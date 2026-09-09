#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, gc, importlib.util, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from convergence_monitoring.detectors import exact_pairwise_scores_by_class

# Reuse the score constructors already validated in Experiment 11.
p11 = ROOT / "experiments" / "11_compare_ckl_vs_le_across_k.py"
spec = importlib.util.spec_from_file_location("exp11", p11)
exp11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp11)

K = 40

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--common-npz", type=Path,
        default=Path("results/common_loss_trajectories/cifar10_noisy_label_loss_trajectories.npz"))
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--q", type=float, default=.10)
    p.add_argument("--run-lengths", type=int, nargs="+", default=(2,3,4,5,7))
    p.add_argument("--window-lengths", type=int, nargs="+", default=(10,20,30))
    p.add_argument("--window-fractions", type=float, nargs="+", default=(.3,.5,.67,.8))
    p.add_argument("--ewma-lambdas", type=float, nargs="+", default=(.05,.1,.2))
    p.add_argument("--output-dir", type=Path,
        default=Path("results/le_dynamic_rank_detectors_k40"))
    return p.parse_args()

def metrics(y,p):
    y=np.asarray(y,bool); p=np.asarray(p,bool)
    tp=np.sum(p&y); fp=np.sum(p&~y); fn=np.sum(~p&y); tn=np.sum(~p&~y)
    tpr=tp/(tp+fn); fpr=fp/(fp+tn); prec=tp/(tp+fp) if tp+fp else np.nan
    f1=2*prec*tpr/(prec+tpr) if np.isfinite(prec) and prec+tpr else np.nan
    return dict(TPR=float(tpr),FPR=float(fpr),precision=float(prec),F1=float(f1))

def write_csv(path, rows):
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def class_quantile_hits(score,labels,start,C,q):
    N,T=score.shape; out=np.zeros((T-start,N),bool)
    for u,t in enumerate(range(start,T)):
        s=score[:,t]
        for c in range(C):
            idx=np.flatnonzero((labels==c)&np.isfinite(s))
            if idx.size:
                thr=np.quantile(s[idx],1-q); out[u,idx]=s[idx]>=thr
    return out


def minrun(hit,m):
    T,N=hit.shape; run=np.zeros(N,np.int16); det=np.zeros(N,bool); out=np.zeros((T,N),bool)
    for t in range(T):
        run=np.where(hit[t],run+1,0); det |= run>=m; out[t]=det
    return out

def sliding(hit,ell,frac):
    T,N=hit.shape; k=int(np.ceil(frac*ell)); det=np.zeros(N,bool); out=np.zeros((T,N),bool)
    cs=np.cumsum(hit.astype(np.int16),axis=0)
    for t in range(T):
        cnt=cs[t].copy()
        if t-ell>=0: cnt-=cs[t-ell]
        det |= cnt>=k; out[t]=det
    return out,k

def percentiles(score,labels,start,C):
    N,T=score.shape; out=np.full((N,T-start),np.nan,np.float32)
    for u,t in enumerate(range(start,T)):
        s=score[:,t]
        for c in range(C):
            idx=np.flatnonzero((labels==c)&np.isfinite(s))
            if idx.size:
                v=s[idx]; order=np.argsort(v,kind="mergesort")
                r=np.empty(idx.size,float); r[order]=np.arange(idx.size)
                out[idx,u]=(1.0 if idx.size==1 else r/(idx.size-1))
    return out

def ewma(p,lam):
    N,T=p.shape; out=np.full((N,T),np.nan,np.float32); e=np.full(N,np.nan)
    for t in range(T):
        x=p[:,t]; fin=np.isfinite(x); init=fin&~np.isfinite(e); e[init]=x[init]
        upd=fin&np.isfinite(e); e[upd]=(1-lam)*e[upd]+lam*x[upd]; out[:,t]=e
    return out

def topq(score,q):
    idx=np.flatnonzero(np.isfinite(score)); k=min(max(1,round(q*score.size)),idx.size)
    sel=np.zeros(score.size,bool); v=score[idx]; loc=np.argpartition(v,-k)[-k:]; sel[idx[loc]]=1
    return sel

def add_ever(rows,method,det,fam,params,ever,y,epochs):
    for t,ep in enumerate(epochs):
        rows.append(dict(method=method,detector=det,threshold_family=fam,epoch=int(ep),**params,**metrics(y,ever[t])))

def add_current(rows,method,det,fam,params,score,y,epochs,q):
    for t,ep in enumerate(epochs):
        rows.append(dict(method=method,detector=det,threshold_family=fam,epoch=int(ep),q=float(q),**params,**metrics(y,topq(score[:,t],q))))

def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    d=np.load(a.common_npz,allow_pickle=False)
    loss=np.asarray(d["loss_traj"],np.float32); labels=np.asarray(d["observed_label"],np.int64)
    y=np.asarray(d["is_anomaly"],bool); epochs=np.asarray(d["epoch"],np.int64)
    G=exp11.build_class_mean(loss,labels,a.num_classes)
    le=exp11.compute_le_next_next(loss,labels,G,K)
    start=K+1; eps=epochs[start:]
    rows=[]

    for method,score in (("LE_GIE",le),):
        qhit=class_quantile_hits(score,labels,start,a.num_classes,a.q)
        for m in a.run_lengths: add_ever(rows,method,"dynamic_min_run","class_quantile",{"m":m,"q":a.q},minrun(qhit,m),y,eps)
        for ell in a.window_lengths:
            for frac in a.window_fractions:
                ev,k=sliding(qhit,ell,frac)
                add_ever(rows,method,"dynamic_sliding","class_quantile",{"ell":ell,"required_fraction":frac,"k":k,"q":a.q},ev,y,eps)


        pct=percentiles(score,labels,start,a.num_classes)
        for lam in a.ewma_lambdas:
            add_current(rows,method,"rank_ewma","within_class_percentile",{"lambda":lam},ewma(pct,lam),y,eps,a.q)

        pair=exact_pairwise_scores_by_class(score,labels,start_index=start)
        ps=np.asarray(pair["cumulative_score"])[:,start:]
        add_current(rows,method,"cumulative_pairwise","fixed_top_q",{},ps,y,eps,a.q)
        gc.collect()

    write_csv(a.output_dir/"le_dynamic_rank_all_cases.csv",rows)

    best=[]
    keys=sorted(set((r["method"],r["detector"],r["threshold_family"]) for r in rows))
    for key in keys:
        g=[r for r in rows if (r["method"],r["detector"],r["threshold_family"])==key]
        feasible=[r for r in g if r["FPR"] <= 0.05]
        pool=feasible if feasible else g
        pool.sort(key=lambda r:(-r["TPR"],r["FPR"],r["epoch"]))
        row=dict(pool[0])
        row["selection_status"]=("best_under_FPR_0.05" if feasible else "no_FPR_feasible_best_TPR")
        best.append(row)
    write_csv(a.output_dir/"le_dynamic_rank_best_cases.csv",best)

    x=np.arange(len(best)); width=.38
    fig,ax=plt.subplots(figsize=(14,6))
    ax.bar(x-width/2,[r["TPR"] for r in best],width,label="TPR")
    ax.bar(x+width/2,[r["FPR"] for r in best],width,label="FPR")
    ax.set_xticks(x); ax.set_xticklabels([f"{r['method']}\n{r['detector']}\n{r['threshold_family']}" for r in best],rotation=25,ha="right")
    ax.set_ylim(0,1); ax.set_ylabel("Fraction"); ax.set_title("LE-GIE dynamic rank/quantile detectors, K=40"); ax.legend()
    fig.tight_layout(); fig.savefig(a.output_dir/"fig_le_dynamic_rank_best_cases.png",dpi=180); plt.close(fig)

    cfg={
      "artifact":"LE_dynamic_rank_detectors_K40",
      "K":40,"q":a.q,
      "LE_error_limit_rule":"next",
      "LE_GIE_limit_rule":"next",
      "dynamic_threshold":"within-observed-class percentile; hit if percentile >= 1-q",
      "detectors":["quantile min-run","quantile sliding","rank-EWMA","cumulative pairwise"],
      "run_lengths":list(a.run_lengths),
      "window_lengths":list(a.window_lengths),
      "window_fractions":list(a.window_fractions),
      "ewma_lambdas":list(a.ewma_lambdas),
      "mean_plus_a_std_included":False,
      "reason":"mean+a*std is equivalent to fixed z>=a; omitted from this genuinely dynamic rank experiment",
      "evaluation_note":"q is fixed at 0.10; FPR is used only to summarize benchmark performance, not by the detector"
    }
    (a.output_dir/"le_dynamic_rank_config.json").write_text(json.dumps(cfg,indent=2))
    for r in best:
        print(f"{r['method']:>7} | {r['detector']:<20} | {r['threshold_family']:<24} | TPR={r['TPR']:.4f} FPR={r['FPR']:.4f} epoch={r['epoch']}")
    print("Outputs:",a.output_dir)

if __name__=="__main__":
    main()
