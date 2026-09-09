#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,gc,importlib.util,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from convergence_monitoring.detectors import exact_pairwise_scores_by_class
from convergence_monitoring.framework import standardize_monitoring_score
P11=ROOT/'experiments'/'11_compare_ckl_vs_le_across_k.py'
spec=importlib.util.spec_from_file_location('exp11',P11); exp11=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp11)
K=40

def parse_args():
 p=argparse.ArgumentParser();
 p.add_argument('--common-npz',type=Path,default=Path('results/common_loss_trajectories/cifar10_noisy_label_loss_trajectories.npz'))
 p.add_argument('--num-classes',type=int,default=10); p.add_argument('--q',type=float,default=.10)
 p.add_argument('--run-lengths',type=int,nargs='+',default=(2,3,4,5,7)); p.add_argument('--window-lengths',type=int,nargs='+',default=(10,20,30)); p.add_argument('--window-fractions',type=float,nargs='+',default=(.3,.5,.67,.8)); p.add_argument('--rank-ewma-lambdas',type=float,nargs='+',default=(.05,.1,.2))
 p.add_argument('--taus',type=float,nargs='+',default=(1.,1.5,2.,2.5,3.)); p.add_argument('--fixed-ewma-lambdas',type=float,nargs='+',default=(.05,.1,.2)); p.add_argument('--fixed-run-lengths',type=int,nargs='+',default=(2,3,4,5,6,7)); p.add_argument('--fixed-sliding-k-fracs',type=float,nargs='+',default=(.3,.5,.67,.8))
 p.add_argument('--target-fpr',type=float,default=.05); p.add_argument('--output-dir',type=Path,default=Path('results/ckl_rank_vs_fixed_threshold_k40')); return p.parse_args()

def metrics(y,p):
 y=np.asarray(y,bool); p=np.asarray(p,bool); tp=np.sum(p&y); fp=np.sum(p&~y); fn=np.sum(~p&y); tn=np.sum(~p&~y); tpr=tp/(tp+fn); fpr=fp/(fp+tn); prec=tp/(tp+fp) if tp+fp else np.nan; f1=2*prec*tpr/(prec+tpr) if np.isfinite(prec) and prec+tpr else np.nan; return dict(TPR=float(tpr),FPR=float(fpr),precision=float(prec),F1=float(f1),n_selected=int(np.sum(p)))

def write_csv(path,rows):
 keys=[]; seen=set();
 for r in rows:
  for k in r:
   if k not in seen: seen.add(k); keys.append(k)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def percentiles(score,labels,start,C):
 N,T=score.shape; out=np.full((N,T-start),np.nan,np.float32)
 for u,t in enumerate(range(start,T)):
  s=score[:,t]
  for c in range(C):
   idx=np.flatnonzero((labels==c)&np.isfinite(s))
   if idx.size:
    v=s[idx]; order=np.argsort(v,kind='mergesort'); r=np.empty(idx.size,float); r[order]=np.arange(idx.size); out[idx,u]=(1. if idx.size==1 else r/(idx.size-1))
 return out

def minrun(hit,m):
 T,N=hit.shape; run=np.zeros(N,np.int16); det=np.zeros(N,bool); out=np.zeros((T,N),bool)
 for t in range(T): run=np.where(hit[t],run+1,0); det|=run>=m; out[t]=det
 return out

def sliding(hit,ell,frac):
 T,N=hit.shape; k=int(np.ceil(frac*ell)); det=np.zeros(N,bool); out=np.zeros((T,N),bool); cs=np.cumsum(hit.astype(np.int16),axis=0)
 for t in range(T):
  cnt=cs[t].copy(); left=t-ell
  if left>=0: cnt-=cs[left]
  det|=cnt>=k; out[t]=det
 return out,k

def rank_ewma(p,lam):
 N,T=p.shape; out=np.full((N,T),np.nan,np.float32); e=np.full(N,np.nan)
 for t in range(T):
  x=p[:,t]; fin=np.isfinite(x); init=fin&~np.isfinite(e); e[init]=x[init]; upd=fin&np.isfinite(e); e[upd]=(1-lam)*e[upd]+lam*x[upd]; out[:,t]=e
 return out

def fixed_ewma(z,lam):
 T,N=z.shape; out=np.full((T,N),np.nan,np.float32); e=np.full(N,np.nan)
 for t in range(T):
  x=z[t]; fin=np.isfinite(x); init=fin&~np.isfinite(e); e[init]=x[init]; upd=fin&np.isfinite(e); e[upd]=(1-lam)*e[upd]+lam*x[upd]; out[t]=e
 return out

def sticky(score_tn,tau):
 hit=score_tn>=tau; det=np.zeros(hit.shape[1],bool); out=np.zeros_like(hit)
 for t in range(hit.shape[0]): det|=hit[t]; out[t]=det
 return out

def topq(score,q):
 idx=np.flatnonzero(np.isfinite(score)); k=min(max(1,int(round(q*score.size))),idx.size); sel=np.zeros(score.size,bool); v=score[idx]; loc=np.argpartition(v,-k)[-k:]; sel[idx[loc]]=1; return sel

def add_ever(rows,fam,det,params,ever,y,eps):
 for t,ep in enumerate(eps): rows.append(dict(method='CKL',family=fam,detector=det,epoch=int(ep),**params,**metrics(y,ever[t])))

def add_current(rows,fam,det,params,score,y,eps,q):
 for t,ep in enumerate(eps): rows.append(dict(method='CKL',family=fam,detector=det,epoch=int(ep),q=float(q),**params,**metrics(y,topq(score[:,t],q))))

def best(rows,fam,target=None):
 out=[]
 for det in sorted(set(r['detector'] for r in rows if r['family']==fam)):
  g=[r for r in rows if r['family']==fam and r['detector']==det and (target is None or r['FPR']<=target)]
  if not g: continue
  g.sort(key=lambda r:(-r['TPR'],r['FPR'],r['epoch'])); out.append(g[0])
 return out

def main():
 a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); d=np.load(a.common_npz,allow_pickle=False); loss=np.asarray(d['loss_traj'],np.float32); labels=np.asarray(d['observed_label'],np.int64); y=np.asarray(d['is_anomaly'],bool); epochs=np.asarray(d['epoch'],np.int64)
 G=exp11.build_class_mean(loss,labels,a.num_classes); print('Computing CKL K=40...'); ckl=exp11.compute_ckl(loss,labels,G,K,a.num_classes); start=K+1; eps=epochs[start:]; rows=[]
 pct=percentiles(ckl,labels,start,a.num_classes); hit=(pct>=1-a.q).T
 for m in a.run_lengths: add_ever(rows,'rank_dynamic','min_run',{'threshold_type':'within_class_top_q','q':a.q,'m':m},minrun(hit,m),y,eps)
 for ell in a.window_lengths:
  for frac in a.window_fractions:
   ev,k=sliding(hit,ell,frac); add_ever(rows,'rank_dynamic','sliding_window',{'threshold_type':'within_class_top_q','q':a.q,'ell':ell,'required_fraction':frac,'k':k},ev,y,eps)
 for lam in a.rank_ewma_lambdas: add_current(rows,'rank_dynamic','ewma',{'threshold_type':'within_class_percentile','lambda':lam},rank_ewma(pct,lam),y,eps,a.q)
 pair=exact_pairwise_scores_by_class(ckl,labels,start_index=start); ps=np.asarray(pair['cumulative_score'])[:,start:]; add_current(rows,'rank_dynamic','cumulative_pairwise',{'threshold_type':'fixed_top_q'},ps,y,eps,a.q)
 zfull=standardize_monitoring_score(ckl.T,labels,direction='higher',num_classes=a.num_classes,start_index=start); z=zfull[start:]
 for tau in a.taus:
  h=z>=tau
  for m in a.fixed_run_lengths: add_ever(rows,'fixed_z','min_run',{'threshold_type':'fixed_z','tau':tau,'m':m},minrun(h,m),y,eps)
  for ell in a.window_lengths:
   for frac in a.fixed_sliding_k_fracs:
    ev,k=sliding(h,ell,frac); add_ever(rows,'fixed_z','sliding_window',{'threshold_type':'fixed_z','tau':tau,'ell':ell,'required_fraction':frac,'k':k},ev,y,eps)
 for lam in a.fixed_ewma_lambdas:
  ew=fixed_ewma(z,lam)
  for tau in a.taus: add_ever(rows,'fixed_z','ewma',{'threshold_type':'fixed_z','tau':tau,'lambda':lam},sticky(ew,tau),y,eps)
 write_csv(a.output_dir/'ckl_rank_vs_fixed_all_cases.csv',rows); dyn=best(rows,'rank_dynamic',None); write_csv(a.output_dir/'ckl_rank_dynamic_best_q10.csv',dyn); fair=best(rows,'rank_dynamic',a.target_fpr)+best(rows,'fixed_z',a.target_fpr); write_csv(a.output_dir/'ckl_rank_vs_fixed_best_under_fpr.csv',fair)
 order=[]
 for fam in ('fixed_z','rank_dynamic'):
  for det in ('min_run','sliding_window','ewma'):
   m=[r for r in fair if r['family']==fam and r['detector']==det]
   if m: order.append(m[0])
 x=np.arange(len(order)); width=.38; fig,ax=plt.subplots(figsize=(11,5.5)); ax.bar(x-width/2,[r['TPR'] for r in order],width,label='TPR'); ax.bar(x+width/2,[r['FPR'] for r in order],width,label='FPR'); ax.axhline(a.target_fpr,linestyle='--',linewidth=1,label=f'FPR budget={a.target_fpr}'); ax.set_xticks(x); ax.set_xticklabels([f"{r['family']}\n{r['detector']}" for r in order],rotation=20,ha='right'); ax.set_ylim(0,1); ax.set_ylabel('Fraction'); ax.set_title('CKL K=40: rank-based vs fixed-z under common FPR budget'); ax.legend(); fig.tight_layout(); fig.savefig(a.output_dir/'fig_ckl_rank_vs_fixed_under_fpr.png',dpi=180); plt.close(fig)
 cfg={'artifact':'CKL_rank_vs_fixed_threshold','K':40,'CKL_limit_rule':'last3 mean','rank_q':a.q,'rank_dynamic_threshold':'within-observed-class percentile >= 1-q','fixed_threshold':'within-observed-class z >= tau','target_fpr_for_fair_benchmark':a.target_fpr,'selection_warning':'hyperparameters and epoch selected on labeled run; freeze before independent validation'}; (a.output_dir/'ckl_rank_vs_fixed_config.json').write_text(json.dumps(cfg,indent=2))
 print('\nRank-based CKL q=0.10:'); [print(f"{r['detector']:<20} TPR={r['TPR']:.4f} FPR={r['FPR']:.4f} epoch={r['epoch']}") for r in dyn]; print('\nFair comparison under FPR<=0.05:'); [print(f"{r['family']:<12} {r['detector']:<20} TPR={r['TPR']:.4f} FPR={r['FPR']:.4f} epoch={r['epoch']}") for r in fair]; print('Outputs:',a.output_dir); del ckl,pct,hit; gc.collect()
if __name__=='__main__': main()
