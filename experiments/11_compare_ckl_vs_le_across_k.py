#!/usr/bin/env python
"""
Compare CKL and LE-GIE across K = 20, 30, 40 on the SAME loss trajectories.

CKL configuration
-----------------
- Uses its existing stabilized GIE construction.
- GIE/sample/reference limit inside each K-window = mean of last 3 points.
- CKL class reference = geometric mean of valid positive W_i and d_i
  within observed class.

LE-GIE configuration
--------------------
- The theoretical estimator is unchanged:
      ell_hat = ell_err_hat + log(abs(m_GIE_hat))
- ell_err limit = next sample.
- GIE sample limit = next sample.
- GIE class-reference limit = next sample.
- GIE reference trajectory = mean loss trajectory of samples sharing
  the observed label.

Fair comparison
---------------
For each K, both CKL and LE are evaluated from the same common start:
    epoch = K + 2
because LE needs:
    boundary x[t-K-1], K-point tail x[t-K:t], and next/current x[t].

For each estimator and K we report:
- best raw ROC-AUC
- best within-class-z ROC-AUC
- best cumulative exact-pairwise ROC-AUC
- best tested current top-q TPR subject to FPR <= 0.05
- epoch and q of that low-FPR operating point

This is an exploratory labeled-run comparison. Freeze the selected
configuration before independent validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence_monitoring.detectors import (
    binary_auc_from_scores,
    exact_pairwise_scores_by_class,
)
from convergence_monitoring.framework import standardize_monitoring_score


EPS = 1e-7
BATCH = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--common-npz",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument(
        "--K-values",
        type=int,
        nargs="+",
        default=(20, 30, 40),
    )
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
    )
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ckl_vs_le_k_sweep"),
    )
    return p.parse_args()


def build_class_mean(loss, labels, num_classes):
    """Compact class mean trajectories, shape [C,T]."""
    x = np.asarray(loss, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    T = x.shape[1]
    Gc = np.full((num_classes, T), np.nan, dtype=np.float32)

    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size:
            Gc[c] = np.nanmean(
                x[idx],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
    return Gc


def gie_batch(phi, G, phi_limit, G_limit, return_W=False):
    """Row-wise GIE/Hill ratio, optionally returning sample boundary W."""
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    phi_limit = np.asarray(phi_limit, dtype=np.float64)
    G_limit = np.asarray(G_limit, dtype=np.float64)

    R = np.abs(phi - phi_limit[:, None])
    FR = np.abs(G - G_limit[:, None])

    with np.errstate(invalid="ignore"):
        w0 = np.max(R, axis=1)
        w1 = np.max(FR, axis=1)

    mask = (
        (R > EPS)
        & (FR > EPS)
        & np.isfinite(R)
        & np.isfinite(FR)
    )

    n_nonzero = mask.sum(axis=1)
    k_internal = n_nonzero - 1

    base_ok = (
        (k_internal > 4)
        & np.isfinite(w0)
        & np.isfinite(w1)
        & (w0 > 0.0)
        & (w1 > 0.0)
    )

    safe_w0 = np.where(base_ok, w0 + EPS, 1.0)
    safe_w1 = np.where(base_ok, w1 + EPS, 1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_num = np.where(
            mask,
            np.log(np.abs(R / safe_w0[:, None])),
            0.0,
        )
        log_den = np.where(
            mask,
            np.log(np.abs(FR / safe_w1[:, None])),
            0.0,
        )

    denom_num = np.sum(log_num, axis=1)
    denom_den = np.sum(log_den, axis=1)

    ok = (
        base_ok
        & np.isfinite(denom_num)
        & np.isfinite(denom_den)
        & (np.abs(denom_num) >= EPS)
        & (np.abs(denom_den) >= EPS)
    )

    hill_num = np.full(phi.shape[0], np.nan, dtype=np.float64)
    hill_den = np.full(phi.shape[0], np.nan, dtype=np.float64)

    hill_num[ok] = -k_internal[ok] / denom_num[ok]
    hill_den[ok] = -k_internal[ok] / denom_den[ok]

    good = (
        ok
        & np.isfinite(hill_num)
        & np.isfinite(hill_den)
        & (np.abs(hill_den) > EPS)
    )

    d = np.full(phi.shape[0], np.nan, dtype=np.float64)
    d[good] = hill_num[good] / hill_den[good]

    if return_W:
        return d, w0
    return d


def geometric_class_reference(values, labels, num_classes):
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        v = values[labels == c]
        v = v[np.isfinite(v) & (v > 0.0)]
        if v.size:
            out[c] = np.exp(np.mean(np.log(np.maximum(v, EPS))))
    return out


def ckl_vectorized(W1, d1, W2, d2):
    """Vectorized equivalent of ckl_finite."""
    W1 = np.asarray(W1, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    out = np.full(W1.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(W1) & np.isfinite(d1)
        & np.isfinite(W2) & np.isfinite(d2)
        & (W1 > 0.0) & (d1 > 0.0)
        & (W2 > 0.0) & (d2 > 0.0)
    )
    if not np.any(valid):
        return out

    a = W1[valid]
    b = d1[valid]
    c = W2[valid]
    e = d2[valid]

    res = np.full(a.shape, np.nan, dtype=np.float64)
    equal = np.abs(a - c) < 1e-12
    lower = (~equal) & (a < c)
    upper = (~equal) & (~lower)

    if np.any(equal):
        aa, bb, cc = a[equal], b[equal], e[equal]
        res[equal] = (
            aa * ((cc - bb) ** 2)
            / (((bb + 1.0) ** 2) * (cc + 1.0) + EPS)
        )

    if np.any(lower):
        aa, bb, ww, dd = a[lower], b[lower], c[lower], e[lower]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            term = (
                aa
                * (
                    (dd / (bb + 1.0)) * np.log(np.maximum(ww / aa, EPS))
                    - (bb - dd) / ((bb + 1.0) ** 2)
                )
                + dd * ((ww - aa) + aa * np.log(np.maximum(aa / ww, EPS)))
            )
            res[lower] = (
                term
                + (bb / (bb + 1.0)) * aa
                - (dd / (dd + 1.0)) * ww
            )

    if np.any(upper):
        aa, bb, ww, dd = a[upper], b[upper], c[upper], e[upper]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = np.maximum(ww / aa, EPS)
            term = (
                aa / ((bb + 1.0) ** 2)
                * (dd * (ratio ** (bb + 1.0)) - bb)
            )
            res[upper] = (
                term
                + (bb / (bb + 1.0)) * aa
                - (dd / (dd + 1.0)) * ww
            )

    out[valid] = res
    return out


def compute_ckl(loss, labels, Gc, K, num_classes):
    """CKL with its stabilized last-three limit."""
    x = np.asarray(loss, dtype=np.float32)
    N, T = x.shape
    out = np.full((N, T), np.nan, dtype=np.float32)

    first_col = K - 1

    for t in range(first_col, T):
        start = t - K + 1

        d_t = np.full(N, np.nan, dtype=np.float64)
        W_t = np.full(N, np.nan, dtype=np.float64)

        G_limit_cls = np.mean(
            Gc[:, t-2:t+1],
            axis=1,
            dtype=np.float64,
        )

        for lo in range(0, N, BATCH):
            hi = min(lo + BATCH, N)
            xb = np.asarray(x[lo:hi, start:t+1], dtype=np.float64)
            lab = labels[lo:hi]
            Gb = np.asarray(Gc[lab, start:t+1], dtype=np.float64)

            Lx = np.mean(xb[:, -3:], axis=1)
            Lg = G_limit_cls[lab]

            db, wb = gie_batch(
                xb, Gb, Lx, Lg, return_W=True
            )
            d_t[lo:hi] = db
            W_t[lo:hi] = wb

        Wc = geometric_class_reference(
            W_t, labels, num_classes
        )
        dc = geometric_class_reference(
            d_t, labels, num_classes
        )

        score = ckl_vectorized(
            W_t,
            d_t,
            Wc[labels],
            dc[labels],
        )
        out[:, t] = score.astype(np.float32)

    return out


def compute_le_next_next(loss, labels, Gc, K):
    """LE-GIE: ell_err L=next and GIE L=next."""
    x = np.asarray(loss, dtype=np.float32)
    N, T = x.shape
    out = np.full((N, T), np.nan, dtype=np.float32)

    inv_j = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    first_col = K + 1

    for t in range(first_col, T):
        boundary = t - K - 1
        start = t - K

        G_limit_cls = np.asarray(Gc[:, t], dtype=np.float64)

        for lo in range(0, N, BATCH):
            hi = min(lo + BATCH, N)
            lab = labels[lo:hi]

            xb = np.asarray(x[lo:hi, start:t], dtype=np.float64)
            L = np.asarray(x[lo:hi, t], dtype=np.float64)

            r0 = np.abs(
                np.asarray(x[lo:hi, boundary], dtype=np.float64) - L
            )
            rj = np.abs(xb - L[:, None])

            valid_err = (
                np.isfinite(L)
                & np.isfinite(r0)
                & (r0 != 0.0)
                & np.all(
                    np.isfinite(rj) & (rj != 0.0),
                    axis=1,
                )
            )

            ell_err = np.full(hi-lo, np.nan, dtype=np.float64)
            if np.any(valid_err):
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    vals = (
                        np.log(
                            rj[valid_err] / r0[valid_err, None]
                        )
                        * inv_j[None, :]
                    )
                ell_err[valid_err] = np.mean(vals, axis=1)

            Gb = np.asarray(
                Gc[lab, start:t],
                dtype=np.float64,
            )
            m = gie_batch(
                xb,
                Gb,
                L,
                G_limit_cls[lab],
                return_W=False,
            )

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                le = ell_err + np.log(np.abs(m))

            valid = (
                valid_err
                & np.isfinite(m)
                & (m != 0.0)
                & np.isfinite(le)
            )

            block = np.full(hi-lo, np.nan, dtype=np.float32)
            block[valid] = le[valid].astype(np.float32)
            out[lo:hi, t] = block

    return out


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    finite = np.isfinite(score)
    if np.sum(finite) < 2:
        return np.nan
    yy = y[finite]
    if np.all(yy) or np.all(~yy):
        return np.nan
    return binary_auc_from_scores(
        score[finite & y],
        score[finite & ~y],
    )


def auc_curve(y, score_nt, start_col):
    auc = np.full(score_nt.shape[1], np.nan, dtype=np.float64)
    for t in range(start_col, score_nt.shape[1]):
        auc[t] = auc_binary(y, score_nt[:, t])
    return auc


def best_auc(auc, epochs):
    finite = np.flatnonzero(np.isfinite(auc))
    t = int(finite[np.nanargmax(auc[finite])])
    return float(auc[t]), int(epochs[t]), t


def topq_metrics(y, score, q):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    finite_idx = np.flatnonzero(np.isfinite(score))
    k = min(
        max(1, int(round(float(q) * y.size))),
        finite_idx.size,
    )

    selected = np.zeros(y.size, dtype=bool)
    vals = score[finite_idx]
    local = np.argpartition(vals, -k)[-k:]
    selected[finite_idx[local]] = True

    TP = np.sum(selected & y)
    FP = np.sum(selected & ~y)
    FN = np.sum(~selected & y)
    TN = np.sum(~selected & ~y)

    return {
        "q": float(q),
        "TPR": float(TP / (TP + FN)),
        "FPR": float(FP / (FP + TN)),
        "precision": float(TP / (TP + FP)),
    }


def evaluate_score(
    method,
    K,
    score_nt,
    *,
    labels,
    y,
    epochs,
    num_classes,
    top_fractions,
    target_fpr,
):
    # Fair per-K common start for BOTH estimators.
    start_col = K + 1
    start_epoch = start_col + 1

    z_tn = standardize_monitoring_score(
        score_nt.T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=start_col,
    )
    z_nt = z_tn.T.astype(np.float32, copy=False)

    raw_auc = auc_curve(y, score_nt, start_col)
    z_auc = auc_curve(y, z_nt, start_col)

    raw_best, raw_ep, _ = best_auc(raw_auc, epochs)
    z_best, z_ep, _ = best_auc(z_auc, epochs)

    pair = exact_pairwise_scores_by_class(
        score_nt,
        labels,
        start_index=start_col,
    )
    pair_score = np.asarray(pair["cumulative_score"])
    pair_auc = auc_curve(y, pair_score, start_col)
    pair_best, pair_ep, _ = best_auc(pair_auc, epochs)

    best_case = None
    for t in range(start_col, score_nt.shape[1]):
        for q in top_fractions:
            m = topq_metrics(y, pair_score[:, t], q)
            if m["FPR"] <= target_fpr:
                cand = {
                    **m,
                    "epoch": int(epochs[t]),
                    "pair_auc_at_epoch": float(pair_auc[t]),
                }
                if (
                    best_case is None
                    or cand["TPR"] > best_case["TPR"]
                    or (
                        np.isclose(cand["TPR"], best_case["TPR"])
                        and cand["FPR"] < best_case["FPR"]
                    )
                ):
                    best_case = cand

    row = {
        "method": method,
        "K": int(K),
        "common_start_epoch": int(start_epoch),
        "raw_best_auc": raw_best,
        "raw_best_epoch": raw_ep,
        "z_best_auc": z_best,
        "z_best_epoch": z_ep,
        "pairwise_best_auc": pair_best,
        "pairwise_best_epoch": pair_ep,
        "best_q": float(best_case["q"]),
        "best_q_epoch": int(best_case["epoch"]),
        "best_q_TPR": float(best_case["TPR"]),
        "best_q_FPR": float(best_case["FPR"]),
        "best_q_precision": float(best_case["precision"]),
        "pair_auc_at_best_q_epoch": float(
            best_case["pair_auc_at_epoch"]
        ),
    }

    return row


def write_csv(path, rows):
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.common_npz, allow_pickle=False)
    loss = np.asarray(d["loss_traj"], dtype=np.float32)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)
    epochs = np.asarray(d["epoch"], dtype=np.int64)

    Gc = build_class_mean(
        loss,
        labels,
        args.num_classes,
    )

    rows = []

    for K in args.K_values:
        print()
        print("=" * 72)
        print(f"K={K}")
        print("=" * 72)

        print("Computing CKL (stable last3 GIE limit)...")
        ckl = compute_ckl(
            loss,
            labels,
            Gc,
            K,
            args.num_classes,
        )
        r = evaluate_score(
            "CKL",
            K,
            ckl,
            labels=labels,
            y=y,
            epochs=epochs,
            num_classes=args.num_classes,
            top_fractions=args.top_fractions,
            target_fpr=args.target_fpr,
        )
        rows.append(r)
        print(
            "CKL    | "
            f"raw={r['raw_best_auc']:.6f} "
            f"z={r['z_best_auc']:.6f} "
            f"pair={r['pairwise_best_auc']:.6f} "
            f"| q={r['best_q']:.2f} "
            f"TPR={r['best_q_TPR']:.4f} "
            f"FPR={r['best_q_FPR']:.4f}"
        )
        del ckl
        gc.collect()

        print("Computing LE-GIE (next for ell_err and GIE)...")
        le = compute_le_next_next(
            loss,
            labels,
            Gc,
            K,
        )
        r = evaluate_score(
            "LE_GIE",
            K,
            le,
            labels=labels,
            y=y,
            epochs=epochs,
            num_classes=args.num_classes,
            top_fractions=args.top_fractions,
            target_fpr=args.target_fpr,
        )
        rows.append(r)
        print(
            "LE_GIE | "
            f"raw={r['raw_best_auc']:.6f} "
            f"z={r['z_best_auc']:.6f} "
            f"pair={r['pairwise_best_auc']:.6f} "
            f"| q={r['best_q']:.2f} "
            f"TPR={r['best_q_TPR']:.4f} "
            f"FPR={r['best_q_FPR']:.4f}"
        )
        del le
        gc.collect()

    write_csv(
        args.output_dir / "ckl_vs_le_k_sweep_summary.csv",
        rows,
    )

    # Pairwise AUC vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("CKL", "LE_GIE"):
        rr = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda r: r["K"],
        )
        ax.plot(
            [r["K"] for r in rr],
            [r["pairwise_best_auc"] for r in rr],
            marker="o",
            label=method,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("Best cumulative pairwise ROC-AUC")
    ax.set_title("CKL vs LE-GIE across K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "fig_ckl_vs_le_pairwise_auc_vs_K.png",
        dpi=180,
    )
    plt.close(fig)

    # Best low-FPR TPR vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("CKL", "LE_GIE"):
        rr = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda r: r["K"],
        )
        ax.plot(
            [r["K"] for r in rr],
            [r["best_q_TPR"] for r in rr],
            marker="o",
            label=method,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("Best tested TPR under FPR <= 0.05")
    ax.set_title("CKL vs LE-GIE: low-FPR detection across K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "fig_ckl_vs_le_best_tpr_vs_K.png",
        dpi=180,
    )
    plt.close(fig)

    config = {
        "artifact": "CKL_vs_LE_K_sweep",
        "K_values": [int(k) for k in args.K_values],
        "common_data": str(args.common_npz),
        "CKL_limit_rule": "mean of last 3 points inside each CKL GIE window",
        "LE_formula": "ell_err + log(abs(m_GIE))",
        "LE_error_limit_rule": "next sample",
        "LE_GIE_limit_rule": "next sample",
        "LE_reference": "observed-class mean trajectory",
        "common_start_rule":
            "for each K, evaluate both estimators starting at epoch K+2",
        "pairwise_rule":
            "exact all finite peers within same observed class; cumulative wins / cumulative comparisons",
        "top_fractions": [float(q) for q in args.top_fractions],
        "target_fpr": float(args.target_fpr),
        "selection_warning":
            "best q and epoch are selected on this labeled run; freeze before independent validation",
        "production_modules_modified": False,
    }
    with (
        args.output_dir / "ckl_vs_le_k_sweep_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print("Done.")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
