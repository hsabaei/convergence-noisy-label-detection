#!/usr/bin/env python
"""
CKL + LE-GIE score-fusion experiment for CIFAR-10 with 5% label noise.

Goal
====
Test whether CKL and LE-GIE contain complementary information and whether
combining them improves noisy-label discrimination.

Frozen estimator settings
=========================
K = 40

CKL:
    GIE limit = last-three mean

LE-GIE:
    ell_err limit = next sample
    GIE limit     = next sample
    signed LE     = ell_err + log(abs(m_GIE))

Fusion
======
Raw CKL and LE scores have different scales, so fusion is performed only
after converting each score to a within-observed-class percentile:

    r_CKL(i,t), r_LE(i,t) in [0,1]

Fusion variants:
    weighted: w*r_CKL + (1-w)*r_LE
    max:      max(r_CKL, r_LE)

Default weights:
    w in {0, .25, .50, .75, 1}

No ground-truth label information is used to build any fused score.

Evaluation
==========
1) score-level ROC-AUC through training;
2) rank-EWMA statistic ROC-AUC + fixed top-q decision;
3) cumulative-pairwise statistic ROC-AUC + fixed top-q decision;
4) dynamic min-run and sliding-window fixed top-q decisions;
5) overlap/complementarity diagnostics for CKL vs LE.

TPR/FPR and ROC-AUC use the known synthetic noise mask for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from convergence_monitoring.detectors import (
    exact_pairwise_scores_by_class,
    binary_auc_from_scores,
)

# Reuse the already validated CKL/LE implementations from experiment 11.
P11 = ROOT / "experiments" / "11_compare_ckl_vs_le_across_k.py"
if not P11.exists():
    raise FileNotFoundError(
        "Expected experiments/11_compare_ckl_vs_le_across_k.py in the repo."
    )

spec = importlib.util.spec_from_file_location("exp11", P11)
exp11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp11)


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
        "--output-dir",
        type=Path,
        default=Path("results/ckl_le_fusion_noise05_k20_k40"),
    )
    p.add_argument("--K-values", type=int, nargs="+", default=(20, 40))
    p.add_argument("--num-classes", type=int, default=10)

    # Fixed operating budget.
    p.add_argument("--q", type=float, default=0.10)

    # Rank-based temporal detectors.
    p.add_argument("--rank-ewma-lambda", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--sliding-ell", type=int, default=30)
    p.add_argument("--sliding-k", type=int, default=15)

    p.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )

    args = p.parse_args()

    if not (0.0 < args.q < 1.0):
        p.error("--q must be in (0,1).")
    if not (0.0 < args.rank_ewma_lambda <= 1.0):
        p.error("--rank-ewma-lambda must be in (0,1].")
    for w in args.weights:
        if not (0.0 <= w <= 1.0):
            p.error("Every fusion weight must lie in [0,1].")

    return args


def write_csv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def auc_binary(y, score):
    """ROC-AUC from continuous scores.

    detectors.binary_auc_from_scores expects two score arrays:
    positive-class scores first, negative-class scores second.
    """
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    valid = np.isfinite(score)
    pos = score[valid & y]
    neg = score[valid & ~y]

    if pos.size == 0 or neg.size == 0:
        return np.nan

    return float(
        binary_auc_from_scores(
            pos,
            neg,
        )
    )


def auc_curve(y, score_nt, start_col):
    score_nt = np.asarray(score_nt)
    auc = np.full(score_nt.shape[1], np.nan, dtype=np.float64)

    for t in range(start_col, score_nt.shape[1]):
        auc[t] = auc_binary(y, score_nt[:, t])

    return auc


def best_auc_row(auc, epochs):
    finite = np.flatnonzero(np.isfinite(auc))
    if finite.size == 0:
        return {
            "best_auc": np.nan,
            "best_epoch": -1,
        }

    t = int(finite[np.argmax(auc[finite])])
    return {
        "best_auc": float(auc[t]),
        "best_epoch": int(epochs[t]),
    }


def within_class_percentiles(
    score_nt,
    labels,
    start_col,
    num_classes,
):
    """Epoch-wise within-observed-class percentile in [0,1]."""
    score_nt = np.asarray(score_nt, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    N, T = score_nt.shape
    out = np.full((N, T), np.nan, dtype=np.float32)

    for t in range(start_col, T):
        s = score_nt[:, t]

        for c in range(num_classes):
            idx = np.flatnonzero(
                (labels == c) & np.isfinite(s)
            )
            if idx.size == 0:
                continue

            vals = s[idx]
            order = np.argsort(
                vals,
                kind="mergesort",
            )

            # Average rank for ties.
            sorted_vals = vals[order]
            rank_sorted = np.empty(
                idx.size,
                dtype=np.float64,
            )

            a = 0
            while a < idx.size:
                b = a + 1
                while (
                    b < idx.size
                    and sorted_vals[b] == sorted_vals[a]
                ):
                    b += 1

                avg_rank = 0.5 * (a + b - 1)
                rank_sorted[a:b] = avg_rank
                a = b

            rank = np.empty(
                idx.size,
                dtype=np.float64,
            )
            rank[order] = rank_sorted

            if idx.size == 1:
                pct = np.ones(1, dtype=np.float64)
            else:
                pct = rank / float(idx.size - 1)

            out[idx, t] = pct.astype(np.float32)

    return out


def fuse_percentiles(ckl_pct, le_pct, weight):
    """Weighted fusion preserving NaN unless both sources are valid."""
    ckl_pct = np.asarray(ckl_pct, dtype=np.float32)
    le_pct = np.asarray(le_pct, dtype=np.float32)

    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)

    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)
    out[valid] = (
        float(weight) * ckl_pct[valid]
        + (1.0 - float(weight)) * le_pct[valid]
    ).astype(np.float32)

    return out


def fuse_max(ckl_pct, le_pct):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)
    out[valid] = np.maximum(
        ckl_pct[valid],
        le_pct[valid],
    )
    return out


def rank_ewma(score_nt, lam, start_col):
    """EWMA of continuous percentile/fused-rank score."""
    score_nt = np.asarray(score_nt, dtype=np.float32)
    N, T = score_nt.shape

    out = np.full((N, T), np.nan, dtype=np.float32)
    state = np.full(N, np.nan, dtype=np.float32)

    for t in range(start_col, T):
        x = score_nt[:, t]
        finite = np.isfinite(x)

        init = finite & ~np.isfinite(state)
        state[init] = x[init]

        upd = finite & np.isfinite(state)
        state[upd] = (
            (1.0 - float(lam)) * state[upd]
            + float(lam) * x[upd]
        ).astype(np.float32)

        out[:, t] = state

    return out


def topq_mask(score, q):
    score = np.asarray(score, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(score))

    pred = np.zeros(score.size, dtype=bool)
    if finite.size == 0:
        return pred

    k = min(
        max(1, int(round(float(q) * score.size))),
        finite.size,
    )

    vals = score[finite]
    local = np.argpartition(vals, -k)[-k:]
    pred[finite[local]] = True
    return pred


def topq_hits_nt(score_nt, q, start_col):
    N, T = score_nt.shape
    hit = np.zeros((N, T), dtype=bool)

    for t in range(start_col, T):
        hit[:, t] = topq_mask(score_nt[:, t], q)

    return hit


def minrun_from_hits(hit_nt, m, start_col):
    N, T = hit_nt.shape
    pred = np.zeros((N, T), dtype=bool)
    run = np.zeros(N, dtype=np.int32)

    for t in range(start_col, T):
        h = hit_nt[:, t]
        run = np.where(h, run + 1, 0)
        pred[:, t] = run >= int(m)

    return pred


def sliding_from_hits(hit_nt, ell, k, start_col):
    N, T = hit_nt.shape
    pred = np.zeros((N, T), dtype=bool)

    # Cumulative sum over time for O(NT).
    cs = np.cumsum(hit_nt.astype(np.int32), axis=1)

    for t in range(start_col, T):
        left = max(start_col, t - int(ell) + 1)

        count = cs[:, t].copy()
        if left > 0:
            count -= cs[:, left - 1]

        pred[:, t] = count >= int(k)

    return pred


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    return {
        "TPR": float(tpr),
        "FPR": float(fpr),
        "TP": tp,
        "FP": fp,
        "n_selected": int(np.sum(pred)),
    }


def best_binary_over_epoch(y, pred_nt, epochs, start_col):
    """Primary summary: highest TPR over epoch; corresponding FPR."""
    best = None

    for t in range(start_col, pred_nt.shape[1]):
        m = binary_metrics(y, pred_nt[:, t])

        row = {
            **m,
            "epoch": int(epochs[t]),
        }

        if best is None:
            best = row
            continue

        if (
            row["TPR"] > best["TPR"]
            or (
                row["TPR"] == best["TPR"]
                and row["FPR"] < best["FPR"]
            )
        ):
            best = row

    return best


def detector_stat_auc_summary(
    name,
    stat_nt,
    y,
    epochs,
    start_col,
):
    auc = auc_curve(y, stat_nt, start_col)
    b = best_auc_row(auc, epochs)

    return {
        "variant": name,
        "best_auc": b["best_auc"],
        "best_epoch": b["best_epoch"],
    }, auc


def calculate_m(T_monitor, alpha, delta):
    return int(
        math.ceil(
            math.log(T_monitor / delta)
            / (-math.log(alpha))
        )
    )


def complementarity_row(
    epoch,
    ckl_score,
    le_score,
    y,
    q,
):
    """Top-q overlap diagnostics at one epoch."""
    ckl_pred = topq_mask(ckl_score, q)
    le_pred = topq_mask(le_score, q)

    noisy = np.asarray(y, dtype=bool)

    ckl_noisy = ckl_pred & noisy
    le_noisy = le_pred & noisy

    both_noisy = int(np.sum(ckl_noisy & le_noisy))
    ckl_only_noisy = int(np.sum(ckl_noisy & ~le_pred))
    le_only_noisy = int(np.sum(le_noisy & ~ckl_pred))
    union_noisy = int(np.sum((ckl_pred | le_pred) & noisy))

    all_inter = int(np.sum(ckl_pred & le_pred))
    all_union = int(np.sum(ckl_pred | le_pred))

    return {
        "epoch": int(epoch),
        "q": float(q),
        "ckl_noisy_detected": int(np.sum(ckl_noisy)),
        "le_noisy_detected": int(np.sum(le_noisy)),
        "both_noisy": both_noisy,
        "ckl_only_noisy": ckl_only_noisy,
        "le_only_noisy": le_only_noisy,
        "union_noisy": union_noisy,
        "noisy_union_TPR": union_noisy / max(int(np.sum(noisy)), 1),
        "all_selected_jaccard": (
            all_inter / all_union
            if all_union > 0
            else np.nan
        ),
        "noisy_detected_jaccard": (
            both_noisy
            / max(
                int(np.sum(ckl_noisy | le_noisy)),
                1,
            )
        ),
    }


def plot_best_auc(score_rows, detector_rows, out_dir):
    # Score fusion plot.
    labels = [r["variant"] for r in score_rows]
    vals = [r["best_auc"] for r in score_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("CKL + LE-GIE score fusion: best score-level ROC-AUC")
    fig.tight_layout()
    fig.savefig(
        out_dir / "fig_fusion_score_best_auc.png",
        dpi=180,
    )
    plt.close(fig)

    # Temporal detector continuous-statistic AUC.
    labels = [
        f"{r['detector']}\n{r['variant']}"
        for r in detector_rows
    ]
    vals = [r["best_stat_auc"] for r in detector_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Best ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Fusion temporal detector statistic ROC-AUC")
    fig.tight_layout()
    fig.savefig(
        out_dir / "fig_fusion_detector_stat_best_auc.png",
        dpi=180,
    )
    plt.close(fig)



def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load common data once.
    d = np.load(args.common_npz, allow_pickle=False)
    loss = np.asarray(d["loss_traj"], dtype=np.float32)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)
    epochs = np.asarray(d["epoch"], dtype=np.int64)

    class_mean = exp11.build_class_mean(
        loss,
        labels,
        args.num_classes,
    )

    all_score_rows = []
    all_detector_rows = []
    all_detector_auc_rows = []
    all_improvement_rows = []
    all_complementarity_rows = []

    for K in args.K_values:
        K = int(K)
        k_dir = args.output_dir / f"K{K}"
        k_dir.mkdir(parents=True, exist_ok=True)

        start_col = K + 1
        start_epoch = int(epochs[start_col])
        T_monitor = len(epochs[start_col:])
        m = calculate_m(
            T_monitor,
            args.alpha,
            args.delta,
        )

        print()
        print("=" * 72)
        print(f"CKL + LE-GIE FUSION EXPERIMENT | K={K}")
        print("=" * 72)
        print(f"start_epoch={start_epoch}")
        print(f"T_monitor={T_monitor}")
        print(f"q={args.q}")
        print(f"rank_EWMA_lambda={args.rank_ewma_lambda}")
        print(f"min-run m={m}")
        print(
            f"sliding ell={args.sliding_ell}, "
            f"k={args.sliding_k}"
        )

        print("Computing CKL ...")
        ckl = exp11.compute_ckl(
            loss,
            labels,
            class_mean,
            K,
            args.num_classes,
        )

        print("Computing LE-GIE next/next ...")
        le = exp11.compute_le_next_next(
            loss,
            labels,
            class_mean,
            K,
        )

        print("Converting CKL and LE to within-class percentiles ...")
        ckl_pct = within_class_percentiles(
            ckl,
            labels,
            start_col,
            args.num_classes,
        )
        le_pct = within_class_percentiles(
            le,
            labels,
            start_col,
            args.num_classes,
        )

        variants = {
            "CKL_rank": ckl_pct,
            "LE_rank": le_pct,
        }

        for w in args.weights:
            if abs(w - 0.0) < 1e-12 or abs(w - 1.0) < 1e-12:
                continue
            variants[f"weighted_w{w:.2f}"] = fuse_percentiles(
                ckl_pct,
                le_pct,
                w,
            )

        variants["mean_50_50"] = fuse_percentiles(
            ckl_pct,
            le_pct,
            0.5,
        )
        variants["max_fusion"] = fuse_max(
            ckl_pct,
            le_pct,
        )

        if "weighted_w0.50" in variants:
            del variants["weighted_w0.50"]

        # --------------------------------------------------------
        # 1) Score-level AUC
        # --------------------------------------------------------
        score_auc_rows = []

        for name, score_nt in variants.items():
            auc = auc_curve(
                y,
                score_nt,
                start_col,
            )
            b = best_auc_row(auc, epochs)

            row = {
                "K": K,
                "variant": name,
                "start_epoch": start_epoch,
                "best_auc": b["best_auc"],
                "best_epoch": b["best_epoch"],
            }
            score_auc_rows.append(row)
            all_score_rows.append(dict(row))

        score_auc_rows.sort(
            key=lambda r: (
                -float(r["best_auc"])
                if np.isfinite(r["best_auc"])
                else np.inf
            )
        )

        write_csv(
            k_dir / "fusion_score_auc_summary.csv",
            score_auc_rows,
        )

        finite_auc_values = [
            float(r["best_auc"])
            for r in score_auc_rows
            if np.isfinite(r["best_auc"])
        ]
        if finite_auc_values and max(finite_auc_values) < 0.5:
            raise RuntimeError(
                f"K={K}: all score-level ROC-AUC values are below 0.5. "
                "Check score direction and AUC argument ordering."
            )

        # --------------------------------------------------------
        # 2) Complementarity
        # --------------------------------------------------------
        complementarity_rows = []
        for t in range(start_col, loss.shape[1]):
            row = complementarity_row(
                epoch=epochs[t],
                ckl_score=ckl_pct[:, t],
                le_score=le_pct[:, t],
                y=y,
                q=args.q,
            )
            row["K"] = K
            complementarity_rows.append(row)
            all_complementarity_rows.append(dict(row))

        write_csv(
            k_dir / "ckl_le_complementarity_by_epoch.csv",
            complementarity_rows,
        )

        # --------------------------------------------------------
        # 3) Temporal detectors
        # --------------------------------------------------------
        detector_rows = []
        detector_auc_rows = []

        for name, score_nt in variants.items():
            print(f"Evaluating temporal detectors: K={K}, {name}")

            hit_nt = topq_hits_nt(
                score_nt,
                args.q,
                start_col,
            )

            # min-run
            minrun_pred = minrun_from_hits(
                hit_nt,
                m,
                start_col,
            )
            best = best_binary_over_epoch(
                y,
                minrun_pred,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "min_run",
                "TPR": best["TPR"],
                "FPR": best["FPR"],
                "binary_operating_point_auc": (
                    1.0 + best["TPR"] - best["FPR"]
                ) / 2.0,
                "epoch": best["epoch"],
                "q": args.q,
                "m": m,
            }
            detector_rows.append(row)
            all_detector_rows.append(dict(row))

            run_stat = np.full(
                hit_nt.shape,
                np.nan,
                dtype=np.float32,
            )
            run = np.zeros(hit_nt.shape[0], dtype=np.int32)
            for t in range(start_col, hit_nt.shape[1]):
                run = np.where(hit_nt[:, t], run + 1, 0)
                run_stat[:, t] = run.astype(np.float32)

            stat_row, _ = detector_stat_auc_summary(
                name,
                run_stat,
                y,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "min_run_stat",
                "best_stat_auc": stat_row["best_auc"],
                "best_stat_auc_epoch": stat_row["best_epoch"],
            }
            detector_auc_rows.append(row)
            all_detector_auc_rows.append(dict(row))

            # sliding
            sliding_pred = sliding_from_hits(
                hit_nt,
                args.sliding_ell,
                args.sliding_k,
                start_col,
            )
            best = best_binary_over_epoch(
                y,
                sliding_pred,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "sliding_window",
                "TPR": best["TPR"],
                "FPR": best["FPR"],
                "binary_operating_point_auc": (
                    1.0 + best["TPR"] - best["FPR"]
                ) / 2.0,
                "epoch": best["epoch"],
                "q": args.q,
                "ell": args.sliding_ell,
                "k": args.sliding_k,
            }
            detector_rows.append(row)
            all_detector_rows.append(dict(row))

            sliding_stat = np.full(
                hit_nt.shape,
                np.nan,
                dtype=np.float32,
            )
            cs = np.cumsum(hit_nt.astype(np.int32), axis=1)
            for t in range(start_col, hit_nt.shape[1]):
                left = max(
                    start_col,
                    t - args.sliding_ell + 1,
                )
                count = cs[:, t].copy()
                if left > 0:
                    count -= cs[:, left - 1]
                sliding_stat[:, t] = count.astype(np.float32)

            stat_row, _ = detector_stat_auc_summary(
                name,
                sliding_stat,
                y,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "sliding_count_stat",
                "best_stat_auc": stat_row["best_auc"],
                "best_stat_auc_epoch": stat_row["best_epoch"],
            }
            detector_auc_rows.append(row)
            all_detector_auc_rows.append(dict(row))

            # rank-EWMA
            ewma_nt = rank_ewma(
                score_nt,
                args.rank_ewma_lambda,
                start_col,
            )
            ewma_pred = np.zeros_like(hit_nt)
            for t in range(start_col, ewma_nt.shape[1]):
                ewma_pred[:, t] = topq_mask(
                    ewma_nt[:, t],
                    args.q,
                )

            best = best_binary_over_epoch(
                y,
                ewma_pred,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "rank_ewma",
                "TPR": best["TPR"],
                "FPR": best["FPR"],
                "binary_operating_point_auc": (
                    1.0 + best["TPR"] - best["FPR"]
                ) / 2.0,
                "epoch": best["epoch"],
                "q": args.q,
                "lambda": args.rank_ewma_lambda,
            }
            detector_rows.append(row)
            all_detector_rows.append(dict(row))

            stat_row, _ = detector_stat_auc_summary(
                name,
                ewma_nt,
                y,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "rank_ewma_stat",
                "best_stat_auc": stat_row["best_auc"],
                "best_stat_auc_epoch": stat_row["best_epoch"],
            }
            detector_auc_rows.append(row)
            all_detector_auc_rows.append(dict(row))

            # cumulative pairwise
            pair = exact_pairwise_scores_by_class(
                score_nt,
                labels,
                start_index=start_col,
            )
            pair_nt = np.asarray(
                pair["cumulative_score"],
                dtype=np.float32,
            )

            pair_pred = np.zeros_like(hit_nt)
            for t in range(start_col, pair_nt.shape[1]):
                pair_pred[:, t] = topq_mask(
                    pair_nt[:, t],
                    args.q,
                )

            best = best_binary_over_epoch(
                y,
                pair_pred,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "cumulative_pairwise",
                "TPR": best["TPR"],
                "FPR": best["FPR"],
                "binary_operating_point_auc": (
                    1.0 + best["TPR"] - best["FPR"]
                ) / 2.0,
                "epoch": best["epoch"],
                "q": args.q,
            }
            detector_rows.append(row)
            all_detector_rows.append(dict(row))

            stat_row, _ = detector_stat_auc_summary(
                name,
                pair_nt,
                y,
                epochs,
                start_col,
            )
            row = {
                "K": K,
                "variant": name,
                "detector": "cumulative_pairwise_stat",
                "best_stat_auc": stat_row["best_auc"],
                "best_stat_auc_epoch": stat_row["best_epoch"],
            }
            detector_auc_rows.append(row)
            all_detector_auc_rows.append(dict(row))

            del (
                hit_nt,
                minrun_pred,
                run_stat,
                sliding_pred,
                sliding_stat,
                ewma_nt,
                ewma_pred,
                pair,
                pair_nt,
                pair_pred,
            )
            gc.collect()

        write_csv(
            k_dir / "fusion_detector_primary_summary.csv",
            detector_rows,
        )
        write_csv(
            k_dir / "fusion_detector_stat_auc_summary.csv",
            detector_auc_rows,
        )

        # --------------------------------------------------------
        # 4) Improvement over best single score
        # --------------------------------------------------------
        improvement_rows = []

        score_map = {
            r["variant"]: r
            for r in score_auc_rows
        }
        score_ref = max(
            score_map["CKL_rank"]["best_auc"],
            score_map["LE_rank"]["best_auc"],
        )

        for r in score_auc_rows:
            is_fusion = r["variant"] not in (
                "CKL_rank",
                "LE_rank",
            )
            row = {
                "K": K,
                "level": "score",
                "detector": "none",
                "variant": r["variant"],
                "metric": "best_auc",
                "value": r["best_auc"],
                "best_single_reference": score_ref,
                "improvement_over_best_single": (
                    r["best_auc"] - score_ref
                    if is_fusion
                    else np.nan
                ),
                "beats_best_single": (
                    bool(r["best_auc"] > score_ref)
                    if is_fusion
                    else False
                ),
            }
            improvement_rows.append(row)
            all_improvement_rows.append(dict(row))

        by_detector = {}
        for r in detector_auc_rows:
            by_detector.setdefault(
                r["detector"],
                {},
            )[r["variant"]] = r

        for detector, rows in by_detector.items():
            if (
                "CKL_rank" not in rows
                or "LE_rank" not in rows
            ):
                continue

            ref = max(
                rows["CKL_rank"]["best_stat_auc"],
                rows["LE_rank"]["best_stat_auc"],
            )

            for variant, r in rows.items():
                is_fusion = variant not in (
                    "CKL_rank",
                    "LE_rank",
                )
                row = {
                    "K": K,
                    "level": "detector_statistic",
                    "detector": detector,
                    "variant": variant,
                    "metric": "best_stat_auc",
                    "value": r["best_stat_auc"],
                    "best_single_reference": ref,
                    "improvement_over_best_single": (
                        r["best_stat_auc"] - ref
                        if is_fusion
                        else np.nan
                    ),
                    "beats_best_single": (
                        bool(r["best_stat_auc"] > ref)
                        if is_fusion
                        else False
                    ),
                }
                improvement_rows.append(row)
                all_improvement_rows.append(dict(row))

        write_csv(
            k_dir / "fusion_improvement_summary.csv",
            improvement_rows,
        )

        plot_best_auc(
            score_auc_rows,
            detector_auc_rows,
            k_dir,
        )

        print()
        print(f"K={K} SCORE-LEVEL BEST AUC")
        for r in score_auc_rows:
            print(
                f"{r['variant']:>18s} | "
                f"AUC={r['best_auc']:.6f} "
                f"epoch={r['best_epoch']}"
            )

        # Free large K-specific arrays before next K.
        del ckl, le, ckl_pct, le_pct, variants
        gc.collect()

    # Combined summaries across K=20 and K=40.
    write_csv(
        args.output_dir / "all_K_fusion_score_auc_summary.csv",
        all_score_rows,
    )
    write_csv(
        args.output_dir / "all_K_fusion_detector_primary_summary.csv",
        all_detector_rows,
    )
    write_csv(
        args.output_dir / "all_K_fusion_detector_stat_auc_summary.csv",
        all_detector_auc_rows,
    )
    write_csv(
        args.output_dir / "all_K_fusion_improvement_summary.csv",
        all_improvement_rows,
    )
    write_csv(
        args.output_dir / "all_K_ckl_le_complementarity_by_epoch.csv",
        all_complementarity_rows,
    )

    config = {
        "artifact": "CKL_LE_score_fusion_noise05_multiK",
        "common_npz": str(args.common_npz),
        "K_values": [int(k) for k in args.K_values],
        "noise_rate_expected": 0.05,
        "CKL_limit_rule": "last-three mean",
        "LE_formula": "signed ell_err + log(abs(m_GIE))",
        "LE_error_limit_rule": "next sample",
        "LE_GIE_limit_rule": "next sample",
        "fusion_input":
            "within-observed-class percentile of each source score",
        "weighted_fusion":
            "w*r_CKL + (1-w)*r_LE",
        "weights": [float(v) for v in args.weights],
        "max_fusion":
            "max(r_CKL,r_LE)",
        "q": args.q,
        "rank_ewma_lambda": args.rank_ewma_lambda,
        "alpha": args.alpha,
        "delta": args.delta,
        "sliding_ell": args.sliding_ell,
        "sliding_k": args.sliding_k,
        "evaluation_note":
            "synthetic anomaly mask is used only for evaluation, never for fusion",
        "selection_note":
            "K and fusion-weight comparisons are exploratory on this labeled 5% run; freeze selected settings before independent validation",
    }

    with (
        args.output_dir / "fusion_multiK_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 72)
    print("FINISHED K COMPARISON")
    print("=" * 72)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
