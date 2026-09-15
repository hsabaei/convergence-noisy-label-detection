#!/usr/bin/env python
"""
TPR/FPR-over-epoch study for CKL, LE-GIE, and CKL+LE fusion.

Noise levels:
    5%, 10%, 20%

K values:
    20, 40

Frozen fusion:
    weighted_w0.75 = 0.75 * r_CKL + 0.25 * r_LE

Also evaluates:
    CKL_rank
    LE_rank
    mean_50_50
    max_fusion

Temporal detectors:
    min_run
    sliding_window
    rank_ewma
    cumulative_pairwise

Outputs:
    - one long CSV with TPR/FPR at every epoch
    - per-noise/per-K/per-detector TPR plots
    - per-noise/per-K/per-detector FPR plots
    - compact plots focused on CKL_rank, LE_rank, weighted_w0.75

Ground-truth noise mask is used only for evaluation.
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
)

# Reuse validated estimator implementations.
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
        "--npz-05",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument(
        "--npz-10",
        type=Path,
        default=Path(
            "results/common_loss_trajectories_noise10/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument(
        "--npz-20",
        type=Path,
        default=Path(
            "results/common_loss_trajectories_noise20/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/fusion_tpr_fpr_over_epochs_noise05_10_20"
        ),
    )

    p.add_argument(
        "--K-values",
        type=int,
        nargs="+",
        default=(20, 40),
    )

    p.add_argument("--num-classes", type=int, default=10)

    # Frozen operating settings.
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--rank-ewma-lambda", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--sliding-ell", type=int, default=30)
    p.add_argument("--sliding-k", type=int, default=15)

    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def calculate_m(T_monitor, alpha, delta):
    return int(
        math.ceil(
            math.log(T_monitor / delta)
            / (-math.log(alpha))
        )
    )


def within_class_percentiles(
    score_nt,
    labels,
    start_col,
    num_classes,
):
    """Epoch-wise percentile within observed class."""
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
            order = np.argsort(vals, kind="mergesort")
            sorted_vals = vals[order]

            rank_sorted = np.empty(idx.size, dtype=np.float64)

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

            rank = np.empty(idx.size, dtype=np.float64)
            rank[order] = rank_sorted

            if idx.size == 1:
                pct = np.ones(1, dtype=np.float64)
            else:
                pct = rank / float(idx.size - 1)

            out[idx, t] = pct.astype(np.float32)

    return out


def fuse_weighted(ckl_pct, le_pct, w):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)
    out[valid] = (
        float(w) * ckl_pct[valid]
        + (1.0 - float(w)) * le_pct[valid]
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
        run = np.where(hit_nt[:, t], run + 1, 0)
        pred[:, t] = run >= int(m)

    return pred


def sliding_from_hits(hit_nt, ell, k, start_col):
    N, T = hit_nt.shape
    pred = np.zeros((N, T), dtype=bool)

    cs = np.cumsum(hit_nt.astype(np.int32), axis=1)

    for t in range(start_col, T):
        left = max(start_col, t - int(ell) + 1)

        count = cs[:, t].copy()
        if left > 0:
            count -= cs[:, left - 1]

        pred[:, t] = count >= int(k)

    return pred


def topq_from_stat_nt(stat_nt, q, start_col):
    pred = np.zeros(stat_nt.shape, dtype=bool)
    for t in range(start_col, stat_nt.shape[1]):
        pred[:, t] = topq_mask(stat_nt[:, t], q)
    return pred


def metrics_over_time(
    y,
    pred_nt,
    epochs,
    start_col,
    noise_rate,
    K,
    variant,
    detector,
):
    y = np.asarray(y, dtype=bool)
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))

    rows = []

    for t in range(start_col, pred_nt.shape[1]):
        pred = pred_nt[:, t]

        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))

        rows.append({
            "noise_rate": float(noise_rate),
            "K": int(K),
            "variant": variant,
            "detector": detector,
            "epoch": int(epochs[t]),
            "TPR": tp / max(n_pos, 1),
            "FPR": fp / max(n_neg, 1),
            "TP": tp,
            "FP": fp,
            "n_selected": int(np.sum(pred)),
        })

    return rows


def plot_metric(
    rows,
    metric,
    noise_rate,
    K,
    detector,
    out_path,
    variants,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in variants:
        sub = [
            r for r in rows
            if (
                r["noise_rate"] == noise_rate
                and r["K"] == K
                and r["detector"] == detector
                and r["variant"] == variant
            )
        ]

        if not sub:
            continue

        sub = sorted(sub, key=lambda r: r["epoch"])
        x = [r["epoch"] for r in sub]
        y = [r[metric] for r in sub]

        ax.plot(x, y, marker=None, label=variant)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)

    if metric == "FPR":
        ax.axhline(
            0.05,
            linestyle="--",
            linewidth=1.2,
            label="FPR=0.05",
        )

    ax.set_title(
        f"{metric} over epochs | noise={int(noise_rate*100)}% "
        f"| K={K} | {detector}"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_compact_all_detectors(
    rows,
    metric,
    noise_rate,
    K,
    out_path,
):
    """Compact figure with one line per detector for the frozen 0.75 fusion."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for detector in (
        "min_run",
        "sliding_window",
        "rank_ewma",
        "cumulative_pairwise",
    ):
        sub = [
            r for r in rows
            if (
                r["noise_rate"] == noise_rate
                and r["K"] == K
                and r["variant"] == "weighted_w0.75"
                and r["detector"] == detector
            )
        ]
        if not sub:
            continue

        sub = sorted(sub, key=lambda r: r["epoch"])
        ax.plot(
            [r["epoch"] for r in sub],
            [r[metric] for r in sub],
            label=detector,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)

    if metric == "FPR":
        ax.axhline(
            0.05,
            linestyle="--",
            linewidth=1.2,
            label="FPR=0.05",
        )

    ax.set_title(
        f"Frozen 0.75 CKL + 0.25 LE fusion: {metric} "
        f"| noise={int(noise_rate*100)}% | K={K}"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (0.05, args.npz_05),
        (0.10, args.npz_10),
        (0.20, args.npz_20),
    ]

    all_rows = []

    variants_to_plot = [
        "CKL_rank",
        "LE_rank",
        "weighted_w0.75",
        "mean_50_50",
        "max_fusion",
    ]

    core_variants = [
        "CKL_rank",
        "LE_rank",
        "weighted_w0.75",
    ]

    detectors = [
        "min_run",
        "sliding_window",
        "rank_ewma",
        "cumulative_pairwise",
    ]

    for noise_rate, npz_path in datasets:
        print()
        print("=" * 76)
        print(f"NOISE = {int(noise_rate * 100)}%")
        print("=" * 76)

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing required trajectory file: {npz_path}"
            )

        d = np.load(npz_path, allow_pickle=False)

        loss = np.asarray(d["loss_traj"], dtype=np.float32)
        labels = np.asarray(d["observed_label"], dtype=np.int64)
        y = np.asarray(d["is_anomaly"], dtype=bool)
        epochs = np.asarray(d["epoch"], dtype=np.int64)

        observed_rate = float(np.mean(y))
        print(
            f"Loaded {npz_path} | "
            f"observed noise fraction={observed_rate:.4f}"
        )

        class_mean = exp11.build_class_mean(
            loss,
            labels,
            args.num_classes,
        )

        for K in args.K_values:
            K = int(K)
            start_col = K + 1
            start_epoch = int(epochs[start_col])
            T_monitor = len(epochs[start_col:])

            m = calculate_m(
                T_monitor,
                args.alpha,
                args.delta,
            )

            print()
            print("-" * 76)
            print(
                f"noise={int(noise_rate*100)}% | K={K} "
                f"| start_epoch={start_epoch} | m={m}"
            )
            print("-" * 76)

            ckl = exp11.compute_ckl(
                loss,
                labels,
                class_mean,
                K,
                args.num_classes,
            )

            le = exp11.compute_le_next_next(
                loss,
                labels,
                class_mean,
                K,
            )

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
                "weighted_w0.75": fuse_weighted(
                    ckl_pct,
                    le_pct,
                    0.75,
                ),
                "mean_50_50": fuse_weighted(
                    ckl_pct,
                    le_pct,
                    0.50,
                ),
                "max_fusion": fuse_max(
                    ckl_pct,
                    le_pct,
                ),
            }

            for variant_name, score_nt in variants.items():
                print(
                    f"  detector trajectories: {variant_name}"
                )

                hit_nt = topq_hits_nt(
                    score_nt,
                    args.q,
                    start_col,
                )

                # 1) min-run
                pred = minrun_from_hits(
                    hit_nt,
                    m,
                    start_col,
                )
                all_rows.extend(
                    metrics_over_time(
                        y,
                        pred,
                        epochs,
                        start_col,
                        noise_rate,
                        K,
                        variant_name,
                        "min_run",
                    )
                )
                del pred

                # 2) sliding window
                pred = sliding_from_hits(
                    hit_nt,
                    args.sliding_ell,
                    args.sliding_k,
                    start_col,
                )
                all_rows.extend(
                    metrics_over_time(
                        y,
                        pred,
                        epochs,
                        start_col,
                        noise_rate,
                        K,
                        variant_name,
                        "sliding_window",
                    )
                )
                del pred

                # 3) rank-EWMA
                ewma = rank_ewma(
                    score_nt,
                    args.rank_ewma_lambda,
                    start_col,
                )
                pred = topq_from_stat_nt(
                    ewma,
                    args.q,
                    start_col,
                )
                all_rows.extend(
                    metrics_over_time(
                        y,
                        pred,
                        epochs,
                        start_col,
                        noise_rate,
                        K,
                        variant_name,
                        "rank_ewma",
                    )
                )
                del ewma, pred

                # 4) cumulative pairwise
                pair = exact_pairwise_scores_by_class(
                    score_nt,
                    labels,
                    start_index=start_col,
                )
                pair_nt = np.asarray(
                    pair["cumulative_score"],
                    dtype=np.float32,
                )
                pred = topq_from_stat_nt(
                    pair_nt,
                    args.q,
                    start_col,
                )
                all_rows.extend(
                    metrics_over_time(
                        y,
                        pred,
                        epochs,
                        start_col,
                        noise_rate,
                        K,
                        variant_name,
                        "cumulative_pairwise",
                    )
                )

                del hit_nt, pair, pair_nt, pred
                gc.collect()

            # Save incremental CSV after every noise/K block.
            write_csv(
                args.output_dir
                / "tpr_fpr_over_epochs_all.csv",
                all_rows,
            )

            # Plot every detector: all variants.
            block_dir = (
                args.output_dir
                / f"noise{int(noise_rate*100):02d}"
                / f"K{K}"
            )
            block_dir.mkdir(parents=True, exist_ok=True)

            for detector in detectors:
                plot_metric(
                    all_rows,
                    "TPR",
                    noise_rate,
                    K,
                    detector,
                    block_dir
                    / f"TPR_over_epochs_{detector}_all_variants.png",
                    variants_to_plot,
                )

                plot_metric(
                    all_rows,
                    "FPR",
                    noise_rate,
                    K,
                    detector,
                    block_dir
                    / f"FPR_over_epochs_{detector}_all_variants.png",
                    variants_to_plot,
                )

                # Cleaner 3-line plot: CKL, LE, frozen fusion.
                plot_metric(
                    all_rows,
                    "TPR",
                    noise_rate,
                    K,
                    detector,
                    block_dir
                    / f"TPR_over_epochs_{detector}_core.png",
                    core_variants,
                )

                plot_metric(
                    all_rows,
                    "FPR",
                    noise_rate,
                    K,
                    detector,
                    block_dir
                    / f"FPR_over_epochs_{detector}_core.png",
                    core_variants,
                )

            # Compact frozen-fusion plot across detectors.
            plot_compact_all_detectors(
                all_rows,
                "TPR",
                noise_rate,
                K,
                block_dir
                / "TPR_over_epochs_frozen_w075_all_detectors.png",
            )

            plot_compact_all_detectors(
                all_rows,
                "FPR",
                noise_rate,
                K,
                block_dir
                / "FPR_over_epochs_frozen_w075_all_detectors.png",
            )

            del ckl, le, ckl_pct, le_pct, variants
            gc.collect()

        del loss, labels, y, epochs, class_mean, d
        gc.collect()

    # ------------------------------------------------------------
    # Best-over-epoch summary for convenience.
    # ------------------------------------------------------------
    best_rows = []

    keys = sorted(
        set(
            (
                r["noise_rate"],
                r["K"],
                r["variant"],
                r["detector"],
            )
            for r in all_rows
        )
    )

    for noise_rate, K, variant, detector in keys:
        sub = [
            r for r in all_rows
            if (
                r["noise_rate"] == noise_rate
                and r["K"] == K
                and r["variant"] == variant
                and r["detector"] == detector
            )
        ]

        # Same primary convention as previous experiments:
        # highest TPR over epoch, report the FPR at that same epoch.
        sub.sort(
            key=lambda r: (
                -r["TPR"],
                r["FPR"],
                r["epoch"],
            )
        )
        best = sub[0]

        best_rows.append({
            "noise_rate": noise_rate,
            "K": K,
            "variant": variant,
            "detector": detector,
            "best_TPR": best["TPR"],
            "corresponding_FPR": best["FPR"],
            "epoch": best["epoch"],
        })

    write_csv(
        args.output_dir
        / "best_over_epoch_summary.csv",
        best_rows,
    )

    config = {
        "artifact":
            "fusion_TPR_FPR_over_epochs_noise05_10_20",
        "noise_rates": [0.05, 0.10, 0.20],
        "K_values": [int(k) for k in args.K_values],
        "CKL_limit_rule": "last-three mean",
        "LE_formula": "signed ell_err + log(abs(m_GIE))",
        "LE_error_limit_rule": "next sample",
        "LE_GIE_limit_rule": "next sample",
        "frozen_fusion":
            "0.75*r_CKL + 0.25*r_LE",
        "fusion_input":
            "within-observed-class percentile",
        "additional_fusions": [
            "0.5*r_CKL + 0.5*r_LE",
            "max(r_CKL,r_LE)",
        ],
        "q": args.q,
        "rank_ewma_lambda": args.rank_ewma_lambda,
        "alpha": args.alpha,
        "delta": args.delta,
        "sliding_ell": args.sliding_ell,
        "sliding_k": args.sliding_k,
        "evaluation_note":
            "known synthetic noise mask is used only to compute TPR/FPR",
        "important_q_note":
            "fixed q=0.10 imposes a maximum achievable TPR of 0.5 when true noise rate is 20%",
    }

    with (
        args.output_dir / "config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 76)
    print("DONE")
    print("=" * 76)
    print(
        args.output_dir
        / "tpr_fpr_over_epochs_all.csv"
    )
    print(
        args.output_dir
        / "best_over_epoch_summary.csv"
    )
    print(f"Plots under: {args.output_dir}/noiseXX/KXX/")


if __name__ == "__main__":
    main()
