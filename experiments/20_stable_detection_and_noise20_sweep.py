#!/usr/bin/env python
"""
Stable early-detection analysis + dedicated 20%-noise hyperparameter sweep.

Part A
------
Evaluate CKL, LE-GIE, and selected fusion rules on 5%, 10%, 20% noise
using an EARLIEST STABLE DETECTION criterion.

The stable target is defined relative to the attainable TPR under the
selection budget q:

    TPR_max(q, pi) = min(1, q / pi)

    target_TPR = target_fraction * TPR_max

Default:
    target_fraction = 0.90
    stable_horizon = 5 epochs

A stable detection time is the first epoch t for which TPR remains above
target_TPR for the next h monitored epochs.

FPR is NOT used as a hard exclusion during this timing comparison because,
for example, q=0.10 with pi=0.05 mathematically implies FPR > 0.05 even at
TPR=1 for exact top-q detectors. FPR is instead reported and used as a
secondary/tie-breaking diagnostic.

Part B
------
For 20% noise at K=40, sweep:

    q in {0.10, 0.15, 0.20, 0.25}

Min-run:
    m(q) = ceil(log(T_monitor/delta) / -log(q))

Sliding-window:
    ell in {10, 20, 30, 40}
    rho in {0.30, 0.50, 0.67, 0.80}
    k = ceil(rho * ell)

Rank-EWMA:
    lambda in {0.02, 0.05, 0.10, 0.20}

Cumulative pairwise:
    no internal temporal hyperparameter; sweep q only.

Score variants:
    CKL_rank
    LE_rank
    weighted_w0.75
    prob_or
    product

Important:
Ground-truth noise labels are used only for evaluation/selection in this
exploratory study. Any chosen setting should be frozen before independent
validation.
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

from convergence_monitoring.detectors import exact_pairwise_scores_by_class


# Reuse the existing estimator implementations.
P11 = ROOT / "experiments" / "11_compare_ckl_vs_le_across_k.py"
if not P11.exists():
    raise FileNotFoundError(
        "Expected experiments/11_compare_ckl_vs_le_across_k.py"
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
            "results/stable_detection_and_noise20_sweep"
        ),
    )

    p.add_argument(
        "--K-values",
        type=int,
        nargs="+",
        default=(20, 40),
    )

    p.add_argument("--num-classes", type=int, default=10)

    # Part A frozen reference settings.
    p.add_argument("--reference-q", type=float, default=0.10)
    p.add_argument("--reference-ewma-lambda", type=float, default=0.05)
    p.add_argument("--reference-sliding-ell", type=int, default=30)
    p.add_argument("--reference-sliding-k", type=int, default=15)

    # Stable-detection definition.
    p.add_argument("--target-fraction", type=float, default=0.90)
    p.add_argument("--stable-horizon", type=int, default=5)

    # General calibration.
    p.add_argument("--delta", type=float, default=0.001)

    # Part B: 20% sweep.
    p.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=(0.10, 0.15, 0.20, 0.25),
    )
    p.add_argument(
        "--ewma-lambdas",
        type=float,
        nargs="+",
        default=(0.02, 0.05, 0.10, 0.20),
    )
    p.add_argument(
        "--sliding-ells",
        type=int,
        nargs="+",
        default=(10, 20, 30, 40),
    )
    p.add_argument(
        "--sliding-rhos",
        type=float,
        nargs="+",
        default=(0.30, 0.50, 0.67, 0.80),
    )

    return p.parse_args()


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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def calculate_m(T_monitor, q, delta):
    """Min-run length using alpha=q for rank top-q hits."""
    return int(
        math.ceil(
            math.log(T_monitor / delta)
            / (-math.log(q))
        )
    )


def attainable_tpr_max(q, noise_rate):
    if noise_rate <= 0:
        return 1.0
    return min(1.0, float(q) / float(noise_rate))


def stable_target_tpr(q, noise_rate, target_fraction):
    return float(target_fraction) * attainable_tpr_max(
        q,
        noise_rate,
    )


def within_class_percentiles(
    score_nt,
    labels,
    start_col,
    num_classes,
):
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

                rank_sorted[a:b] = 0.5 * (a + b - 1)
                a = b

            rank = np.empty(idx.size, dtype=np.float64)
            rank[order] = rank_sorted

            if idx.size == 1:
                pct = np.ones(1, dtype=np.float64)
            else:
                pct = rank / float(idx.size - 1)

            out[idx, t] = pct.astype(np.float32)

    return out


def fuse_weighted(ckl_pct, le_pct, w=0.75):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)
    out[valid] = (
        float(w) * ckl_pct[valid]
        + (1.0 - float(w)) * le_pct[valid]
    ).astype(np.float32)
    return out


def fuse_prob_or(ckl_pct, le_pct):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)

    c = np.clip(ckl_pct[valid], 0.0, 1.0)
    l = np.clip(le_pct[valid], 0.0, 1.0)

    out[valid] = (
        1.0 - (1.0 - c) * (1.0 - l)
    ).astype(np.float32)
    return out


def fuse_product(ckl_pct, le_pct):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)

    out[valid] = (
        np.clip(ckl_pct[valid], 0.0, 1.0)
        * np.clip(le_pct[valid], 0.0, 1.0)
    ).astype(np.float32)
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
    out = np.zeros(score_nt.shape, dtype=bool)
    for t in range(start_col, score_nt.shape[1]):
        out[:, t] = topq_mask(score_nt[:, t], q)
    return out


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


def topq_from_stat_nt(stat_nt, q, start_col):
    pred = np.zeros(stat_nt.shape, dtype=bool)
    for t in range(start_col, stat_nt.shape[1]):
        pred[:, t] = topq_mask(stat_nt[:, t], q)
    return pred


def trajectory_metrics(
    y,
    pred_nt,
    epochs,
    start_col,
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
            "epoch": int(epochs[t]),
            "TPR": tp / max(n_pos, 1),
            "FPR": fp / max(n_neg, 1),
            "TP": tp,
            "FP": fp,
            "n_selected": int(np.sum(pred)),
        })

    return rows


def summarize_stable(
    rows,
    target_tpr,
    stable_horizon,
    sustained_start=60,
    sustained_end=120,
    late_start=180,
    late_end=200,
):
    """Summarize earliest stable detection and temporal quality."""
    if not rows:
        return {}

    rows = sorted(rows, key=lambda r: r["epoch"])

    epoch = np.asarray([r["epoch"] for r in rows], dtype=np.int64)
    tpr = np.asarray([r["TPR"] for r in rows], dtype=np.float64)
    fpr = np.asarray([r["FPR"] for r in rows], dtype=np.float64)

    h = int(stable_horizon)
    first_idx = -1

    for i in range(0, len(rows) - h + 1):
        if np.all(tpr[i : i + h] >= float(target_tpr)):
            first_idx = i
            break

    out = {
        "target_TPR": float(target_tpr),
        "stable_horizon": h,
        "first_stable_epoch": (
            int(epoch[first_idx])
            if first_idx >= 0
            else -1
        ),
    }

    if first_idx >= 0:
        sl = slice(first_idx, first_idx + h)
        out["stable_window_mean_TPR"] = float(np.mean(tpr[sl]))
        out["stable_window_min_TPR"] = float(np.min(tpr[sl]))
        out["stable_window_mean_FPR"] = float(np.mean(fpr[sl]))
        out["stable_window_max_FPR"] = float(np.max(fpr[sl]))
    else:
        out["stable_window_mean_TPR"] = np.nan
        out["stable_window_min_TPR"] = np.nan
        out["stable_window_mean_FPR"] = np.nan
        out["stable_window_max_FPR"] = np.nan

    peak_idx = int(np.argmax(tpr))
    out["peak_TPR"] = float(tpr[peak_idx])
    out["peak_epoch"] = int(epoch[peak_idx])
    out["FPR_at_peak"] = float(fpr[peak_idx])

    out["final_TPR"] = float(tpr[-1])
    out["final_FPR"] = float(fpr[-1])
    out["peak_to_final_decay"] = float(
        tpr[peak_idx] - tpr[-1]
    )

    sustained = (
        (epoch >= sustained_start)
        & (epoch <= sustained_end)
    )
    if np.any(sustained):
        out["sustained_mean_TPR"] = float(
            np.mean(tpr[sustained])
        )
        out["sustained_mean_FPR"] = float(
            np.mean(fpr[sustained])
        )
    else:
        out["sustained_mean_TPR"] = np.nan
        out["sustained_mean_FPR"] = np.nan

    late = (
        (epoch >= late_start)
        & (epoch <= late_end)
    )
    if np.any(late):
        out["late_mean_TPR"] = float(np.mean(tpr[late]))
        out["late_mean_FPR"] = float(np.mean(fpr[late]))
    else:
        out["late_mean_TPR"] = np.nan
        out["late_mean_FPR"] = np.nan

    # Average temporal quality across the monitored period.
    out["mean_TPR_over_monitoring"] = float(np.mean(tpr))
    out["mean_FPR_over_monitoring"] = float(np.mean(fpr))

    return out


def stable_sort_key(row):
    """Practical exploratory ranking: early stable detection first."""
    e = int(row.get("first_stable_epoch", -1))
    return (
        0 if e >= 0 else 1,
        e if e >= 0 else 10**9,
        float(row.get("stable_window_mean_FPR", np.inf)),
        -float(row.get("sustained_mean_TPR", -np.inf)),
        float(row.get("peak_to_final_decay", np.inf)),
    )


def build_score_variants(
    loss,
    labels,
    K,
    num_classes,
    start_col,
):
    class_mean = exp11.build_class_mean(
        loss,
        labels,
        num_classes,
    )

    ckl = exp11.compute_ckl(
        loss,
        labels,
        class_mean,
        K,
        num_classes,
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
        num_classes,
    )
    le_pct = within_class_percentiles(
        le,
        labels,
        start_col,
        num_classes,
    )

    variants = {
        "CKL_rank": ckl_pct,
        "LE_rank": le_pct,
        "weighted_w0.75": fuse_weighted(
            ckl_pct,
            le_pct,
            0.75,
        ),
        "prob_or": fuse_prob_or(
            ckl_pct,
            le_pct,
        ),
        "product": fuse_product(
            ckl_pct,
            le_pct,
        ),
    }

    return variants


def evaluate_detector_configuration(
    y,
    epochs,
    score_nt,
    labels,
    start_col,
    detector,
    q,
    *,
    m=None,
    ell=None,
    k=None,
    lam=None,
):
    hit_nt = None
    aux = None

    if detector == "min_run":
        hit_nt = topq_hits_nt(
            score_nt,
            q,
            start_col,
        )
        pred = minrun_from_hits(
            hit_nt,
            m,
            start_col,
        )

    elif detector == "sliding_window":
        hit_nt = topq_hits_nt(
            score_nt,
            q,
            start_col,
        )
        pred = sliding_from_hits(
            hit_nt,
            ell,
            k,
            start_col,
        )

    elif detector == "rank_ewma":
        aux = rank_ewma(
            score_nt,
            lam,
            start_col,
        )
        pred = topq_from_stat_nt(
            aux,
            q,
            start_col,
        )

    elif detector == "cumulative_pairwise":
        aux = exact_pairwise_scores_by_class(
            score_nt,
            labels,
            start_index=start_col,
        )
        pair_nt = np.asarray(
            aux["cumulative_score"],
            dtype=np.float32,
        )
        pred = topq_from_stat_nt(
            pair_nt,
            q,
            start_col,
        )
        aux = pair_nt

    else:
        raise ValueError(detector)

    rows = trajectory_metrics(
        y,
        pred,
        epochs,
        start_col,
    )

    del pred
    if hit_nt is not None:
        del hit_nt
    if aux is not None:
        del aux
    gc.collect()

    return rows


def plot_best_configs(
    all_traj_rows,
    best_rows,
    output_dir,
):
    """Plot best configuration per detector at 20% noise."""
    detectors = [
        "min_run",
        "sliding_window",
        "rank_ewma",
        "cumulative_pairwise",
    ]

    for metric in ("TPR", "FPR"):
        fig, ax = plt.subplots(figsize=(10, 6))

        for detector in detectors:
            candidates = [
                r for r in best_rows
                if r["detector"] == detector
            ]
            if not candidates:
                continue

            best = sorted(
                candidates,
                key=stable_sort_key,
            )[0]

            sub = [
                r for r in all_traj_rows
                if (
                    r["detector"] == detector
                    and r["variant"] == best["variant"]
                    and r["q"] == best["q"]
                    and r.get("m") == best.get("m")
                    and r.get("ell") == best.get("ell")
                    and r.get("k") == best.get("k")
                    and r.get("lambda") == best.get("lambda")
                )
            ]
            sub.sort(key=lambda r: r["epoch"])

            label = (
                f"{detector} | {best['variant']} | "
                f"q={best['q']}"
            )

            ax.plot(
                [r["epoch"] for r in sub],
                [r[metric] for r in sub],
                label=label,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"20% noise, K=40: best stable configurations ({metric})"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"noise20_best_stable_configs_{metric}.png",
            dpi=180,
        )
        plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (0.05, args.npz_05),
        (0.10, args.npz_10),
        (0.20, args.npz_20),
    ]

    # ============================================================
    # Part A: stable early-detection summary at frozen settings.
    # ============================================================
    frozen_summary = []

    for noise_rate, npz_path in datasets:
        print()
        print("=" * 76)
        print(
            f"PART A | stable detection | noise={noise_rate:.2f}"
        )
        print("=" * 76)

        d = np.load(npz_path, allow_pickle=False)

        loss = np.asarray(d["loss_traj"], dtype=np.float32)
        labels = np.asarray(d["observed_label"], dtype=np.int64)
        y = np.asarray(d["is_anomaly"], dtype=bool)
        epochs = np.asarray(d["epoch"], dtype=np.int64)

        observed_noise = float(np.mean(y))
        print(f"Observed noise fraction = {observed_noise:.4f}")

        for K in args.K_values:
            K = int(K)
            start_col = K + 1
            T_monitor = len(epochs[start_col:])

            q = float(args.reference_q)
            m = calculate_m(
                T_monitor,
                q,
                args.delta,
            )

            target_tpr = stable_target_tpr(
                q,
                noise_rate,
                args.target_fraction,
            )

            variants = build_score_variants(
                loss,
                labels,
                K,
                args.num_classes,
                start_col,
            )

            for variant, score_nt in variants.items():
                configs = [
                    (
                        "min_run",
                        {
                            "m": m,
                        },
                    ),
                    (
                        "sliding_window",
                        {
                            "ell": args.reference_sliding_ell,
                            "k": args.reference_sliding_k,
                        },
                    ),
                    (
                        "rank_ewma",
                        {
                            "lam": args.reference_ewma_lambda,
                        },
                    ),
                    (
                        "cumulative_pairwise",
                        {},
                    ),
                ]

                for detector, hp in configs:
                    traj = evaluate_detector_configuration(
                        y,
                        epochs,
                        score_nt,
                        labels,
                        start_col,
                        detector,
                        q,
                        **hp,
                    )

                    summary = summarize_stable(
                        traj,
                        target_tpr,
                        args.stable_horizon,
                    )

                    frozen_summary.append({
                        "noise_rate": noise_rate,
                        "K": K,
                        "variant": variant,
                        "detector": detector,
                        "q": q,
                        "attainable_TPR_max":
                            attainable_tpr_max(q, noise_rate),
                        "target_fraction":
                            args.target_fraction,
                        "m": hp.get("m"),
                        "ell": hp.get("ell"),
                        "k": hp.get("k"),
                        "lambda": hp.get("lam"),
                        **summary,
                    })

            del variants
            gc.collect()

        del d, loss, labels, y, epochs
        gc.collect()

    frozen_summary.sort(
        key=lambda r: (
            r["noise_rate"],
            r["K"],
            *stable_sort_key(r),
        )
    )

    write_csv(
        args.output_dir
        / "stable_detection_frozen_settings.csv",
        frozen_summary,
    )

    # ============================================================
    # Part B: 20% dedicated q + detector hyperparameter sweep.
    # ============================================================
    print()
    print("=" * 76)
    print("PART B | 20% noise | K=40 hyperparameter sweep")
    print("=" * 76)

    noise_rate = 0.20
    K = 40
    d = np.load(args.npz_20, allow_pickle=False)

    loss = np.asarray(d["loss_traj"], dtype=np.float32)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)
    epochs = np.asarray(d["epoch"], dtype=np.int64)

    start_col = K + 1
    T_monitor = len(epochs[start_col:])

    variants = build_score_variants(
        loss,
        labels,
        K,
        args.num_classes,
        start_col,
    )

    sweep_summary = []
    sweep_traj_rows = []

    for variant, score_nt in variants.items():
        print(f"Score variant: {variant}")

        for q in args.q_values:
            q = float(q)
            target_tpr = stable_target_tpr(
                q,
                noise_rate,
                args.target_fraction,
            )

            # ----------------------------------------------------
            # Min-run: m derived from q.
            # ----------------------------------------------------
            m = calculate_m(
                T_monitor,
                q,
                args.delta,
            )

            traj = evaluate_detector_configuration(
                y,
                epochs,
                score_nt,
                labels,
                start_col,
                "min_run",
                q,
                m=m,
            )

            summary = summarize_stable(
                traj,
                target_tpr,
                args.stable_horizon,
            )

            row = {
                "noise_rate": noise_rate,
                "K": K,
                "variant": variant,
                "detector": "min_run",
                "q": q,
                "attainable_TPR_max":
                    attainable_tpr_max(q, noise_rate),
                "target_fraction":
                    args.target_fraction,
                "m": m,
                "ell": None,
                "k": None,
                "rho": None,
                "lambda": None,
                **summary,
            }
            sweep_summary.append(row)

            for r in traj:
                sweep_traj_rows.append({
                    **r,
                    "variant": variant,
                    "detector": "min_run",
                    "q": q,
                    "m": m,
                    "ell": None,
                    "k": None,
                    "rho": None,
                    "lambda": None,
                })

            # ----------------------------------------------------
            # Sliding-window sweep.
            # ----------------------------------------------------
            for ell in args.sliding_ells:
                for rho in args.sliding_rhos:
                    k = int(math.ceil(float(rho) * int(ell)))

                    traj = evaluate_detector_configuration(
                        y,
                        epochs,
                        score_nt,
                        labels,
                        start_col,
                        "sliding_window",
                        q,
                        ell=int(ell),
                        k=k,
                    )

                    summary = summarize_stable(
                        traj,
                        target_tpr,
                        args.stable_horizon,
                    )

                    row = {
                        "noise_rate": noise_rate,
                        "K": K,
                        "variant": variant,
                        "detector": "sliding_window",
                        "q": q,
                        "attainable_TPR_max":
                            attainable_tpr_max(q, noise_rate),
                        "target_fraction":
                            args.target_fraction,
                        "m": None,
                        "ell": int(ell),
                        "k": k,
                        "rho": float(rho),
                        "lambda": None,
                        **summary,
                    }
                    sweep_summary.append(row)

                    for r in traj:
                        sweep_traj_rows.append({
                            **r,
                            "variant": variant,
                            "detector": "sliding_window",
                            "q": q,
                            "m": None,
                            "ell": int(ell),
                            "k": k,
                            "rho": float(rho),
                            "lambda": None,
                        })

            # ----------------------------------------------------
            # Rank-EWMA sweep.
            # ----------------------------------------------------
            for lam in args.ewma_lambdas:
                traj = evaluate_detector_configuration(
                    y,
                    epochs,
                    score_nt,
                    labels,
                    start_col,
                    "rank_ewma",
                    q,
                    lam=float(lam),
                )

                summary = summarize_stable(
                    traj,
                    target_tpr,
                    args.stable_horizon,
                )

                row = {
                    "noise_rate": noise_rate,
                    "K": K,
                    "variant": variant,
                    "detector": "rank_ewma",
                    "q": q,
                    "attainable_TPR_max":
                        attainable_tpr_max(q, noise_rate),
                    "target_fraction":
                        args.target_fraction,
                    "m": None,
                    "ell": None,
                    "k": None,
                    "rho": None,
                    "lambda": float(lam),
                    **summary,
                }
                sweep_summary.append(row)

                for r in traj:
                    sweep_traj_rows.append({
                        **r,
                        "variant": variant,
                        "detector": "rank_ewma",
                        "q": q,
                        "m": None,
                        "ell": None,
                        "k": None,
                        "rho": None,
                        "lambda": float(lam),
                    })

            # ----------------------------------------------------
            # Cumulative pairwise: q only.
            # ----------------------------------------------------
            traj = evaluate_detector_configuration(
                y,
                epochs,
                score_nt,
                labels,
                start_col,
                "cumulative_pairwise",
                q,
            )

            summary = summarize_stable(
                traj,
                target_tpr,
                args.stable_horizon,
            )

            row = {
                "noise_rate": noise_rate,
                "K": K,
                "variant": variant,
                "detector": "cumulative_pairwise",
                "q": q,
                "attainable_TPR_max":
                    attainable_tpr_max(q, noise_rate),
                "target_fraction":
                    args.target_fraction,
                "m": None,
                "ell": None,
                "k": None,
                "rho": None,
                "lambda": None,
                **summary,
            }
            sweep_summary.append(row)

            for r in traj:
                sweep_traj_rows.append({
                    **r,
                    "variant": variant,
                    "detector": "cumulative_pairwise",
                    "q": q,
                    "m": None,
                    "ell": None,
                    "k": None,
                    "rho": None,
                    "lambda": None,
                })

    write_csv(
        args.output_dir
        / "noise20_hyperparameter_sweep_summary.csv",
        sweep_summary,
    )
    write_csv(
        args.output_dir
        / "noise20_hyperparameter_sweep_trajectories.csv",
        sweep_traj_rows,
    )

    # Best configuration within each detector/variant.
    best_rows = []

    detector_variant_keys = sorted(
        set(
            (r["detector"], r["variant"])
            for r in sweep_summary
        )
    )

    for detector, variant in detector_variant_keys:
        sub = [
            r for r in sweep_summary
            if (
                r["detector"] == detector
                and r["variant"] == variant
            )
        ]

        sub.sort(key=stable_sort_key)
        best = dict(sub[0])
        best["rank_within_detector_variant"] = 1
        best_rows.append(best)

    write_csv(
        args.output_dir
        / "noise20_best_by_detector_variant.csv",
        best_rows,
    )

    # Overall best configuration within each detector.
    best_detector_rows = []

    for detector in sorted(
        set(r["detector"] for r in sweep_summary)
    ):
        sub = [
            r for r in sweep_summary
            if r["detector"] == detector
        ]
        sub.sort(key=stable_sort_key)

        best = dict(sub[0])
        best["rank_within_detector"] = 1
        best_detector_rows.append(best)

    write_csv(
        args.output_dir
        / "noise20_best_by_detector.csv",
        best_detector_rows,
    )

    plot_best_configs(
        sweep_traj_rows,
        best_detector_rows,
        args.output_dir,
    )

    config = {
        "artifact":
            "stable_detection_plus_noise20_hyperparameter_sweep",
        "noise_rates_part_A": [0.05, 0.10, 0.20],
        "K_values_part_A": [int(v) for v in args.K_values],
        "part_A_reference_q": args.reference_q,
        "part_A_reference_ewma_lambda":
            args.reference_ewma_lambda,
        "part_A_reference_sliding": {
            "ell": args.reference_sliding_ell,
            "k": args.reference_sliding_k,
        },
        "stable_detection": {
            "target_fraction_of_attainable_TPR":
                args.target_fraction,
            "stable_horizon_epochs":
                args.stable_horizon,
            "TPR_max_formula":
                "min(1,q/noise_rate)",
            "selection_priority":
                "earliest stable detection first; FPR over stable window is secondary",
        },
        "noise20_sweep": {
            "K": 40,
            "q_values": [float(v) for v in args.q_values],
            "ewma_lambdas":
                [float(v) for v in args.ewma_lambdas],
            "sliding_ells":
                [int(v) for v in args.sliding_ells],
            "sliding_rhos":
                [float(v) for v in args.sliding_rhos],
            "minrun_rule":
                "m=ceil(log(T_monitor/delta)/(-log(q)))",
            "delta": args.delta,
        },
        "score_variants": [
            "CKL_rank",
            "LE_rank",
            "weighted_w0.75",
            "prob_or",
            "product",
        ],
        "estimator_settings": {
            "CKL_limit_rule": "last-three mean",
            "LE_formula":
                "signed ell_err + log(abs(m_GIE))",
            "LE_error_limit_rule": "next sample",
            "LE_GIE_limit_rule": "next sample",
        },
        "evaluation_warning":
            "this sweep uses known synthetic noise labels to compare configurations; freeze selected settings before independent validation",
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
        / "stable_detection_frozen_settings.csv"
    )
    print(
        args.output_dir
        / "noise20_hyperparameter_sweep_summary.csv"
    )
    print(
        args.output_dir
        / "noise20_best_by_detector.csv"
    )
    print(
        args.output_dir
        / "noise20_best_stable_configs_TPR.png"
    )
    print(
        args.output_dir
        / "noise20_best_stable_configs_FPR.png"
    )


if __name__ == "__main__":
    main()
