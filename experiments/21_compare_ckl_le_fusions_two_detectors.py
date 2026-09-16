#!/usr/bin/env python
"""
Compare CKL, LE-GIE, and CKL+LE fusion using ONLY the two strongest
temporal detectors:

    1) rank-EWMA
    2) cumulative pairwise

The decision is based on the FULL temporal behavior, not only max TPR.

Noise levels:
    5%, 10%, 20%

K values:
    configurable; default {20, 40}

Score constructions:
    CKL_rank
    LE_rank
    weighted_w0.75 = 0.75*r_CKL + 0.25*r_LE
    prob_or        = r_CKL + r_LE - r_CKL*r_LE
    product        = r_CKL*r_LE

For each detector/score/noise/K combination, report:
    - earliest stable detection epoch
    - stable-window TPR/FPR
    - early mean TPR/FPR
    - sustained mean TPR/FPR
    - late mean TPR/FPR
    - peak TPR and epoch
    - final TPR/FPR
    - peak-to-final decay
    - temporal AUC of TPR and FPR

Stable detection is defined relative to the maximum recall attainable
under the fixed top-q budget:

    TPR_max = min(1, q / noise_rate)
    stable target = target_fraction * TPR_max

Default target_fraction = 0.90 and stable horizon = 5 epochs.

Ground-truth noise labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from convergence_monitoring.detectors import exact_pairwise_scores_by_class


# Reuse the already validated estimator implementations.
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
            "results/ckl_le_two_detector_comparison"
        ),
    )

    p.add_argument(
        "--K-values",
        type=int,
        nargs="+",
        default=(20, 40),
    )

    p.add_argument("--num-classes", type=int, default=10)

    # Same q is used for all score constructions within each noise level.
    # 20% defaults to q=.20 so recall is not artificially capped at .50.
    p.add_argument("--q-05", type=float, default=0.10)
    p.add_argument("--q-10", type=float, default=0.10)
    p.add_argument("--q-20", type=float, default=0.20)

    p.add_argument("--rank-ewma-lambda", type=float, default=0.05)

    # Stable-detection metrics.
    p.add_argument("--target-fraction", type=float, default=0.90)
    p.add_argument("--stable-horizon", type=int, default=5)

    # Temporal intervals.
    p.add_argument("--early-start", type=int, default=50)
    p.add_argument("--early-end", type=int, default=80)
    p.add_argument("--sustained-start", type=int, default=60)
    p.add_argument("--sustained-end", type=int, default=120)
    p.add_argument("--late-start", type=int, default=180)
    p.add_argument("--late-end", type=int, default=200)

    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def fuse_weighted(ckl_pct, le_pct, weight=0.75):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)

    out[valid] = (
        float(weight) * ckl_pct[valid]
        + (1.0 - float(weight)) * le_pct[valid]
    ).astype(np.float32)

    return out


def fuse_prob_or(ckl_pct, le_pct):
    out = np.full_like(ckl_pct, np.nan, dtype=np.float32)
    valid = np.isfinite(ckl_pct) & np.isfinite(le_pct)

    c = np.clip(ckl_pct[valid], 0.0, 1.0)
    l = np.clip(le_pct[valid], 0.0, 1.0)

    out[valid] = (
        c + l - c * l
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
        pred[:, t] = topq_mask(
            stat_nt[:, t],
            q,
        )

    return pred


def trajectory_rows(
    y,
    pred_nt,
    epochs,
    start_col,
    *,
    noise_rate,
    K,
    q,
    detector,
    variant,
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
            "q": float(q),
            "detector": detector,
            "variant": variant,
            "epoch": int(epochs[t]),
            "TPR": tp / max(n_pos, 1),
            "FPR": fp / max(n_neg, 1),
            "TP": tp,
            "FP": fp,
            "n_selected": int(np.sum(pred)),
        })

    return rows


def interval_mean(epoch, values, start, end):
    mask = (
        (epoch >= int(start))
        & (epoch <= int(end))
    )

    if not np.any(mask):
        return np.nan

    return float(np.mean(values[mask]))


def normalized_time_auc(epoch, values):
    if len(epoch) <= 1:
        return float(values[0])

    span = float(epoch[-1] - epoch[0])
    if span <= 0:
        return float(values[-1])

    return float(
        np.trapz(values, epoch) / span
    )


def summarize_trajectory(
    rows,
    *,
    noise_rate,
    q,
    target_fraction,
    stable_horizon,
    early_start,
    early_end,
    sustained_start,
    sustained_end,
    late_start,
    late_end,
):
    rows = sorted(rows, key=lambda r: r["epoch"])

    epoch = np.asarray(
        [r["epoch"] for r in rows],
        dtype=np.int64,
    )
    tpr = np.asarray(
        [r["TPR"] for r in rows],
        dtype=np.float64,
    )
    fpr = np.asarray(
        [r["FPR"] for r in rows],
        dtype=np.float64,
    )

    tpr_max_attainable = min(
        1.0,
        float(q) / float(noise_rate),
    )
    target_tpr = (
        float(target_fraction)
        * tpr_max_attainable
    )

    h = int(stable_horizon)
    first_stable = -1

    for i in range(0, len(rows) - h + 1):
        if np.all(
            tpr[i : i + h] >= target_tpr
        ):
            first_stable = i
            break

    peak_idx = int(np.argmax(tpr))

    summary = {
        "attainable_TPR_max": float(tpr_max_attainable),
        "stable_target_TPR": float(target_tpr),
        "stable_horizon": h,

        "first_stable_epoch": (
            int(epoch[first_stable])
            if first_stable >= 0
            else -1
        ),

        "peak_TPR": float(tpr[peak_idx]),
        "peak_epoch": int(epoch[peak_idx]),
        "FPR_at_peak_TPR": float(fpr[peak_idx]),

        "final_TPR": float(tpr[-1]),
        "final_FPR": float(fpr[-1]),

        "peak_to_final_TPR_decay": float(
            tpr[peak_idx] - tpr[-1]
        ),

        "early_mean_TPR": interval_mean(
            epoch, tpr, early_start, early_end
        ),
        "early_mean_FPR": interval_mean(
            epoch, fpr, early_start, early_end
        ),

        "sustained_mean_TPR": interval_mean(
            epoch, tpr, sustained_start, sustained_end
        ),
        "sustained_mean_FPR": interval_mean(
            epoch, fpr, sustained_start, sustained_end
        ),

        "late_mean_TPR": interval_mean(
            epoch, tpr, late_start, late_end
        ),
        "late_mean_FPR": interval_mean(
            epoch, fpr, late_start, late_end
        ),

        "TPR_time_AUC": normalized_time_auc(
            epoch,
            tpr,
        ),
        "FPR_time_AUC": normalized_time_auc(
            epoch,
            fpr,
        ),
    }

    if first_stable >= 0:
        sl = slice(
            first_stable,
            first_stable + h,
        )

        summary.update({
            "stable_window_mean_TPR": float(
                np.mean(tpr[sl])
            ),
            "stable_window_min_TPR": float(
                np.min(tpr[sl])
            ),
            "stable_window_mean_FPR": float(
                np.mean(fpr[sl])
            ),
            "stable_window_max_FPR": float(
                np.max(fpr[sl])
            ),
        })
    else:
        summary.update({
            "stable_window_mean_TPR": np.nan,
            "stable_window_min_TPR": np.nan,
            "stable_window_mean_FPR": np.nan,
            "stable_window_max_FPR": np.nan,
        })

    return summary


def pareto_dominates(a, b):
    """Trajectory-oriented dominance.

    A dominates B if A is at least as good on all of:
      - stable detection time (earlier)
      - early TPR (higher)
      - sustained TPR (higher)
      - late TPR (higher)
      - TPR time-AUC (higher)
      - sustained FPR (lower)
      - decay (lower)

    and strictly better on at least one.
    """
    a_epoch = a["first_stable_epoch"]
    b_epoch = b["first_stable_epoch"]

    # A failed stable detection is treated as infinitely late.
    a_epoch_cmp = (
        a_epoch
        if a_epoch >= 0
        else 10**9
    )
    b_epoch_cmp = (
        b_epoch
        if b_epoch >= 0
        else 10**9
    )

    comparisons = [
        a_epoch_cmp <= b_epoch_cmp,
        a["early_mean_TPR"] >= b["early_mean_TPR"],
        a["sustained_mean_TPR"] >= b["sustained_mean_TPR"],
        a["late_mean_TPR"] >= b["late_mean_TPR"],
        a["TPR_time_AUC"] >= b["TPR_time_AUC"],
        a["sustained_mean_FPR"] <= b["sustained_mean_FPR"],
        a["peak_to_final_TPR_decay"]
        <= b["peak_to_final_TPR_decay"],
    ]

    strict = [
        a_epoch_cmp < b_epoch_cmp,
        a["early_mean_TPR"] > b["early_mean_TPR"],
        a["sustained_mean_TPR"] > b["sustained_mean_TPR"],
        a["late_mean_TPR"] > b["late_mean_TPR"],
        a["TPR_time_AUC"] > b["TPR_time_AUC"],
        a["sustained_mean_FPR"] < b["sustained_mean_FPR"],
        a["peak_to_final_TPR_decay"]
        < b["peak_to_final_TPR_decay"],
    ]

    return all(comparisons) and any(strict)


def mark_pareto(summary_rows):
    out = []

    block_keys = sorted(
        set(
            (
                r["noise_rate"],
                r["K"],
                r["detector"],
            )
            for r in summary_rows
        )
    )

    for noise_rate, K, detector in block_keys:
        block = [
            dict(r)
            for r in summary_rows
            if (
                r["noise_rate"] == noise_rate
                and r["K"] == K
                and r["detector"] == detector
            )
        ]

        for i, row in enumerate(block):
            dominated = False

            for j, other in enumerate(block):
                if i == j:
                    continue

                if pareto_dominates(other, row):
                    dominated = True
                    break

            row["pareto_nondominated"] = not dominated
            out.append(row)

    return out


def plot_detector(
    trajectory_rows_all,
    *,
    noise_rate,
    K,
    detector,
    metric,
    output_path,
):
    variants = [
        "CKL_rank",
        "LE_rank",
        "weighted_w0.75",
        "prob_or",
        "product",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in variants:
        sub = [
            r for r in trajectory_rows_all
            if (
                r["noise_rate"] == noise_rate
                and r["K"] == K
                and r["detector"] == detector
                and r["variant"] == variant
            )
        ]

        if not sub:
            continue

        sub.sort(key=lambda r: r["epoch"])

        ax.plot(
            [r["epoch"] for r in sub],
            [r[metric] for r in sub],
            label=variant,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"{metric} over epochs | "
        f"noise={int(noise_rate*100)}% | "
        f"K={K} | {detector}"
    )
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = [
        (
            0.05,
            args.q_05,
            args.npz_05,
        ),
        (
            0.10,
            args.q_10,
            args.npz_10,
        ),
        (
            0.20,
            args.q_20,
            args.npz_20,
        ),
    ]

    all_trajectory_rows = []
    summary_rows = []

    for noise_rate, q, npz_path in datasets:
        print()
        print("=" * 76)
        print(
            f"noise={noise_rate:.2f} | q={q:.2f}"
        )
        print("=" * 76)

        d = np.load(
            npz_path,
            allow_pickle=False,
        )

        loss = np.asarray(
            d["loss_traj"],
            dtype=np.float32,
        )
        labels = np.asarray(
            d["observed_label"],
            dtype=np.int64,
        )
        y = np.asarray(
            d["is_anomaly"],
            dtype=bool,
        )
        epochs = np.asarray(
            d["epoch"],
            dtype=np.int64,
        )

        observed_noise = float(np.mean(y))
        print(
            f"observed_noise_fraction="
            f"{observed_noise:.4f}"
        )

        class_mean = exp11.build_class_mean(
            loss,
            labels,
            args.num_classes,
        )

        for K in args.K_values:
            K = int(K)
            start_col = K + 1

            print()
            print(f"K={K}")

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
                "prob_or": fuse_prob_or(
                    ckl_pct,
                    le_pct,
                ),
                "product": fuse_product(
                    ckl_pct,
                    le_pct,
                ),
            }

            for variant_name, score_nt in variants.items():
                print(
                    f"  {variant_name}"
                )

                # ------------------------------------------------
                # 1) Rank-EWMA
                # ------------------------------------------------
                ewma_nt = rank_ewma(
                    score_nt,
                    args.rank_ewma_lambda,
                    start_col,
                )

                ewma_pred = topq_from_stat_nt(
                    ewma_nt,
                    q,
                    start_col,
                )

                rows = trajectory_rows(
                    y,
                    ewma_pred,
                    epochs,
                    start_col,
                    noise_rate=noise_rate,
                    K=K,
                    q=q,
                    detector="rank_ewma",
                    variant=variant_name,
                )

                all_trajectory_rows.extend(rows)

                summary = summarize_trajectory(
                    rows,
                    noise_rate=noise_rate,
                    q=q,
                    target_fraction=args.target_fraction,
                    stable_horizon=args.stable_horizon,
                    early_start=args.early_start,
                    early_end=args.early_end,
                    sustained_start=args.sustained_start,
                    sustained_end=args.sustained_end,
                    late_start=args.late_start,
                    late_end=args.late_end,
                )

                summary_rows.append({
                    "noise_rate": noise_rate,
                    "K": K,
                    "q": q,
                    "detector": "rank_ewma",
                    "variant": variant_name,
                    "lambda":
                        args.rank_ewma_lambda,
                    **summary,
                })

                del ewma_nt, ewma_pred
                gc.collect()

                # ------------------------------------------------
                # 2) Cumulative pairwise
                # ------------------------------------------------
                pair = exact_pairwise_scores_by_class(
                    score_nt,
                    labels,
                    start_index=start_col,
                )

                pair_nt = np.asarray(
                    pair["cumulative_score"],
                    dtype=np.float32,
                )

                pair_pred = topq_from_stat_nt(
                    pair_nt,
                    q,
                    start_col,
                )

                rows = trajectory_rows(
                    y,
                    pair_pred,
                    epochs,
                    start_col,
                    noise_rate=noise_rate,
                    K=K,
                    q=q,
                    detector="cumulative_pairwise",
                    variant=variant_name,
                )

                all_trajectory_rows.extend(rows)

                summary = summarize_trajectory(
                    rows,
                    noise_rate=noise_rate,
                    q=q,
                    target_fraction=args.target_fraction,
                    stable_horizon=args.stable_horizon,
                    early_start=args.early_start,
                    early_end=args.early_end,
                    sustained_start=args.sustained_start,
                    sustained_end=args.sustained_end,
                    late_start=args.late_start,
                    late_end=args.late_end,
                )

                summary_rows.append({
                    "noise_rate": noise_rate,
                    "K": K,
                    "q": q,
                    "detector":
                        "cumulative_pairwise",
                    "variant": variant_name,
                    "lambda": None,
                    **summary,
                })

                del pair, pair_nt, pair_pred
                gc.collect()

            # Save incrementally.
            write_csv(
                args.output_dir
                / "trajectory_metrics_over_epochs.csv",
                all_trajectory_rows,
            )

            write_csv(
                args.output_dir
                / "trajectory_summary.csv",
                summary_rows,
            )

            # Plots for both detectors.
            plot_dir = (
                args.output_dir
                / f"noise{int(noise_rate*100):02d}"
                / f"K{K}"
            )
            plot_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for detector in (
                "rank_ewma",
                "cumulative_pairwise",
            ):
                for metric in (
                    "TPR",
                    "FPR",
                ):
                    plot_detector(
                        all_trajectory_rows,
                        noise_rate=noise_rate,
                        K=K,
                        detector=detector,
                        metric=metric,
                        output_path=(
                            plot_dir
                            / f"{detector}_{metric}_over_epochs.png"
                        ),
                    )

            del ckl, le, ckl_pct, le_pct, variants
            gc.collect()

        del d, loss, labels, y, epochs, class_mean
        gc.collect()

    # Mark Pareto-nondominated score constructions separately for each
    # noise/K/detector block.
    pareto_rows = mark_pareto(
        summary_rows
    )

    write_csv(
        args.output_dir
        / "trajectory_summary_with_pareto.csv",
        pareto_rows,
    )

    # Detector-level compact comparison:
    # only the nondominated score constructions.
    nondominated_rows = [
        r for r in pareto_rows
        if r["pareto_nondominated"]
    ]

    write_csv(
        args.output_dir
        / "pareto_nondominated_methods.csv",
        nondominated_rows,
    )

    config = {
        "artifact":
            "CKL_LE_fusion_two_detector_trajectory_comparison",
        "noise_rates": [0.05, 0.10, 0.20],
        "K_values": [int(v) for v in args.K_values],
        "q_by_noise": {
            "0.05": args.q_05,
            "0.10": args.q_10,
            "0.20": args.q_20,
        },
        "detectors": [
            "rank_ewma",
            "cumulative_pairwise",
        ],
        "rank_ewma_lambda":
            args.rank_ewma_lambda,
        "score_variants": [
            "CKL_rank",
            "LE_rank",
            "weighted_w0.75",
            "prob_or",
            "product",
        ],
        "fusion_formulas": {
            "weighted_w0.75":
                "0.75*r_CKL + 0.25*r_LE",
            "prob_or":
                "r_CKL + r_LE - r_CKL*r_LE",
            "product":
                "r_CKL*r_LE",
        },
        "stable_detection": {
            "target_fraction":
                args.target_fraction,
            "stable_horizon":
                args.stable_horizon,
            "target_formula":
                "target_fraction * min(1,q/noise_rate)",
        },
        "temporal_intervals": {
            "early": [
                args.early_start,
                args.early_end,
            ],
            "sustained": [
                args.sustained_start,
                args.sustained_end,
            ],
            "late": [
                args.late_start,
                args.late_end,
            ],
        },
        "decision_note":
            "do not choose only by maximum TPR; use stable detection time, early/sustained/late TPR, FPR, temporal AUC, and decay",
        "evaluation_only_note":
            "true synthetic noise labels are used only for evaluation",
    }

    with (
        args.output_dir / "config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print("DONE")
    print("=" * 76)
    print(
        args.output_dir
        / "trajectory_summary.csv"
    )
    print(
        args.output_dir
        / "trajectory_summary_with_pareto.csv"
    )
    print(
        args.output_dir
        / "pareto_nondominated_methods.csv"
    )


if __name__ == "__main__":
    main()
