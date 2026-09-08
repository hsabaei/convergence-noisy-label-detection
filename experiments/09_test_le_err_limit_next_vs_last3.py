#!/usr/bin/env python
"""
Focused sensitivity study for the LE error-component limit estimate.

Goal
----
Keep the GIE construction fixed at the user's stabilised choice:

    K = 40
    GIE sample/reference limits = mean of latest 3 observations
    class reference = observed-class mean

Change ONLY the finite-sample limit used by the error-decay component:

    A) last3_mean
    B) next

The theoretical estimator remains unchanged:

    ell_hat = ell_err_hat + log(abs(m_GIE_hat))

This experiment answers one question only:

    Does using the next sample for ell_err improve LE without materially
    harming temporal stability, while GIE keeps its last-three averaging?

No production estimator/detector module is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from convergence_monitoring.framework import (
    standardize_monitoring_score,
)


EPS = 1e-7
K = 40


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
        default=Path(
            "results/le_err_limit_sensitivity_k40"
        ),
    )

    args = p.parse_args()

    for q in args.top_fractions:
        if not 0.0 < q < 1.0:
            p.error("--top-fractions must lie in (0,1).")

    if not 0.0 < args.target_fpr < 1.0:
        p.error("--target-fpr must lie in (0,1).")

    return args


def build_class_mean_reference(loss, labels, num_classes):
    x = np.asarray(loss, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    G = np.full_like(x, np.nan, dtype=np.float64)

    for c in range(int(num_classes)):
        members = np.flatnonzero(labels == c)
        if members.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            ref = np.nanmean(x[members], axis=0)
        G[members] = ref[None, :]

    return G


def last3_limit(traj, t):
    return np.mean(
        traj[:, t - 2:t + 1],
        axis=1,
    )


def next_limit(traj, t):
    return traj[:, t].copy()


def vectorized_gie_window_with_limits(
    phi,
    G,
    phi_limit,
    G_limit,
):
    """Same row-wise GIE/Hill formula used in the prior corrected study."""
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    phi_limit = np.asarray(phi_limit, dtype=np.float64)
    G_limit = np.asarray(G_limit, dtype=np.float64)

    N, _ = phi.shape

    R = np.abs(phi - phi_limit[:, None])
    FR = np.abs(G - G_limit[:, None])

    with np.errstate(invalid="ignore"):
        w0 = np.max(R, axis=1)
        w1 = np.max(FR, axis=1)

    d = np.full(N, np.nan, dtype=np.float64)

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

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
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

    hill_num = np.full(N, np.nan)
    hill_den = np.full(N, np.nan)

    hill_num[ok] = -k_internal[ok] / denom_num[ok]
    hill_den[ok] = -k_internal[ok] / denom_den[ok]

    good = (
        ok
        & np.isfinite(hill_num)
        & np.isfinite(hill_den)
        & (np.abs(hill_den) > EPS)
    )

    d[good] = hill_num[good] / hill_den[good]
    return d


def rolling_le_variant(
    loss,
    labels,
    *,
    err_limit_method,
    num_classes,
):
    """K=40 LE with GIE always using last3 mean.

    At output epoch t:
        boundary = x[t-K-1]
        tail     = x[t-K:t]  (K points, ending at x[t-1])

    Error component:
        err_limit_method = last3_mean OR next

    GIE:
        sample limit     = last3_mean
        reference limit  = last3_mean
    """
    x = np.asarray(loss, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    N, T = x.shape
    G = build_class_mean_reference(
        x,
        labels,
        num_classes,
    )

    ell_err_traj = np.full((N, T), np.nan)
    id_gie_traj = np.full((N, T), np.nan)
    le_traj = np.full((N, T), np.nan)
    err_limit_traj = np.full((N, T), np.nan)
    gie_sample_limit_traj = np.full((N, T), np.nan)
    gie_reference_limit_traj = np.full((N, T), np.nan)

    inv_j = 1.0 / np.arange(1, K + 1, dtype=np.float64)

    first_col = K + 1

    for t in range(first_col, T):
        boundary = t - K - 1
        tail_start = t - K

        if err_limit_method == "last3_mean":
            L_err = last3_limit(x, t)
        elif err_limit_method == "next":
            L_err = next_limit(x, t)
        else:
            raise ValueError(err_limit_method)

        # GIE remains stabilised with last-three averaging.
        L_gie_sample = last3_limit(x, t)
        L_gie_ref = last3_limit(G, t)

        err_limit_traj[:, t] = L_err
        gie_sample_limit_traj[:, t] = L_gie_sample
        gie_reference_limit_traj[:, t] = L_gie_ref

        tail = x[:, tail_start:t]
        G_tail = G[:, tail_start:t]

        # ----------------------------------------
        # ell_err with its own tested L_err.
        # ----------------------------------------
        r0 = np.abs(x[:, boundary] - L_err)
        rj = np.abs(tail - L_err[:, None])

        valid_err = (
            np.isfinite(L_err)
            & np.isfinite(r0)
            & (r0 != 0.0)
            & np.all(
                np.isfinite(rj)
                & (rj != 0.0),
                axis=1,
            )
        )

        ell_err = np.full(N, np.nan)

        if np.any(valid_err):
            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
            ):
                vals = (
                    np.log(
                        rj[valid_err]
                        / r0[valid_err, None]
                    )
                    * inv_j[None, :]
                )

            ell_err[valid_err] = np.mean(
                vals,
                axis=1,
            )

        # ----------------------------------------
        # GIE always with stable last3 limits.
        # ----------------------------------------
        id_gie = vectorized_gie_window_with_limits(
            tail,
            G_tail,
            L_gie_sample,
            L_gie_ref,
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            le = ell_err + np.log(np.abs(id_gie))

        valid = (
            valid_err
            & np.isfinite(id_gie)
            & (id_gie != 0.0)
            & np.isfinite(le)
        )

        ell_err_traj[valid, t] = ell_err[valid]
        id_gie_traj[valid, t] = id_gie[valid]
        le_traj[valid, t] = le[valid]

    return {
        "le_traj": le_traj,
        "ell_err_traj": ell_err_traj,
        "id_gie_traj": id_gie_traj,
        "err_limit_traj": err_limit_traj,
        "gie_sample_limit_traj": gie_sample_limit_traj,
        "gie_reference_limit_traj": gie_reference_limit_traj,
        "first_available_column": first_col,
        "first_available_epoch": first_col + 1,
    }


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

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


def auc_curve(y, score_nt):
    return np.asarray(
        [
            auc_binary(y, score_nt[:, t])
            for t in range(score_nt.shape[1])
        ],
        dtype=np.float64,
    )


def best_auc(auc, epochs):
    finite = np.flatnonzero(np.isfinite(auc))
    t = int(finite[np.nanargmax(auc[finite])])
    return float(auc[t]), int(epochs[t]), t


def temporal_stability(score_nt, start_col):
    x = np.asarray(score_nt[:, start_col:], dtype=np.float64)

    a = x[:, :-1]
    b = x[:, 1:]
    valid = np.isfinite(a) & np.isfinite(b)

    steps = np.abs(b[valid] - a[valid])

    return {
        "median_abs_step": float(np.median(steps)),
        "p95_abs_step": float(np.quantile(steps, 0.95)),
        "n_steps": int(steps.size),
    }


def topq_metrics(y, score, q):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    N = y.size
    finite_idx = np.flatnonzero(np.isfinite(score))
    k = min(
        max(1, int(round(float(q) * N))),
        finite_idx.size,
    )

    selected = np.zeros(N, dtype=bool)
    fs = score[finite_idx]
    local = np.argpartition(fs, -k)[-k:]
    selected[finite_idx[local]] = True

    TP = int(np.sum(selected & y))
    FP = int(np.sum(selected & ~y))
    FN = int(np.sum(~selected & y))
    TN = int(np.sum(~selected & ~y))

    return {
        "q": float(q),
        "TPR": TP / (TP + FN),
        "FPR": FP / (FP + TN),
        "precision": TP / (TP + FP),
        "n_selected": int(np.sum(selected)),
    }


def oracle_max_tpr(y, score, target_fpr):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    finite = np.isfinite(score)
    yy = y[finite]
    ss = score[finite]

    order = np.argsort(-ss, kind="mergesort")
    yy = yy[order]
    ss = ss[order]

    n_pos = int(np.sum(yy))
    n_neg = int(np.sum(~yy))

    tp = np.cumsum(yy.astype(np.int64))
    fp = np.cumsum((~yy).astype(np.int64))

    tpr = tp / n_pos
    fpr = fp / n_neg

    feasible = np.flatnonzero(fpr <= target_fpr)
    best_tpr = np.max(tpr[feasible])
    cand = feasible[np.isclose(tpr[feasible], best_tpr)]
    best = int(cand[np.argmin(fpr[cand])])

    return {
        "oracle_TPR": float(tpr[best]),
        "oracle_FPR": float(fpr[best]),
        "oracle_selected_fraction": float((best + 1) / y.size),
    }


def write_csv(path, rows):
    fields = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def evaluate(
    name,
    result,
    *,
    labels,
    y,
    epochs,
    num_classes,
    top_fractions,
    target_fpr,
):
    le = result["le_traj"]
    ell_err = result["ell_err_traj"]
    id_gie = result["id_gie_traj"]
    start_col = int(result["first_available_column"])

    z_tn = standardize_monitoring_score(
        le.T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=start_col,
    )
    z_nt = z_tn.T

    raw_auc = auc_curve(y, le)
    z_auc = auc_curve(y, z_nt)
    err_auc = auc_curve(y, ell_err)

    log_m = np.full_like(id_gie, np.nan)
    valid_m = np.isfinite(id_gie) & (id_gie != 0.0)
    log_m[valid_m] = np.log(np.abs(id_gie[valid_m]))
    m_auc = auc_curve(y, log_m)

    raw_best, raw_ep, _ = best_auc(raw_auc, epochs)
    z_best, z_ep, _ = best_auc(z_auc, epochs)
    err_best, err_ep, _ = best_auc(err_auc, epochs)
    m_best, m_ep, _ = best_auc(m_auc, epochs)

    pair = exact_pairwise_scores_by_class(
        le,
        labels,
        start_index=start_col,
    )

    pair_score = np.asarray(
        pair["cumulative_score"],
        dtype=np.float64,
    )

    pair_auc = auc_curve(y, pair_score)
    pair_best, pair_ep, pair_t = best_auc(
        pair_auc,
        epochs,
    )

    stability = temporal_stability(
        le,
        start_col,
    )

    # Best tested q+epoch under the common 5% FPR budget.
    best_q_case = None
    topq_rows = []

    for t in range(start_col, le.shape[1]):
        for q in top_fractions:
            m = topq_metrics(
                y,
                pair_score[:, t],
                q,
            )
            row = {
                "variant": name,
                "epoch": int(epochs[t]),
                "pairwise_auc": float(pair_auc[t]),
                **m,
            }
            topq_rows.append(row)

            if m["FPR"] <= target_fpr:
                if (
                    best_q_case is None
                    or m["TPR"] > best_q_case["TPR"]
                    or (
                        np.isclose(m["TPR"], best_q_case["TPR"])
                        and m["FPR"] < best_q_case["FPR"]
                    )
                ):
                    best_q_case = row

    oracle_rows = []
    for t in range(start_col, le.shape[1]):
        o = oracle_max_tpr(
            y,
            pair_score[:, t],
            target_fpr,
        )
        oracle_rows.append({
            "epoch": int(epochs[t]),
            **o,
        })

    best_oracle = max(
        oracle_rows,
        key=lambda r: (r["oracle_TPR"], -r["oracle_FPR"]),
    )

    summary = {
        "variant": name,
        "K": K,
        "err_limit_method":
            "next" if name == "err_next__gie_last3"
            else "last3_mean",
        "gie_limit_method": "last3_mean",
        "reference_method": "mean",
        "raw_le_best_auc": raw_best,
        "raw_le_best_epoch": raw_ep,
        "z_le_best_auc": z_best,
        "z_le_best_epoch": z_ep,
        "ell_err_best_auc": err_best,
        "ell_err_best_epoch": err_ep,
        "log_abs_gie_best_auc": m_best,
        "log_abs_gie_best_epoch": m_ep,
        "pairwise_best_auc": pair_best,
        "pairwise_best_epoch": pair_ep,
        "le_median_abs_epoch_step":
            stability["median_abs_step"],
        "le_p95_abs_epoch_step":
            stability["p95_abs_step"],
        "best_tested_q": float(best_q_case["q"]),
        "best_tested_q_epoch": int(best_q_case["epoch"]),
        "best_tested_q_TPR": float(best_q_case["TPR"]),
        "best_tested_q_FPR": float(best_q_case["FPR"]),
        "oracle_TPR": float(best_oracle["oracle_TPR"]),
        "oracle_FPR": float(best_oracle["oracle_FPR"]),
        "oracle_epoch": int(best_oracle["epoch"]),
        "oracle_selected_fraction":
            float(best_oracle["oracle_selected_fraction"]),
    }

    return summary, topq_rows, pair_auc


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.common_npz, allow_pickle=False)

    loss = np.asarray(d["loss_traj"], dtype=np.float64)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)
    epochs = np.asarray(d["epoch"], dtype=np.int64)

    variants = {}

    for err_method in ("last3_mean", "next"):
        name = (
            "err_last3__gie_last3"
            if err_method == "last3_mean"
            else "err_next__gie_last3"
        )

        print(
            f"Computing {name}: "
            f"K={K}, err_limit={err_method}, "
            "GIE_limit=last3_mean ..."
        )

        variants[name] = rolling_le_variant(
            loss,
            labels,
            err_limit_method=err_method,
            num_classes=args.num_classes,
        )

    summary_rows = []
    all_topq = []
    pair_auc_by_variant = {}

    for name, result in variants.items():
        summary, topq, pair_auc = evaluate(
            name,
            result,
            labels=labels,
            y=y,
            epochs=epochs,
            num_classes=args.num_classes,
            top_fractions=args.top_fractions,
            target_fpr=args.target_fpr,
        )

        summary_rows.append(summary)
        all_topq.extend(topq)
        pair_auc_by_variant[name] = pair_auc

        print(
            f"{name:>24s} | "
            f"raw={summary['raw_le_best_auc']:.6f} "
            f"z={summary['z_le_best_auc']:.6f} "
            f"pair={summary['pairwise_best_auc']:.6f} "
            f"| dLE50={summary['le_median_abs_epoch_step']:.6g} "
            f"dLE95={summary['le_p95_abs_epoch_step']:.6g} "
            f"| best tested q={summary['best_tested_q']:.2f} "
            f"TPR={summary['best_tested_q_TPR']:.4f} "
            f"FPR={summary['best_tested_q_FPR']:.4f}"
        )

    write_csv(
        args.output_dir / "err_limit_sensitivity_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output_dir / "err_limit_sensitivity_topq.csv",
        all_topq,
    )

    # Save both score trajectories for direct inspection if desired.
    np.savez_compressed(
        args.output_dir / "err_limit_sensitivity_score_trajectories.npz",
        epoch=epochs,
        observed_label=labels,
        is_anomaly=y,
        le_err_last3_gie_last3=variants[
            "err_last3__gie_last3"
        ]["le_traj"].astype(np.float32),
        le_err_next_gie_last3=variants[
            "err_next__gie_last3"
        ]["le_traj"].astype(np.float32),
        ell_err_last3=variants[
            "err_last3__gie_last3"
        ]["ell_err_traj"].astype(np.float32),
        ell_err_next=variants[
            "err_next__gie_last3"
        ]["ell_err_traj"].astype(np.float32),
        id_gie_last3=variants[
            "err_last3__gie_last3"
        ]["id_gie_traj"].astype(np.float32),
    )

    # Plot cumulative pairwise AUC.
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, auc in pair_auc_by_variant.items():
        ax.plot(
            epochs,
            auc,
            label=name,
        )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative pairwise ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(
        "LE-GIE: error-limit sensitivity with GIE fixed at last3"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "fig_err_limit_pairwise_auc.png",
        dpi=180,
    )
    plt.close(fig)

    # Stability comparison.
    labels_plot = [
        r["variant"]
        for r in summary_rows
    ]
    med = [
        r["le_median_abs_epoch_step"]
        for r in summary_rows
    ]
    p95 = [
        r["le_p95_abs_epoch_step"]
        for r in summary_rows
    ]

    xloc = np.arange(len(summary_rows), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xloc - width/2, med, width, label="Median |ΔLE|")
    ax.bar(xloc + width/2, p95, width, label="95th percentile |ΔLE|")
    ax.set_xticks(xloc)
    ax.set_xticklabels(labels_plot)
    ax.set_ylabel("Absolute epoch-to-epoch LE change")
    ax.set_title("LE stability: next vs last3 only in error component")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "fig_err_limit_stability.png",
        dpi=180,
    )
    plt.close(fig)

    config = {
        "artifact": "LE_error_limit_sensitivity_K40",
        "K": K,
        "theoretical_estimator":
            "ell_hat = ell_err_hat + log(abs(m_GIE_hat))",
        "reference_method": "mean",
        "gie_limit_method": "last3_mean",
        "tested_error_limit_methods": [
            "last3_mean",
            "next",
        ],
        "top_fractions": [
            float(q)
            for q in args.top_fractions
        ],
        "target_fpr": float(args.target_fpr),
        "production_modules_modified": False,
        "purpose":
            "test whether next-sample L for ell_err improves LE while GIE retains last3 stability",
        "selection_warning":
            "Exploratory labeled-run comparison; freeze chosen variant before independent validation.",
    }

    with (
        args.output_dir / "err_limit_sensitivity_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print(
        "Done. GIE remained fixed at last3_mean for BOTH variants."
    )
    print(
        f"Outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
