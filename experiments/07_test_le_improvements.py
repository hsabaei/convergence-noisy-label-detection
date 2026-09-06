#!/usr/bin/env python
"""
Staged finite-sample sensitivity study for the proposed LE-GIE score.

The theoretical estimator is NEVER changed:

    ell_hat = ell_err_hat + log(abs(m_GIE_hat)).

Only finite-sample estimation choices are varied.

Stage A — K sensitivity
    K in {10,15,20,30,40}
    class reference = ordinary observed-class mean
    GIE limit = original last-3 mean
    error-component limit = guarded Aitken

Stage B — GIE class-reference sensitivity
    K = 20
    ordinary mean
    leave-one-out mean
    median
    5% symmetric trimmed mean
    10% symmetric trimmed mean

Stage C — GIE limit sensitivity
    K = 20
    class reference = ordinary observed-class mean
    last-3 mean
    next-sample
    guarded Aitken

For every variant we report:
    - best raw LE AUC
    - best class-z LE AUC
    - best exact cumulative-pairwise LE AUC
    - top-q TPR/FPR at the pairwise best-AUC epoch
    - evaluation-only maximum TPR under FPR <= target at that same epoch
    - component AUCs for ell_err and log|m_GIE|

This is an exploratory sensitivity experiment.  Do not select a final
configuration from this run and claim generalization without an independent
training seed/run.
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
from convergence_monitoring.estimators import (
    rolling_class_reference_gie_batch,
)
from convergence_monitoring.framework import (
    standardize_monitoring_score,
)
from convergence_monitoring.proposed_le import (
    compose_le_from_error_and_m,
    rolling_proposed_le_batch,
)


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
        "--k-values",
        type=int,
        nargs="+",
        default=(10, 15, 20, 30, 40),
    )

    p.add_argument("--reference-k", type=int, default=20)
    p.add_argument("--limit-k", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=10)

    p.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20),
    )
    p.add_argument("--target-fpr", type=float, default=0.05)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/le_improvement_sensitivity"),
    )

    args = p.parse_args()

    if any(k < 6 for k in args.k_values):
        p.error("All --k-values must be at least 6.")
    if args.reference_k < 6 or args.limit_k < 6:
        p.error("--reference-k and --limit-k must be at least 6.")
    if not 0.0 < args.target_fpr < 1.0:
        p.error("--target-fpr must lie in (0,1).")
    for q in args.top_fractions:
        if not 0.0 < q < 1.0:
            p.error("--top-fractions must lie in (0,1).")

    return args


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
    score_nt = np.asarray(score_nt, dtype=np.float64)
    out = np.full(score_nt.shape[1], np.nan, dtype=np.float64)
    for t in range(score_nt.shape[1]):
        out[t] = auc_binary(y, score_nt[:, t])
    return out


def best_auc(auc, epochs):
    finite = np.flatnonzero(np.isfinite(auc))
    if finite.size == 0:
        return np.nan, -1, -1
    t = int(finite[np.nanargmax(auc[finite])])
    return float(auc[t]), int(epochs[t]), t


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
    if k:
        fs = score[finite_idx]
        local = np.argpartition(fs, -k)[-k:]
        selected[finite_idx[local]] = True

    TP = int(np.sum(selected & y))
    FP = int(np.sum(selected & ~y))
    FN = int(np.sum(~selected & y))
    TN = int(np.sum(~selected & ~y))

    TPR = TP / (TP + FN) if TP + FN else np.nan
    FPR = FP / (FP + TN) if FP + TN else np.nan
    precision = TP / (TP + FP) if TP + FP else np.nan

    return {
        "q": float(q),
        "n_selected": int(k),
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "TPR": float(TPR),
        "FPR": float(FPR),
        "precision": float(precision),
    }


def oracle_max_tpr_under_fpr(y, score, target_fpr):
    """Evaluation-only ROC operating point at one epoch."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    finite = np.isfinite(score)
    yy = y[finite]
    ss = score[finite]

    n_pos = int(np.sum(yy))
    n_neg = int(np.sum(~yy))

    if n_pos == 0 or n_neg == 0:
        return {
            "oracle_TPR": np.nan,
            "oracle_FPR": np.nan,
            "oracle_threshold": np.nan,
            "oracle_selected_fraction": np.nan,
        }

    order = np.argsort(-ss, kind="mergesort")
    yy = yy[order]
    ss = ss[order]

    tp = np.cumsum(yy.astype(np.int64))
    fp = np.cumsum((~yy).astype(np.int64))

    tpr = tp / n_pos
    fpr = fp / n_neg

    feasible = np.flatnonzero(fpr <= float(target_fpr))
    if feasible.size == 0:
        return {
            "oracle_TPR": 0.0,
            "oracle_FPR": 0.0,
            "oracle_threshold": np.inf,
            "oracle_selected_fraction": 0.0,
        }

    best_tpr = np.max(tpr[feasible])
    candidates = feasible[np.isclose(tpr[feasible], best_tpr)]

    # Among equal TPR, take the smallest FPR.
    best = int(
        candidates[
            np.argmin(fpr[candidates])
        ]
    )

    return {
        "oracle_TPR": float(tpr[best]),
        "oracle_FPR": float(fpr[best]),
        "oracle_threshold": float(ss[best]),
        "oracle_selected_fraction": float((best + 1) / y.size),
    }


def write_csv(path, rows):
    if not rows:
        return

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


def evaluate_variant(
    *,
    stage,
    variant,
    K,
    reference_method,
    trim_fraction,
    gie_limit_method,
    ell_err_nt,
    id_gie_nt,
    labels,
    y,
    epochs,
    top_fractions,
    target_fpr,
    num_classes,
):
    le_nt = compose_le_from_error_and_m(
        ell_err_nt,
        id_gie_nt,
    )

    start_candidates = np.flatnonzero(
        np.any(np.isfinite(le_nt), axis=0)
    )
    if start_candidates.size == 0:
        raise RuntimeError(f"No finite LE for variant {variant}.")

    start_col = int(start_candidates[0])

    z_tn = standardize_monitoring_score(
        le_nt.T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=start_col,
    )
    z_nt = z_tn.T

    log_m_nt = np.full_like(id_gie_nt, np.nan, dtype=np.float64)
    valid_m = (
        np.isfinite(id_gie_nt)
        & (id_gie_nt != 0.0)
    )
    log_m_nt[valid_m] = np.log(
        np.abs(id_gie_nt[valid_m])
    )

    auc_raw = auc_curve(y, le_nt)
    auc_z = auc_curve(y, z_nt)
    auc_err = auc_curve(y, ell_err_nt)
    auc_log_m = auc_curve(y, log_m_nt)

    raw_best, raw_ep, _ = best_auc(auc_raw, epochs)
    z_best, z_ep, _ = best_auc(auc_z, epochs)
    err_best, err_ep, _ = best_auc(auc_err, epochs)
    m_best, m_ep, _ = best_auc(auc_log_m, epochs)

    pair = exact_pairwise_scores_by_class(
        le_nt,
        labels,
        start_index=start_col,
    )
    pair_cum_nt = np.asarray(
        pair["cumulative_score"],
        dtype=np.float64,
    )
    auc_pair = auc_curve(y, pair_cum_nt)
    pair_best, pair_ep, pair_t = best_auc(
        auc_pair,
        epochs,
    )

    score_best = pair_cum_nt[:, pair_t]
    oracle = oracle_max_tpr_under_fpr(
        y,
        score_best,
        target_fpr,
    )

    finite_frac = float(
        np.mean(np.isfinite(le_nt[:, pair_t]))
    )

    summary = {
        "stage": stage,
        "variant": variant,
        "K": int(K),
        "reference_method": reference_method,
        "trim_fraction": float(trim_fraction),
        "gie_limit_method": gie_limit_method,
        "theoretical_formula": "ell_err + log(abs(m_GIE))",
        "first_available_epoch": int(epochs[start_col]),
        "raw_le_best_auc": raw_best,
        "raw_le_best_epoch": raw_ep,
        "z_le_best_auc": z_best,
        "z_le_best_epoch": z_ep,
        "ell_err_best_auc": err_best,
        "ell_err_best_epoch": err_ep,
        "log_abs_gie_best_auc": m_best,
        "log_abs_gie_best_epoch": m_ep,
        "pairwise_cumulative_best_auc": pair_best,
        "pairwise_cumulative_best_epoch": pair_ep,
        "finite_fraction_at_pairwise_best": finite_frac,
        **oracle,
    }

    topq_rows = []
    for q in top_fractions:
        topq_rows.append({
            "stage": stage,
            "variant": variant,
            "K": int(K),
            "reference_method": reference_method,
            "trim_fraction": float(trim_fraction),
            "gie_limit_method": gie_limit_method,
            "epoch": pair_ep,
            "pairwise_auc": pair_best,
            **topq_metrics(y, score_best, q),
        })

    epoch_rows = []
    for t, ep in enumerate(epochs):
        if (
            np.isfinite(auc_raw[t])
            or np.isfinite(auc_z[t])
            or np.isfinite(auc_pair[t])
        ):
            epoch_rows.append({
                "stage": stage,
                "variant": variant,
                "K": int(K),
                "reference_method": reference_method,
                "trim_fraction": float(trim_fraction),
                "gie_limit_method": gie_limit_method,
                "epoch": int(ep),
                "raw_le_auc": float(auc_raw[t]) if np.isfinite(auc_raw[t]) else np.nan,
                "z_le_auc": float(auc_z[t]) if np.isfinite(auc_z[t]) else np.nan,
                "pairwise_cumulative_auc": (
                    float(auc_pair[t])
                    if np.isfinite(auc_pair[t])
                    else np.nan
                ),
            })

    return summary, topq_rows, epoch_rows


def plot_stage(epoch_rows, stage, path):
    rows = [r for r in epoch_rows if r["stage"] == stage]
    variants = sorted(set(r["variant"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in variants:
        rr = [r for r in rows if r["variant"] == variant]
        rr.sort(key=lambda r: r["epoch"])
        ax.plot(
            [r["epoch"] for r in rr],
            [r["pairwise_cumulative_auc"] for r in rr],
            label=variant,
        )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative pairwise ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"LE-GIE sensitivity — {stage}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.common_npz, allow_pickle=False)

    required = {
        "sample_index",
        "epoch",
        "loss_traj",
        "observed_label",
        "true_label",
        "is_anomaly",
    }
    missing = required - set(d.files)
    if missing:
        raise KeyError(f"Missing arrays in common NPZ: {sorted(missing)}")

    loss = np.asarray(d["loss_traj"], dtype=np.float64)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)
    epochs = np.asarray(d["epoch"], dtype=np.int64)

    summary_rows = []
    topq_rows = []
    epoch_rows = []

    ell_cache = {}
    gie_cache = {}
    eval_cache = {}

    def get_ell(K):
        K = int(K)
        if K not in ell_cache:
            print(f"Computing ell_err for K={K} ...", flush=True)
            out = rolling_proposed_le_batch(
                loss,
                K=K,
                limit_method="aitken_guarded",
            )
            ell_cache[K] = np.asarray(
                out["ell_err_traj"],
                dtype=np.float64,
            )
        return ell_cache[K]

    def get_gie(K, reference_method, trim_fraction, limit_method):
        key = (
            int(K),
            str(reference_method),
            float(trim_fraction),
            str(limit_method),
        )
        if key not in gie_cache:
            print(
                "Computing GIE "
                f"K={K}, ref={reference_method}, trim={trim_fraction}, "
                f"limit={limit_method} ...",
                flush=True,
            )
            out = rolling_class_reference_gie_batch(
                loss,
                labels,
                K=int(K),
                num_classes=args.num_classes,
                reference_method=reference_method,
                trim_fraction=float(trim_fraction),
                limit_method=limit_method,
            )
            gie_cache[key] = np.asarray(
                out["id_gie_traj"],
                dtype=np.float64,
            )
        return gie_cache[key]

    def run_variant(stage, variant, K, ref, trim, limit):
        eval_key = (
            int(K), str(ref), float(trim), str(limit)
        )

        if eval_key not in eval_cache:
            summary, tq, er = evaluate_variant(
                stage=stage,
                variant=variant,
                K=K,
                reference_method=ref,
                trim_fraction=trim,
                gie_limit_method=limit,
                ell_err_nt=get_ell(K),
                id_gie_nt=get_gie(K, ref, trim, limit),
                labels=labels,
                y=y,
                epochs=epochs,
                top_fractions=args.top_fractions,
                target_fpr=args.target_fpr,
                num_classes=args.num_classes,
            )
            eval_cache[eval_key] = (summary, tq, er)
        else:
            base_summary, base_tq, base_er = eval_cache[eval_key]
            summary = dict(base_summary)
            summary.update({
                "stage": stage,
                "variant": variant,
            })

            tq = []
            for r in base_tq:
                q = dict(r)
                q.update({"stage": stage, "variant": variant})
                tq.append(q)

            er = []
            for r in base_er:
                q = dict(r)
                q.update({"stage": stage, "variant": variant})
                er.append(q)

        summary_rows.append(summary)
        topq_rows.extend(tq)
        epoch_rows.extend(er)

        print(
            f"{stage:>12s} | {variant:<26s} | "
            f"raw={summary['raw_le_best_auc']:.6f} "
            f"z={summary['z_le_best_auc']:.6f} "
            f"pair={summary['pairwise_cumulative_best_auc']:.6f} "
            f"@ e{summary['pairwise_cumulative_best_epoch']} "
            f"oracleTPR@FPR<={args.target_fpr:g}="
            f"{summary['oracle_TPR']:.4f}",
            flush=True,
        )

    # ----------------------------------------------------------
    # Stage A: K sensitivity.
    # ----------------------------------------------------------
    for K in args.k_values:
        run_variant(
            "K_sensitivity",
            f"K={int(K)}",
            int(K),
            "mean",
            0.10,
            "last3_mean",
        )

    # ----------------------------------------------------------
    # Stage B: class-reference sensitivity at K=20.
    # ----------------------------------------------------------
    K = int(args.reference_k)

    run_variant(
        "reference_sensitivity",
        "mean",
        K, "mean", 0.10, "last3_mean",
    )
    run_variant(
        "reference_sensitivity",
        "leave_one_out_mean",
        K, "leave_one_out_mean", 0.10, "last3_mean",
    )
    run_variant(
        "reference_sensitivity",
        "median",
        K, "median", 0.10, "last3_mean",
    )
    run_variant(
        "reference_sensitivity",
        "trimmed_mean_05",
        K, "trimmed_mean", 0.05, "last3_mean",
    )
    run_variant(
        "reference_sensitivity",
        "trimmed_mean_10",
        K, "trimmed_mean", 0.10, "last3_mean",
    )

    # ----------------------------------------------------------
    # Stage C: GIE limit sensitivity at K=20.
    # ----------------------------------------------------------
    K = int(args.limit_k)

    for limit in (
        "last3_mean",
        "next",
        "aitken_guarded",
    ):
        run_variant(
            "gie_limit_sensitivity",
            limit,
            K, "mean", 0.10, limit,
        )

    write_csv(
        args.output_dir / "le_improvement_summary.csv",
        summary_rows,
    )
    write_csv(
        args.output_dir / "le_improvement_topq.csv",
        topq_rows,
    )
    write_csv(
        args.output_dir / "le_improvement_auc_by_epoch.csv",
        epoch_rows,
    )

    for stage, filename in (
        ("K_sensitivity", "fig_k_sensitivity_pairwise_auc.png"),
        ("reference_sensitivity", "fig_reference_sensitivity_pairwise_auc.png"),
        ("gie_limit_sensitivity", "fig_gie_limit_sensitivity_pairwise_auc.png"),
    ):
        plot_stage(
            epoch_rows,
            stage,
            args.output_dir / filename,
        )

    metadata = {
        "artifact": "LE_GIE_finite_sample_sensitivity",
        "theoretical_estimator":
            "ell_hat = ell_err_hat + log(abs(m_GIE_hat))",
        "theoretical_formula_changed": False,
        "error_limit_method": "aitken_guarded",
        "k_values": [int(k) for k in args.k_values],
        "reference_methods": [
            "mean",
            "leave_one_out_mean",
            "median",
            "trimmed_mean_05",
            "trimmed_mean_10",
        ],
        "gie_limit_methods": [
            "last3_mean",
            "next",
            "aitken_guarded",
        ],
        "top_fractions": [float(q) for q in args.top_fractions],
        "target_fpr": float(args.target_fpr),
        "oracle_warning":
            "oracle TPR uses known anomaly labels and is evaluation-only",
        "selection_warning":
            "This run is exploratory. Validate any chosen configuration on an independent training seed/run.",
    }

    with (
        args.output_dir / "le_improvement_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
