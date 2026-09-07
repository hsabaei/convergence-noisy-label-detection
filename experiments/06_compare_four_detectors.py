#!/usr/bin/env python
"""
Compare four temporal noisy-label detectors for CKL and LE-GIE.

Frozen score definitions
------------------------
CKL:
    raw CKL trajectory, standardized within observed class at each epoch.

LE-GIE:
    ell_hat = ell_err + log(abs(m_GIE)),
    then standardized within observed class at each epoch.

Detector 1 — Min-run
--------------------
I_i(t) = 1{ z_i(t) >= tau }
R_i(t) = I_i(t) * (R_i(t-1) + 1)
Detect when R_i(t) >= m,
m = ceil(log(Ta/delta) / -log(alpha)).

Detector 2 — Sliding-window
---------------------------
Uses the same I_i(t).  Over an ending window of length ell,
W_i(t) = sum I_i(s).
Detect when W_i(t) >= k, where k is the Chernoff-calibrated threshold.

Detector 3 — EWMA
-----------------
E_i(t) = (1-lambda) E_i(t-1) + lambda z_i(t)
Detect when E_i(t) >= tau.

Detector 4 — Cumulative exact pairwise
--------------------------------------
At each epoch, compare a sample with every other finite sample having the same
observed label.  Higher CKL or higher signed LE-GIE is treated as more
noisy-like.  Ties contribute 0.5.

The cumulative pairwise score is cumulative win mass divided by cumulative
valid pair comparisons.  This cumulative statistic IS the temporal detector.
It is evaluated directly as a current ranking score (AUC / top-q).  No extra
sticky "ever crossed rho" threshold is applied on top of it.

Important
---------
Only observed labels are used to construct z-scores, GIE class references, and
same-class pairwise comparisons.  The true noisy-label mask is used only for
evaluation and for an empirical diagnostic of the alpha assumption.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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
    chernoff_window_k,
    compute_minrun_m,
    exact_pairwise_scores_by_class,
    threshold_continuous_score,
)
from convergence_monitoring.estimators import (
    rolling_class_reference_gie_batch,
)
from convergence_monitoring.framework import (
    run_temporal_detectors,
    standardize_monitoring_score,
)
from convergence_monitoring.proposed_le import (
    compose_le_from_error_and_m,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare min-run, sliding, EWMA, and cumulative pairwise detectors for CKL and LE-GIE."
    )

    p.add_argument(
        "--common-npz",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument(
        "--ckl-npz",
        type=Path,
        default=Path(
            "results/ckl_scores/"
            "ckl_score_trajectories.npz"
        ),
    )
    p.add_argument(
        "--le-npz",
        type=Path,
        default=Path(
            "results/proposed_le_scores_aitken_k20/"
            "proposed_le_score_trajectories.npz"
        ),
    )

    p.add_argument("--K", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=10)

    # Common z-score threshold grid.  Since both methods are class-wise
    # standardized, the SAME grid is used for CKL and LE-GIE.
    p.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=(1.0, 1.5, 2.0, 2.5, 3.0),
    )

    # Same grid used in the earlier detector study.
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.1, 0.01, 0.001),
    )
    p.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=(1e-3,),
    )
    p.add_argument(
        "--window-lengths",
        type=int,
        nargs="+",
        default=(10, 20, 30),
    )
    p.add_argument(
        "--ewma-lambdas",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2),
    )

    # Cumulative pairwise is itself the temporal detector.
    # We therefore evaluate the CURRENT cumulative score C_i(t) by ranking,
    # without adding another sticky "ever crossed rho" layer.
    p.add_argument(
        "--primary-pairwise-top-fraction",
        type=float,
        default=0.05,
        help=(
            "Primary current-score operating point for cumulative pairwise. "
            "For the 5%% synthetic-noise experiment, 0.05 matches the known "
            "injected corruption fraction."
        ),
    )
    p.add_argument(
        "--pairwise-top-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20),
        help="Sensitivity grid for current cumulative-pairwise ranking.",
    )

    p.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20),
    )

    # Used only to make an exploratory "best" table.
    p.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/four_detector_comparison_pairwise_current"
        ),
    )

    args = p.parse_args()

    if args.K < 6:
        p.error("--K must be at least 6.")
    if not 0.0 < args.target_fpr < 1.0:
        p.error("--target-fpr must be in (0,1).")

    for tau in args.taus:
        if not np.isfinite(tau):
            p.error("--taus must be finite.")
    for a in args.alphas:
        if not 0.0 < a < 1.0:
            p.error("--alphas must lie in (0,1).")
    for d in args.deltas:
        if not 0.0 < d < 1.0:
            p.error("--deltas must lie in (0,1).")
    for ell in args.window_lengths:
        if ell < 1:
            p.error("--window-lengths must be positive.")
    for lam in args.ewma_lambdas:
        if not 0.0 < lam <= 1.0:
            p.error("--ewma-lambdas must lie in (0,1].")
    if not 0.0 < args.primary_pairwise_top_fraction < 1.0:
        p.error("--primary-pairwise-top-fraction must lie in (0,1).")
    for q in args.pairwise_top_fractions:
        if not 0.0 < q < 1.0:
            p.error("--pairwise-top-fractions must lie in (0,1).")
    if args.primary_pairwise_top_fraction not in args.pairwise_top_fractions:
        args.pairwise_top_fractions = tuple(
            sorted(
                set(args.pairwise_top_fractions)
                | {float(args.primary_pairwise_top_fraction)}
            )
        )
    for q in args.top_fractions:
        if not 0.0 < q < 1.0:
            p.error("--top-fractions must lie in (0,1).")

    return args


def require_npz(path: Path, required, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} NPZ not found: {path}"
        )

    d = np.load(path, allow_pickle=False)
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(
            f"{name} NPZ missing arrays: {missing}"
        )
    return d


def check_same(name, a, b):
    if not np.array_equal(a, b):
        raise ValueError(f"{name} differs across artifacts.")


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    return binary_auc_from_scores(
        score[y],
        score[~y],
    )


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    TP = int(np.sum(pred & y))
    FP = int(np.sum(pred & ~y))
    FN = int(np.sum(~pred & y))
    TN = int(np.sum(~pred & ~y))

    TPR = TP / (TP + FN) if TP + FN else np.nan
    FPR = FP / (FP + TN) if FP + TN else np.nan
    precision = TP / (TP + FP) if TP + FP else np.nan
    recall = TPR
    F1 = (
        2.0 * precision * recall / (precision + recall)
        if (
            np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0.0
        )
        else 0.0
    )
    specificity = TN / (TN + FP) if TN + FP else np.nan
    balanced_accuracy = (
        0.5 * (TPR + specificity)
        if np.isfinite(TPR) and np.isfinite(specificity)
        else np.nan
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "TPR": float(TPR),
        "FPR": float(FPR),
        "precision": float(precision),
        "F1": float(F1),
        "balanced_accuracy": float(balanced_accuracy),
    }


def first_hit_epoch(hit_tn, start_epoch):
    hit_tn = np.asarray(hit_tn, dtype=bool)
    N = hit_tn.shape[1]
    out = np.full(N, -1, dtype=np.int32)

    any_hit = hit_tn.any(axis=0)
    if np.any(any_hit):
        out[any_hit] = (
            np.argmax(hit_tn[:, any_hit], axis=0)
            + int(start_epoch)
        ).astype(np.int32)

    return out


def detection_curve(ever_tn, y):
    ever_tn = np.asarray(ever_tn, dtype=bool)
    y = np.asarray(y, dtype=bool)
    return (
        np.mean(ever_tn[:, y], axis=1),
        np.mean(ever_tn[:, ~y], axis=1),
    )


def first_epoch_curve_reaches(curve, epochs, threshold):
    idx = np.flatnonzero(
        np.asarray(curve) >= float(threshold)
    )
    return int(epochs[idx[0]]) if idx.size else -1


def empirical_clean_alpha_sup(z_tn, y, tau):
    """Evaluation-only check of P(z>=tau | clean), not used by the detector."""
    clean = ~np.asarray(y, dtype=bool)
    z = np.asarray(z_tn, dtype=np.float64)

    vals = []
    for t in range(z.shape[0]):
        x = z[t, clean]
        finite = np.isfinite(x)
        if np.any(finite):
            vals.append(float(np.mean(x[finite] >= float(tau))))
    return float(max(vals)) if vals else np.nan


def summarize_sequential(
    *,
    method,
    detector,
    params,
    hit_tn,
    ever_tn,
    y,
    epochs,
    base_auc_max,
    base_auc_epoch,
    alpha_empirical_clean_sup=np.nan,
    calibration_feasible=True,
):
    pred = ever_tn[-1]
    metrics = binary_metrics(y, pred)

    noisy_curve, clean_curve = detection_curve(
        ever_tn,
        y,
    )

    fh = first_hit_epoch(
        hit_tn,
        int(epochs[0]),
    )

    noisy_fh = fh[np.asarray(y, dtype=bool)]
    clean_fh = fh[~np.asarray(y, dtype=bool)]
    noisy_fh = noisy_fh[noisy_fh >= 0]
    clean_fh = clean_fh[clean_fh >= 0]

    return {
        "method": method,
        "detector": detector,
        "tau": params.get("tau", ""),
        "alpha": params.get("alpha", ""),
        "delta": params.get("delta", ""),
        "m": params.get("m", ""),
        "ell": params.get("ell", ""),
        "k": params.get("k", ""),
        "lambda": params.get("lambda", ""),
        "rho": params.get("rho", ""),
        "calibration_feasible": bool(calibration_feasible),
        "empirical_clean_alpha_sup": alpha_empirical_clean_sup,
        "alpha_bound_satisfied_empirically": (
            bool(params["alpha"] >= alpha_empirical_clean_sup)
            if (
                "alpha" in params
                and np.isfinite(alpha_empirical_clean_sup)
            )
            else ""
        ),
        **metrics,
        "final_noisy_detection_rate": float(noisy_curve[-1]),
        "final_clean_false_alarm_rate": float(clean_curve[-1]),
        "median_first_hit_noisy": (
            float(np.median(noisy_fh))
            if noisy_fh.size
            else np.nan
        ),
        "median_first_hit_clean": (
            float(np.median(clean_fh))
            if clean_fh.size
            else np.nan
        ),
        "epoch_at_50pct_noisy_detection":
            first_epoch_curve_reaches(noisy_curve, epochs, 0.50),
        "epoch_at_80pct_noisy_detection":
            first_epoch_curve_reaches(noisy_curve, epochs, 0.80),
        "epoch_at_90pct_noisy_detection":
            first_epoch_curve_reaches(noisy_curve, epochs, 0.90),
        "base_score_max_auc": float(base_auc_max),
        "base_score_argmax_epoch": int(base_auc_epoch),
    }


def top_fraction_metrics(y, score, q):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    N = y.size
    finite_idx = np.flatnonzero(np.isfinite(score))
    target_k = max(1, int(round(float(q) * N)))
    k = min(target_k, finite_idx.size)

    selected = np.zeros(N, dtype=bool)
    if k:
        fs = score[finite_idx]
        local = np.argpartition(fs, -k)[-k:]
        selected[finite_idx[local]] = True

    out = binary_metrics(y, selected)
    out.update({
        "q": float(q),
        "n_selected": int(np.sum(selected)),
    })
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


def select_best(rows, target_fpr):
    """Strict exploratory selection under the stated constraints.

    Selection requirements
    ----------------------
    1. ``calibration_feasible`` must be True.
    2. For min-run and sliding-window, the empirical clean exceedance check
       must satisfy alpha_empirical <= alpha.  This is an evaluation-only
       diagnostic that uses the known clean/noisy mask; it is not a deployable
       calibration procedure.
    3. Final cumulative detector FPR must satisfy FPR <= target_fpr.

    There is intentionally NO fallback to best F1 when no configuration meets
    the constraints.  Such a method-detector pair is reported explicitly as
    having no feasible tested configuration.
    """
    best_rows = []
    no_feasible_rows = []

    keys = sorted(
        set(
            (r["method"], r["detector"])
            for r in rows
        )
    )

    for method, detector in keys:
        group = [
            r for r in rows
            if (
                r["method"] == method
                and r["detector"] == detector
            )
        ]

        calibration_valid = []
        for r in group:
            if not bool(r["calibration_feasible"]):
                continue

            if detector in {"min_run", "sliding_window"}:
                if r.get("alpha_bound_satisfied_empirically", "") is not True:
                    continue

            calibration_valid.append(r)

        fpr_feasible = [
            r for r in calibration_valid
            if (
                np.isfinite(r["FPR"])
                and r["FPR"] <= target_fpr
            )
        ]

        if not fpr_feasible:
            finite_fprs = [
                float(r["FPR"])
                for r in calibration_valid
                if np.isfinite(r["FPR"])
            ]

            no_feasible_rows.append({
                "method": method,
                "detector": detector,
                "selection_status": "no_feasible_tested_configuration",
                "target_fpr": float(target_fpr),
                "n_total_tested": int(len(group)),
                "n_calibration_valid": int(len(calibration_valid)),
                "minimum_fpr_among_calibration_valid": (
                    float(min(finite_fprs))
                    if finite_fprs
                    else np.nan
                ),
                "reason": (
                    "No tested configuration simultaneously satisfied "
                    "calibration requirements and FPR <= target."
                ),
            })
            continue

        candidates = sorted(
            fpr_feasible,
            key=lambda r: (
                -r["TPR"],
                r["FPR"],
                -r["precision"],
                (
                    r["median_first_hit_noisy"]
                    if np.isfinite(r["median_first_hit_noisy"])
                    else float("inf")
                ),
            ),
        )

        best = dict(candidates[0])
        best["selection_status"] = "feasible"
        best["target_fpr"] = float(target_fpr)
        best_rows.append(best)

    return best_rows, no_feasible_rows


def base_auc_curve(score_nt, y, epochs, start_col):
    rows = []
    aucs = np.full(score_nt.shape[1], np.nan, dtype=np.float64)

    for t in range(start_col, score_nt.shape[1]):
        aucs[t] = auc_binary(
            y,
            score_nt[:, t],
        )
        rows.append({
            "epoch": int(epochs[t]),
            "auc": aucs[t],
        })

    finite = np.flatnonzero(np.isfinite(aucs))
    best = finite[np.nanargmax(aucs[finite])]
    return rows, aucs, float(aucs[best]), int(epochs[best])


def build_detector_rows(
    *,
    method,
    z_tn,
    y,
    epochs_analysis,
    args,
    base_auc_max,
    base_auc_epoch,
):
    rows = []
    Ta = z_tn.shape[0]

    alpha_emp_by_tau = {
        float(tau): empirical_clean_alpha_sup(
            z_tn,
            y,
            tau,
        )
        for tau in args.taus
    }

    # ----------------------------------------------------------
    # 1) Min-run
    # ----------------------------------------------------------
    for tau, alpha, delta in itertools.product(
        args.taus,
        args.alphas,
        args.deltas,
    ):
        m = compute_minrun_m(
            Ta,
            alpha,
            delta,
        )

        result = run_temporal_detectors(
            z_tn,
            tau=tau,
            minrun_m=m,
            sliding_length=1,
            sliding_k=2,  # deliberately impossible placeholder; output ignored
            ewma_lambda=1.0,
        )

        d = result["minrun"]

        rows.append(
            summarize_sequential(
                method=method,
                detector="min_run",
                params={
                    "tau": float(tau),
                    "alpha": float(alpha),
                    "delta": float(delta),
                    "m": int(m),
                },
                hit_tn=d["hit"],
                ever_tn=d["ever_detected"],
                y=y,
                epochs=epochs_analysis,
                base_auc_max=base_auc_max,
                base_auc_epoch=base_auc_epoch,
                alpha_empirical_clean_sup=alpha_emp_by_tau[float(tau)],
                calibration_feasible=(m <= Ta),
            )
        )

    # ----------------------------------------------------------
    # 2) Sliding window
    # ----------------------------------------------------------
    for tau, alpha, delta, ell in itertools.product(
        args.taus,
        args.alphas,
        args.deltas,
        args.window_lengths,
    ):
        if ell > Ta:
            continue

        k, mu, L = chernoff_window_k(
            Ta,
            ell,
            alpha,
            delta,
        )

        result = run_temporal_detectors(
            z_tn,
            tau=tau,
            minrun_m=Ta + 1,  # impossible placeholder; output ignored
            sliding_length=ell,
            sliding_k=k,
            ewma_lambda=1.0,
        )

        d = result["sliding_window"]

        row = summarize_sequential(
            method=method,
            detector="sliding_window",
            params={
                "tau": float(tau),
                "alpha": float(alpha),
                "delta": float(delta),
                "ell": int(ell),
                "k": int(k),
            },
            hit_tn=d["hit"],
            ever_tn=d["ever_detected"],
            y=y,
            epochs=epochs_analysis,
            base_auc_max=base_auc_max,
            base_auc_epoch=base_auc_epoch,
            alpha_empirical_clean_sup=alpha_emp_by_tau[float(tau)],
            calibration_feasible=(k <= ell),
        )
        row["chernoff_mu"] = float(mu)
        row["chernoff_L"] = float(L)
        rows.append(row)

    # ----------------------------------------------------------
    # 3) EWMA
    # ----------------------------------------------------------
    for tau, lam in itertools.product(
        args.taus,
        args.ewma_lambdas,
    ):
        result = run_temporal_detectors(
            z_tn,
            tau=tau,
            minrun_m=Ta + 1,
            sliding_length=1,
            sliding_k=2,
            ewma_lambda=lam,
        )

        d = result["ewma"]

        rows.append(
            summarize_sequential(
                method=method,
                detector="ewma",
                params={
                    "tau": float(tau),
                    "lambda": float(lam),
                },
                hit_tn=d["hit"],
                ever_tn=d["ever_detected"],
                y=y,
                epochs=epochs_analysis,
                base_auc_max=base_auc_max,
                base_auc_epoch=base_auc_epoch,
            )
        )

    return rows


def top_fraction_selection(score, q):
    """Boolean top-q selection from one continuous score vector."""
    score = np.asarray(score, dtype=np.float64)
    N = score.size
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
    return selected


def top_fraction_selection_trajectory(score_nt, q):
    """Current (non-sticky) top-q selection at every epoch."""
    score_nt = np.asarray(score_nt, dtype=np.float64)
    N, T = score_nt.shape
    out = np.zeros((T, N), dtype=bool)
    for t in range(T):
        out[t] = top_fraction_selection(score_nt[:, t], q)
    return out


def cumulative_pairwise_primary_row(
    *,
    method,
    cumulative_nt,
    y,
    epochs_analysis,
    primary_q,
    base_auc_max,
    base_auc_epoch,
):
    """Primary fourth-detector row.

    The cumulative pairwise statistic C_i(t) already aggregates temporal
    evidence.  We therefore evaluate C_i(t) directly at the epoch where its
    ranking AUC is maximal, using a fixed top-q current-score operating point.

    This intentionally does NOT apply an additional sticky "ever crossed rho"
    rule.
    """
    cumulative_nt = np.asarray(cumulative_nt, dtype=np.float64)

    aucs = np.full(cumulative_nt.shape[1], np.nan, dtype=np.float64)
    for t in range(cumulative_nt.shape[1]):
        aucs[t] = auc_binary(y, cumulative_nt[:, t])

    finite = np.flatnonzero(np.isfinite(aucs))
    if finite.size == 0:
        raise RuntimeError(f"No finite cumulative-pairwise AUC for {method}.")

    best_t = int(finite[np.nanargmax(aucs[finite])])
    best_epoch = int(epochs_analysis[best_t])

    metrics = top_fraction_metrics(
        y,
        cumulative_nt[:, best_t],
        primary_q,
    )

    return {
        "method": method,
        "detector": "cumulative_pairwise",
        "selection_status": "primary_current_score_top_q",
        "q": float(primary_q),
        "evaluation_epoch": best_epoch,
        "pairwise_auc_at_evaluation_epoch": float(aucs[best_t]),
        "tau": "",
        "alpha": "",
        "delta": "",
        "m": "",
        "ell": "",
        "k": "",
        "lambda": "",
        "rho": "",
        "calibration_feasible": True,
        "empirical_clean_alpha_sup": np.nan,
        "alpha_bound_satisfied_empirically": "",
        **metrics,
        "final_noisy_detection_rate": np.nan,
        "final_clean_false_alarm_rate": np.nan,
        "median_first_hit_noisy": np.nan,
        "median_first_hit_clean": np.nan,
        "epoch_at_50pct_noisy_detection": -1,
        "epoch_at_80pct_noisy_detection": -1,
        "epoch_at_90pct_noisy_detection": -1,
        "base_score_max_auc": float(base_auc_max),
        "base_score_argmax_epoch": int(base_auc_epoch),
        "note": (
            "Current cumulative pairwise ranking at its best-AUC epoch; "
            "no extra sticky threshold layer."
        ),
    }


def max_tpr_under_fpr_budget(y, score, target_fpr):
    """Evaluation-only ROC operating point.

    Scans score thresholds and returns the largest TPR with FPR <= target_fpr.
    This uses the known noisy-label mask and is therefore an oracle diagnostic,
    not a deployable threshold-selection procedure.
    """
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    finite = np.isfinite(score)
    y_f = y[finite]
    s_f = score[finite]

    n_pos = int(np.sum(y_f))
    n_neg = int(np.sum(~y_f))
    if n_pos == 0 or n_neg == 0 or s_f.size == 0:
        return {
            "threshold": np.nan,
            "TPR": np.nan,
            "FPR": np.nan,
            "precision": np.nan,
            "n_selected": 0,
        }

    order = np.argsort(-s_f, kind="mergesort")
    ys = y_f[order]
    ss = s_f[order]

    tp = np.cumsum(ys.astype(np.int64))
    fp = np.cumsum((~ys).astype(np.int64))

    tpr = tp / n_pos
    fpr = fp / n_neg

    ok = np.flatnonzero(fpr <= float(target_fpr))
    if ok.size == 0:
        return {
            "threshold": np.inf,
            "TPR": 0.0,
            "FPR": 0.0,
            "precision": np.nan,
            "n_selected": 0,
        }

    # Max TPR; among ties prefer lower FPR and fewer selected.
    best_tpr = np.nanmax(tpr[ok])
    cand = ok[np.isclose(tpr[ok], best_tpr)]
    if cand.size > 1:
        cand = cand[np.argsort(fpr[cand], kind="mergesort")]
    idx = int(cand[0])

    selected = idx + 1
    precision = tp[idx] / selected

    return {
        "threshold": float(ss[idx]),
        "TPR": float(tpr[idx]),
        "FPR": float(fpr[idx]),
        "precision": float(precision),
        "n_selected": int(selected),
        "selection_fraction": float(selected / y.size),
    }



def reconstruct_best_curves(
    row,
    *,
    z_tn,
    pair_cum_nt,
    epochs_analysis,
):
    detector = row["detector"]

    if detector == "min_run":
        result = run_temporal_detectors(
            z_tn,
            tau=float(row["tau"]),
            minrun_m=int(row["m"]),
            sliding_length=1,
            sliding_k=2,
            ewma_lambda=1.0,
        )
        return result["minrun"]["ever_detected"]

    if detector == "sliding_window":
        result = run_temporal_detectors(
            z_tn,
            tau=float(row["tau"]),
            minrun_m=z_tn.shape[0] + 1,
            sliding_length=int(row["ell"]),
            sliding_k=int(row["k"]),
            ewma_lambda=1.0,
        )
        return result["sliding_window"]["ever_detected"]

    if detector == "ewma":
        result = run_temporal_detectors(
            z_tn,
            tau=float(row["tau"]),
            minrun_m=z_tn.shape[0] + 1,
            sliding_length=1,
            sliding_k=2,
            ewma_lambda=float(row["lambda"]),
        )
        return result["ewma"]["ever_detected"]

    if detector == "cumulative_pairwise":
        # Current top-q selection from the already cumulative statistic.
        # No "ever crossed threshold" / sticky layer is applied.
        return top_fraction_selection_trajectory(
            pair_cum_nt,
            float(row["q"]),
        )

    raise ValueError(detector)


def plot_best_by_method(
    best_rows,
    *,
    method,
    z_tn,
    pair_cum_nt,
    y,
    epochs_analysis,
    path,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for row in best_rows:
        if row["method"] != method:
            continue

        ever = reconstruct_best_curves(
            row,
            z_tn=z_tn,
            pair_cum_nt=pair_cum_nt,
            epochs_analysis=epochs_analysis,
        )

        tpr, fpr = detection_curve(
            ever,
            y,
        )

        ax.plot(
            epochs_analysis,
            tpr,
            label=f"{row['detector']} TPR",
        )
        ax.plot(
            epochs_analysis,
            fpr,
            linestyle="--",
            label=f"{row['detector']} FPR",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detected / selected fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"{method}: best four temporal detectors"
    )
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ckl_vs_le_for_detector(
    best_rows,
    *,
    detector,
    z_ckl_tn,
    z_le_tn,
    pair_ckl_nt,
    pair_le_nt,
    y,
    epochs_analysis,
    path,
):
    rows = {
        r["method"]: r
        for r in best_rows
        if r["detector"] == detector
    }
    if "CKL" not in rows or "LE_GIE" not in rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for method, z, pair in (
        ("CKL", z_ckl_tn, pair_ckl_nt),
        ("LE_GIE", z_le_tn, pair_le_nt),
    ):
        ever = reconstruct_best_curves(
            rows[method],
            z_tn=z,
            pair_cum_nt=pair,
            epochs_analysis=epochs_analysis,
        )
        tpr, fpr = detection_curve(ever, y)

        ax.plot(
            epochs_analysis,
            tpr,
            label=f"{method} TPR",
        )
        ax.plot(
            epochs_analysis,
            fpr,
            linestyle="--",
            label=f"{method} FPR",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detected / selected fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"CKL vs LE-GIE — {detector}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    common = require_npz(
        args.common_npz,
        (
            "sample_index",
            "epoch",
            "loss_traj",
            "observed_label",
            "true_label",
            "is_anomaly",
        ),
        "common loss",
    )
    ckl = require_npz(
        args.ckl_npz,
        (
            "sample_index",
            "epoch",
            "observed_label",
            "is_anomaly",
            "ckl_traj",
        ),
        "CKL",
    )
    le = require_npz(
        args.le_npz,
        (
            "sample_index",
            "epoch",
            "observed_label",
            "is_anomaly",
        ),
        "LE",
    )

    sample_index = np.asarray(common["sample_index"], dtype=np.int64)
    epochs = np.asarray(common["epoch"], dtype=np.int64)
    labels = np.asarray(common["observed_label"], dtype=np.int64)
    true_label = np.asarray(common["true_label"], dtype=np.int64)
    y = np.asarray(common["is_anomaly"], dtype=bool)
    loss_traj = np.asarray(common["loss_traj"], dtype=np.float64)

    check_same(
        "sample_index CKL/common",
        sample_index,
        np.asarray(ckl["sample_index"], dtype=np.int64),
    )
    check_same(
        "sample_index LE/common",
        sample_index,
        np.asarray(le["sample_index"], dtype=np.int64),
    )
    check_same(
        "epoch CKL/common",
        epochs,
        np.asarray(ckl["epoch"], dtype=np.int64),
    )
    check_same(
        "epoch LE/common",
        epochs,
        np.asarray(le["epoch"], dtype=np.int64),
    )
    check_same(
        "observed labels CKL/common",
        labels,
        np.asarray(ckl["observed_label"], dtype=np.int64),
    )
    check_same(
        "observed labels LE/common",
        labels,
        np.asarray(le["observed_label"], dtype=np.int64),
    )
    check_same(
        "noisy mask CKL/common",
        y,
        np.asarray(ckl["is_anomaly"], dtype=bool),
    )
    check_same(
        "noisy mask LE/common",
        y,
        np.asarray(le["is_anomaly"], dtype=bool),
    )

    ckl_nt = np.asarray(
        ckl["ckl_traj"],
        dtype=np.float64,
    )
    # ----------------------------------------------------------
    # LE-GIE source
    # ----------------------------------------------------------
    # Final/frozen artifacts contain the COMPLETE consistent LE trajectory.
    # Prefer that directly.  Legacy artifacts containing only ell_err_traj
    # still follow the previous recomputation path unchanged.
    if "le_gie_traj" in le.files:
        le_gie_nt = np.asarray(
            le["le_gie_traj"],
            dtype=np.float64,
        )
        le_score_source = "precomputed_le_gie_traj"
    elif "lambda_traj" in le.files:
        le_gie_nt = np.asarray(
            le["lambda_traj"],
            dtype=np.float64,
        )
        le_score_source = "precomputed_lambda_traj"
    elif "ell_err_traj" in le.files:
        ell_err_nt = np.asarray(
            le["ell_err_traj"],
            dtype=np.float64,
        )

        gie = rolling_class_reference_gie_batch(
            loss_traj,
            labels,
            K=args.K,
            num_classes=args.num_classes,
        )
        id_gie_nt = np.asarray(
            gie["id_gie_traj"],
            dtype=np.float64,
        )
        le_gie_nt = compose_le_from_error_and_m(
            ell_err_nt,
            id_gie_nt,
        )
        le_score_source = "legacy_ell_err_plus_recomputed_gie"
    else:
        raise KeyError(
            "LE artifact must contain le_gie_traj, lambda_traj, "
            "or legacy ell_err_traj."
        )

    # Common first epoch where both raw score families are available.
    common_cols = np.flatnonzero(
        np.any(np.isfinite(ckl_nt), axis=0)
        & np.any(np.isfinite(le_gie_nt), axis=0)
    )
    if common_cols.size == 0:
        raise RuntimeError("No common finite CKL/LE-GIE epoch.")

    start_col = int(common_cols[0])
    start_epoch = int(epochs[start_col])
    epochs_analysis = epochs[start_col:]
    Ta = int(epochs_analysis.size)

    # ----------------------------------------------------------
    # Class-wise z-score is the common input to detectors 1-3.
    # No extra class-mean subtraction after this step.
    # ----------------------------------------------------------
    z_ckl_full_tn = standardize_monitoring_score(
        ckl_nt.T,
        labels,
        direction="higher",
        num_classes=args.num_classes,
        start_index=start_col,
    )
    z_le_full_tn = standardize_monitoring_score(
        le_gie_nt.T,
        labels,
        direction="higher",
        num_classes=args.num_classes,
        start_index=start_col,
    )

    z_ckl_tn = z_ckl_full_tn[start_col:]
    z_le_tn = z_le_full_tn[start_col:]

    # ----------------------------------------------------------
    # Report BOTH raw-score and class-z AUCs on the same interval.
    #
    # Detectors 1-3 consume class-wise z-scores, so the ambiguous
    # ``base_score_max_auc`` field below is defined as the CLASS-Z AUC.
    # Raw AUC is retained explicitly as a separate diagnostic.
    # ----------------------------------------------------------
    ckl_raw_auc_rows, ckl_raw_auc_curve, ckl_raw_auc_max, ckl_raw_auc_ep = (
        base_auc_curve(
            ckl_nt,
            y,
            epochs,
            start_col,
        )
    )
    le_raw_auc_rows, le_raw_auc_curve, le_raw_auc_max, le_raw_auc_ep = (
        base_auc_curve(
            le_gie_nt,
            y,
            epochs,
            start_col,
        )
    )

    ckl_z_nt = z_ckl_full_tn.T
    le_z_nt = z_le_full_tn.T

    ckl_z_auc_rows, ckl_z_auc_curve, ckl_z_auc_max, ckl_z_auc_ep = (
        base_auc_curve(
            ckl_z_nt,
            y,
            epochs,
            start_col,
        )
    )
    le_z_auc_rows, le_z_auc_curve, le_z_auc_max, le_z_auc_ep = (
        base_auc_curve(
            le_z_nt,
            y,
            epochs,
            start_col,
        )
    )

    base_auc_rows = []
    for method, variant, rr in (
        ("CKL", "raw", ckl_raw_auc_rows),
        ("CKL", "class_z", ckl_z_auc_rows),
        ("LE_GIE", "raw", le_raw_auc_rows),
        ("LE_GIE", "class_z", le_z_auc_rows),
    ):
        for r in rr:
            base_auc_rows.append({
                "method": method,
                "score_variant": variant,
                **r,
            })

    write_csv(
        args.output_dir / "base_score_auc_by_epoch.csv",
        base_auc_rows,
    )

    # ----------------------------------------------------------
    # Detectors 1-3.
    # ----------------------------------------------------------
    rows = []
    rows.extend(
        build_detector_rows(
            method="CKL",
            z_tn=z_ckl_tn,
            y=y,
            epochs_analysis=epochs_analysis,
            args=args,
            base_auc_max=ckl_z_auc_max,
            base_auc_epoch=ckl_z_auc_ep,
        )
    )
    rows.extend(
        build_detector_rows(
            method="LE_GIE",
            z_tn=z_le_tn,
            y=y,
            epochs_analysis=epochs_analysis,
            args=args,
            base_auc_max=le_z_auc_max,
            base_auc_epoch=le_z_auc_ep,
        )
    )

    # ----------------------------------------------------------
    # Detector 4: exact all-peer cumulative pairwise.
    # Pairwise ordering is identical for raw and within-class z-score at
    # a fixed epoch.  We use the raw oriented scores.
    # ----------------------------------------------------------
    pair_ckl = exact_pairwise_scores_by_class(
        ckl_nt,
        labels,
        start_index=start_col,
    )
    pair_le = exact_pairwise_scores_by_class(
        le_gie_nt,
        labels,
        start_index=start_col,
    )

    pair_ckl_cum_nt = np.asarray(
        pair_ckl["cumulative_score"],
        dtype=np.float64,
    )[:, start_col:]
    pair_le_cum_nt = np.asarray(
        pair_le["cumulative_score"],
        dtype=np.float64,
    )[:, start_col:]

    # The fourth detector is the cumulative pairwise statistic itself.
    # Do NOT add a second sticky threshold layer on top of it.
    pairwise_primary_rows = [
        cumulative_pairwise_primary_row(
            method="CKL",
            cumulative_nt=pair_ckl_cum_nt,
            y=y,
            epochs_analysis=epochs_analysis,
            primary_q=args.primary_pairwise_top_fraction,
            base_auc_max=ckl_z_auc_max,
            base_auc_epoch=ckl_z_auc_ep,
        ),
        cumulative_pairwise_primary_row(
            method="LE_GIE",
            cumulative_nt=pair_le_cum_nt,
            y=y,
            epochs_analysis=epochs_analysis,
            primary_q=args.primary_pairwise_top_fraction,
            base_auc_max=le_z_auc_max,
            base_auc_epoch=le_z_auc_ep,
        ),
    ]

    # Add explicit score-level summaries.  ``base_score_max_auc`` already
    # equals class-z AUC for compatibility with the previous table.
    score_summaries = {
        "CKL": {
            "base_raw_score_max_auc": float(ckl_raw_auc_max),
            "base_raw_score_argmax_epoch": int(ckl_raw_auc_ep),
            "base_z_score_max_auc": float(ckl_z_auc_max),
            "base_z_score_argmax_epoch": int(ckl_z_auc_ep),
        },
        "LE_GIE": {
            "base_raw_score_max_auc": float(le_raw_auc_max),
            "base_raw_score_argmax_epoch": int(le_raw_auc_ep),
            "base_z_score_max_auc": float(le_z_auc_max),
            "base_z_score_argmax_epoch": int(le_z_auc_ep),
        },
    }

    for row in rows:
        row["base_score_variant"] = "class_z"
        row.update(score_summaries[row["method"]])

    write_csv(
        args.output_dir / "all_detector_combinations.csv",
        rows,
    )

    # Strict FPR/calibration selection applies to the first three
    # z-threshold-based detectors.  Cumulative pairwise is evaluated by its
    # fixed current-score top-q operating point, matching the prior experiment.
    best_rows, no_feasible_rows = select_best(
        rows,
        target_fpr=args.target_fpr,
    )
    best_rows.extend(pairwise_primary_rows)

    write_csv(
        args.output_dir / "best_by_method_detector.csv",
        best_rows,
    )

    if no_feasible_rows:
        write_csv(
            args.output_dir / "no_feasible_method_detector.csv",
            no_feasible_rows,
        )

    # ----------------------------------------------------------
    # Continuous cumulative-pairwise AUC + top-q by epoch.
    # ----------------------------------------------------------
    pair_rows = []
    for method, score_nt in (
        ("CKL", pair_ckl_cum_nt),
        ("LE_GIE", pair_le_cum_nt),
    ):
        for t, ep in enumerate(epochs_analysis):
            score = score_nt[:, t]
            auc = auc_binary(y, score)

            base = {
                "method": method,
                "epoch": int(ep),
                "auc": auc,
            }

            for q in args.top_fractions:
                m = top_fraction_metrics(
                    y,
                    score,
                    q,
                )
                pair_rows.append({
                    **base,
                    **m,
                })

    write_csv(
        args.output_dir / "cumulative_pairwise_auc_topq_by_epoch.csv",
        pair_rows,
    )

    # ----------------------------------------------------------
    # Extra cumulative-LE/CKL diagnostics:
    #   (a) top-q sensitivity at the best-AUC epoch;
    #   (b) maximum TPR attainable at each epoch under FPR <= target.
    #
    # (b) uses the known anomaly mask and is explicitly ORACLE / evaluation-only.
    # ----------------------------------------------------------
    pairwise_sensitivity_rows = []
    pairwise_fpr_budget_rows = []

    for method, score_nt in (
        ("CKL", pair_ckl_cum_nt),
        ("LE_GIE", pair_le_cum_nt),
    ):
        aucs = np.asarray(
            [
                auc_binary(y, score_nt[:, t])
                for t in range(score_nt.shape[1])
            ],
            dtype=np.float64,
        )
        finite = np.flatnonzero(np.isfinite(aucs))
        best_t = int(finite[np.nanargmax(aucs[finite])])
        best_ep = int(epochs_analysis[best_t])

        for q in args.pairwise_top_fractions:
            m = top_fraction_metrics(
                y,
                score_nt[:, best_t],
                q,
            )
            pairwise_sensitivity_rows.append({
                "method": method,
                "epoch": best_ep,
                "auc": float(aucs[best_t]),
                **m,
            })

        for t, ep in enumerate(epochs_analysis):
            op = max_tpr_under_fpr_budget(
                y,
                score_nt[:, t],
                args.target_fpr,
            )
            pairwise_fpr_budget_rows.append({
                "method": method,
                "epoch": int(ep),
                "auc": float(aucs[t]),
                "target_fpr": float(args.target_fpr),
                **op,
                "oracle_evaluation_only": True,
            })

    write_csv(
        args.output_dir / "pairwise_topq_sensitivity_at_best_auc.csv",
        pairwise_sensitivity_rows,
    )
    write_csv(
        args.output_dir / "pairwise_oracle_max_tpr_under_fpr_budget_by_epoch.csv",
        pairwise_fpr_budget_rows,
    )

    # Save compact arrays.
    np.savez_compressed(
        args.output_dir / "four_detector_score_trajectories.npz",
        sample_index=sample_index,
        epoch=epochs,
        observed_label=labels,
        true_label=true_label,
        is_anomaly=y,
        ckl_raw=ckl_nt.astype(np.float32),
        le_gie=le_gie_nt.astype(np.float32),
        z_ckl=z_ckl_full_tn.T.astype(np.float32),
        z_le_gie=z_le_full_tn.T.astype(np.float32),
        ckl_pairwise_all_cumulative=np.asarray(
            pair_ckl["cumulative_score"],
            dtype=np.float32,
        ),
        le_pairwise_all_cumulative=np.asarray(
            pair_le["cumulative_score"],
            dtype=np.float32,
        ),
        common_start_col=np.asarray(start_col, dtype=np.int64),
        common_start_epoch=np.asarray(start_epoch, dtype=np.int64),
    )

    # ----------------------------------------------------------
    # Plots for the exploratory best configurations.
    # ----------------------------------------------------------
    plot_best_by_method(
        best_rows,
        method="CKL",
        z_tn=z_ckl_tn,
        pair_cum_nt=pair_ckl_cum_nt,
        y=y,
        epochs_analysis=epochs_analysis,
        path=args.output_dir / "fig_best_four_detectors_ckl.png",
    )
    plot_best_by_method(
        best_rows,
        method="LE_GIE",
        z_tn=z_le_tn,
        pair_cum_nt=pair_le_cum_nt,
        y=y,
        epochs_analysis=epochs_analysis,
        path=args.output_dir / "fig_best_four_detectors_le_gie.png",
    )

    for detector in (
        "min_run",
        "sliding_window",
        "ewma",
        "cumulative_pairwise",
    ):
        plot_ckl_vs_le_for_detector(
            best_rows,
            detector=detector,
            z_ckl_tn=z_ckl_tn,
            z_le_tn=z_le_tn,
            pair_ckl_nt=pair_ckl_cum_nt,
            pair_le_nt=pair_le_cum_nt,
            y=y,
            epochs_analysis=epochs_analysis,
            path=args.output_dir / f"fig_ckl_vs_le_{detector}.png",
        )

    metadata = {
        "artifact": "four_temporal_detector_comparison",
        "common_npz": str(args.common_npz),
        "ckl_npz": str(args.ckl_npz),
        "le_npz": str(args.le_npz),
        "K": int(args.K),
        "le_formula": "ell_err + log(abs(m_GIE))",
        "le_score_source": le_score_source,
        "gie_reference": "mean loss trajectory of samples sharing observed label",
        "common_start_epoch": start_epoch,
        "first_three_detector_input":
            "within-observed-class z-score at each epoch",
        "no_second_class_mean_subtraction": True,
        "minrun_reset_rule":
            "R(t)=I(t)*(R(t-1)+1)",
        "sliding_k_policy":
            "unclipped Chernoff k; k>ell is retained and marked infeasible",
        "pairwise_mode":
            "exact all finite same-observed-class peers",
        "pairwise_accumulation":
            "cumulative win mass / cumulative valid comparisons",
        "taus": [float(x) for x in args.taus],
        "alphas": [float(x) for x in args.alphas],
        "deltas": [float(x) for x in args.deltas],
        "window_lengths": [int(x) for x in args.window_lengths],
        "ewma_lambdas": [float(x) for x in args.ewma_lambdas],
        "primary_pairwise_top_fraction":
            float(args.primary_pairwise_top_fraction),
        "pairwise_top_fractions":
            [float(x) for x in args.pairwise_top_fractions],
        "cumulative_pairwise_decision_rule":
            "current cumulative score ranked at each epoch; primary decision is current top-q; no extra sticky threshold",
        "pairwise_oracle_fpr_budget_diagnostic":
            "evaluation-only maximum TPR subject to FPR target, computed using known anomaly labels",
        "base_score_auc_reporting":
            "both raw and within-observed-class z-score AUC are reported; base_score_max_auc in detector tables denotes class_z",
        "strict_best_selection":
            "applies to min-run, sliding-window, and EWMA only; cumulative pairwise uses fixed current-score top-q",
        "selection_rule":
            "calibration_feasible; min-run/sliding also require empirical clean alpha bound; FPR <= target; maximize TPR; no fallback to best F1",
        "alpha_bound_check_uses_known_clean_mask":
            True,
        "top_fractions": [float(x) for x in args.top_fractions],
        "target_fpr_for_exploratory_best_table":
            float(args.target_fpr),
        "best_table_warning":
            "best hyperparameters and the empirical alpha-bound check use this labeled evaluation run; validate on an independent seed/run before final claims",
        "n_samples": int(sample_index.size),
        "n_noisy": int(np.sum(y)),
    }

    with (
        args.output_dir / "four_detector_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("========================================")
    print("Four temporal detectors: CKL vs LE-GIE")
    print("========================================")
    print(f"Common monitoring start: epoch {start_epoch}")
    print("First 3 detectors use class-wise z-score directly.")
    print("Fourth detector uses exact all-peer cumulative comparison.")
    print()
    print(
        "Base-score AUC summary:"
    )
    print(
        f"    CKL raw={ckl_raw_auc_max:.6f} @ {ckl_raw_auc_ep}, "
        f"class-z={ckl_z_auc_max:.6f} @ {ckl_z_auc_ep}"
    )
    print(
        f" LE-GIE raw={le_raw_auc_max:.6f} @ {le_raw_auc_ep}, "
        f"class-z={le_z_auc_max:.6f} @ {le_z_auc_ep}"
    )
    print()

    print(
        f"Strict feasible configurations under FPR <= "
        f"{args.target_fpr:.3f}:"
    )

    for row in best_rows:
        hp = []
        for key in ("tau", "alpha", "delta", "m", "ell", "k", "lambda", "q", "evaluation_epoch"):
            value = row.get(key, "")
            if value != "":
                hp.append(f"{key}={value}")

        print(
            f"{row['method']:>7s} | "
            f"{row['detector']:<20s} | "
            f"TPR={row['TPR']:.4f} "
            f"FPR={row['FPR']:.4f} "
            f"F1={row['F1']:.4f} | "
            + ", ".join(hp)
        )

    if no_feasible_rows:
        print()
        print("No feasible tested configuration:")
        for row in no_feasible_rows:
            min_fpr = row["minimum_fpr_among_calibration_valid"]
            min_fpr_text = (
                f"{min_fpr:.4f}"
                if np.isfinite(min_fpr)
                else "N/A"
            )
            print(
                f"{row['method']:>7s} | "
                f"{row['detector']:<20s} | "
                f"minimum calibration-valid FPR={min_fpr_text}"
            )

    # Evaluation-only indication of whether the cumulative score can exceed
    # the primary top-5% TPR while still staying under the same FPR budget.
    le_oracle = [
        r for r in pairwise_fpr_budget_rows
        if r["method"] == "LE_GIE"
        and np.isfinite(r["TPR"])
    ]
    if le_oracle:
        best_oracle = max(
            le_oracle,
            key=lambda r: (r["TPR"], -r["FPR"]),
        )
        print()
        print(
            "LE-GIE cumulative pairwise oracle under "
            f"FPR <= {args.target_fpr:.3f}: "
            f"TPR={best_oracle['TPR']:.4f}, "
            f"FPR={best_oracle['FPR']:.4f}, "
            f"epoch={best_oracle['epoch']}, "
            f"selected_fraction={best_oracle['selection_fraction']:.4f}"
        )
        print(
            "  (evaluation-only threshold; do not treat this as a "
            "deployable tuned detector without independent calibration)"
        )

    print()
    print(f"LE score source: {le_score_source}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
