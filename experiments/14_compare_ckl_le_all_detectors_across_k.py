#!/usr/bin/env python
"""
Revised CKL-vs-LE all-detector comparison for K = 20, 30, 40.

Corrections implemented
=======================
1) Min-run length m is NOT tuned.
   It is calculated from
       m = ceil( log(T_monitor / delta) / (-log(alpha)) )
   with alpha=0.10 and delta=0.001.

2) CKL uses ONE COMMON z-threshold tau for all three threshold-based
   temporal detectors:
       - min-run
       - sliding-window
       - EWMA
   The same tau is also used across K=20,30,40.

3) CKL score configuration:
       K in {20,30,40}
       CKL limit rule = last-three mean

4) LE-GIE score configuration:
       K in {20,30,40}
       ell_err limit = next sample
       GIE sample limit = next sample
       GIE class-reference limit = next sample

5) LE temporal detectors are rank-based:
       q = 0.10 fixed
       - dynamic min-run: within-class top-q hit
       - dynamic sliding: within-class top-q hit
       - rank-EWMA: EWMA of within-class percentile, current top-q
       - cumulative pairwise: current top-q

6) Cumulative pairwise for CKL also uses fixed q=0.10.

Common CKL threshold selection
==============================
We test tau in a small predefined grid. For each candidate tau, each CKL
threshold detector is evaluated for every K. Under the benchmark constraint
FPR <= 0.05, we take each detector/K pair's best TPR over epoch.

A candidate tau is eligible only if ALL 3 CKL threshold detectors have at
least one feasible epoch at ALL 3 K values. Among eligible candidates, choose
the single tau that maximizes the mean feasible TPR over the 9 combinations
(3 detectors x 3 K values). Tie-break by higher minimum TPR, then lower mean FPR.

This gives ONE CKL tau for all three detectors and all K values.

Important
=========
The common-tau selection and all best-epoch summaries are exploratory because
they use the labeled seed-66 benchmark. After this run, freeze the chosen tau
and detector settings before independent-seed validation.
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

from convergence_monitoring.detectors import exact_pairwise_scores_by_class, binary_auc_from_scores
from convergence_monitoring.framework import standardize_monitoring_score

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
        "--K-values",
        type=int,
        nargs="+",
        default=(20, 30, 40),
    )
    p.add_argument("--num-classes", type=int, default=10)

    # Min-run formula.
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.001)

    # CKL common threshold.
    p.add_argument(
        "--ckl-taus",
        type=float,
        nargs="+",
        default=(1.0, 1.5, 2.0, 2.5, 3.0),
    )

    # Fixed non-threshold hyperparameters for CKL.
    p.add_argument("--ckl-sliding-ell", type=int, default=20)
    p.add_argument("--ckl-sliding-k", type=int, default=14)
    p.add_argument("--ckl-ewma-lambda", type=float, default=0.20)

    # LE rank-based settings.
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--le-sliding-ell", type=int, default=30)
    p.add_argument("--le-sliding-k", type=int, default=15)
    p.add_argument("--le-rank-ewma-lambda", type=float, default=0.05)

    # Benchmark-only diagnostic.
    p.add_argument("--target-fpr", type=float, default=0.05)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/revised_ckl_vs_le_all_detectors_k_sweep"
        ),
    )
    return p.parse_args()


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if np.isfinite(precision) and precision + tpr > 0
        else np.nan
    )

    return {
        "TPR": float(tpr),
        "FPR": float(fpr),
        "precision": float(precision),
        "F1": float(f1),
        "n_selected": int(np.sum(pred)),
    }


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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def calculate_m(T_monitor, alpha, delta):
    return int(
        math.ceil(
            math.log(float(T_monitor) / float(delta))
            / (-math.log(float(alpha)))
        )
    )


def minrun_from_hits(hit_tn, m):
    T, N = hit_tn.shape
    run = np.zeros(N, dtype=np.int16)
    detected = np.zeros(N, dtype=bool)
    ever = np.zeros((T, N), dtype=bool)

    for t in range(T):
        run = np.where(hit_tn[t], run + 1, 0)
        detected |= run >= int(m)
        ever[t] = detected

    return ever


def sliding_from_hits(hit_tn, ell, k):
    T, N = hit_tn.shape
    detected = np.zeros(N, dtype=bool)
    ever = np.zeros((T, N), dtype=bool)
    csum = np.cumsum(hit_tn.astype(np.int16), axis=0)

    for t in range(T):
        count = csum[t].copy()
        left = t - int(ell)
        if left >= 0:
            count -= csum[left]

        detected |= count >= int(k)
        ever[t] = detected

    return ever


def ewma_continuous(score_tn, lam):
    score_tn = np.asarray(score_tn, dtype=np.float64)

    T, N = score_tn.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    e = np.full(N, np.nan, dtype=np.float64)

    for t in range(T):
        x = score_tn[t]
        finite = np.isfinite(x)

        init = finite & ~np.isfinite(e)
        e[init] = x[init]

        upd = finite & np.isfinite(e)
        e[upd] = (
            (1.0 - float(lam)) * e[upd]
            + float(lam) * x[upd]
        )

        out[t] = e.astype(np.float32)

    return out


def sticky_threshold(score_tn, tau):
    hit = np.asarray(score_tn >= float(tau), dtype=bool)
    detected = np.zeros(hit.shape[1], dtype=bool)
    ever = np.zeros_like(hit, dtype=bool)

    for t in range(hit.shape[0]):
        detected |= hit[t]
        ever[t] = detected

    return ever


def within_class_percentiles(score_nt, labels, start_col, num_classes):
    score_nt = np.asarray(score_nt)
    labels = np.asarray(labels, dtype=np.int64)

    N, T = score_nt.shape
    out = np.full((N, T - start_col), np.nan, dtype=np.float32)

    for u, t in enumerate(range(start_col, T)):
        s = score_nt[:, t]

        for c in range(num_classes):
            idx = np.flatnonzero(
                (labels == c) & np.isfinite(s)
            )
            if idx.size == 0:
                continue

            vals = s[idx]
            order = np.argsort(vals, kind="mergesort")

            ranks = np.empty(idx.size, dtype=np.float64)
            ranks[order] = np.arange(idx.size, dtype=np.float64)

            if idx.size == 1:
                pct = np.ones(1, dtype=np.float64)
            else:
                pct = ranks / (idx.size - 1)

            out[idx, u] = pct.astype(np.float32)

    return out


def rank_ewma(percentile_nt, lam):
    p = np.asarray(percentile_nt, dtype=np.float64)

    N, T = p.shape
    out = np.full((N, T), np.nan, dtype=np.float32)
    e = np.full(N, np.nan, dtype=np.float64)

    for t in range(T):
        x = p[:, t]
        finite = np.isfinite(x)

        init = finite & ~np.isfinite(e)
        e[init] = x[init]

        upd = finite & np.isfinite(e)
        e[upd] = (
            (1.0 - float(lam)) * e[upd]
            + float(lam) * x[upd]
        )

        out[:, t] = e.astype(np.float32)

    return out


def topq_selection(score, q):
    score = np.asarray(score)
    finite_idx = np.flatnonzero(np.isfinite(score))

    k = min(
        max(1, int(round(float(q) * score.size))),
        finite_idx.size,
    )

    selected = np.zeros(score.size, dtype=bool)
    if k == 0:
        return selected

    vals = score[finite_idx]
    local = np.argpartition(vals, -k)[-k:]
    selected[finite_idx[local]] = True

    return selected


def current_topq_trajectory(score_nt, q):
    N, T = score_nt.shape
    out = np.zeros((T, N), dtype=bool)

    for t in range(T):
        out[t] = topq_selection(score_nt[:, t], q)

    return out


def trajectory_rows(
    method,
    K,
    detector,
    family,
    params,
    pred_tn,
    y,
    epochs_analysis,
):
    rows = []

    for t, ep in enumerate(epochs_analysis):
        rows.append({
            "method": method,
            "K": int(K),
            "detector": detector,
            "detector_family": family,
            "epoch": int(ep),
            **params,
            **binary_metrics(y, pred_tn[t]),
        })

    return rows


def best_over_epoch(rows, target_fpr=None):
    group = rows

    if target_fpr is not None:
        group = [
            r for r in rows
            if (
                np.isfinite(r["FPR"])
                and r["FPR"] <= float(target_fpr)
            )
        ]

    if not group:
        return None

    group = sorted(
        group,
        key=lambda r: (
            -float(r["TPR"]),
            float(r["FPR"]),
            int(r["epoch"]),
        ),
    )
    return dict(group[0])


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    finite = np.isfinite(score)
    if np.sum(finite) < 2:
        return np.nan
    yy = y[finite]
    if np.all(yy) or np.all(~yy):
        return np.nan
    return float(binary_auc_from_scores(score[finite & y], score[finite & ~y]))

def auc_curve(y, score_nt, start_col):
    out = np.full(score_nt.shape[1], np.nan, dtype=np.float64)
    for t in range(start_col, score_nt.shape[1]):
        out[t] = auc_binary(y, score_nt[:, t])
    return out

def best_auc(auc, epochs):
    finite = np.flatnonzero(np.isfinite(auc))
    if finite.size == 0:
        return {"best_auc": np.nan, "best_epoch": -1}
    t = int(finite[np.argmax(auc[finite])])
    return {"best_auc": float(auc[t]), "best_epoch": int(epochs[t])}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Store all score trajectories only one K at a time.
    per_k = {}
    auc_rows = []

    for K in args.K_values:
        print()
        print("=" * 72)
        print(f"Computing scores for K={K}")
        print("=" * 72)

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

        start_col = K + 1
        epochs_analysis = epochs[start_col:]
        T_monitor = len(epochs_analysis)

        m = calculate_m(
            T_monitor,
            args.alpha,
            args.delta,
        )

        print(
            f"K={K}: start_epoch={int(epochs_analysis[0])}, "
            f"T_monitor={T_monitor}, m={m}"
        )

        # CKL z trajectory.
        z_full_tn = standardize_monitoring_score(
            ckl.T,
            labels,
            direction="higher",
            num_classes=args.num_classes,
            start_index=start_col,
        )
        z_tn = np.asarray(z_full_tn[start_col:])

        # LE percentile trajectory.
        pct_nt = within_class_percentiles(
            le,
            labels,
            start_col,
            args.num_classes,
        )

        # Pairwise trajectories for both.
        ckl_pair = exact_pairwise_scores_by_class(
            ckl,
            labels,
            start_index=start_col,
        )
        ckl_pair_nt = np.asarray(
            ckl_pair["cumulative_score"]
        )[:, start_col:]

        le_pair = exact_pairwise_scores_by_class(
            le,
            labels,
            start_index=start_col,
        )
        le_pair_nt = np.asarray(
            le_pair["cumulative_score"]
        )[:, start_col:]

        # AUC summaries for raw score, class-z score, and cumulative pairwise.
        ckl_raw_auc = auc_curve(y, ckl, start_col)
        ckl_z_nt = np.asarray(z_full_tn.T, dtype=np.float32)
        ckl_z_auc = auc_curve(y, ckl_z_nt, start_col)
        ckl_pair_auc = auc_curve(y, np.asarray(ckl_pair["cumulative_score"]), start_col)

        le_raw_auc = auc_curve(y, le, start_col)
        le_z_full_tn = standardize_monitoring_score(
            le.T, labels, direction="higher",
            num_classes=args.num_classes, start_index=start_col,
        )
        le_z_nt = np.asarray(le_z_full_tn.T, dtype=np.float32)
        le_z_auc = auc_curve(y, le_z_nt, start_col)
        le_pair_auc = auc_curve(y, np.asarray(le_pair["cumulative_score"]), start_col)

        for method, raw_auc, z_auc, pair_auc in (
            ("CKL", ckl_raw_auc, ckl_z_auc, ckl_pair_auc),
            ("LE_GIE", le_raw_auc, le_z_auc, le_pair_auc),
        ):
            rb = best_auc(raw_auc, epochs)
            zb = best_auc(z_auc, epochs)
            pb = best_auc(pair_auc, epochs)
            auc_rows.append({
                "method": method,
                "K": int(K),
                "common_start_epoch": int(epochs_analysis[0]),
                "raw_best_auc": rb["best_auc"],
                "raw_best_epoch": rb["best_epoch"],
                "class_z_best_auc": zb["best_auc"],
                "class_z_best_epoch": zb["best_epoch"],
                "cumulative_pairwise_best_auc": pb["best_auc"],
                "cumulative_pairwise_best_epoch": pb["best_epoch"],
            })

        per_k[int(K)] = {
            "epochs_analysis": epochs_analysis,
            "T_monitor": int(T_monitor),
            "m": int(m),
            "z_tn": z_tn.astype(np.float32, copy=False),
            "pct_nt": pct_nt.astype(np.float32, copy=False),
            "ckl_pair_nt": ckl_pair_nt.astype(np.float32, copy=False),
            "le_pair_nt": le_pair_nt.astype(np.float32, copy=False),
        }

        del ckl, le
        gc.collect()

    write_csv(
        args.output_dir / "base_and_pairwise_auc_summary.csv",
        auc_rows,
    )

    # ============================================================
    # Select ONE common CKL tau across all K and all 3 detectors.
    # ============================================================
    tau_selection_rows = []

    for tau in args.ckl_taus:
        combo_rows = []
        eligible = True

        for K in args.K_values:
            dK = per_k[int(K)]
            z_tn = dK["z_tn"]
            eps = dK["epochs_analysis"]
            m = dK["m"]

            # min-run
            hit = np.asarray(z_tn >= float(tau), dtype=bool)
            pred = minrun_from_hits(hit, m)
            rows = trajectory_rows(
                "CKL",
                K,
                "min_run",
                "fixed_z_common_tau",
                {
                    "tau": float(tau),
                    "m": int(m),
                },
                pred,
                y,
                eps,
            )
            best = best_over_epoch(
                rows,
                args.target_fpr,
            )
            if best is None:
                eligible = False
            else:
                combo_rows.append(best)

            # sliding
            pred = sliding_from_hits(
                hit,
                args.ckl_sliding_ell,
                args.ckl_sliding_k,
            )
            rows = trajectory_rows(
                "CKL",
                K,
                "sliding_window",
                "fixed_z_common_tau",
                {
                    "tau": float(tau),
                    "ell": int(args.ckl_sliding_ell),
                    "k": int(args.ckl_sliding_k),
                },
                pred,
                y,
                eps,
            )
            best = best_over_epoch(
                rows,
                args.target_fpr,
            )
            if best is None:
                eligible = False
            else:
                combo_rows.append(best)

            # EWMA with SAME tau
            ewma_score_tn = ewma_continuous(
                z_tn,
                args.ckl_ewma_lambda,
            )
            pred = sticky_threshold(
                ewma_score_tn,
                tau,
            )
            rows = trajectory_rows(
                "CKL",
                K,
                "ewma",
                "fixed_z_common_tau",
                {
                    "tau": float(tau),
                    "lambda": float(args.ckl_ewma_lambda),
                },
                pred,
                y,
                eps,
            )
            best = best_over_epoch(
                rows,
                args.target_fpr,
            )
            if best is None:
                eligible = False
            else:
                combo_rows.append(best)

        if eligible and len(combo_rows) == 9:
            tprs = np.asarray(
                [r["TPR"] for r in combo_rows],
                dtype=np.float64,
            )
            fprs = np.asarray(
                [r["FPR"] for r in combo_rows],
                dtype=np.float64,
            )

            tau_selection_rows.append({
                "tau": float(tau),
                "eligible_all_9": True,
                "mean_TPR": float(np.mean(tprs)),
                "min_TPR": float(np.min(tprs)),
                "mean_FPR": float(np.mean(fprs)),
            })
        else:
            tau_selection_rows.append({
                "tau": float(tau),
                "eligible_all_9": False,
                "mean_TPR": np.nan,
                "min_TPR": np.nan,
                "mean_FPR": np.nan,
            })

    eligible = [
        r for r in tau_selection_rows
        if r["eligible_all_9"]
    ]

    if not eligible:
        raise RuntimeError(
            "No common CKL tau was feasible for all 3 detectors "
            "at all K values under the requested FPR budget."
        )

    eligible.sort(
        key=lambda r: (
            -float(r["mean_TPR"]),
            -float(r["min_TPR"]),
            float(r["mean_FPR"]),
            float(r["tau"]),
        )
    )

    chosen_tau = float(
        eligible[0]["tau"]
    )

    print()
    print(
        f"Chosen common CKL tau = {chosen_tau}"
    )

    write_csv(
        args.output_dir
        / "ckl_common_tau_selection.csv",
        tau_selection_rows,
    )

    # ============================================================
    # Build final revised trajectories for chosen tau.
    # ============================================================
    all_rows = []

    for K in args.K_values:
        dK = per_k[int(K)]
        z_tn = dK["z_tn"]
        pct_nt = dK["pct_nt"]
        eps = dK["epochs_analysis"]
        m = dK["m"]

        # -------------------------
        # CKL fixed-z common tau
        # -------------------------
        hit = np.asarray(
            z_tn >= chosen_tau,
            dtype=bool,
        )

        ckl_min = minrun_from_hits(
            hit,
            m,
        )

        all_rows.extend(
            trajectory_rows(
                "CKL",
                K,
                "min_run",
                "fixed_z_common_tau",
                {
                    "tau": chosen_tau,
                    "m": int(m),
                    "alpha": float(args.alpha),
                    "delta": float(args.delta),
                },
                ckl_min,
                y,
                eps,
            )
        )

        ckl_slide = sliding_from_hits(
            hit,
            args.ckl_sliding_ell,
            args.ckl_sliding_k,
        )

        all_rows.extend(
            trajectory_rows(
                "CKL",
                K,
                "sliding_window",
                "fixed_z_common_tau",
                {
                    "tau": chosen_tau,
                    "ell": int(args.ckl_sliding_ell),
                    "k": int(args.ckl_sliding_k),
                },
                ckl_slide,
                y,
                eps,
            )
        )

        ckl_ewma_score_tn = ewma_continuous(
            z_tn,
            args.ckl_ewma_lambda,
        )

        ckl_ewma = sticky_threshold(
            ckl_ewma_score_tn,
            chosen_tau,
        )

        all_rows.extend(
            trajectory_rows(
                "CKL",
                K,
                "ewma",
                "fixed_z_common_tau",
                {
                    "tau": chosen_tau,
                    "lambda": float(args.ckl_ewma_lambda),
                },
                ckl_ewma,
                y,
                eps,
            )
        )

        ckl_pair_pred = current_topq_trajectory(
            dK["ckl_pair_nt"],
            args.q,
        )

        all_rows.extend(
            trajectory_rows(
                "CKL",
                K,
                "cumulative_pairwise",
                "fixed_top_q",
                {
                    "q": float(args.q),
                },
                ckl_pair_pred,
                y,
                eps,
            )
        )

        # -------------------------
        # LE rank-dynamic
        # -------------------------
        le_hit_tn = np.asarray(
            pct_nt >= (1.0 - float(args.q)),
            dtype=bool,
        ).T

        le_min = minrun_from_hits(
            le_hit_tn,
            m,
        )

        all_rows.extend(
            trajectory_rows(
                "LE_GIE",
                K,
                "min_run",
                "rank_dynamic",
                {
                    "q": float(args.q),
                    "m": int(m),
                    "alpha": float(args.alpha),
                    "delta": float(args.delta),
                },
                le_min,
                y,
                eps,
            )
        )

        le_slide = sliding_from_hits(
            le_hit_tn,
            args.le_sliding_ell,
            args.le_sliding_k,
        )

        all_rows.extend(
            trajectory_rows(
                "LE_GIE",
                K,
                "sliding_window",
                "rank_dynamic",
                {
                    "q": float(args.q),
                    "ell": int(args.le_sliding_ell),
                    "k": int(args.le_sliding_k),
                },
                le_slide,
                y,
                eps,
            )
        )

        le_rank_ewma_nt = rank_ewma(
            pct_nt,
            args.le_rank_ewma_lambda,
        )

        le_ewma_pred = current_topq_trajectory(
            le_rank_ewma_nt,
            args.q,
        )

        all_rows.extend(
            trajectory_rows(
                "LE_GIE",
                K,
                "ewma",
                "rank_dynamic",
                {
                    "q": float(args.q),
                    "lambda": float(args.le_rank_ewma_lambda),
                },
                le_ewma_pred,
                y,
                eps,
            )
        )

        le_pair_pred = current_topq_trajectory(
            dK["le_pair_nt"],
            args.q,
        )

        all_rows.extend(
            trajectory_rows(
                "LE_GIE",
                K,
                "cumulative_pairwise",
                "fixed_top_q",
                {
                    "q": float(args.q),
                },
                le_pair_pred,
                y,
                eps,
            )
        )

    write_csv(
        args.output_dir
        / "all_detector_trajectories.csv",
        all_rows,
    )

    # ============================================================
    # Two summaries:
    # A) fixed-parameter best TPR over epoch (FPR evaluation-only)
    # B) benchmark best under FPR <= 0.05
    # ============================================================
    deployment_rows = []
    benchmark_rows = []

    for method in ("CKL", "LE_GIE"):
        for K in args.K_values:
            for detector in (
                "min_run",
                "sliding_window",
                "ewma",
                "cumulative_pairwise",
            ):
                group = [
                    r for r in all_rows
                    if (
                        r["method"] == method
                        and int(r["K"]) == int(K)
                        and r["detector"] == detector
                    )
                ]

                best_dep = best_over_epoch(
                    group,
                    target_fpr=None,
                )
                if best_dep is not None:
                    deployment_rows.append(
                        best_dep
                    )

                best_fpr = best_over_epoch(
                    group,
                    target_fpr=args.target_fpr,
                )
                if best_fpr is not None:
                    benchmark_rows.append(
                        best_fpr
                    )

    write_csv(
        args.output_dir
        / "fixed_parameter_best_over_epoch.csv",
        deployment_rows,
    )

    write_csv(
        args.output_dir
        / "benchmark_best_under_fpr.csv",
        benchmark_rows,
    )

    # ============================================================
    # PRIMARY plots:
    # fixed detector parameters, no FPR filtering.
    # FPR is shown as an evaluation outcome.
    # ============================================================
    detectors = (
        "min_run",
        "sliding_window",
        "ewma",
        "cumulative_pairwise",
    )

    for detector in detectors:
        # TPR vs K
        fig, ax = plt.subplots(figsize=(7.5, 5))

        for method in ("CKL", "LE_GIE"):
            rr = sorted(
                [
                    r for r in deployment_rows
                    if (
                        r["method"] == method
                        and r["detector"] == detector
                    )
                ],
                key=lambda r: int(r["K"]),
            )

            ax.plot(
                [int(r["K"]) for r in rr],
                [float(r["TPR"]) for r in rr],
                marker="o",
                label=method,
            )

        ax.set_xlabel("K")
        ax.set_ylabel("Best TPR with frozen detector parameters")
        ax.set_title(
            f"Primary fixed-parameter comparison: {detector}"
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            args.output_dir
            / f"fig_primary_{detector}_tpr_vs_K.png",
            dpi=180,
        )
        plt.close(fig)

        # FPR vs K
        fig, ax = plt.subplots(figsize=(7.5, 5))

        for method in ("CKL", "LE_GIE"):
            rr = sorted(
                [
                    r for r in deployment_rows
                    if (
                        r["method"] == method
                        and r["detector"] == detector
                    )
                ],
                key=lambda r: int(r["K"]),
            )

            ax.plot(
                [int(r["K"]) for r in rr],
                [float(r["FPR"]) for r in rr],
                marker="o",
                label=method,
            )

        ax.axhline(
            args.target_fpr,
            linestyle="--",
            linewidth=1,
            label=f"reference FPR={args.target_fpr}",
        )
        ax.set_xlabel("K")
        ax.set_ylabel("Evaluation FPR")
        ax.set_title(
            f"Primary fixed-parameter FPR: {detector}"
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            args.output_dir
            / f"fig_primary_{detector}_fpr_vs_K.png",
            dpi=180,
        )
        plt.close(fig)

    # One compact primary summary at K=40.
    k40_rows = [
        r for r in deployment_rows
        if int(r["K"]) == 40
    ]

    x = np.arange(len(detectors), dtype=float)
    width = 0.36

    ckl40 = {
        r["detector"]: r
        for r in k40_rows
        if r["method"] == "CKL"
    }
    le40 = {
        r["detector"]: r
        for r in k40_rows
        if r["method"] == "LE_GIE"
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        [ckl40[d]["TPR"] for d in detectors],
        width,
        label="CKL",
    )
    ax.bar(
        x + width / 2,
        [le40[d]["TPR"] for d in detectors],
        width,
        label="LE-GIE",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("TPR")
    ax.set_title("Primary frozen detector comparison at K=40")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "fig_primary_all_detectors_K40_tpr.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        [ckl40[d]["FPR"] for d in detectors],
        width,
        label="CKL",
    )
    ax.bar(
        x + width / 2,
        [le40[d]["FPR"] for d in detectors],
        width,
        label="LE-GIE",
    )
    ax.axhline(
        args.target_fpr,
        linestyle="--",
        linewidth=1,
        label=f"reference FPR={args.target_fpr}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=15)
    ax.set_ylabel("Evaluation FPR")
    ax.set_title("Primary frozen detector FPR at K=40")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "fig_primary_all_detectors_K40_fpr.png",
        dpi=180,
    )
    plt.close(fig)

    # ============================================================
    # SECONDARY benchmark plots:
    # only rows feasible under FPR <= target.
    # ============================================================
    for detector in detectors:
        fig, ax = plt.subplots(figsize=(7.5, 5))

        any_line = False
        for method in ("CKL", "LE_GIE"):
            rr = sorted(
                [
                    r for r in benchmark_rows
                    if (
                        r["method"] == method
                        and r["detector"] == detector
                    )
                ],
                key=lambda r: int(r["K"]),
            )

            if not rr:
                continue

            any_line = True
            ax.plot(
                [int(r["K"]) for r in rr],
                [float(r["TPR"]) for r in rr],
                marker="o",
                label=method,
            )

        if any_line:
            ax.set_xlabel("K")
            ax.set_ylabel(
                f"Best TPR under FPR <= {args.target_fpr}"
            )
            ax.set_title(
                f"Secondary low-FPR benchmark: {detector}"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                args.output_dir
                / f"fig_secondary_{detector}_tpr_vs_K.png",
                dpi=180,
            )
        plt.close(fig)

    # AUC comparison across K.
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("CKL", "LE_GIE"):
        rr = sorted([r for r in auc_rows if r["method"] == method], key=lambda r: int(r["K"]))
        ax.plot(
            [int(r["K"]) for r in rr],
            [float(r["cumulative_pairwise_best_auc"]) for r in rr],
            marker="o", label=method,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("Best cumulative pairwise ROC-AUC")
    ax.set_title("CKL vs LE-GIE cumulative pairwise AUC across K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_auc_cumulative_pairwise_vs_K.png", dpi=180)
    plt.close(fig)

    config = {
        "artifact":
            "revised_CKL_vs_LE_all_detectors_K_sweep",
        "K_values": [
            int(k) for k in args.K_values
        ],
        "minrun_formula":
            "ceil(log(T_monitor/delta)/(-log(alpha)))",
        "alpha": float(args.alpha),
        "delta": float(args.delta),
        "m_by_K": {
            str(K): int(per_k[int(K)]["m"])
            for K in args.K_values
        },
        "CKL": {
            "limit_rule": "last3 mean",
            "common_tau_candidates": [
                float(v) for v in args.ckl_taus
            ],
            "chosen_common_tau": chosen_tau,
            "tau_scope":
                "same tau for CKL min-run, sliding-window, EWMA, and for all tested K values",
            "sliding_ell":
                int(args.ckl_sliding_ell),
            "sliding_k":
                int(args.ckl_sliding_k),
            "ewma_lambda":
                float(args.ckl_ewma_lambda),
            "pairwise_q":
                float(args.q),
        },
        "LE_GIE": {
            "error_limit_rule": "next",
            "GIE_limit_rule": "next",
            "rank_q": float(args.q),
            "minrun_m_by_K": {
                str(K): int(per_k[int(K)]["m"])
                for K in args.K_values
            },
            "sliding_ell":
                int(args.le_sliding_ell),
            "sliding_k":
                int(args.le_sliding_k),
            "rank_ewma_lambda":
                float(args.le_rank_ewma_lambda),
            "pairwise_q":
                float(args.q),
        },
        "primary_summary":
            "fixed detector parameters; best TPR over epoch; no FPR filtering; FPR reported only as benchmark outcome",
        "secondary_summary":
            "best TPR over epoch subject to FPR <= target_fpr_benchmark",
        "auc_output":
            "base_and_pairwise_auc_summary.csv contains raw, class-z, and cumulative-pairwise best ROC-AUC for each estimator and K",
        "target_fpr_benchmark":
            float(args.target_fpr),
        "selection_warning":
            "common tau and benchmark-best epochs use labeled seed-66 data; freeze chosen settings before independent validation",
        "production_modules_modified":
            False,
    }

    with (
        args.output_dir
        / "revised_ckl_le_all_detectors_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("FINAL REVISED SUMMARY")
    print("=" * 72)
    print(
        f"Common CKL tau = {chosen_tau}"
    )
    print(
        "m by K = "
        + ", ".join(
            f"{K}:{per_k[int(K)]['m']}"
            for K in args.K_values
        )
    )
    print()

    print("PRIMARY fixed-parameter results (no FPR filtering):")
    for row in deployment_rows:
        print(
            f"{row['method']:>6s} | "
            f"K={int(row['K']):>2d} | "
            f"{row['detector']:<20s} | "
            f"TPR={row['TPR']:.4f} "
            f"FPR={row['FPR']:.4f} "
            f"epoch={row['epoch']}"
        )

    print()
    print(
        f"SECONDARY benchmark best under FPR <= "
        f"{args.target_fpr:.3f}:"
    )

    for row in benchmark_rows:
        print(
            f"{row['method']:>6s} | "
            f"K={int(row['K']):>2d} | "
            f"{row['detector']:<20s} | "
            f"TPR={row['TPR']:.4f} "
            f"FPR={row['FPR']:.4f} "
            f"epoch={row['epoch']}"
        )

    print()
    print("AUC summary:")
    for row in auc_rows:
        print(
            f"{row['method']:>6s} | K={int(row['K']):>2d} | "
            f"raw={row['raw_best_auc']:.6f} "
            f"z={row['class_z_best_auc']:.6f} "
            f"pair={row['cumulative_pairwise_best_auc']:.6f}"
        )

    print()
    print(
        f"Outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
