#!/usr/bin/env python
"""
Corrected LE-GIE finite-sample sensitivity study.

IMPORTANT CORRECTION
--------------------
The limit estimate L_hat is a choice of the WHOLE LE estimator.  Therefore the
same sample limit L_hat_i(t) is used in BOTH:

    1) the error-decay component ell_err_hat, and
    2) the sample-side residuals inside m_GIE_hat.

The class-reference trajectory G_c(t) has its own corresponding limit
L_hat_G,c(t), computed with the SAME limit-estimation rule.

The theoretical estimator is unchanged:

    ell_hat = ell_err_hat + log(abs(m_GIE_hat)).

At output column t (0-based), the latest observation x_t is available.
The common estimation tail is

    x_{t-K}, ..., x_{t-1}            (K points)

and the boundary used by ell_err is

    x_{t-K-1}.

The same information horizon is used for every tested limit rule:

    last3_mean:
        L_hat_i(t) = mean(x_{i,t-2}, x_{i,t-1}, x_{i,t})

    next:
        L_hat_i(t) = x_{i,t}

    aitken_guarded:
        guarded Aitken Delta^2 from x_{i,t-2}, x_{i,t-1}, x_{i,t},
        with fallback to x_{i,t}.

For the GIE class-reference trajectory G_c(t), the corresponding
L_hat_G,c(t) is computed by the same rule.

This script is deliberately experiment-local.  It does NOT modify
proposed_le.py, estimators.py, detectors.py, or any earlier experiment.

Stages
------
A. K sensitivity:
       K in {10,15,20,30,40}
       class reference = observed-class mean
       common limit rule = last3_mean

B. Class-reference sensitivity:
       K = 20
       common limit rule = last3_mean
       references:
           mean
           leave_one_out_mean
           median
           5% trimmed mean
           10% trimmed mean

C. Whole-estimator limit sensitivity:
       K = 20
       class reference = observed-class mean
       common limit rule:
           last3_mean
           next
           aitken_guarded

For each variant we report:
    - raw LE best ROC-AUC
    - class-z LE best ROC-AUC
    - ell_err best ROC-AUC
    - log|m_GIE| best ROC-AUC
    - exact cumulative-pairwise best ROC-AUC
    - top-q TPR/FPR at the pairwise best-AUC epoch
    - evaluation-only max TPR under FPR <= target
    - temporal LE stability:
          median |ell_hat(t)-ell_hat(t-1)|
          95th percentile |ell_hat(t)-ell_hat(t-1)|

The oracle FPR-budget result uses the known noisy-label mask and is
evaluation-only.  Any selected configuration must be validated on an
independent training run before final claims.
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


# ============================================================
# CLI
# ============================================================

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
        default=Path(
            "results/le_improvement_consistent_limit"
        ),
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


# ============================================================
# Class-reference construction
# ============================================================

def build_class_reference_trajectory(
    loss_traj,
    observed_label,
    *,
    num_classes=10,
    reference_method="mean",
    trim_fraction=0.10,
):
    """Build G_i(t) using only the observed class of sample i.

    This implementation is experiment-local so no production estimator
    module needs to be changed.
    """
    x = np.asarray(loss_traj, dtype=np.float64)
    labels = np.asarray(observed_label, dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(
            f"loss_traj must have shape [N,T], got {x.shape}."
        )

    N, T = x.shape

    if labels.shape != (N,):
        raise ValueError(
            f"observed_label must have shape ({N},), got {labels.shape}."
        )

    method = str(reference_method).lower()

    allowed = {
        "mean",
        "leave_one_out_mean",
        "median",
        "trimmed_mean",
    }

    if method not in allowed:
        raise ValueError(
            f"Unknown reference_method={reference_method!r}."
        )

    trim_fraction = float(trim_fraction)

    if method == "trimmed_mean":
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError(
                "trim_fraction must lie in [0,0.5)."
            )

    G = np.full((N, T), np.nan, dtype=np.float64)

    for c in range(int(num_classes)):
        members = np.flatnonzero(labels == c)

        if members.size == 0:
            continue

        sub = x[members]

        if method == "mean":
            ref = np.nanmean(sub, axis=0)
            G[members] = ref[None, :]
            continue

        if method == "median":
            ref = np.nanmedian(sub, axis=0)
            G[members] = ref[None, :]
            continue

        if method == "leave_one_out_mean":
            finite = np.isfinite(sub)
            sums = np.nansum(sub, axis=0)
            counts = finite.sum(axis=0)

            numer = (
                sums[None, :]
                - np.where(finite, sub, 0.0)
            )

            denom = (
                counts[None, :]
                - finite.astype(np.int64)
            )

            ref = np.full_like(
                sub,
                np.nan,
                dtype=np.float64,
            )

            np.divide(
                numer,
                denom,
                out=ref,
                where=(denom > 0),
            )

            G[members] = ref
            continue

        # Symmetric trimmed mean, independently at each epoch.
        ref = np.full(T, np.nan, dtype=np.float64)

        for t in range(T):
            vals = sub[:, t]
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            vals = np.sort(vals)
            n_trim = int(
                np.floor(
                    trim_fraction * vals.size
                )
            )

            if (
                n_trim > 0
                and 2 * n_trim < vals.size
            ):
                vals = vals[
                    n_trim:
                    vals.size - n_trim
                ]

            ref[t] = float(
                np.mean(vals)
            )

        G[members] = ref[None, :]

    return G


# ============================================================
# Common limit estimators
# ============================================================

def guarded_aitken_batch(a, b, c):
    """Guarded Aitken Delta^2 with fallback to latest value c."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    d0 = b - a
    d1 = c - b
    denominator = d1 - d0

    scale = np.maximum(
        np.maximum(
            np.abs(d0),
            np.abs(d1),
        ),
        np.finfo(np.float64).tiny,
    )

    tol = (
        np.sqrt(np.finfo(np.float64).eps)
        * scale
    )

    use_aitken = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(c)
        & np.isfinite(denominator)
        & (np.abs(denominator) > tol)
    )

    L = np.full_like(
        a,
        np.nan,
        dtype=np.float64,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        L[use_aitken] = (
            a[use_aitken]
            - (
                d0[use_aitken] ** 2
                / denominator[use_aitken]
            )
        )

    fallback = (
        ~use_aitken
        | ~np.isfinite(L)
    )

    L[fallback] = c[fallback]

    return L, fallback


def estimate_common_limit_at_t(
    traj,
    t,
    method,
):
    """Estimate the row-wise limit using observations through column t."""
    traj = np.asarray(
        traj,
        dtype=np.float64,
    )

    method = str(method).lower()

    if t < 2:
        raise ValueError(
            "Need at least three observations for common limit estimation."
        )

    if method == "last3_mean":
        return (
            np.mean(
                traj[:, t - 2:t + 1],
                axis=1,
            ),
            np.zeros(
                traj.shape[0],
                dtype=bool,
            ),
        )

    if method == "next":
        return (
            traj[:, t].copy(),
            np.zeros(
                traj.shape[0],
                dtype=bool,
            ),
        )

    if method == "aitken_guarded":
        return guarded_aitken_batch(
            traj[:, t - 2],
            traj[:, t - 1],
            traj[:, t],
        )

    raise ValueError(
        "limit_method must be one of "
        "{'last3_mean','next','aitken_guarded'}."
    )


# ============================================================
# GIE with externally supplied limits
# ============================================================

def vectorized_gie_window_with_limits(
    phi,
    G,
    phi_limit,
    G_limit,
):
    """Row-wise GIE using supplied common estimator limits.

    This keeps the existing GIE/Hill formula.  Only the limit values are
    supplied explicitly so the same L_hat used by ell_err is also used by
    the sample side of GIE.
    """
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    phi_limit = np.asarray(
        phi_limit,
        dtype=np.float64,
    )
    G_limit = np.asarray(
        G_limit,
        dtype=np.float64,
    )

    if phi.shape != G.shape or phi.ndim != 2:
        raise ValueError(
            "phi and G must have matching shape [N,K]."
        )

    N, _ = phi.shape

    if phi_limit.shape != (N,):
        raise ValueError(
            f"phi_limit must have shape ({N},)."
        )

    if G_limit.shape != (N,):
        raise ValueError(
            f"G_limit must have shape ({N},)."
        )

    R = np.abs(
        phi - phi_limit[:, None]
    )

    FR = np.abs(
        G - G_limit[:, None]
    )

    with np.errstate(invalid="ignore"):
        w0 = np.max(R, axis=1)
        w1 = np.max(FR, axis=1)

    d = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )

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

    if not np.any(base_ok):
        return d

    safe_w0 = np.where(
        base_ok,
        w0 + EPS,
        1.0,
    )

    safe_w1 = np.where(
        base_ok,
        w1 + EPS,
        1.0,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        log_num = np.where(
            mask,
            np.log(
                np.abs(
                    R / safe_w0[:, None]
                )
            ),
            0.0,
        )

        log_den = np.where(
            mask,
            np.log(
                np.abs(
                    FR / safe_w1[:, None]
                )
            ),
            0.0,
        )

    denom_num = np.sum(
        log_num,
        axis=1,
    )

    denom_den = np.sum(
        log_den,
        axis=1,
    )

    ok = (
        base_ok
        & np.isfinite(denom_num)
        & np.isfinite(denom_den)
        & (np.abs(denom_num) >= EPS)
        & (np.abs(denom_den) >= EPS)
    )

    hill_num = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )

    hill_den = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )

    hill_num[ok] = (
        -k_internal[ok]
        / denom_num[ok]
    )

    hill_den[ok] = (
        -k_internal[ok]
        / denom_den[ok]
    )

    good = (
        ok
        & np.isfinite(hill_num)
        & np.isfinite(hill_den)
        & (np.abs(hill_den) > EPS)
    )

    d[good] = (
        hill_num[good]
        / hill_den[good]
    )

    return d


# ============================================================
# Whole LE estimator with ONE common L_hat rule
# ============================================================

def rolling_common_limit_le_gie_batch(
    loss_traj,
    observed_label,
    *,
    K,
    num_classes=10,
    reference_method="mean",
    trim_fraction=0.10,
    limit_method="last3_mean",
):
    """Rolling LE-GIE with one consistent limit rule.

    At output column t:

        boundary:
            x_{t-K-1}

        K-point common tail:
            x_{t-K}, ..., x_{t-1}

        latest available observation for limit estimation:
            x_t

    The SAME sample L_hat_i(t) is used for:
        - ell_err residuals
        - sample-side GIE residuals

    The class-reference limit uses the SAME rule on G_i(t).
    """
    x = np.asarray(
        loss_traj,
        dtype=np.float64,
    )

    labels = np.asarray(
        observed_label,
        dtype=np.int64,
    )

    if x.ndim != 2:
        raise ValueError(
            f"loss_traj must be [N,T], got {x.shape}."
        )

    N, T = x.shape

    if labels.shape != (N,):
        raise ValueError(
            f"observed_label must have shape ({N},)."
        )

    K = int(K)

    if K < 6:
        raise ValueError(
            "K must be at least 6."
        )

    if T < K + 2:
        raise ValueError(
            f"Need at least K+2={K+2} epochs; got {T}."
        )

    G = build_class_reference_trajectory(
        x,
        labels,
        num_classes=num_classes,
        reference_method=reference_method,
        trim_fraction=trim_fraction,
    )

    ell_err_traj = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    id_gie_traj = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    lambda_traj = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    sample_limit_traj = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    reference_limit_traj = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    valid_traj = np.zeros(
        (N, T),
        dtype=bool,
    )

    limit_fallback_traj = np.zeros(
        (N, T),
        dtype=bool,
    )

    inv_j = (
        1.0
        / np.arange(
            1,
            K + 1,
            dtype=np.float64,
        )
    )

    # t is the latest available observation used by L_hat.
    # n = t-1 is the end of the K-point estimation tail.
    first_col = K + 1

    for t in range(first_col, T):

        n = t - 1
        boundary = t - K - 1
        tail_start = t - K

        sample_limit, fallback_sample = (
            estimate_common_limit_at_t(
                x,
                t,
                limit_method,
            )
        )

        reference_limit, fallback_reference = (
            estimate_common_limit_at_t(
                G,
                t,
                limit_method,
            )
        )

        sample_limit_traj[:, t] = sample_limit
        reference_limit_traj[:, t] = reference_limit

        limit_fallback_traj[:, t] = (
            fallback_sample
            | fallback_reference
        )

        # ----------------------------------------------------
        # 1) Error-decay component with THE SAME sample L_hat.
        # ----------------------------------------------------

        r0 = np.abs(
            x[:, boundary]
            - sample_limit
        )

        # K residuals corresponding to j=1,...,K:
        # x_{t-K}, ..., x_{t-1}
        tail = x[:, tail_start:t]

        rj = np.abs(
            tail
            - sample_limit[:, None]
        )

        valid_ell = (
            np.isfinite(sample_limit)
            & np.isfinite(r0)
            & (r0 != 0.0)
            & np.all(
                np.isfinite(rj)
                & (rj != 0.0),
                axis=1,
            )
        )

        ell_err = np.full(
            N,
            np.nan,
            dtype=np.float64,
        )

        if np.any(valid_ell):
            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
            ):
                ell_values = (
                    np.log(
                        rj[valid_ell]
                        / r0[valid_ell, None]
                    )
                    * inv_j[None, :]
                )

            ell_err[valid_ell] = np.mean(
                ell_values,
                axis=1,
            )

        # ----------------------------------------------------
        # 2) GIE using the SAME sample L_hat and same tail.
        # ----------------------------------------------------

        G_tail = G[:, tail_start:t]

        id_gie = (
            vectorized_gie_window_with_limits(
                tail,
                G_tail,
                sample_limit,
                reference_limit,
            )
        )

        # ----------------------------------------------------
        # 3) Final theoretical LE, unchanged.
        # ----------------------------------------------------

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            lambda_hat = (
                ell_err
                + np.log(
                    np.abs(id_gie)
                )
            )

        valid = (
            valid_ell
            & np.isfinite(id_gie)
            & (id_gie != 0.0)
            & np.isfinite(lambda_hat)
        )

        ell_err_traj[valid, t] = (
            ell_err[valid]
        )

        id_gie_traj[valid, t] = (
            id_gie[valid]
        )

        lambda_traj[valid, t] = (
            lambda_hat[valid]
        )

        valid_traj[valid, t] = True

    return {
        "lambda_traj": lambda_traj,
        "ell_err_traj": ell_err_traj,
        "id_gie_traj": id_gie_traj,
        "sample_limit_traj": sample_limit_traj,
        "reference_limit_traj": reference_limit_traj,
        "valid_traj": valid_traj,
        "limit_fallback_traj": limit_fallback_traj,
        "first_available_column": int(first_col),
        "first_available_epoch": int(first_col + 1),
        "K": int(K),
        "reference_method": str(reference_method),
        "trim_fraction": float(trim_fraction),
        "limit_method": str(limit_method),
        "window_definition": (
            "boundary=x[t-K-1], "
            "tail=x[t-K:t], "
            "latest_for_limit=x[t]"
        ),
    }


# ============================================================
# Evaluation helpers
# ============================================================

def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(
        score,
        dtype=np.float64,
    )

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
    score_nt = np.asarray(
        score_nt,
        dtype=np.float64,
    )

    out = np.full(
        score_nt.shape[1],
        np.nan,
        dtype=np.float64,
    )

    for t in range(score_nt.shape[1]):
        out[t] = auc_binary(
            y,
            score_nt[:, t],
        )

    return out


def best_auc(auc, epochs):
    finite = np.flatnonzero(
        np.isfinite(auc)
    )

    if finite.size == 0:
        return np.nan, -1, -1

    t = int(
        finite[
            np.nanargmax(
                auc[finite]
            )
        ]
    )

    return (
        float(auc[t]),
        int(epochs[t]),
        t,
    )


def topq_metrics(y, score, q):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(
        score,
        dtype=np.float64,
    )

    N = y.size
    finite_idx = np.flatnonzero(
        np.isfinite(score)
    )

    k = min(
        max(
            1,
            int(round(float(q) * N)),
        ),
        finite_idx.size,
    )

    selected = np.zeros(
        N,
        dtype=bool,
    )

    if k:
        fs = score[finite_idx]
        local = np.argpartition(
            fs,
            -k,
        )[-k:]

        selected[
            finite_idx[local]
        ] = True

    TP = int(
        np.sum(selected & y)
    )

    FP = int(
        np.sum(selected & ~y)
    )

    FN = int(
        np.sum(~selected & y)
    )

    TN = int(
        np.sum(~selected & ~y)
    )

    TPR = (
        TP / (TP + FN)
        if TP + FN
        else np.nan
    )

    FPR = (
        FP / (FP + TN)
        if FP + TN
        else np.nan
    )

    precision = (
        TP / (TP + FP)
        if TP + FP
        else np.nan
    )

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


def oracle_max_tpr_under_fpr(
    y,
    score,
    target_fpr,
):
    """Evaluation-only ROC operating point."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(
        score,
        dtype=np.float64,
    )

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

    order = np.argsort(
        -ss,
        kind="mergesort",
    )

    yy = yy[order]
    ss = ss[order]

    tp = np.cumsum(
        yy.astype(np.int64)
    )

    fp = np.cumsum(
        (~yy).astype(np.int64)
    )

    tpr = tp / n_pos
    fpr = fp / n_neg

    feasible = np.flatnonzero(
        fpr <= float(target_fpr)
    )

    if feasible.size == 0:
        return {
            "oracle_TPR": 0.0,
            "oracle_FPR": 0.0,
            "oracle_threshold": np.inf,
            "oracle_selected_fraction": 0.0,
        }

    best_tpr = np.max(
        tpr[feasible]
    )

    candidates = feasible[
        np.isclose(
            tpr[feasible],
            best_tpr,
        )
    ]

    best = int(
        candidates[
            np.argmin(
                fpr[candidates]
            )
        ]
    )

    return {
        "oracle_TPR": float(tpr[best]),
        "oracle_FPR": float(fpr[best]),
        "oracle_threshold": float(ss[best]),
        "oracle_selected_fraction": float(
            (best + 1) / y.size
        ),
    }


def temporal_stability(score_nt):
    """Summarize adjacent-epoch absolute changes of a score trajectory."""
    score = np.asarray(
        score_nt,
        dtype=np.float64,
    )

    if score.shape[1] < 2:
        return {
            "median_abs_step": np.nan,
            "p95_abs_step": np.nan,
            "n_valid_steps": 0,
        }

    left = score[:, :-1]
    right = score[:, 1:]

    valid = (
        np.isfinite(left)
        & np.isfinite(right)
    )

    if not np.any(valid):
        return {
            "median_abs_step": np.nan,
            "p95_abs_step": np.nan,
            "n_valid_steps": 0,
        }

    steps = np.abs(
        right[valid]
        - left[valid]
    )

    return {
        "median_abs_step": float(
            np.median(steps)
        ),
        "p95_abs_step": float(
            np.quantile(
                steps,
                0.95,
            )
        ),
        "n_valid_steps": int(
            steps.size
        ),
    }


def write_csv(path, rows):
    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# One variant
# ============================================================

def evaluate_variant(
    *,
    stage,
    variant,
    K,
    reference_method,
    trim_fraction,
    limit_method,
    result,
    labels,
    y,
    epochs,
    top_fractions,
    target_fpr,
    num_classes,
):
    le_nt = np.asarray(
        result["lambda_traj"],
        dtype=np.float64,
    )

    ell_err_nt = np.asarray(
        result["ell_err_traj"],
        dtype=np.float64,
    )

    id_gie_nt = np.asarray(
        result["id_gie_traj"],
        dtype=np.float64,
    )

    start_col = int(
        result["first_available_column"]
    )

    z_tn = standardize_monitoring_score(
        le_nt.T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=start_col,
    )

    z_nt = z_tn.T

    log_m_nt = np.full_like(
        id_gie_nt,
        np.nan,
        dtype=np.float64,
    )

    valid_m = (
        np.isfinite(id_gie_nt)
        & (id_gie_nt != 0.0)
    )

    log_m_nt[valid_m] = np.log(
        np.abs(
            id_gie_nt[valid_m]
        )
    )

    auc_raw = auc_curve(
        y,
        le_nt,
    )

    auc_z = auc_curve(
        y,
        z_nt,
    )

    auc_err = auc_curve(
        y,
        ell_err_nt,
    )

    auc_log_m = auc_curve(
        y,
        log_m_nt,
    )

    raw_best, raw_ep, _ = (
        best_auc(
            auc_raw,
            epochs,
        )
    )

    z_best, z_ep, _ = (
        best_auc(
            auc_z,
            epochs,
        )
    )

    err_best, err_ep, _ = (
        best_auc(
            auc_err,
            epochs,
        )
    )

    m_best, m_ep, _ = (
        best_auc(
            auc_log_m,
            epochs,
        )
    )

    pair = exact_pairwise_scores_by_class(
        le_nt,
        labels,
        start_index=start_col,
    )

    pair_cum_nt = np.asarray(
        pair["cumulative_score"],
        dtype=np.float64,
    )

    auc_pair = auc_curve(
        y,
        pair_cum_nt,
    )

    pair_best, pair_ep, pair_t = (
        best_auc(
            auc_pair,
            epochs,
        )
    )

    score_best = (
        pair_cum_nt[:, pair_t]
    )

    oracle = (
        oracle_max_tpr_under_fpr(
            y,
            score_best,
            target_fpr,
        )
    )

    le_stability = temporal_stability(
        le_nt[:, start_col:]
    )

    limit_stability = temporal_stability(
        result["sample_limit_traj"][
            :,
            start_col:,
        ]
    )

    fallback_fraction = float(
        np.mean(
            result["limit_fallback_traj"][
                :,
                start_col:,
            ]
        )
    )

    summary = {
        "stage": stage,
        "variant": variant,
        "K": int(K),
        "reference_method": str(reference_method),
        "trim_fraction": float(trim_fraction),
        "common_limit_method": str(limit_method),
        "theoretical_formula":
            "ell_err + log(abs(m_GIE))",
        "common_limit_enforced": True,
        "first_available_epoch": int(
            result["first_available_epoch"]
        ),
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
        "le_median_abs_epoch_step":
            le_stability["median_abs_step"],
        "le_p95_abs_epoch_step":
            le_stability["p95_abs_step"],
        "limit_median_abs_epoch_step":
            limit_stability["median_abs_step"],
        "limit_p95_abs_epoch_step":
            limit_stability["p95_abs_step"],
        "aitken_fallback_fraction":
            fallback_fraction,
        **oracle,
    }

    topq_rows = []

    for q in top_fractions:
        topq_rows.append({
            "stage": stage,
            "variant": variant,
            "K": int(K),
            "reference_method": str(
                reference_method
            ),
            "trim_fraction": float(
                trim_fraction
            ),
            "common_limit_method": str(
                limit_method
            ),
            "epoch": pair_ep,
            "pairwise_auc": pair_best,
            **topq_metrics(
                y,
                score_best,
                q,
            ),
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
                "reference_method": str(
                    reference_method
                ),
                "trim_fraction": float(
                    trim_fraction
                ),
                "common_limit_method": str(
                    limit_method
                ),
                "epoch": int(ep),
                "raw_le_auc": (
                    float(auc_raw[t])
                    if np.isfinite(auc_raw[t])
                    else np.nan
                ),
                "z_le_auc": (
                    float(auc_z[t])
                    if np.isfinite(auc_z[t])
                    else np.nan
                ),
                "pairwise_cumulative_auc": (
                    float(auc_pair[t])
                    if np.isfinite(auc_pair[t])
                    else np.nan
                ),
            })

    return (
        summary,
        topq_rows,
        epoch_rows,
    )


# ============================================================
# Plots
# ============================================================

def plot_stage_auc(
    epoch_rows,
    stage,
    path,
):
    rows = [
        r
        for r in epoch_rows
        if r["stage"] == stage
    ]

    variants = sorted(
        set(
            r["variant"]
            for r in rows
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for variant in variants:
        rr = [
            r
            for r in rows
            if r["variant"] == variant
        ]

        rr.sort(
            key=lambda r: r["epoch"]
        )

        ax.plot(
            [
                r["epoch"]
                for r in rr
            ],
            [
                r["pairwise_cumulative_auc"]
                for r in rr
            ],
            label=variant,
        )

    ax.axhline(
        0.5,
        linewidth=1,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Cumulative pairwise ROC-AUC"
    )

    ax.set_ylim(
        0.5,
        1.0,
    )

    ax.set_title(
        f"LE-GIE sensitivity — {stage}"
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


def plot_limit_stability(
    summary_rows,
    path,
):
    rows = [
        r
        for r in summary_rows
        if r["stage"]
        == "common_limit_sensitivity"
    ]

    labels = [
        r["variant"]
        for r in rows
    ]

    med = [
        r["le_median_abs_epoch_step"]
        for r in rows
    ]

    p95 = [
        r["le_p95_abs_epoch_step"]
        for r in rows
    ]

    x = np.arange(
        len(rows),
        dtype=float,
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        med,
        width,
        label="Median |LE(t)-LE(t-1)|",
    )

    ax.bar(
        x + width / 2,
        p95,
        width,
        label="95th percentile",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(
        "Absolute epoch-to-epoch LE change"
    )

    ax.set_title(
        "LE temporal stability by common limit estimator"
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    d = np.load(
        args.common_npz,
        allow_pickle=False,
    )

    required = {
        "epoch",
        "loss_traj",
        "observed_label",
        "is_anomaly",
    }

    missing = (
        required - set(d.files)
    )

    if missing:
        raise KeyError(
            f"Missing arrays in common NPZ: "
            f"{sorted(missing)}"
        )

    loss = np.asarray(
        d["loss_traj"],
        dtype=np.float64,
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

    summary_rows = []
    topq_rows = []
    epoch_rows = []

    cache = {}

    def get_result(
        K,
        reference_method,
        trim_fraction,
        limit_method,
    ):
        key = (
            int(K),
            str(reference_method),
            float(trim_fraction),
            str(limit_method),
        )

        if key not in cache:
            print(
                "Computing consistent LE-GIE: "
                f"K={K}, "
                f"ref={reference_method}, "
                f"trim={trim_fraction}, "
                f"limit={limit_method} ...",
                flush=True,
            )

            cache[key] = (
                rolling_common_limit_le_gie_batch(
                    loss,
                    labels,
                    K=int(K),
                    num_classes=args.num_classes,
                    reference_method=reference_method,
                    trim_fraction=float(
                        trim_fraction
                    ),
                    limit_method=limit_method,
                )
            )

        return cache[key]

    def run_variant(
        stage,
        variant,
        K,
        reference_method,
        trim_fraction,
        limit_method,
    ):
        result = get_result(
            K,
            reference_method,
            trim_fraction,
            limit_method,
        )

        summary, tq, er = (
            evaluate_variant(
                stage=stage,
                variant=variant,
                K=K,
                reference_method=
                    reference_method,
                trim_fraction=
                    trim_fraction,
                limit_method=
                    limit_method,
                result=result,
                labels=labels,
                y=y,
                epochs=epochs,
                top_fractions=
                    args.top_fractions,
                target_fpr=
                    args.target_fpr,
                num_classes=
                    args.num_classes,
            )
        )

        summary_rows.append(summary)
        topq_rows.extend(tq)
        epoch_rows.extend(er)

        print(
            f"{stage:>26s} | "
            f"{variant:<24s} | "
            f"raw={summary['raw_le_best_auc']:.6f} "
            f"z={summary['z_le_best_auc']:.6f} "
            f"pair={summary['pairwise_cumulative_best_auc']:.6f} "
            f"@ e{summary['pairwise_cumulative_best_epoch']} "
            f"oracleTPR@FPR<={args.target_fpr:g}="
            f"{summary['oracle_TPR']:.4f} "
            f"LEstep50={summary['le_median_abs_epoch_step']:.6g} "
            f"LEstep95={summary['le_p95_abs_epoch_step']:.6g}",
            flush=True,
        )

    # --------------------------------------------------------
    # Stage A: K sensitivity with the user's stable last-3
    # limit applied to THE WHOLE estimator.
    # --------------------------------------------------------

    for K in args.k_values:
        run_variant(
            "K_sensitivity",
            f"K={int(K)}",
            int(K),
            "mean",
            0.10,
            "last3_mean",
        )

    # --------------------------------------------------------
    # Stage B: reference sensitivity, still with ONE common
    # last-3 limit rule everywhere.
    # --------------------------------------------------------

    K = int(
        args.reference_k
    )

    reference_variants = [
        (
            "mean",
            "mean",
            0.10,
        ),
        (
            "leave_one_out_mean",
            "leave_one_out_mean",
            0.10,
        ),
        (
            "median",
            "median",
            0.10,
        ),
        (
            "trimmed_mean_05",
            "trimmed_mean",
            0.05,
        ),
        (
            "trimmed_mean_10",
            "trimmed_mean",
            0.10,
        ),
    ]

    for (
        variant,
        reference_method,
        trim_fraction,
    ) in reference_variants:
        run_variant(
            "reference_sensitivity",
            variant,
            K,
            reference_method,
            trim_fraction,
            "last3_mean",
        )

    # --------------------------------------------------------
    # Stage C: the limit estimator changes for THE WHOLE LE.
    # --------------------------------------------------------

    K = int(
        args.limit_k
    )

    for method in (
        "last3_mean",
        "next",
        "aitken_guarded",
    ):
        run_variant(
            "common_limit_sensitivity",
            method,
            K,
            "mean",
            0.10,
            method,
        )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    write_csv(
        args.output_dir
        / "le_improvement_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output_dir
        / "le_improvement_topq.csv",
        topq_rows,
    )

    write_csv(
        args.output_dir
        / "le_improvement_auc_by_epoch.csv",
        epoch_rows,
    )

    for stage, filename in (
        (
            "K_sensitivity",
            "fig_k_sensitivity_pairwise_auc.png",
        ),
        (
            "reference_sensitivity",
            "fig_reference_sensitivity_pairwise_auc.png",
        ),
        (
            "common_limit_sensitivity",
            "fig_common_limit_sensitivity_pairwise_auc.png",
        ),
    ):
        plot_stage_auc(
            epoch_rows,
            stage,
            args.output_dir / filename,
        )

    plot_limit_stability(
        summary_rows,
        args.output_dir
        / "fig_common_limit_stability.png",
    )

    metadata = {
        "artifact":
            "LE_GIE_consistent_common_limit_sensitivity",
        "theoretical_estimator":
            "ell_hat = ell_err_hat + log(abs(m_GIE_hat))",
        "theoretical_formula_changed":
            False,
        "common_limit_enforced":
            True,
        "common_limit_statement":
            (
                "The same sample L_hat_i(t) is used in ell_err "
                "and the sample-side GIE residuals; the class-reference "
                "limit is computed by the same rule on G_i(t)."
            ),
        "window_definition":
            (
                "At latest observation t: "
                "boundary=x[t-K-1], "
                "K-point tail=x[t-K:t], "
                "limit uses observations through x[t]."
            ),
        "primary_limit_method":
            "last3_mean",
        "primary_limit_rationale":
            (
                "Mean of the latest three observations is retained "
                "as the primary stability safeguard."
            ),
        "k_values": [
            int(k)
            for k in args.k_values
        ],
        "reference_methods": [
            "mean",
            "leave_one_out_mean",
            "median",
            "trimmed_mean_05",
            "trimmed_mean_10",
        ],
        "common_limit_methods": [
            "last3_mean",
            "next",
            "aitken_guarded",
        ],
        "top_fractions": [
            float(q)
            for q in args.top_fractions
        ],
        "target_fpr":
            float(args.target_fpr),
        "stability_metrics": [
            "median absolute epoch-to-epoch LE change",
            "95th percentile absolute epoch-to-epoch LE change",
        ],
        "oracle_warning":
            (
                "oracle TPR uses known anomaly labels "
                "and is evaluation-only"
            ),
        "selection_warning":
            (
                "This run is exploratory. Validate any chosen "
                "configuration on an independent training run."
            ),
        "production_modules_modified":
            False,
    }

    with (
        args.output_dir
        / "le_improvement_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "IMPORTANT: common L_hat was enforced for the whole estimator."
    )
    print(
        "Previous production estimator/detector modules were not modified."
    )
    print(
        f"Outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
