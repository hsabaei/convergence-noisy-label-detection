#!/usr/bin/env python
"""
Paired intervention experiment for CKL, LE, and CKL+LE.

One run trains TWO models:
    A) baseline: ordinary CE on observed/noisy labels
    B) adaptive: sample-specific target intervention guided by one score source
       and one temporal detector.

Supported score sources
-----------------------
    ckl       : CKL with the frozen last-three limit
    le        : signed LE-GIE with next/next limits
    combined  : r_CKL + r_LE - r_CKL*r_LE

Supported temporal detectors
----------------------------
    ewma
    cumulative_pairwise

Frozen defaults
---------------
    K = 40
    q = 0.10
    EWMA lambda = 0.05
    gamma_max = 2.0
    alpha_floor = 0.20

Causal intervention
-------------------
At epoch t:
    1) train using alpha values computed after epoch t-1
    2) evaluate deterministic per-sample observed-label loss
    3) compute the requested score
    4) update EWMA or cumulative pairwise statistic
    5) select the current top-q suspicious samples
    6) compute alpha for epoch t+1

Only selected samples are corrected:
    y* = alpha*y_observed + (1-alpha)*stopgrad(p_theta)

All other samples keep alpha = 1.

IMPORTANT
---------
* True labels and the synthetic-noise mask are evaluation-only.
* Detection/intervention NEVER uses true labels or the noise mask.
* CKL and LE are first converted to within-observed-class percentiles.
* The signed LE formula is unchanged:
      ell_hat = ell_err_hat + log(abs(m_GIE_hat))
  We do NOT take abs(ell_hat).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convergence_monitoring.data import NoisyCIFAR10WithIndex
from convergence_monitoring.models import CNN12_Model
from convergence_monitoring.training import set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

EPS = 1e-7
ESTIMATOR_BATCH = 2000


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "score_guided_intervention",
    )
    p.add_argument("--download", action="store_true")

    p.add_argument(
        "--score-source",
        choices=("ckl", "le", "combined"),
        required=True,
    )
    p.add_argument(
        "--detector",
        choices=("ewma", "cumulative_pairwise"),
        required=True,
    )

    p.add_argument("--noisy-frac", type=float, default=0.05)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    # Frozen detector/intervention settings.
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--ewma-lambda", type=float, default=0.05)
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--alpha-floor", type=float, default=0.20)

    p.add_argument("--save-models", action="store_true")

    args = p.parse_args()

    if not (0.0 <= args.noisy_frac < 1.0):
        p.error("--noisy-frac must be in [0,1).")
    if not (0.0 < args.q < 1.0):
        p.error("--q must be in (0,1).")
    if not (0.0 < args.ewma_lambda <= 1.0):
        p.error("--ewma-lambda must be in (0,1].")
    if args.K < 6:
        p.error("--K must be >= 6 for the GIE estimator.")
    if args.gamma_max < 0.0:
        p.error("--gamma-max must be nonnegative.")
    if not (0.0 <= args.alpha_floor <= 1.0):
        p.error("--alpha-floor must be in [0,1].")

    return args


# ---------------------------------------------------------------------
# Reproducibility / data
# ---------------------------------------------------------------------

def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def capture_torch_rng(device):
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state


def restore_torch_rng(state, device):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


# ---------------------------------------------------------------------
# Adaptive target
# ---------------------------------------------------------------------

def adaptive_target_cross_entropy(logits, y_observed, alpha, num_classes):
    """Sample-specific mixed target.

    y* = alpha*y_observed + (1-alpha)*stopgrad(p_theta)
    """
    alpha = alpha.to(dtype=logits.dtype).view(-1, 1)
    y_onehot = F.one_hot(
        y_observed,
        num_classes=num_classes,
    ).to(dtype=logits.dtype)

    p_detached = torch.softmax(
        logits.detach(),
        dim=1,
    )

    target = (
        alpha * y_onehot
        + (1.0 - alpha) * p_detached
    )

    return -(
        target * F.log_softmax(logits, dim=1)
    ).sum(dim=1).mean()


def correction_strength(epoch, first_detector_epoch, total_epochs, gamma_max):
    """Gradually increase intervention strength after detector activation."""
    if epoch < first_detector_epoch:
        return 0.0

    denom = max(
        total_epochs - first_detector_epoch,
        1,
    )
    frac = np.clip(
        (epoch - first_detector_epoch) / denom,
        0.0,
        1.0,
    )
    return float(gamma_max) * float(frac)


def alpha_from_detector_stat(stat, detected, gamma, alpha_floor):
    """Only detected samples are modified; all others keep alpha=1."""
    stat = np.asarray(stat, dtype=np.float32)
    detected = np.asarray(detected, dtype=bool)

    alpha = np.ones(stat.shape[0], dtype=np.float32)

    active = (
        detected
        & np.isfinite(stat)
    )

    if np.any(active):
        strength = np.clip(
            stat[active],
            0.0,
            1.0,
        )
        val = np.exp(
            -float(gamma) * strength
        )
        alpha[active] = np.maximum(
            val,
            float(alpha_floor),
        ).astype(np.float32)

    return alpha


# ---------------------------------------------------------------------
# CKL / LE estimators
# ---------------------------------------------------------------------

def build_class_mean(loss_history, labels, num_classes):
    x = np.asarray(loss_history, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    out = np.full(
        (num_classes, x.shape[1]),
        np.nan,
        dtype=np.float32,
    )

    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size:
            out[c] = np.nanmean(
                x[idx],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

    return out


def gie_batch(phi, G, phi_limit, G_limit, return_W=False):
    """Frozen GIE/Hill construction used in the CKL/LE experiments."""
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    phi_limit = np.asarray(phi_limit, dtype=np.float64)
    G_limit = np.asarray(G_limit, dtype=np.float64)

    R = np.abs(
        phi - phi_limit[:, None]
    )
    FR = np.abs(
        G - G_limit[:, None]
    )

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
        phi.shape[0],
        np.nan,
        dtype=np.float64,
    )
    hill_den = np.full(
        phi.shape[0],
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

    d = np.full(
        phi.shape[0],
        np.nan,
        dtype=np.float64,
    )
    d[good] = (
        hill_num[good]
        / hill_den[good]
    )

    if return_W:
        return d, w0

    return d


def geometric_class_reference(values, labels, num_classes):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    out = np.full(
        num_classes,
        np.nan,
        dtype=np.float64,
    )

    for c in range(num_classes):
        v = values[
            labels == c
        ]
        v = v[
            np.isfinite(v)
            & (v > 0.0)
        ]

        if v.size:
            out[c] = np.exp(
                np.mean(
                    np.log(
                        np.maximum(v, EPS)
                    )
                )
            )

    return out


def ckl_vectorized(W1, d1, W2, d2):
    W1 = np.asarray(W1, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    out = np.full(
        W1.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(W1)
        & np.isfinite(d1)
        & np.isfinite(W2)
        & np.isfinite(d2)
        & (W1 > 0.0)
        & (d1 > 0.0)
        & (W2 > 0.0)
        & (d2 > 0.0)
    )

    if not np.any(valid):
        return out

    a = W1[valid]
    b = d1[valid]
    c = W2[valid]
    e = d2[valid]

    res = np.full(
        a.shape,
        np.nan,
        dtype=np.float64,
    )

    equal = np.abs(a - c) < 1e-12
    lower = (
        (~equal)
        & (a < c)
    )
    upper = (
        (~equal)
        & (~lower)
    )

    if np.any(equal):
        aa = a[equal]
        bb = b[equal]
        cc = e[equal]

        res[equal] = (
            aa
            * ((cc - bb) ** 2)
            / (
                ((bb + 1.0) ** 2)
                * (cc + 1.0)
                + EPS
            )
        )

    if np.any(lower):
        aa = a[lower]
        bb = b[lower]
        ww = c[lower]
        dd = e[lower]

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            term = (
                aa
                * (
                    (
                        dd / (bb + 1.0)
                    )
                    * np.log(
                        np.maximum(
                            ww / aa,
                            EPS,
                        )
                    )
                    - (
                        (bb - dd)
                        / ((bb + 1.0) ** 2)
                    )
                )
                + dd
                * (
                    (ww - aa)
                    + aa
                    * np.log(
                        np.maximum(
                            aa / ww,
                            EPS,
                        )
                    )
                )
            )

            res[lower] = (
                term
                + (
                    bb / (bb + 1.0)
                )
                * aa
                - (
                    dd / (dd + 1.0)
                )
                * ww
            )

    if np.any(upper):
        aa = a[upper]
        bb = b[upper]
        ww = c[upper]
        dd = e[upper]

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            ratio = np.maximum(
                ww / aa,
                EPS,
            )

            term = (
                aa
                / ((bb + 1.0) ** 2)
                * (
                    dd
                    * (
                        ratio ** (bb + 1.0)
                    )
                    - bb
                )
            )

            res[upper] = (
                term
                + (
                    bb / (bb + 1.0)
                )
                * aa
                - (
                    dd / (dd + 1.0)
                )
                * ww
            )

    out[valid] = res
    return out


def latest_ckl_last3(loss_history, labels, K, num_classes):
    """Latest CKL using the frozen last-three limit."""
    x = np.asarray(
        loss_history,
        dtype=np.float32,
    )
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    N, T = x.shape

    if T < K:
        return np.full(
            N,
            np.nan,
            dtype=np.float32,
        )

    t = T - 1
    start = t - K + 1

    Gc = build_class_mean(
        x,
        labels,
        num_classes,
    )

    G_limit_cls = np.mean(
        Gc[:, t - 2:t + 1],
        axis=1,
        dtype=np.float64,
    )

    d_t = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )
    W_t = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )

    for lo in range(
        0,
        N,
        ESTIMATOR_BATCH,
    ):
        hi = min(
            lo + ESTIMATOR_BATCH,
            N,
        )

        xb = np.asarray(
            x[lo:hi, start:t + 1],
            dtype=np.float64,
        )
        lab = labels[lo:hi]
        Gb = np.asarray(
            Gc[lab, start:t + 1],
            dtype=np.float64,
        )

        Lx = np.mean(
            xb[:, -3:],
            axis=1,
        )
        Lg = G_limit_cls[lab]

        db, wb = gie_batch(
            xb,
            Gb,
            Lx,
            Lg,
            return_W=True,
        )

        d_t[lo:hi] = db
        W_t[lo:hi] = wb

    Wc = geometric_class_reference(
        W_t,
        labels,
        num_classes,
    )
    dc = geometric_class_reference(
        d_t,
        labels,
        num_classes,
    )

    score = ckl_vectorized(
        W_t,
        d_t,
        Wc[labels],
        dc[labels],
    )

    return score.astype(
        np.float32,
    )


def latest_le_next_next(loss_history, labels, K, num_classes):
    """Latest signed LE-GIE using next-sample L in both components.

    The theoretical formula remains:
        ell_hat = ell_err_hat + log(abs(m_GIE_hat))

    We intentionally do NOT take abs(ell_hat).
    """
    x = np.asarray(
        loss_history,
        dtype=np.float32,
    )
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    N, T = x.shape

    if T < K + 2:
        return np.full(
            N,
            np.nan,
            dtype=np.float32,
        )

    t = T - 1
    boundary = t - K - 1
    start = t - K

    Gc = build_class_mean(
        x,
        labels,
        num_classes,
    )

    G_limit_cls = np.asarray(
        Gc[:, t],
        dtype=np.float64,
    )

    inv_j = (
        1.0
        / np.arange(
            1,
            K + 1,
            dtype=np.float64,
        )
    )

    out = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    for lo in range(
        0,
        N,
        ESTIMATOR_BATCH,
    ):
        hi = min(
            lo + ESTIMATOR_BATCH,
            N,
        )

        lab = labels[lo:hi]

        xb = np.asarray(
            x[lo:hi, start:t],
            dtype=np.float64,
        )
        L = np.asarray(
            x[lo:hi, t],
            dtype=np.float64,
        )

        r0 = np.abs(
            np.asarray(
                x[lo:hi, boundary],
                dtype=np.float64,
            )
            - L
        )

        rj = np.abs(
            xb
            - L[:, None]
        )

        valid_err = (
            np.isfinite(L)
            & np.isfinite(r0)
            & (r0 != 0.0)
            & np.all(
                np.isfinite(rj)
                & (rj != 0.0),
                axis=1,
            )
        )

        ell_err = np.full(
            hi - lo,
            np.nan,
            dtype=np.float64,
        )

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

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            le = (
                ell_err
                + np.log(
                    np.abs(m)
                )
            )

        valid = (
            valid_err
            & np.isfinite(m)
            & (m != 0.0)
            & np.isfinite(le)
        )

        block = np.full(
            hi - lo,
            np.nan,
            dtype=np.float32,
        )
        block[valid] = le[
            valid
        ].astype(np.float32)

        out[lo:hi] = block

    return out


# ---------------------------------------------------------------------
# Rank / combination / temporal detectors
# ---------------------------------------------------------------------

def within_class_percentile(score, labels, num_classes):
    """Tie-aware percentile within observed class, in [0,1]."""
    score = np.asarray(
        score,
        dtype=np.float64,
    )
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    out = np.full(
        score.shape,
        np.nan,
        dtype=np.float32,
    )

    for c in range(num_classes):
        idx = np.flatnonzero(
            (labels == c)
            & np.isfinite(score)
        )

        if idx.size == 0:
            continue

        vals = score[idx]
        order = np.argsort(
            vals,
            kind="mergesort",
        )
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
                and sorted_vals[b]
                == sorted_vals[a]
            ):
                b += 1

            rank_sorted[a:b] = (
                0.5
                * (a + b - 1)
            )
            a = b

        ranks = np.empty(
            idx.size,
            dtype=np.float64,
        )
        ranks[order] = rank_sorted

        if idx.size == 1:
            pct = np.ones(
                1,
                dtype=np.float64,
            )
        else:
            pct = (
                ranks
                / float(idx.size - 1)
            )

        out[idx] = pct.astype(
            np.float32,
        )

    return out


def combined_score(ckl_rank, le_rank):
    out = np.full_like(
        ckl_rank,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(ckl_rank)
        & np.isfinite(le_rank)
    )

    c = np.clip(
        ckl_rank[valid],
        0.0,
        1.0,
    )
    l = np.clip(
        le_rank[valid],
        0.0,
        1.0,
    )

    out[valid] = (
        c + l - c * l
    ).astype(np.float32)

    return out


def update_ewma(previous, current, lam):
    previous = np.asarray(
        previous,
        dtype=np.float32,
    ).copy()

    current = np.asarray(
        current,
        dtype=np.float32,
    )

    finite = np.isfinite(
        current
    )

    init = (
        finite
        & ~np.isfinite(previous)
    )
    previous[init] = current[init]

    update = (
        finite
        & np.isfinite(previous)
    )
    previous[update] = (
        (1.0 - float(lam))
        * previous[update]
        + float(lam)
        * current[update]
    ).astype(np.float32)

    return previous


def update_cumulative_pairwise(
    current_score,
    labels,
    running_wins,
    running_count,
):
    """One-epoch update matching exact all-peer same-class pairwise logic."""
    score = np.asarray(
        current_score,
        dtype=np.float64,
    )
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    wins_t = np.zeros(
        score.shape[0],
        dtype=np.float64,
    )
    count_t = np.zeros(
        score.shape[0],
        dtype=np.int64,
    )

    for c in np.unique(labels):
        members = np.flatnonzero(
            labels == c
        )

        vals = score[members]
        finite_local = np.flatnonzero(
            np.isfinite(vals)
        )

        n_valid = int(
            finite_local.size
        )

        if n_valid <= 1:
            continue

        finite_members = members[
            finite_local
        ]
        finite_vals = vals[
            finite_local
        ]

        order = np.argsort(
            finite_vals,
            kind="mergesort",
        )
        sorted_vals = finite_vals[
            order
        ]
        sorted_members = finite_members[
            order
        ]

        group_start = 0

        while group_start < n_valid:
            group_end = (
                group_start + 1
            )

            value = sorted_vals[
                group_start
            ]

            while (
                group_end < n_valid
                and sorted_vals[group_end]
                == value
            ):
                group_end += 1

            group_size = (
                group_end
                - group_start
            )

            lower_count = (
                group_start
            )
            tied_other = (
                group_size - 1
            )

            win_mass = (
                float(lower_count)
                + 0.5
                * float(tied_other)
            )
            denom = (
                n_valid - 1
            )

            group_members = (
                sorted_members[
                    group_start:group_end
                ]
            )

            wins_t[
                group_members
            ] = win_mass

            count_t[
                group_members
            ] = denom

            group_start = group_end

    running_wins = (
        running_wins
        + wins_t
    )
    running_count = (
        running_count
        + count_t
    )

    cumulative = np.full(
        score.shape[0],
        np.nan,
        dtype=np.float32,
    )

    usable = (
        running_count > 0
    )

    cumulative[usable] = (
        running_wins[usable]
        / running_count[usable]
    ).astype(np.float32)

    return (
        running_wins,
        running_count,
        cumulative,
    )


def topq_mask(score, q):
    score = np.asarray(
        score,
        dtype=np.float64,
    )

    finite_idx = np.flatnonzero(
        np.isfinite(score)
    )

    selected = np.zeros(
        score.shape[0],
        dtype=bool,
    )

    if finite_idx.size == 0:
        return selected

    k = min(
        max(
            1,
            int(
                round(
                    float(q)
                    * score.size
                )
            ),
        ),
        finite_idx.size,
    )

    vals = score[
        finite_idx
    ]

    local = np.argpartition(
        vals,
        -k,
    )[-k:]

    selected[
        finite_idx[local]
    ] = True

    return selected


def current_detector_input(
    score_source,
    loss_history,
    labels,
    K,
    num_classes,
):
    """Return [0,1]-scale input for the temporal detector."""
    if score_source == "ckl":
        ckl = latest_ckl_last3(
            loss_history,
            labels,
            K,
            num_classes,
        )
        return within_class_percentile(
            ckl,
            labels,
            num_classes,
        )

    if score_source == "le":
        le = latest_le_next_next(
            loss_history,
            labels,
            K,
            num_classes,
        )
        return within_class_percentile(
            le,
            labels,
            num_classes,
        )

    if score_source == "combined":
        ckl = latest_ckl_last3(
            loss_history,
            labels,
            K,
            num_classes,
        )
        le = latest_le_next_next(
            loss_history,
            labels,
            K,
            num_classes,
        )

        ckl_rank = within_class_percentile(
            ckl,
            labels,
            num_classes,
        )
        le_rank = within_class_percentile(
            le,
            labels,
            num_classes,
        )

        return combined_score(
            ckl_rank,
            le_rank,
        )

    raise ValueError(
        f"Unknown score_source={score_source}"
    )


# ---------------------------------------------------------------------
# Train/evaluate
# ---------------------------------------------------------------------

def train_paired_one_epoch(
    baseline_model,
    adaptive_model,
    loader,
    baseline_optimizer,
    adaptive_optimizer,
    device,
    alpha_table,
    num_classes,
):
    """Same minibatch, same augmented tensor, synchronized dropout RNG."""
    baseline_model.train()
    adaptive_model.train()

    baseline_loss_sum = 0.0
    adaptive_loss_sum = 0.0
    total = 0

    for x, y_observed, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y_observed = y_observed.to(
            device,
            non_blocking=True,
        )

        idx_np = np.asarray(
            idx,
            dtype=np.int64,
        )

        alpha = torch.from_numpy(
            alpha_table[idx_np]
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        baseline_optimizer.zero_grad(
            set_to_none=True
        )
        adaptive_optimizer.zero_grad(
            set_to_none=True
        )

        paired_rng = capture_torch_rng(
            device
        )

        baseline_logits = baseline_model(
            x
        )

        restore_torch_rng(
            paired_rng,
            device,
        )

        adaptive_logits = adaptive_model(
            x
        )

        baseline_loss = F.cross_entropy(
            baseline_logits,
            y_observed,
        )

        adaptive_loss = adaptive_target_cross_entropy(
            adaptive_logits,
            y_observed,
            alpha,
            num_classes,
        )

        baseline_loss.backward()
        adaptive_loss.backward()

        baseline_optimizer.step()
        adaptive_optimizer.step()

        n = int(
            y_observed.numel()
        )

        baseline_loss_sum += (
            float(
                baseline_loss.item()
            )
            * n
        )
        adaptive_loss_sum += (
            float(
                adaptive_loss.item()
            )
            * n
        )
        total += n

    return {
        "baseline_optimization_loss":
            baseline_loss_sum
            / max(total, 1),
        "adaptive_optimization_loss":
            adaptive_loss_sum
            / max(total, 1),
    }


@torch.no_grad()
def evaluate_training_set(
    model,
    loader,
    device,
    true_labels,
    n_samples,
):
    """Deterministic training-set pass.

    Detector uses only observed-label CE loss.
    True-label correctness is returned only for evaluation.
    """
    model.eval()

    loss_observed = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )
    prediction = np.full(
        n_samples,
        -1,
        dtype=np.int64,
    )
    true_correct = np.zeros(
        n_samples,
        dtype=bool,
    )

    for x, y_observed, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y_observed = y_observed.to(
            device,
            non_blocking=True,
        )

        idx_np = np.asarray(
            idx,
            dtype=np.int64,
        )

        logits = model(x)

        per_loss = F.cross_entropy(
            logits,
            y_observed,
            reduction="none",
        )

        pred = (
            logits.argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        loss_observed[idx_np] = (
            per_loss
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        prediction[
            idx_np
        ] = pred

        true_correct[
            idx_np
        ] = (
            pred
            == true_labels[
                idx_np
            ]
        )

    if np.any(
        prediction < 0
    ):
        raise RuntimeError(
            "Evaluation did not cover every training sample."
        )

    return {
        "loss_observed":
            loss_observed,
        "prediction":
            prediction,
        "true_correct":
            true_correct,
    }


@torch.no_grad()
def evaluate_test_accuracy(
    model,
    loader,
    device,
):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits,
            y,
            reduction="sum",
        )

        pred = logits.argmax(
            dim=1
        )

        correct += int(
            (pred == y)
            .sum()
            .item()
        )
        total += int(
            y.numel()
        )
        loss_sum += float(
            loss.item()
        )

    return {
        "test_accuracy":
            correct
            / max(total, 1),
        "test_loss":
            loss_sum
            / max(total, 1),
    }


def masked_accuracy(correct, mask):
    mask = np.asarray(
        mask,
        dtype=bool,
    )
    count = int(
        np.sum(mask)
    )

    if count == 0:
        return np.nan

    return float(
        np.mean(
            np.asarray(
                correct,
                dtype=bool,
            )[mask]
        )
    )


def max_parameter_difference(model_a, model_b):
    maximum = 0.0

    with torch.no_grad():
        for pa, pb in zip(
            model_a.parameters(),
            model_b.parameters(),
        ):
            diff = float(
                torch.max(
                    torch.abs(
                        pa - pb
                    )
                ).item()
            )
            maximum = max(
                maximum,
                diff,
            )

    return maximum


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

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )
    set_seed(
        args.seed
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_tf, eval_tf = build_transforms()

    train_ds = NoisyCIFAR10WithIndex(
        root=str(
            args.data_root
        ),
        train=True,
        transform=train_tf,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )

    eval_ds = NoisyCIFAR10WithIndex(
        root=str(
            args.data_root
        ),
        train=True,
        transform=eval_tf,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )

    # Clean CIFAR-10 test set.
    test_ds = CIFAR10(
        root=str(
            args.data_root
        ),
        train=False,
        transform=eval_tf,
        download=args.download,
    )

    if not np.array_equal(
        train_ds.targets,
        eval_ds.targets,
    ):
        raise RuntimeError(
            "Training/evaluation observed labels differ."
        )

    if not np.array_equal(
        train_ds.true_targets,
        eval_ds.true_targets,
    ):
        raise RuntimeError(
            "Training/evaluation true labels differ."
        )

    if not np.array_equal(
        train_ds.is_anomaly,
        eval_ds.is_anomaly,
    ):
        raise RuntimeError(
            "Training/evaluation noise masks differ."
        )

    observed_labels = np.asarray(
        train_ds.targets,
        dtype=np.int64,
    )
    true_labels = np.asarray(
        train_ds.true_targets,
        dtype=np.int64,
    )
    noisy_mask = np.asarray(
        train_ds.is_anomaly,
        dtype=bool,
    )
    clean_mask = ~noisy_mask
    n_samples = len(
        train_ds
    )

    generator = torch.Generator().manual_seed(
        args.seed
    )
    pin = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        generator=generator,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    baseline_model = CNN12_Model(
        num_classes=args.num_classes
    ).to(device)

    adaptive_model = copy.deepcopy(
        baseline_model
    ).to(device)

    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    adaptive_optimizer = torch.optim.AdamW(
        adaptive_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Detection uses ONLY adaptive-model observed-label loss trajectories.
    adaptive_loss_traj = np.full(
        (
            n_samples,
            args.num_epochs,
        ),
        np.nan,
        dtype=np.float32,
    )

    detector_input_traj = np.full(
        (
            n_samples,
            args.num_epochs,
        ),
        np.nan,
        dtype=np.float32,
    )

    detector_stat_traj = np.full(
        (
            n_samples,
            args.num_epochs,
        ),
        np.nan,
        dtype=np.float32,
    )

    detected_traj = np.zeros(
        (
            n_samples,
            args.num_epochs,
        ),
        dtype=bool,
    )

    alpha_traj = np.ones(
        (
            n_samples,
            args.num_epochs,
        ),
        dtype=np.float32,
    )

    baseline_true_correct_traj = np.zeros(
        (
            n_samples,
            args.num_epochs,
        ),
        dtype=bool,
    )

    adaptive_true_correct_traj = np.zeros(
        (
            n_samples,
            args.num_epochs,
        ),
        dtype=bool,
    )

    alpha_table = np.ones(
        n_samples,
        dtype=np.float32,
    )

    ewma_state = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )

    pair_running_wins = np.zeros(
        n_samples,
        dtype=np.float64,
    )
    pair_running_count = np.zeros(
        n_samples,
        dtype=np.int64,
    )

    # Common start for CKL, LE, and CKL+LE.
    # LE K=40 next/next first becomes valid at epoch 42.
    first_detector_epoch = (
        args.K + 2
    )

    epoch_rows = []

    print("=" * 78)
    print("PAIRED SCORE-GUIDED INTERVENTION")
    print("=" * 78)
    print(f"device={device}")
    print(f"score_source={args.score_source}")
    print(f"detector={args.detector}")
    print(f"noise={args.noisy_frac:.3f}")
    print(f"K={args.K}")
    print(f"q={args.q}")
    print(f"EWMA lambda={args.ewma_lambda}")
    print(f"gamma_max={args.gamma_max}")
    print(f"alpha_floor={args.alpha_floor}")
    print(f"first_detector_epoch={first_detector_epoch}")
    print("baseline target = observed label")
    print(
        "adaptive target = "
        "alpha*y_observed + (1-alpha)*stopgrad(p)"
    )
    print(
        "true labels/noise mask are evaluation-only"
    )
    print()

    for epoch_idx in range(
        args.num_epochs
    ):
        epoch = (
            epoch_idx + 1
        )

        # alpha computed after previous epoch is used now.
        alpha_traj[
            :,
            epoch_idx,
        ] = alpha_table

        train_stats = train_paired_one_epoch(
            baseline_model=baseline_model,
            adaptive_model=adaptive_model,
            loader=train_loader,
            baseline_optimizer=baseline_optimizer,
            adaptive_optimizer=adaptive_optimizer,
            device=device,
            alpha_table=alpha_table,
            num_classes=args.num_classes,
        )

        baseline_eval = evaluate_training_set(
            baseline_model,
            eval_loader,
            device,
            true_labels,
            n_samples,
        )

        adaptive_eval = evaluate_training_set(
            adaptive_model,
            eval_loader,
            device,
            true_labels,
            n_samples,
        )

        baseline_test = evaluate_test_accuracy(
            baseline_model,
            test_loader,
            device,
        )

        adaptive_test = evaluate_test_accuracy(
            adaptive_model,
            test_loader,
            device,
        )

        baseline_true_correct_traj[
            :,
            epoch_idx,
        ] = baseline_eval[
            "true_correct"
        ]

        adaptive_true_correct_traj[
            :,
            epoch_idx,
        ] = adaptive_eval[
            "true_correct"
        ]

        adaptive_loss_traj[
            :,
            epoch_idx,
        ] = adaptive_eval[
            "loss_observed"
        ]

        detector_input = np.full(
            n_samples,
            np.nan,
            dtype=np.float32,
        )
        detector_stat = np.full(
            n_samples,
            np.nan,
            dtype=np.float32,
        )
        detected = np.zeros(
            n_samples,
            dtype=bool,
        )
        gamma = 0.0

        if epoch >= first_detector_epoch:
            detector_input = current_detector_input(
                score_source=args.score_source,
                loss_history=adaptive_loss_traj[
                    :,
                    :epoch,
                ],
                labels=observed_labels,
                K=args.K,
                num_classes=args.num_classes,
            )

            if args.detector == "ewma":
                ewma_state = update_ewma(
                    ewma_state,
                    detector_input,
                    args.ewma_lambda,
                )
                detector_stat = (
                    ewma_state.copy()
                )

            elif args.detector == "cumulative_pairwise":
                (
                    pair_running_wins,
                    pair_running_count,
                    detector_stat,
                ) = update_cumulative_pairwise(
                    current_score=detector_input,
                    labels=observed_labels,
                    running_wins=pair_running_wins,
                    running_count=pair_running_count,
                )

            else:
                raise ValueError(
                    args.detector
                )

            detected = topq_mask(
                detector_stat,
                args.q,
            )

            gamma = correction_strength(
                epoch=epoch,
                first_detector_epoch=first_detector_epoch,
                total_epochs=args.num_epochs,
                gamma_max=args.gamma_max,
            )

            # CAUSAL: alpha computed here is used at epoch t+1.
            alpha_table = alpha_from_detector_stat(
                detector_stat,
                detected,
                gamma,
                args.alpha_floor,
            )

        detector_input_traj[
            :,
            epoch_idx,
        ] = detector_input

        detector_stat_traj[
            :,
            epoch_idx,
        ] = detector_stat

        detected_traj[
            :,
            epoch_idx,
        ] = detected

        # Evaluation-only detector metrics.
        tp = int(
            np.sum(
                detected
                & noisy_mask
            )
        )
        fp = int(
            np.sum(
                detected
                & clean_mask
            )
        )

        n_noisy = int(
            np.sum(
                noisy_mask
            )
        )
        n_clean = int(
            np.sum(
                clean_mask
            )
        )

        tpr = (
            tp
            / max(
                n_noisy,
                1,
            )
        )
        fpr = (
            fp
            / max(
                n_clean,
                1,
            )
        )

        base_all_acc = float(
            np.mean(
                baseline_eval[
                    "true_correct"
                ]
            )
        )
        adapt_all_acc = float(
            np.mean(
                adaptive_eval[
                    "true_correct"
                ]
            )
        )

        base_clean_acc = masked_accuracy(
            baseline_eval[
                "true_correct"
            ],
            clean_mask,
        )
        adapt_clean_acc = masked_accuracy(
            adaptive_eval[
                "true_correct"
            ],
            clean_mask,
        )

        base_noisy_acc = masked_accuracy(
            baseline_eval[
                "true_correct"
            ],
            noisy_mask,
        )
        adapt_noisy_acc = masked_accuracy(
            adaptive_eval[
                "true_correct"
            ],
            noisy_mask,
        )

        param_diff = max_parameter_difference(
            baseline_model,
            adaptive_model,
        )

        row = {
            "epoch": epoch,
            "score_source":
                args.score_source,
            "detector":
                args.detector,
            "noisy_frac":
                args.noisy_frac,
            "K":
                args.K,
            "q":
                args.q,

            "baseline_optimization_loss":
                train_stats[
                    "baseline_optimization_loss"
                ],
            "adaptive_optimization_loss":
                train_stats[
                    "adaptive_optimization_loss"
                ],

            # Training accuracy against ORIGINAL TRUE labels.
            "baseline_train_true_accuracy_all":
                base_all_acc,
            "adaptive_train_true_accuracy_all":
                adapt_all_acc,
            "train_true_accuracy_gain_all":
                adapt_all_acc
                - base_all_acc,

            "baseline_train_true_accuracy_clean":
                base_clean_acc,
            "adaptive_train_true_accuracy_clean":
                adapt_clean_acc,

            "baseline_train_true_accuracy_noisy":
                base_noisy_acc,
            "adaptive_train_true_accuracy_noisy":
                adapt_noisy_acc,
            "train_true_accuracy_gain_noisy":
                adapt_noisy_acc
                - base_noisy_acc,

            # Clean CIFAR-10 test set.
            "baseline_test_accuracy":
                baseline_test[
                    "test_accuracy"
                ],
            "adaptive_test_accuracy":
                adaptive_test[
                    "test_accuracy"
                ],
            "test_accuracy_gain":
                adaptive_test[
                    "test_accuracy"
                ]
                - baseline_test[
                    "test_accuracy"
                ],
            "baseline_test_loss":
                baseline_test[
                    "test_loss"
                ],
            "adaptive_test_loss":
                adaptive_test[
                    "test_loss"
                ],

            "detector_active":
                bool(
                    epoch
                    >= first_detector_epoch
                ),
            "n_detected":
                int(
                    np.sum(
                        detected
                    )
                ),
            "detected_fraction":
                float(
                    np.mean(
                        detected
                    )
                ),
            "detector_TPR_eval_only":
                float(tpr),
            "detector_FPR_eval_only":
                float(fpr),
            "gamma":
                float(gamma),

            # alpha_table is for NEXT epoch.
            "mean_alpha_next_epoch":
                float(
                    np.mean(
                        alpha_table
                    )
                ),
            "mean_alpha_detected_next_epoch":
                (
                    float(
                        np.mean(
                            alpha_table[
                                detected
                            ]
                        )
                    )
                    if np.any(
                        detected
                    )
                    else np.nan
                ),
            "mean_alpha_noisy_next_epoch_eval_only":
                float(
                    np.mean(
                        alpha_table[
                            noisy_mask
                        ]
                    )
                ),
            "mean_alpha_clean_next_epoch_eval_only":
                float(
                    np.mean(
                        alpha_table[
                            clean_mask
                        ]
                    )
                ),

            "max_parameter_abs_difference":
                float(
                    param_diff
                ),
        }

        epoch_rows.append(
            row
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == first_detector_epoch
            or epoch == args.num_epochs
        ):
            print(
                f"epoch={epoch:03d} | "
                f"test base={row['baseline_test_accuracy']:.4f} "
                f"adapt={row['adaptive_test_accuracy']:.4f} "
                f"gain={row['test_accuracy_gain']:+.4f} | "
                f"train-true noisy base={base_noisy_acc:.4f} "
                f"adapt={adapt_noisy_acc:.4f} | "
                f"det={row['n_detected']:5d} "
                f"TPR={tpr:.4f} "
                f"FPR={fpr:.4f} "
                f"gamma={gamma:.3f} "
                f"alpha_det="
                f"{row['mean_alpha_detected_next_epoch']:.3f}"
            )

    write_csv(
        out_dir
        / "epoch_summary.csv",
        epoch_rows,
    )

    np.savez_compressed(
        out_dir
        / "intervention_trajectories.npz",
        sample_index=np.arange(
            n_samples,
            dtype=np.int64,
        ),
        epoch=np.arange(
            1,
            args.num_epochs + 1,
            dtype=np.int64,
        ),
        observed_label=observed_labels,
        true_label=true_labels,
        is_anomaly=noisy_mask,
        adaptive_loss_traj=adaptive_loss_traj,
        detector_input_traj=detector_input_traj,
        detector_stat_traj=detector_stat_traj,
        detected_traj=detected_traj,
        alpha_traj=alpha_traj,
        baseline_true_correct_traj=
            baseline_true_correct_traj,
        adaptive_true_correct_traj=
            adaptive_true_correct_traj,
    )

    config = {
        "artifact":
            "paired_score_guided_intervention",
        "score_source":
            args.score_source,
        "detector":
            args.detector,
        "score_definition": {
            "ckl":
                "CKL with last-three limit",
            "le":
                "signed LE-GIE: ell_err next + log(abs(m_GIE next))",
            "combined":
                "r_CKL + r_LE - r_CKL*r_LE",
        }[
            args.score_source
        ],
        "paired_design": {
            "same_initial_weights":
                True,
            "same_minibatches":
                True,
            "same_augmented_images":
                True,
            "synchronized_dropout_rng":
                True,
            "same_optimizer_hyperparameters":
                True,
        },
        "intervention": {
            "target":
                "alpha*y_observed + (1-alpha)*stopgrad(model_probability)",
            "only_detected_samples_modified":
                True,
            "alpha_rule":
                "max(alpha_floor, exp(-gamma*detector_stat))",
            "causal":
                "epoch t detection updates alpha used at epoch t+1",
            "gamma_max":
                args.gamma_max,
            "alpha_floor":
                args.alpha_floor,
        },
        "detector_settings": {
            "K":
                args.K,
            "q":
                args.q,
            "ewma_lambda":
                args.ewma_lambda,
            "first_detector_epoch":
                first_detector_epoch,
        },
        "training": {
            "noisy_frac":
                args.noisy_frac,
            "num_epochs":
                args.num_epochs,
            "seed":
                args.seed,
            "batch_size":
                args.batch_size,
            "optimizer":
                "AdamW",
            "lr":
                args.lr,
            "weight_decay":
                args.weight_decay,
        },
        "evaluation": {
            "true_labels_and_noise_mask":
                "evaluation-only",
            "training_accuracy":
                "prediction vs original true CIFAR-10 label",
            "test_accuracy":
                "clean CIFAR-10 test labels",
        },
    }

    with (
        out_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    if args.save_models:
        torch.save(
            baseline_model.state_dict(),
            out_dir
            / "baseline_model.pt",
        )
        torch.save(
            adaptive_model.state_dict(),
            out_dir
            / "adaptive_model.pt",
        )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(
        out_dir
        / "epoch_summary.csv"
    )
    print(
        out_dir
        / "intervention_trajectories.npz"
    )
    print(
        out_dir
        / "config.json"
    )


if __name__ == "__main__":
    main()
