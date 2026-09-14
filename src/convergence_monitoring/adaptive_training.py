"""Causal LE-GIE rank-EWMA detector + D2L-style adaptive targets."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-7
BATCH = 2000


def adaptive_target_cross_entropy(logits, y, alpha, num_classes):
    """y* = alpha*y + (1-alpha)*stopgrad(p_theta)."""
    alpha = alpha.to(dtype=logits.dtype).view(-1, 1)
    y_onehot = F.one_hot(y, num_classes=num_classes).to(dtype=logits.dtype)
    p_detached = torch.softmax(logits.detach(), dim=1)
    target = alpha * y_onehot + (1.0 - alpha) * p_detached
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def build_class_mean(loss, labels, num_classes):
    x = np.asarray(loss, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    out = np.full((num_classes, x.shape[1]), np.nan, dtype=np.float32)
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size:
            out[c] = np.nanmean(x[idx], axis=0, dtype=np.float64).astype(np.float32)
    return out


def gie_batch(phi, G, phi_limit, G_limit):
    """Same GIE/Hill construction used in the frozen LE experiment."""
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    phi_limit = np.asarray(phi_limit, dtype=np.float64)
    G_limit = np.asarray(G_limit, dtype=np.float64)

    R = np.abs(phi - phi_limit[:, None])
    FR = np.abs(G - G_limit[:, None])
    with np.errstate(invalid="ignore"):
        w0 = np.max(R, axis=1)
        w1 = np.max(FR, axis=1)

    mask = (R > EPS) & (FR > EPS) & np.isfinite(R) & np.isfinite(FR)
    n_nonzero = mask.sum(axis=1)
    k_internal = n_nonzero - 1
    base_ok = (
        (k_internal > 4) & np.isfinite(w0) & np.isfinite(w1)
        & (w0 > 0.0) & (w1 > 0.0)
    )
    safe_w0 = np.where(base_ok, w0 + EPS, 1.0)
    safe_w1 = np.where(base_ok, w1 + EPS, 1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_num = np.where(mask, np.log(np.abs(R / safe_w0[:, None])), 0.0)
        log_den = np.where(mask, np.log(np.abs(FR / safe_w1[:, None])), 0.0)

    denom_num = np.sum(log_num, axis=1)
    denom_den = np.sum(log_den, axis=1)
    ok = (
        base_ok & np.isfinite(denom_num) & np.isfinite(denom_den)
        & (np.abs(denom_num) >= EPS) & (np.abs(denom_den) >= EPS)
    )

    hill_num = np.full(phi.shape[0], np.nan, dtype=np.float64)
    hill_den = np.full(phi.shape[0], np.nan, dtype=np.float64)
    hill_num[ok] = -k_internal[ok] / denom_num[ok]
    hill_den[ok] = -k_internal[ok] / denom_den[ok]

    good = ok & np.isfinite(hill_num) & np.isfinite(hill_den) & (np.abs(hill_den) > EPS)
    d = np.full(phi.shape[0], np.nan, dtype=np.float64)
    d[good] = hill_num[good] / hill_den[good]
    return d


def latest_le_next_next(loss_history, labels, K, num_classes):
    """Latest signed LE-GIE using next-sample L in ell_err and GIE.

    Formula is unchanged: ell_hat = ell_err_hat + log(abs(m_GIE_hat)).
    """
    x = np.asarray(loss_history, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    N, T = x.shape
    if T < K + 2:
        return np.full(N, np.nan, dtype=np.float32)

    t = T - 1
    boundary = t - K - 1
    start = t - K
    Gc = build_class_mean(x, labels, num_classes)
    G_limit_cls = np.asarray(Gc[:, t], dtype=np.float64)
    inv_j = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    out = np.full(N, np.nan, dtype=np.float32)

    for lo in range(0, N, BATCH):
        hi = min(lo + BATCH, N)
        lab = labels[lo:hi]
        xb = np.asarray(x[lo:hi, start:t], dtype=np.float64)
        L = np.asarray(x[lo:hi, t], dtype=np.float64)
        r0 = np.abs(np.asarray(x[lo:hi, boundary], dtype=np.float64) - L)
        rj = np.abs(xb - L[:, None])

        valid_err = (
            np.isfinite(L) & np.isfinite(r0) & (r0 != 0.0)
            & np.all(np.isfinite(rj) & (rj != 0.0), axis=1)
        )
        ell_err = np.full(hi - lo, np.nan, dtype=np.float64)
        if np.any(valid_err):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                vals = np.log(rj[valid_err] / r0[valid_err, None]) * inv_j[None, :]
            ell_err[valid_err] = np.mean(vals, axis=1)

        Gb = np.asarray(Gc[lab, start:t], dtype=np.float64)
        m = gie_batch(xb, Gb, L, G_limit_cls[lab])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            le = ell_err + np.log(np.abs(m))

        valid = valid_err & np.isfinite(m) & (m != 0.0) & np.isfinite(le)
        block = np.full(hi - lo, np.nan, dtype=np.float32)
        block[valid] = le[valid].astype(np.float32)
        out[lo:hi] = block
    return out


def within_class_percentile(score, labels, num_classes):
    score = np.asarray(score)
    labels = np.asarray(labels, dtype=np.int64)
    out = np.full(score.shape, np.nan, dtype=np.float32)
    for c in range(num_classes):
        idx = np.flatnonzero((labels == c) & np.isfinite(score))
        if idx.size == 0:
            continue
        vals = score[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(idx.size, dtype=np.float64)
        ranks[order] = np.arange(idx.size, dtype=np.float64)
        pct = np.ones(1) if idx.size == 1 else ranks / (idx.size - 1)
        out[idx] = pct.astype(np.float32)
    return out


def update_rank_ewma(previous, current_percentile, lam):
    previous = np.asarray(previous, dtype=np.float32).copy()
    x = np.asarray(current_percentile, dtype=np.float32)
    finite = np.isfinite(x)
    init = finite & ~np.isfinite(previous)
    previous[init] = x[init]
    upd = finite & np.isfinite(previous)
    previous[upd] = ((1.0 - lam) * previous[upd] + lam * x[upd]).astype(np.float32)
    return previous


def topq_mask(score, q):
    score = np.asarray(score)
    finite_idx = np.flatnonzero(np.isfinite(score))
    selected = np.zeros(score.shape[0], dtype=bool)
    if finite_idx.size == 0:
        return selected
    k = min(max(1, int(round(float(q) * score.size))), finite_idx.size)
    vals = score[finite_idx]
    local = np.argpartition(vals, -k)[-k:]
    selected[finite_idx[local]] = True
    return selected


def correction_strength(epoch, first_detector_epoch, total_epochs, gamma_max):
    """D2L-like gradual schedule from 0 to gamma_max."""
    if epoch < first_detector_epoch:
        return 0.0
    denom = max(total_epochs - first_detector_epoch, 1)
    frac = np.clip((epoch - first_detector_epoch) / denom, 0.0, 1.0)
    return float(gamma_max) * float(frac)


def alpha_from_rank_ewma(rank_ewma, detected, gamma, alpha_floor):
    """Only detected samples are corrected; others keep alpha=1."""
    rank_ewma = np.asarray(rank_ewma, dtype=np.float32)
    detected = np.asarray(detected, dtype=bool)
    alpha = np.ones(rank_ewma.shape[0], dtype=np.float32)
    active = detected & np.isfinite(rank_ewma)
    if np.any(active):
        val = np.exp(-float(gamma) * np.clip(rank_ewma[active], 0.0, 1.0))
        alpha[active] = np.maximum(val, float(alpha_floor)).astype(np.float32)
    return alpha
