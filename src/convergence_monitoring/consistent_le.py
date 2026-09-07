"""
Consistent-limit LE-GIE estimator.

This module is ADDITIVE: it does not replace or alter the earlier
proposed_le.py / estimators.py implementations.

The theoretical estimator is

    ell_hat = ell_err_hat + log(abs(m_GIE_hat)).

A single sample-limit rule L_hat_i(t) is used consistently in
    1) ell_err_hat, and
    2) the sample-side residuals inside m_GIE_hat.

The class-reference trajectory has its own corresponding limit, computed
with the SAME rule.

Frozen final setting used by the final CIFAR-10 experiment:
    K = 40
    reference_method = "mean"
    limit_method = "last3_mean"
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

EPS = 1e-7


def build_class_reference_trajectory(
    loss_traj: np.ndarray,
    observed_label: np.ndarray,
    *,
    num_classes: int = 10,
    reference_method: str = "mean",
    trim_fraction: float = 0.10,
) -> np.ndarray:
    """Build the observed-class reference trajectory G_i(t)."""
    x = np.asarray(loss_traj, dtype=np.float64)
    labels = np.asarray(observed_label, dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(f"loss_traj must have shape [N,T], got {x.shape}.")

    N, T = x.shape
    if labels.shape != (N,):
        raise ValueError(
            f"observed_label must have shape ({N},), got {labels.shape}."
        )

    method = str(reference_method).lower()
    allowed = {"mean", "leave_one_out_mean", "median", "trimmed_mean"}
    if method not in allowed:
        raise ValueError(
            f"reference_method must be one of {sorted(allowed)}, got {reference_method!r}."
        )

    trim_fraction = float(trim_fraction)
    if method == "trimmed_mean" and not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0,0.5).")

    G = np.full((N, T), np.nan, dtype=np.float64)

    for c in range(int(num_classes)):
        members = np.flatnonzero(labels == c)
        if members.size == 0:
            continue

        sub = x[members]

        if method == "mean":
            with np.errstate(invalid="ignore"):
                ref = np.nanmean(sub, axis=0)
            G[members] = ref[None, :]
            continue

        if method == "median":
            with np.errstate(invalid="ignore"):
                ref = np.nanmedian(sub, axis=0)
            G[members] = ref[None, :]
            continue

        if method == "leave_one_out_mean":
            finite = np.isfinite(sub)
            sums = np.nansum(sub, axis=0)
            counts = finite.sum(axis=0)

            numer = sums[None, :] - np.where(finite, sub, 0.0)
            denom = counts[None, :] - finite.astype(np.int64)

            ref = np.full_like(sub, np.nan, dtype=np.float64)
            np.divide(numer, denom, out=ref, where=(denom > 0))
            G[members] = ref
            continue

        ref = np.full(T, np.nan, dtype=np.float64)
        for t in range(T):
            vals = sub[:, t]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            vals = np.sort(vals)
            n_trim = int(np.floor(trim_fraction * vals.size))
            if n_trim > 0 and 2 * n_trim < vals.size:
                vals = vals[n_trim:vals.size - n_trim]
            ref[t] = float(np.mean(vals))

        G[members] = ref[None, :]

    return G


def guarded_aitken_batch(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Guarded Aitken Delta^2 with fallback to the latest observation c."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    d0 = b - a
    d1 = c - b
    denominator = d1 - d0

    scale = np.maximum(
        np.maximum(np.abs(d0), np.abs(d1)),
        np.finfo(np.float64).tiny,
    )
    tol = np.sqrt(np.finfo(np.float64).eps) * scale

    use_aitken = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(c)
        & np.isfinite(denominator)
        & (np.abs(denominator) > tol)
    )

    L = np.full_like(a, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        L[use_aitken] = (
            a[use_aitken]
            - d0[use_aitken] ** 2 / denominator[use_aitken]
        )

    fallback = (~use_aitken) | (~np.isfinite(L))
    L[fallback] = c[fallback]

    return L, fallback


def estimate_limit_at_t(
    traj: np.ndarray,
    t: int,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Row-wise limit estimate using observations through column t."""
    traj = np.asarray(traj, dtype=np.float64)
    method = str(method).lower()

    if t < 2:
        raise ValueError("Need at least three observations.")

    if method == "last3_mean":
        return (
            np.mean(traj[:, t - 2:t + 1], axis=1),
            np.zeros(traj.shape[0], dtype=bool),
        )

    if method == "next":
        return traj[:, t].copy(), np.zeros(traj.shape[0], dtype=bool)

    if method == "aitken_guarded":
        return guarded_aitken_batch(
            traj[:, t - 2],
            traj[:, t - 1],
            traj[:, t],
        )

    raise ValueError(
        "limit_method must be one of "
        "{'last3_mean', 'next', 'aitken_guarded'}."
    )


def vectorized_gie_window_with_limits(
    phi: np.ndarray,
    G: np.ndarray,
    phi_limit: np.ndarray,
    G_limit: np.ndarray,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """Existing row-wise GIE/Hill formula with explicit consistent limits."""
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    phi_limit = np.asarray(phi_limit, dtype=np.float64)
    G_limit = np.asarray(G_limit, dtype=np.float64)

    if phi.shape != G.shape or phi.ndim != 2:
        raise ValueError("phi and G must have matching shape [N,K].")

    N, _ = phi.shape
    if phi_limit.shape != (N,) or G_limit.shape != (N,):
        raise ValueError("phi_limit and G_limit must each have shape [N].")

    R = np.abs(phi - phi_limit[:, None])
    FR = np.abs(G - G_limit[:, None])

    with np.errstate(invalid="ignore"):
        w0 = np.max(R, axis=1)
        w1 = np.max(FR, axis=1)

    d = np.full(N, np.nan, dtype=np.float64)

    mask = (
        (R > eps)
        & (FR > eps)
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

    safe_w0 = np.where(base_ok, w0 + eps, 1.0)
    safe_w1 = np.where(base_ok, w1 + eps, 1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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
        & (np.abs(denom_num) >= eps)
        & (np.abs(denom_den) >= eps)
    )

    hill_num = np.full(N, np.nan, dtype=np.float64)
    hill_den = np.full(N, np.nan, dtype=np.float64)

    hill_num[ok] = -k_internal[ok] / denom_num[ok]
    hill_den[ok] = -k_internal[ok] / denom_den[ok]

    good = (
        ok
        & np.isfinite(hill_num)
        & np.isfinite(hill_den)
        & (np.abs(hill_den) > eps)
    )

    d[good] = hill_num[good] / hill_den[good]
    return d


def rolling_common_limit_le_gie_batch(
    loss_traj: np.ndarray,
    observed_label: np.ndarray,
    *,
    K: int,
    num_classes: int = 10,
    reference_method: str = "mean",
    trim_fraction: float = 0.10,
    limit_method: str = "last3_mean",
) -> Dict[str, np.ndarray]:
    """Compute rolling LE-GIE with one consistent sample-limit rule.

    At output column t:

        boundary        = x[t-K-1]
        K-point tail    = x[t-K : t]      # x[t-K], ..., x[t-1]
        latest for Lhat = x[t]

    The SAME L_hat_i(t) is used in ell_err and sample-side GIE residuals.
    """
    x = np.asarray(loss_traj, dtype=np.float64)
    labels = np.asarray(observed_label, dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(f"loss_traj must be [N,T], got {x.shape}.")

    N, T = x.shape
    if labels.shape != (N,):
        raise ValueError(f"observed_label must have shape ({N},).")

    K = int(K)
    if K < 6:
        raise ValueError("K must be at least 6.")
    if T < K + 2:
        raise ValueError(f"Need at least K+2={K+2} observations; got {T}.")

    G = build_class_reference_trajectory(
        x,
        labels,
        num_classes=num_classes,
        reference_method=reference_method,
        trim_fraction=trim_fraction,
    )

    ell_err_traj = np.full((N, T), np.nan, dtype=np.float64)
    id_gie_traj = np.full((N, T), np.nan, dtype=np.float64)
    lambda_traj = np.full((N, T), np.nan, dtype=np.float64)
    sample_limit_traj = np.full((N, T), np.nan, dtype=np.float64)
    reference_limit_traj = np.full((N, T), np.nan, dtype=np.float64)
    valid_traj = np.zeros((N, T), dtype=bool)
    limit_fallback_traj = np.zeros((N, T), dtype=bool)

    inv_j = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    first_col = K + 1

    for t in range(first_col, T):
        boundary = t - K - 1
        tail_start = t - K

        sample_limit, fallback_sample = estimate_limit_at_t(
            x, t, limit_method
        )
        reference_limit, fallback_reference = estimate_limit_at_t(
            G, t, limit_method
        )

        sample_limit_traj[:, t] = sample_limit
        reference_limit_traj[:, t] = reference_limit
        limit_fallback_traj[:, t] = (
            fallback_sample | fallback_reference
        )

        # Error component with the same sample L_hat.
        r0 = np.abs(x[:, boundary] - sample_limit)
        tail = x[:, tail_start:t]
        rj = np.abs(tail - sample_limit[:, None])

        valid_ell = (
            np.isfinite(sample_limit)
            & np.isfinite(r0)
            & (r0 != 0.0)
            & np.all(
                np.isfinite(rj) & (rj != 0.0),
                axis=1,
            )
        )

        ell_err = np.full(N, np.nan, dtype=np.float64)

        if np.any(valid_ell):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ell_values = (
                    np.log(rj[valid_ell] / r0[valid_ell, None])
                    * inv_j[None, :]
                )

            ell_err[valid_ell] = np.mean(ell_values, axis=1)

        # GIE with the same sample L_hat and corresponding class-reference Lhat.
        G_tail = G[:, tail_start:t]
        id_gie = vectorized_gie_window_with_limits(
            tail,
            G_tail,
            sample_limit,
            reference_limit,
        )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lambda_hat = ell_err + np.log(np.abs(id_gie))

        valid = (
            valid_ell
            & np.isfinite(id_gie)
            & (id_gie != 0.0)
            & np.isfinite(lambda_hat)
        )

        ell_err_traj[valid, t] = ell_err[valid]
        id_gie_traj[valid, t] = id_gie[valid]
        lambda_traj[valid, t] = lambda_hat[valid]
        valid_traj[valid, t] = True

    return {
        "lambda_traj": lambda_traj,
        "le_gie_traj": lambda_traj,
        "ell_err_traj": ell_err_traj,
        "id_gie_traj": id_gie_traj,
        "sample_limit_traj": sample_limit_traj,
        "reference_limit_traj": reference_limit_traj,
        "valid_traj": valid_traj,
        "limit_fallback_traj": limit_fallback_traj,
        "first_available_column": np.asarray(first_col, dtype=np.int64),
        "first_available_epoch": np.asarray(first_col + 1, dtype=np.int64),
        "K": np.asarray(K, dtype=np.int64),
    }
