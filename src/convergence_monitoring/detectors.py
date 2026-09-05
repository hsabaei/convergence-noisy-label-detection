import math

import numpy as np
from .estimators import EPS


def normalized_local_window(phi, eps: float = EPS):
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 1 or phi.size == 0:
        return None, np.nan

    limit = np.mean(phi[-3:]) if phi.size >= 3 else np.mean(phi)
    R = np.abs(phi - limit)

    w = float(np.max(R)) if R.size else np.nan
    if not np.isfinite(w) or w <= eps:
        return None, np.nan

    z = np.clip(R / w, eps, 1.0)
    u = np.log(z)
    return u.astype(np.float64), w


def fit_diag_gaussian(X: np.ndarray, eps: float = 1e-6):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return None, None

    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=1)
    var = np.maximum(var, eps)
    return mu, var


def gaussian_nll(x: np.ndarray, mu: np.ndarray, var: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    return float(0.5 * np.sum(np.log(var) + ((x - mu) ** 2) / var))


def zscore_by_class(
    score_all: np.ndarray,
    y_all: np.ndarray,
    num_classes: int = 10,
    eps: float = 1e-12,
    start_index: int = 0,
):
    """Standardize each epoch within class, matching the CKL calibration.

    Rows before ``start_index`` remain NaN.  This lets CKL and LRT use the
    same detector start (epoch 40 corresponds to ``start_index=39`` when the
    array row 0 is epoch 1).
    """
    score_all = np.asarray(score_all, dtype=np.float64)
    y_all = np.asarray(y_all)

    if score_all.ndim != 2:
        raise ValueError("score_all must have shape (T, N)")

    T, N = score_all.shape
    if y_all.ndim != 1 or y_all.size != N:
        raise ValueError("y_all must have shape (N,) matching score_all")

    start_index = int(start_index)
    if start_index < 0 or start_index > T:
        raise ValueError(f"start_index must be in [0, {T}], got {start_index}")

    z_all = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(start_index, T):
        x_t = score_all[t]
        for c in range(num_classes):
            idx = (y_all == c)
            vals = x_t[idx]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            mu = np.mean(vals)
            sigma = max(np.std(vals), eps)

            x_tc = x_t[idx]
            finite = np.isfinite(x_tc)
            z_all[t, np.where(idx)[0][finite]] = (x_tc[finite] - mu) / sigma

    return z_all


def z_from_ckl(
    ckl_all: np.ndarray,
    y_all: np.ndarray,
    num_classes: int = 10,
    eps: float = 1e-12,
    start_index: int = 0,
):
    return zscore_by_class(
        ckl_all,
        y_all,
        num_classes=num_classes,
        eps=eps,
        start_index=start_index,
    )


def z_from_lrt_variants(
    lrt_all: np.ndarray,
    y_all: np.ndarray,
    num_classes: int = 10,
    eps: float = 1e-12,
    start_index: int = 0,
):
    """Return the four standardized LRT inputs used by the detectors.

    ``z_abs_lrt`` means z(abs(raw LRT)).
    ``abs_z_lrt`` means abs(z(raw LRT)).
    These are intentionally both returned because they are not equivalent.
    """
    lrt_all = np.asarray(lrt_all, dtype=np.float64)

    z_lrt = zscore_by_class(
        lrt_all,
        y_all,
        num_classes=num_classes,
        eps=eps,
        start_index=start_index,
    )
    z_neg_lrt = zscore_by_class(
        -lrt_all,
        y_all,
        num_classes=num_classes,
        eps=eps,
        start_index=start_index,
    )
    z_abs_lrt = zscore_by_class(
        np.abs(lrt_all),
        y_all,
        num_classes=num_classes,
        eps=eps,
        start_index=start_index,
    )
    abs_z_lrt = np.abs(z_lrt)

    return {
        "z_lrt": z_lrt,
        "z_neg_lrt": z_neg_lrt,
        "z_abs_lrt": z_abs_lrt,
        "abs_z_lrt": abs_z_lrt,
    }


def standardized_detector_inputs(
    ckl_all: np.ndarray,
    lrt_all: np.ndarray,
    y_all: np.ndarray,
    num_classes: int = 10,
    eps: float = 1e-12,
    start_index: int = 0,
):
    """Build CKL and all standardized LRT inputs with one calibration rule."""
    inputs = {
        "z_ckl": z_from_ckl(
            ckl_all,
            y_all,
            num_classes=num_classes,
            eps=eps,
            start_index=start_index,
        )
    }
    inputs.update(
        z_from_lrt_variants(
            lrt_all,
            y_all,
            num_classes=num_classes,
            eps=eps,
            start_index=start_index,
        )
    )
    return inputs


robust_z_from_ckl = z_from_ckl


def estimate_alpha_sup_from_h0(z_all: np.ndarray, tau: float, idx_h0: np.ndarray) -> float:
    I_h0 = z_all[:, idx_h0] >= tau
    alpha_hat = float(np.max(np.mean(I_h0, axis=1)))
    return min(0.999999, max(1e-12, alpha_hat))


def exceedance_from_z(z_all: np.ndarray, tau: float) -> np.ndarray:
    return z_all >= tau


def compute_minrun_m(T: int, alpha: float, delta: float) -> int:
    return int(math.ceil(math.log(T / delta) / (-math.log(alpha))))


def minrun_detector(I: np.ndarray, m: int):
    T, N = I.shape
    runlen = np.zeros((T, N), dtype=np.int32)
    drift = np.zeros((T, N), dtype=bool)

    runlen[0] = I[0].astype(np.int32)
    drift[0] = runlen[0] >= m

    for t in range(1, T):
        runlen[t] = (runlen[t - 1] + 1) * I[t]
        drift[t] = runlen[t] >= m

    first_hit = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        ts = np.where(drift[:, i])[0]
        if ts.size > 0:
            first_hit[i] = int(ts[0])

    return runlen, drift, first_hit


def chernoff_window_k(T: int, ell: int, alpha: float, delta: float):
    mu = ell * alpha
    L = math.log((T - ell + 1) / delta)
    k_real = mu + 0.5 * (L + math.sqrt(L * L + 8.0 * mu * L))
    k = int(math.ceil(k_real))
    k = max(1, min(k, ell))
    return k, mu, L


def forward_window_sums(I: np.ndarray, ell: int) -> np.ndarray:
    T, N = I.shape
    cs = np.cumsum(I.astype(np.int32), axis=0)
    W = cs[ell - 1:].copy()
    if W.shape[0] > 1:
        W[1:] -= cs[:-ell]
    return W


def sliding_window_detector(I: np.ndarray, ell: int, k: int):
    W = forward_window_sums(I, ell)
    hit = W >= k
    any_hit = hit.any(axis=0)

    N = I.shape[1]
    first_t = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        ts = np.where(hit[:, i])[0]
        if ts.size > 0:
            first_t[i] = int(ts[0])

    return W, hit, any_hit, first_t


def ewma_smooth_z(z_all: np.ndarray, lam: float = 0.2) -> np.ndarray:
    T, N = z_all.shape
    z_ewma = np.full((T, N), np.nan, dtype=np.float64)
    z_ewma[0] = z_all[0]

    for t in range(1, T):
        prev = z_ewma[t - 1]
        x = z_all[t]
        finite_x = np.isfinite(x)

        z_ewma[t] = prev

        upd = finite_x & np.isfinite(prev)
        z_ewma[t, upd] = (1.0 - lam) * prev[upd] + lam * x[upd]

        start = finite_x & (~np.isfinite(prev))
        z_ewma[t, start] = x[start]

    return z_ewma


def ewma_detector_same_tau(z_all: np.ndarray, tau: float, lam: float = 0.2):
    z_ewma = ewma_smooth_z(z_all, lam=lam)
    drift = np.isfinite(z_ewma) & (z_ewma >= tau)

    N = z_all.shape[1]
    first_hit = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        ts = np.where(drift[:, i])[0]
        if ts.size > 0:
            first_hit[i] = int(ts[0])

    return z_ewma, drift, first_hit


def binary_auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)

    pos_scores = pos_scores[np.isfinite(pos_scores)]
    neg_scores = neg_scores[np.isfinite(neg_scores)]

    if pos_scores.size == 0 or neg_scores.size == 0:
        return np.nan

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([
        np.ones(pos_scores.size, dtype=np.int64),
        np.zeros(neg_scores.size, dtype=np.int64),
    ])

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    n_pos = pos_scores.size
    n_neg = neg_scores.size
    sum_ranks_pos = ranks[labels == 1].sum()

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pairwise_scores_by_class(
    score_nt: np.ndarray,
    labels: np.ndarray,
    *,
    n_peers: int = 10,
    start_index: int = 0,
    seed: int = 66,
):
    """Compare each sample with same-observed-class peers over time.

    Parameters
    ----------
    score_nt:
        Score array in [N,T] orientation. Larger values must mean
        "more noisy-like".
    labels:
        Observed labels, shape [N].
    n_peers:
        Number of same-class peers used per sample per epoch.
    start_index:
        Zero-based epoch column where pairwise monitoring begins.
    seed:
        Random seed controlling the per-epoch class permutations.

    Returns
    -------
    dict with:
        instantaneous_score [N,T]
            Fraction of valid pairwise comparisons won at that epoch.
        cumulative_score [N,T]
            Fraction of all valid pairwise comparisons won from start_index
            through the current epoch.
        valid_comparisons [N,T]
            Number of usable pairwise comparisons at each epoch.
        cumulative_comparisons [N,T]
            Cumulative number of usable comparisons.
        cumulative_wins [N,T]
            Cumulative pairwise win mass. Ties contribute 0.5.

    Notes
    -----
    At every epoch and within every class, members are randomly permuted.
    Each sample is compared with ``n_peers`` distinct cyclic offsets in that
    permutation. This gives unique, self-excluding peers for that sample at
    that epoch while avoiding an N x N comparison matrix.

    A win is score_i > score_j. A tie contributes 0.5.
    """
    score = np.asarray(score_nt, dtype=np.float64)
    labels = np.asarray(labels)

    if score.ndim != 2:
        raise ValueError(
            f"score_nt must have shape [N,T], got {score.shape}."
        )

    N, T = score.shape

    if labels.shape != (N,):
        raise ValueError(
            f"labels must have shape ({N},), got {labels.shape}."
        )

    n_peers = int(n_peers)
    start_index = int(start_index)

    if n_peers < 1:
        raise ValueError("n_peers must be positive.")
    if start_index < 0 or start_index >= T:
        raise ValueError(
            f"start_index must be in [0, {T-1}], got {start_index}."
        )

    classes = np.unique(labels)

    for c in classes:
        nc = int(np.sum(labels == c))
        if nc <= n_peers:
            raise ValueError(
                f"Class {c} has {nc} samples, but n_peers={n_peers}; "
                "need class size > n_peers."
            )

    instantaneous = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )
    cumulative = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )
    valid_comparisons = np.zeros(
        (N, T),
        dtype=np.int16,
    )
    cumulative_comparisons = np.zeros(
        (N, T),
        dtype=np.int32,
    )
    cumulative_wins = np.zeros(
        (N, T),
        dtype=np.float64,
    )

    running_wins = np.zeros(N, dtype=np.float64)
    running_count = np.zeros(N, dtype=np.int32)

    rng = np.random.default_rng(int(seed))

    class_members = {
        c: np.flatnonzero(labels == c)
        for c in classes
    }

    for t in range(start_index, T):

        wins_t = np.zeros(N, dtype=np.float64)
        count_t = np.zeros(N, dtype=np.int16)

        for c in classes:

            members = class_members[c]
            nc = members.size

            # Fresh peer assignment at every epoch.
            perm = rng.permutation(members)

            # Distinct offsets ensure no repeated peer for the same sample
            # during this epoch, and offset 0 is excluded to avoid self-pairs.
            offsets = rng.choice(
                np.arange(1, nc, dtype=np.int64),
                size=n_peers,
                replace=False,
            )

            x_i = score[perm, t]

            for offset in offsets:
                peers = np.roll(perm, int(offset))
                x_j = score[peers, t]

                valid = (
                    np.isfinite(x_i)
                    & np.isfinite(x_j)
                )

                if not np.any(valid):
                    continue

                ii = perm[valid]
                vi = x_i[valid]
                vj = x_j[valid]

                win_mass = np.where(
                    vi > vj,
                    1.0,
                    np.where(vi == vj, 0.5, 0.0),
                )

                wins_t[ii] += win_mass
                count_t[ii] += 1

        usable = count_t > 0

        instantaneous[usable, t] = (
            wins_t[usable]
            / count_t[usable]
        )

        running_wins += wins_t
        running_count += count_t.astype(np.int32)

        cumulative_wins[:, t] = running_wins
        cumulative_comparisons[:, t] = running_count
        valid_comparisons[:, t] = count_t

        cumulative_usable = running_count > 0
        cumulative[cumulative_usable, t] = (
            running_wins[cumulative_usable]
            / running_count[cumulative_usable]
        )

    return {
        "instantaneous_score": instantaneous,
        "cumulative_score": cumulative,
        "valid_comparisons": valid_comparisons,
        "cumulative_comparisons": cumulative_comparisons,
        "cumulative_wins": cumulative_wins,
        "n_peers": int(n_peers),
        "start_index": int(start_index),
        "seed": int(seed),
    }
