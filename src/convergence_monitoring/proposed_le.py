import time

import numpy as np
import mpmath as mp


# ============================================================
# Hill estimator -- float64
# ============================================================

def _hill_estimator_float(V):

    V = np.asarray(
        V,
        dtype=float,
    )

    V = V[
        np.isfinite(V)
        & (V > 0.0)
    ]

    k = V.size

    if k < 2:
        return np.nan

    w = np.max(V)

    if (
        not np.isfinite(w)
        or w <= 0.0
    ):
        return np.nan

    denominator = np.sum(
        np.log(V / w)
    )

    if (
        not np.isfinite(denominator)
        or denominator == 0.0
    ):
        return np.nan

    hill = (
        -k / denominator
    )

    if (
        not np.isfinite(hill)
        or hill <= 0.0
    ):
        return np.nan

    return float(hill)


# ============================================================
# Hill estimator -- arbitrary precision
# ============================================================

def _hill_estimator_mp(V):

    values = [
        v
        for v in V
        if mp.isfinite(v)
        and v > 0
    ]

    k = len(values)

    if k < 2:
        return mp.nan

    w = max(values)

    if (
        not mp.isfinite(w)
        or w <= 0
    ):
        return mp.nan

    log_ratios = [
        mp.log(v / w)
        for v in values
    ]

    denominator = mp.fsum(
        log_ratios
    )

    if (
        not mp.isfinite(denominator)
        or denominator == 0
    ):
        return mp.nan

    hill = (
        -mp.mpf(k)
        / denominator
    )

    if (
        not mp.isfinite(hill)
        or hill <= 0
    ):
        return mp.nan

    return hill


# ============================================================
# Float64 estimator
# ============================================================

def _proposed_float(
    x,
    K,
    end_index,
    limit_method,
    L_hat_override,
):

    start_time = time.perf_counter()

    x = np.asarray(
        x,
        dtype=float,
    )

    if x.ndim != 1:
        raise ValueError(
            "x must be one-dimensional."
        )

    if end_index is None:
        n = len(x) - 2
    else:
        n = int(end_index)

    if n < K:
        raise ValueError(
            f"end_index={n} is too early for K={K}."
        )

    if n + 1 >= len(x):
        raise ValueError(
            "x[n+1] is unavailable."
        )

    t0 = n - K

    L_hat = np.nan
    used_limit_method = limit_method

    def failure(
        reason,
        ell_err_hat=np.nan,
        id_fie_hat=np.nan,
    ):

        return {
            "lambda_hat": np.nan,
            "ell_err_hat": ell_err_hat,
            "id_fie_hat": id_fie_hat,
            "L_hat": L_hat,
            "limit_method": used_limit_method,
            "end_index": int(n),
            "window_start": int(t0),
            "window_end": int(n),
            "latest_observation_index": int(n + 1),
            "K": int(K),
            "runtime_sec":
                time.perf_counter() - start_time,
            "valid": False,
            "failure_reason": reason,
        }

    # ========================================================
    # Limit estimate
    # ========================================================

    if L_hat_override is not None:

        L_hat = float(
            L_hat_override
        )

        used_limit_method = "oracle"

    elif limit_method == "next":

        L_hat = float(
            x[n + 1]
        )

    elif limit_method in ("aitken", "aitken_guarded"):

        d0 = x[n] - x[n - 1]
        d1 = x[n + 1] - x[n]
        denominator = d1 - d0

        if limit_method == "aitken":
            if denominator == 0.0:
                return failure(
                    "Aitken denominator is zero."
                )
            use_aitken = True
        else:
            # Scale-aware guard for a nearly-zero second difference.
            scale = max(abs(d0), abs(d1), np.finfo(float).tiny)
            tol = np.sqrt(np.finfo(float).eps) * scale
            use_aitken = (
                np.isfinite(denominator)
                and abs(denominator) > tol
            )

        if use_aitken:
            L_hat = (
                x[n - 1]
                - (d0 ** 2) / denominator
            )
        else:
            # Same information horizon as "next": x[n+1] is available.
            L_hat = float(x[n + 1])
            used_limit_method = "aitken_guarded_fallback_next"

    else:

        raise ValueError(
            "limit_method must be "
            "'next', 'aitken', or 'aitken_guarded'."
        )

    if not np.isfinite(L_hat):
        return failure(
            "Limit estimate is nonfinite."
        )

    # ========================================================
    # Error component
    # ========================================================

    r0 = abs(
        x[t0] - L_hat
    )

    if (
        not np.isfinite(r0)
        or r0 == 0.0
    ):
        return failure(
            "Initial residual is zero or nonfinite."
        )

    ell_values = []

    for j in range(1, K + 1):

        rj = abs(
            x[t0 + j]
            - L_hat
        )

        if (
            not np.isfinite(rj)
            or rj == 0.0
        ):
            return failure(
                f"Residual invalid at j={j}."
            )

        ell_j = (
            np.log(rj / r0)
            / j
        )

        ell_values.append(
            ell_j
        )

    ell_err_hat = float(
        np.mean(ell_values)
    )

    # ========================================================
    # FIE
    # ========================================================

    R = np.abs(
        x[t0:n]
        - L_hat
    )

    FR = np.abs(
        x[t0 + 1:n + 1]
        - L_hat
    )

    hill_R = (
        _hill_estimator_float(R)
    )

    hill_FR = (
        _hill_estimator_float(FR)
    )

    if not np.isfinite(hill_R):
        return failure(
            "Hill(R) is invalid.",
            ell_err_hat,
        )

    if not np.isfinite(hill_FR):
        return failure(
            "Hill(FR) is invalid.",
            ell_err_hat,
        )

    id_fie_hat = (
        hill_R / hill_FR
    )

    if (
        not np.isfinite(id_fie_hat)
        or id_fie_hat == 0.0
    ):
        return failure(
            "FIE estimate is zero or nonfinite.",
            ell_err_hat,
            id_fie_hat,
        )

    # ========================================================
    # Final LE
    # ========================================================

    # Theory requires the magnitude of the asymptotic LID contribution:
    #     ell_rel = log |ID_F^*|
    lambda_hat = (
        ell_err_hat
        + np.log(abs(id_fie_hat))
    )

    return {
        "lambda_hat":
            float(lambda_hat),

        "ell_err_hat":
            float(ell_err_hat),

        "id_fie_hat":
            float(id_fie_hat),

        "L_hat":
            float(L_hat),

        "limit_method":
            used_limit_method,

        "end_index":
            int(n),

        "window_start":
            int(t0),

        "window_end":
            int(n),

        "latest_observation_index":
            int(n + 1),

        "K":
            int(K),

        "runtime_sec":
            float(
                time.perf_counter()
                - start_time
            ),

        "valid":
            True,

        "failure_reason":
            "",
    }


# ============================================================
# High-precision estimator
# ============================================================

def _proposed_mp(
    x,
    K,
    end_index,
    limit_method,
    L_hat_override,
):

    start_time = time.perf_counter()

    # IMPORTANT:
    # Do not convert through float.
    x = [
        value
        if isinstance(value, mp.mpf)
        else mp.mpf(str(value))
        for value in x
    ]

    if end_index is None:
        n = len(x) - 2
    else:
        n = int(end_index)

    if n < K:
        raise ValueError(
            f"end_index={n} is too early for K={K}."
        )

    if n + 1 >= len(x):
        raise ValueError(
            "x[n+1] is unavailable."
        )

    t0 = n - K

    L_hat = mp.nan
    used_limit_method = limit_method

    def failure(
        reason,
        ell_err_hat=mp.nan,
        id_fie_hat=mp.nan,
    ):

        return {
            "lambda_hat": mp.nan,
            "ell_err_hat": ell_err_hat,
            "id_fie_hat": id_fie_hat,
            "L_hat": L_hat,
            "limit_method": used_limit_method,
            "end_index": int(n),
            "window_start": int(t0),
            "window_end": int(n),
            "latest_observation_index": int(n + 1),
            "K": int(K),
            "runtime_sec":
                time.perf_counter() - start_time,
            "valid": False,
            "failure_reason": reason,
        }

    # ========================================================
    # Limit estimate
    # ========================================================

    if L_hat_override is not None:

        L_hat = (
            L_hat_override
            if isinstance(
                L_hat_override,
                mp.mpf,
            )
            else mp.mpf(
                str(L_hat_override)
            )
        )

        used_limit_method = "oracle"

    elif limit_method == "next":

        L_hat = x[n + 1]

    elif limit_method in ("aitken", "aitken_guarded"):

        d0 = x[n] - x[n - 1]
        d1 = x[n + 1] - x[n]
        denominator = d1 - d0

        if limit_method == "aitken":
            if denominator == 0:
                return failure(
                    "Aitken denominator is zero."
                )
            use_aitken = True
        else:
            scale = max(mp.fabs(d0), mp.fabs(d1), mp.mpf("1e-100"))
            tol = mp.sqrt(mp.eps) * scale
            use_aitken = (
                mp.isfinite(denominator)
                and mp.fabs(denominator) > tol
            )

        if use_aitken:
            L_hat = (
                x[n - 1]
                - (d0 ** 2) / denominator
            )
        else:
            L_hat = x[n + 1]
            used_limit_method = "aitken_guarded_fallback_next"

    else:

        raise ValueError(
            "limit_method must be "
            "'next', 'aitken', or 'aitken_guarded'."
        )

    if not mp.isfinite(L_hat):
        return failure(
            "Limit estimate is nonfinite."
        )

    # ========================================================
    # Error-decay component
    # ========================================================

    r0 = mp.fabs(
        x[t0] - L_hat
    )

    if (
        not mp.isfinite(r0)
        or r0 == 0
    ):
        return failure(
            "Initial residual is zero or nonfinite."
        )

    ell_values = []

    for j in range(1, K + 1):

        rj = mp.fabs(
            x[t0 + j]
            - L_hat
        )

        if (
            not mp.isfinite(rj)
            or rj == 0
        ):
            return failure(
                f"Residual invalid at j={j}."
            )

        ell_j = (
            mp.log(rj / r0)
            / mp.mpf(j)
        )

        ell_values.append(
            ell_j
        )

    ell_err_hat = (
        mp.fsum(ell_values)
        / mp.mpf(K)
    )

    # ========================================================
    # FIE component
    # ========================================================

    R = [
        mp.fabs(
            x[i] - L_hat
        )
        for i in range(
            t0,
            n,
        )
    ]

    FR = [
        mp.fabs(
            x[i] - L_hat
        )
        for i in range(
            t0 + 1,
            n + 1,
        )
    ]

    if len(R) != K:
        return failure(
            "R has incorrect length.",
            ell_err_hat,
        )

    if len(FR) != K:
        return failure(
            "FR has incorrect length.",
            ell_err_hat,
        )

    hill_R = (
        _hill_estimator_mp(R)
    )

    hill_FR = (
        _hill_estimator_mp(FR)
    )

    if not mp.isfinite(hill_R):
        return failure(
            "Hill(R) is invalid.",
            ell_err_hat,
        )

    if not mp.isfinite(hill_FR):
        return failure(
            "Hill(FR) is invalid.",
            ell_err_hat,
        )

    id_fie_hat = (
        hill_R
        / hill_FR
    )

    if (
        not mp.isfinite(id_fie_hat)
        or id_fie_hat == 0
    ):
        return failure(
            "FIE estimate is zero or nonfinite.",
            ell_err_hat,
            id_fie_hat,
        )

    # ========================================================
    # Final Lyapunov estimate
    # ========================================================

    # Theory requires the magnitude of the asymptotic LID contribution:
    #     ell_rel = log |ID_F^*|
    lambda_hat = (
        ell_err_hat
        + mp.log(mp.fabs(id_fie_hat))
    )

    if not mp.isfinite(lambda_hat):
        return failure(
            "Final LE is nonfinite.",
            ell_err_hat,
            id_fie_hat,
        )

    return {
        "lambda_hat":
            lambda_hat,

        "ell_err_hat":
            ell_err_hat,

        "id_fie_hat":
            id_fie_hat,

        "L_hat":
            L_hat,

        "limit_method":
            used_limit_method,

        "end_index":
            int(n),

        "window_start":
            int(t0),

        "window_end":
            int(n),

        "latest_observation_index":
            int(n + 1),

        "K":
            int(K),

        "runtime_sec":
            float(
                time.perf_counter()
                - start_time
            ),

        "valid":
            True,

        "failure_reason":
            "",
    }


# ============================================================
# Public estimator
# ============================================================

def proposed_le_estimator(
    x,
    K,
    end_index=None,
    limit_method="next",
    L_hat_override=None,
    numeric_backend="float64",
):
    """
    Proposed Lyapunov exponent estimator.

    numeric_backend:
        "float64"
        "mpmath"
    """

    if K < 1:
        raise ValueError(
            "K must be positive."
        )

    if numeric_backend == "float64":

        return _proposed_float(
            x=x,
            K=K,
            end_index=end_index,
            limit_method=limit_method,
            L_hat_override=L_hat_override,
        )

    if numeric_backend == "mpmath":

        return _proposed_mp(
            x=x,
            K=K,
            end_index=end_index,
            limit_method=limit_method,
            L_hat_override=L_hat_override,
        )

    raise ValueError(
        "numeric_backend must be "
        "'float64' or 'mpmath'."
    )


# ============================================================
# Rolling estimator
# ============================================================

def rolling_proposed_le_estimator(
    x,
    K,
    limit_method="next",
    numeric_backend="float64",
):
    """
    Compute all rolling estimates.

    For endpoint n:

        window = x_{n-K}, ..., x_n

    and x_{n+1} is available for the
    practical limit estimate.

    First endpoint:

        n = K

    Last endpoint:

        n = len(x) - 2
    """

    first_end = K
    last_end = len(x) - 2

    if first_end > last_end:
        raise ValueError(
            "Trajectory is too short "
            "for the requested K."
        )

    results = []

    for n in range(
        first_end,
        last_end + 1,
    ):

        result = proposed_le_estimator(
            x=x,
            K=K,
            end_index=n,
            limit_method=limit_method,
            numeric_backend=numeric_backend,
        )

        results.append(
            result
        )

    return results


# ============================================================
# Vectorized rolling estimator for many trajectories
# ============================================================

def _rowwise_hill_float(V):
    """Row-wise Hill estimator for a 2-D float64 array."""
    V = np.asarray(V, dtype=np.float64)

    if V.ndim != 2:
        raise ValueError("V must be two-dimensional.")

    valid = np.isfinite(V) & (V > 0.0)
    k = valid.sum(axis=1).astype(np.float64)

    masked = np.where(valid, V, -np.inf)
    w = np.max(masked, axis=1)

    out = np.full(V.shape[0], np.nan, dtype=np.float64)

    base_ok = (
        (k >= 2.0)
        & np.isfinite(w)
        & (w > 0.0)
    )

    if not np.any(base_ok):
        return out

    safe_w = np.where(base_ok, w, 1.0)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        logs = np.where(
            valid,
            np.log(V / safe_w[:, None]),
            0.0,
        )

    denominator = np.sum(logs, axis=1)

    ok = (
        base_ok
        & np.isfinite(denominator)
        & (denominator != 0.0)
    )

    hill = np.full(V.shape[0], np.nan, dtype=np.float64)
    hill[ok] = -k[ok] / denominator[ok]

    good = (
        ok
        & np.isfinite(hill)
        & (hill > 0.0)
    )

    out[good] = hill[good]
    return out


def rolling_proposed_le_batch(
    x,
    K,
    limit_method="next",
):
    """Vectorized rolling proposed-LE estimator for many trajectories.

    Parameters
    ----------
    x : array-like, shape (N, T)
        N scalar trajectories observed over T epochs.
    K : int
        Estimation-window parameter used by ``proposed_le_estimator``.
    limit_method : {"next", "aitken", "aitken_guarded"}
        Practical fixed-point estimate.

    Returns
    -------
    dict
        Arrays use shape (N, T).  Observation column ``t`` contains the
        estimate whose endpoint is ``n=t-1``; therefore the first available
        one-based epoch is ``K+2``.

    Notes
    -----
    This is the batched float64 implementation of the same estimator defined
    in this module.  In particular,

        lambda_hat = ell_err_hat + log(abs(id_fie_hat)).

    ``aitken_guarded`` uses Aitken Delta^2 when its second difference is
    numerically resolved, otherwise it falls back to the same ``x[n+1]``
    limit used by ``limit_method="next"``.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(
            f"x must have shape (N,T); got {x.shape}."
        )

    if K < 1:
        raise ValueError("K must be positive.")

    if limit_method not in (
        "next",
        "aitken",
        "aitken_guarded",
    ):
        raise ValueError(
            "limit_method must be 'next', 'aitken', "
            "or 'aitken_guarded'."
        )

    N, T = x.shape

    if T < K + 2:
        raise ValueError(
            f"Need at least K+2={K+2} observations; found T={T}."
        )

    lambda_traj = np.full((N, T), np.nan, dtype=np.float64)
    ell_err_traj = np.full((N, T), np.nan, dtype=np.float64)
    id_fie_traj = np.full((N, T), np.nan, dtype=np.float64)
    limit_traj = np.full((N, T), np.nan, dtype=np.float64)
    valid_traj = np.zeros((N, T), dtype=bool)
    aitken_fallback_traj = np.zeros((N, T), dtype=bool)

    inv_j = (
        1.0
        / np.arange(
            1,
            K + 1,
            dtype=np.float64,
        )
    )

    # Column t is the latest observation x[n+1], hence n=t-1.
    for t in range(K + 1, T):

        n = t - 1
        t0 = n - K

        if limit_method == "next":

            L_hat = x[:, t].copy()

        else:

            d0 = x[:, t - 1] - x[:, t - 2]
            d1 = x[:, t] - x[:, t - 1]
            denominator = d1 - d0

            L_hat = np.full(
                N,
                np.nan,
                dtype=np.float64,
            )

            if limit_method == "aitken":

                use_aitken = (
                    np.isfinite(denominator)
                    & (denominator != 0.0)
                )

            else:

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
                    np.isfinite(denominator)
                    & (np.abs(denominator) > tol)
                )

            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
            ):
                L_hat[use_aitken] = (
                    x[use_aitken, t - 2]
                    - (
                        d0[use_aitken] ** 2
                        / denominator[use_aitken]
                    )
                )

            if limit_method == "aitken_guarded":

                fallback = (
                    ~use_aitken
                    | ~np.isfinite(L_hat)
                )

                L_hat[fallback] = x[fallback, t]
                aitken_fallback_traj[fallback, t] = True

        limit_traj[:, t] = L_hat

        # ----------------------------------------------------
        # Error-decay component
        # ----------------------------------------------------

        r0 = np.abs(
            x[:, t0]
            - L_hat
        )

        rj = np.abs(
            x[:, t0 + 1:t]
            - L_hat[:, None]
        )

        valid_ell = (
            np.isfinite(L_hat)
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
        # FIE component
        # ----------------------------------------------------

        R = np.abs(
            x[:, t0:n]
            - L_hat[:, None]
        )

        FR = np.abs(
            x[:, t0 + 1:n + 1]
            - L_hat[:, None]
        )

        hill_R = _rowwise_hill_float(R)
        hill_FR = _rowwise_hill_float(FR)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            id_fie = hill_R / hill_FR

            # Corollary/theory:
            #     ell_rel = log |ID_F^*|
            lambda_hat = (
                ell_err
                + np.log(np.abs(id_fie))
            )

        valid = (
            valid_ell
            & np.isfinite(hill_R)
            & np.isfinite(hill_FR)
            & np.isfinite(id_fie)
            & (id_fie != 0.0)
            & np.isfinite(lambda_hat)
        )

        ell_err_traj[valid, t] = ell_err[valid]
        id_fie_traj[valid, t] = id_fie[valid]
        lambda_traj[valid, t] = lambda_hat[valid]
        valid_traj[valid, t] = True

    return {
        "lambda_traj": lambda_traj,
        "ell_err_traj": ell_err_traj,
        "id_fie_traj": id_fie_traj,
        "limit_traj": limit_traj,
        "valid_traj": valid_traj,
        "aitken_fallback_traj": aitken_fallback_traj,
        "first_available_column": K + 1,
        "first_available_epoch": K + 2,
        "K": int(K),
        "limit_method": limit_method,
    }
