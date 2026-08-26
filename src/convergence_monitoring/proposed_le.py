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

    elif limit_method == "aitken":

        denominator = (
            x[n + 1]
            - 2.0 * x[n]
            + x[n - 1]
        )

        if denominator == 0.0:
            return failure(
                "Aitken denominator is zero."
            )

        L_hat = (
            x[n - 1]
            -
            (
                (x[n] - x[n - 1]) ** 2
                / denominator
            )
        )

    else:

        raise ValueError(
            "limit_method must be "
            "'next' or 'aitken'."
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
        or id_fie_hat <= 0
    ):
        return failure(
            "FIE estimate is invalid.",
            ell_err_hat,
            id_fie_hat,
        )

    # ========================================================
    # Final LE
    # ========================================================

    lambda_hat = (
        ell_err_hat
        + np.log(id_fie_hat)
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

    elif limit_method == "aitken":

        denominator = (
            x[n + 1]
            - 2 * x[n]
            + x[n - 1]
        )

        if denominator == 0:
            return failure(
                "Aitken denominator is zero."
            )

        L_hat = (
            x[n - 1]
            -
            (
                (x[n] - x[n - 1]) ** 2
                / denominator
            )
        )

    else:

        raise ValueError(
            "limit_method must be "
            "'next' or 'aitken'."
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
        or id_fie_hat <= 0
    ):
        return failure(
            "FIE estimate is invalid.",
            ell_err_hat,
            id_fie_hat,
        )

    # ========================================================
    # Final Lyapunov estimate
    # ========================================================

    lambda_hat = (
        ell_err_hat
        + mp.log(id_fie_hat)
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