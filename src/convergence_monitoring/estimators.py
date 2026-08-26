import math
from typing import Dict, Optional, Tuple

import numpy as np

EPS = 1e-7


class LIDEstimators:
    def __init__(self, device="cpu"):
        self.device = device

    def compute_GIE_LID(self, phi, G, phi_limit=None, G_limit=None):
        epsilon = 1e-7

        phi = np.asarray(phi, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)

        if phi_limit is None:
            phi_limit = np.mean(phi[-3:]) if phi.size >= 3 else np.mean(phi)
        if G_limit is None:
            G_limit = np.mean(G[-3:]) if G.size >= 3 else np.mean(G)

        R = np.abs(phi - phi_limit)
        FR = np.abs(G - G_limit)

        w0 = np.max(R) if R.size else EPS
        w1 = np.max(FR) if FR.size else EPS

        mask = (R > EPS) & (FR > EPS)
        R_non_zero = R[mask]
        FR_non_zero = FR[mask]

        Wmax = float(w0)

        k = int(R_non_zero.shape[0] - 1)
        if k <= 4:
            return float(EPS), float(Wmax)

        denom_num = np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon))))
        denom_den = np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon))))

        if abs(denom_num) < EPS or abs(denom_den) < EPS:
            return float(EPS), float(Wmax)

        hill_num = -(k / denom_num)
        hill_den = -(k / denom_den)

        gie = hill_num / hill_den if abs(hill_den) > EPS else np.nan
        return float(gie), float(Wmax)

    def compute_FIE_LID(self, phi, phi_limit=None, G_limit=None):
        """
        FIE version of compute_GIE_LID.

        Instead of using an external reference trajectory G, this uses the
        one-step forward loss trajectory:

            phi_for_estimator = phi[:-1]
            G_for_estimator   = phi[1:]

        The numerical estimator is otherwise exactly compute_GIE_LID.
        """
        phi = np.asarray(phi, dtype=np.float64)
        if phi.ndim != 1 or phi.size < 2:
            Wmax = float(np.max(np.abs(phi))) if phi.size else float(EPS)
            return float(EPS), float(Wmax)

        return self.compute_GIE_LID(
            phi[:-1],
            phi[1:],
            phi_limit=phi_limit,
            G_limit=G_limit,
        )


    def compute_Bayes_GIE(self, phi, G, Num0=0.0, Den0=0.0, phi_limit=None, G_limit=None):
        """
        Bayesian/smoothed GIE estimate using cumulative reciprocal-Hill sums.

        This follows the update the user proposed:

            hill_num = Hill(phi residuals)
            hill_den = Hill(G residuals)
            Num1 = 1 / hill_den
            Den1 = 1 / hill_num
            d_bayes = (Num0 + Num1) / (Den0 + Den1)

        If the current window is invalid, the increment is zero and the
        returned estimate is the previous cumulative ratio when possible.
        """
        epsilon = 1e-7

        phi = np.asarray(phi, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)

        Num0 = float(Num0)
        Den0 = float(Den0)

        if phi.ndim != 1 or G.ndim != 1 or phi.size == 0 or G.size == 0 or phi.size != G.size:
            prev = Num0 / Den0 if abs(Den0) > EPS else EPS
            Wmax = float(np.max(np.abs(phi))) if phi.size else float(EPS)
            return float(prev), float(Wmax), 0.0, 0.0

        if phi_limit is None:
            phi_limit = np.mean(phi[-3:]) if phi.size >= 3 else np.mean(phi)
        if G_limit is None:
            G_limit = np.mean(G[-3:]) if G.size >= 3 else np.mean(G)

        R = np.abs(phi - float(phi_limit))
        FR = np.abs(G - float(G_limit))

        w0 = np.max(R) if R.size else EPS
        w1 = np.max(FR) if FR.size else EPS
        Wmax = float(w0)

        if not (np.isfinite(w0) and np.isfinite(w1)) or w0 <= EPS or w1 <= EPS:
            prev = Num0 / Den0 if abs(Den0) > EPS else EPS
            return float(prev), float(Wmax), 0.0, 0.0

        # Paired filtering: remove any index where either deviation is zero.
        mask = (R > EPS) & (FR > EPS) & np.isfinite(R) & np.isfinite(FR)
        R_non_zero = R[mask]
        FR_non_zero = FR[mask]

        k = int(R_non_zero.shape[0] - 1)
        if k <= 4:
            prev = Num0 / Den0 if abs(Den0) > EPS else EPS
            return float(prev), float(Wmax), 0.0, 0.0

        denom_num = np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon))))
        denom_den = np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon))))

        Num1, Den1 = 0.0, 0.0
        if abs(denom_num) >= EPS and abs(denom_den) >= EPS:
            hill_num = -(k / denom_num)
            hill_den = -(k / denom_den)
            if (
                np.isfinite(hill_num)
                and np.isfinite(hill_den)
                and abs(hill_num) > EPS
                and abs(hill_den) > EPS
            ):
                # Denominator Hill goes in numerator's cumulative sum.
                Num1 = 1.0 / hill_den
                # Numerator Hill goes in denominator's cumulative sum.
                Den1 = 1.0 / hill_num

        Num_cumulative = Num0 + Num1
        Den_cumulative = Den0 + Den1

        if abs(Den_cumulative) > EPS and np.isfinite(Num_cumulative) and np.isfinite(Den_cumulative):
            LID_Bayes = Num_cumulative / Den_cumulative
        else:
            LID_Bayes = EPS

        if not np.isfinite(LID_Bayes) or LID_Bayes <= EPS:
            LID_Bayes = EPS

        return float(LID_Bayes), float(Wmax), float(Num1), float(Den1)

    def compute_Bayes_FIE(self, phi, Num0=0.0, Den0=0.0, phi_limit=None, G_limit=None):
        """
        Bayesian/smoothed FIE estimate.

        This is compute_Bayes_GIE applied to the one-step self map:

            phi_for_estimator = phi[:-1]
            G_for_estimator   = phi[1:]
        """
        phi = np.asarray(phi, dtype=np.float64)
        if phi.ndim != 1 or phi.size < 2:
            prev = float(Num0) / float(Den0) if abs(float(Den0)) > EPS else EPS
            Wmax = float(np.max(np.abs(phi))) if phi.size else float(EPS)
            return float(prev), float(Wmax), 0.0, 0.0

        return self.compute_Bayes_GIE(
            phi[:-1],
            phi[1:],
            Num0=Num0,
            Den0=Den0,
            phi_limit=phi_limit,
            G_limit=G_limit,
        )

def _ckl_equal_W(W, d1, d2):
    return W * ((d2 - d1) ** 2) / (((d1 + 1.0) ** 2) * (d2 + 1.0) + EPS)


def _ckl_case_A(W1, d1, W2, d2):
    term = (
        W1
        * (
            (d2 / (d1 + 1.0)) * math.log(max(W2 / W1, EPS))
            - (d1 - d2) / ((d1 + 1.0) ** 2)
        )
        + d2 * ((W2 - W1) + W1 * math.log(max(W1 / W2, EPS)))
    )
    return term + (d1 / (d1 + 1.0)) * W1 - (d2 / (d2 + 1.0)) * W2


def _ckl_case_B(W1, d1, W2, d2):
    ratio = max(W2 / W1, EPS)
    term = (W1 / ((d1 + 1.0) ** 2)) * (d2 * (ratio ** (d1 + 1.0)) - d1)
    return term + (d1 / (d1 + 1.0)) * W1 - (d2 / (d2 + 1.0)) * W2


def ckl_finite(W1, d1, W2, d2):
    vals = [W1, W2, d1, d2]
    if not all(np.isfinite(vals)) or min(vals) <= 0:
        return np.nan
    if abs(W1 - W2) < 1e-12:
        return _ckl_equal_W(W1, d1, d2)
    return _ckl_case_A(W1, d1, W2, d2) if W1 < W2 else _ckl_case_B(W1, d1, W2, d2)


def _safe_log(x: float, eps: float = EPS) -> float:
    return math.log(max(float(x), eps))


def _conditional_upper_trimmed_last3_limit(
    x: np.ndarray,
    eta: Optional[float],
    eps: float = EPS,
) -> float:
    """
    Common-limit estimator with optional conditional upper trimming.

    If eta is None, this is the original last-3 mean.

    If eta is given, sort the last three values x_(1) <= x_(2) <= x_(3).
    Trim x_(3) only when it is isolated from x_(2):

        x_(2) / x_(3) < eta.

    When the last three values are close, this ratio is near one, so no
    trimming occurs.  This protects the finite-sample limit estimate from a
    single upward tail spike without hiding a genuinely high / divergent tail.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    tail = x[-3:] if x.size >= 3 else x
    if tail.size < 3 or eta is None:
        return float(np.mean(tail))

    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0.0:
        return float(np.mean(tail))

    s = np.sort(tail)
    largest = float(s[-1])
    second = float(s[-2])

    # For cross-entropy losses largest should be positive.  If it is not,
    # fall back to the ordinary mean rather than making an unstable decision.
    if largest <= eps:
        return float(np.mean(tail))

    ratio = second / max(largest, eps)
    if ratio < eta:
        return float(np.mean(s[:-1]))
    return float(np.mean(tail))


def _common_limit(x: np.ndarray, limit_trim_eta: Optional[float] = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    return _conditional_upper_trimmed_last3_limit(x, eta=limit_trim_eta, eps=EPS)


def _residual_bounds(phi_segment, limit, eps: float = EPS) -> Tuple[float, float]:
    phi_segment = np.asarray(phi_segment, dtype=np.float64)
    R = np.abs(phi_segment - float(limit))
    R = R[np.isfinite(R) & (R > eps)]
    if R.size == 0:
        return np.nan, np.nan
    return float(np.min(R)), float(np.max(R))


def compute_three_gie(
    phi_hist_i,
    G_hist_i,
    lid_est: LIDEstimators,
    K: int,
    phi_limit_trim_eta: Optional[float] = None,
    G_limit_trim_eta: Optional[float] = None,
):
    """
    Split a length-K trajectory window into outer/inner halves and estimate
    d_I, d_O, d_c and the corresponding empirical residual scales.
    """
    phi = np.asarray(phi_hist_i, dtype=np.float64)
    G = np.asarray(G_hist_i, dtype=np.float64)

    if phi.size < K or G.size < K:
        return (np.nan,) * 7

    mid = K // 2
    if mid < 5 or (K - mid) < 5:
        return (np.nan,) * 7

    phi_outer, G_outer = phi[:mid], G[:mid]
    phi_inner, G_inner = phi[mid:K], G[mid:K]
    phi_common, G_common = phi[:K], G[:K]

    phi_limit_common = _common_limit(phi_common, limit_trim_eta=phi_limit_trim_eta)
    G_limit_common = _common_limit(G_common, limit_trim_eta=G_limit_trim_eta)

    dI, WI = lid_est.compute_GIE_LID(
        phi_inner, G_inner,
        phi_limit=phi_limit_common,
        G_limit=G_limit_common,
    )

    dO, WO = lid_est.compute_GIE_LID(
        phi_outer, G_outer,
        phi_limit=phi_limit_common,
        G_limit=G_limit_common,
    )

    dc, WO0 = lid_est.compute_GIE_LID(
        phi_common, G_common,
        phi_limit=phi_limit_common,
        G_limit=G_limit_common,
    )

    WL, _ = _residual_bounds(phi_outer, phi_limit_common, eps=EPS)

    vals = [dI, dO, dc, WI, WL, WO, WO0]
    if not all(np.isfinite(vals)) or min(vals) <= EPS:
        return (np.nan,) * 7

    WO = max(float(WO), float(WL) + EPS)
    WO0 = max(float(WO0), float(WI) + EPS)

    return float(dI), float(dO), float(dc), float(WI), float(WL), float(WO), float(WO0)


def compute_three_gie_lrt_limit(
    phi_hist_i,
    G_hist_i,
    lid_est: LIDEstimators,
    K: int,
    limit_trim_eta: float,
    trim_G_limit: bool = False,
):
    """
    LRT/GIE plug-in estimates using a conditional last-3 limit for phi.

    This keeps the same GIE formula and the same common origin for outer,
    inner, and common windows.  Only the finite-sample estimate of the common
    phi limit is changed.  By default the G limit remains the original last-3
    mean, because the observed instability was caused by phi-limit
    contamination.  Set trim_G_limit=True only for a separate diagnostic.
    """
    return compute_three_gie(
        phi_hist_i=phi_hist_i,
        G_hist_i=G_hist_i,
        lid_est=lid_est,
        K=K,
        phi_limit_trim_eta=limit_trim_eta,
        G_limit_trim_eta=limit_trim_eta if trim_G_limit else None,
    )


def compute_three_fie(
    phi_hist_i,
    lid_est: LIDEstimators,
    K: int,
    limit_trim_eta: Optional[float] = None,
):
    """
    FIE analogue of compute_three_gie.

    This estimates d_I, d_O, and d_c from a single example trajectory by
    calling compute_FIE_LID on each window.  Internally this means each window
    uses the paired arrays phi[:-1] and phi[1:], rather than an external
    class/population trajectory G.
    """
    phi = np.asarray(phi_hist_i, dtype=np.float64)

    if phi.size < K:
        return (np.nan,) * 7

    mid = K // 2
    if mid < 5 or (K - mid) < 5:
        return (np.nan,) * 7

    phi_outer = phi[:mid]
    phi_inner = phi[mid:K]
    phi_common = phi[:K]

    # For FIE the source and forward image have the same fixed-point limit.
    phi_limit_common = _common_limit(phi_common, limit_trim_eta=limit_trim_eta)

    dI, WI = lid_est.compute_FIE_LID(
        phi_inner,
        phi_limit=phi_limit_common,
        G_limit=phi_limit_common,
    )

    dO, WO = lid_est.compute_FIE_LID(
        phi_outer,
        phi_limit=phi_limit_common,
        G_limit=phi_limit_common,
    )

    dc, WO0 = lid_est.compute_FIE_LID(
        phi_common,
        phi_limit=phi_limit_common,
        G_limit=phi_limit_common,
    )

    WL, _ = _residual_bounds(phi_outer, phi_limit_common, eps=EPS)

    vals = [dI, dO, dc, WI, WL, WO, WO0]
    if not all(np.isfinite(vals)) or min(vals) <= EPS:
        return (np.nan,) * 7

    WO = max(float(WO), float(WL) + EPS)
    WO0 = max(float(WO0), float(WI) + EPS)

    return float(dI), float(dO), float(dc), float(WI), float(WL), float(WO), float(WO0)


def compute_three_fie_lrt_limit(
    phi_hist_i,
    lid_est: LIDEstimators,
    K: int,
    limit_trim_eta: float,
):
    """
    Limit-trim FIE analogue used for LRT diagnostics.
    """
    return compute_three_fie(
        phi_hist_i=phi_hist_i,
        lid_est=lid_est,
        K=K,
        limit_trim_eta=limit_trim_eta,
    )



def compute_three_bayes_fie(
    phi_hist_i,
    lid_est: LIDEstimators,
    K: int,
    Num0: Dict[str, float],
    Den0: Dict[str, float],
    limit_trim_eta: Optional[float] = None,
):
    """
    Bayesian/smoothed FIE analogue of compute_three_fie.

    ``Num0`` and ``Den0`` are cumulative sums for one example with keys:

        inner, outer, common

    The function returns

        dI, dO, dc, WI, WL, WO, WO0, Num_inc, Den_inc

    where Num_inc and Den_inc contain the current-epoch increments that should
    be added to the cumulative state by the caller.  Keeping the update outside
    the function makes the state explicit and easy to save/debug.
    """
    phi = np.asarray(phi_hist_i, dtype=np.float64)

    zero_inc = {"inner": 0.0, "outer": 0.0, "common": 0.0}
    if phi.size < K:
        return (np.nan,) * 7 + (zero_inc.copy(), zero_inc.copy())

    mid = K // 2
    if mid < 5 or (K - mid) < 5:
        return (np.nan,) * 7 + (zero_inc.copy(), zero_inc.copy())

    phi_outer = phi[:mid]
    phi_inner = phi[mid:K]
    phi_common = phi[:K]

    phi_limit_common = _common_limit(phi_common, limit_trim_eta=limit_trim_eta)

    Num_inc = zero_inc.copy()
    Den_inc = zero_inc.copy()

    dI, WI, n1, d1 = lid_est.compute_Bayes_FIE(
        phi_inner,
        Num0=float(Num0.get("inner", 0.0)),
        Den0=float(Den0.get("inner", 0.0)),
        phi_limit=phi_limit_common,
        G_limit=phi_limit_common,
    )
    Num_inc["inner"] = n1
    Den_inc["inner"] = d1

    dO, WO, n1, d1 = lid_est.compute_Bayes_FIE(
        phi_outer,
        Num0=float(Num0.get("outer", 0.0)),
        Den0=float(Den0.get("outer", 0.0)),
        phi_limit=phi_limit_common,
        G_limit=phi_limit_common,
    )
    Num_inc["outer"] = n1
    Den_inc["outer"] = d1

    dc, WO0, n1, d1 = lid_est.compute_Bayes_FIE(
        phi_common,
        Num0=float(Num0.get("common", 0.0)),
        Den0=float(Den0.get("common", 0.0)),
        phi_limit=phi_limit_common,
        G_limit=phi_limit_common,
    )
    Num_inc["common"] = n1
    Den_inc["common"] = d1

    WL, _ = _residual_bounds(phi_outer, phi_limit_common, eps=EPS)

    vals = [dI, dO, dc, WI, WL, WO, WO0]
    if not all(np.isfinite(vals)) or min(vals) <= EPS:
        return (np.nan,) * 7 + (Num_inc, Den_inc)

    WO = max(float(WO), float(WL) + EPS)
    WO0 = max(float(WO0), float(WI) + EPS)

    return float(dI), float(dO), float(dc), float(WI), float(WL), float(WO), float(WO0), Num_inc, Den_inc


def compute_three_bayes_fie_lrt_limit(
    phi_hist_i,
    lid_est: LIDEstimators,
    K: int,
    Num0: Dict[str, float],
    Den0: Dict[str, float],
    limit_trim_eta: float,
):
    """Limit-trim version of Bayesian FIE."""
    return compute_three_bayes_fie(
        phi_hist_i=phi_hist_i,
        lid_est=lid_est,
        K=K,
        Num0=Num0,
        Den0=Den0,
        limit_trim_eta=limit_trim_eta,
    )

# ============================================================
# LRT-only boundary helpers
# ============================================================
# These functions do not change compute_GIE_LID.  CKL continues to use the
# original max-boundary GIE estimator.  The robust boundary option below is
# used only for LRT, where the boundary enters Lambda_w directly and
# Lambda_alpha indirectly through alpha0(WI, WO0, dc).


def _residual_bounds_lrt(phi_segment, limit, eps: float = EPS, boundary_quantile=None) -> Tuple[float, float]:
    """
    Residual lower/upper bounds for LRT.

    boundary_quantile=None gives the original max boundary.
    boundary_quantile=q in (0,1) gives a robust upper boundary Q_q(R).

    The lower boundary remains the empirical minimum positive residual.  Only
    the upper boundary is robustified, because the spike sensitivity we are
    diagnosing comes from the max/order-statistic upper boundary.
    """
    phi_segment = np.asarray(phi_segment, dtype=np.float64)
    R = np.abs(phi_segment - float(limit))
    R = R[np.isfinite(R) & (R > eps)]
    if R.size == 0:
        return np.nan, np.nan

    lower = float(np.min(R))
    if boundary_quantile is None:
        upper = float(np.max(R))
    else:
        q = float(boundary_quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError("boundary_quantile must be in (0,1] or None")
        upper = float(np.quantile(R, q))
        upper = max(upper, lower + eps)
    return lower, upper


def compute_three_gie_lrt_boundaries(
    phi_hist_i,
    G_hist_i,
    lid_est: LIDEstimators,
    K: int,
    boundary_quantile=None,
):
    """
    LRT version of compute_three_gie with an optional robust boundary.

    Important: this function does not alter compute_GIE_LID, so CKL and the
    original GIE/W estimates are unchanged.  The exponents d_I, d_O, d_c are
    still computed by the original GIE estimator.  Only the LRT boundary
    values WI, WL, WO, WO0 are replaced by residual bounds computed from phi.
    """
    phi = np.asarray(phi_hist_i, dtype=np.float64)
    G = np.asarray(G_hist_i, dtype=np.float64)

    if phi.size < K or G.size < K:
        return (np.nan,) * 7

    mid = K // 2
    if mid < 5 or (K - mid) < 5:
        return (np.nan,) * 7

    phi_outer, G_outer = phi[:mid], G[:mid]
    phi_inner, G_inner = phi[mid:K], G[mid:K]
    phi_common, G_common = phi[:K], G[:K]

    phi_limit_common = _common_limit(phi_common)
    G_limit_common = _common_limit(G_common)

    # Exponents are unchanged.  We discard the W values returned by
    # compute_GIE_LID so that robust boundaries affect only LRT.
    dI, _ = lid_est.compute_GIE_LID(
        phi_inner, G_inner,
        phi_limit=phi_limit_common,
        G_limit=G_limit_common,
    )
    dO, _ = lid_est.compute_GIE_LID(
        phi_outer, G_outer,
        phi_limit=phi_limit_common,
        G_limit=G_limit_common,
    )
    dc, _ = lid_est.compute_GIE_LID(
        phi_common, G_common,
        phi_limit=phi_limit_common,
        G_limit=G_limit_common,
    )

    _, WI = _residual_bounds_lrt(
        phi_inner, phi_limit_common, eps=EPS, boundary_quantile=boundary_quantile
    )
    WL, WO = _residual_bounds_lrt(
        phi_outer, phi_limit_common, eps=EPS, boundary_quantile=boundary_quantile
    )
    _, WO0 = _residual_bounds_lrt(
        phi_common, phi_limit_common, eps=EPS, boundary_quantile=boundary_quantile
    )

    vals = [dI, dO, dc, WI, WL, WO, WO0]
    if not all(np.isfinite(vals)) or min(vals) <= EPS:
        return (np.nan,) * 7

    WO = max(float(WO), float(WL) + EPS)
    WO0 = max(float(WO0), float(WI) + EPS)

    return float(dI), float(dO), float(dc), float(WI), float(WL), float(WO), float(WO0)

def lrt_lambda_d_gie(dI: float, dO: float, dc: float, m: int, n: int, eps: float = EPS) -> float:
    """
    GIE plug-in exponent contribution.
    """
    dI = max(float(dI), eps)
    dO = max(float(dO), eps)
    dc = max(float(dc), eps)
    m = int(m)
    n = int(n)

    if m <= 0 or n <= 0:
        return np.nan

    alpha = np.clip(m / (m + n), eps, 1.0 - eps)

    inner = 2.0 * m * (_safe_log(dI / dc, eps) - 1.0 + dc / dI)

    power = dc / dO
    alpha_power = alpha ** power
    ratio_term = max((1.0 - alpha_power) / (1.0 - alpha), eps)

    outer = (
        2.0 * n * _safe_log(dO / dc, eps)
        - (2.0 * (dO - dc) / dO) * (n + m * _safe_log(alpha, eps))
        + 2.0 * n * _safe_log(ratio_term, eps)
    )

    val = inner + outer
    return float(val) if np.isfinite(val) else np.nan


def lrt_lambda_w(
    dc: float,
    WI: float,
    WL: float,
    WO: float,
    WO0: float,
    n: int,
    eps: float = EPS,
) -> float:
    dc = max(float(dc), eps)
    WI = max(float(WI), eps)
    WL = max(float(WL), eps)
    WO = max(float(WO), WL + eps)
    WO0 = max(float(WO0), WI + eps)
    n = int(n)

    if n <= 0:
        return np.nan

    numerator = max((WO0 ** dc) - (WI ** dc), eps)
    denominator = max((WO ** dc) - (WL ** dc), eps)

    val = 2.0 * n * np.log(numerator / denominator)
    return float(val) if np.isfinite(val) else np.nan


def lrt_alpha0(WI: float, WO: float, dc: float, eps: float = EPS) -> float:
    WI = max(float(WI), eps)
    WO = max(float(WO), eps)
    dc = max(float(dc), eps)
    ratio = np.clip(WI / max(WO, eps), eps, 1.0)
    return float(ratio ** dc)


def lrt_lambda_alpha(m: int, n: int, alpha0: float, eps: float = EPS) -> float:
    m = int(m)
    n = int(n)
    if m <= 0 or n <= 0:
        return np.nan

    alpha_hat = np.clip(m / (m + n), eps, 1.0 - eps)
    alpha0 = np.clip(float(alpha0), eps, 1.0 - 1e-12)

    val = (
        2.0 * m * _safe_log(alpha_hat / alpha0, eps)
        + 2.0 * n * _safe_log((1.0 - alpha_hat) / (1.0 - alpha0), eps)
    )
    return float(val) if np.isfinite(val) else np.nan


def geom_mean_pos(x, eps=EPS):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.maximum(x, eps)
    return float(np.exp(np.mean(np.log(x))))


def class_geom_means(W, d, y, num_classes=10, eps=EPS):
    W = np.asarray(W, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    W2 = np.full(num_classes, np.nan, dtype=np.float64)
    d2 = np.full(num_classes, np.nan, dtype=np.float64)

    for c in range(num_classes):
        idx = (y == c)
        W2[c] = geom_mean_pos(W[idx], eps=eps)
        d2[c] = geom_mean_pos(d[idx], eps=eps)
    return W2, d2

# ============================================================
# Direct finite-boundary LRT helpers
# ============================================================
# The component functions above use closed-form / GIE plug-in contrasts.
# The helpers below evaluate the actual log-likelihoods ell_A, ell_B,
# ell_C, and ell_D from the residual data in the current window.  This lets
# you compare
#
#     direct LRT = 2(ell_A - ell_D)
#
# against the old componentwise value Lambda_d + Lambda_w + Lambda_alpha.
# For fixed parameters, the direct components below always sum exactly to the
# direct full value by construction.


def _log_power_gap(upper: float, lower: float, d: float, eps: float = EPS) -> float:
    """
    Stable computation of log(upper**d - lower**d).

    Returns NaN if the interval is invalid.  This function avoids overflow for
    large d by working on the log scale.
    """
    upper = float(upper)
    lower = float(lower)
    d = float(d)

    if not (np.isfinite(upper) and np.isfinite(lower) and np.isfinite(d)):
        return np.nan
    if upper <= eps or lower < 0.0 or d <= eps or upper <= lower:
        return np.nan

    log_u = d * math.log(max(upper, eps))
    if lower <= eps:
        return float(log_u)

    log_l = d * math.log(max(lower, eps))
    if log_l >= log_u:
        return np.nan

    # log(exp(log_u) - exp(log_l)) = log_u + log(1 - exp(log_l - log_u))
    return float(log_u + math.log1p(-math.exp(log_l - log_u)))


def _lrt_residual_split(
    phi_hist_i,
    K: int,
    eps: float = EPS,
    limit_trim_eta: Optional[float] = None,
):
    """
    Build the inner, outer, and combined positive residual arrays used by LRT.

    The same common limit used elsewhere in the file is used here.  Residuals
    are floored at eps so that all K time points remain in the likelihood.
    """
    phi = np.asarray(phi_hist_i, dtype=np.float64)
    if phi.size < K:
        return None, None, None, np.nan

    phi = phi[:K]
    mid = K // 2
    if mid <= 0 or K - mid <= 0:
        return None, None, None, np.nan

    limit = _common_limit(phi, limit_trim_eta=limit_trim_eta)
    R = np.abs(phi - limit)
    R = np.maximum(R, eps)
    R = R[np.isfinite(R)]
    if R.size != K:
        return None, None, None, np.nan

    Y = R[:mid]      # outer/earlier half
    X = R[mid:K]     # inner/later half
    return X, Y, R, float(limit)


def _lrt_loglik_A(
    X: np.ndarray,
    Y: np.ndarray,
    dI: float,
    dO: float,
    WI: float,
    WL: float,
    WO: float,
    alpha: float,
    eps: float = EPS,
) -> float:
    """Full alternative log-likelihood ell_A."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    kI = int(X.size)
    kO = int(Y.size)

    dI = max(float(dI), eps)
    dO = max(float(dO), eps)
    WI = max(float(WI), eps)
    WL = max(float(WL), eps)
    WO = max(float(WO), WL + eps)
    alpha = float(np.clip(alpha, eps, 1.0 - eps))

    SX = float(np.sum(np.log(np.maximum(X, eps))))
    SY = float(np.sum(np.log(np.maximum(Y, eps))))
    gap_O = _log_power_gap(WO, WL, dO, eps=eps)
    if not np.isfinite(gap_O):
        return np.nan

    val = (
        kI * _safe_log(alpha, eps)
        + kO * _safe_log(1.0 - alpha, eps)
        + kI * _safe_log(dI, eps)
        + (dI - 1.0) * SX
        - kI * dI * _safe_log(WI, eps)
        + kO * _safe_log(dO, eps)
        + (dO - 1.0) * SY
        - kO * gap_O
    )
    return float(val) if np.isfinite(val) else np.nan


def _lrt_loglik_B(
    X: np.ndarray,
    Y: np.ndarray,
    dc: float,
    WI: float,
    WL: float,
    WO: float,
    alpha: float,
    eps: float = EPS,
) -> float:
    """Common-exponent intermediate log-likelihood ell_B."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    kI = int(X.size)
    kO = int(Y.size)
    K = kI + kO

    dc = max(float(dc), eps)
    WI = max(float(WI), eps)
    WL = max(float(WL), eps)
    WO = max(float(WO), WL + eps)
    alpha = float(np.clip(alpha, eps, 1.0 - eps))

    S = float(np.sum(np.log(np.maximum(np.concatenate([X, Y]), eps))))
    gap_O = _log_power_gap(WO, WL, dc, eps=eps)
    if not np.isfinite(gap_O):
        return np.nan

    val = (
        kI * _safe_log(alpha, eps)
        + kO * _safe_log(1.0 - alpha, eps)
        + K * _safe_log(dc, eps)
        + (dc - 1.0) * S
        - kI * dc * _safe_log(WI, eps)
        - kO * gap_O
    )
    return float(val) if np.isfinite(val) else np.nan


def _lrt_loglik_C(
    X: np.ndarray,
    Y: np.ndarray,
    dc: float,
    WI: float,
    WO0: float,
    alpha: float,
    eps: float = EPS,
) -> float:
    """Common-boundary intermediate log-likelihood ell_C."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    kI = int(X.size)
    kO = int(Y.size)
    K = kI + kO

    dc = max(float(dc), eps)
    WI = max(float(WI), eps)
    WO0 = max(float(WO0), WI + eps)
    alpha = float(np.clip(alpha, eps, 1.0 - eps))

    S = float(np.sum(np.log(np.maximum(np.concatenate([X, Y]), eps))))
    gap_0 = _log_power_gap(WO0, WI, dc, eps=eps)
    if not np.isfinite(gap_0):
        return np.nan

    val = (
        kI * _safe_log(alpha, eps)
        + kO * _safe_log(1.0 - alpha, eps)
        + K * _safe_log(dc, eps)
        + (dc - 1.0) * S
        - kI * dc * _safe_log(WI, eps)
        - kO * gap_0
    )
    return float(val) if np.isfinite(val) else np.nan


def _lrt_loglik_D(
    R: np.ndarray,
    dc: float,
    WO0: float,
    eps: float = EPS,
) -> float:
    """Restricted null log-likelihood ell_D = ell_0."""
    R = np.asarray(R, dtype=np.float64)
    K = int(R.size)
    dc = max(float(dc), eps)
    WO0 = max(float(WO0), eps)

    S = float(np.sum(np.log(np.maximum(R, eps))))
    val = K * _safe_log(dc, eps) + (dc - 1.0) * S - K * dc * _safe_log(WO0, eps)
    return float(val) if np.isfinite(val) else np.nan


def compute_lrt_direct_plugin(
    phi_hist_i,
    K: int,
    dI: float,
    dO: float,
    dc: float,
    WI: float,
    WL: float,
    WO: float,
    WO0: float,
    eps: float = EPS,
    limit_trim_eta: Optional[float] = None,
) -> Dict[str, float]:
    """
    Direct plug-in evaluation of the finite-boundary LRT.

    This function uses the same plug-in estimates (dI, dO, dc, WI, WL, WO,
    WO0) that you already use for the component formulas.  It then evaluates
    the four log-likelihoods ell_A, ell_B, ell_C, ell_D directly from the
    residual arrays and returns:

        lrt_direct = 2 * (ell_A - ell_D)

    together with direct versions of the three components.  These direct
    components sum exactly to lrt_direct, so comparing them with the old
    Lambda_d/Lambda_w/Lambda_alpha tells us whether a component formula or a
    plug-in approximation is causing the discrepancy.
    """
    X, Y, R, limit = _lrt_residual_split(
        phi_hist_i, K=K, eps=eps, limit_trim_eta=limit_trim_eta
    )
    if X is None or Y is None or R is None:
        return {
            "lrt": np.nan,
            "lambda_d": np.nan,
            "lambda_w": np.nan,
            "lambda_alpha": np.nan,
            "ell_A": np.nan,
            "ell_B": np.nan,
            "ell_C": np.nan,
            "ell_D": np.nan,

            # Raw plug-in parameters used by the direct LRT.
            "alpha_hat": np.nan,
            "alpha0": np.nan,
            "dI_hat": np.nan,
            "dO_hat": np.nan,
            "dc_hat": np.nan,
            "S_X": np.nan,
            "S_Y": np.nan,
            "wI_hat": np.nan,
            "wL_hat": np.nan,
            "wO_hat": np.nan,
            "wO0_hat": np.nan,

            # Exact 8 displayed-formula terms, each already multiplied by 2.
            "Lambda_alpha": np.nan,
            "Lambda_dI": np.nan,
            "Lambda_wI": np.nan,
            "Lambda_dO": np.nan,
            "Lambda_wO": np.nan,
            "Lambda_dc": np.nan,
            "Lambda_Sc": np.nan,
            "Lambda_w0": np.nan,
            "Lambda_8_sum": np.nan,
            "Lambda_8_minus_lrt": np.nan,

            "ordered_boundary_ok": 0.0,
            "support_ok": 0.0,
            "limit": np.nan,
            "limit_trim_eta": float(limit_trim_eta) if limit_trim_eta is not None else np.nan,
        }

    kI = int(X.size)
    kO = int(Y.size)
    alpha_hat = float(np.clip(kI / max(kI + kO, 1), eps, 1.0 - eps))

    # Keep the likelihood supports at least as large as the observed residuals.
    # For the original max-boundary LRT these max() calls are no-ops.  For
    # quantile boundaries they prevent accidental support violations, but the
    # resulting value should be interpreted only as a diagnostic.
    WI_eff = max(float(WI), float(np.max(X)), eps)
    WL_eff = max(float(WL), eps)
    WO_eff = max(float(WO), float(np.max(Y)), WL_eff + eps)
    WO0_eff = max(float(WO0), float(np.max(R)), WI_eff + eps)

    # Raw sufficient quantities used inside the displayed 8-term formula.
    # These are saved so explanations do not have to reconstruct them later.
    S_X = float(np.sum(np.log(np.maximum(X, eps))))
    S_Y = float(np.sum(np.log(np.maximum(Y, eps))))

    dI_eff = max(float(dI), eps)
    dO_eff = max(float(dO), eps)
    dc_eff = max(float(dc), eps)

    gap_O = _log_power_gap(WO_eff, WL_eff, dO_eff, eps=eps)

    if np.isfinite(gap_O):
        Lambda_alpha = 2.0 * (
            kI * _safe_log(alpha_hat, eps)
            + kO * _safe_log(1.0 - alpha_hat, eps)
        )
        Lambda_dI = 2.0 * (
            kI * _safe_log(dI_eff, eps)
            + (dI_eff - 1.0) * S_X
        )
        Lambda_wI = 2.0 * (
            -kI * dI_eff * _safe_log(WI_eff, eps)
        )
        Lambda_dO = 2.0 * (
            kO * _safe_log(dO_eff, eps)
            + (dO_eff - 1.0) * S_Y
        )
        Lambda_wO = 2.0 * (
            -kO * gap_O
        )
        Lambda_dc = 2.0 * (
            -(kI + kO) * _safe_log(dc_eff, eps)
        )
        Lambda_Sc = 2.0 * (
            -(dc_eff - 1.0) * (S_X + S_Y)
        )
        Lambda_w0 = 2.0 * (
            (kI + kO) * dc_eff * _safe_log(WO0_eff, eps)
        )
        Lambda_8_sum = (
            Lambda_alpha
            + Lambda_dI
            + Lambda_wI
            + Lambda_dO
            + Lambda_wO
            + Lambda_dc
            + Lambda_Sc
            + Lambda_w0
        )
    else:
        Lambda_alpha = Lambda_dI = Lambda_wI = Lambda_dO = np.nan
        Lambda_wO = Lambda_dc = Lambda_Sc = Lambda_w0 = np.nan
        Lambda_8_sum = np.nan

    ell_A = _lrt_loglik_A(X, Y, dI, dO, WI_eff, WL_eff, WO_eff, alpha_hat, eps=eps)
    ell_B = _lrt_loglik_B(X, Y, dc, WI_eff, WL_eff, WO_eff, alpha_hat, eps=eps)
    ell_C = _lrt_loglik_C(X, Y, dc, WI_eff, WO0_eff, alpha_hat, eps=eps)
    ell_D = _lrt_loglik_D(R, dc, WO0_eff, eps=eps)

    if not all(np.isfinite([ell_A, ell_B, ell_C, ell_D])):
        lrt = lam_d = lam_w = lam_alpha = np.nan
    else:
        lam_d = 2.0 * (ell_A - ell_B)
        lam_w = 2.0 * (ell_B - ell_C)
        lam_alpha = 2.0 * (ell_C - ell_D)
        lrt = 2.0 * (ell_A - ell_D)

    alpha0 = lrt_alpha0(WI_eff, WO0_eff, dc, eps=eps)

    support_ok = float(
        np.all(X <= WI_eff + 10 * eps)
        and np.all(Y >= WL_eff - 10 * eps)
        and np.all(Y <= WO_eff + 10 * eps)
        and np.all(R <= WO0_eff + 10 * eps)
    )
    ordered_ok = float(WI_eff <= WL_eff + 10 * eps)

    return {
        "lrt": float(lrt) if np.isfinite(lrt) else np.nan,
        "lambda_d": float(lam_d) if np.isfinite(lam_d) else np.nan,
        "lambda_w": float(lam_w) if np.isfinite(lam_w) else np.nan,
        "lambda_alpha": float(lam_alpha) if np.isfinite(lam_alpha) else np.nan,
        "ell_A": float(ell_A) if np.isfinite(ell_A) else np.nan,
        "ell_B": float(ell_B) if np.isfinite(ell_B) else np.nan,
        "ell_C": float(ell_C) if np.isfinite(ell_C) else np.nan,
        "ell_D": float(ell_D) if np.isfinite(ell_D) else np.nan,
        # Existing alpha quantities.
        "alpha_hat": float(alpha_hat),
        "alpha0": float(alpha0) if np.isfinite(alpha0) else np.nan,

        # Raw plug-in parameters used by the direct LRT.
        "dI_hat": float(dI_eff),
        "dO_hat": float(dO_eff),
        "dc_hat": float(dc_eff),
        "S_X": float(S_X) if np.isfinite(S_X) else np.nan,
        "S_Y": float(S_Y) if np.isfinite(S_Y) else np.nan,
        "wI_hat": float(WI_eff),
        "wL_hat": float(WL_eff),
        "wO_hat": float(WO_eff),
        "wO0_hat": float(WO0_eff),

        # Exact 8 displayed-formula terms, each already multiplied by 2.
        "Lambda_alpha": float(Lambda_alpha) if np.isfinite(Lambda_alpha) else np.nan,
        "Lambda_dI": float(Lambda_dI) if np.isfinite(Lambda_dI) else np.nan,
        "Lambda_wI": float(Lambda_wI) if np.isfinite(Lambda_wI) else np.nan,
        "Lambda_dO": float(Lambda_dO) if np.isfinite(Lambda_dO) else np.nan,
        "Lambda_wO": float(Lambda_wO) if np.isfinite(Lambda_wO) else np.nan,
        "Lambda_dc": float(Lambda_dc) if np.isfinite(Lambda_dc) else np.nan,
        "Lambda_Sc": float(Lambda_Sc) if np.isfinite(Lambda_Sc) else np.nan,
        "Lambda_w0": float(Lambda_w0) if np.isfinite(Lambda_w0) else np.nan,
        "Lambda_8_sum": float(Lambda_8_sum) if np.isfinite(Lambda_8_sum) else np.nan,
        "Lambda_8_minus_lrt": (
            float(Lambda_8_sum - lrt)
            if np.isfinite(Lambda_8_sum) and np.isfinite(lrt)
            else np.nan
        ),

        "ordered_boundary_ok": ordered_ok,
        "support_ok": support_ok,
        "limit": float(limit) if np.isfinite(limit) else np.nan,
        "limit_trim_eta": float(limit_trim_eta) if limit_trim_eta is not None else np.nan,
    }

