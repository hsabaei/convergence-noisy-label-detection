"""Shared score-normalization and temporal-detection interface.

Both CKL and LE monitoring methods are expected to produce a score array with
shape (T, N): epochs by training samples.  This module applies the same
class-wise calibration and temporal detectors to either score.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .detectors import (
    cumulative_ever_detected,
    ewma_detector_same_tau,
    exceedance_from_z,
    first_hit_from_boolean,
    minrun_detector,
    sliding_window_detector,
    zscore_by_class,
)

ScoreDirection = Literal["higher", "lower"]


def standardize_monitoring_score(
    score_all: np.ndarray,
    labels: np.ndarray,
    *,
    direction: ScoreDirection = "higher",
    num_classes: int = 10,
    start_index: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return class-wise z-scores with positive values interpreted as noisier.

    Parameters
    ----------
    score_all:
        Monitoring statistic with shape (T, N).
    labels:
        Observed training labels, shape (N,).
    direction:
        ``"higher"`` if larger raw scores are more noisy-like; ``"lower"``
        if smaller raw scores are more noisy-like.
    start_index:
        Epoch-array index before which output remains NaN.
    """
    score_all = np.asarray(score_all, dtype=np.float64)
    if direction == "higher":
        oriented = score_all
    elif direction == "lower":
        oriented = -score_all
    else:
        raise ValueError("direction must be 'higher' or 'lower'.")

    return zscore_by_class(
        oriented,
        labels,
        num_classes=num_classes,
        eps=eps,
        start_index=start_index,
    )


def run_temporal_detectors(
    z_all: np.ndarray,
    *,
    tau: float,
    minrun_m: int,
    sliding_length: int,
    sliding_k: int,
    ewma_lambda: float,
) -> dict:
    """Apply the three z-score-based temporal detectors.

    ``z_all`` is already standardized within observed class and epoch, so the
    detectors compare z directly with a global z-threshold ``tau``.  No second
    subtraction of the class mean is performed.

    Min-run and sliding-window operate on
        I_{i,t} = 1{z_{i,t} >= tau}.
    EWMA operates directly on z.

    The returned ``ever_detected`` arrays implement the sequential declaration
    rule: once a sample is detected, it remains detected.
    """
    z_all = np.asarray(z_all, dtype=np.float64)
    if z_all.ndim != 2:
        raise ValueError("z_all must have shape [T,N].")

    exceedance = exceedance_from_z(
        z_all,
        tau=float(tau),
    )

    # True consecutive-run recursion; it resets to zero when I_{i,t}=0.
    runlen, minrun_hit, _ = minrun_detector(
        exceedance,
        m=int(minrun_m),
    )
    minrun_ever = cumulative_ever_detected(minrun_hit)
    minrun_first = first_hit_from_boolean(minrun_hit)

    window_sum, window_hit_compact, _, _ = sliding_window_detector(
        exceedance,
        ell=int(sliding_length),
        k=int(sliding_k),
    )

    # sliding_window_detector returns one row per complete window.
    T, N = z_all.shape
    window_hit = np.zeros((T, N), dtype=bool)
    window_stat = np.full((T, N), np.nan, dtype=np.float64)
    first_window_end = int(sliding_length) - 1
    window_hit[first_window_end:] = window_hit_compact
    window_stat[first_window_end:] = window_sum
    window_ever = cumulative_ever_detected(window_hit)
    window_first = first_hit_from_boolean(window_hit)

    z_ewma, ewma_hit, _ = ewma_detector_same_tau(
        z_all,
        tau=float(tau),
        lam=float(ewma_lambda),
    )
    ewma_ever = cumulative_ever_detected(ewma_hit)
    ewma_first = first_hit_from_boolean(ewma_hit)

    return {
        "z": z_all,
        "exceedance": exceedance,
        "minrun": {
            "run_length": runlen,
            "hit": minrun_hit,
            "ever_detected": minrun_ever,
            "first_hit": minrun_first,
        },
        "sliding_window": {
            "window_sum": window_stat,
            "hit": window_hit,
            "ever_detected": window_ever,
            "first_hit": window_first,
        },
        "ewma": {
            "smoothed_z": z_ewma,
            "hit": ewma_hit,
            "ever_detected": ewma_ever,
            "first_hit": ewma_first,
        },
    }

