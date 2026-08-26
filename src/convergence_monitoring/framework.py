"""Shared score-normalization and temporal-detection interface.

Both CKL and LE monitoring methods are expected to produce a score array with
shape (T, N): epochs by training samples.  This module applies the same
class-wise calibration and temporal detectors to either score.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .detectors import (
    ewma_detector_same_tau,
    exceedance_from_z,
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
    """Apply min-run, sliding-window, and EWMA detectors to one z-score trace."""
    z_all = np.asarray(z_all, dtype=np.float64)
    exceedance = exceedance_from_z(z_all, tau=float(tau))

    runlen, minrun_drift, minrun_first = minrun_detector(
        exceedance,
        m=int(minrun_m),
    )
    window_sum, window_hit, window_any, window_first = sliding_window_detector(
        exceedance,
        ell=int(sliding_length),
        k=int(sliding_k),
    )
    z_ewma, ewma_drift, ewma_first = ewma_detector_same_tau(
        z_all,
        tau=float(tau),
        lam=float(ewma_lambda),
    )

    return {
        "z": z_all,
        "exceedance": exceedance,
        "minrun": {
            "run_length": runlen,
            "decision": minrun_drift,
            "first_hit": minrun_first,
        },
        "sliding_window": {
            "window_sum": window_sum,
            "decision": window_hit,
            "ever_detected": window_any,
            "first_hit": window_first,
        },
        "ewma": {
            "smoothed_z": z_ewma,
            "decision": ewma_drift,
            "first_hit": ewma_first,
        },
    }
