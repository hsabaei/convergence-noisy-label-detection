"""Convergence-based noisy-label detection framework."""

from .data import CIFAR10WithIndex, MixedCIFAR10WithInjectedCIFAR100, NoisyCIFAR10WithIndex
from .models import CNN12, CNN12_Model, ConvBlock
from .training import (
    evaluate,
    evaluate_accuracy,
    evaluate_group_diagnostics,
    make_frozen_mask,
    make_subset_mask,
    set_seed,
)
from .estimators import EPS, LIDEstimators, ckl_finite
from .proposed_le import proposed_le_estimator, rolling_proposed_le_estimator
from .detectors import (
    binary_auc_from_scores,
    compute_minrun_m,
    ewma_detector_same_tau,
    ewma_smooth_z,
    exceedance_from_z,
    minrun_detector,
    sliding_window_detector,
    zscore_by_class,
)
from .framework import standardize_monitoring_score, run_temporal_detectors

__all__ = [
    "CIFAR10WithIndex",
    "MixedCIFAR10WithInjectedCIFAR100",
    "NoisyCIFAR10WithIndex",
    "CNN12",
    "CNN12_Model",
    "ConvBlock",
    "evaluate",
    "evaluate_accuracy",
    "evaluate_group_diagnostics",
    "make_frozen_mask",
    "make_subset_mask",
    "set_seed",
    "EPS",
    "LIDEstimators",
    "ckl_finite",
    "proposed_le_estimator",
    "rolling_proposed_le_estimator",
    "binary_auc_from_scores",
    "compute_minrun_m",
    "ewma_detector_same_tau",
    "ewma_smooth_z",
    "exceedance_from_z",
    "minrun_detector",
    "sliding_window_detector",
    "zscore_by_class",
    "standardize_monitoring_score",
    "run_temporal_detectors",
]
