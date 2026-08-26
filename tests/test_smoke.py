import numpy as np

from convergence_monitoring.detectors import zscore_by_class
from convergence_monitoring.framework import run_temporal_detectors
from convergence_monitoring.proposed_le import proposed_le_estimator


def test_proposed_le_returns_expected_fields():
    x = np.asarray([1.0, 0.82, 0.68, 0.57, 0.49, 0.43, 0.39], dtype=float)
    result = proposed_le_estimator(x=x, K=4, end_index=5, limit_method="next")
    assert "lambda_hat" in result
    assert "valid" in result


def test_detector_pipeline_shapes():
    score = np.asarray([
        [0.0, 0.1, 1.0, 1.1],
        [0.1, 0.2, 1.1, 1.3],
        [0.2, 0.3, 1.2, 1.5],
        [0.3, 0.4, 1.3, 1.7],
    ])
    labels = np.asarray([0, 0, 1, 1])
    z = zscore_by_class(score, labels, num_classes=2)
    out = run_temporal_detectors(
        z,
        tau=0.5,
        minrun_m=2,
        sliding_length=2,
        sliding_k=1,
        ewma_lambda=0.2,
    )
    assert out["minrun"]["decision"].shape == score.shape
    assert out["ewma"]["decision"].shape == score.shape
