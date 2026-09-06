import numpy as np

from convergence_monitoring.estimators import (
    EPS,
    ckl_finite,
    build_class_reference_trajectory,
    vectorized_gie_window,
    rolling_class_reference_gie_batch,
)
from convergence_monitoring.detectors import (
    binary_auc_from_scores,
    pairwise_scores_by_class,
    exact_pairwise_scores_by_class,
    compute_minrun_m,
    chernoff_window_k,
)


def test_legacy_symbols_still_import():
    # Earlier CKL / LE / detector experiments import these names.
    assert EPS > 0
    assert callable(ckl_finite)
    assert callable(binary_auc_from_scores)
    assert callable(pairwise_scores_by_class)
    assert callable(exact_pairwise_scores_by_class)
    assert callable(compute_minrun_m)
    assert callable(chernoff_window_k)


def test_default_class_reference_is_legacy_mean():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [10.0, 8.0, 6.0],
            [14.0, 12.0, 10.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    got = build_class_reference_trajectory(
        x,
        labels,
        num_classes=2,
    )

    expected = np.array(
        [
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
            [12.0, 10.0, 8.0],
            [12.0, 10.0, 8.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(got, expected)


def test_default_gie_equals_explicit_legacy_options():
    rng = np.random.default_rng(66)
    x = np.abs(rng.normal(size=(40, 25))) + 0.05
    labels = np.repeat(np.arange(4), 10)

    default = rolling_class_reference_gie_batch(
        x,
        labels,
        K=10,
        num_classes=4,
    )

    explicit = rolling_class_reference_gie_batch(
        x,
        labels,
        K=10,
        num_classes=4,
        reference_method="mean",
        trim_fraction=0.10,
        limit_method="last3_mean",
    )

    for key in ("id_gie_traj", "W_gie_traj"):
        np.testing.assert_array_equal(
            default[key],
            explicit[key],
        )

    np.testing.assert_array_equal(
        default["valid_traj"],
        explicit["valid_traj"],
    )

    assert default["first_available_column"] == 9
    assert default["first_available_epoch"] == 10
    assert default["reference"] == "observed-class mean loss trajectory"


def test_vectorized_gie_default_limit_is_legacy_last3_mean():
    rng = np.random.default_rng(7)
    phi = np.abs(rng.normal(size=(12, 10))) + 0.1
    G = np.abs(rng.normal(size=(12, 10))) + 0.1

    d_default, w_default = vectorized_gie_window(phi, G)

    phi_limit = np.mean(phi[:, -3:], axis=1)
    G_limit = np.mean(G[:, -3:], axis=1)

    d_explicit, w_explicit = vectorized_gie_window(
        phi,
        G,
        phi_limit=phi_limit,
        G_limit=G_limit,
    )

    np.testing.assert_array_equal(d_default, d_explicit)
    np.testing.assert_array_equal(w_default, w_explicit)


def test_exact_pairwise_matches_bruteforce_with_ties_and_nans():
    score = np.array(
        [
            [0.1, 0.2, 0.2],
            [0.1, 0.4, 0.3],
            [0.5, 0.1, 0.3],
            [0.2, 0.8, np.nan],
            [0.7, 0.6, 0.9],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    out = exact_pairwise_scores_by_class(
        score,
        labels,
        start_index=0,
    )
    got = out["instantaneous_score"]

    expected = np.full_like(score, np.nan)

    for t in range(score.shape[1]):
        for i in range(score.shape[0]):
            if not np.isfinite(score[i, t]):
                continue

            peers = np.flatnonzero(
                (labels == labels[i])
                & (np.arange(labels.size) != i)
                & np.isfinite(score[:, t])
            )
            if peers.size == 0:
                continue

            wins = (
                (score[i, t] > score[peers, t]).astype(float)
                + 0.5
                * (score[i, t] == score[peers, t]).astype(float)
            )
            expected[i, t] = np.mean(wins)

    np.testing.assert_array_equal(got, expected)
