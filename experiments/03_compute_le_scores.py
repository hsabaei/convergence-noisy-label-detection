"""Compute rolling proposed-LE scores from the shared CIFAR-10 loss trajectories.

This experiment does not retrain the network.  It loads the common artifact
produced by ``01_generate_loss_trajectories.py`` and computes the proposed
Lyapunov-exponent estimate for every sample at every usable epoch.

The implementation follows ``convergence_monitoring.proposed_le`` exactly for
``numeric_backend='float64'`` but vectorizes the computation over samples.
For observation column t (0-based), endpoint n=t-1 is used so x[n+1]=x[t] is
available for the practical limit estimate.  Therefore, with window length K,
the first available score is at epoch K+2 (1-based).

Outputs include raw LE trajectories, the error-decay and FIE components,
class-wise z-scores for both possible anomaly directions, validity rates, and
epoch-wise ROC-AUC.  Both directions are reported deliberately: whether noisy
labels correspond to larger or smaller LE should be determined empirically,
not assumed in advance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convergence_monitoring.detectors import binary_auc_from_scores
from convergence_monitoring.framework import standardize_monitoring_score
from convergence_monitoring.proposed_le import rolling_proposed_le_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rolling proposed Lyapunov-exponent scores from shared loss trajectories."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "common_loss_trajectories"
        / "cifar10_noisy_label_loss_trajectories.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "proposed_le_scores",
    )
    parser.add_argument("--K", type=int, default=20, help="Proposed-LE window length.")
    parser.add_argument(
        "--limit-method",
        choices=("next", "aitken", "aitken_guarded"),
        default="next",
        help="Practical limit estimate used by the proposed estimator.",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    if args.K < 1:
        parser.error("--K must be positive.")
    if args.num_classes < 2:
        parser.error("--num-classes must be at least 2.")
    return args


def auc_by_epoch(
    score_nt: np.ndarray,
    is_anomaly: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return AUC for higher-is-noisy and lower-is-noisy directions."""
    score_nt = np.asarray(score_nt, dtype=np.float64)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    _, T = score_nt.shape

    auc_higher = np.full(T, np.nan, dtype=np.float64)
    auc_lower = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        s = score_nt[:, t]
        auc_higher[t] = binary_auc_from_scores(s[is_anomaly], s[~is_anomaly])
        auc_lower[t] = binary_auc_from_scores(-s[is_anomaly], -s[~is_anomaly])

    return auc_higher, auc_lower


def write_epoch_summary(
    path: Path,
    *,
    epochs: np.ndarray,
    lambda_traj: np.ndarray,
    valid_traj: np.ndarray,
    is_anomaly: np.ndarray,
    auc_higher: np.ndarray,
    auc_lower: np.ndarray,
) -> None:
    clean = ~is_anomaly
    noisy = is_anomaly

    fieldnames = [
        "epoch",
        "n_valid",
        "valid_fraction",
        "mean_le_clean",
        "median_le_clean",
        "mean_le_noisy",
        "median_le_noisy",
        "auc_higher_is_noisy",
        "auc_lower_is_noisy",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t, epoch in enumerate(epochs):
            s = lambda_traj[:, t]
            vc = np.isfinite(s) & clean
            vn = np.isfinite(s) & noisy
            v = valid_traj[:, t]

            writer.writerow(
                {
                    "epoch": int(epoch),
                    "n_valid": int(v.sum()),
                    "valid_fraction": float(v.mean()),
                    "mean_le_clean": float(np.mean(s[vc])) if np.any(vc) else np.nan,
                    "median_le_clean": float(np.median(s[vc])) if np.any(vc) else np.nan,
                    "mean_le_noisy": float(np.mean(s[vn])) if np.any(vn) else np.nan,
                    "median_le_noisy": float(np.median(s[vn])) if np.any(vn) else np.nan,
                    "auc_higher_is_noisy": float(auc_higher[t]),
                    "auc_lower_is_noisy": float(auc_lower[t]),
                }
            )


def main() -> None:
    args = parse_args()
    input_path = args.input_npz.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Shared loss-trajectory artifact not found: {input_path}\n"
            "Run experiments/01_generate_loss_trajectories.py first."
        )

    data = np.load(input_path, allow_pickle=False)
    required = {"loss_traj", "epoch", "observed_label", "true_label", "is_anomaly", "sample_index"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Input NPZ is missing required arrays: {missing}")

    loss_traj = np.asarray(data["loss_traj"], dtype=np.float64)
    epochs = np.asarray(data["epoch"], dtype=np.int64)
    observed_label = np.asarray(data["observed_label"], dtype=np.int64)
    true_label = np.asarray(data["true_label"], dtype=np.int64)
    is_anomaly = np.asarray(data["is_anomaly"], dtype=bool)
    sample_index = np.asarray(data["sample_index"], dtype=np.int64)

    if loss_traj.ndim != 2:
        raise ValueError(f"loss_traj must be [N,T]; found {loss_traj.shape}.")
    N, T = loss_traj.shape
    if epochs.shape != (T,):
        raise ValueError("epoch array does not match loss_traj time dimension.")
    for name, arr in {
        "observed_label": observed_label,
        "true_label": true_label,
        "is_anomaly": is_anomaly,
        "sample_index": sample_index,
    }.items():
        if arr.shape != (N,):
            raise ValueError(f"{name} must have shape ({N},), found {arr.shape}.")

    print("=== Proposed LE score experiment ===")
    print(f"input: {input_path}")
    print(f"loss_traj shape: {loss_traj.shape}")
    print(f"K: {args.K}")
    print(f"limit_method: {args.limit_method}")
    print(f"n_noisy: {int(is_anomaly.sum())}")

    le = rolling_proposed_le_batch(
        loss_traj,
        K=args.K,
        limit_method=args.limit_method,
    )

    lambda_traj = np.asarray(le["lambda_traj"], dtype=np.float64)
    valid_traj = np.asarray(le["valid_traj"], dtype=bool)
    first_col = int(le["first_available_column"])
    first_epoch = int(le["first_available_epoch"])

    # framework.py uses [T,N], whereas the trajectory artifact stores [N,T].
    z_higher_tn = standardize_monitoring_score(
        lambda_traj.T,
        observed_label,
        direction="higher",
        num_classes=args.num_classes,
        start_index=first_col,
    )
    z_lower_tn = standardize_monitoring_score(
        lambda_traj.T,
        observed_label,
        direction="lower",
        num_classes=args.num_classes,
        start_index=first_col,
    )

    auc_higher, auc_lower = auc_by_epoch(lambda_traj, is_anomaly)

    score_path = output_dir / "proposed_le_score_trajectories.npz"
    np.savez_compressed(
        score_path,
        sample_index=sample_index,
        epoch=epochs,
        observed_label=observed_label,
        true_label=true_label,
        is_anomaly=is_anomaly,
        lambda_traj=lambda_traj.astype(np.float32),
        ell_err_traj=np.asarray(le["ell_err_traj"], dtype=np.float32),
        id_fie_traj=np.asarray(le["id_fie_traj"], dtype=np.float32),
        limit_traj=np.asarray(le["limit_traj"], dtype=np.float32),
        valid_traj=valid_traj,
        aitken_fallback_traj=np.asarray(le["aitken_fallback_traj"], dtype=bool),
        z_higher_is_noisy=z_higher_tn.T.astype(np.float32),
        z_lower_is_noisy=z_lower_tn.T.astype(np.float32),
        auc_higher_is_noisy=auc_higher,
        auc_lower_is_noisy=auc_lower,
        K=np.asarray(args.K, dtype=np.int64),
        first_available_epoch=np.asarray(first_epoch, dtype=np.int64),
    )

    summary_path = output_dir / "proposed_le_epoch_summary.csv"
    write_epoch_summary(
        summary_path,
        epochs=epochs,
        lambda_traj=lambda_traj,
        valid_traj=valid_traj,
        is_anomaly=is_anomaly,
        auc_higher=auc_higher,
        auc_lower=auc_lower,
    )

    finite_h = np.flatnonzero(np.isfinite(auc_higher))
    finite_l = np.flatnonzero(np.isfinite(auc_lower))
    best_h = int(finite_h[np.nanargmax(auc_higher[finite_h])]) if finite_h.size else None
    best_l = int(finite_l[np.nanargmax(auc_lower[finite_l])]) if finite_l.size else None

    metadata = {
        "artifact": "proposed_le_score_trajectories",
        "input_npz": str(input_path),
        "K": int(args.K),
        "limit_method": args.limit_method,
        "numeric_backend": "float64-batch-from-src",
        "estimator_implementation": "convergence_monitoring.proposed_le.rolling_proposed_le_batch",
        "relative_component": "log(abs(ID_FIE_hat))",
        "trajectory_orientation": "arrays are [sample_index, epoch_index] = [N,T]",
        "first_available_column_zero_based": first_col,
        "first_available_epoch_one_based": first_epoch,
        "normalization": "within observed CIFAR-10 class at each epoch",
        "direction_policy": (
            "Both higher-is-noisy and lower-is-noisy are saved. "
            "Direction is not fixed before score-level evaluation."
        ),
        "n_samples": int(N),
        "n_epochs": int(T),
        "n_noisy": int(is_anomaly.sum()),
    }
    with (output_dir / "proposed_le_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved proposed-LE artifacts:")
    print(f"  scores: {score_path}")
    print(f"  epoch summary: {summary_path}")
    print(f"  config: {output_dir / 'proposed_le_config.json'}")
    print(f"  first available epoch: {first_epoch}")

    if best_h is not None:
        print(
            f"  best raw-score AUC (higher LE = noisy): "
            f"{auc_higher[best_h]:.6f} at epoch {epochs[best_h]}"
        )
    if best_l is not None:
        print(
            f"  best raw-score AUC (lower LE = noisy): "
            f"{auc_lower[best_l]:.6f} at epoch {epochs[best_l]}"
        )

    print(
        "\nDo not choose the anomaly direction from theory alone; compare the two "
        "AUC trajectories before running the temporal detector grid."
    )


if __name__ == "__main__":
    main()
