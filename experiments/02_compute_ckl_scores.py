"""Compute rolling CKL scores from the shared CIFAR-10 loss trajectories.

This experiment does not retrain the network. It loads the common artifact
produced by ``01_generate_loss_trajectories.py`` and computes the CKL-based
convergence score for every sample at every usable epoch.

For each epoch t and sample i:

1. Use the most recent ``K`` per-sample loss observations as phi_i.
2. Build a class-reference loss trajectory G_i from the observed/noisy label:
       G_i(s) = mean loss at epoch s among samples with observed label y_i.
3. Estimate the stabilized GIE dimension d_i and sample boundary W_i with the
   existing ``LIDEstimators.compute_GIE_LID`` formula.
4. Within each observed class, form geometric-mean references
       W_c = geom_mean(W_i), d_c = geom_mean(d_i).
5. Compute
       CKL_i = ckl_finite(W_i, d_i, W_c, d_c).
6. Standardize CKL within observed class at each epoch and report ROC-AUC.

Arrays are saved in the same [N,T] orientation as the shared loss artifact.

Important:
- This script uses the *same deterministic evaluation loss trajectories* as
  the proposed-LE experiment, so the CKL-vs-LE comparison is paired.
- The default GIE window is K=20, matching the framework's stated CKL/GIE
  window choice.
- The default reporting/monitoring start is epoch 40. Scores before epoch 40
  are still computed once K observations exist, but standardized detector
  input is left NaN before the chosen monitoring start.
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

from convergence_monitoring.estimators import EPS, ckl_finite
from convergence_monitoring.detectors import binary_auc_from_scores
from convergence_monitoring.framework import standardize_monitoring_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rolling CKL scores from shared noisy-label CIFAR-10 loss trajectories."
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
        default=REPO_ROOT / "results" / "ckl_scores",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=20,
        help="Rolling GIE/CKL window length.",
    )
    parser.add_argument(
        "--monitor-start-epoch",
        type=int,
        default=40,
        help="One-based epoch at which standardized monitoring output begins.",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    if args.K < 6:
        parser.error("--K must be at least 6 for the existing GIE estimator.")
    if args.monitor_start_epoch < args.K:
        parser.error("--monitor-start-epoch must be >= K.")
    if args.num_classes < 2:
        parser.error("--num-classes must be at least 2.")
    return args


def build_class_reference(loss_traj: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return class-mean loss trajectory, shape [N,T], using observed labels."""
    loss_traj = np.asarray(loss_traj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    N, T = loss_traj.shape
    G = np.full((N, T), np.nan, dtype=np.float64)

    for c in range(num_classes):
        idx = labels == c
        if not np.any(idx):
            continue
        vals = loss_traj[idx]  # [Nc,T]
        with np.errstate(invalid="ignore"):
            mean_t = np.nanmean(vals, axis=0)
        G[idx] = mean_t[None, :]

    return G


def vectorized_gie_window(
    phi: np.ndarray,
    G: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of LIDEstimators.compute_GIE_LID over rows.

    Parameters
    ----------
    phi, G:
        Arrays of shape [N,K].

    Returns
    -------
    d, W:
        GIE estimate and sample boundary for each row.
    """
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    if phi.shape != G.shape or phi.ndim != 2:
        raise ValueError("phi and G must have matching shape [N,K].")

    N, K = phi.shape
    epsilon = 1e-7

    # Existing estimator uses mean of the last three observations as limit.
    phi_limit = np.mean(phi[:, -3:], axis=1)
    G_limit = np.mean(G[:, -3:], axis=1)

    R = np.abs(phi - phi_limit[:, None])
    FR = np.abs(G - G_limit[:, None])

    with np.errstate(invalid="ignore"):
        w0 = np.max(R, axis=1)
        w1 = np.max(FR, axis=1)

    W = w0.copy()
    d = np.full(N, np.nan, dtype=np.float64)

    # Paired filtering, exactly as in the estimator.
    mask = (
        (R > eps)
        & (FR > eps)
        & np.isfinite(R)
        & np.isfinite(FR)
    )
    n_nonzero = mask.sum(axis=1)
    k_internal = n_nonzero - 1

    base_ok = (
        (k_internal > 4)
        & np.isfinite(w0)
        & np.isfinite(w1)
        & (w0 > 0.0)
        & (w1 > 0.0)
    )
    if not np.any(base_ok):
        return d, W

    denom_num = np.zeros(N, dtype=np.float64)
    denom_den = np.zeros(N, dtype=np.float64)

    safe_w0 = np.where(base_ok, w0 + epsilon, 1.0)
    safe_w1 = np.where(base_ok, w1 + epsilon, 1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_num = np.where(mask, np.log(np.abs(R / safe_w0[:, None])), 0.0)
        log_den = np.where(mask, np.log(np.abs(FR / safe_w1[:, None])), 0.0)

    denom_num = np.sum(log_num, axis=1)
    denom_den = np.sum(log_den, axis=1)

    ok = (
        base_ok
        & np.isfinite(denom_num)
        & np.isfinite(denom_den)
        & (np.abs(denom_num) >= eps)
        & (np.abs(denom_den) >= eps)
    )
    if not np.any(ok):
        return d, W

    hill_num = np.full(N, np.nan, dtype=np.float64)
    hill_den = np.full(N, np.nan, dtype=np.float64)

    hill_num[ok] = -(k_internal[ok] / denom_num[ok])
    hill_den[ok] = -(k_internal[ok] / denom_den[ok])

    good = (
        ok
        & np.isfinite(hill_num)
        & np.isfinite(hill_den)
        & (np.abs(hill_den) > eps)
    )
    d[good] = hill_num[good] / hill_den[good]

    # The scalar implementation can return negative/nonfinite values; CKL
    # itself requires strictly positive parameters, so invalid CKL inputs are
    # filtered at the next stage rather than silently altered here.
    return d, W


def class_geometric_references(
    W: np.ndarray,
    d: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Class geometric means used by the existing CKL implementation."""
    W_cls = np.full(num_classes, np.nan, dtype=np.float64)
    d_cls = np.full(num_classes, np.nan, dtype=np.float64)

    for c in range(num_classes):
        idx = labels == c

        w = W[idx]
        w = w[np.isfinite(w) & (w > 0.0)]
        if w.size:
            W_cls[c] = float(np.exp(np.mean(np.log(np.maximum(w, eps)))))

        dc = d[idx]
        dc = dc[np.isfinite(dc) & (dc > 0.0)]
        if dc.size:
            d_cls[c] = float(np.exp(np.mean(np.log(np.maximum(dc, eps)))))

    return W_cls, d_cls


def compute_ckl_trajectories(
    loss_traj: np.ndarray,
    observed_label: np.ndarray,
    *,
    K: int,
    num_classes: int,
) -> dict[str, np.ndarray | int]:
    """Compute W, d, class references, and CKL in [N,T] orientation."""
    x = np.asarray(loss_traj, dtype=np.float64)
    labels = np.asarray(observed_label, dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(f"loss_traj must be [N,T], got {x.shape}.")
    N, T = x.shape
    if labels.shape != (N,):
        raise ValueError(f"observed_label must have shape ({N},).")
    if T < K:
        raise ValueError(f"Need at least K={K} epochs; found T={T}.")

    G = build_class_reference(x, labels, num_classes)

    W_traj = np.full((N, T), np.nan, dtype=np.float64)
    d_traj = np.full((N, T), np.nan, dtype=np.float64)
    ckl_traj = np.full((N, T), np.nan, dtype=np.float64)
    valid_traj = np.zeros((N, T), dtype=bool)

    W_class_traj = np.full((num_classes, T), np.nan, dtype=np.float64)
    d_class_traj = np.full((num_classes, T), np.nan, dtype=np.float64)

    first_col = K - 1

    for t in range(first_col, T):
        start = t - K + 1
        phi_w = x[:, start : t + 1]
        G_w = G[:, start : t + 1]

        d_t, W_t = vectorized_gie_window(phi_w, G_w)
        W_cls, d_cls = class_geometric_references(
            W_t, d_t, labels, num_classes=num_classes
        )

        ckl_t = np.full(N, np.nan, dtype=np.float64)

        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            if not (
                np.isfinite(W_cls[c]) and W_cls[c] > 0.0
                and np.isfinite(d_cls[c]) and d_cls[c] > 0.0
            ):
                continue

            for i in idx:
                if not (
                    np.isfinite(W_t[i]) and W_t[i] > 0.0
                    and np.isfinite(d_t[i]) and d_t[i] > 0.0
                ):
                    continue
                ckl_t[i] = ckl_finite(
                    W_t[i], d_t[i], W_cls[c], d_cls[c]
                )

        valid = np.isfinite(ckl_t)

        W_traj[:, t] = W_t
        d_traj[:, t] = d_t
        ckl_traj[:, t] = ckl_t
        valid_traj[:, t] = valid
        W_class_traj[:, t] = W_cls
        d_class_traj[:, t] = d_cls

        epoch = t + 1
        if epoch == K or epoch % 10 == 0 or epoch == T:
            print(
                f"epoch={epoch:03d}/{T} "
                f"valid_CKL={int(valid.sum())}/{N} "
                f"({100.0 * valid.mean():.2f}%)"
            )

    return {
        "class_reference_loss_traj": G,
        "W_traj": W_traj,
        "d_traj": d_traj,
        "ckl_traj": ckl_traj,
        "valid_traj": valid_traj,
        "W_class_traj": W_class_traj,
        "d_class_traj": d_class_traj,
        "first_available_column": first_col,
        "first_available_epoch": K,
    }


def auc_by_epoch(
    score_nt: np.ndarray,
    is_anomaly: np.ndarray,
) -> np.ndarray:
    score_nt = np.asarray(score_nt, dtype=np.float64)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)

    _, T = score_nt.shape
    auc = np.full(T, np.nan, dtype=np.float64)

    for t in range(T):
        s = score_nt[:, t]
        auc[t] = binary_auc_from_scores(
            s[is_anomaly],
            s[~is_anomaly],
        )
    return auc


def write_epoch_summary(
    path: Path,
    *,
    epochs: np.ndarray,
    ckl_traj: np.ndarray,
    valid_traj: np.ndarray,
    is_anomaly: np.ndarray,
    auc_raw: np.ndarray,
    auc_z: np.ndarray,
) -> None:
    clean = ~is_anomaly
    noisy = is_anomaly

    fieldnames = [
        "epoch",
        "n_valid",
        "valid_fraction",
        "mean_ckl_clean",
        "median_ckl_clean",
        "mean_ckl_noisy",
        "median_ckl_noisy",
        "auc_ckl",
        "auc_z_ckl",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t, epoch in enumerate(epochs):
            s = ckl_traj[:, t]
            vc = np.isfinite(s) & clean
            vn = np.isfinite(s) & noisy
            v = valid_traj[:, t]

            writer.writerow(
                {
                    "epoch": int(epoch),
                    "n_valid": int(v.sum()),
                    "valid_fraction": float(v.mean()),
                    "mean_ckl_clean": float(np.mean(s[vc])) if np.any(vc) else np.nan,
                    "median_ckl_clean": float(np.median(s[vc])) if np.any(vc) else np.nan,
                    "mean_ckl_noisy": float(np.mean(s[vn])) if np.any(vn) else np.nan,
                    "median_ckl_noisy": float(np.median(s[vn])) if np.any(vn) else np.nan,
                    "auc_ckl": float(auc_raw[t]),
                    "auc_z_ckl": float(auc_z[t]),
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
    required = {
        "loss_traj",
        "epoch",
        "observed_label",
        "true_label",
        "is_anomaly",
        "sample_index",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Input NPZ is missing required arrays: {missing}")

    loss_traj = np.asarray(data["loss_traj"], dtype=np.float64)
    epochs = np.asarray(data["epoch"], dtype=np.int64)
    observed_label = np.asarray(data["observed_label"], dtype=np.int64)
    true_label = np.asarray(data["true_label"], dtype=np.int64)
    is_anomaly = np.asarray(data["is_anomaly"], dtype=bool)
    sample_index = np.asarray(data["sample_index"], dtype=np.int64)

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

    print("=== CKL score experiment ===")
    print(f"input: {input_path}")
    print(f"loss_traj shape: {loss_traj.shape}")
    print(f"K: {args.K}")
    print(f"monitor_start_epoch: {args.monitor_start_epoch}")
    print(f"n_noisy: {int(is_anomaly.sum())}")

    out = compute_ckl_trajectories(
        loss_traj,
        observed_label,
        K=args.K,
        num_classes=args.num_classes,
    )

    ckl_traj = np.asarray(out["ckl_traj"], dtype=np.float64)
    valid_traj = np.asarray(out["valid_traj"], dtype=bool)

    monitor_start_col = args.monitor_start_epoch - 1

    # framework.py expects [T,N].
    z_ckl_tn = standardize_monitoring_score(
        ckl_traj.T,
        observed_label,
        direction="higher",
        num_classes=args.num_classes,
        start_index=monitor_start_col,
    )
    z_ckl_nt = z_ckl_tn.T

    auc_raw = auc_by_epoch(ckl_traj, is_anomaly)
    auc_z = auc_by_epoch(z_ckl_nt, is_anomaly)

    score_path = output_dir / "ckl_score_trajectories.npz"
    np.savez_compressed(
        score_path,
        sample_index=sample_index,
        epoch=epochs,
        observed_label=observed_label,
        true_label=true_label,
        is_anomaly=is_anomaly,
        ckl_traj=ckl_traj.astype(np.float32),
        z_ckl=z_ckl_nt.astype(np.float32),
        W_traj=np.asarray(out["W_traj"], dtype=np.float32),
        d_traj=np.asarray(out["d_traj"], dtype=np.float32),
        W_class_traj=np.asarray(out["W_class_traj"], dtype=np.float32),
        d_class_traj=np.asarray(out["d_class_traj"], dtype=np.float32),
        valid_traj=valid_traj,
        auc_ckl=auc_raw,
        auc_z_ckl=auc_z,
        K=np.asarray(args.K, dtype=np.int64),
        first_available_epoch=np.asarray(out["first_available_epoch"], dtype=np.int64),
        monitor_start_epoch=np.asarray(args.monitor_start_epoch, dtype=np.int64),
    )

    summary_path = output_dir / "ckl_epoch_summary.csv"
    write_epoch_summary(
        summary_path,
        epochs=epochs,
        ckl_traj=ckl_traj,
        valid_traj=valid_traj,
        is_anomaly=is_anomaly,
        auc_raw=auc_raw,
        auc_z=auc_z,
    )

    finite_raw = np.flatnonzero(np.isfinite(auc_raw))
    finite_z = np.flatnonzero(np.isfinite(auc_z))
    best_raw = int(finite_raw[np.nanargmax(auc_raw[finite_raw])]) if finite_raw.size else None
    best_z = int(finite_z[np.nanargmax(auc_z[finite_z])]) if finite_z.size else None

    metadata = {
        "artifact": "ckl_score_trajectories",
        "input_npz": str(input_path),
        "trajectory_orientation": "arrays are [sample_index, epoch_index] = [N,T]",
        "K": int(args.K),
        "first_available_epoch_one_based": int(out["first_available_epoch"]),
        "monitor_start_epoch_one_based": int(args.monitor_start_epoch),
        "sample_dimension_estimator": "existing stabilized GIE formula",
        "sample_boundary": "max absolute residual to last-3 mean within rolling K-window",
        "G_reference": "mean loss trajectory of samples sharing the observed/noisy CIFAR-10 label",
        "class_reference": "geometric mean of positive W_i and d_i within observed class",
        "ckl_direction": "larger CKL is more anomalous",
        "normalization": "within observed CIFAR-10 class at each epoch",
        "n_samples": int(N),
        "n_epochs": int(T),
        "n_noisy": int(is_anomaly.sum()),
        "best_raw_auc_epoch": int(epochs[best_raw]) if best_raw is not None else None,
        "best_raw_auc": float(auc_raw[best_raw]) if best_raw is not None else None,
        "best_z_auc_epoch": int(epochs[best_z]) if best_z is not None else None,
        "best_z_auc": float(auc_z[best_z]) if best_z is not None else None,
    }

    config_path = output_dir / "ckl_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved CKL artifacts:")
    print(f"  scores:  {score_path}")
    print(f"  summary: {summary_path}")
    print(f"  config:  {config_path}")

    if best_raw is not None:
        print(
            f"\nBest raw CKL AUC: {auc_raw[best_raw]:.6f} "
            f"at epoch {int(epochs[best_raw])}"
        )
    if best_z is not None:
        print(
            f"Best z(CKL) AUC: {auc_z[best_z]:.6f} "
            f"at epoch {int(epochs[best_z])}"
        )


if __name__ == "__main__":
    main()
