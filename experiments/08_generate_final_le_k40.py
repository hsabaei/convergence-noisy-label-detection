#!/usr/bin/env python
"""
Generate the frozen final LE-GIE score artifact.

Frozen estimator:
    K = 40
    class reference = observed-class mean
    limit rule = mean of latest 3 observations

The same sample L_hat is used consistently in ell_err and sample-side GIE.

This script writes a NEW artifact and never overwrites the previous K=20
experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from convergence_monitoring.consistent_le import (
    rolling_common_limit_le_gie_batch,
)
from convergence_monitoring.detectors import binary_auc_from_scores
from convergence_monitoring.framework import standardize_monitoring_score


FROZEN_K = 40
FROZEN_REFERENCE = "mean"
FROZEN_LIMIT = "last3_mean"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--common-npz",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/proposed_le_scores_consistent_k40"
        ),
    )

    return p.parse_args()


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(s)

    if np.sum(finite) < 2:
        return np.nan

    yy = y[finite]
    if np.all(yy) or np.all(~yy):
        return np.nan

    return binary_auc_from_scores(
        s[finite & y],
        s[finite & ~y],
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.common_npz, allow_pickle=False)

    required = (
        "sample_index",
        "epoch",
        "loss_traj",
        "observed_label",
        "true_label",
        "is_anomaly",
    )
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"Common NPZ missing arrays: {missing}")

    sample_index = np.asarray(d["sample_index"], dtype=np.int64)
    epochs = np.asarray(d["epoch"], dtype=np.int64)
    loss = np.asarray(d["loss_traj"], dtype=np.float64)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    true_label = np.asarray(d["true_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)

    print("========================================")
    print("Frozen final LE-GIE artifact")
    print("========================================")
    print(f"K: {FROZEN_K}")
    print(f"class reference: {FROZEN_REFERENCE}")
    print(f"limit rule: {FROZEN_LIMIT}")
    print("common L_hat: enforced across ell_err and sample-side GIE")

    out = rolling_common_limit_le_gie_batch(
        loss,
        labels,
        K=FROZEN_K,
        num_classes=args.num_classes,
        reference_method=FROZEN_REFERENCE,
        limit_method=FROZEN_LIMIT,
    )

    le = np.asarray(out["le_gie_traj"], dtype=np.float64)
    ell_err = np.asarray(out["ell_err_traj"], dtype=np.float64)
    id_gie = np.asarray(out["id_gie_traj"], dtype=np.float64)
    valid = np.asarray(out["valid_traj"], dtype=bool)

    start_col = int(np.asarray(out["first_available_column"]))
    start_epoch = int(np.asarray(out["first_available_epoch"]))

    z_tn = standardize_monitoring_score(
        le.T,
        labels,
        direction="higher",
        num_classes=args.num_classes,
        start_index=start_col,
    )
    z_le = z_tn.T

    rows = []
    raw_auc = np.full(epochs.size, np.nan)
    z_auc = np.full(epochs.size, np.nan)

    for t, epoch in enumerate(epochs):
        raw_auc[t] = auc_binary(y, le[:, t])
        z_auc[t] = auc_binary(y, z_le[:, t])

        if t >= start_col:
            rows.append({
                "epoch": int(epoch),
                "raw_le_auc": float(raw_auc[t]) if np.isfinite(raw_auc[t]) else np.nan,
                "z_le_auc": float(z_auc[t]) if np.isfinite(z_auc[t]) else np.nan,
                "finite_fraction": float(np.mean(np.isfinite(le[:, t]))),
            })

    raw_finite = np.flatnonzero(np.isfinite(raw_auc))
    z_finite = np.flatnonzero(np.isfinite(z_auc))

    raw_best_t = int(raw_finite[np.nanargmax(raw_auc[raw_finite])])
    z_best_t = int(z_finite[np.nanargmax(z_auc[z_finite])])

    artifact = args.output_dir / "proposed_le_score_trajectories.npz"

    np.savez_compressed(
        artifact,
        sample_index=sample_index,
        epoch=epochs,
        observed_label=labels,
        true_label=true_label,
        is_anomaly=y,
        ell_err_traj=ell_err.astype(np.float32),
        id_gie_traj=id_gie.astype(np.float32),
        le_gie_traj=le.astype(np.float32),
        lambda_traj=le.astype(np.float32),  # explicit alias
        z_le_gie_traj=z_le.astype(np.float32),
        sample_limit_traj=np.asarray(
            out["sample_limit_traj"],
            dtype=np.float32,
        ),
        reference_limit_traj=np.asarray(
            out["reference_limit_traj"],
            dtype=np.float32,
        ),
        valid_traj=valid,
        first_available_column=np.asarray(start_col, dtype=np.int64),
        first_available_epoch=np.asarray(start_epoch, dtype=np.int64),
        K=np.asarray(FROZEN_K, dtype=np.int64),
        common_limit_enforced=np.asarray(True),
        reference_method=np.asarray(FROZEN_REFERENCE),
        limit_method=np.asarray(FROZEN_LIMIT),
    )

    csv_path = args.output_dir / "proposed_le_epoch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "raw_le_auc",
                "z_le_auc",
                "finite_fraction",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    config = {
        "artifact": "frozen_final_consistent_LE_GIE",
        "theoretical_estimator":
            "ell_hat = ell_err_hat + log(abs(m_GIE_hat))",
        "K": FROZEN_K,
        "reference_method": FROZEN_REFERENCE,
        "limit_method": FROZEN_LIMIT,
        "common_limit_enforced": True,
        "common_limit_statement":
            "same sample L_hat used in ell_err and sample-side GIE; same rule on class reference",
        "window_definition":
            "at latest t: boundary=x[t-K-1], tail=x[t-K:t], L_hat uses observations through x[t]",
        "first_available_epoch": start_epoch,
        "raw_best_auc": float(raw_auc[raw_best_t]),
        "raw_best_auc_epoch": int(epochs[raw_best_t]),
        "z_best_auc": float(z_auc[z_best_t]),
        "z_best_auc_epoch": int(epochs[z_best_t]),
        "source_common_npz": str(args.common_npz),
        "previous_artifacts_overwritten": False,
    }

    with (
        args.output_dir / "proposed_le_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"First available epoch: {start_epoch}")
    print(
        f"Best raw LE AUC: {raw_auc[raw_best_t]:.6f} "
        f"@ epoch {epochs[raw_best_t]}"
    )
    print(
        f"Best class-z LE AUC: {z_auc[z_best_t]:.6f} "
        f"@ epoch {epochs[z_best_t]}"
    )
    print(f"Saved: {artifact}")


if __name__ == "__main__":
    main()
