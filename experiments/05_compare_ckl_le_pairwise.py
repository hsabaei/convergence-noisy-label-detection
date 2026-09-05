#!/usr/bin/env python
"""
Compare CKL with direct and pairwise LE-GIE noisy-label detection.

Primary scientific comparison
-----------------------------
CKL:
    class/distribution divergence score.

LE-GIE:
    ell_hat_i(t) = ell_err_i(t) + log(abs(m_GIE_i(t)))

Pairwise LE-GIE:
    compare signed LE-GIE values among samples sharing the same observed
    class, then accumulate the fraction of pairwise wins over training.

The pairwise detector intentionally stays simple:
    - same observed class peers
    - M peers/sample/epoch
    - no gamma/delta decomposition
    - no pairwise z-score
    - no Beta posterior
    - no learned threshold
    - higher LE-GIE = more noisy-like

The script evaluates:
    1) CKL, class-wise standardized
    2) direct LE-GIE, raw
    3) direct LE-GIE, class-wise standardized
    4) instantaneous pairwise LE-GIE win fraction
    5) cumulative pairwise LE-GIE win fraction

All methods are evaluated on the same noisy-label ground truth.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence_monitoring.detectors import (
    binary_auc_from_scores,
    pairwise_scores_by_class,
)
from convergence_monitoring.estimators import (
    rolling_class_reference_gie_batch,
)
from convergence_monitoring.framework import (
    standardize_monitoring_score,
)
from convergence_monitoring.proposed_le import (
    compose_le_from_error_and_m,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare CKL with direct and pairwise LE-GIE."
    )

    p.add_argument(
        "--common-npz",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument(
        "--ckl-npz",
        type=Path,
        default=Path(
            "results/ckl_scores/"
            "ckl_score_trajectories.npz"
        ),
    )
    p.add_argument(
        "--le-npz",
        type=Path,
        default=Path(
            "results/proposed_le_scores_aitken_k20/"
            "proposed_le_score_trajectories.npz"
        ),
        help=(
            "Corrected K=20 LE artifact used only for ell_err. "
            "Default is guarded-Aitken."
        ),
    )
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--n-peers", type=int, default=10)
    p.add_argument("--pairwise-seed", type=int, default=66)

    p.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/ckl_vs_le_gie_pairwise"
        ),
    )

    args = p.parse_args()

    if args.K < 6:
        p.error("--K must be at least 6.")
    if args.n_peers < 1:
        p.error("--n-peers must be positive.")
    for q in args.top_fractions:
        if not 0.0 < q < 1.0:
            p.error("--top-fractions values must be in (0,1).")

    return args


def require_npz(path: Path, required, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} NPZ not found: {path}"
        )

    d = np.load(path, allow_pickle=False)

    missing = [
        key for key in required
        if key not in d.files
    ]
    if missing:
        raise KeyError(
            f"{name} NPZ missing arrays: {missing}"
        )

    return d


def check_same_vector(name, a, b):
    if not np.array_equal(a, b):
        raise ValueError(
            f"{name} differs across input artifacts."
        )


def auc_from_labels(y, score):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=np.float64)

    return binary_auc_from_scores(
        s[y],
        s[~y],
    )


def finite_fraction(score):
    return float(
        np.mean(np.isfinite(score))
    )


def top_fraction_metrics(
    y_true,
    score,
    q,
):
    """Select the highest q*N finite scores and evaluate against noisy labels."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)

    N = y.size
    finite_idx = np.flatnonzero(
        np.isfinite(s)
    )

    target_k = int(
        max(1, round(float(q) * N))
    )
    k = min(
        target_k,
        finite_idx.size,
    )

    selected = np.zeros(
        N,
        dtype=bool,
    )

    if k > 0:
        finite_scores = s[finite_idx]

        # argpartition is enough because only the selected set matters.
        local = np.argpartition(
            finite_scores,
            -k,
        )[-k:]

        selected[
            finite_idx[local]
        ] = True

    TP = int(
        np.sum(selected & y)
    )
    FP = int(
        np.sum(selected & ~y)
    )
    FN = int(
        np.sum((~selected) & y)
    )
    TN = int(
        np.sum((~selected) & (~y))
    )

    tpr = (
        TP / (TP + FN)
        if TP + FN > 0
        else np.nan
    )
    fpr = (
        FP / (FP + TN)
        if FP + TN > 0
        else np.nan
    )
    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else np.nan
    )
    f1 = (
        2.0 * precision * tpr
        / (precision + tpr)
        if (
            np.isfinite(precision)
            and np.isfinite(tpr)
            and precision + tpr > 0
        )
        else np.nan
    )

    return {
        "q": float(q),
        "target_selected": target_k,
        "n_selected": int(np.sum(selected)),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "TPR": tpr,
        "FPR": fpr,
        "precision": precision,
        "F1": f1,
    }


def write_csv(path: Path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_auc(
    epochs,
    y,
    methods,
    common_start_col,
):
    rows = []

    for method_name, score in methods.items():

        for t in range(score.shape[1]):

            if t < common_start_col:
                auc = np.nan
            else:
                auc = auc_from_labels(
                    y,
                    score[:, t],
                )

            rows.append({
                "method": method_name,
                "epoch": int(epochs[t]),
                "auc_higher_is_noisy": auc,
                "finite_fraction": finite_fraction(
                    score[:, t]
                ),
            })

    return rows


def summarize_best_auc(rows):
    out = []

    methods = sorted(
        set(r["method"] for r in rows)
    )

    for method in methods:
        rr = [
            r for r in rows
            if (
                r["method"] == method
                and np.isfinite(
                    r["auc_higher_is_noisy"]
                )
            )
        ]

        if not rr:
            continue

        best = max(
            rr,
            key=lambda r:
                r["auc_higher_is_noisy"],
        )

        out.append({
            "method": method,
            "best_auc": best["auc_higher_is_noisy"],
            "best_epoch": best["epoch"],
            "finite_fraction_at_best": best[
                "finite_fraction"
            ],
        })

    return out


def summarize_topq(
    epochs,
    y,
    methods,
    top_fractions,
    common_start_col,
):
    rows = []

    for method_name, score in methods.items():

        for t in range(
            common_start_col,
            score.shape[1],
        ):

            for q in top_fractions:
                m = top_fraction_metrics(
                    y,
                    score[:, t],
                    q,
                )

                rows.append({
                    "method": method_name,
                    "epoch": int(epochs[t]),
                    **m,
                })

    return rows


def summarize_pairwise(
    epochs,
    pairwise,
    common_start_col,
):
    rows = []

    inst = pairwise[
        "valid_comparisons"
    ]
    cum = pairwise[
        "cumulative_comparisons"
    ]

    for t in range(
        common_start_col,
        inst.shape[1],
    ):
        rows.append({
            "epoch": int(epochs[t]),
            "mean_valid_pairs_this_epoch":
                float(np.mean(inst[:, t])),
            "fraction_with_at_least_one_pair":
                float(np.mean(inst[:, t] > 0)),
            "median_cumulative_pairs":
                float(np.median(cum[:, t])),
            "min_cumulative_pairs":
                int(np.min(cum[:, t])),
            "max_cumulative_pairs":
                int(np.max(cum[:, t])),
        })

    return rows


def plot_auc(
    epochs,
    methods,
    y,
    common_start_col,
    path,
):
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for name, score in methods.items():
        auc = np.full(
            score.shape[1],
            np.nan,
            dtype=np.float64,
        )

        for t in range(
            common_start_col,
            score.shape[1],
        ):
            auc[t] = auc_from_labels(
                y,
                score[:, t],
            )

        ax.plot(
            epochs,
            auc,
            label=name,
        )

    ax.axhline(
        0.5,
        linewidth=1,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "ROC-AUC (higher score = noisy)"
    )
    ax.set_title(
        "CKL vs direct and pairwise LE-GIE"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


def plot_top5_tpr(
    epochs,
    y,
    methods,
    common_start_col,
    path,
):
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for name, score in methods.items():

        vals = np.full(
            score.shape[1],
            np.nan,
            dtype=np.float64,
        )

        for t in range(
            common_start_col,
            score.shape[1],
        ):
            vals[t] = top_fraction_metrics(
                y,
                score[:, t],
                0.05,
            )["TPR"]

        ax.plot(
            epochs,
            vals,
            label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "TPR at top 5% selection"
    )
    ax.set_title(
        "Noisy-label recovery at the 5% operating point"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    common = require_npz(
        args.common_npz,
        required=(
            "sample_index",
            "epoch",
            "loss_traj",
            "observed_label",
            "true_label",
            "is_anomaly",
        ),
        name="common loss",
    )

    ckl = require_npz(
        args.ckl_npz,
        required=(
            "sample_index",
            "epoch",
            "observed_label",
            "is_anomaly",
            "ckl_traj",
        ),
        name="CKL",
    )

    le = require_npz(
        args.le_npz,
        required=(
            "sample_index",
            "epoch",
            "observed_label",
            "is_anomaly",
            "ell_err_traj",
        ),
        name="LE",
    )

    sample_index = np.asarray(
        common["sample_index"],
        dtype=np.int64,
    )
    epochs = np.asarray(
        common["epoch"],
        dtype=np.int64,
    )
    labels = np.asarray(
        common["observed_label"],
        dtype=np.int64,
    )
    true_label = np.asarray(
        common["true_label"],
        dtype=np.int64,
    )
    is_anomaly = np.asarray(
        common["is_anomaly"],
        dtype=bool,
    )
    loss_traj = np.asarray(
        common["loss_traj"],
        dtype=np.float64,
    )

    check_same_vector(
        "sample_index CKL/common",
        sample_index,
        np.asarray(
            ckl["sample_index"],
            dtype=np.int64,
        ),
    )
    check_same_vector(
        "sample_index LE/common",
        sample_index,
        np.asarray(
            le["sample_index"],
            dtype=np.int64,
        ),
    )
    check_same_vector(
        "epoch CKL/common",
        epochs,
        np.asarray(
            ckl["epoch"],
            dtype=np.int64,
        ),
    )
    check_same_vector(
        "epoch LE/common",
        epochs,
        np.asarray(
            le["epoch"],
            dtype=np.int64,
        ),
    )
    check_same_vector(
        "observed labels CKL/common",
        labels,
        np.asarray(
            ckl["observed_label"],
            dtype=np.int64,
        ),
    )
    check_same_vector(
        "observed labels LE/common",
        labels,
        np.asarray(
            le["observed_label"],
            dtype=np.int64,
        ),
    )
    check_same_vector(
        "noisy-label mask CKL/common",
        is_anomaly,
        np.asarray(
            ckl["is_anomaly"],
            dtype=bool,
        ),
    )
    check_same_vector(
        "noisy-label mask LE/common",
        is_anomaly,
        np.asarray(
            le["is_anomaly"],
            dtype=bool,
        ),
    )

    ckl_raw = np.asarray(
        ckl["ckl_traj"],
        dtype=np.float64,
    )
    ell_err = np.asarray(
        le["ell_err_traj"],
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # LE-GIE
    # --------------------------------------------------------------

    gie = rolling_class_reference_gie_batch(
        loss_traj,
        labels,
        K=args.K,
        num_classes=args.num_classes,
    )

    id_gie = np.asarray(
        gie["id_gie_traj"],
        dtype=np.float64,
    )

    le_gie = compose_le_from_error_and_m(
        ell_err,
        id_gie,
    )

    # Primary comparison begins when CKL and LE-GIE are both available.
    finite_ckl_by_t = np.any(
        np.isfinite(ckl_raw),
        axis=0,
    )
    finite_le_by_t = np.any(
        np.isfinite(le_gie),
        axis=0,
    )

    common_cols = np.flatnonzero(
        finite_ckl_by_t
        & finite_le_by_t
    )

    if common_cols.size == 0:
        raise RuntimeError(
            "CKL and LE-GIE have no common finite epoch."
        )

    common_start_col = int(
        common_cols[0]
    )
    common_start_epoch = int(
        epochs[common_start_col]
    )

    # Standardized direct scores use the same observed-class calibration.
    z_ckl = standardize_monitoring_score(
        ckl_raw.T,
        labels,
        direction="higher",
        num_classes=args.num_classes,
        start_index=common_start_col,
    ).T

    z_le_gie = standardize_monitoring_score(
        le_gie.T,
        labels,
        direction="higher",
        num_classes=args.num_classes,
        start_index=common_start_col,
    ).T

    # --------------------------------------------------------------
    # Professor-motivated pairwise LE detector.
    # --------------------------------------------------------------

    pairwise = pairwise_scores_by_class(
        le_gie,
        labels,
        n_peers=args.n_peers,
        start_index=common_start_col,
        seed=args.pairwise_seed,
    )

    pair_inst = np.asarray(
        pairwise["instantaneous_score"],
        dtype=np.float64,
    )
    pair_cum = np.asarray(
        pairwise["cumulative_score"],
        dtype=np.float64,
    )

    methods = {
        "CKL_z": z_ckl,
        "LE_GIE_direct_raw": le_gie,
        "LE_GIE_direct_z": z_le_gie,
        "LE_GIE_pairwise_instant": pair_inst,
        "LE_GIE_pairwise_cumulative": pair_cum,
    }

    auc_rows = summarize_auc(
        epochs,
        is_anomaly,
        methods,
        common_start_col,
    )
    topq_rows = summarize_topq(
        epochs,
        is_anomaly,
        methods,
        args.top_fractions,
        common_start_col,
    )
    best_rows = summarize_best_auc(
        auc_rows
    )
    pairwise_rows = summarize_pairwise(
        epochs,
        pairwise,
        common_start_col,
    )

    write_csv(
        args.output_dir
        / "auc_by_epoch.csv",
        auc_rows,
    )
    write_csv(
        args.output_dir
        / "best_auc_summary.csv",
        best_rows,
    )
    write_csv(
        args.output_dir
        / "topq_metrics_by_epoch.csv",
        topq_rows,
    )
    write_csv(
        args.output_dir
        / "pairwise_coverage_by_epoch.csv",
        pairwise_rows,
    )

    np.savez_compressed(
        args.output_dir
        / "comparison_score_trajectories.npz",
        sample_index=sample_index,
        epoch=epochs,
        observed_label=labels,
        true_label=true_label,
        is_anomaly=is_anomaly,
        ckl_raw=ckl_raw.astype(np.float32),
        z_ckl=z_ckl.astype(np.float32),
        ell_err=ell_err.astype(np.float32),
        id_gie=id_gie.astype(np.float32),
        le_gie=le_gie.astype(np.float32),
        z_le_gie=z_le_gie.astype(np.float32),
        pairwise_instantaneous=pair_inst.astype(np.float32),
        pairwise_cumulative=pair_cum.astype(np.float32),
        pairwise_valid_comparisons=np.asarray(
            pairwise["valid_comparisons"],
            dtype=np.int16,
        ),
        pairwise_cumulative_comparisons=np.asarray(
            pairwise["cumulative_comparisons"],
            dtype=np.int32,
        ),
        K=np.asarray(args.K, dtype=np.int64),
        n_peers=np.asarray(
            args.n_peers,
            dtype=np.int64,
        ),
        pairwise_seed=np.asarray(
            args.pairwise_seed,
            dtype=np.int64,
        ),
        common_start_col=np.asarray(
            common_start_col,
            dtype=np.int64,
        ),
        common_start_epoch=np.asarray(
            common_start_epoch,
            dtype=np.int64,
        ),
    )

    plot_auc(
        epochs,
        methods,
        is_anomaly,
        common_start_col,
        args.output_dir
        / "fig_auc_ckl_vs_le_pairwise.png",
    )

    plot_top5_tpr(
        epochs,
        is_anomaly,
        methods,
        common_start_col,
        args.output_dir
        / "fig_top5_tpr_ckl_vs_le_pairwise.png",
    )

    metadata = {
        "artifact":
            "ckl_vs_le_gie_pairwise_comparison",
        "common_npz":
            str(args.common_npz),
        "ckl_npz":
            str(args.ckl_npz),
        "le_npz":
            str(args.le_npz),
        "K":
            int(args.K),
        "le_formula":
            "ell_err + log(abs(m_GIE))",
        "gie_reference":
            "mean loss trajectory of samples sharing observed label",
        "pairwise_peer_scope":
            "same observed class",
        "pairwise_n_peers_per_sample_per_epoch":
            int(args.n_peers),
        "pairwise_seed":
            int(args.pairwise_seed),
        "pairwise_rule":
            "higher signed LE-GIE wins; tie contributes 0.5",
        "pairwise_accumulation":
            "cumulative wins / cumulative valid comparisons",
        "common_start_col_zero_based":
            int(common_start_col),
        "common_start_epoch_one_based":
            int(common_start_epoch),
        "primary_operating_point":
            "top 5% selection",
        "top_fractions":
            [float(q) for q in args.top_fractions],
        "n_samples":
            int(sample_index.size),
        "n_noisy":
            int(np.sum(is_anomaly)),
    }

    with (
        args.output_dir
        / "comparison_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("========================================")
    print("CKL vs LE-GIE pairwise comparison")
    print("========================================")
    print(
        f"Common monitoring start: epoch "
        f"{common_start_epoch}"
    )
    print(
        f"Pairwise peers/sample/epoch: "
        f"{args.n_peers}"
    )
    print()

    for row in best_rows:
        print(
            f"{row['method']:>28s} | "
            f"best AUC={row['best_auc']:.6f} "
            f"at epoch {row['best_epoch']}"
        )

    print()
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
