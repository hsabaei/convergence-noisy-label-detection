#!/usr/bin/env python
"""
Diagnostic comparison of Proposed-Next vs Proposed-Aitken-Guarded LE scores.

This script DOES NOT re-estimate LE. It only analyzes already-computed score
artifacts produced by experiments/03_compute_le_scores.py.

For each LE variant and epoch it evaluates:
    1) ell_err_hat
    2) log(abs(ID_FIE_hat))
    3) final signed LE = ell_err_hat + log(abs(ID_FIE_hat))

Outputs:
    le_component_epoch_summary.csv
    le_component_best_auc_summary.csv
    le_variant_pairwise_summary.csv
    le_validity_summary.csv

    fig_auc_components_next.png
    fig_auc_components_aitken.png
    fig_medians_components_next.png
    fig_medians_components_aitken.png
    fig_final_le_auc_next_vs_aitken.png
    fig_valid_fraction_next_vs_aitken.png
    fig_aitken_fallback_fraction.png   (when available)

Important:
    AUC is reported in both orientations:
        higher score = noisy
        lower score  = noisy

    The signed LE itself is never replaced by abs(LE).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the repository's src-layout importable when this file is executed
# directly as: python experiments/04_analyze_le_components.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence_monitoring.framework import standardize_monitoring_score
from convergence_monitoring.estimators import rolling_class_reference_gie_batch


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze LE components for next vs guarded-Aitken runs."
    )
    parser.add_argument(
        "--next-npz",
        type=Path,
        default=Path(
            "results/proposed_le_scores_next_k20/"
            "proposed_le_score_trajectories.npz"
        ),
        help="Corrected Proposed-Next K=20 NPZ.",
    )
    parser.add_argument(
        "--aitken-npz",
        type=Path,
        default=Path(
            "results/proposed_le_scores_aitken_k20/"
            "proposed_le_score_trajectories.npz"
        ),
        help="Corrected Proposed-Aitken-Guarded K=20 NPZ.",
    )
    parser.add_argument(
        "--common-npz",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
        help="Common loss-trajectory artifact used to compute class-reference GIE.",
    )
    parser.add_argument(
        "--gie-K",
        type=int,
        default=20,
        help="Rolling class-reference GIE window length.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/le_component_diagnostics"),
    )
    return parser.parse_args()


def binary_auc_from_scores(y_true, scores):
    """
    ROC-AUC from ranks. Returns NaN if either class is absent.
    """
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)

    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]

    if y.size == 0:
        return np.nan

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]

    ranks = np.empty_like(s, dtype=np.float64)

    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and sorted_s[j] == sorted_s[i]:
            j += 1
        # average 1-based rank for ties
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = np.sum(ranks[y == 1])
    auc = (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def auc_pair(y_true, scores):
    auc_higher = binary_auc_from_scores(y_true, scores)
    auc_lower = binary_auc_from_scores(y_true, -np.asarray(scores))
    return auc_higher, auc_lower


def safe_log_abs(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)

    ok = np.isfinite(x) & (x != 0.0)
    out[ok] = np.log(np.abs(x[ok]))
    return out


def median_by_group(values, is_anomaly):
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(is_anomaly, dtype=bool)

    clean = values[(~y) & np.isfinite(values)]
    noisy = values[y & np.isfinite(values)]

    return (
        float(np.median(clean)) if clean.size else np.nan,
        float(np.median(noisy)) if noisy.size else np.nan,
        int(clean.size),
        int(noisy.size),
    )


def load_variant(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} NPZ not found: {path}")

    d = np.load(path, allow_pickle=False)

    required = [
        "epoch",
        "observed_label",
        "is_anomaly",
        "lambda_traj",
        "ell_err_traj",
        "id_fie_traj",
        "valid_traj",
    ]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(
            f"{name} NPZ is missing required arrays: {missing}"
        )

    out = {
        "name": name,
        "epoch": np.asarray(d["epoch"], dtype=np.int64),
        "observed_label": np.asarray(d["observed_label"], dtype=np.int64),
        "is_anomaly": np.asarray(d["is_anomaly"], dtype=bool),
        "lambda": np.asarray(d["lambda_traj"], dtype=np.float64),
        "ell_err": np.asarray(d["ell_err_traj"], dtype=np.float64),
        "id_fie": np.asarray(d["id_fie_traj"], dtype=np.float64),
        "valid": np.asarray(d["valid_traj"], dtype=bool),
    }

    out["log_abs_id"] = safe_log_abs(out["id_fie"])

    # Reconstruct final LE independently as a consistency diagnostic.
    with np.errstate(invalid="ignore"):
        out["lambda_reconstructed"] = (
            out["ell_err"] + out["log_abs_id"]
        )

    if "aitken_fallback_traj" in d.files:
        out["aitken_fallback"] = np.asarray(
            d["aitken_fallback_traj"], dtype=bool
        )
    else:
        out["aitken_fallback"] = None

    # Shape checks.
    N, T = out["lambda"].shape
    if out["epoch"].shape != (T,):
        raise ValueError(
            f"{name}: epoch has shape {out['epoch'].shape}; expected {(T,)}"
        )
    if out["observed_label"].shape != (N,):
        raise ValueError(
            f"{name}: observed_label has shape {out['observed_label'].shape}; "
            f"expected {(N,)}"
        )

    if out["is_anomaly"].shape != (N,):
        raise ValueError(
            f"{name}: is_anomaly has shape {out['is_anomaly'].shape}; "
            f"expected {(N,)}"
        )

    for key in ("ell_err", "id_fie", "log_abs_id", "valid"):
        if out[key].shape != (N, T):
            raise ValueError(
                f"{name}: {key} has shape {out[key].shape}; "
                f"expected {(N, T)}"
            )

    # Formula consistency check.
    mask = (
        np.isfinite(out["lambda"])
        & np.isfinite(out["lambda_reconstructed"])
    )
    if np.any(mask):
        out["max_formula_abs_diff"] = float(
            np.max(
                np.abs(
                    out["lambda"][mask]
                    - out["lambda_reconstructed"][mask]
                )
            )
        )
    else:
        out["max_formula_abs_diff"] = np.nan

    return out


def summarize_variant(v):
    rows = []
    y = v["is_anomaly"]
    epoch = v["epoch"]
    N = y.size

    components = {
        "ell_err": v["ell_err"],
        "log_abs_id_fie": v["log_abs_id"],
        "lambda": v["lambda"],
    }

    for t_idx, ep in enumerate(epoch):
        valid_count = int(np.sum(v["valid"][:, t_idx]))
        valid_fraction = valid_count / N

        fallback_count = (
            int(np.sum(v["aitken_fallback"][:, t_idx]))
            if v["aitken_fallback"] is not None
            else 0
        )
        fallback_fraction = fallback_count / N

        for comp_name, arr in components.items():
            vals = arr[:, t_idx]
            auc_h, auc_l = auc_pair(y, vals)
            med_clean, med_noisy, n_clean, n_noisy = median_by_group(
                vals, y
            )

            rows.append({
                "variant": v["name"],
                "epoch": int(ep),
                "component": comp_name,
                "auc_higher_is_noisy": auc_h,
                "auc_lower_is_noisy": auc_l,
                "median_clean": med_clean,
                "median_noisy": med_noisy,
                "median_noisy_minus_clean": (
                    med_noisy - med_clean
                    if np.isfinite(med_clean) and np.isfinite(med_noisy)
                    else np.nan
                ),
                "n_finite_clean": n_clean,
                "n_finite_noisy": n_noisy,
                "valid_count": valid_count,
                "valid_fraction": valid_fraction,
                "aitken_fallback_count": fallback_count,
                "aitken_fallback_fraction": fallback_fraction,
            })

    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_auc_rows(component_rows):
    out = []
    keys = sorted(
        set((r["variant"], r["component"]) for r in component_rows)
    )

    for variant, component in keys:
        subset = [
            r for r in component_rows
            if r["variant"] == variant
            and r["component"] == component
        ]

        for direction_col, direction_name in (
            ("auc_higher_is_noisy", "higher_is_noisy"),
            ("auc_lower_is_noisy", "lower_is_noisy"),
        ):
            finite = [
                r for r in subset
                if np.isfinite(r[direction_col])
            ]
            if not finite:
                continue

            best = max(finite, key=lambda r: r[direction_col])

            out.append({
                "variant": variant,
                "component": component,
                "direction": direction_name,
                "best_auc": best[direction_col],
                "best_epoch": best["epoch"],
                "median_clean_at_best": best["median_clean"],
                "median_noisy_at_best": best["median_noisy"],
                "valid_fraction_at_best": best["valid_fraction"],
            })

    return out


def pairwise_summary(next_v, aitken_v):
    if not np.array_equal(next_v["epoch"], aitken_v["epoch"]):
        raise ValueError("Next and Aitken epoch arrays are not identical.")
    if not np.array_equal(
        next_v["is_anomaly"], aitken_v["is_anomaly"]
    ):
        raise ValueError(
            "Next and Aitken anomaly masks are not identical."
        )

    rows = []
    y = next_v["is_anomaly"]

    for t_idx, ep in enumerate(next_v["epoch"]):
        for comp in ("ell_err", "log_abs_id", "lambda"):
            a = next_v[comp][:, t_idx]
            b = aitken_v[comp][:, t_idx]

            auc_n_h, auc_n_l = auc_pair(y, a)
            auc_a_h, auc_a_l = auc_pair(y, b)

            common = np.isfinite(a) & np.isfinite(b)

            rows.append({
                "epoch": int(ep),
                "component": (
                    "log_abs_id_fie" if comp == "log_abs_id" else comp
                ),
                "next_auc_higher": auc_n_h,
                "aitken_auc_higher": auc_a_h,
                "aitken_minus_next_auc_higher": (
                    auc_a_h - auc_n_h
                    if np.isfinite(auc_a_h) and np.isfinite(auc_n_h)
                    else np.nan
                ),
                "next_auc_lower": auc_n_l,
                "aitken_auc_lower": auc_a_l,
                "aitken_minus_next_auc_lower": (
                    auc_a_l - auc_n_l
                    if np.isfinite(auc_a_l) and np.isfinite(auc_n_l)
                    else np.nan
                ),
                "n_common_finite": int(np.sum(common)),
                "median_aitken_minus_next": (
                    float(np.median(b[common] - a[common]))
                    if np.any(common)
                    else np.nan
                ),
            })

    return rows



# ---------------------------------------------------------------------
# Weighted convergence-detection score
# ---------------------------------------------------------------------

WEIGHT_ALPHAS = (0.0, 0.25, 0.5, 1.0)


def build_weighted_scores(v, num_classes=10):
    """Build class-wise standardized weighted component scores.

    The theoretical LE estimator remains unchanged:

        lambda_hat = ell_err_hat + log(abs(ID_FIE_hat)).

    For detection only, define

        S_alpha = z(ell_err_hat) + alpha * z(log(abs(ID_FIE_hat))),

    where each z-score is computed within observed class and epoch using the
    project's shared standardization function.
    """
    labels = v["observed_label"]
    K = None

    # Infer the first usable column from the first epoch containing any finite LE.
    finite_any = np.any(np.isfinite(v["lambda"]), axis=0)
    usable_cols = np.flatnonzero(finite_any)
    if usable_cols.size == 0:
        raise ValueError(f"{v['name']}: no finite LE values found.")
    first_col = int(usable_cols[0])

    z_err_tn = standardize_monitoring_score(
        v["ell_err"].T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=first_col,
    )
    z_id_tn = standardize_monitoring_score(
        v["log_abs_id"].T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=first_col,
    )

    z_err = z_err_tn.T
    z_id = z_id_tn.T

    scores = {}
    for alpha in WEIGHT_ALPHAS:
        scores[float(alpha)] = z_err + float(alpha) * z_id

    return {
        "first_available_column": first_col,
        "z_err": z_err,
        "z_id": z_id,
        "scores": scores,
    }


def summarize_weighted_scores(v, weighted):
    rows = []
    y = v["is_anomaly"]

    for alpha, score in weighted["scores"].items():
        for t_idx, ep in enumerate(v["epoch"]):
            vals = score[:, t_idx]

            auc_h, auc_l = auc_pair(y, vals)
            med_clean, med_noisy, n_clean, n_noisy = median_by_group(
                vals, y
            )

            rows.append({
                "variant": v["name"],
                "alpha": float(alpha),
                "epoch": int(ep),
                "auc_higher_is_noisy": auc_h,
                "auc_lower_is_noisy": auc_l,
                "median_clean": med_clean,
                "median_noisy": med_noisy,
                "median_noisy_minus_clean": (
                    med_noisy - med_clean
                    if np.isfinite(med_clean) and np.isfinite(med_noisy)
                    else np.nan
                ),
                "n_finite_clean": n_clean,
                "n_finite_noisy": n_noisy,
            })

    return rows


def summarize_weighted_best(weighted_rows):
    rows = []

    keys = sorted(
        set(
            (r["variant"], float(r["alpha"]))
            for r in weighted_rows
        )
    )

    for variant, alpha in keys:
        subset = [
            r for r in weighted_rows
            if r["variant"] == variant
            and float(r["alpha"]) == alpha
        ]

        for direction_col, direction_name in (
            ("auc_higher_is_noisy", "higher_is_noisy"),
            ("auc_lower_is_noisy", "lower_is_noisy"),
        ):
            finite = [
                r for r in subset
                if np.isfinite(r[direction_col])
            ]
            if not finite:
                continue

            best = max(finite, key=lambda r: r[direction_col])

            rows.append({
                "variant": variant,
                "alpha": alpha,
                "direction": direction_name,
                "best_auc": best[direction_col],
                "best_epoch": best["epoch"],
                "median_clean_at_best": best["median_clean"],
                "median_noisy_at_best": best["median_noisy"],
            })

    return rows


def summarize_weighted_fixed_windows(weighted_rows):
    """Summaries over pre-specified early windows without optimizing an epoch."""
    windows = (
        (22, 30),
        (22, 40),
        (22, 60),
    )

    rows = []

    keys = sorted(
        set(
            (r["variant"], float(r["alpha"]))
            for r in weighted_rows
        )
    )

    for variant, alpha in keys:
        subset = [
            r for r in weighted_rows
            if r["variant"] == variant
            and float(r["alpha"]) == alpha
        ]

        for start_ep, end_ep in windows:
            rr = [
                r for r in subset
                if start_ep <= int(r["epoch"]) <= end_ep
                and np.isfinite(r["auc_higher_is_noisy"])
            ]

            if not rr:
                continue

            aucs = np.asarray(
                [r["auc_higher_is_noisy"] for r in rr],
                dtype=np.float64,
            )

            rows.append({
                "variant": variant,
                "alpha": alpha,
                "start_epoch": start_ep,
                "end_epoch": end_ep,
                "mean_auc_higher_is_noisy": float(np.mean(aucs)),
                "median_auc_higher_is_noisy": float(np.median(aucs)),
                "min_auc_higher_is_noisy": float(np.min(aucs)),
                "max_auc_higher_is_noisy": float(np.max(aucs)),
                "n_epochs": int(aucs.size),
            })

    return rows


def plot_weighted_auc(weighted_rows, variant, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alpha in WEIGHT_ALPHAS:
        rr = [
            r for r in weighted_rows
            if r["variant"] == variant
            and float(r["alpha"]) == float(alpha)
        ]

        ax.plot(
            [r["epoch"] for r in rr],
            [r["auc_higher_is_noisy"] for r in rr],
            label=f"alpha={alpha:g}",
        )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher weighted score = noisy)")
    ax.set_title(
        "Weighted convergence-detection score "
        f"— {variant}"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_weighted_early_auc(weighted_rows, variant, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alpha in WEIGHT_ALPHAS:
        rr = [
            r for r in weighted_rows
            if r["variant"] == variant
            and float(r["alpha"]) == float(alpha)
            and 22 <= int(r["epoch"]) <= 60
        ]

        ax.plot(
            [r["epoch"] for r in rr],
            [r["auc_higher_is_noisy"] for r in rr],
            label=f"alpha={alpha:g}",
        )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher weighted score = noisy)")
    ax.set_title(
        "Weighted score AUC, early training "
        f"— {variant}"
    )
    ax.set_ylim(0.4, 0.85)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



# ---------------------------------------------------------------------
# Class-reference GIE analysis
# ---------------------------------------------------------------------

def load_common_loss_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Common loss NPZ not found: {path}"
        )

    d = np.load(path, allow_pickle=False)

    required = [
        "loss_traj",
        "epoch",
        "observed_label",
        "is_anomaly",
    ]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(
            f"Common loss NPZ is missing required arrays: {missing}"
        )

    return {
        "loss_traj": np.asarray(d["loss_traj"], dtype=np.float64),
        "epoch": np.asarray(d["epoch"], dtype=np.int64),
        "observed_label": np.asarray(
            d["observed_label"],
            dtype=np.int64,
        ),
        "is_anomaly": np.asarray(
            d["is_anomaly"],
            dtype=bool,
        ),
    }


def validate_common_against_le(common, v):
    if not np.array_equal(common["epoch"], v["epoch"]):
        raise ValueError(
            f"{v['name']}: epochs differ from common loss artifact."
        )

    if not np.array_equal(
        common["observed_label"],
        v["observed_label"],
    ):
        raise ValueError(
            f"{v['name']}: observed labels differ from common loss artifact."
        )

    if not np.array_equal(
        common["is_anomaly"],
        v["is_anomaly"],
    ):
        raise ValueError(
            f"{v['name']}: noisy-label mask differs from common loss artifact."
        )


def summarize_single_component(
    name,
    values,
    epochs,
    is_anomaly,
):
    rows = []

    for t_idx, ep in enumerate(epochs):
        vals = values[:, t_idx]
        auc_h, auc_l = auc_pair(is_anomaly, vals)
        med_clean, med_noisy, n_clean, n_noisy = median_by_group(
            vals,
            is_anomaly,
        )

        rows.append({
            "component": name,
            "epoch": int(ep),
            "auc_higher_is_noisy": auc_h,
            "auc_lower_is_noisy": auc_l,
            "median_clean": med_clean,
            "median_noisy": med_noisy,
            "median_noisy_minus_clean": (
                med_noisy - med_clean
                if np.isfinite(med_clean)
                and np.isfinite(med_noisy)
                else np.nan
            ),
            "n_finite_clean": n_clean,
            "n_finite_noisy": n_noisy,
            "finite_fraction": (
                (n_clean + n_noisy) / len(is_anomaly)
            ),
        })

    return rows


def best_single_component_rows(rows):
    out = []

    for component in sorted(set(r["component"] for r in rows)):
        subset = [
            r for r in rows
            if r["component"] == component
        ]

        for col, direction in (
            ("auc_higher_is_noisy", "higher_is_noisy"),
            ("auc_lower_is_noisy", "lower_is_noisy"),
        ):
            finite = [
                r for r in subset
                if np.isfinite(r[col])
            ]
            if not finite:
                continue

            best = max(
                finite,
                key=lambda r: r[col],
            )

            out.append({
                "component": component,
                "direction": direction,
                "best_auc": best[col],
                "best_epoch": best["epoch"],
                "median_clean_at_best": best["median_clean"],
                "median_noisy_at_best": best["median_noisy"],
                "finite_fraction_at_best": best["finite_fraction"],
            })

    return out


def build_weighted_scores_external_id(
    v,
    log_abs_id_external,
    *,
    num_classes=10,
):
    """Build S_alpha = z(ell_err) + alpha*z(external log|ID|)."""
    labels = v["observed_label"]

    finite_le = np.any(
        np.isfinite(v["ell_err"]),
        axis=0,
    )
    finite_id = np.any(
        np.isfinite(log_abs_id_external),
        axis=0,
    )

    common_cols = np.flatnonzero(
        finite_le & finite_id
    )
    if common_cols.size == 0:
        raise ValueError(
            f"{v['name']}: no overlapping finite ell_err and GIE columns."
        )

    first_col = int(common_cols[0])

    z_err = standardize_monitoring_score(
        v["ell_err"].T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=first_col,
    ).T

    z_id = standardize_monitoring_score(
        log_abs_id_external.T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=first_col,
    ).T

    scores = {
        float(alpha):
            z_err + float(alpha) * z_id
        for alpha in WEIGHT_ALPHAS
    }

    return {
        "first_available_column": first_col,
        "z_err": z_err,
        "z_id": z_id,
        "scores": scores,
    }


def summarize_weighted_scores_named(
    v,
    weighted,
    id_source,
):
    rows = []
    y = v["is_anomaly"]

    for alpha, score in weighted["scores"].items():
        for t_idx, ep in enumerate(v["epoch"]):
            vals = score[:, t_idx]
            auc_h, auc_l = auc_pair(y, vals)
            med_clean, med_noisy, n_clean, n_noisy = median_by_group(
                vals,
                y,
            )

            rows.append({
                "variant": v["name"],
                "id_source": id_source,
                "alpha": float(alpha),
                "epoch": int(ep),
                "auc_higher_is_noisy": auc_h,
                "auc_lower_is_noisy": auc_l,
                "median_clean": med_clean,
                "median_noisy": med_noisy,
                "median_noisy_minus_clean": (
                    med_noisy - med_clean
                    if np.isfinite(med_clean)
                    and np.isfinite(med_noisy)
                    else np.nan
                ),
                "n_finite_clean": n_clean,
                "n_finite_noisy": n_noisy,
            })

    return rows


def summarize_weighted_named_best(rows):
    out = []
    keys = sorted(
        set(
            (
                r["variant"],
                r["id_source"],
                float(r["alpha"]),
            )
            for r in rows
        )
    )

    for variant, id_source, alpha in keys:
        subset = [
            r for r in rows
            if r["variant"] == variant
            and r["id_source"] == id_source
            and float(r["alpha"]) == alpha
        ]

        for col, direction in (
            ("auc_higher_is_noisy", "higher_is_noisy"),
            ("auc_lower_is_noisy", "lower_is_noisy"),
        ):
            finite = [
                r for r in subset
                if np.isfinite(r[col])
            ]
            if not finite:
                continue

            best = max(
                finite,
                key=lambda r: r[col],
            )

            out.append({
                "variant": variant,
                "id_source": id_source,
                "alpha": alpha,
                "direction": direction,
                "best_auc": best[col],
                "best_epoch": best["epoch"],
                "median_clean_at_best": best["median_clean"],
                "median_noisy_at_best": best["median_noisy"],
            })

    return out


def summarize_weighted_named_windows(rows):
    windows = (
        (22, 30),
        (22, 40),
        (22, 60),
    )

    out = []
    keys = sorted(
        set(
            (
                r["variant"],
                r["id_source"],
                float(r["alpha"]),
            )
            for r in rows
        )
    )

    for variant, id_source, alpha in keys:
        subset = [
            r for r in rows
            if r["variant"] == variant
            and r["id_source"] == id_source
            and float(r["alpha"]) == alpha
        ]

        for start_ep, end_ep in windows:
            rr = [
                r for r in subset
                if start_ep <= int(r["epoch"]) <= end_ep
                and np.isfinite(r["auc_higher_is_noisy"])
            ]
            if not rr:
                continue

            aucs = np.asarray(
                [r["auc_higher_is_noisy"] for r in rr],
                dtype=np.float64,
            )

            out.append({
                "variant": variant,
                "id_source": id_source,
                "alpha": alpha,
                "start_epoch": start_ep,
                "end_epoch": end_ep,
                "mean_auc_higher_is_noisy": float(np.mean(aucs)),
                "median_auc_higher_is_noisy": float(np.median(aucs)),
                "min_auc_higher_is_noisy": float(np.min(aucs)),
                "max_auc_higher_is_noisy": float(np.max(aucs)),
                "n_epochs": int(aucs.size),
            })

    return out


def summarize_gie_augmented_score(
    v,
    log_abs_gie,
):
    """Analyze ell_err + log|ID_GIE|.

    This is intentionally called a GIE-augmented detection score, not an LE.
    """
    score = v["ell_err"] + log_abs_gie
    rows = []

    for t_idx, ep in enumerate(v["epoch"]):
        vals = score[:, t_idx]
        auc_h, auc_l = auc_pair(
            v["is_anomaly"],
            vals,
        )
        med_clean, med_noisy, n_clean, n_noisy = median_by_group(
            vals,
            v["is_anomaly"],
        )

        rows.append({
            "variant": v["name"],
            "score": "ell_err_plus_log_abs_id_gie",
            "epoch": int(ep),
            "auc_higher_is_noisy": auc_h,
            "auc_lower_is_noisy": auc_l,
            "median_clean": med_clean,
            "median_noisy": med_noisy,
            "n_finite_clean": n_clean,
            "n_finite_noisy": n_noisy,
        })

    return rows


def plot_id_fie_vs_gie(
    next_v,
    aitken_v,
    log_abs_gie,
    out_path,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    series = (
        (
            "FIE — next",
            next_v["log_abs_id"],
            next_v["is_anomaly"],
        ),
        (
            "FIE — guarded Aitken",
            aitken_v["log_abs_id"],
            aitken_v["is_anomaly"],
        ),
        (
            "class-reference GIE",
            log_abs_gie,
            next_v["is_anomaly"],
        ),
    )

    for label, arr, y in series:
        aucs = [
            auc_pair(y, arr[:, t])[0]
            for t in range(arr.shape[1])
        ]
        ax.plot(
            next_v["epoch"],
            aucs,
            label=label,
        )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher log|ID| = noisy)")
    ax.set_title("FIE vs class-reference GIE component AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gie_augmented_vs_le(
    v,
    log_abs_gie,
    out_path,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    auc_le = [
        auc_pair(
            v["is_anomaly"],
            v["lambda"][:, t],
        )[0]
        for t in range(v["lambda"].shape[1])
    ]

    gie_score = (
        v["ell_err"]
        + log_abs_gie
    )
    auc_gie = [
        auc_pair(
            v["is_anomaly"],
            gie_score[:, t],
        )[0]
        for t in range(gie_score.shape[1])
    ]

    ax.plot(
        v["epoch"],
        auc_le,
        label="theoretical LE with FIE",
    )
    ax.plot(
        v["epoch"],
        auc_gie,
        label="ell_err + log|ID_GIE|",
    )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher score = noisy)")
    ax.set_title(
        f"FIE LE vs GIE-augmented score — {v['name']}"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_weighted_fie_vs_gie(
    fie_rows,
    gie_rows,
    variant,
    alpha,
    out_path,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    rr_fie = [
        r for r in fie_rows
        if r["variant"] == variant
        and float(r["alpha"]) == float(alpha)
    ]
    rr_gie = [
        r for r in gie_rows
        if r["variant"] == variant
        and r["id_source"] == "gie"
        and float(r["alpha"]) == float(alpha)
    ]

    ax.plot(
        [r["epoch"] for r in rr_fie],
        [r["auc_higher_is_noisy"] for r in rr_fie],
        label=f"FIE alpha={alpha:g}",
    )
    ax.plot(
        [r["epoch"] for r in rr_gie],
        [r["auc_higher_is_noisy"] for r in rr_gie],
        label=f"GIE alpha={alpha:g}",
    )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher weighted score = noisy)")
    ax.set_title(
        f"Weighted FIE vs GIE — {variant}, alpha={alpha:g}"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def rows_for(rows, variant, component):
    return [
        r for r in rows
        if r["variant"] == variant
        and r["component"] == component
    ]


def plot_auc_components(rows, variant, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = {
        "ell_err": r"$\hat{\ell}_{err}$",
        "log_abs_id_fie": r"$\ln|\widehat{ID}_{FIE}|$",
        "lambda": r"$\hat{\ell}$",
    }

    for comp in ("ell_err", "log_abs_id_fie", "lambda"):
        rr = rows_for(rows, variant, comp)
        x = [r["epoch"] for r in rr]
        y = [r["auc_higher_is_noisy"] for r in rr]
        ax.plot(x, y, label=labels[comp])

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher score = noisy)")
    ax.set_title(f"LE component AUC — {variant}")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_component_medians(rows, variant, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for comp, display in (
        ("ell_err", "ell_err"),
        ("log_abs_id_fie", "log|ID_FIE|"),
        ("lambda", "LE"),
    ):
        rr = rows_for(rows, variant, comp)
        x = [r["epoch"] for r in rr]

        ax.plot(
            x,
            [r["median_clean"] for r in rr],
            label=f"{display}: clean",
        )
        ax.plot(
            x,
            [r["median_noisy"] for r in rr],
            linestyle="--",
            label=f"{display}: noisy",
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Median score")
    ax.set_title(f"Clean vs noisy LE-component medians — {variant}")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_le_comparison(rows, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for variant in ("next", "aitken_guarded"):
        rr = rows_for(rows, variant, "lambda")
        ax.plot(
            [r["epoch"] for r in rr],
            [r["auc_higher_is_noisy"] for r in rr],
            label=variant,
        )

    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC (higher signed LE = noisy)")
    ax.set_title("Final signed LE AUC: Next vs guarded Aitken")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_valid_fraction(rows, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for variant in ("next", "aitken_guarded"):
        rr = rows_for(rows, variant, "lambda")
        ax.plot(
            [r["epoch"] for r in rr],
            [r["valid_fraction"] for r in rr],
            label=variant,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Finite LE fraction")
    ax.set_title("LE numerical availability")
    ax.set_ylim(0.0, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fallback_fraction(rows, out_path):
    rr = rows_for(rows, "aitken_guarded", "lambda")

    if not rr:
        return

    vals = np.asarray(
        [r["aitken_fallback_fraction"] for r in rr],
        dtype=np.float64,
    )

    if not np.any(vals > 0):
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        [r["epoch"] for r in rr],
        vals,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fallback fraction")
    ax.set_title("Guarded-Aitken fallback to next-sample limit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    next_v = load_variant(args.next_npz, "next")
    aitken_v = load_variant(args.aitken_npz, "aitken_guarded")

    if not np.array_equal(next_v["epoch"], aitken_v["epoch"]):
        raise ValueError("Epoch arrays differ between variants.")

    if not np.array_equal(
        next_v["is_anomaly"],
        aitken_v["is_anomaly"],
    ):
        raise ValueError("Noisy-label masks differ between variants.")

    # --------------------------------------------------------------
    # Compute class-reference GIE once from the common loss artifact.
    # G_i(t) is the mean loss trajectory of samples sharing observed label.
    # --------------------------------------------------------------

    common = load_common_loss_artifact(args.common_npz)
    validate_common_against_le(common, next_v)
    validate_common_against_le(common, aitken_v)

    gie = rolling_class_reference_gie_batch(
        common["loss_traj"],
        common["observed_label"],
        K=args.gie_K,
        num_classes=args.num_classes,
    )

    id_gie = np.asarray(
        gie["id_gie_traj"],
        dtype=np.float64,
    )
    log_abs_gie = safe_log_abs(id_gie)

    gie_component_rows = summarize_single_component(
        "log_abs_id_gie",
        log_abs_gie,
        common["epoch"],
        common["is_anomaly"],
    )
    write_csv(
        args.output_dir / "gie_component_epoch_summary.csv",
        gie_component_rows,
    )
    write_csv(
        args.output_dir / "gie_component_best_auc_summary.csv",
        best_single_component_rows(gie_component_rows),
    )

    gie_augmented_rows = (
        summarize_gie_augmented_score(next_v, log_abs_gie)
        + summarize_gie_augmented_score(aitken_v, log_abs_gie)
    )
    write_csv(
        args.output_dir / "gie_augmented_score_epoch_summary.csv",
        gie_augmented_rows,
    )

    rows = (
        summarize_variant(next_v)
        + summarize_variant(aitken_v)
    )

    write_csv(
        args.output_dir / "le_component_epoch_summary.csv",
        rows,
    )

    best = best_auc_rows(rows)
    write_csv(
        args.output_dir / "le_component_best_auc_summary.csv",
        best,
    )

    pairwise = pairwise_summary(next_v, aitken_v)
    write_csv(
        args.output_dir / "le_variant_pairwise_summary.csv",
        pairwise,
    )


    # --------------------------------------------------------------
    # Detection-score ablation:
    # S_alpha = z(ell_err) + alpha * z(log|ID_FIE|)
    #
    # The LE estimator itself is NOT modified.
    # --------------------------------------------------------------

    next_weighted = build_weighted_scores(next_v)
    aitken_weighted = build_weighted_scores(aitken_v)

    weighted_rows = (
        summarize_weighted_scores(next_v, next_weighted)
        + summarize_weighted_scores(aitken_v, aitken_weighted)
    )

    write_csv(
        args.output_dir / "weighted_score_epoch_summary.csv",
        weighted_rows,
    )

    weighted_best = summarize_weighted_best(weighted_rows)
    write_csv(
        args.output_dir / "weighted_score_best_auc_summary.csv",
        weighted_best,
    )

    weighted_windows = summarize_weighted_fixed_windows(
        weighted_rows
    )
    write_csv(
        args.output_dir / "weighted_score_early_window_summary.csv",
        weighted_windows,
    )

    # --------------------------------------------------------------
    # Same prespecified alpha grid, replacing FIE with class-reference GIE.
    # --------------------------------------------------------------

    next_weighted_gie = build_weighted_scores_external_id(
        next_v,
        log_abs_gie,
        num_classes=args.num_classes,
    )
    aitken_weighted_gie = build_weighted_scores_external_id(
        aitken_v,
        log_abs_gie,
        num_classes=args.num_classes,
    )

    weighted_gie_rows = (
        summarize_weighted_scores_named(
            next_v,
            next_weighted_gie,
            "gie",
        )
        + summarize_weighted_scores_named(
            aitken_v,
            aitken_weighted_gie,
            "gie",
        )
    )

    write_csv(
        args.output_dir / "weighted_gie_score_epoch_summary.csv",
        weighted_gie_rows,
    )
    write_csv(
        args.output_dir / "weighted_gie_score_best_auc_summary.csv",
        summarize_weighted_named_best(weighted_gie_rows),
    )
    write_csv(
        args.output_dir / "weighted_gie_score_early_window_summary.csv",
        summarize_weighted_named_windows(weighted_gie_rows),
    )

    validity_rows = []
    for v in (next_v, aitken_v):
        N, T = v["lambda"].shape
        finite_lambda = np.isfinite(v["lambda"])
        fallback = v["aitken_fallback"]

        validity_rows.append({
            "variant": v["name"],
            "n_samples": N,
            "n_epochs": T,
            "finite_le_count": int(np.sum(finite_lambda)),
            "finite_le_fraction_over_full_array": float(
                np.mean(finite_lambda)
            ),
            "max_formula_abs_diff": v["max_formula_abs_diff"],
            "aitken_fallback_count": (
                int(np.sum(fallback))
                if fallback is not None
                else 0
            ),
            "aitken_fallback_fraction_over_full_array": (
                float(np.mean(fallback))
                if fallback is not None
                else 0.0
            ),
        })

    write_csv(
        args.output_dir / "le_validity_summary.csv",
        validity_rows,
    )

    plot_auc_components(
        rows,
        "next",
        args.output_dir / "fig_auc_components_next.png",
    )
    plot_auc_components(
        rows,
        "aitken_guarded",
        args.output_dir / "fig_auc_components_aitken.png",
    )
    plot_component_medians(
        rows,
        "next",
        args.output_dir / "fig_medians_components_next.png",
    )
    plot_component_medians(
        rows,
        "aitken_guarded",
        args.output_dir / "fig_medians_components_aitken.png",
    )
    plot_final_le_comparison(
        rows,
        args.output_dir / "fig_final_le_auc_next_vs_aitken.png",
    )
    plot_valid_fraction(
        rows,
        args.output_dir / "fig_valid_fraction_next_vs_aitken.png",
    )
    plot_fallback_fraction(
        rows,
        args.output_dir / "fig_aitken_fallback_fraction.png",
    )


    plot_weighted_auc(
        weighted_rows,
        "next",
        args.output_dir / "fig_weighted_auc_next.png",
    )
    plot_weighted_auc(
        weighted_rows,
        "aitken_guarded",
        args.output_dir / "fig_weighted_auc_aitken.png",
    )
    plot_weighted_early_auc(
        weighted_rows,
        "next",
        args.output_dir / "fig_weighted_auc_early_next.png",
    )
    plot_weighted_early_auc(
        weighted_rows,
        "aitken_guarded",
        args.output_dir / "fig_weighted_auc_early_aitken.png",
    )

    plot_id_fie_vs_gie(
        next_v,
        aitken_v,
        log_abs_gie,
        args.output_dir / "fig_id_component_auc_fie_vs_gie.png",
    )
    plot_gie_augmented_vs_le(
        next_v,
        log_abs_gie,
        args.output_dir / "fig_gie_augmented_vs_le_next.png",
    )
    plot_gie_augmented_vs_le(
        aitken_v,
        log_abs_gie,
        args.output_dir / "fig_gie_augmented_vs_le_aitken.png",
    )
    plot_weighted_fie_vs_gie(
        weighted_rows,
        weighted_gie_rows,
        "aitken_guarded",
        0.25,
        args.output_dir / "fig_weighted_fie_vs_gie_aitken_alpha025.png",
    )
    plot_weighted_fie_vs_gie(
        weighted_rows,
        weighted_gie_rows,
        "aitken_guarded",
        1.0,
        args.output_dir / "fig_weighted_fie_vs_gie_aitken_alpha1.png",
    )

    print("========================================")
    print("LE component diagnostic complete")
    print("========================================")
    print(f"Next formula max abs diff:   {next_v['max_formula_abs_diff']:.3e}")
    print(f"Aitken formula max abs diff: {aitken_v['max_formula_abs_diff']:.3e}")
    print()

    for r in best:
        if r["component"] == "lambda":
            print(
                f"{r['variant']:>16s} | "
                f"{r['direction']:>16s} | "
                f"best AUC={r['best_auc']:.6f} "
                f"at epoch {r['best_epoch']}"
            )

    print()
    print("Weighted detection-score results:")
    for r in weighted_best:
        if r["direction"] == "higher_is_noisy":
            print(
                f"{r['variant']:>16s} | "
                f"alpha={r['alpha']:<4g} | "
                f"best AUC={r['best_auc']:.6f} "
                f"at epoch {r['best_epoch']}"
            )

    print()
    print("GIE component / weighted-score results:")
    gie_best = best_single_component_rows(gie_component_rows)
    for r in gie_best:
        if r["direction"] == "higher_is_noisy":
            print(
                f"{r['component']:>20s} | "
                f"best AUC={r['best_auc']:.6f} "
                f"at epoch {r['best_epoch']}"
            )

    for r in summarize_weighted_named_best(weighted_gie_rows):
        if (
            r["variant"] == "aitken_guarded"
            and r["direction"] == "higher_is_noisy"
        ):
            print(
                f"GIE weighted | alpha={r['alpha']:<4g} | "
                f"best AUC={r['best_auc']:.6f} "
                f"at epoch {r['best_epoch']}"
            )

    print()
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
