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
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
