#!/usr/bin/env python3
"""
Create presentation-ready CKL vs LE vs CKL+LE (OR fusion) figures.

For each temporal detector:
  1) EWMA
  2) Cumulative pairwise

The figure shows:
  - TPR and FPR over epochs for CKL, LE, and CKL+LE
  - 5%, 10%, and 20% noise in three panels
  - a compact summary table underneath

The table uses plain-language labels:
  Early TPR   = mean TPR over epochs 50-80
  Average TPR = mean TPR over epochs 60-120
  Late TPR    = mean TPR over epochs 180-200
  Average FPR = mean FPR over epochs 60-120

Default: K=40.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = ["CKL_rank", "LE_rank", "prob_or"]
VARIANT_LABELS = {
    "CKL_rank": "CKL",
    "LE_rank": "LE",
    "prob_or": "CKL+LE",
}

DETECTORS = ["rank_ewma", "cumulative_pairwise"]
DETECTOR_LABELS = {
    "rank_ewma": "EWMA",
    "cumulative_pairwise": "Cumulative pairwise",
}

# Fixed colors make CKL/LE/fusion visually consistent across both figures.
COLORS = {
    "CKL_rank": "tab:blue",
    "LE_rank": "tab:orange",
    "prob_or": "tab:green",
}

NOISE_LEVELS = [0.05, 0.10, 0.20]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trajectory-csv",
        type=Path,
        default=Path("/mnt/data/trajectory_metrics_over_epochs.csv"),
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/mnt/data/trajectory_summary.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/final_ckl_le_fusion_presentation_v1/output"),
    )
    p.add_argument("--K", type=int, default=40)
    return p.parse_args()


def load_and_filter(path: Path, K: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[
        (df["K"] == K)
        & (df["variant"].isin(VARIANTS))
        & (df["detector"].isin(DETECTORS))
        & (df["noise_rate"].isin(NOISE_LEVELS))
    ].copy()


def make_compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[[
        "noise_rate",
        "detector",
        "variant",
        "early_mean_TPR",
        "sustained_mean_TPR",
        "late_mean_TPR",
        "sustained_mean_FPR",
    ]].copy()

    out["Noise"] = (100 * out["noise_rate"]).round().astype(int).astype(str) + "%"
    out["Detector"] = out["detector"].map(DETECTOR_LABELS)
    out["Method"] = out["variant"].map(VARIANT_LABELS)
    out["Early TPR (50-80)"] = out["early_mean_TPR"]
    out["Average TPR (60-120)"] = out["sustained_mean_TPR"]
    out["Late TPR (180-200)"] = out["late_mean_TPR"]
    out["Average FPR (60-120)"] = out["sustained_mean_FPR"]

    return out[[
        "Noise",
        "Detector",
        "Method",
        "Early TPR (50-80)",
        "Average TPR (60-120)",
        "Late TPR (180-200)",
        "Average FPR (60-120)",
    ]].sort_values(["Detector", "Noise", "Method"])


def format_table_rows(summary: pd.DataFrame, detector: str):
    rows = []
    for noise in NOISE_LEVELS:
        for variant in VARIANTS:
            r = summary[
                (summary["detector"] == detector)
                & np.isclose(summary["noise_rate"], noise)
                & (summary["variant"] == variant)
            ]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([
                f"{int(noise * 100)}%",
                VARIANT_LABELS[variant],
                f"{r['early_mean_TPR']:.3f}",
                f"{r['sustained_mean_TPR']:.3f}",
                f"{r['late_mean_TPR']:.3f}",
                f"{r['sustained_mean_FPR']:.3f}",
            ])
    return rows


def make_figure(
    trajectories: pd.DataFrame,
    summary: pd.DataFrame,
    detector: str,
    K: int,
    output_dir: Path,
):
    fig = plt.figure(figsize=(16.5, 10.2))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[2.1, 1.35],
        hspace=0.35,
        wspace=0.22,
    )

    legend_handles = []
    legend_labels = []

    for col, noise in enumerate(NOISE_LEVELS):
        ax = fig.add_subplot(gs[0, col])
        sub = trajectories[
            (trajectories["detector"] == detector)
            & np.isclose(trajectories["noise_rate"], noise)
        ]

        for variant in VARIANTS:
            d = sub[sub["variant"] == variant].sort_values("epoch")
            if d.empty:
                continue

            color = COLORS[variant]
            label = VARIANT_LABELS[variant]

            line_tpr, = ax.plot(
                d["epoch"],
                d["TPR"],
                color=color,
                linewidth=2.2,
                linestyle="-",
                label=f"{label} TPR",
            )
            line_fpr, = ax.plot(
                d["epoch"],
                d["FPR"],
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{label} FPR",
            )

            if col == 0:
                legend_handles.extend([line_tpr, line_fpr])
                legend_labels.extend([f"{label} TPR", f"{label} FPR"])

        ax.set_title(f"{int(noise * 100)}% label noise", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        if col == 0:
            ax.set_ylabel("Rate", fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)

    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")

    cell_text = format_table_rows(summary, detector)
    col_labels = [
        "Noise",
        "Method",
        "Early TPR\n(50-80)",
        "Average TPR\n(60-120)",
        "Late TPR\n(180-200)",
        "Average FPR\n(60-120)",
    ]

    table = table_ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.04, 0.02, 0.92, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.28)

    # Bold the header row only.
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    detector_name = DETECTOR_LABELS[detector]
    fig.suptitle(
        f"CKL vs LE vs CKL+LE using {detector_name} (K={K})",
        fontsize=20,
        y=0.985,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=6,
        frameon=False,
        fontsize=10.5,
    )

    fig.text(
        0.5,
        0.005,
        "Solid lines: TPR    |    Dashed lines: FPR",
        ha="center",
        fontsize=11,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{detector_name.lower().replace(' ', '_')}_ckl_le_or_fusion_K{K}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"

    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traj = load_and_filter(args.trajectory_csv, args.K)
    summary = load_and_filter(args.summary_csv, args.K)

    compact = make_compact_summary(summary)
    compact_path = args.output_dir / f"ckl_le_or_fusion_summary_K{args.K}.csv"
    compact.to_csv(compact_path, index=False)

    print(f"Saved summary: {compact_path}")

    for detector in DETECTORS:
        png, pdf = make_figure(
            trajectories=traj,
            summary=summary,
            detector=detector,
            K=args.K,
            output_dir=args.output_dir,
        )
        print(png)
        print(pdf)


if __name__ == "__main__":
    main()
