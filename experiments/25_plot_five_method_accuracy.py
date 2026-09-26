#!/usr/bin/env python3
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_columns(df, path, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_score_guided(path, adaptive=True):
    df = read_csv(path)

    if adaptive:
        train_col = "adaptive_train_true_accuracy_all"
        test_col = "adaptive_test_accuracy"
    else:
        train_col = "baseline_train_true_accuracy_all"
        test_col = "baseline_test_accuracy"

    require_columns(df, path, ["epoch", train_col, test_col])

    return pd.DataFrame({
        "epoch": df["epoch"],
        "train_acc": df[train_col],
        "test_acc": df[test_col],
    })


def load_d2l(path):
    df = read_csv(path)

    # These are the column names produced by
    # experiments/26_train_d2l_intervention.py
    train_candidates = [
        "d2l_train_true_accuracy_all",
        "train_true_accuracy_all",
    ]
    test_candidates = [
        "d2l_test_accuracy",
        "test_accuracy",
    ]

    train_col = next((c for c in train_candidates if c in df.columns), None)
    test_col = next((c for c in test_candidates if c in df.columns), None)

    if train_col is None or test_col is None:
        raise KeyError(
            f"Could not find D2L accuracy columns in {path}\n"
            f"Tried training: {train_candidates}\n"
            f"Tried test: {test_candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return pd.DataFrame({
        "epoch": df["epoch"],
        "train_acc": df[train_col],
        "test_acc": df[test_col],
    })


def plot_curves(series, metric, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for label, df in series.items():
        ax.plot(
            df["epoch"],
            100.0 * df[metric],
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=1)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ckl-csv",
        default="results/score_guided_intervention/ckl__ewma/epoch_summary.csv",
    )
    p.add_argument(
        "--le-csv",
        default="results/score_guided_intervention/le__ewma/epoch_summary.csv",
    )
    p.add_argument(
        "--combined-csv",
        default="results/score_guided_intervention/combined__ewma/epoch_summary.csv",
    )
    p.add_argument(
        "--d2l-csv",
        default="results/d2l_intervention/epoch_summary.csv",
    )
    p.add_argument(
        "--output-dir",
        default="results/five_method_accuracy_plots",
    )

    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The paired baseline is identical across CKL/LE/combined.
    # Use the baseline columns from the combined run.
    baseline = load_score_guided(args.combined_csv, adaptive=False)
    ckl = load_score_guided(args.ckl_csv, adaptive=True)
    le = load_score_guided(args.le_csv, adaptive=True)
    combined = load_score_guided(args.combined_csv, adaptive=True)
    d2l = load_d2l(args.d2l_csv)

    series = {
        "Baseline": baseline,
        "CKL": ckl,
        "LE": le,
        "CKL+LE": combined,
        "D2L": d2l,
    }

    # Overall training accuracy against original true CIFAR-10 labels.
    plot_curves(
        series,
        metric="train_acc",
        ylabel="Training accuracy (%)",
        title="Training Accuracy During Training",
        output_path=outdir / "training_accuracy_five_methods",
    )

    # Clean CIFAR-10 test accuracy.
    plot_curves(
        series,
        metric="test_acc",
        ylabel="Test accuracy (%)",
        title="Test Accuracy During Training",
        output_path=outdir / "test_accuracy_five_methods",
    )

    rows = []
    for label, df in series.items():
        best_test_i = df["test_acc"].idxmax()
        rows.append({
            "method": label,
            "final_train_accuracy": float(df["train_acc"].iloc[-1]),
            "final_test_accuracy": float(df["test_acc"].iloc[-1]),
            "best_test_accuracy": float(df.loc[best_test_i, "test_acc"]),
            "best_test_epoch": int(df.loc[best_test_i, "epoch"]),
        })

    pd.DataFrame(rows).to_csv(
        outdir / "five_method_accuracy_summary.csv",
        index=False,
    )

    print("Created:")
    print(outdir / "training_accuracy_five_methods.png")
    print(outdir / "training_accuracy_five_methods.pdf")
    print(outdir / "test_accuracy_five_methods.png")
    print(outdir / "test_accuracy_five_methods.pdf")
    print(outdir / "five_method_accuracy_summary.csv")


if __name__ == "__main__":
    main()
