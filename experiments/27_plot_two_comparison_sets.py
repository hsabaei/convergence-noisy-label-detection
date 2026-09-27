#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def require(df, path, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_score_run(path):
    df = read_csv(path)
    cols = [
        "epoch",
        "adaptive_train_true_accuracy_all",
        "adaptive_train_true_accuracy_noisy",
        "adaptive_test_accuracy",
        "baseline_train_true_accuracy_all",
        "baseline_train_true_accuracy_noisy",
        "baseline_test_accuracy",
    ]
    require(df, path, cols)

    return {
        "adaptive": pd.DataFrame({
            "epoch": df["epoch"],
            "train_all": df["adaptive_train_true_accuracy_all"],
            "train_noisy": df["adaptive_train_true_accuracy_noisy"],
            "test": df["adaptive_test_accuracy"],
        }),
        "baseline": pd.DataFrame({
            "epoch": df["epoch"],
            "train_all": df["baseline_train_true_accuracy_all"],
            "train_noisy": df["baseline_train_true_accuracy_noisy"],
            "test": df["baseline_test_accuracy"],
        }),
    }


def load_d2l(path):
    df = read_csv(path)

    train_all = next((c for c in [
        "d2l_train_true_accuracy_all",
        "train_true_accuracy_all",
    ] if c in df.columns), None)

    train_noisy = next((c for c in [
        "d2l_train_true_accuracy_noisy",
        "train_true_accuracy_noisy",
    ] if c in df.columns), None)

    test = next((c for c in [
        "d2l_test_accuracy",
        "test_accuracy",
    ] if c in df.columns), None)

    if train_all is None or train_noisy is None or test is None:
        raise KeyError(
            f"Could not find D2L accuracy columns in {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    return pd.DataFrame({
        "epoch": df["epoch"],
        "train_all": df[train_all],
        "train_noisy": df[train_noisy],
        "test": df[test],
    })


def auto_ylim(series, metric, start_epoch=40, pad=0.25, step=0.5):
    vals = []
    for df in series.values():
        shown = df[df["epoch"] >= start_epoch]
        vals.extend((100.0 * shown[metric]).tolist())

    lo = min(vals) - pad
    hi = max(vals) + pad

    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step

    if hi - lo < 2.0:
        mid = 0.5 * (hi + lo)
        lo = math.floor((mid - 1.0) / step) * step
        hi = math.ceil((mid + 1.0) / step) * step

    return lo, hi


def plot(series, metric, title, ylabel, outpath,
         start_epoch=40, full_scale=False):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, df in series.items():
        shown = df if start_epoch is None else df[df["epoch"] >= start_epoch]
        ax.plot(shown["epoch"], 100.0 * shown[metric],
                linewidth=2.1, label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    if start_epoch is not None:
        ax.set_xlim(left=start_epoch)
    else:
        ax.set_xlim(left=1)

    if full_scale:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(*auto_ylim(series, metric, start_epoch=start_epoch))

    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_summary_rows(rows, group, series):
    for label, df in series.items():
        late = df[df["epoch"].between(150, 200)]
        best_i = df["test"].idxmax()
        rows.append({
            "group": group,
            "method": label,
            "final_test_accuracy": float(df["test"].iloc[-1]),
            "best_test_accuracy": float(df.loc[best_i, "test"]),
            "best_test_epoch": int(df.loc[best_i, "epoch"]),
            "avg_test_accuracy_150_200": float(late["test"].mean()),
            "final_train_true_accuracy_all": float(df["train_all"].iloc[-1]),
            "final_train_true_accuracy_noisy": float(df["train_noisy"].iloc[-1]),
            "avg_train_true_noisy_150_200": float(late["train_noisy"].mean()),
        })


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--ckl-ewma", required=True)
    p.add_argument("--le-ewma", required=True)
    p.add_argument("--combined-ewma", required=True)
    p.add_argument("--ckl-pairwise", required=True)
    p.add_argument("--le-pairwise", required=True)
    p.add_argument("--combined-pairwise", required=True)
    p.add_argument("--d2l", required=True)
    p.add_argument("--output-dir", default="results/two_set_intervention_plots")
    p.add_argument("--zoom-start-epoch", type=int, default=40)

    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckl_e = load_score_run(args.ckl_ewma)
    le_e = load_score_run(args.le_ewma)
    comb_e = load_score_run(args.combined_ewma)

    ckl_p = load_score_run(args.ckl_pairwise)
    le_p = load_score_run(args.le_pairwise)
    comb_p = load_score_run(args.combined_pairwise)

    d2l = load_d2l(args.d2l)

    # Set 1: CKL, LE, CKL+LE for both temporal methods.
    set1 = {
        "CKL - EWMA": ckl_e["adaptive"],
        "LE - EWMA": le_e["adaptive"],
        "CKL+LE - EWMA": comb_e["adaptive"],
        "CKL - Pairwise": ckl_p["adaptive"],
        "LE - Pairwise": le_p["adaptive"],
        "CKL+LE - Pairwise": comb_p["adaptive"],
    }

    plot(
        set1, "test",
        "CKL, LE, and CKL+LE: EWMA vs Cumulative Pairwise",
        "Test accuracy (%)",
        outdir / "set1_test_accuracy",
        start_epoch=args.zoom_start_epoch,
    )

    plot(
        set1, "train_all",
        "CKL, LE, and CKL+LE: Overall Training Accuracy",
        "Training accuracy (%)",
        outdir / "set1_training_accuracy_all",
        start_epoch=args.zoom_start_epoch,
    )

    plot(
        set1, "train_noisy",
        "CKL, LE, and CKL+LE: True-Label Accuracy on Noisy Samples",
        "Accuracy on noisy samples (%)",
        outdir / "set1_training_accuracy_noisy",
        start_epoch=None,
        full_scale=True,
    )

    # Set 2: two CKL+LE methods vs D2L and baseline.
    set2 = {
        "Baseline": comb_e["baseline"],
        "CKL+LE - EWMA": comb_e["adaptive"],
        "CKL+LE - Pairwise": comb_p["adaptive"],
        "D2L": d2l,
    }

    plot(
        set2, "test",
        "CKL+LE Intervention vs D2L and Baseline",
        "Test accuracy (%)",
        outdir / "set2_test_accuracy",
        start_epoch=args.zoom_start_epoch,
    )

    plot(
        set2, "train_all",
        "CKL+LE Intervention vs D2L and Baseline: Training Accuracy",
        "Training accuracy (%)",
        outdir / "set2_training_accuracy_all",
        start_epoch=args.zoom_start_epoch,
    )

    plot(
        set2, "train_noisy",
        "CKL+LE Intervention vs D2L and Baseline: Noisy Samples",
        "Accuracy on noisy samples (%)",
        outdir / "set2_training_accuracy_noisy",
        start_epoch=None,
        full_scale=True,
    )

    rows = []
    add_summary_rows(rows, "set1", set1)
    add_summary_rows(rows, "set2", set2)
    pd.DataFrame(rows).to_csv(outdir / "two_set_summary.csv", index=False)

    print("Created:")
    for name in [
        "set1_test_accuracy.png",
        "set1_test_accuracy.pdf",
        "set1_training_accuracy_all.png",
        "set1_training_accuracy_all.pdf",
        "set1_training_accuracy_noisy.png",
        "set1_training_accuracy_noisy.pdf",
        "set2_test_accuracy.png",
        "set2_test_accuracy.pdf",
        "set2_training_accuracy_all.png",
        "set2_training_accuracy_all.pdf",
        "set2_training_accuracy_noisy.png",
        "set2_training_accuracy_noisy.pdf",
        "two_set_summary.csv",
    ]:
        print(outdir / name)


if __name__ == "__main__":
    main()
