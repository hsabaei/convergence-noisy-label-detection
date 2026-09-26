#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df, candidates, name, csv_path):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find column for '{name}' in {csv_path}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_case(csv_path, case_name, test_candidates, train_candidates):
    df = pd.read_csv(csv_path)

    epoch_col = pick_col(
        df,
        ["epoch", "Epoch", "epochs"],
        f"{case_name}: epoch",
        csv_path,
    )
    test_col = pick_col(df, test_candidates, f"{case_name}: test acc", csv_path)
    train_col = pick_col(df, train_candidates, f"{case_name}: train acc", csv_path)

    out = pd.DataFrame({
        "epoch": df[epoch_col].to_numpy(),
        "test_acc": df[test_col].to_numpy(),
        "train_acc": df[train_col].to_numpy(),
    })
    return out


def plot_metric(case_to_df, metric, ylabel, title, out_png, out_pdf):
    plt.figure(figsize=(10, 6))

    order = ["baseline", "ckl", "le", "combined", "d2l"]
    pretty = {
        "baseline": "Baseline",
        "ckl": "CKL",
        "le": "LE",
        "combined": "CKL+LE",
        "d2l": "D2L",
    }

    for key in order:
        if key not in case_to_df:
            continue
        df = case_to_df[key]
        plt.plot(df["epoch"], df[metric], label=pretty[key], linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def write_summary(case_to_df, out_csv):
    rows = []
    pretty = {
        "baseline": "Baseline",
        "ckl": "CKL",
        "le": "LE",
        "combined": "CKL+LE",
        "d2l": "D2L",
    }

    for key, df in case_to_df.items():
        best_test_idx = df["test_acc"].idxmax()
        best_train_idx = df["train_acc"].idxmax()

        rows.append({
            "method": pretty.get(key, key),
            "final_epoch": int(df["epoch"].iloc[-1]),
            "final_test_acc": float(df["test_acc"].iloc[-1]),
            "best_test_acc": float(df["test_acc"].iloc[best_test_idx]),
            "best_test_epoch": int(df["epoch"].iloc[best_test_idx]),
            "final_train_true_acc": float(df["train_acc"].iloc[-1]),
            "best_train_true_acc": float(df["train_acc"].iloc[best_train_idx]),
            "best_train_true_epoch": int(df["epoch"].iloc[best_train_idx]),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("method")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary to: {out_csv}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckl-csv",
        default="results/score_guided_intervention/ckl__ewma/epoch_summary.csv",
    )
    parser.add_argument(
        "--le-csv",
        default="results/score_guided_intervention/le__ewma/epoch_summary.csv",
    )
    parser.add_argument(
        "--combined-csv",
        default="results/score_guided_intervention/combined__ewma/epoch_summary.csv",
    )
    parser.add_argument(
        "--d2l-csv",
        default="results/d2l_intervention/epoch_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/five_method_accuracy_plots",
    )

    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Baseline is taken from the BASE columns of the combined run
    baseline_df = load_case(
        args.combined_csv,
        "baseline",
        test_candidates=[
            "test_base_acc",
            "base_test_acc",
            "test_acc_base",
        ],
        train_candidates=[
            "train_true_noisy_base_acc",
            "base_train_true_noisy_acc",
            "train_true_base_acc",
        ],
    )

    # Score-guided adaptive runs
    ckl_df = load_case(
        args.ckl_csv,
        "ckl",
        test_candidates=[
            "test_adapt_acc",
            "adaptive_test_acc",
            "test_acc_adapt",
        ],
        train_candidates=[
            "train_true_noisy_adapt_acc",
            "adaptive_train_true_noisy_acc",
            "train_true_adapt_acc",
        ],
    )

    le_df = load_case(
        args.le_csv,
        "le",
        test_candidates=[
            "test_adapt_acc",
            "adaptive_test_acc",
            "test_acc_adapt",
        ],
        train_candidates=[
            "train_true_noisy_adapt_acc",
            "adaptive_train_true_noisy_acc",
            "train_true_adapt_acc",
        ],
    )

    combined_df = load_case(
        args.combined_csv,
        "combined",
        test_candidates=[
            "test_adapt_acc",
            "adaptive_test_acc",
            "test_acc_adapt",
        ],
        train_candidates=[
            "train_true_noisy_adapt_acc",
            "adaptive_train_true_noisy_acc",
            "train_true_adapt_acc",
        ],
    )

    # D2L run
    d2l_df = load_case(
        args.d2l_csv,
        "d2l",
        test_candidates=[
            "test_d2l_acc",
            "d2l_test_acc",
            "test_acc_d2l",
        ],
        train_candidates=[
            "train_true_noisy_d2l_acc",
            "d2l_train_true_noisy_acc",
            "train_true_d2l_acc",
        ],
    )

    case_to_df = {
        "baseline": baseline_df,
        "ckl": ckl_df,
        "le": le_df,
        "combined": combined_df,
        "d2l": d2l_df,
    }

    plot_metric(
        case_to_df=case_to_df,
        metric="test_acc",
        ylabel="Test Accuracy",
        title="Test Accuracy During Training",
        out_png=outdir / "test_accuracy_five_methods.png",
        out_pdf=outdir / "test_accuracy_five_methods.pdf",
    )

    plot_metric(
        case_to_df=case_to_df,
        metric="train_acc",
        ylabel="Training Accuracy (measured against true labels)",
        title="Training Accuracy During Training (True Labels)",
        out_png=outdir / "train_true_accuracy_five_methods.png",
        out_pdf=outdir / "train_true_accuracy_five_methods.pdf",
    )

    write_summary(case_to_df, outdir / "five_method_accuracy_summary.csv")

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()