
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
        train_all_col = "adaptive_train_true_accuracy_all"
        train_noisy_col = "adaptive_train_true_accuracy_noisy"
        test_col = "adaptive_test_accuracy"
    else:
        train_all_col = "baseline_train_true_accuracy_all"
        train_noisy_col = "baseline_train_true_accuracy_noisy"
        test_col = "baseline_test_accuracy"

    require_columns(
        df,
        path,
        ["epoch", train_all_col, train_noisy_col, test_col],
    )

    return pd.DataFrame({
        "epoch": df["epoch"],
        "train_all": df[train_all_col],
        "train_noisy": df[train_noisy_col],
        "test": df[test_col],
    })


def load_d2l(path):
    df = read_csv(path)

    train_all_candidates = [
        "d2l_train_true_accuracy_all",
        "train_true_accuracy_all",
    ]
    train_noisy_candidates = [
        "d2l_train_true_accuracy_noisy",
        "train_true_accuracy_noisy",
    ]
    test_candidates = [
        "d2l_test_accuracy",
        "test_accuracy",
    ]

    def pick(cands, what):
        col = next((c for c in cands if c in df.columns), None)
        if col is None:
            raise KeyError(
                f"Could not find {what} in {path}\n"
                f"Tried: {cands}\n"
                f"Available columns: {list(df.columns)}"
            )
        return col

    train_all_col = pick(train_all_candidates, "D2L overall training accuracy")
    train_noisy_col = pick(train_noisy_candidates, "D2L noisy-subset training accuracy")
    test_col = pick(test_candidates, "D2L test accuracy")

    return pd.DataFrame({
        "epoch": df["epoch"],
        "train_all": df[train_all_col],
        "train_noisy": df[train_noisy_col],
        "test": df[test_col],
    })


def plot_curves(series, metric, ylabel, title, output_path, ylim):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for label, df in series.items():
        ax.plot(
            df["epoch"],
            100.0 * df[metric],
            linewidth=2.1,
            label=label,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.set_xlim(left=1)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()

    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(series, out_csv):
    rows = []

    for label, df in series.items():
        best_test_idx = df["test"].idxmax()

        late = df[df["epoch"].between(150, 200)]

        rows.append({
            "method": label,
            "final_train_true_accuracy_all":
                float(df["train_all"].iloc[-1]),
            "final_train_true_accuracy_noisy":
                float(df["train_noisy"].iloc[-1]),
            "final_test_accuracy":
                float(df["test"].iloc[-1]),
            "best_test_accuracy":
                float(df.loc[best_test_idx, "test"]),
            "best_test_epoch":
                int(df.loc[best_test_idx, "epoch"]),
            "avg_test_accuracy_epochs_150_200":
                float(late["test"].mean()),
            "avg_train_true_noisy_epochs_150_200":
                float(late["train_noisy"].mean()),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)


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
        default="results/five_method_accuracy_plots_v2",
    )

    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    # 1) Zoomed clean test accuracy.
    plot_curves(
        series,
        metric="test",
        ylabel="Test accuracy (%)",
        title="Test Accuracy During Training",
        output_path=outdir / "test_accuracy_five_methods_zoomed",
        ylim=(89, 93),
    )

    # 2) Zoomed overall true-label training accuracy.
    plot_curves(
        series,
        metric="train_all",
        ylabel="Training accuracy (%)",
        title="Training Accuracy During Training",
        output_path=outdir / "training_accuracy_all_five_methods_zoomed",
        ylim=(93, 99),
    )

    # 3) True-label training accuracy on only the synthetically noisy samples.
    plot_curves(
        series,
        metric="train_noisy",
        ylabel="Accuracy on noisy samples (%)",
        title="True-Label Accuracy on Noisy Training Samples",
        output_path=outdir / "training_accuracy_noisy_subset_five_methods",
        ylim=(0, 100),
    )

    write_summary(
        series,
        outdir / "five_method_accuracy_summary_v2.csv",
    )

    print("Created:")
    for name in [
        "test_accuracy_five_methods_zoomed.png",
        "test_accuracy_five_methods_zoomed.pdf",
        "training_accuracy_all_five_methods_zoomed.png",
        "training_accuracy_all_five_methods_zoomed.pdf",
        "training_accuracy_noisy_subset_five_methods.png",
        "training_accuracy_noisy_subset_five_methods.pdf",
        "five_method_accuracy_summary_v2.csv",
    ]:
        print(outdir / name)


if __name__ == "__main__":
    main()
