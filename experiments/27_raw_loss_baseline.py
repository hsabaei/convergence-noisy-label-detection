from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def first_existing_key(data, candidates):
    for key in candidates:
        if key in data.files:
            return key
    return None


def load_common_npz(path):
    data = np.load(path, allow_pickle=True)
    print("Available keys in input NPZ:")
    for k in data.files:
        arr = data[k]
        print(f"  {k:30s} shape={getattr(arr, 'shape', None)}")

    loss_key = first_existing_key(data, [
        "loss_trajectories", "loss_traj", "losses",
        "per_sample_losses", "sample_losses",
        "train_loss_trajectories",
    ])
    observed_key = first_existing_key(data, [
        "observed_labels", "observed_label", "noisy_labels",
        "train_labels", "targets", "labels",
    ])
    true_key = first_existing_key(data, [
        "true_labels", "true_label", "original_labels", "clean_labels",
    ])
    noisy_key = first_existing_key(data, [
        "is_anomaly", "noisy_mask", "is_noisy", "corruption_mask",
    ])
    epoch_key = first_existing_key(data, ["epochs", "epoch"])

    if loss_key is None:
        raise KeyError(f"Could not locate per-sample losses. Available keys: {data.files}")
    if observed_key is None:
        raise KeyError(f"Could not locate observed labels. Available keys: {data.files}")

    loss = np.asarray(data[loss_key], dtype=np.float64)
    observed = np.asarray(data[observed_key], dtype=np.int64).reshape(-1)

    if noisy_key is not None:
        noisy = np.asarray(data[noisy_key], dtype=bool).reshape(-1)
    elif true_key is not None:
        true = np.asarray(data[true_key], dtype=np.int64).reshape(-1)
        noisy = observed != true
    else:
        raise KeyError("Need either a noise mask or true labels for evaluation.")

    if loss.ndim != 2:
        raise ValueError(f"Expected 2-D loss array, got {loss.shape}")

    n = len(observed)
    if loss.shape[0] == n:
        pass
    elif loss.shape[1] == n:
        print("Transposing loss trajectories from [T,N] to [N,T].")
        loss = loss.T
    else:
        raise ValueError(f"Loss shape {loss.shape} does not match labels {n}.")

    T = loss.shape[1]
    if epoch_key is not None:
        epochs = np.asarray(data[epoch_key]).reshape(-1)
        if len(epochs) != T:
            epochs = np.arange(1, T + 1)
    else:
        epochs = np.arange(1, T + 1)

    return {"loss": loss, "observed": observed, "noisy": noisy, "epochs": epochs}


def average_ranks_01(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    x = values[valid]
    m = len(x)
    if m == 0:
        return out
    if m == 1:
        out[np.where(valid)[0][0]] = 0.5
        return out
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(m, dtype=np.float64)
    start = 0
    while start < m:
        end = start + 1
        while end < m and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_pos = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_pos / (m - 1)
        start = end
    out[valid] = ranks
    return out


def average_ranks_unscaled(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_x = values[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def rank_by_observed_class(score, observed_labels):
    score = np.asarray(score, dtype=np.float64)
    N, T = score.shape
    ranks = np.full((N, T), np.nan, dtype=np.float64)
    for c in np.unique(observed_labels):
        idx = np.where(observed_labels == c)[0]
        for t in range(T):
            ranks[idx, t] = average_ranks_01(score[idx, t])
    return ranks


def binary_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    y = y_true[valid]
    s = score[valid]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = average_ranks_unscaled(s)
    rank_sum_pos = ranks[y].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def class_top_q(score_t, observed_labels, q):
    """Instantaneous top-q within each observed class. No temporal memory."""
    score_t = np.asarray(score_t, dtype=np.float64)
    detected = np.zeros(len(score_t), dtype=bool)
    for c in np.unique(observed_labels):
        idx = np.where(observed_labels == c)[0]
        finite_idx = idx[np.isfinite(score_t[idx])]
        if len(finite_idx) == 0:
            continue
        k = int(np.ceil(q * len(finite_idx)))
        k = max(1, min(k, len(finite_idx)))
        vals = score_t[finite_idx]
        chosen = np.argpartition(vals, len(vals) - k)[-k:]
        detected[finite_idx[chosen]] = True
    return detected


def tpr_fpr(detected, noisy_mask):
    detected = np.asarray(detected, dtype=bool)
    noisy_mask = np.asarray(noisy_mask, dtype=bool)
    pos = noisy_mask
    neg = ~noisy_mask
    tpr = detected[pos].mean() if pos.any() else np.nan
    fpr = detected[neg].mean() if neg.any() else np.nan
    return float(tpr), float(fpr)


def range_mean(df, column, lo, hi):
    x = df.loc[df["epoch"].between(lo, hi), column]
    return float(x.mean()) if len(x) else np.nan


def plot_auc(df, outpath):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["epoch"], df["raw_loss_auc"], linewidth=2, label="Raw loss")
    ax.axhline(0.5, linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Instantaneous Raw-Loss Noisy-Label Discrimination")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tpr_fpr(df, q, outpath):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["epoch"], df["TPR"], linewidth=2, label="TPR")
    ax.plot(df["epoch"], df["FPR"], linewidth=2, label="FPR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Instantaneous Raw Loss: Class-Conditional Top-{q:g}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=(
        "Instantaneous raw-loss noisy-label baseline. No EWMA, cumulative pairwise, "
        "min-run, sliding window, or any other temporal detector is used."
    ))
    p.add_argument("--input", default=(
        "results/common_loss_trajectories/"
        "cifar10_noisy_label_loss_trajectories.npz"
    ))
    p.add_argument("--output-dir", default="results/raw_loss_baseline_instantaneous")
    p.add_argument("--q", type=float, default=0.10, help=(
        "Instantaneous top fraction within each observed class. This is only an "
        "operating point for TPR/FPR, not a temporal detector."
    ))
    p.add_argument("--start-epoch", type=int, default=42, help=(
        "First reported epoch. Use 42 to align with the K=40 CKL/LE comparison."
    ))
    args = p.parse_args()

    if not (0.0 < args.q < 1.0):
        raise ValueError("--q must lie in (0,1).")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = load_common_npz(args.input)
    loss = d["loss"]
    observed = d["observed"]
    noisy = d["noisy"]
    epochs = d["epochs"]
    N, T = loss.shape

    candidates = np.where(epochs >= args.start_epoch)[0]
    if len(candidates) == 0:
        raise ValueError(f"start epoch {args.start_epoch} is beyond available epochs.")
    start_col = int(candidates[0])

    print()
    print("=" * 72)
    print("INSTANTANEOUS RAW LOSS BASELINE")
    print("=" * 72)
    print(f"N={N}, T={T}, noisy={int(noisy.sum())}, noise fraction={noisy.mean():.4f}")
    print(f"q={args.q}, start epoch={args.start_epoch}")
    print("Temporal detector: NONE")

    # Class ranks are computed independently at each epoch. This is not temporal smoothing.
    loss_class_rank = rank_by_observed_class(loss, observed)

    rows = []
    for t in range(start_col, T):
        auc_raw = binary_auc(noisy, loss[:, t])
        auc_class_rank = binary_auc(noisy, loss_class_rank[:, t])

        # Main operating-point baseline: top-q raw loss within observed class at this epoch.
        detected = class_top_q(loss[:, t], observed, args.q)
        tpr, fpr = tpr_fpr(detected, noisy)

        rows.append({
            "epoch": int(epochs[t]),
            "raw_loss_auc": auc_raw,
            "class_rank_loss_auc": auc_class_rank,
            "TPR": tpr,
            "FPR": fpr,
            "selected": int(detected.sum()),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "raw_loss_instantaneous_metrics.csv", index=False)

    summary = pd.DataFrame([{
        "score": "Raw loss (instantaneous)",
        "detector": "None",
        "EarlyAUC_50_80": range_mean(metrics_df, "raw_loss_auc", 50, 80),
        "AvgAUC_60_120": range_mean(metrics_df, "raw_loss_auc", 60, 120),
        "LateAUC_180_200": range_mean(metrics_df, "raw_loss_auc", 180, 200),
        "EarlyTPR_50_80": range_mean(metrics_df, "TPR", 50, 80),
        "AvgTPR_60_120": range_mean(metrics_df, "TPR", 60, 120),
        "LateTPR_180_200": range_mean(metrics_df, "TPR", 180, 200),
        "AvgFPR_60_120": range_mean(metrics_df, "FPR", 60, 120),
    }])
    summary.to_csv(outdir / "raw_loss_instantaneous_summary.csv", index=False)

    np.savez_compressed(
        outdir / "raw_loss_instantaneous_scores.npz",
        epoch=epochs,
        raw_loss=loss.astype(np.float32),
        raw_loss_class_rank=loss_class_rank.astype(np.float32),
        observed_label=observed,
        is_anomaly=noisy,
    )

    plot_auc(metrics_df, outdir / "raw_loss_instantaneous_auc.png")
    plot_tpr_fpr(metrics_df, args.q, outdir / "raw_loss_instantaneous_tpr_fpr.png")

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Saved to:", outdir)


if __name__ == "__main__":
    main()
