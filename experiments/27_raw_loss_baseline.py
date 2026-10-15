from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Helpers for loading the common trajectory artifact
# ============================================================

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

    loss_key = first_existing_key(
        data,
        [
            "loss_trajectories",
            "loss_traj",
            "losses",
            "per_sample_losses",
            "sample_losses",
            "train_loss_trajectories",
        ],
    )

    observed_key = first_existing_key(
        data,
        [
            "observed_labels",
            "observed_label",
            "noisy_labels",
            "train_labels",
            "targets",
            "labels",
        ],
    )

    true_key = first_existing_key(
        data,
        [
            "true_labels",
            "true_label",
            "original_labels",
            "clean_labels",
        ],
    )

    noisy_key = first_existing_key(
        data,
        [
            "is_anomaly",
            "noisy_mask",
            "is_noisy",
            "corruption_mask",
        ],
    )

    epoch_key = first_existing_key(
        data,
        [
            "epochs",
            "epoch",
        ],
    )

    if loss_key is None:
        raise KeyError(
            "Could not locate the per-sample loss trajectory array.\n"
            f"Available keys: {data.files}"
        )

    if observed_key is None:
        raise KeyError(
            "Could not locate observed/noisy training labels.\n"
            f"Available keys: {data.files}"
        )

    loss = np.asarray(data[loss_key], dtype=np.float64)
    observed = np.asarray(data[observed_key], dtype=np.int64).reshape(-1)

    if noisy_key is not None:
        noisy = np.asarray(data[noisy_key], dtype=bool).reshape(-1)
    elif true_key is not None:
        true = np.asarray(data[true_key], dtype=np.int64).reshape(-1)
        noisy = observed != true
    else:
        raise KeyError(
            "Need either a noise mask or true labels for evaluation."
        )

    # Expected orientation in your score artifacts is [N, T].
    # Detect and correct [T, N] if necessary.
    if loss.ndim != 2:
        raise ValueError(
            f"Expected a 2-D loss trajectory array, got {loss.shape}"
        )

    n = len(observed)

    if loss.shape[0] == n:
        pass
    elif loss.shape[1] == n:
        print("Transposing loss trajectories from [T,N] to [N,T].")
        loss = loss.T
    else:
        raise ValueError(
            "Loss array dimensions do not match number of labels:\n"
            f"loss shape = {loss.shape}, labels = {n}"
        )

    T = loss.shape[1]

    if epoch_key is not None:
        epochs = np.asarray(data[epoch_key]).reshape(-1)
        if len(epochs) != T:
            epochs = np.arange(1, T + 1)
    else:
        epochs = np.arange(1, T + 1)

    return {
        "loss": loss,
        "observed": observed,
        "noisy": noisy,
        "epochs": epochs,
    }


# ============================================================
# Rank computation
# ============================================================

def average_ranks_01(values):
    """
    Average empirical rank in [0,1].

    Lowest finite value -> near 0
    Highest finite value -> near 1

    Ties receive average rank.
    """
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

        # Positions 0,...,m-1 scaled to [0,1].
        average_position = 0.5 * (start + end - 1)
        rank_value = average_position / (m - 1)

        ranks[order[start:end]] = rank_value
        start = end

    out[valid] = ranks
    return out


def rank_by_observed_class(score, observed_labels):
    """
    score: [N,T]
    returns class-conditional empirical ranks [N,T].
    """
    score = np.asarray(score, dtype=np.float64)
    N, T = score.shape

    ranks = np.full((N, T), np.nan, dtype=np.float64)

    for c in np.unique(observed_labels):
        idx = np.where(observed_labels == c)[0]

        for t in range(T):
            ranks[idx, t] = average_ranks_01(score[idx, t])

    return ranks


# ============================================================
# Binary ROC-AUC without sklearn dependency
# ============================================================

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

    # AUC = probability that a random positive outranks a random negative.
    # Use average ranks for ties.
    ranks = average_ranks_unscaled(s)

    rank_sum_pos = ranks[y].sum()

    auc = (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def average_ranks_unscaled(values):
    """
    Average ranks numbered 1,...,n.
    Used for Mann-Whitney / AUC.
    """
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

        # 1-based average rank
        avg = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg

        start = end

    return ranks


# ============================================================
# Class-conditional top-q selection
# ============================================================

def class_top_q(score_t, observed_labels, q):
    """
    Select approximately the top q fraction within each observed class.

    score_t: [N]
    returns bool [N]
    """
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

        # Higher score = more suspicious.
        chosen_local = np.argpartition(
            vals,
            len(vals) - k,
        )[-k:]

        detected[finite_idx[chosen_local]] = True

    return detected


# ============================================================
# EWMA
# ============================================================

def compute_ewma(rank_score, start_col, lam):
    """
    EWMA of the within-class raw-loss rank.

    Before start_col values remain NaN.
    """
    N, T = rank_score.shape

    ewma = np.full((N, T), np.nan, dtype=np.float64)

    ewma[:, start_col] = rank_score[:, start_col]

    for t in range(start_col + 1, T):
        current = rank_score[:, t]
        previous = ewma[:, t - 1]

        both = np.isfinite(current) & np.isfinite(previous)
        only_current = np.isfinite(current) & ~np.isfinite(previous)

        ewma[both, t] = (
            (1.0 - lam) * previous[both]
            + lam * current[both]
        )

        ewma[only_current, t] = current[only_current]

    return ewma


# ============================================================
# Exact all-peer cumulative pairwise detector
# ============================================================

def epoch_pairwise_win_mass(score_t, observed_labels):
    """
    Compute exact same-class pairwise win mass efficiently using ranks.

    For each finite sample:
        win_mass =
            number of peers with lower score
            + 0.5 * number of peers with equal score

    The sample itself is excluded.

    Returns
    -------
    wins : [N]
    comparisons : [N]
    """
    score_t = np.asarray(score_t, dtype=np.float64)
    N = len(score_t)

    wins = np.zeros(N, dtype=np.float64)
    comparisons = np.zeros(N, dtype=np.float64)

    for c in np.unique(observed_labels):
        class_idx = np.where(observed_labels == c)[0]
        finite_mask = np.isfinite(score_t[class_idx])
        idx = class_idx[finite_mask]

        m = len(idx)

        if m <= 1:
            continue

        x = score_t[idx]

        order = np.argsort(x, kind="mergesort")
        sorted_x = x[order]

        sorted_win = np.zeros(m, dtype=np.float64)

        start = 0

        while start < m:
            end = start + 1

            while (
                end < m
                and sorted_x[end] == sorted_x[start]
            ):
                end += 1

            n_lower = start
            tie_group_size = end - start

            # Exclude self from tied peers.
            tied_other = tie_group_size - 1

            win_value = (
                n_lower
                + 0.5 * tied_other
            )

            sorted_win[start:end] = win_value
            start = end

        local_win = np.empty(m, dtype=np.float64)
        local_win[order] = sorted_win

        wins[idx] = local_win
        comparisons[idx] = m - 1

    return wins, comparisons


def compute_cumulative_pairwise(
    raw_score,
    observed_labels,
    start_col,
):
    """
    Accumulate exact wins and comparison counts:

        P_cum(i,t) =
            sum_tau W_i(tau)
            /
            sum_tau M_i(tau)

    Important:
    this is not an average of epoch percentiles.
    """
    N, T = raw_score.shape

    cumulative_score = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    total_wins = np.zeros(N, dtype=np.float64)
    total_comparisons = np.zeros(N, dtype=np.float64)

    for t in range(start_col, T):
        wins, comps = epoch_pairwise_win_mass(
            raw_score[:, t],
            observed_labels,
        )

        total_wins += wins
        total_comparisons += comps

        valid = total_comparisons > 0

        cumulative_score[valid, t] = (
            total_wins[valid]
            / total_comparisons[valid]
        )

    return cumulative_score


# ============================================================
# Metrics
# ============================================================

def tpr_fpr(detected, noisy_mask):
    detected = np.asarray(detected, dtype=bool)
    noisy_mask = np.asarray(noisy_mask, dtype=bool)

    pos = noisy_mask
    neg = ~noisy_mask

    tpr = (
        detected[pos].mean()
        if pos.any()
        else np.nan
    )

    fpr = (
        detected[neg].mean()
        if neg.any()
        else np.nan
    )

    return float(tpr), float(fpr)


def epoch_detector_metrics(
    detector_score,
    observed_labels,
    noisy_mask,
    epochs,
    q,
    start_col,
):
    rows = []

    for t in range(start_col, detector_score.shape[1]):
        detected = class_top_q(
            detector_score[:, t],
            observed_labels,
            q,
        )

        tpr, fpr = tpr_fpr(
            detected,
            noisy_mask,
        )

        rows.append(
            {
                "epoch": int(epochs[t]),
                "TPR": tpr,
                "FPR": fpr,
                "selected": int(detected.sum()),
            }
        )

    return pd.DataFrame(rows)


def range_mean(df, column, lo, hi):
    x = df.loc[
        df["epoch"].between(lo, hi),
        column,
    ]

    if len(x) == 0:
        return np.nan

    return float(x.mean())


def summarize_detector(name, df):
    return {
        "score": "Raw loss",
        "detector": name,
        "EarlyTPR_50_80":
            range_mean(df, "TPR", 50, 80),
        "AvgTPR_60_120":
            range_mean(df, "TPR", 60, 120),
        "LateTPR_180_200":
            range_mean(df, "TPR", 180, 200),
        "AvgFPR_60_120":
            range_mean(df, "FPR", 60, 120),
    }


# ============================================================
# Plotting
# ============================================================

def plot_detector(df, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        df["epoch"],
        df["TPR"],
        linewidth=2,
        label="TPR",
    )

    ax.plot(
        df["epoch"],
        df["FPR"],
        linewidth=2,
        label="FPR",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default=(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )

    p.add_argument(
        "--output-dir",
        default="results/raw_loss_baseline",
    )

    p.add_argument(
        "--q",
        type=float,
        default=0.10,
        help=(
            "Upper within-class detection fraction. "
            "Use 0.10 for 5/10%% experiments and "
            "0.20 for standard 20%% detector comparison."
        ),
    )

    p.add_argument(
        "--ewma-lambda",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--start-epoch",
        type=int,
        default=42,
        help=(
            "First epoch included in monitoring. "
            "42 aligns with the K=40 LE/CKL experiment."
        ),
    )

    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    d = load_common_npz(args.input)

    loss = d["loss"]
    observed = d["observed"]
    noisy = d["noisy"]
    epochs = d["epochs"]

    N, T = loss.shape

    print()
    print("=" * 72)
    print("RAW LOSS BASELINE")
    print("=" * 72)
    print(f"N={N}")
    print(f"T={T}")
    print(f"noisy={int(noisy.sum())}")
    print(f"noise fraction={noisy.mean():.4f}")
    print(f"q={args.q}")
    print(f"EWMA lambda={args.ewma_lambda}")
    print(f"start epoch={args.start_epoch}")

    # --------------------------------------------------------
    # Align start column
    # --------------------------------------------------------

    candidates = np.where(
        epochs >= args.start_epoch
    )[0]

    if len(candidates) == 0:
        raise ValueError(
            f"start epoch {args.start_epoch} "
            f"is beyond available epochs."
        )

    start_col = int(candidates[0])

    # --------------------------------------------------------
    # 1. Instantaneous raw-loss score AUC
    # --------------------------------------------------------

    auc_rows = []

    for t in range(start_col, T):
        auc_raw = binary_auc(
            noisy,
            loss[:, t],
        )

        auc_rows.append(
            {
                "epoch": int(epochs[t]),
                "raw_loss_auc": auc_raw,
            }
        )

    auc_df = pd.DataFrame(auc_rows)

    # --------------------------------------------------------
    # 2. Within-observed-class loss rank
    # --------------------------------------------------------

    print(
        "Computing within-observed-class raw-loss ranks..."
    )

    loss_rank = rank_by_observed_class(
        loss,
        observed,
    )

    # --------------------------------------------------------
    # 3. EWMA
    # --------------------------------------------------------

    print("Computing EWMA raw-loss baseline...")

    loss_ewma = compute_ewma(
        loss_rank,
        start_col=start_col,
        lam=args.ewma_lambda,
    )

    ewma_df = epoch_detector_metrics(
        detector_score=loss_ewma,
        observed_labels=observed,
        noisy_mask=noisy,
        epochs=epochs,
        q=args.q,
        start_col=start_col,
    )

    ewma_df["detector"] = "EWMA"

    # --------------------------------------------------------
    # 4. Cumulative pairwise
    # --------------------------------------------------------

    print(
        "Computing exact cumulative pairwise "
        "raw-loss baseline..."
    )

    pairwise_score = compute_cumulative_pairwise(
        raw_score=loss,
        observed_labels=observed,
        start_col=start_col,
    )

    pairwise_df = epoch_detector_metrics(
        detector_score=pairwise_score,
        observed_labels=observed,
        noisy_mask=noisy,
        epochs=epochs,
        q=args.q,
        start_col=start_col,
    )

    pairwise_df["detector"] = (
        "Cumulative pairwise"
    )

    # --------------------------------------------------------
    # Save epoch metrics
    # --------------------------------------------------------

    epoch_df = pd.concat(
        [
            ewma_df,
            pairwise_df,
        ],
        ignore_index=True,
    )

    epoch_df.to_csv(
        outdir / "raw_loss_epoch_metrics.csv",
        index=False,
    )

    auc_df.to_csv(
        outdir / "raw_loss_score_auc.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary metrics matching thesis tables
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            summarize_detector(
                "EWMA",
                ewma_df,
            ),
            summarize_detector(
                "Cumulative pairwise",
                pairwise_df,
            ),
        ]
    )

    summary.to_csv(
        outdir / "raw_loss_summary.csv",
        index=False,
    )

    print()
    print("Summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Save score arrays
    # --------------------------------------------------------

    np.savez_compressed(
        outdir / "raw_loss_baseline_scores.npz",
        epoch=epochs,
        raw_loss=loss.astype(np.float32),
        raw_loss_class_rank=loss_rank.astype(
            np.float32
        ),
        raw_loss_ewma=loss_ewma.astype(
            np.float32
        ),
        raw_loss_cumulative_pairwise=(
            pairwise_score.astype(np.float32)
        ),
        observed_label=observed,
        is_anomaly=noisy,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_detector(
        ewma_df,
        (
            "Raw Loss Baseline: EWMA "
            f"(q={args.q:g}, lambda={args.ewma_lambda:g})"
        ),
        outdir
        / "raw_loss_baseline_ewma_tpr_fpr.png",
    )

    plot_detector(
        pairwise_df,
        (
            "Raw Loss Baseline: "
            f"Cumulative Pairwise (q={args.q:g})"
        ),
        outdir
        / "raw_loss_baseline_pairwise_tpr_fpr.png",
    )

    print()
    print("Saved:")
    print(
        outdir / "raw_loss_summary.csv"
    )
    print(
        outdir / "raw_loss_epoch_metrics.csv"
    )
    print(
        outdir / "raw_loss_score_auc.csv"
    )
    print(
        outdir / "raw_loss_baseline_scores.npz"
    )


if __name__ == "__main__":
    main()