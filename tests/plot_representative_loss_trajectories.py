# plot_representative_loss_trajectories.py

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Plot style
# ============================================================

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 14
LINEWIDTH = 3.0


# ============================================================
# Loading helpers
# ============================================================

def first_existing_key(data, candidates):
    for key in candidates:
        if key in data.files:
            return key
    return None


def load_trajectory_file(path):
    data = np.load(path, allow_pickle=True)

    print("Available keys:")
    for key in data.files:
        arr = data[key]
        print(f"  {key:30s} shape={getattr(arr, 'shape', None)}")

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
            f"Could not find loss trajectories. Available keys: {data.files}"
        )

    if observed_key is None:
        raise KeyError(
            f"Could not find observed labels. Available keys: {data.files}"
        )

    losses = np.asarray(data[loss_key], dtype=np.float64)
    observed = np.asarray(data[observed_key], dtype=np.int64).reshape(-1)

    # Determine which samples are noisy
    if noisy_key is not None:
        noisy = np.asarray(data[noisy_key], dtype=bool).reshape(-1)

    elif true_key is not None:
        true = np.asarray(data[true_key], dtype=np.int64).reshape(-1)
        noisy = observed != true

    else:
        raise KeyError(
            "Need either a noisy-sample mask or true labels."
        )

    n = len(observed)

    # Ensure trajectory shape is [N, T]
    if losses.ndim != 2:
        raise ValueError(
            f"Expected a 2-D trajectory array, got shape {losses.shape}"
        )

    if losses.shape[0] == n:
        pass

    elif losses.shape[1] == n:
        print("Transposing trajectories from [T, N] to [N, T].")
        losses = losses.T

    else:
        raise ValueError(
            f"Loss shape {losses.shape} does not match "
            f"number of samples {n}."
        )

    # Epoch axis
    if epoch_key is not None:
        epochs = np.asarray(data[epoch_key]).reshape(-1)

        if len(epochs) != losses.shape[1]:
            print(
                "Epoch array length does not match trajectory length. "
                "Using 1,...,T instead."
            )
            epochs = np.arange(1, losses.shape[1] + 1)

    else:
        epochs = np.arange(1, losses.shape[1] + 1)

    return losses, observed, noisy, epochs


# ============================================================
# Representative sample selection
# ============================================================

def select_typical_sample(losses, indices):
    """
    Select the sample whose overall mean loss is closest to
    the median overall mean loss of the group.
    """

    indices = np.asarray(indices)

    if len(indices) == 0:
        return None

    mean_loss = np.mean(losses[indices], axis=1)

    median_value = np.median(mean_loss)

    closest_position = np.argmin(
        np.abs(mean_loss - median_value)
    )

    return indices[closest_position]


def select_difficult_clean(losses, clean_indices):
    """
    Select a difficult clean sample.

    We define difficulty using the largest average loss across
    training among clean samples.
    """

    clean_indices = np.asarray(clean_indices)

    if len(clean_indices) == 0:
        return None

    mean_loss = np.mean(
        losses[clean_indices],
        axis=1,
    )

    return clean_indices[np.argmax(mean_loss)]


def select_representatives(
    losses,
    observed,
    noisy,
    observed_class=None,
):
    """
    Select:
        1. Typical clean
        2. Difficult clean
        3. Typical noisy

    If observed_class is provided, all three are selected from
    the same observed class.
    """

    if observed_class is None:
        candidate_mask = np.ones(
            len(noisy),
            dtype=bool,
        )

    else:
        candidate_mask = observed == observed_class

    clean_indices = np.where(
        (~noisy) & candidate_mask
    )[0]

    noisy_indices = np.where(
        noisy & candidate_mask
    )[0]

    if len(clean_indices) == 0:
        raise RuntimeError(
            "No clean samples found for the requested selection."
        )

    if len(noisy_indices) == 0:
        raise RuntimeError(
            "No noisy samples found for the requested selection. "
            "Try another observed class or omit --observed-class."
        )

    typical_clean = select_typical_sample(
        losses,
        clean_indices,
    )

    difficult_clean = select_difficult_clean(
        losses,
        clean_indices,
    )

    typical_noisy = select_typical_sample(
        losses,
        noisy_indices,
    )

    # Avoid selecting the same clean sample twice
    if difficult_clean == typical_clean and len(clean_indices) > 1:

        mean_loss = np.mean(
            losses[clean_indices],
            axis=1,
        )

        order = np.argsort(mean_loss)

        difficult_clean = clean_indices[order[-2]]

    return (
        typical_clean,
        difficult_clean,
        typical_noisy,
    )


# ============================================================
# Plot
# ============================================================

def plot_representative_trajectories(
    losses,
    epochs,
    typical_clean,
    difficult_clean,
    typical_noisy,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # --------------------------------------------------------
    # Typical clean
    # --------------------------------------------------------

    ax.plot(
        epochs,
        losses[typical_clean],
        linewidth=LINEWIDTH,
        label="Typical clean",
    )

    # --------------------------------------------------------
    # Difficult clean
    # --------------------------------------------------------

    ax.plot(
        epochs,
        losses[difficult_clean],
        linewidth=LINEWIDTH,
        linestyle="--",
        label="Difficult clean",
    )

    # --------------------------------------------------------
    # Typical noisy
    # --------------------------------------------------------

    ax.plot(
        epochs,
        losses[typical_noisy],
        linewidth=LINEWIDTH,
        linestyle="-.",
        label="Typical noisy",
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "Per-Sample Loss Trajectories",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=15,
    )

    # --------------------------------------------------------
    # Axis titles
    # --------------------------------------------------------

    ax.set_xlabel(
        "Epoch",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        labelpad=8,
    )

    ax.set_ylabel(
        "Loss",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        labelpad=8,
    )

    # --------------------------------------------------------
    # Tick numbers
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        width=1.5,
    )

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        loc="best",
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.25,
    )

    # Slightly thicker plot borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved figure to:")
    print(output_path)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot three representative per-sample loss trajectories: "
            "typical clean, difficult clean, and typical noisy."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to loss-trajectory NPZ file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/loss_trajectory_figures/"
            "per_sample_loss_trajectories.png"
        ),
        help="Output PNG path.",
    )

    parser.add_argument(
        "--observed-class",
        type=int,
        default=None,
        help=(
            "Optional observed class. "
            "If supplied, all selected samples belong to "
            "the same observed class."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    losses, observed, noisy, epochs = load_trajectory_file(
        args.input
    )

    print("\nDataset summary")
    print("----------------------------")
    print(f"Number of samples : {len(noisy)}")
    print(f"Number clean      : {(~noisy).sum()}")
    print(f"Number noisy      : {noisy.sum()}")
    print(f"Number epochs     : {losses.shape[1]}")

    if args.observed_class is not None:

        class_mask = observed == args.observed_class

        print(
            f"Observed class     : {args.observed_class}"
        )

        print(
            "Clean in class     : "
            f"{np.sum((~noisy) & class_mask)}"
        )

        print(
            "Noisy in class     : "
            f"{np.sum(noisy & class_mask)}"
        )

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    (
        typical_clean,
        difficult_clean,
        typical_noisy,
    ) = select_representatives(
        losses=losses,
        observed=observed,
        noisy=noisy,
        observed_class=args.observed_class,
    )

    print("\nSelected samples")
    print("----------------------------")
    print(
        f"Typical clean  : sample {typical_clean}"
    )
    print(
        f"Difficult clean: sample {difficult_clean}"
    )
    print(
        f"Typical noisy  : sample {typical_noisy}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_representative_trajectories(
        losses=losses,
        epochs=epochs,
        typical_clean=typical_clean,
        difficult_clean=difficult_clean,
        typical_noisy=typical_noisy,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()