# plot_representative_loss_trajectories.py

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


def first_existing_key(data, candidates):
    for key in candidates:
        if key in data.files:
            return key
    return None


def load_trajectory_file(path):
    data = np.load(path, allow_pickle=True)

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
        raise KeyError(f"Could not find loss trajectories. Keys: {data.files}")

    if observed_key is None:
        raise KeyError(f"Could not find observed labels. Keys: {data.files}")

    losses = np.asarray(data[loss_key], dtype=np.float64)
    observed = np.asarray(data[observed_key], dtype=np.int64).reshape(-1)

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

    # Convert to [N, T]
    if losses.shape[0] == n:
        pass
    elif losses.shape[1] == n:
        losses = losses.T
    else:
        raise ValueError(
            f"Loss shape {losses.shape} does not match number of samples {n}"
        )

    if epoch_key is not None:
        epochs = np.asarray(data[epoch_key]).reshape(-1)
        if len(epochs) != losses.shape[1]:
            epochs = np.arange(1, losses.shape[1] + 1)
    else:
        epochs = np.arange(1, losses.shape[1] + 1)

    return losses, observed, noisy, epochs


def select_representative_indices(losses, mask, n_select=4):
    """
    Select samples whose mean trajectory is close to the median
    trajectory level of the requested group.
    """
    indices = np.where(mask)[0]

    if len(indices) < n_select:
        return indices

    mean_loss = np.mean(losses[indices], axis=1)
    median_loss = np.median(mean_loss)

    order = np.argsort(np.abs(mean_loss - median_loss))

    return indices[order[:n_select]]


def select_difficult_clean(losses, clean_mask):
    """
    Clean sample with unusually high average loss.
    """
    indices = np.where(clean_mask)[0]
    mean_loss = np.mean(losses[indices], axis=1)

    return indices[np.argmax(mean_loss)]


def select_memorized_noisy(losses, noisy_mask):
    """
    Select a noisy sample that begins relatively high
    but ends relatively low.

    This is a simple heuristic for a possible memorization example.
    """
    indices = np.where(noisy_mask)[0]

    if len(indices) == 0:
        return None

    T = losses.shape[1]

    early_end = max(5, int(0.20 * T))
    late_start = max(0, int(0.80 * T))

    early = np.mean(losses[indices, :early_end], axis=1)
    late = np.mean(losses[indices, late_start:], axis=1)

    drop = early - late

    return indices[np.argmax(drop)]


def plot_group(
    losses,
    epochs,
    clean_indices,
    noisy_indices,
    difficult_clean=None,
    memorized_noisy=None,
    title="Representative per-sample loss trajectories",
    output_path="representative_loss_trajectories.png",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Typical clean trajectories
    for idx in clean_indices:
        ax.plot(
            epochs,
            losses[idx],
            linewidth=1.5,
            alpha=0.8,
            label="Typical clean" if idx == clean_indices[0] else None,
        )

    # Typical noisy trajectories
    for idx in noisy_indices:
        ax.plot(
            epochs,
            losses[idx],
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label="Typical noisy" if idx == noisy_indices[0] else None,
        )

    # Difficult clean sample
    if difficult_clean is not None:
        ax.plot(
            epochs,
            losses[difficult_clean],
            linewidth=2.5,
            label=f"Difficult clean (sample {difficult_clean})",
        )

    # Memorized noisy sample
    if memorized_noisy is not None:
        ax.plot(
            epochs,
            losses[memorized_noisy],
            linewidth=2.5,
            linestyle=":",
            label=f"Noisy with large loss drop (sample {memorized_noisy})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-sample loss")
    ax.set_title(title)

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to common loss-trajectory NPZ file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="representative_loss_trajectories.png",
    )

    parser.add_argument(
        "--observed-class",
        type=int,
        default=None,
        help="Optional observed class to restrict the comparison",
    )

    parser.add_argument(
        "--n-clean",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-noisy",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Representative per-sample loss trajectories",
    )

    args = parser.parse_args()

    losses, observed, noisy, epochs = load_trajectory_file(args.input)

    clean = ~noisy

    # Optional: restrict everything to one observed class
    if args.observed_class is not None:
        class_mask = observed == args.observed_class
    else:
        class_mask = np.ones_like(noisy, dtype=bool)

    clean_mask = clean & class_mask
    noisy_mask = noisy & class_mask

    print(f"Total samples: {len(noisy)}")
    print(f"Clean in selected group: {clean_mask.sum()}")
    print(f"Noisy in selected group: {noisy_mask.sum()}")

    if clean_mask.sum() == 0:
        raise RuntimeError("No clean samples found.")

    if noisy_mask.sum() == 0:
        raise RuntimeError(
            "No noisy samples found. Try another observed class or omit "
            "--observed-class."
        )

    clean_indices = select_representative_indices(
        losses,
        clean_mask,
        args.n_clean,
    )

    noisy_indices = select_representative_indices(
        losses,
        noisy_mask,
        args.n_noisy,
    )

    difficult_clean = select_difficult_clean(
        losses,
        clean_mask,
    )

    memorized_noisy = select_memorized_noisy(
        losses,
        noisy_mask,
    )

    print("Typical clean indices:", clean_indices)
    print("Typical noisy indices:", noisy_indices)
    print("Difficult clean index:", difficult_clean)
    print("Noisy large-drop index:", memorized_noisy)

    plot_group(
        losses=losses,
        epochs=epochs,
        clean_indices=clean_indices,
        noisy_indices=noisy_indices,
        difficult_clean=difficult_clean,
        memorized_noisy=memorized_noisy,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()