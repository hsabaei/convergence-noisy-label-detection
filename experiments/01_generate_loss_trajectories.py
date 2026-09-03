"""Generate the common CIFAR-10 noisy-label loss trajectories.

This script performs the shared training run used by both CKL and proposed-LE
experiments.  It trains CNN12 on one synthetically corrupted CIFAR-10 training
set and, after every epoch, evaluates the same samples with deterministic
preprocessing to record per-sample cross-entropy loss trajectories.

The saved trajectory artifact is intentionally method-agnostic: later CKL and
LE scripts must load this same file so that their comparison is paired and does
not depend on separate network-training runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


# Allow this script to be executed directly from a source checkout without
# requiring an editable install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convergence_monitoring.data import NoisyCIFAR10WithIndex
from convergence_monitoring.models import CNN12_Model
from convergence_monitoring.training import evaluate_group_diagnostics, set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CNN12 once on synthetically corrupted CIFAR-10 and save the "
            "common per-sample loss trajectories for CKL and proposed-LE."
        )
    )

    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "common_loss_trajectories")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10.")

    parser.add_argument("--noisy-frac", type=float, default=0.05)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Also save the final model/optimizer state. The score pipelines do not require it.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.noisy_frac < 1.0):
        parser.error("--noisy-frac must be in [0, 1).")
    if args.num_epochs < 1:
        parser.error("--num-epochs must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative.")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative.")

    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(requested)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_n = int(y.numel())
            total_loss += float(loss.item()) * batch_n
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_n

    return {
        "train_loss": total_loss / max(total, 1),
        "train_acc": total_correct / max(total, 1),
    }


def write_epoch_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "eval_loss_all",
        "eval_acc_all",
        "eval_loss_clean",
        "eval_acc_clean",
        "eval_loss_noisy",
        "eval_acc_noisy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    print("=== Common noisy-label loss-trajectory run ===")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"noisy_frac: {args.noisy_frac}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"output_dir: {output_dir}")

    train_transform, eval_transform = build_transforms()

    # Two dataset views are constructed with the same seed.  They therefore
    # share the exact same corrupted labels and anomaly mask; only transforms
    # differ.  Training uses augmentation, while trajectory measurement is
    # deterministic.
    train_ds = NoisyCIFAR10WithIndex(
        root=str(args.data_root),
        train=True,
        transform=train_transform,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )
    eval_ds = NoisyCIFAR10WithIndex(
        root=str(args.data_root),
        train=True,
        transform=eval_transform,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )

    if not np.array_equal(train_ds.true_targets, eval_ds.true_targets):
        raise RuntimeError("Training and evaluation datasets have different true labels.")
    if not np.array_equal(train_ds.targets, eval_ds.targets):
        raise RuntimeError("Training and evaluation datasets have different corrupted labels.")
    if not np.array_equal(train_ds.is_anomaly, eval_ds.is_anomaly):
        raise RuntimeError("Training and evaluation datasets have different noisy-label masks.")

    n_samples = len(train_ds)
    n_noisy = int(np.sum(train_ds.is_anomaly))
    n_clean = n_samples - n_noisy

    print(f"n_samples: {n_samples}")
    print(f"n_clean: {n_clean}")
    print(f"n_noisy: {n_noisy}")

    pin_memory = device.type == "cuda"

    # The generator fixes the order used by the shuffled training loader.
    loader_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = CNN12_Model(num_classes=args.num_classes).to(device)
    # Preserve the repo's CNN12 dropout setting while allowing it to be changed
    # without editing the script.
    if hasattr(model, "s1") and args.dropout != 0.1:
        # CNN12_Model does not expose p_drop, so reconstruct only when needed.
        from convergence_monitoring.models import CNN12

        model = CNN12(num_classes=args.num_classes, p_drop=args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # evaluate_group_diagnostics needs a subset mask for its legacy summary
    # fields.  A zero mask is sufficient because this experiment uses the
    # per-sample loss/correctness arrays, not those subset summaries.
    dummy_subset_mask = torch.zeros(n_samples, dtype=torch.bool)

    loss_history: list[np.ndarray] = []
    correct_history: list[np.ndarray] = []
    epoch_rows: list[dict[str, Any]] = []

    clean_mask = ~train_ds.is_anomaly
    noisy_mask = train_ds.is_anomaly

    for epoch_idx in range(args.num_epochs):
        epoch = epoch_idx + 1

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        diag = evaluate_group_diagnostics(
            model=model,
            loader=eval_loader,
            device=str(device),
            subset_mask_table=dummy_subset_mask,
            num_classes=args.num_classes,
        )

        loss_full = np.asarray(diag["loss_full"], dtype=np.float64)
        correct_full = np.asarray(diag["correct_full"], dtype=np.float64)
        observed_from_loader = np.asarray(diag["label_full"], dtype=np.int64)

        if not np.array_equal(observed_from_loader, eval_ds.targets):
            raise RuntimeError(f"Observed labels changed during evaluation at epoch {epoch}.")
        if not np.all(np.isfinite(loss_full)):
            raise RuntimeError(f"Nonfinite per-sample loss encountered at epoch {epoch}.")

        loss_history.append(loss_full)
        correct_history.append(correct_full)

        eval_loss_all = float(np.mean(loss_full))
        eval_acc_all = float(np.mean(correct_full))
        eval_loss_clean = float(np.mean(loss_full[clean_mask])) if np.any(clean_mask) else np.nan
        eval_acc_clean = float(np.mean(correct_full[clean_mask])) if np.any(clean_mask) else np.nan
        eval_loss_noisy = float(np.mean(loss_full[noisy_mask])) if np.any(noisy_mask) else np.nan
        eval_acc_noisy = float(np.mean(correct_full[noisy_mask])) if np.any(noisy_mask) else np.nan

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["train_loss"],
                "train_acc": train_stats["train_acc"],
                "eval_loss_all": eval_loss_all,
                "eval_acc_all": eval_acc_all,
                "eval_loss_clean": eval_loss_clean,
                "eval_acc_clean": eval_acc_clean,
                "eval_loss_noisy": eval_loss_noisy,
                "eval_acc_noisy": eval_acc_noisy,
            }
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.num_epochs:
            print(
                f"epoch={epoch:03d}/{args.num_epochs} "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"train_acc={train_stats['train_acc']:.4f} "
                f"clean_loss={eval_loss_clean:.4f} "
                f"noisy_loss={eval_loss_noisy:.4f}"
            )

    # Historical code uses [N, T], so keep that orientation for downstream
    # compatibility with CKL and proposed-LE score construction.
    loss_traj = np.stack(loss_history, axis=1)
    correct_traj = np.stack(correct_history, axis=1)

    trajectory_path = output_dir / "cifar10_noisy_label_loss_trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        sample_index=np.arange(n_samples, dtype=np.int64),
        epoch=np.arange(1, args.num_epochs + 1, dtype=np.int64),
        loss_traj=loss_traj.astype(np.float32),
        correct_traj=correct_traj.astype(np.float32),
        true_label=np.asarray(train_ds.true_targets, dtype=np.int64),
        observed_label=np.asarray(train_ds.targets, dtype=np.int64),
        is_anomaly=np.asarray(train_ds.is_anomaly, dtype=bool),
        noisy_indices=np.flatnonzero(train_ds.is_anomaly).astype(np.int64),
    )

    summary_path = output_dir / "epoch_summary.csv"
    write_epoch_summary(summary_path, epoch_rows)

    metadata = {
        "artifact": "common_cifar10_noisy_label_loss_trajectories",
        "trajectory_orientation": "loss_traj[sample_index, epoch_index] = shape [N, T]",
        "dataset": "CIFAR-10 train",
        "n_samples": n_samples,
        "n_clean": n_clean,
        "n_noisy": n_noisy,
        "num_classes": args.num_classes,
        "noisy_frac_requested": args.noisy_frac,
        "noisy_frac_realized": n_noisy / n_samples,
        "corruption_rule": "uniformly replace selected label with one of the other classes",
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model": "CNN12",
        "dropout": args.dropout,
        "train_augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
        "evaluation_transform": "deterministic ToTensor + Normalize",
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    metadata_path = output_dir / "run_config.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.save_model:
        model_path = output_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metadata": metadata,
            },
            model_path,
        )
        print(f"saved model: {model_path}")

    print("\nSaved common experiment artifacts:")
    print(f"  trajectories: {trajectory_path}")
    print(f"  epoch summary: {summary_path}")
    print(f"  run config: {metadata_path}")
    print(f"  loss_traj shape: {loss_traj.shape}")
    print("\nUse this same trajectory NPZ for both CKL and proposed-LE experiments.")


if __name__ == "__main__":
    main()
