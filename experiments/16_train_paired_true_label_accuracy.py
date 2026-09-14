#!/usr/bin/env python
"""Paired baseline-vs-detector training with TRUE-label accuracy.

Two models are trained inside the SAME run:

    A) Baseline: ordinary cross-entropy on the observed/noisy labels.
    B) Adaptive: LE-GIE + rank-EWMA detector + D2L-style adaptive target.

Fairness controls
-----------------
* same noisy-label realization
* exact same initial model weights
* exact same minibatches
* exact same augmented images
* synchronized dropout RNG for the two forward passes
* same optimizer type and hyperparameters

IMPORTANT
---------
All reported TRAINING ACCURACIES are computed against the ORIGINAL TRUE
CIFAR-10 labels, never against the corrupted observed labels.

The observed labels are used only:
1) as the baseline training targets,
2) inside the adaptive target y*,
3) to construct the class-reference trajectory used by LE-GIE.

The true labels/noise mask are evaluation-only and never affect detection
or alpha.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convergence_monitoring.data import NoisyCIFAR10WithIndex
from convergence_monitoring.models import CNN12_Model
from convergence_monitoring.training import set_seed
from convergence_monitoring.adaptive_training import (
    adaptive_target_cross_entropy,
    alpha_from_rank_ewma,
    correction_strength,
    latest_le_next_next,
    topq_mask,
    update_rank_ewma,
    within_class_percentile,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "paired_true_label_training",
    )
    p.add_argument("--download", action="store_true")

    p.add_argument("--noisy-frac", type=float, default=0.05)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    # Frozen detector settings.
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--rank-ewma-lambda", type=float, default=0.05)

    # New adaptive-training controls.
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--alpha-floor", type=float, default=0.20)

    p.add_argument("--save-models", action="store_true")

    args = p.parse_args()

    if not (0.0 <= args.noisy_frac < 1.0):
        p.error("--noisy-frac must be in [0,1).")
    if not (0.0 < args.q < 1.0):
        p.error("--q must be in (0,1).")
    if not (0.0 < args.rank_ewma_lambda <= 1.0):
        p.error("--rank-ewma-lambda must be in (0,1].")
    if not (0.0 <= args.alpha_floor <= 1.0):
        p.error("--alpha-floor must be in [0,1].")
    if args.gamma_max < 0.0:
        p.error("--gamma-max must be nonnegative.")

    return args


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def build_transforms():
    # One shared training dataset/loader is used, so each augmented image is
    # generated once and fed to BOTH models.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Deterministic view of the training set for per-sample loss/detection
    # and true-label training accuracy.
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def capture_torch_rng(device):
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state


def restore_torch_rng(state, device):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def train_paired_one_epoch(
    baseline_model,
    adaptive_model,
    loader,
    baseline_optimizer,
    adaptive_optimizer,
    device,
    alpha_table,
    num_classes,
):
    """Train both models on exactly the same mini-batches/images.

    Dropout RNG is synchronized so the two models also receive the same
    dropout realization before their training objectives diverge.
    """
    baseline_model.train()
    adaptive_model.train()

    baseline_loss_sum = 0.0
    adaptive_loss_sum = 0.0
    total = 0

    for x, y_observed, idx in loader:
        x = x.to(device, non_blocking=True)
        y_observed = y_observed.to(device, non_blocking=True)
        idx_np = np.asarray(idx, dtype=np.int64)

        alpha = torch.from_numpy(alpha_table[idx_np]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        baseline_optimizer.zero_grad(set_to_none=True)
        adaptive_optimizer.zero_grad(set_to_none=True)

        # Save RNG immediately before baseline forward, then restore it before
        # adaptive forward. This makes dropout masks match.
        paired_rng = capture_torch_rng(device)

        baseline_logits = baseline_model(x)

        restore_torch_rng(paired_rng, device)
        adaptive_logits = adaptive_model(x)

        # Baseline always optimizes ordinary CE on the observed/noisy label.
        baseline_loss = F.cross_entropy(
            baseline_logits,
            y_observed,
        )

        # Adaptive model uses the D2L-style sample-specific target.
        adaptive_loss = adaptive_target_cross_entropy(
            adaptive_logits,
            y_observed,
            alpha,
            num_classes,
        )

        baseline_loss.backward()
        adaptive_loss.backward()

        baseline_optimizer.step()
        adaptive_optimizer.step()

        n = int(y_observed.numel())
        baseline_loss_sum += float(baseline_loss.item()) * n
        adaptive_loss_sum += float(adaptive_loss.item()) * n
        total += n

    return {
        "baseline_optimization_loss": baseline_loss_sum / max(total, 1),
        "adaptive_optimization_loss": adaptive_loss_sum / max(total, 1),
    }


@torch.no_grad()
def evaluate_training_set_true_labels(
    model,
    loader,
    device,
    true_labels,
    observed_labels,
    n_samples,
):
    """Deterministic pass over training set.

    Returns:
    * per-sample observed-label CE loss (needed by detector),
    * prediction,
    * correctness measured ONLY against true labels.
    """
    model.eval()

    loss_observed = np.full(n_samples, np.nan, dtype=np.float32)
    prediction = np.full(n_samples, -1, dtype=np.int64)
    true_correct = np.zeros(n_samples, dtype=bool)

    for x, y_observed_batch, idx in loader:
        x = x.to(device, non_blocking=True)
        y_observed_batch = y_observed_batch.to(device, non_blocking=True)
        idx_np = np.asarray(idx, dtype=np.int64)

        logits = model(x)
        per_loss = F.cross_entropy(
            logits,
            y_observed_batch,
            reduction="none",
        )
        pred = logits.argmax(dim=1).detach().cpu().numpy()

        loss_observed[idx_np] = per_loss.detach().cpu().numpy().astype(np.float32)
        prediction[idx_np] = pred

        true_batch = true_labels[idx_np]
        true_correct[idx_np] = pred == true_batch

    if np.any(prediction < 0):
        raise RuntimeError("Evaluation did not cover every training sample.")

    return {
        "loss_observed": loss_observed,
        "prediction": prediction,
        "true_correct": true_correct,
    }


def safe_accuracy(correct, mask):
    count = int(np.sum(mask))
    if count == 0:
        return np.nan, 0
    return float(np.mean(correct[mask])), count


def true_accuracy_summary(correct, true_labels, noisy_mask, num_classes):
    clean_mask = ~noisy_mask

    overall, overall_n = safe_accuracy(
        correct,
        np.ones_like(noisy_mask, dtype=bool),
    )
    clean_acc, clean_n = safe_accuracy(correct, clean_mask)
    noisy_acc, noisy_n = safe_accuracy(correct, noisy_mask)

    class_rows = []
    for c in range(num_classes):
        class_mask = true_labels == c

        for subset_name, subset_mask in (
            ("all", class_mask),
            ("clean", class_mask & clean_mask),
            ("noisy", class_mask & noisy_mask),
        ):
            acc, count = safe_accuracy(correct, subset_mask)
            class_rows.append({
                "class_id": int(c),
                "subset": subset_name,
                "true_accuracy": acc,
                "count": count,
            })

    return {
        "overall_true_accuracy": overall,
        "overall_count": overall_n,
        "clean_true_accuracy": clean_acc,
        "clean_count": clean_n,
        "noisy_true_accuracy": noisy_acc,
        "noisy_count": noisy_n,
        "class_rows": class_rows,
    }


def max_parameter_difference(model_a, model_b):
    maximum = 0.0
    with torch.no_grad():
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            diff = float(torch.max(torch.abs(pa - pb)).item())
            maximum = max(maximum, diff)
    return maximum


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    # Improves reproducibility. We do not force
    # torch.use_deterministic_algorithms(True), because some CUDA kernels in
    # user environments may not support it.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms()

    train_ds = NoisyCIFAR10WithIndex(
        root=str(args.data_root),
        train=True,
        transform=train_tf,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )
    eval_ds = NoisyCIFAR10WithIndex(
        root=str(args.data_root),
        train=True,
        transform=eval_tf,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )

    # Verify that training/evaluation views represent the same corruption.
    if not np.array_equal(train_ds.targets, eval_ds.targets):
        raise RuntimeError("Training/evaluation observed labels differ.")
    if not np.array_equal(train_ds.true_targets, eval_ds.true_targets):
        raise RuntimeError("Training/evaluation true labels differ.")
    if not np.array_equal(train_ds.is_anomaly, eval_ds.is_anomaly):
        raise RuntimeError("Training/evaluation noise masks differ.")

    observed_labels = np.asarray(train_ds.targets, dtype=np.int64)
    true_labels = np.asarray(train_ds.true_targets, dtype=np.int64)
    noisy_mask = np.asarray(train_ds.is_anomaly, dtype=bool)
    clean_mask = ~noisy_mask
    n_samples = len(train_ds)

    class_names = getattr(
        train_ds,
        "classes",
        [str(i) for i in range(args.num_classes)],
    )

    # One shared shuffled loader: same minibatches and augmented tensors.
    generator = torch.Generator().manual_seed(args.seed)
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        generator=generator,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # Initialize once, then deep-copy: exact same initial parameters/buffers.
    baseline_model = CNN12_Model(
        num_classes=args.num_classes
    ).to(device)

    adaptive_model = copy.deepcopy(baseline_model).to(device)

    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    adaptive_optimizer = torch.optim.AdamW(
        adaptive_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Adaptive detector state. Detection uses ONLY adaptive-model observed-label
    # loss trajectories; true labels are evaluation-only.
    adaptive_loss_traj = np.full(
        (n_samples, args.num_epochs),
        np.nan,
        dtype=np.float32,
    )
    alpha_traj = np.ones(
        (n_samples, args.num_epochs),
        dtype=np.float32,
    )
    rank_ewma_traj = np.full(
        (n_samples, args.num_epochs),
        np.nan,
        dtype=np.float32,
    )
    detected_traj = np.zeros(
        (n_samples, args.num_epochs),
        dtype=bool,
    )

    # Save true-label correctness for later plotting/analysis.
    baseline_true_correct_traj = np.zeros(
        (n_samples, args.num_epochs),
        dtype=bool,
    )
    adaptive_true_correct_traj = np.zeros(
        (n_samples, args.num_epochs),
        dtype=bool,
    )

    alpha_table = np.ones(n_samples, dtype=np.float32)
    rank_ewma_state = np.full(n_samples, np.nan, dtype=np.float32)

    # Frozen LE convention: K=40 gives first valid score at epoch 42.
    first_detector_epoch = args.K + 2

    epoch_rows = []
    class_rows = []

    print("=== PAIRED TRUE-LABEL TRAINING EXPERIMENT ===")
    print(f"device={device}")
    print(f"noise={args.noisy_frac}")
    print("baseline = ordinary CE on observed/noisy labels")
    print("adaptive = LE-GIE + rank-EWMA + D2L-style target")
    print("ALL REPORTED ACCURACY = accuracy against ORIGINAL TRUE labels")
    print(f"K={args.K}, q={args.q}, rank_ewma_lambda={args.rank_ewma_lambda}")
    print("LE-GIE limits: next for ell_err and next for GIE")
    print(f"first_detector_epoch={first_detector_epoch}")
    print(f"gamma_max={args.gamma_max}, alpha_floor={args.alpha_floor}")
    print()

    for epoch_idx in range(args.num_epochs):
        epoch = epoch_idx + 1

        # alpha used for this epoch was computed after previous epoch.
        alpha_traj[:, epoch_idx] = alpha_table

        train_stats = train_paired_one_epoch(
            baseline_model=baseline_model,
            adaptive_model=adaptive_model,
            loader=train_loader,
            baseline_optimizer=baseline_optimizer,
            adaptive_optimizer=adaptive_optimizer,
            device=device,
            alpha_table=alpha_table,
            num_classes=args.num_classes,
        )

        # Deterministic training-set evaluations.
        baseline_eval = evaluate_training_set_true_labels(
            model=baseline_model,
            loader=eval_loader,
            device=device,
            true_labels=true_labels,
            observed_labels=observed_labels,
            n_samples=n_samples,
        )
        adaptive_eval = evaluate_training_set_true_labels(
            model=adaptive_model,
            loader=eval_loader,
            device=device,
            true_labels=true_labels,
            observed_labels=observed_labels,
            n_samples=n_samples,
        )

        baseline_correct = baseline_eval["true_correct"]
        adaptive_correct = adaptive_eval["true_correct"]

        baseline_true_correct_traj[:, epoch_idx] = baseline_correct
        adaptive_true_correct_traj[:, epoch_idx] = adaptive_correct

        baseline_summary = true_accuracy_summary(
            baseline_correct,
            true_labels,
            noisy_mask,
            args.num_classes,
        )
        adaptive_summary = true_accuracy_summary(
            adaptive_correct,
            true_labels,
            noisy_mask,
            args.num_classes,
        )

        # Detector uses adaptive-model observed-label losses ONLY.
        adaptive_loss_traj[:, epoch_idx] = adaptive_eval["loss_observed"]

        detected = np.zeros(n_samples, dtype=bool)
        gamma = 0.0

        if epoch >= first_detector_epoch:
            le_score = latest_le_next_next(
                adaptive_loss_traj[:, :epoch],
                observed_labels,
                args.K,
                args.num_classes,
            )

            current_rank = within_class_percentile(
                le_score,
                observed_labels,
                args.num_classes,
            )

            rank_ewma_state = update_rank_ewma(
                rank_ewma_state,
                current_rank,
                args.rank_ewma_lambda,
            )

            detected = topq_mask(
                rank_ewma_state,
                args.q,
            )

            gamma = correction_strength(
                epoch,
                first_detector_epoch,
                args.num_epochs,
                args.gamma_max,
            )

            # Causal: update after epoch t; use during epoch t+1.
            alpha_table = alpha_from_rank_ewma(
                rank_ewma_state,
                detected,
                gamma,
                args.alpha_floor,
            )

        rank_ewma_traj[:, epoch_idx] = rank_ewma_state
        detected_traj[:, epoch_idx] = detected

        # Detector TPR/FPR are evaluation-only diagnostics.
        tp = int(np.sum(detected & noisy_mask))
        fp = int(np.sum(detected & clean_mask))
        n_noisy = int(np.sum(noisy_mask))
        n_clean = int(np.sum(clean_mask))
        detector_tpr = tp / max(n_noisy, 1)
        detector_fpr = fp / max(n_clean, 1)

        param_diff = max_parameter_difference(
            baseline_model,
            adaptive_model,
        )

        epoch_rows.append({
            "epoch": epoch,

            # Losses are optimization losses on observed/noisy labels/targets.
            "baseline_optimization_loss": train_stats["baseline_optimization_loss"],
            "adaptive_optimization_loss": train_stats["adaptive_optimization_loss"],

            # ALL accuracy columns below use TRUE labels.
            "baseline_true_accuracy_all": baseline_summary["overall_true_accuracy"],
            "adaptive_true_accuracy_all": adaptive_summary["overall_true_accuracy"],

            "baseline_true_accuracy_clean": baseline_summary["clean_true_accuracy"],
            "adaptive_true_accuracy_clean": adaptive_summary["clean_true_accuracy"],

            "baseline_true_accuracy_noisy": baseline_summary["noisy_true_accuracy"],
            "adaptive_true_accuracy_noisy": adaptive_summary["noisy_true_accuracy"],

            "true_accuracy_gain_all_adaptive_minus_baseline": (
                adaptive_summary["overall_true_accuracy"]
                - baseline_summary["overall_true_accuracy"]
            ),
            "true_accuracy_gain_noisy_adaptive_minus_baseline": (
                adaptive_summary["noisy_true_accuracy"]
                - baseline_summary["noisy_true_accuracy"]
            ),

            "detector_active": bool(epoch >= first_detector_epoch),
            "n_detected": int(np.sum(detected)),
            "detector_TPR_eval_only": float(detector_tpr),
            "detector_FPR_eval_only": float(detector_fpr),
            "gamma": float(gamma),

            # alpha_table is for NEXT epoch.
            "mean_alpha_next_epoch": float(np.mean(alpha_table)),
            "mean_alpha_noisy_next_epoch": float(np.mean(alpha_table[noisy_mask])),
            "mean_alpha_clean_next_epoch": float(np.mean(alpha_table[clean_mask])),
            "mean_alpha_detected_next_epoch": (
                float(np.mean(alpha_table[detected]))
                if np.any(detected)
                else np.nan
            ),

            # Fairness diagnostic: should stay ~0 before correction has effect.
            "max_parameter_abs_difference": float(param_diff),
        })

        # Long-format per-class TRUE-label accuracy.
        for model_name, summary in (
            ("baseline_no_detector", baseline_summary),
            ("adaptive_detector", adaptive_summary),
        ):
            for r in summary["class_rows"]:
                c = int(r["class_id"])
                class_rows.append({
                    "epoch": epoch,
                    "model": model_name,
                    "class_id": c,
                    "class_name": (
                        class_names[c]
                        if c < len(class_names)
                        else str(c)
                    ),
                    "subset": r["subset"],
                    "true_accuracy": r["true_accuracy"],
                    "count": r["count"],
                })

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == first_detector_epoch
            or epoch == args.num_epochs
        ):
            print(
                f"epoch={epoch:03d} "
                f"TRUE_ACC all: "
                f"base={baseline_summary['overall_true_accuracy']:.4f} "
                f"adapt={adaptive_summary['overall_true_accuracy']:.4f} | "
                f"TRUE_ACC noisy: "
                f"base={baseline_summary['noisy_true_accuracy']:.4f} "
                f"adapt={adaptive_summary['noisy_true_accuracy']:.4f} | "
                f"det={int(np.sum(detected)):5d} "
                f"TPR={detector_tpr:.4f} "
                f"FPR={detector_fpr:.4f} "
                f"gamma={gamma:.3f} "
                f"param_diff={param_diff:.3e}"
            )

    write_csv(
        out_dir / "epoch_true_accuracy_summary.csv",
        epoch_rows,
    )
    write_csv(
        out_dir / "per_class_true_accuracy.csv",
        class_rows,
    )

    np.savez_compressed(
        out_dir / "paired_true_label_trajectories.npz",
        sample_index=np.arange(n_samples, dtype=np.int64),
        epoch=np.arange(1, args.num_epochs + 1, dtype=np.int64),
        observed_label=observed_labels,
        true_label=true_labels,
        is_anomaly=noisy_mask,
        adaptive_loss_traj=adaptive_loss_traj,
        alpha_traj=alpha_traj,
        rank_ewma_traj=rank_ewma_traj,
        detected_traj=detected_traj,
        baseline_true_correct_traj=baseline_true_correct_traj,
        adaptive_true_correct_traj=adaptive_true_correct_traj,
    )

    config = {
        "artifact": "paired_baseline_vs_detector_true_label_accuracy",
        "paired_design": {
            "same_run": True,
            "same_initial_weights": True,
            "same_minibatches": True,
            "same_augmented_images": True,
            "synchronized_dropout_rng": True,
            "same_optimizer_hyperparameters": True,
        },
        "accuracy_definition":
            "ALL reported training accuracy is prediction vs original true CIFAR-10 label",
        "baseline_training_target":
            "observed/noisy label with ordinary cross-entropy",
        "adaptive_training_target":
            "y*=alpha*y_observed+(1-alpha)*stopgrad(model_probability)",
        "detector":
            "signed LE-GIE next/next -> within-class percentile -> rank-EWMA -> top-q",
        "causal_update":
            "detector after epoch t updates alpha used in epoch t+1",
        "K": args.K,
        "q": args.q,
        "rank_ewma_lambda": args.rank_ewma_lambda,
        "gamma_max": args.gamma_max,
        "alpha_floor": args.alpha_floor,
        "first_detector_epoch": first_detector_epoch,
        "noisy_frac": args.noisy_frac,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "evaluation_only_note":
            "true labels and true noise mask never affect detector or alpha",
    }

    with (out_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    if args.save_models:
        torch.save(
            {
                "model_state_dict": baseline_model.state_dict(),
                "optimizer_state_dict": baseline_optimizer.state_dict(),
                "config": config,
            },
            out_dir / "baseline_final.pt",
        )
        torch.save(
            {
                "model_state_dict": adaptive_model.state_dict(),
                "optimizer_state_dict": adaptive_optimizer.state_dict(),
                "config": config,
            },
            out_dir / "adaptive_final.pt",
        )

    print()
    print("Saved:")
    print(out_dir / "epoch_true_accuracy_summary.csv")
    print(out_dir / "per_class_true_accuracy.csv")
    print(out_dir / "paired_true_label_trajectories.npz")
    print(out_dir / "run_config.json")


if __name__ == "__main__":
    main()
