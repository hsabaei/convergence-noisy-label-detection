#!/usr/bin/env python
"""Causal detector-guided training inspired by D2L.

At epoch t:
  1) train with alpha values computed after epoch t-1;
  2) record deterministic per-sample losses;
  3) compute frozen LE-GIE(K=40, next/next);
  4) update within-class rank-EWMA;
  5) detect current top-q suspicious samples;
  6) compute alpha for epoch t+1.

Adaptive target:
    y* = alpha*y + (1-alpha)*stopgrad(p_theta)

The true noisy-label mask is used only for evaluation metrics, never by the detector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convergence_monitoring.data import NoisyCIFAR10WithIndex
from convergence_monitoring.models import CNN12_Model
from convergence_monitoring.training import evaluate_group_diagnostics, set_seed
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
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "adaptive_le_rank_ewma")
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

    # Frozen detector configuration from the detection study.
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--rank-ewma-lambda", type=float, default=0.05)

    # New training-control parameters only.
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--adaptive-off", action="store_true")
    p.add_argument("--save-model", action="store_true")

    args = p.parse_args()
    if not (0 <= args.noisy_frac < 1):
        p.error("--noisy-frac must be in [0,1).")
    if not (0 < args.q < 1):
        p.error("--q must be in (0,1).")
    if not (0 < args.rank_ewma_lambda <= 1):
        p.error("--rank-ewma-lambda must be in (0,1].")
    if not (0 <= args.alpha_floor <= 1):
        p.error("--alpha-floor must be in [0,1].")
    if args.gamma_max < 0:
        p.error("--gamma-max must be nonnegative.")
    return args


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def train_one_epoch(model, loader, optimizer, device, alpha_table, num_classes):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx_np = np.asarray(idx, dtype=np.int64)
        alpha = torch.from_numpy(alpha_table[idx_np]).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = adaptive_target_cross_entropy(logits, y, alpha, num_classes)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            n = int(y.numel())
            total_loss += float(loss.item()) * n
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += n

    return {
        "train_loss": total_loss / max(total, 1),
        "train_acc_observed": total_correct / max(total, 1),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms()
    train_ds = NoisyCIFAR10WithIndex(
        root=str(args.data_root), train=True, transform=train_tf,
        download=args.download, noisy_frac=args.noisy_frac,
        seed=args.seed, num_classes=args.num_classes,
    )
    eval_ds = NoisyCIFAR10WithIndex(
        root=str(args.data_root), train=True, transform=eval_tf,
        download=args.download, noisy_frac=args.noisy_frac,
        seed=args.seed, num_classes=args.num_classes,
    )

    if not np.array_equal(train_ds.targets, eval_ds.targets):
        raise RuntimeError("Training/evaluation corrupted labels differ.")
    if not np.array_equal(train_ds.is_anomaly, eval_ds.is_anomaly):
        raise RuntimeError("Training/evaluation noise masks differ.")

    labels = np.asarray(train_ds.targets, dtype=np.int64)
    noisy_mask = np.asarray(train_ds.is_anomaly, dtype=bool)
    clean_mask = ~noisy_mask
    n_samples = len(train_ds)

    generator = torch.Generator().manual_seed(args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, generator=generator,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    model = CNN12_Model(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dummy_subset_mask = torch.zeros(n_samples, dtype=torch.bool)

    loss_traj = np.full((n_samples, args.num_epochs), np.nan, dtype=np.float32)
    alpha_traj = np.ones((n_samples, args.num_epochs), dtype=np.float32)
    rank_ewma_traj = np.full((n_samples, args.num_epochs), np.nan, dtype=np.float32)
    detected_traj = np.zeros((n_samples, args.num_epochs), dtype=bool)

    alpha_table = np.ones(n_samples, dtype=np.float32)
    rank_ewma_state = np.full(n_samples, np.nan, dtype=np.float32)

    # Frozen LE convention: K=40 -> first valid score at epoch 42.
    first_detector_epoch = args.K + 2
    rows = []

    print("=== D2L-style detector-guided training ===")
    print(f"device={device}")
    print(f"noise={args.noisy_frac}")
    print(f"K={args.K}, q={args.q}, rank_ewma_lambda={args.rank_ewma_lambda}")
    print("LE-GIE limits: next for ell_err and next for GIE")
    print(f"first_detector_epoch={first_detector_epoch}")
    print(f"gamma_max={args.gamma_max}, alpha_floor={args.alpha_floor}")
    print(f"adaptive_off={args.adaptive_off}")

    for epoch_idx in range(args.num_epochs):
        epoch = epoch_idx + 1
        alpha_traj[:, epoch_idx] = alpha_table

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            alpha_table, args.num_classes,
        )

        diag = evaluate_group_diagnostics(
            model=model,
            loader=eval_loader,
            device=str(device),
            subset_mask_table=dummy_subset_mask,
            num_classes=args.num_classes,
        )
        loss_full = np.asarray(diag["loss_full"], dtype=np.float32)
        correct_full = np.asarray(diag["correct_full"], dtype=np.float32)
        loss_traj[:, epoch_idx] = loss_full

        detected = np.zeros(n_samples, dtype=bool)
        gamma = 0.0

        if (not args.adaptive_off) and epoch >= first_detector_epoch:
            le_score = latest_le_next_next(
                loss_traj[:, :epoch], labels, args.K, args.num_classes
            )
            current_rank = within_class_percentile(
                le_score, labels, args.num_classes
            )
            rank_ewma_state = update_rank_ewma(
                rank_ewma_state, current_rank, args.rank_ewma_lambda
            )
            detected = topq_mask(rank_ewma_state, args.q)
            gamma = correction_strength(
                epoch, first_detector_epoch, args.num_epochs, args.gamma_max
            )

            # Causal update: these alpha values are used in the NEXT epoch.
            alpha_table = alpha_from_rank_ewma(
                rank_ewma_state, detected, gamma, args.alpha_floor
            )
        elif args.adaptive_off:
            alpha_table.fill(1.0)

        rank_ewma_traj[:, epoch_idx] = rank_ewma_state
        detected_traj[:, epoch_idx] = detected

        n_detected = int(np.sum(detected))
        tp = int(np.sum(detected & noisy_mask))
        fp = int(np.sum(detected & clean_mask))
        tpr = tp / max(int(np.sum(noisy_mask)), 1)
        fpr = fp / max(int(np.sum(clean_mask)), 1)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_acc_observed": train_stats["train_acc_observed"],
            "eval_loss_all": float(np.mean(loss_full)),
            "eval_acc_observed": float(np.mean(correct_full)),
            "eval_loss_clean": float(np.mean(loss_full[clean_mask])),
            "eval_loss_noisy": float(np.mean(loss_full[noisy_mask])),
            "detector_active": bool((not args.adaptive_off) and epoch >= first_detector_epoch),
            "gamma": float(gamma),
            "n_detected": n_detected,
            "detected_fraction": n_detected / n_samples,
            "detector_TPR_eval_only": float(tpr),
            "detector_FPR_eval_only": float(fpr),
            "mean_alpha_next_epoch": float(np.mean(alpha_table)),
            "mean_alpha_detected_next_epoch": (
                float(np.mean(alpha_table[detected])) if n_detected else np.nan
            ),
        }
        rows.append(row)

        if epoch == 1 or epoch % 5 == 0 or epoch == first_detector_epoch or epoch == args.num_epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={row['train_loss']:.4f} "
                f"eval_acc={row['eval_acc_observed']:.4f} "
                f"det={n_detected:5d} "
                f"TPR={tpr:.4f} FPR={fpr:.4f} "
                f"gamma={gamma:.3f} "
                f"alpha_det={row['mean_alpha_detected_next_epoch']:.3f}"
            )

    write_csv(out_dir / "epoch_summary.csv", rows)
    np.savez_compressed(
        out_dir / "adaptive_training_trajectories.npz",
        sample_index=np.arange(n_samples, dtype=np.int64),
        epoch=np.arange(1, args.num_epochs + 1, dtype=np.int64),
        loss_traj=loss_traj,
        alpha_traj=alpha_traj,
        rank_ewma_traj=rank_ewma_traj,
        detected_traj=detected_traj,
        observed_label=labels,
        true_label=np.asarray(train_ds.true_targets, dtype=np.int64),
        is_anomaly=noisy_mask,
    )

    config = {
        "artifact": "detector_guided_adaptive_target_training",
        "adaptive_target": "y*=alpha*y+(1-alpha)*stopgrad(model_probability)",
        "causal_update": "epoch t detector updates alpha for epoch t+1",
        "score": "signed LE-GIE = ell_err + log(abs(m_GIE))",
        "LE_error_limit": "next",
        "LE_GIE_limit": "next",
        "K": args.K,
        "q": args.q,
        "rank_ewma_lambda": args.rank_ewma_lambda,
        "gamma_max": args.gamma_max,
        "alpha_floor": args.alpha_floor,
        "first_detector_epoch": first_detector_epoch,
        "noisy_frac": args.noisy_frac,
        "seed": args.seed,
        "adaptive_off": args.adaptive_off,
        "evaluation_only_note": "true noise mask is never used by detector or alpha",
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.save_model:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, out_dir / "final_model.pt")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
