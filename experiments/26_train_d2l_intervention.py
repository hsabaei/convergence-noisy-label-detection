#!/usr/bin/env python
"""
Paired D2L intervention experiment on the SAME CNN12/noisy-CIFAR10 setup
used for the CKL/LE intervention experiments.

This is a PyTorch implementation of the D2L intervention described in:
    Ma et al., "Dimensionality-Driven Learning with Noisy Labels", ICML 2018.

Purpose
-------
Compare:
    A) Baseline: ordinary CE on observed/noisy labels
    B) D2L: global LID-driven intervention

The network architecture, noisy labels, initialization, minibatches,
augmentations, optimizer hyperparameters, and dropout RNG are paired.

D2L logic
---------
1) Train normally at first.
2) After each epoch, estimate ONE mean LID value from the penultimate-layer
   representations of 10 random batches x 128 samples.
3) Detect the turning point when the current LID is more than 2 standard
   deviations above the mean of the previous 5 LID values, after a CIFAR-10
   initialization period of 40 epochs.
4) At the first turning point, roll the D2L model weights back to the model
   state from the previous epoch.
5) After the turning point, use one GLOBAL alpha for every training sample:

       y* = alpha * y_observed + (1-alpha) * y_hat

   where y_hat is the detached HARD predicted one-hot label.

6) The D2L alpha is

       alpha_t = exp( - lambda_t * expansion_t )

       lambda_t = t / T
       expansion_t = LID_t / min_{j < t} LID_j

Important
---------
* D2L does NOT flag individual noisy samples.
* The same alpha is used for every sample in a given epoch.
* True labels and the synthetic-noise mask are evaluation-only.
* Train accuracy is evaluated against the ORIGINAL true CIFAR-10 labels.
* Clean CIFAR-10 test accuracy is recorded every epoch.

Fair-comparison choice
----------------------
The original paper used SGD and 120 epochs for CIFAR-10. Here we deliberately
keep the SAME CNN12 + AdamW + 200-epoch setup as the CKL/LE experiments so
that only the intervention mechanism changes.

CIFAR-10 D2L monitoring defaults
--------------------------------
The authors' released implementation uses:
    init_epoch = 40
    epoch_win  = 5
    lid_subset_size = 1280
    lid_k = 20

We use the same monitoring defaults here.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convergence_monitoring.data import NoisyCIFAR10WithIndex
from convergence_monitoring.models import CNN12_Model
from convergence_monitoring.training import set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-12


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mmfs1/home/hs833/data"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "d2l_intervention",
    )
    p.add_argument("--download", action="store_true")

    p.add_argument("--noisy-frac", type=float, default=0.05)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    # D2L monitoring settings.
    p.add_argument("--lid-k", type=int, default=20)
    p.add_argument("--lid-batch-size", type=int, default=128)
    p.add_argument("--lid-num-batches", type=int, default=10)
    p.add_argument("--turning-init-epoch", type=int, default=40)
    p.add_argument("--lid-window", type=int, default=5)
    p.add_argument("--turning-z", type=float, default=2.0)

    p.add_argument("--save-models", action="store_true")

    args = p.parse_args()

    if not (0.0 <= args.noisy_frac < 1.0):
        p.error("--noisy-frac must be in [0,1).")
    if args.lid_k < 2:
        p.error("--lid-k must be >= 2.")
    if args.lid_batch_size <= args.lid_k:
        p.error("--lid-batch-size must be larger than --lid-k.")
    if args.lid_num_batches < 1:
        p.error("--lid-num-batches must be >= 1.")
    if args.lid_window < 2:
        p.error("--lid-window must be >= 2.")
    if args.turning_init_epoch < 0:
        p.error("--turning-init-epoch must be nonnegative.")
    if args.turning_z <= 0:
        p.error("--turning-z must be positive.")

    return args


# ---------------------------------------------------------------------
# Reproducibility / data
# ---------------------------------------------------------------------

def resolve_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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


def clone_model_state_cpu(model):
    """Deep CPU copy of model weights for rollback."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


# ---------------------------------------------------------------------
# D2L target loss
# ---------------------------------------------------------------------

def d2l_mixed_target_cross_entropy(
    logits,
    y_observed,
    alpha,
    num_classes,
):
    """Original D2L-style hard predicted label mixture.

    y* = alpha*y_observed + (1-alpha)*y_hat

    y_hat is detached and one-hot.
    alpha is global for the epoch.
    """
    if alpha >= 1.0 - 1e-12:
        return F.cross_entropy(
            logits,
            y_observed,
        )

    y_raw = F.one_hot(
        y_observed,
        num_classes=num_classes,
    ).to(dtype=logits.dtype)

    pred_class = logits.detach().argmax(
        dim=1
    )
    y_hat = F.one_hot(
        pred_class,
        num_classes=num_classes,
    ).to(dtype=logits.dtype)

    y_star = (
        float(alpha) * y_raw
        + (1.0 - float(alpha)) * y_hat
    )

    return -(
        y_star
        * F.log_softmax(logits, dim=1)
    ).sum(dim=1).mean()


# ---------------------------------------------------------------------
# Penultimate feature extraction
# ---------------------------------------------------------------------

class PenultimateFeatureHook:
    """Capture the input to the final nn.Linear layer.

    This is the representation immediately before the classifier.
    """

    def __init__(self, model):
        linear_modules = [
            m for m in model.modules()
            if isinstance(m, nn.Linear)
        ]

        if not linear_modules:
            raise RuntimeError(
                "CNN12_Model contains no nn.Linear layer; "
                "cannot identify classifier input."
            )

        self.final_linear = linear_modules[-1]
        self.features = None

        self.handle = self.final_linear.register_forward_pre_hook(
            self._hook
        )

    def _hook(self, module, inputs):
        if not inputs:
            raise RuntimeError(
                "Final linear layer received no input."
            )

        feat = inputs[0]

        if feat.ndim > 2:
            feat = torch.flatten(
                feat,
                start_dim=1,
            )

        self.features = feat.detach()

    def close(self):
        self.handle.remove()


# ---------------------------------------------------------------------
# LID
# ---------------------------------------------------------------------

def lid_mle_from_features(features, k):
    """Per-sample LID MLE using k nearest OTHER samples.

    For each x:
        LID(x) = - [ (1/k) sum_i log(r_i / r_k) ]^{-1}

    r_k is the largest of the k nearest-neighbor distances.
    """
    features = features.float()

    n = features.shape[0]
    if n <= k:
        raise ValueError(
            f"Need batch size > k, got n={n}, k={k}."
        )

    # Pairwise Euclidean distance in representation space.
    dist = torch.cdist(
        features,
        features,
        p=2,
    )

    # Numerical self-distance may be tiny rather than exactly zero.
    diag = torch.arange(
        n,
        device=features.device,
    )
    dist[diag, diag] = float("inf")

    knn, _ = torch.topk(
        dist,
        k=k,
        dim=1,
        largest=False,
        sorted=True,
    )

    knn = torch.clamp(
        knn,
        min=EPS,
    )

    r_max = knn[:, -1:]
    log_ratio = torch.log(
        knn / r_max
    )

    denom = torch.mean(
        log_ratio,
        dim=1,
    )

    lid = -1.0 / denom

    valid = (
        torch.isfinite(lid)
        & (lid > 0)
    )

    return lid[valid]


@torch.no_grad()
def estimate_mean_lid(
    model,
    eval_dataset,
    device,
    rng,
    *,
    k,
    batch_size,
    num_batches,
    num_workers,
):
    """Paper-style LID estimate from m random batches.

    We sample m*batch_size distinct examples, split them into m batches,
    compute LID within each batch, and average all valid sample LIDs.
    """
    model.eval()

    n_needed = (
        int(batch_size)
        * int(num_batches)
    )

    if n_needed > len(eval_dataset):
        raise ValueError(
            f"LID subset needs {n_needed} samples but dataset "
            f"contains {len(eval_dataset)}."
        )

    subset_idx = rng.choice(
        len(eval_dataset),
        size=n_needed,
        replace=False,
    )

    hook = PenultimateFeatureHook(
        model
    )

    all_lids = []

    try:
        for b in range(num_batches):
            ids = subset_idx[
                b * batch_size:
                (b + 1) * batch_size
            ]

            # Fetch deterministic eval-transformed samples.
            xs = []
            for idx in ids:
                x, _y, _idx = eval_dataset[
                    int(idx)
                ]
                xs.append(x)

            x = torch.stack(
                xs,
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            hook.features = None
            _ = model(x)

            if hook.features is None:
                raise RuntimeError(
                    "Penultimate feature hook did not fire."
                )

            lids = lid_mle_from_features(
                hook.features,
                k=k,
            )

            if lids.numel():
                all_lids.append(
                    lids.detach().cpu()
                )

    finally:
        hook.close()

    if not all_lids:
        return float("nan")

    all_lids = torch.cat(
        all_lids,
        dim=0,
    )

    return float(
        all_lids.mean().item()
    )


# ---------------------------------------------------------------------
# Turning-point / alpha logic
# ---------------------------------------------------------------------

def turning_point_now(
    lids,
    epoch,
    init_epoch,
    window,
    z_threshold,
):
    """Return detection info for the FIRST turning point only.

    Current LID is compared with the immediately preceding `window` LIDs.
    """
    if epoch <= init_epoch:
        return False, np.nan, np.nan

    if len(lids) < window + 1:
        return False, np.nan, np.nan

    previous = np.asarray(
        lids[-window-1:-1],
        dtype=np.float64,
    )
    current = float(
        lids[-1]
    )

    if (
        not np.isfinite(current)
        or not np.all(
            np.isfinite(previous)
        )
    ):
        return False, np.nan, np.nan

    mean = float(
        np.mean(previous)
    )
    std = float(
        np.std(previous)
    )

    threshold = (
        mean
        + float(z_threshold)
        * std
    )

    return (
        current > threshold,
        mean,
        std,
    )


def d2l_alpha(
    current_lid,
    previous_lids,
    epoch,
    total_epochs,
):
    """Paper D2L alpha after the turning point."""
    prev = np.asarray(
        previous_lids,
        dtype=np.float64,
    )

    prev = prev[
        np.isfinite(prev)
        & (prev > 0)
    ]

    if (
        len(prev) == 0
        or not np.isfinite(current_lid)
        or current_lid <= 0
    ):
        return 1.0, np.nan, np.nan

    min_previous = float(
        np.min(prev)
    )

    expansion = (
        float(current_lid)
        / min_previous
    )

    # Paper notation i/T. Our logged epochs are 1..T.
    pace = float(epoch) / float(
        total_epochs
    )

    alpha = float(
        np.exp(
            -pace * expansion
        )
    )

    # Mathematical range is already (0,1], clamp only for numerical safety.
    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    return (
        alpha,
        pace,
        expansion,
    )


# ---------------------------------------------------------------------
# Paired training / evaluation
# ---------------------------------------------------------------------

def train_paired_one_epoch(
    baseline_model,
    d2l_model,
    loader,
    baseline_optimizer,
    d2l_optimizer,
    device,
    d2l_alpha_value,
    num_classes,
):
    """Same minibatch + augmentation, synchronized dropout RNG."""
    baseline_model.train()
    d2l_model.train()

    baseline_loss_sum = 0.0
    d2l_loss_sum = 0.0
    total = 0

    for x, y_observed, _idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y_observed = y_observed.to(
            device,
            non_blocking=True,
        )

        baseline_optimizer.zero_grad(
            set_to_none=True
        )
        d2l_optimizer.zero_grad(
            set_to_none=True
        )

        paired_rng = capture_torch_rng(
            device
        )

        baseline_logits = baseline_model(
            x
        )

        restore_torch_rng(
            paired_rng,
            device,
        )

        d2l_logits = d2l_model(
            x
        )

        baseline_loss = F.cross_entropy(
            baseline_logits,
            y_observed,
        )

        d2l_loss = d2l_mixed_target_cross_entropy(
            d2l_logits,
            y_observed,
            d2l_alpha_value,
            num_classes,
        )

        baseline_loss.backward()
        d2l_loss.backward()

        baseline_optimizer.step()
        d2l_optimizer.step()

        n = int(
            y_observed.numel()
        )

        baseline_loss_sum += (
            float(
                baseline_loss.item()
            )
            * n
        )
        d2l_loss_sum += (
            float(
                d2l_loss.item()
            )
            * n
        )
        total += n

    return {
        "baseline_optimization_loss":
            baseline_loss_sum
            / max(total, 1),
        "d2l_optimization_loss":
            d2l_loss_sum
            / max(total, 1),
    }


@torch.no_grad()
def evaluate_training_set(
    model,
    loader,
    device,
    true_labels,
    n_samples,
):
    """Training accuracy against ORIGINAL true labels.

    True labels are used only here, never for training or D2L decisions.
    """
    model.eval()

    prediction = np.full(
        n_samples,
        -1,
        dtype=np.int64,
    )

    true_correct = np.zeros(
        n_samples,
        dtype=bool,
    )

    for x, _y_observed, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        idx_np = np.asarray(
            idx,
            dtype=np.int64,
        )

        logits = model(x)

        pred = (
            logits.argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        prediction[
            idx_np
        ] = pred

        true_correct[
            idx_np
        ] = (
            pred
            == true_labels[
                idx_np
            ]
        )

    if np.any(
        prediction < 0
    ):
        raise RuntimeError(
            "Evaluation did not cover every training sample."
        )

    return {
        "prediction":
            prediction,
        "true_correct":
            true_correct,
    }


@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits,
            y,
            reduction="sum",
        )

        pred = logits.argmax(
            dim=1
        )

        correct += int(
            (pred == y)
            .sum()
            .item()
        )
        total += int(
            y.numel()
        )
        loss_sum += float(
            loss.item()
        )

    return {
        "test_accuracy":
            correct
            / max(total, 1),
        "test_loss":
            loss_sum
            / max(total, 1),
    }


def masked_accuracy(
    correct,
    mask,
):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if int(
        np.sum(mask)
    ) == 0:
        return np.nan

    return float(
        np.mean(
            np.asarray(
                correct,
                dtype=bool,
            )[mask]
        )
    )


def max_parameter_difference(
    model_a,
    model_b,
):
    maximum = 0.0

    with torch.no_grad():
        for pa, pb in zip(
            model_a.parameters(),
            model_b.parameters(),
        ):
            diff = float(
                torch.max(
                    torch.abs(
                        pa - pb
                    )
                ).item()
            )
            maximum = max(
                maximum,
                diff,
            )

    return maximum


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    keys = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    set_seed(
        args.seed
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_tf, eval_tf = build_transforms()

    train_ds = NoisyCIFAR10WithIndex(
        root=str(
            args.data_root
        ),
        train=True,
        transform=train_tf,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )

    eval_ds = NoisyCIFAR10WithIndex(
        root=str(
            args.data_root
        ),
        train=True,
        transform=eval_tf,
        download=args.download,
        noisy_frac=args.noisy_frac,
        seed=args.seed,
        num_classes=args.num_classes,
    )

    test_ds = CIFAR10(
        root=str(
            args.data_root
        ),
        train=False,
        transform=eval_tf,
        download=args.download,
    )

    if not np.array_equal(
        train_ds.targets,
        eval_ds.targets,
    ):
        raise RuntimeError(
            "Observed labels differ between train/eval datasets."
        )

    if not np.array_equal(
        train_ds.true_targets,
        eval_ds.true_targets,
    ):
        raise RuntimeError(
            "True labels differ between train/eval datasets."
        )

    if not np.array_equal(
        train_ds.is_anomaly,
        eval_ds.is_anomaly,
    ):
        raise RuntimeError(
            "Noise masks differ between train/eval datasets."
        )

    true_labels = np.asarray(
        train_ds.true_targets,
        dtype=np.int64,
    )
    noisy_mask = np.asarray(
        train_ds.is_anomaly,
        dtype=bool,
    )
    clean_mask = ~noisy_mask

    n_samples = len(
        train_ds
    )

    generator = torch.Generator().manual_seed(
        args.seed
    )

    pin = (
        device.type == "cuda"
    )

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

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    baseline_model = CNN12_Model(
        num_classes=args.num_classes
    ).to(
        device
    )

    d2l_model = copy.deepcopy(
        baseline_model
    ).to(
        device
    )

    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    d2l_optimizer = torch.optim.AdamW(
        d2l_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Independent RNG for selecting LID subsets so it does not perturb
    # training minibatch shuffling or dropout RNG.
    lid_rng = np.random.default_rng(
        args.seed + 2026
    )

    lids = []
    turning_epoch = -1

    # Alpha used to TRAIN the current epoch.
    alpha_current = 1.0

    epoch_rows = []

    print("=" * 78)
    print("PAIRED D2L INTERVENTION")
    print("=" * 78)
    print(f"device={device}")
    print(f"noise={args.noisy_frac:.3f}")
    print(f"epochs={args.num_epochs}")
    print(f"lid_k={args.lid_k}")
    print(
        f"lid sampling="
        f"{args.lid_num_batches} x {args.lid_batch_size}"
    )
    print(
        f"turning init epoch={args.turning_init_epoch}"
    )
    print(
        f"turning window={args.lid_window}"
    )
    print(
        f"turning z={args.turning_z}"
    )
    print(
        "D2L target = alpha*y_observed "
        "+ (1-alpha)*hard_predicted_onehot"
    )
    print(
        "D2L alpha is GLOBAL for all samples"
    )
    print(
        "true labels/noise mask are evaluation-only"
    )
    print()

    for epoch_idx in range(
        args.num_epochs
    ):
        epoch = (
            epoch_idx + 1
        )

        # Save the PRE-EPOCH D2L model state.
        # If the turning point is detected after this epoch, restore this.
        d2l_pre_epoch_state = clone_model_state_cpu(
            d2l_model
        )

        alpha_used = float(
            alpha_current
        )

        train_stats = train_paired_one_epoch(
            baseline_model=baseline_model,
            d2l_model=d2l_model,
            loader=train_loader,
            baseline_optimizer=baseline_optimizer,
            d2l_optimizer=d2l_optimizer,
            device=device,
            d2l_alpha_value=alpha_used,
            num_classes=args.num_classes,
        )

        # -------------------------------------------------------------
        # D2L monitoring after training the epoch
        # -------------------------------------------------------------
        mean_lid = estimate_mean_lid(
            model=d2l_model,
            eval_dataset=eval_ds,
            device=device,
            rng=lid_rng,
            k=args.lid_k,
            batch_size=args.lid_batch_size,
            num_batches=args.lid_num_batches,
            num_workers=args.num_workers,
        )

        if (
            not np.isfinite(mean_lid)
            or mean_lid <= 0
        ):
            if lids:
                mean_lid = float(
                    lids[-1]
                )
            else:
                raise RuntimeError(
                    "First LID estimate is invalid."
                )

        lids.append(
            float(mean_lid)
        )

        turning_detected_now = False
        previous_window_mean = np.nan
        previous_window_std = np.nan
        rolled_back = False

        if turning_epoch < 0:
            (
                turning_detected_now,
                previous_window_mean,
                previous_window_std,
            ) = turning_point_now(
                lids=lids,
                epoch=epoch,
                init_epoch=args.turning_init_epoch,
                window=args.lid_window,
                z_threshold=args.turning_z,
            )

            if turning_detected_now:
                # Paper: u <- i-1; rollback h(x) to previous epoch.
                turning_epoch = max(
                    epoch - 1,
                    0,
                )

                d2l_model.load_state_dict(
                    d2l_pre_epoch_state
                )

                rolled_back = True

        # -------------------------------------------------------------
        # Compute alpha for NEXT epoch.
        # Once turning point exists, update every epoch.
        # -------------------------------------------------------------
        alpha_next = 1.0
        pace_lambda = float(
            epoch
        ) / float(
            args.num_epochs
        )
        expansion = np.nan

        if turning_epoch >= 0:
            # min over previous epochs; exclude the current value.
            previous_lids = lids[:-1]

            alpha_next, pace_lambda, expansion = d2l_alpha(
                current_lid=float(
                    mean_lid
                ),
                previous_lids=previous_lids,
                epoch=epoch,
                total_epochs=args.num_epochs,
            )

        alpha_current = float(
            alpha_next
        )

        # -------------------------------------------------------------
        # Evaluate the ACTUAL state that moves to the next epoch.
        # This is after rollback if a turning point was found.
        # -------------------------------------------------------------
        baseline_train_eval = evaluate_training_set(
            baseline_model,
            eval_loader,
            device,
            true_labels,
            n_samples,
        )

        d2l_train_eval = evaluate_training_set(
            d2l_model,
            eval_loader,
            device,
            true_labels,
            n_samples,
        )

        baseline_test_eval = evaluate_test(
            baseline_model,
            test_loader,
            device,
        )

        d2l_test_eval = evaluate_test(
            d2l_model,
            test_loader,
            device,
        )

        baseline_train_true_all = float(
            np.mean(
                baseline_train_eval[
                    "true_correct"
                ]
            )
        )

        d2l_train_true_all = float(
            np.mean(
                d2l_train_eval[
                    "true_correct"
                ]
            )
        )

        baseline_train_true_clean = masked_accuracy(
            baseline_train_eval[
                "true_correct"
            ],
            clean_mask,
        )

        d2l_train_true_clean = masked_accuracy(
            d2l_train_eval[
                "true_correct"
            ],
            clean_mask,
        )

        baseline_train_true_noisy = masked_accuracy(
            baseline_train_eval[
                "true_correct"
            ],
            noisy_mask,
        )

        d2l_train_true_noisy = masked_accuracy(
            d2l_train_eval[
                "true_correct"
            ],
            noisy_mask,
        )

        param_diff = max_parameter_difference(
            baseline_model,
            d2l_model,
        )

        row = {
            "epoch":
                epoch,

            "baseline_optimization_loss":
                train_stats[
                    "baseline_optimization_loss"
                ],
            "d2l_optimization_loss":
                train_stats[
                    "d2l_optimization_loss"
                ],

            # Training accuracy against ORIGINAL true labels.
            "baseline_train_true_accuracy_all":
                baseline_train_true_all,
            "train_true_accuracy_all":
                d2l_train_true_all,
            "d2l_train_true_accuracy_all":
                d2l_train_true_all,
            "train_true_accuracy_gain_all":
                d2l_train_true_all
                - baseline_train_true_all,

            "baseline_train_true_accuracy_clean":
                baseline_train_true_clean,
            "d2l_train_true_accuracy_clean":
                d2l_train_true_clean,

            "baseline_train_true_accuracy_noisy":
                baseline_train_true_noisy,
            "d2l_train_true_accuracy_noisy":
                d2l_train_true_noisy,
            "train_true_accuracy_gain_noisy":
                d2l_train_true_noisy
                - baseline_train_true_noisy,

            # Clean test set.
            "baseline_test_accuracy":
                baseline_test_eval[
                    "test_accuracy"
                ],
            "test_accuracy":
                d2l_test_eval[
                    "test_accuracy"
                ],
            "d2l_test_accuracy":
                d2l_test_eval[
                    "test_accuracy"
                ],
            "test_accuracy_gain":
                d2l_test_eval[
                    "test_accuracy"
                ]
                - baseline_test_eval[
                    "test_accuracy"
                ],
            "baseline_test_loss":
                baseline_test_eval[
                    "test_loss"
                ],
            "d2l_test_loss":
                d2l_test_eval[
                    "test_loss"
                ],

            # D2L state.
            "mean_lid":
                float(mean_lid),
            "previous_lid_window_mean":
                float(previous_window_mean),
            "previous_lid_window_std":
                float(previous_window_std),
            "turning_detected_now":
                bool(
                    turning_detected_now
                ),
            "turning_epoch":
                int(turning_epoch),
            "rolled_back_now":
                bool(
                    rolled_back
                ),
            "alpha_used_this_epoch":
                float(alpha_used),
            "alpha_next_epoch":
                float(alpha_next),
            "lambda":
                float(pace_lambda),
            "lid_expansion":
                float(expansion),
            "max_parameter_abs_difference":
                float(param_diff),
        }

        epoch_rows.append(
            row
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or turning_detected_now
            or epoch == args.num_epochs
        ):
            print(
                f"epoch={epoch:03d} | "
                f"LID={mean_lid:.4f} "
                f"turn={turning_epoch:3d} "
                f"rollback={int(rolled_back)} "
                f"alpha_used={alpha_used:.4f} "
                f"alpha_next={alpha_next:.4f} | "
                f"test base="
                f"{baseline_test_eval['test_accuracy']:.4f} "
                f"d2l={d2l_test_eval['test_accuracy']:.4f} "
                f"gain="
                f"{row['test_accuracy_gain']:+.4f} | "
                f"train-true noisy base="
                f"{baseline_train_true_noisy:.4f} "
                f"d2l={d2l_train_true_noisy:.4f}"
            )

    write_csv(
        out_dir
        / "epoch_summary.csv",
        epoch_rows,
    )

    np.savez_compressed(
        out_dir
        / "d2l_trajectories.npz",
        epoch=np.arange(
            1,
            args.num_epochs + 1,
            dtype=np.int64,
        ),
        mean_lid=np.asarray(
            lids,
            dtype=np.float32,
        ),
        true_label=true_labels,
        is_anomaly=noisy_mask,
    )

    config = {
        "artifact":
            "paired_d2l_intervention_same_training_setup",

        "method":
            "Dimensionality-Driven Learning (D2L)",

        "paper_rule": {
            "lid_layer":
                "input to final Linear layer (penultimate representation)",
            "lid_estimator":
                "MLE using k nearest neighbors in each sampled batch",
            "global_epoch_lid":
                "mean of sample LIDs over random batches",
            "turning_rule":
                "current LID > mean(previous window) + 2*std(previous window)",
            "rollback":
                "model weights restored to state immediately before triggering epoch",
            "target":
                "alpha*y_observed + (1-alpha)*hard_predicted_onehot",
            "alpha":
                "exp(-(epoch/T)*(current_LID/min_previous_LID))",
            "global_alpha":
                True,
        },

        "cifar10_d2l_monitoring": {
            "lid_k":
                args.lid_k,
            "lid_batch_size":
                args.lid_batch_size,
            "lid_num_batches":
                args.lid_num_batches,
            "lid_subset_size":
                args.lid_batch_size
                * args.lid_num_batches,
            "turning_init_epoch":
                args.turning_init_epoch,
            "lid_window":
                args.lid_window,
            "turning_z":
                args.turning_z,
        },

        "fair_comparison_training": {
            "note":
                "D2L intervention is evaluated under the same training setup as CKL/LE, not the paper's SGD/120-epoch recipe.",
            "same_initial_weights":
                True,
            "same_minibatches":
                True,
            "same_augmented_images":
                True,
            "synchronized_dropout_rng":
                True,
            "optimizer":
                "AdamW",
            "lr":
                args.lr,
            "weight_decay":
                args.weight_decay,
            "epochs":
                args.num_epochs,
            "batch_size":
                args.batch_size,
            "seed":
                args.seed,
            "noisy_frac":
                args.noisy_frac,
        },

        "evaluation": {
            "true_labels_and_noise_mask":
                "evaluation-only",
            "training_accuracy":
                "prediction vs original true CIFAR-10 label",
            "test_accuracy":
                "clean CIFAR-10 test labels",
        },
    }

    with (
        out_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    if args.save_models:
        torch.save(
            baseline_model.state_dict(),
            out_dir
            / "baseline_model.pt",
        )
        torch.save(
            d2l_model.state_dict(),
            out_dir
            / "d2l_model.pt",
        )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(
        out_dir
        / "epoch_summary.csv"
    )
    print(
        out_dir
        / "d2l_trajectories.npz"
    )
    print(
        out_dir
        / "config.json"
    )


if __name__ == "__main__":
    main()
