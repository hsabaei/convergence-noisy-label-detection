import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_subset_mask(n: int, frac: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = max(1, int(round(frac * n)))
    subset_idx = perm[:k]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[subset_idx] = True
    return mask


def make_frozen_mask(n: int, frac: float, seed: int) -> torch.Tensor:
    return make_subset_mask(n, frac, seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


evaluate_accuracy = evaluate


@torch.no_grad()
def evaluate_group_diagnostics(
    model: nn.Module,
    loader,
    device: str,
    subset_mask_table: torch.Tensor,
    num_classes: int = 10,
):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    N = len(loader.dataset)

    loss_full = np.full(N, np.nan, dtype=np.float64)
    correct_full = np.full(N, np.nan, dtype=np.float64)
    label_full = np.full(N, -1, dtype=np.int64)

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        per_loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        idx_np = idx.numpy()
        loss_full[idx_np] = per_loss.cpu().numpy()
        correct_full[idx_np] = (pred == y).cpu().numpy().astype(np.float64)
        label_full[idx_np] = y.cpu().numpy()

    subset_np = subset_mask_table.numpy()
    other_np = ~subset_np

    subset_loss = float(np.mean(loss_full[subset_np])) if np.any(subset_np) else np.nan
    other_loss = float(np.mean(loss_full[other_np])) if np.any(other_np) else np.nan

    subset_acc = float(np.mean(correct_full[subset_np])) if np.any(subset_np) else np.nan
    other_acc = float(np.mean(correct_full[other_np])) if np.any(other_np) else np.nan

    subset_loss_by_class = np.full(num_classes, np.nan, dtype=np.float64)
    other_loss_by_class = np.full(num_classes, np.nan, dtype=np.float64)
    subset_acc_by_class = np.full(num_classes, np.nan, dtype=np.float64)
    other_acc_by_class = np.full(num_classes, np.nan, dtype=np.float64)

    for c in range(num_classes):
        mask_c_subset = subset_np & (label_full == c)
        mask_c_other = other_np & (label_full == c)

        if np.any(mask_c_subset):
            subset_loss_by_class[c] = float(np.mean(loss_full[mask_c_subset]))
            subset_acc_by_class[c] = float(np.mean(correct_full[mask_c_subset]))

        if np.any(mask_c_other):
            other_loss_by_class[c] = float(np.mean(loss_full[mask_c_other]))
            other_acc_by_class[c] = float(np.mean(correct_full[mask_c_other]))

    return {
        "loss_full": loss_full,
        "correct_full": correct_full,
        "label_full": label_full,
        "subset_loss": subset_loss,
        "other_loss": other_loss,
        "subset_acc": subset_acc,
        "other_acc": other_acc,
        "subset_loss_by_class": subset_loss_by_class,
        "other_loss_by_class": other_loss_by_class,
        "subset_acc_by_class": subset_acc_by_class,
        "other_acc_by_class": other_acc_by_class,
    }
