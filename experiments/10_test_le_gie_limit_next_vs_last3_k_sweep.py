"""
Focused sensitivity study for the GIE limit estimate inside LE.

Keep fixed:
        ell_err limit = next sample
    class reference = observed-class mean

Change ONLY the GIE limit rule:
    A) last-three mean
    B) next sample

The theoretical estimator remains:
    ell_hat = ell_err_hat + log(abs(m_GIE_hat))

No production estimator/detector module is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence_monitoring.detectors import (
    binary_auc_from_scores,
    exact_pairwise_scores_by_class,
)
from convergence_monitoring.framework import (
    standardize_monitoring_score,
)


EPS = 1e-7


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--common-npz",
        type=Path,
        default=Path(
            "results/common_loss_trajectories/"
            "cifar10_noisy_label_loss_trajectories.npz"
        ),
    )
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument(
        "--K-values",
        type=int,
        nargs="+",
        default=(10, 15, 20, 30, 40),
        help="K values to test.",
    )
    p.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
    )
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/le_gie_limit_k_sweep"
        ),
    )

    args = p.parse_args()

    for q in args.top_fractions:
        if not 0.0 < q < 1.0:
            p.error("--top-fractions must lie in (0,1).")

    if not 0.0 < args.target_fpr < 1.0:
        p.error("--target-fpr must lie in (0,1).")

    return args


def build_class_mean_reference(loss, labels, num_classes):
    """Return compact observed-class mean trajectories with shape C x T."""
    x = np.asarray(loss)
    labels = np.asarray(labels, dtype=np.int64)

    T = x.shape[1]
    Gc = np.full((int(num_classes), T), np.nan, dtype=np.float32)

    for c in range(int(num_classes)):
        members = np.flatnonzero(labels == c)
        if members.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            Gc[c] = np.nanmean(
                x[members],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

    return Gc

def last3_limit(traj, t):
    return np.mean(
        traj[:, t - 2:t + 1],
        axis=1,
    )


def next_limit(traj, t):
    return traj[:, t].copy()


def vectorized_gie_window_with_limits(
    phi,
    G,
    phi_limit,
    G_limit,
    batch_size=2000,
):
    """Row-wise GIE/Hill formula evaluated in sample batches.

    Batching preserves the estimator exactly while avoiding simultaneous
    N x K temporary arrays for R, FR, mask, log_num, and log_den.
    """
    phi = np.asarray(phi)
    G = np.asarray(G)
    phi_limit = np.asarray(phi_limit)
    G_limit = np.asarray(G_limit)

    N = phi.shape[0]
    d = np.full(N, np.nan, dtype=np.float64)

    for lo in range(0, N, int(batch_size)):
        hi = min(lo + int(batch_size), N)

        p = np.asarray(phi[lo:hi], dtype=np.float64)
        g = np.asarray(G[lo:hi], dtype=np.float64)
        pl = np.asarray(phi_limit[lo:hi], dtype=np.float64)
        gl = np.asarray(G_limit[lo:hi], dtype=np.float64)

        R = np.abs(p - pl[:, None])
        FR = np.abs(g - gl[:, None])

        with np.errstate(invalid="ignore"):
            w0 = np.max(R, axis=1)
            w1 = np.max(FR, axis=1)

        mask = (
            (R > EPS)
            & (FR > EPS)
            & np.isfinite(R)
            & np.isfinite(FR)
        )

        n_nonzero = mask.sum(axis=1)
        k_internal = n_nonzero - 1

        base_ok = (
            (k_internal > 4)
            & np.isfinite(w0)
            & np.isfinite(w1)
            & (w0 > 0.0)
            & (w1 > 0.0)
        )

        safe_w0 = np.where(base_ok, w0 + EPS, 1.0)
        safe_w1 = np.where(base_ok, w1 + EPS, 1.0)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            log_num = np.where(
                mask,
                np.log(np.abs(R / safe_w0[:, None])),
                0.0,
            )
            log_den = np.where(
                mask,
                np.log(np.abs(FR / safe_w1[:, None])),
                0.0,
            )

        denom_num = np.sum(log_num, axis=1)
        denom_den = np.sum(log_den, axis=1)

        ok = (
            base_ok
            & np.isfinite(denom_num)
            & np.isfinite(denom_den)
            & (np.abs(denom_num) >= EPS)
            & (np.abs(denom_den) >= EPS)
        )

        hill_num = np.full(hi - lo, np.nan, dtype=np.float64)
        hill_den = np.full(hi - lo, np.nan, dtype=np.float64)

        hill_num[ok] = -k_internal[ok] / denom_num[ok]
        hill_den[ok] = -k_internal[ok] / denom_den[ok]

        good = (
            ok
            & np.isfinite(hill_num)
            & np.isfinite(hill_den)
            & (np.abs(hill_den) > EPS)
        )

        out = np.full(hi - lo, np.nan, dtype=np.float64)
        out[good] = hill_num[good] / hill_den[good]
        d[lo:hi] = out

    return d

def rolling_le_variant(
    loss,
    labels,
    *,
    K,
    gie_limit_method,
    num_classes,
):
    """Compute LE trajectory with low peak memory.

    Fixed for every variant:
        ell_err limit = next sample

    Tested inside GIE:
        last3_mean versus next

    The observed-class reference is stored only once per class (C x T),
    not repeated for all N samples.
    """
    x = np.asarray(loss, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    N, T = x.shape
    Gc = build_class_mean_reference(
        x,
        labels,
        num_classes,
    )

    # Only the final LE trajectory is retained.
    le_traj = np.full((N, T), np.nan, dtype=np.float32)

    inv_j = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    first_col = K + 1
    batch_size = 2000

    for t in range(first_col, T):
        boundary = t - K - 1
        tail_start = t - K

        # Error component is always next-sample L.
        L_err = np.asarray(x[:, t], dtype=np.float32)

        if gie_limit_method == "last3_mean":
            L_gie_sample = np.mean(
                x[:, t - 2:t + 1],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
            G_limit_by_class = np.mean(
                Gc[:, t - 2:t + 1],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
        elif gie_limit_method == "next":
            L_gie_sample = np.asarray(x[:, t], dtype=np.float32)
            G_limit_by_class = np.asarray(Gc[:, t], dtype=np.float32)
        else:
            raise ValueError(gie_limit_method)

        # Process samples in batches and write LE directly.
        for lo in range(0, N, batch_size):
            hi = min(lo + batch_size, N)

            xb = np.asarray(
                x[lo:hi, tail_start:t],
                dtype=np.float64,
            )
            boundary_b = np.asarray(
                x[lo:hi, boundary],
                dtype=np.float64,
            )
            Lerr_b = np.asarray(
                L_err[lo:hi],
                dtype=np.float64,
            )

            r0 = np.abs(boundary_b - Lerr_b)
            rj = np.abs(xb - Lerr_b[:, None])

            valid_err = (
                np.isfinite(Lerr_b)
                & np.isfinite(r0)
                & (r0 != 0.0)
                & np.all(
                    np.isfinite(rj) & (rj != 0.0),
                    axis=1,
                )
            )

            ell_err = np.full(hi - lo, np.nan, dtype=np.float64)
            if np.any(valid_err):
                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                ):
                    vals = (
                        np.log(
                            rj[valid_err]
                            / r0[valid_err, None]
                        )
                        * inv_j[None, :]
                    )
                ell_err[valid_err] = np.mean(vals, axis=1)

            labels_b = labels[lo:hi]
            G_tail_b = np.asarray(
                Gc[labels_b, tail_start:t],
                dtype=np.float64,
            )
            Lgie_b = np.asarray(
                L_gie_sample[lo:hi],
                dtype=np.float64,
            )
            G_limit_b = np.asarray(
                G_limit_by_class[labels_b],
                dtype=np.float64,
            )

            id_gie = vectorized_gie_window_with_limits(
                xb,
                G_tail_b,
                Lgie_b,
                G_limit_b,
                batch_size=hi - lo,
            )

            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
            ):
                le = ell_err + np.log(np.abs(id_gie))

            valid = (
                valid_err
                & np.isfinite(id_gie)
                & (id_gie != 0.0)
                & np.isfinite(le)
            )

            if np.any(valid):
                block = np.full(hi - lo, np.nan, dtype=np.float32)
                block[valid] = le[valid].astype(np.float32)
                le_traj[lo:hi, t] = block

    return {
        "le_traj": le_traj,
        "first_available_column": first_col,
        "first_available_epoch": first_col + 1,
    }

def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)

    finite = np.isfinite(score)
    if np.sum(finite) < 2:
        return np.nan

    yy = y[finite]
    if np.all(yy) or np.all(~yy):
        return np.nan

    return binary_auc_from_scores(
        score[finite & y],
        score[finite & ~y],
    )


def auc_curve(y, score_nt):
    return np.asarray(
        [
            auc_binary(y, score_nt[:, t])
            for t in range(score_nt.shape[1])
        ],
        dtype=np.float64,
    )


def best_auc(auc, epochs):
    finite = np.flatnonzero(np.isfinite(auc))
    t = int(finite[np.nanargmax(auc[finite])])
    return float(auc[t]), int(epochs[t]), t


def temporal_stability(score_nt, start_col):
    x = np.asarray(score_nt[:, start_col:])

    a = x[:, :-1]
    b = x[:, 1:]
    valid = np.isfinite(a) & np.isfinite(b)

    steps = np.abs(b[valid] - a[valid])

    return {
        "median_abs_step": float(np.median(steps)),
        "p95_abs_step": float(np.quantile(steps, 0.95)),
        "n_steps": int(steps.size),
    }


def topq_metrics(y, score, q):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    N = y.size
    finite_idx = np.flatnonzero(np.isfinite(score))
    k = min(
        max(1, int(round(float(q) * N))),
        finite_idx.size,
    )

    selected = np.zeros(N, dtype=bool)
    fs = score[finite_idx]
    local = np.argpartition(fs, -k)[-k:]
    selected[finite_idx[local]] = True

    TP = int(np.sum(selected & y))
    FP = int(np.sum(selected & ~y))
    FN = int(np.sum(~selected & y))
    TN = int(np.sum(~selected & ~y))

    return {
        "q": float(q),
        "TPR": TP / (TP + FN),
        "FPR": FP / (FP + TN),
        "precision": TP / (TP + FP),
        "n_selected": int(np.sum(selected)),
    }


def oracle_max_tpr(y, score, target_fpr):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    finite = np.isfinite(score)
    yy = y[finite]
    ss = score[finite]

    order = np.argsort(-ss, kind="mergesort")
    yy = yy[order]
    ss = ss[order]

    n_pos = int(np.sum(yy))
    n_neg = int(np.sum(~yy))

    tp = np.cumsum(yy.astype(np.int64))
    fp = np.cumsum((~yy).astype(np.int64))

    tpr = tp / n_pos
    fpr = fp / n_neg

    feasible = np.flatnonzero(fpr <= target_fpr)
    best_tpr = np.max(tpr[feasible])
    cand = feasible[np.isclose(tpr[feasible], best_tpr)]
    best = int(cand[np.argmin(fpr[cand])])

    return {
        "oracle_TPR": float(tpr[best]),
        "oracle_FPR": float(fpr[best]),
        "oracle_selected_fraction": float((best + 1) / y.size),
    }


def write_csv(path, rows):
    fields = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def evaluate(
    name,
    result,
    *,
    K,
    labels,
    y,
    epochs,
    num_classes,
    top_fractions,
    target_fpr,
):
    le = result["le_traj"]
    start_col = int(result["first_available_column"])

    z_tn = standardize_monitoring_score(
        le.T,
        labels,
        direction="higher",
        num_classes=num_classes,
        start_index=start_col,
    )
    z_nt = z_tn.T

    raw_auc = auc_curve(y, le)
    z_auc = auc_curve(y, z_nt)
    raw_best, raw_ep, _ = best_auc(raw_auc, epochs)
    z_best, z_ep, _ = best_auc(z_auc, epochs)

    pair = exact_pairwise_scores_by_class(
        le,
        labels,
        start_index=start_col,
    )

    pair_score = np.asarray(
        pair["cumulative_score"],
    )

    pair_auc = auc_curve(y, pair_score)
    pair_best, pair_ep, pair_t = best_auc(
        pair_auc,
        epochs,
    )

    stability = temporal_stability(
        le,
        start_col,
    )

    # Best tested q+epoch under the common 5% FPR budget.
    best_q_case = None
    topq_rows = []

    for t in range(start_col, le.shape[1]):
        for q in top_fractions:
            m = topq_metrics(
                y,
                pair_score[:, t],
                q,
            )
            row = {
                "variant": name,
                "epoch": int(epochs[t]),
                "pairwise_auc": float(pair_auc[t]),
                **m,
            }
            topq_rows.append(row)

            if m["FPR"] <= target_fpr:
                if (
                    best_q_case is None
                    or m["TPR"] > best_q_case["TPR"]
                    or (
                        np.isclose(m["TPR"], best_q_case["TPR"])
                        and m["FPR"] < best_q_case["FPR"]
                    )
                ):
                    best_q_case = row

    oracle_rows = []
    for t in range(start_col, le.shape[1]):
        o = oracle_max_tpr(
            y,
            pair_score[:, t],
            target_fpr,
        )
        oracle_rows.append({
            "epoch": int(epochs[t]),
            **o,
        })

    best_oracle = max(
        oracle_rows,
        key=lambda r: (r["oracle_TPR"], -r["oracle_FPR"]),
    )

    summary = {
        "variant": name,
        "K": K,
        "err_limit_method": "next",
        "gie_limit_method":
            "next" if "err_next__gie_next" in name
            else "last3_mean",
        "reference_method": "mean",
        "raw_le_best_auc": raw_best,
        "raw_le_best_epoch": raw_ep,
        "z_le_best_auc": z_best,
        "z_le_best_epoch": z_ep,
        "pairwise_best_auc": pair_best,
        "pairwise_best_epoch": pair_ep,
        "le_median_abs_epoch_step":
            stability["median_abs_step"],
        "le_p95_abs_epoch_step":
            stability["p95_abs_step"],
        "best_tested_q": float(best_q_case["q"]),
        "best_tested_q_epoch": int(best_q_case["epoch"]),
        "best_tested_q_TPR": float(best_q_case["TPR"]),
        "best_tested_q_FPR": float(best_q_case["FPR"]),
        "oracle_TPR": float(best_oracle["oracle_TPR"]),
        "oracle_FPR": float(best_oracle["oracle_FPR"]),
        "oracle_epoch": int(best_oracle["epoch"]),
        "oracle_selected_fraction":
            float(best_oracle["oracle_selected_fraction"]),
    }

    return summary, topq_rows, pair_auc


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.common_npz, allow_pickle=False)

    loss = np.asarray(d["loss_traj"], dtype=np.float32)
    labels = np.asarray(d["observed_label"], dtype=np.int64)
    y = np.asarray(d["is_anomaly"], dtype=bool)
    epochs = np.asarray(d["epoch"], dtype=np.int64)

    summary_rows = []
    all_topq = []
    pair_auc_by_variant = {}

    for K in args.K_values:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")

        for gie_method in ("last3_mean", "next"):
            base_name = (
                "err_next__gie_last3"
                if gie_method == "last3_mean"
                else "err_next__gie_next"
            )
            name = f"{base_name}__K{K}"

            print(
                f"Computing {name}: "
                f"K={K}, err_limit=next, "
                f"GIE_limit={gie_method} ..."
            )

            result = rolling_le_variant(
                loss, labels, K=K,
                gie_limit_method=gie_method,
                num_classes=args.num_classes,
            )

            summary, topq, pair_auc = evaluate(
                name, result, K=K, labels=labels, y=y,
                epochs=epochs, num_classes=args.num_classes,
                top_fractions=args.top_fractions,
                target_fpr=args.target_fpr,
            )
            summary["variant_base"] = base_name
            summary_rows.append(summary)
            all_topq.extend(topq)
            pair_auc_by_variant[name] = pair_auc

            print(
                f"{name:>30s} | "
                f"raw={summary['raw_le_best_auc']:.6f} "
                f"z={summary['z_le_best_auc']:.6f} "
                f"pair={summary['pairwise_best_auc']:.6f} "
                f"| dLE50={summary['le_median_abs_epoch_step']:.6g} "
                f"dLE95={summary['le_p95_abs_epoch_step']:.6g} "
                f"| best tested q={summary['best_tested_q']:.2f} "
                f"TPR={summary['best_tested_q_TPR']:.4f} "
                f"FPR={summary['best_tested_q_FPR']:.4f}"
            )
            del result
            import gc
            gc.collect()

    write_csv(
        args.output_dir / "gie_limit_k_sweep_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output_dir / "gie_limit_k_sweep_topq.csv",
        all_topq,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for base_name in ("err_next__gie_last3", "err_next__gie_next"):
        rr = sorted([r for r in summary_rows if r["variant_base"] == base_name], key=lambda r: int(r["K"]))
        ax.plot([int(r["K"]) for r in rr], [float(r["pairwise_best_auc"]) for r in rr], marker="o", label=base_name)
    ax.set_xlabel("K")
    ax.set_ylabel("Best cumulative pairwise ROC-AUC")
    ax.set_title("GIE limit comparison across K")
    ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "fig_gie_limit_pairwise_auc_vs_K.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for base_name in ("err_next__gie_last3", "err_next__gie_next"):
        rr = sorted([r for r in summary_rows if r["variant_base"] == base_name], key=lambda r: int(r["K"]))
        ax.plot([int(r["K"]) for r in rr], [float(r["best_tested_q_TPR"]) for r in rr], marker="o", label=base_name)
    ax.set_xlabel("K"); ax.set_ylabel("Best tested TPR under FPR <= target")
    ax.set_title("Low-FPR detection vs K"); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "fig_gie_limit_best_tpr_vs_K.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for base_name in ("err_next__gie_last3", "err_next__gie_next"):
        rr = sorted([r for r in summary_rows if r["variant_base"] == base_name], key=lambda r: int(r["K"]))
        ax.plot([int(r["K"]) for r in rr], [float(r["le_median_abs_epoch_step"]) for r in rr], marker="o", label=base_name)
    ax.set_xlabel("K"); ax.set_ylabel("Median |ΔLE|")
    ax.set_title("LE temporal stability vs K"); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "fig_gie_limit_stability_vs_K.png", dpi=180); plt.close(fig)

    config = {
        "artifact": "LE_GIE_limit_sensitivity_K_sweep",
        "K_values": [int(k) for k in args.K_values],
        "theoretical_estimator":
            "ell_hat = ell_err_hat + log(abs(m_GIE_hat))",
        "reference_method": "mean",
        "error_limit_method": "next",
        "tested_gie_limit_methods": [
            "last3_mean",
            "next",
        ],
        "top_fractions": [
            float(q)
            for q in args.top_fractions
        ],
        "target_fpr": float(args.target_fpr),
        "production_modules_modified": False,
        "memory_mode": "float32 loss + compact class reference + one float32 LE trajectory + sample batching; component trajectories not retained",
        "purpose":
            "test next-sample versus last-three-mean L inside GIE across K while ell_err remains fixed at next-sample L",
        "selection_warning":
            "Exploratory labeled-run comparison; freeze chosen variant before independent validation.",
    }

    with (
        args.output_dir / "gie_limit_k_sweep_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print(
        "Done. Error-component L remained fixed at next for ALL K values and BOTH GIE variants."
    )
    print(
        f"Outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()