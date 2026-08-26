# Convergence-Based Noisy-Label Detection

Research code for evaluating whether convergence-monitoring statistics can identify corrupted training labels during neural-network training.

The framework compares two monitoring statistics:

1. **CKL-based convergence score**
2. **Proposed Lyapunov-exponent (LE) estimator**

Both produce a time series for every training sample.  The same downstream pipeline is then used for both methods:

**per-sample score trajectory → class-wise z-score normalization → temporal detector → noisy-label decision**

The three temporal detectors are:

- consecutive-run (min-run) detection;
- sliding-window detection;
- exponentially weighted moving average (EWMA) detection.

## Experimental setting

The main benchmark is CIFAR-10 with synthetic label corruption.  A fixed fraction of training labels is randomly replaced by an incorrect class label.  The corruption mask is retained only for evaluation, giving exact clean/noisy ground truth.

## Repository layout

```text
convergence-noisy-label-detection/
├── src/convergence_monitoring/
│   ├── data.py          # CIFAR-10 datasets and synthetic label corruption
│   ├── models.py        # CNN-12 model
│   ├── training.py      # training/evaluation utilities
│   ├── estimators.py    # CKL/GIE-related estimators
│   ├── proposed_le.py   # proposed LE estimator
│   ├── detectors.py     # z-scores, min-run, sliding-window, EWMA, AUC
│   └── framework.py     # common CKL/LE detector interface
├── experiments/         # experiment entry points
├── tests/               # unit/smoke tests
├── docs/                # method notes
└── results/             # generated outputs (ignored by git)
```

## Monitoring statistics

### CKL

The CKL implementation models finite-boundary power-law tails and evaluates cumulative KL divergence between sample and reference tail models.  The existing estimator implementation is retained in `estimators.py`.

### Proposed LE estimator

For a scalar trajectory `x`, window length `K`, and endpoint `n`, the proposed estimator uses a finite error-decay component together with an FIE correction:

```text
lambda_hat = ell_err_hat + log(id_fie_hat)
```

The practical `next` limit estimate uses `x[n+1]`, and a rolling interface is provided in `proposed_le.py`.

## Common normalization

Scores are standardized within observed CIFAR-10 classes at each epoch.  `framework.standardize_monitoring_score` also lets the experiment explicitly specify whether larger or smaller raw values are interpreted as more noisy-like.

## Temporal detectors

All three detectors consume the same standardized score trajectory:

- **Min-run:** flag after `m` consecutive threshold exceedances.
- **Sliding window:** flag when at least `k` of the latest `ell` observations exceed the threshold.
- **EWMA:** exponentially smooth the z-score trajectory and threshold the smoothed score.

## Evaluation

The intended evaluation has two levels:

**Score level**
- epoch-wise ROC-AUC.

**Detector level**
- true positive rate (TPR);
- false positive rate (FPR);
- precision;
- F1 score;
- median first-detection epoch.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Status

This repository currently contains the core dataset, model, CKL/estimator, proposed-LE, normalization, and detector implementations.  Experiment entry points can be added under `experiments/` as the CKL and LE experimental protocols are frozen.
