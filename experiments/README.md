# Experiments

The experimental comparison is intentionally staged so CKL and the proposed LE
are evaluated on **exactly the same network-training run**.

## 1. Generate the common loss trajectories

`01_generate_loss_trajectories.py` trains CNN12 once on synthetically corrupted
CIFAR-10 and records the deterministic per-sample cross-entropy loss after every
epoch.  It saves:

- `cifar10_noisy_label_loss_trajectories.npz` — shared `[N, T]` loss and
  correctness trajectories plus true/observed labels and the noisy-label mask;
- `epoch_summary.csv` — training and clean/noisy group diagnostics by epoch;
- `run_config.json` — the exact dataset/training configuration.

Example from the repository root:

```bash
python experiments/01_generate_loss_trajectories.py --download
```

For Wulver, pass the existing CIFAR-10 location rather than downloading:

```bash
python experiments/01_generate_loss_trajectories.py \
  --data-root /mmfs1/home/hs833/Distribution-Divergence/data \
  --output-dir results/common_loss_trajectories
```

Defaults reproduce the current framework setup: 5% noisy labels, seed 66,
CNN12, AdamW, 200 epochs, batch size 128, learning rate `1e-3`, and weight
decay `5e-4`.

## 2. Score the same trajectories

The next scripts should load the NPZ generated in Step 1 rather than retraining:

- `02_compute_ckl_scores.py`
- `03_compute_le_scores.py`

## 3. Compare scores and temporal detectors

Then use shared evaluation scripts:

- `04_score_level_evaluation.py` — epoch-wise ROC-AUC and score diagnostics;
- `05_temporal_detector_grid.py` — min-run, sliding-window, and EWMA using the
  same normalization/evaluation rules for CKL and LE.

This separation prevents training stochasticity from being mistaken for a
method difference between CKL and LE.
