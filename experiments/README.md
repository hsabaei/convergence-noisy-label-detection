# Experiments

Place executable experiment scripts here.

Recommended separation:

- `run_cifar10_ckl.py`: train once, record per-sample losses, construct the CKL score trajectory, normalize by observed class, then run all three temporal detectors.
- `run_cifar10_le.py`: use the same training/loss protocol, construct the proposed-LE score trajectory, normalize by observed class, then run the same detector grid.
- `compare_scores.py`: load saved score traces and produce epoch-wise ROC-AUC and detector-level summary tables.

The CKL and LE runs should share dataset seed, corruption mask, model, optimizer, training epochs, and evaluation cadence whenever a paired comparison is intended.
