# Noisy-Label Detection Framework

## Objective

Identify training samples whose observed labels have been corrupted while model training proceeds normally.

## Ground truth

A sample is noisy when its training label differs from its original CIFAR-10 class label. Synthetic corruption provides the clean/noisy mask used only for evaluation.

## Monitoring statistics

For every sample and epoch, record a convergence statistic. Two statistics are compared:

1. CKL-based convergence score.
2. Proposed LE estimator.

Each statistic produces an epoch-by-sample score matrix.

## Normalization

At each epoch, raw scores are standardized within the observed class. The score orientation must be specified explicitly so that positive z-scores always denote more noisy-like behavior.

## Temporal detectors

Three mechanisms convert score trajectories into binary decisions:

1. consecutive-run (min-run);
2. sliding-window;
3. EWMA.

## Hyperparameters

The experimental grid may vary threshold, persistence/run length, sliding-window length and required exceedances, and EWMA smoothing parameter.

## Evaluation

### Score level

Compute ROC-AUC independently at each epoch.

### Detector level

Report TPR, FPR, precision, F1, and the median first-detection epoch among noisy samples that are detected.
