"""
Small regression check for the corrected common-limit LE experiment.

This test is standalone and does not require pytest.
Run from anywhere inside the repository with:

    python tests/test_common_limit_consistency.py
"""

import importlib.util
from pathlib import Path
import numpy as np


# tests/test_common_limit_consistency.py
# -> repository root is parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO_ROOT / "experiments" / "07_test_le_improvements.py"

if not EXPERIMENT.exists():
    raise FileNotFoundError(
        f"Could not find corrected experiment at: {EXPERIMENT}"
    )

spec = importlib.util.spec_from_file_location(
    "le_exp",
    EXPERIMENT,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


rng = np.random.default_rng(66)

N = 30
T = 25
K = 8

# Smooth positive trajectories with small sample-specific perturbations.
base = np.linspace(1.0, 0.2, T)
x = (
    base[None, :]
    + 0.02 * rng.normal(size=(N, T))
)

labels = np.repeat(
    np.arange(3),
    10,
)

for method in (
    "last3_mean",
    "next",
    "aitken_guarded",
):
    out = mod.rolling_common_limit_le_gie_batch(
        x,
        labels,
        K=K,
        num_classes=3,
        reference_method="mean",
        limit_method=method,
    )

    t = out["first_available_column"]

    L = out["sample_limit_traj"][:, t]

    if method == "last3_mean":
        expected = np.mean(
            x[:, t - 2:t + 1],
            axis=1,
        )
        np.testing.assert_allclose(
            L,
            expected,
        )

    elif method == "next":
        np.testing.assert_allclose(
            L,
            x[:, t],
        )

    # Verify that the stored LE is composed from ell_err and GIE
    # produced under the same common-limit call.
    ell = out["ell_err_traj"][:, t]
    m = out["id_gie_traj"][:, t]
    lam = out["lambda_traj"][:, t]

    valid = (
        np.isfinite(ell)
        & np.isfinite(m)
        & (m != 0.0)
        & np.isfinite(lam)
    )

    expected_lam = (
        ell[valid]
        + np.log(np.abs(m[valid]))
    )

    np.testing.assert_allclose(
        lam[valid],
        expected_lam,
    )

print(
    "PASS: the same common L_hat rule is applied to the whole LE estimator."
)
print(
    "PASS: ell_hat = ell_err_hat + log(abs(m_GIE_hat)) is unchanged."
)
