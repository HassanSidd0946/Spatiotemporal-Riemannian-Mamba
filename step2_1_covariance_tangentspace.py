# step2_1_covariance_tangentspace.py

"""
BCI Research Pipeline — Phase 2, Step 2.1
Task : Covariance Estimation → Tangent Space Projection
       Independently verify the core Riemannian geometry transformation.

Input : Mock numpy array X_mock of shape (410, 62, 251)  [n_epochs, n_channels, n_times]
Output: Tangent space vectors of shape (410, 1953)        [n_epochs, n_features]

Formula for tangent space dimensionality:
  n_features = n_channels * (n_channels + 1) / 2
             = 62 * 63 / 2
             = 1953
"""

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# ── Configuration ─────────────────────────────────────────────────────────────
N_EPOCHS   = 410
N_CHANNELS = 62
N_TIMES    = 251
RANDOM_SEED = 42

# ── Step 1 · Mock Data Injection ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 1 · Mock Data Injection")
print(f"{'='*60}")

rng = np.random.default_rng(RANDOM_SEED)

X_mock = rng.standard_normal((N_EPOCHS, N_CHANNELS, N_TIMES)).astype(np.float64)
y_mock = np.array([0] * (N_EPOCHS // 2) + [1] * (N_EPOCHS // 2), dtype=np.int32)

print(f"  X_mock shape : {X_mock.shape}")
print(f"               [n_epochs={X_mock.shape[0]}, n_channels={X_mock.shape[1]}, n_times={X_mock.shape[2]}]")
print(f"  y_mock shape : {y_mock.shape}")
print(f"  Class 0 count: {(y_mock == 0).sum()}")
print(f"  Class 1 count: {(y_mock == 1).sum()}")

# ── Step 2 · Covariance Estimation ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 2 · Covariance Estimation  [estimator='oas']")
print(f"{'='*60}")

cov_estimator = Covariances(estimator='oas')
X_cov = cov_estimator.fit_transform(X_mock)   # shape: (n_epochs, n_channels, n_channels)

print(f"  Covariance matrices shape : {X_cov.shape}")
print(f"                             [n_epochs={X_cov.shape[0]}, n_channels={X_cov.shape[1]}, n_channels={X_cov.shape[2]}]")

# Verify SPD property on a sample matrix
sample_cov = X_cov[0]
eigenvalues = np.linalg.eigvalsh(sample_cov)
is_symmetric = np.allclose(sample_cov, sample_cov.T, atol=1e-10)
is_pos_def   = np.all(eigenvalues > 0)

print(f"\n  SPD verification on X_cov[0]:")
print(f"    Symmetric     : {is_symmetric}")
print(f"    Positive def  : {is_pos_def}  (min eigenvalue = {eigenvalues.min():.6e})")

expected_cov_shape = (N_EPOCHS, N_CHANNELS, N_CHANNELS)
assert X_cov.shape == expected_cov_shape, (
    f"Shape mismatch! Expected {expected_cov_shape}, got {X_cov.shape}"
)
print(f"\n  ✓ Shape assertion passed: {X_cov.shape} == {expected_cov_shape}")

# ── Step 3 · Tangent Space Mapping ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 3 · Tangent Space Mapping  [metric='riemann']")
print(f"{'='*60}")

ts = TangentSpace(metric='riemann')
X_ts = ts.fit_transform(X_cov)   # shape: (n_epochs, n_features)

print(f"  Tangent space vectors shape : {X_ts.shape}")
print(f"                               [n_epochs={X_ts.shape[0]}, n_features={X_ts.shape[1]}]")

# ── Step 4 · Verification ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 4 · Shape Verification Summary")
print(f"{'='*60}")

expected_n_features = N_CHANNELS * (N_CHANNELS + 1) // 2  # = 1953
expected_ts_shape   = (N_EPOCHS, expected_n_features)

print(f"\n  Formula  : n_features = n_channels*(n_channels+1)/2")
print(f"           = {N_CHANNELS}*{N_CHANNELS+1}/2 = {expected_n_features}")

print(f"\n  {'Stage':<35} {'Expected':<20} {'Actual':<20} {'Pass?'}")
print(f"  {'-'*80}")

raw_ok = X_mock.shape == (N_EPOCHS, N_CHANNELS, N_TIMES)
cov_ok = X_cov.shape  == (N_EPOCHS, N_CHANNELS, N_CHANNELS)
ts_ok  = X_ts.shape   == expected_ts_shape

print(f"  {'Raw mock data':<35} {str((N_EPOCHS, N_CHANNELS, N_TIMES)):<20} {str(X_mock.shape):<20} {'✓' if raw_ok else '✗'}")
print(f"  {'Covariance matrices':<35} {str((N_EPOCHS, N_CHANNELS, N_CHANNELS)):<20} {str(X_cov.shape):<20} {'✓' if cov_ok else '✗'}")
print(f"  {'Tangent space vectors':<35} {str(expected_ts_shape):<20} {str(X_ts.shape):<20} {'✓' if ts_ok else '✗'}")

assert raw_ok and cov_ok and ts_ok, "One or more shape assertions failed!"

print(f"\n  X_ts stats (sanity check):")
print(f"    mean  = {X_ts.mean():.6f}")
print(f"    std   = {X_ts.std():.6f}")
print(f"    min   = {X_ts.min():.6f}")
print(f"    max   = {X_ts.max():.6f}")

print(f"\n{'='*60}")
print("  Step 2.1 complete — Riemannian geometry transformation verified.")
print("  X_ts is ready for Step 2.2 (artifact rejection / classification).")
print(f"{'='*60}\n")