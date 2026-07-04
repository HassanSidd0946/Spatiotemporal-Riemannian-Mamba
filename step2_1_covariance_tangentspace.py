# # step2_1_covariance_tangentspace.py

# """
# BCI Research Pipeline — Phase 2, Step 2.1
# Task : Covariance Estimation → Tangent Space Projection
#        Independently verify the core Riemannian geometry transformation.

# Input : Mock numpy array X_mock of shape (410, 62, 251)  [n_epochs, n_channels, n_times]
# Output: Tangent space vectors of shape (410, 1953)        [n_epochs, n_features]

# Formula for tangent space dimensionality:
#   n_features = n_channels * (n_channels + 1) / 2
#              = 62 * 63 / 2
#              = 1953
# """

# import numpy as np
# from pyriemann.estimation import Covariances
# from pyriemann.tangentspace import TangentSpace

# # ── Configuration ─────────────────────────────────────────────────────────────
# N_EPOCHS   = 410
# N_CHANNELS = 62
# N_TIMES    = 251
# RANDOM_SEED = 42

# # ── Step 1 · Mock Data Injection ──────────────────────────────────────────────
# print(f"\n{'='*60}")
# print("  Step 1 · Mock Data Injection")
# print(f"{'='*60}")

# rng = np.random.default_rng(RANDOM_SEED)

# X_mock = rng.standard_normal((N_EPOCHS, N_CHANNELS, N_TIMES)).astype(np.float64)
# y_mock = np.array([0] * (N_EPOCHS // 2) + [1] * (N_EPOCHS // 2), dtype=np.int32)

# print(f"  X_mock shape : {X_mock.shape}")
# print(f"               [n_epochs={X_mock.shape[0]}, n_channels={X_mock.shape[1]}, n_times={X_mock.shape[2]}]")
# print(f"  y_mock shape : {y_mock.shape}")
# print(f"  Class 0 count: {(y_mock == 0).sum()}")
# print(f"  Class 1 count: {(y_mock == 1).sum()}")

# # ── Step 2 · Covariance Estimation ────────────────────────────────────────────
# print(f"\n{'='*60}")
# print("  Step 2 · Covariance Estimation  [estimator='oas']")
# print(f"{'='*60}")

# cov_estimator = Covariances(estimator='oas')
# X_cov = cov_estimator.fit_transform(X_mock)   # shape: (n_epochs, n_channels, n_channels)

# print(f"  Covariance matrices shape : {X_cov.shape}")
# print(f"                             [n_epochs={X_cov.shape[0]}, n_channels={X_cov.shape[1]}, n_channels={X_cov.shape[2]}]")

# # Verify SPD property on a sample matrix
# sample_cov = X_cov[0]
# eigenvalues = np.linalg.eigvalsh(sample_cov)
# is_symmetric = np.allclose(sample_cov, sample_cov.T, atol=1e-10)
# is_pos_def   = np.all(eigenvalues > 0)

# print(f"\n  SPD verification on X_cov[0]:")
# print(f"    Symmetric     : {is_symmetric}")
# print(f"    Positive def  : {is_pos_def}  (min eigenvalue = {eigenvalues.min():.6e})")

# expected_cov_shape = (N_EPOCHS, N_CHANNELS, N_CHANNELS)
# assert X_cov.shape == expected_cov_shape, (
#     f"Shape mismatch! Expected {expected_cov_shape}, got {X_cov.shape}"
# )
# print(f"\n  ✓ Shape assertion passed: {X_cov.shape} == {expected_cov_shape}")

# # ── Step 3 · Tangent Space Mapping ────────────────────────────────────────────
# print(f"\n{'='*60}")
# print("  Step 3 · Tangent Space Mapping  [metric='riemann']")
# print(f"{'='*60}")

# ts = TangentSpace(metric='riemann')
# X_ts = ts.fit_transform(X_cov)   # shape: (n_epochs, n_features)

# print(f"  Tangent space vectors shape : {X_ts.shape}")
# print(f"                               [n_epochs={X_ts.shape[0]}, n_features={X_ts.shape[1]}]")

# # ── Step 4 · Verification ─────────────────────────────────────────────────────
# print(f"\n{'='*60}")
# print("  Step 4 · Shape Verification Summary")
# print(f"{'='*60}")

# expected_n_features = N_CHANNELS * (N_CHANNELS + 1) // 2  # = 1953
# expected_ts_shape   = (N_EPOCHS, expected_n_features)

# print(f"\n  Formula  : n_features = n_channels*(n_channels+1)/2")
# print(f"           = {N_CHANNELS}*{N_CHANNELS+1}/2 = {expected_n_features}")

# print(f"\n  {'Stage':<35} {'Expected':<20} {'Actual':<20} {'Pass?'}")
# print(f"  {'-'*80}")

# raw_ok = X_mock.shape == (N_EPOCHS, N_CHANNELS, N_TIMES)
# cov_ok = X_cov.shape  == (N_EPOCHS, N_CHANNELS, N_CHANNELS)
# ts_ok  = X_ts.shape   == expected_ts_shape

# print(f"  {'Raw mock data':<35} {str((N_EPOCHS, N_CHANNELS, N_TIMES)):<20} {str(X_mock.shape):<20} {'✓' if raw_ok else '✗'}")
# print(f"  {'Covariance matrices':<35} {str((N_EPOCHS, N_CHANNELS, N_CHANNELS)):<20} {str(X_cov.shape):<20} {'✓' if cov_ok else '✗'}")
# print(f"  {'Tangent space vectors':<35} {str(expected_ts_shape):<20} {str(X_ts.shape):<20} {'✓' if ts_ok else '✗'}")

# assert raw_ok and cov_ok and ts_ok, "One or more shape assertions failed!"

# print(f"\n  X_ts stats (sanity check):")
# print(f"    mean  = {X_ts.mean():.6f}")
# print(f"    std   = {X_ts.std():.6f}")
# print(f"    min   = {X_ts.min():.6f}")
# print(f"    max   = {X_ts.max():.6f}")

# print(f"\n{'='*60}")
# print("  Step 2.1 complete — Riemannian geometry transformation verified.")
# print("  X_ts is ready for Step 2.2 (artifact rejection / classification).")
# print(f"{'='*60}\n")




































# step2_1_covariance_tangentspace.py

"""
BCI Research Pipeline — Phase 2, Step 2.1  (Condition 3: + Euclidean Alignment)
Task : Load real ds005189 data → Euclidean Alignment (per subject) →
       Covariance Estimation → Tangent Space Projection → save features.

Input : processed_eeg_all_subjects.npz  containing:
          X        (N_total, 62, 251)   [n_epochs, n_channels, n_times]
          y        (N_total,)           labels
          subjects (N_total,)           subject id per epoch
Output: tangent_features_ea.npz containing:
          X_ts      (N_total, 1953)     tangent space features, post-EA
          y         (N_total,)
          subjects  (N_total,)

Formula for tangent space dimensionality:
  n_features = n_channels * (n_channels + 1) / 2
             = 62 * 63 / 2
             = 1953
"""

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from step2_0_euclidean_alignment import align_dataset_per_subject

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH   = "processed_eeg_all_subjects.npz"   # ← point this at your real file
OUTPUT_PATH = "tangent_features_ea.npz"
N_CHANNELS  = 62
EA_EPS      = 1e-6

# ── Step 1 · Load Real ds005189 Data ──────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 1 · Load Real Data")
print(f"{'='*60}")

raw          = np.load(DATA_PATH, allow_pickle=True)
X            = raw["X"].astype(np.float64)      # (N_total, 62, 251)
y            = raw["y"]
subjects     = raw["subjects"]

N_EPOCHS, N_CH, N_TIMES = X.shape
assert N_CH == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {N_CH}"

print(f"  X shape        : {X.shape}")
print(f"                 [n_epochs={X.shape[0]}, n_channels={X.shape[1]}, n_times={X.shape[2]}]")
print(f"  y shape        : {y.shape}")
print(f"  Subjects       : {sorted(np.unique(subjects).tolist())}")
print(f"  Class counts   : {dict(zip(*np.unique(y, return_counts=True)))}")

# ── Step 2 · Euclidean Alignment (per subject, unsupervised) ──────────────────
print(f"\n{'='*60}")
print("  Step 2 · Euclidean Alignment  [per subject, pre-covariance]")
print(f"{'='*60}")

X_ea = align_dataset_per_subject(X, subjects, eps=EA_EPS)
print(f"  X_ea shape     : {X_ea.shape}  (same shape, whitened per subject)")

# ── Step 3 · Covariance Estimation (on EA-aligned epochs) ─────────────────────
print(f"\n{'='*60}")
print("  Step 3 · Covariance Estimation  [estimator='oas']")
print(f"{'='*60}")

cov_estimator = Covariances(estimator='oas')
X_cov = cov_estimator.fit_transform(X_ea)   # shape: (n_epochs, n_channels, n_channels)

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

# ── Step 4 · Tangent Space Mapping ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 4 · Tangent Space Mapping  [metric='riemann']")
print(f"{'='*60}")

ts = TangentSpace(metric='riemann')
X_ts = ts.fit_transform(X_cov)   # shape: (n_epochs, n_features)

print(f"  Tangent space vectors shape : {X_ts.shape}")
print(f"                               [n_epochs={X_ts.shape[0]}, n_features={X_ts.shape[1]}]")

# NOTE on the TangentSpace reference point (read before using in LOSO):
#   ts.fit_transform() computes ONE global Frechet mean across ALL epochs
#   (all subjects) and projects every point relative to it. That mean is
#   unsupervised (no labels) but it IS influenced by the held-out subject's
#   own covariance matrices when you later run LOSO on these precomputed
#   features. This is a much softer leak than label leakage -- it's the same
#   category of "population statistic" leak as fitting a global scaler on
#   all data -- but it is technically non-zero. Your run_loso_trainer_on_modal.py
#   model does NOT have this issue: its TangentSpaceLayer uses a direct
#   matrix-log (implicit identity reference), not a fitted global mean, so
#   it is already leak-free per fold. If your Condition 3 IEEE writeup claims
#   "zero data leakage," verify which of these two paths (this offline export
#   vs. the in-model TangentSpaceLayer) actually feeds your reported LOSO
#   numbers, and disclose whichever mean-fitting behavior applies.

# ── Step 5 · Verification ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 5 · Shape Verification Summary")
print(f"{'='*60}")

expected_n_features = N_CHANNELS * (N_CHANNELS + 1) // 2  # = 1953
expected_ts_shape   = (N_EPOCHS, expected_n_features)

print(f"\n  Formula  : n_features = n_channels*(n_channels+1)/2")
print(f"           = {N_CHANNELS}*{N_CHANNELS+1}/2 = {expected_n_features}")

print(f"\n  {'Stage':<35} {'Expected':<20} {'Actual':<20} {'Pass?'}")
print(f"  {'-'*80}")

raw_ok = X.shape    == (N_EPOCHS, N_CHANNELS, N_TIMES)
ea_ok  = X_ea.shape == (N_EPOCHS, N_CHANNELS, N_TIMES)
cov_ok = X_cov.shape == (N_EPOCHS, N_CHANNELS, N_CHANNELS)
ts_ok  = X_ts.shape  == expected_ts_shape

print(f"  {'Raw real data':<35} {str((N_EPOCHS, N_CHANNELS, N_TIMES)):<20} {str(X.shape):<20} {'✓' if raw_ok else '✗'}")
print(f"  {'EA-aligned data':<35} {str((N_EPOCHS, N_CHANNELS, N_TIMES)):<20} {str(X_ea.shape):<20} {'✓' if ea_ok else '✗'}")
print(f"  {'Covariance matrices':<35} {str((N_EPOCHS, N_CHANNELS, N_CHANNELS)):<20} {str(X_cov.shape):<20} {'✓' if cov_ok else '✗'}")
print(f"  {'Tangent space vectors':<35} {str(expected_ts_shape):<20} {str(X_ts.shape):<20} {'✓' if ts_ok else '✗'}")

assert raw_ok and ea_ok and cov_ok and ts_ok, "One or more shape assertions failed!"

print(f"\n  X_ts stats (sanity check):")
print(f"    mean  = {X_ts.mean():.6f}")
print(f"    std   = {X_ts.std():.6f}")
print(f"    min   = {X_ts.min():.6f}")
print(f"    max   = {X_ts.max():.6f}")

# ── Step 6 · Save ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 6 · Saving Aligned Tangent-Space Features")
print(f"{'='*60}")

np.savez(OUTPUT_PATH, X_ts=X_ts.astype(np.float32), y=y, subjects=subjects)
print(f"  Saved  : {OUTPUT_PATH}")
print(f"    X_ts      : {X_ts.shape}")
print(f"    y         : {y.shape}")
print(f"    subjects  : {subjects.shape}")

print(f"\n{'='*60}")
print("  Step 2.1 complete — EA + Riemannian geometry transformation verified.")
print("  tangent_features_ea.npz is ready for Condition 3 LOSO evaluation.")
print(f"{'='*60}\n")