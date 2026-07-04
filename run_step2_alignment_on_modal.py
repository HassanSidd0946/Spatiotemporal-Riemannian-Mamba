# =============================================================================
# run_step2_alignment_on_modal.py
# Phase 2, Step 2.1 (Condition 3) — runs directly on Modal against the
# persistent eeg-data-vol volume, mirroring run_loso_trainer_on_modal.py's
# App / Volume / mount conventions.
#
# Pipeline: Load (cloud) -> Euclidean Alignment -> OAS Covariance ->
#           SPD Verification -> Riemannian Tangent Space -> Save (cloud)
#
# Usage: modal run run_step2_alignment_on_modal.py
#   (run from the same directory as step2_0_euclidean_alignment.py)
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE  (mirrors run_loso_trainer_on_modal.py)
# =============================================================================

app    = modal.App("bci-step2-alignment")
volume = modal.Volume.from_name("eeg-data-vol")

DATA_PATH   = "/data/processed_eeg_all_subjects.npz"
OUTPUT_PATH = "/data/tangent_features_ea.npz"
VOLUME_PATH = "/data"

step2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2",
        "scikit-learn",
        "pyriemann",
    )
    # Bundles the local step2_0_euclidean_alignment.py into the container
    # image so `from step2_0_euclidean_alignment import align_dataset_per_subject`
    # works exactly like a normal local import. Requires the file to sit in
    # the same directory you run `modal run` from.
    .add_local_python_source("step2_0_euclidean_alignment")
)


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================

@app.function(
    image=step2_image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    memory=16384,
)
def run_step2_alignment():

    import numpy as np
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    import logging

    from step2_0_euclidean_alignment import align_dataset_per_subject

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("step2-alignment")

    N_CHANNELS = 62
    EA_EPS     = 1e-6

    # =========================================================================
    # STEP 1: DATA LOAD  (from the Modal volume)
    # =========================================================================
    log.info(f"{'='*60}")
    log.info("  Step 1 · Load Real Data (from Modal volume)")
    log.info(f"{'='*60}")

    log.info(f"Loading {DATA_PATH} ...")
    raw       = np.load(DATA_PATH, allow_pickle=True)
    X         = raw["X"].astype(np.float64)
    y         = raw["y"]
    subjects  = raw["subjects"]

    N_EPOCHS, N_CH, N_TIMES = X.shape
    assert N_CH == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {N_CH}"

    log.info(f"  X shape        : {X.shape}  [n_epochs, n_channels, n_times]")
    log.info(f"  y shape        : {y.shape}")
    log.info(f"  Subjects       : {sorted(np.unique(subjects).tolist())}")
    log.info(f"  Class counts   : {dict(zip(*np.unique(y, return_counts=True)))}")

    # =========================================================================
    # STEP 2: EUCLIDEAN ALIGNMENT (per subject, unsupervised)
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info("  Step 2 · Euclidean Alignment  [per subject, pre-covariance]")
    log.info(f"{'='*60}")

    X_ea = align_dataset_per_subject(X, subjects, eps=EA_EPS)
    log.info(f"  X_ea shape     : {X_ea.shape}  (same shape, whitened per subject)")

    # =========================================================================
    # STEP 3: OAS COVARIANCE ESTIMATION (on EA-aligned epochs)
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info("  Step 3 · Covariance Estimation  [estimator='oas']")
    log.info(f"{'='*60}")

    cov_estimator = Covariances(estimator='oas')
    X_cov = cov_estimator.fit_transform(X_ea)

    log.info(f"  Covariance matrices shape : {X_cov.shape}")

    # =========================================================================
    # STEP 4: SPD VERIFICATION
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info("  Step 4 · SPD Verification")
    log.info(f"{'='*60}")

    sample_cov    = X_cov[0]
    eigenvalues   = np.linalg.eigvalsh(sample_cov)
    is_symmetric  = np.allclose(sample_cov, sample_cov.T, atol=1e-10)
    is_pos_def    = np.all(eigenvalues > 0)

    log.info(f"  Symmetric     : {is_symmetric}")
    log.info(f"  Positive def  : {is_pos_def}  (min eigenvalue = {eigenvalues.min():.6e})")

    expected_cov_shape = (N_EPOCHS, N_CHANNELS, N_CHANNELS)
    assert X_cov.shape == expected_cov_shape, (
        f"Shape mismatch! Expected {expected_cov_shape}, got {X_cov.shape}"
    )
    assert is_symmetric and is_pos_def, "SPD verification failed!"
    log.info(f"  ✓ SPD assertion passed: {X_cov.shape} == {expected_cov_shape}")

    # =========================================================================
    # STEP 5: RIEMANNIAN TANGENT SPACE MAPPING
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info("  Step 5 · Tangent Space Mapping  [metric='riemann']")
    log.info(f"{'='*60}")

    ts = TangentSpace(metric='riemann')
    X_ts = ts.fit_transform(X_cov)

    expected_n_features = N_CHANNELS * (N_CHANNELS + 1) // 2  # = 1953
    expected_ts_shape    = (N_EPOCHS, expected_n_features)

    log.info(f"  Tangent space vectors shape : {X_ts.shape}")
    log.info(f"  Formula check: {N_CHANNELS}*{N_CHANNELS+1}/2 = {expected_n_features}")

    ts_ok = X_ts.shape == expected_ts_shape
    log.info(f"  {'✓' if ts_ok else '✗'} Shape assertion: {X_ts.shape} == {expected_ts_shape}")
    assert ts_ok, "Tangent space shape mismatch!"

    log.info(f"\n  X_ts stats (sanity check):")
    log.info(f"    mean  = {X_ts.mean():.6f}")
    log.info(f"    std   = {X_ts.std():.6f}")
    log.info(f"    min   = {X_ts.min():.6f}")
    log.info(f"    max   = {X_ts.max():.6f}")

    # Same caveat as the local script: this TangentSpace reference is a
    # single global Frechet mean over ALL subjects, computed here once
    # (unsupervised, no labels) -- soft population-level leak, distinct
    # from the in-model TangentSpaceLayer's identity-reference matrix-log
    # used inside run_loso_trainer_on_modal.py. Disclose whichever path
    # feeds your reported Condition 3 numbers.

    # =========================================================================
    # STEP 6: SAVE BACK TO THE MODAL VOLUME + COMMIT
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info("  Step 6 · Saving Aligned Tangent-Space Features to Volume")
    log.info(f"{'='*60}")

    np.savez(OUTPUT_PATH, X_ts=X_ts.astype(np.float32), y=y, subjects=subjects)
    log.info(f"  Saved  : {OUTPUT_PATH}")
    log.info(f"    X_ts      : {X_ts.shape}")
    log.info(f"    y         : {y.shape}")
    log.info(f"    subjects  : {subjects.shape}")

    # Persist the write across containers -- without this, the file may
    # only live in this container's ephemeral view of the volume.
    volume.commit()
    log.info("  ✓ volume.commit() complete — tangent_features_ea.npz is now")
    log.info("    durably available to run_loso_trainer_on_modal.py and any")
    log.info("    future container that mounts eeg-data-vol.")

    log.info(f"\n{'='*60}")
    log.info("  Step 2.1 complete — EA + Riemannian geometry transformation verified.")
    log.info(f"{'='*60}\n")

    return {
        "n_epochs"        : int(N_EPOCHS),
        "n_channels"      : int(N_CHANNELS),
        "n_features"      : int(expected_n_features),
        "output_path"     : OUTPUT_PATH,
        "spd_symmetric"   : bool(is_symmetric),
        "spd_pos_def"     : bool(is_pos_def),
    }


# =============================================================================
# SECTION 3: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  BCI Step 2.1 — Euclidean Alignment + Riemannian Tangent Space")
    print("  Running on Modal against eeg-data-vol")
    print("="*70 + "\n")

    result = run_step2_alignment.remote()

    print("\n" + "="*70)
    print("  RESULT SUMMARY")
    print("="*70)
    for key, val in result.items():
        print(f"  {key:<15} : {val}")
    print("="*70)
    print(
        "\n  tangent_features_ea.npz has been committed to eeg-data-vol."
        "\n  It is now available at /data/tangent_features_ea.npz for any"
        "\n  Modal function (e.g. run_loso_trainer_on_modal.py) that mounts"
        "\n  the same volume.\n"
    )