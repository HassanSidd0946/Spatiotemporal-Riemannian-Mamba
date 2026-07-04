# =============================================================================
# command to run : modal run run_step4_condition4_subspace_calibration.py::main
# run_step4_condition4_subspace_calibration.py
# Condition 4 (v2): PCA Subspace-Regularized 15% Few-Shot Calibration
# Strict 29-Subject LOSO Benchmark, Precomputed Tangent-Space Features
#
# WHY THIS REWRITE EXISTS:
#   The first Condition 4 attempt fed the full 1953-d tangent vector into a
#   linear head fine-tuned on only ~60 calibration samples (15% of one
#   subject). That's a classic N << P regime (60 samples, 1953 features):
#   the head can always find a direction that perfectly separates the tiny
#   calibration set by chance, and that direction generalizes poorly to the
#   85% held-out test set. Result: 52.27% mean accuracy, worse than the
#   55.52% zero-shot Condition 3 baseline it was supposed to beat.
#
#   The fix is dimensionality REDUCTION before adaptation, not more
#   capacity. Projecting into a ~35-dim PCA subspace fit on the 28 training
#   subjects flips the ratio to N_cal (~60) > P_subspace (~35), which is a
#   regime linear classifiers are well-behaved in. Blending the subject's
#   local classifier with the group's global classifier via a
#   cross-validated shrinkage weight further damps variance from the small
#   calibration set, borrowing statistical strength from the 28-subject
#   group model exactly where the calibration set alone is unreliable.
#
# LEAKAGE CONTROLS (all four preserved / strengthened from v1):
#   1. StandardScaler AND PCA are fit ONLY on the 28 training subjects'
#      tangent vectors, then applied transform-only to D_cal and D_test for
#      the held-out subject. The held-out subject never influences the
#      subspace itself.
#   2. The 15%/85% split of subject k's data is a single stratified
#      shuffle split (exact class balance in D_cal); disjointness of D_cal
#      and D_test is verified with an explicit assertion.
#   3. The global classifier is fit only on the 28-subject training pool.
#      The local classifier and the shrinkage weight are both selected
#      using D_cal ONLY (via internal stratified k-fold CV on D_cal) —
#      D_test is touched exactly once, for final evaluation.
#   4. No component of the pipeline (scaler, PCA, global classifier, local
#      classifier, or shrinkage weight) is ever fit using D_test.
#
# Usage: modal run run_step4_condition4_subspace_calibration.py::main
#        modal run run_step4_condition4_subspace_calibration.py::inspect
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-condition4-subspace-calibration")
volume = modal.Volume.from_name("eeg-data-vol")

DATA_PATH        = "/data/tangent_features_ea.npz"
CONDITION3_JSON  = "/data/results_condition3_ea_zeroshot.json"
OUTPUT_JSON      = "/data/results_condition4_subspace_calibrated.json"
VOLUME_PATH      = "/data"

CONDITION2_BASELINE_ACC = 0.5553   # Mamba, zero-shot LOSO, no EA
CONDITION3_BASELINE_ACC = 0.5552   # SpatioMamba + EA, zero-shot LOSO

CAL_FRACTION       = 0.15   # 15% subject-specific calibration set
RANDOM_SEED        = 42
PCA_N_COMPONENTS   = 35     # int -> fixed component count; float in (0,1) -> explained-variance target
LOGREG_C           = 1.0    # inverse L2 regularization strength
SHRINK_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS    = 3       # internal CV folds, computed ONLY on D_cal
RESCUE_LOW_THRESH  = 0.55    # Condition-3 acc below this = "collapsed"
RESCUE_HIGH_THRESH = 0.60    # post-calibration acc at/above this = "rescued"

calib_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "scikit-learn==1.4.2",
        "scipy",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run run_step4_condition4_subspace_calibration.py::inspect
# =============================================================================

@app.function(image=calib_image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_npz():
    import numpy as np
    raw = np.load(DATA_PATH, allow_pickle=True)
    print(f"\nArchive: {DATA_PATH}")
    print(f"Keys found: {raw.files}\n")
    for k in raw.files:
        arr = raw[k]
        print(f"  '{k}': shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim <= 1 and arr.size <= 40:
            print(f"      sample values: {arr[:10]}")
    return raw.files


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    keys = inspect_npz.remote()
    print(f"\nKeys in {DATA_PATH}: {keys}")


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================

@app.function(
    image=calib_image,
    cpu=4.0,
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=16384,
)
def run_condition4_subspace_calibration():

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, json, os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-subspace")

    np.random.seed(RANDOM_SEED)

    # =========================================================================
    # SECTION 3: DATA LOADING (precomputed tangent-space features)
    # =========================================================================
    log.info(f"Loading {DATA_PATH} ...")
    raw = np.load(DATA_PATH, allow_pickle=True)
    log.info(f"Archive keys found: {raw.files}")

    def _pick_key(candidates, purpose):
        for c in candidates:
            if c in raw.files:
                return c
        raise KeyError(
            f"Could not find a key for '{purpose}' in {DATA_PATH}. "
            f"Tried {candidates}, but archive only contains {raw.files}. "
            f"Run `modal run run_step4_condition4_subspace_calibration.py::inspect` "
            f"to see exact keys/shapes and update the candidate list."
        )

    X_KEY   = _pick_key(
        ["X", "X_ts", "features", "tangent_features", "tangent_vectors", "data", "X_tangent"],
        "feature matrix",
    )
    Y_KEY   = _pick_key(["y", "labels", "label", "Y", "targets"], "labels")
    SUB_KEY = _pick_key(
        ["subjects", "subject", "subject_ids", "subject_id", "groups", "sub_ids"],
        "subject IDs",
    )
    log.info(f"Using keys → features: '{X_KEY}', labels: '{Y_KEY}', subjects: '{SUB_KEY}'")

    X_np        = raw[X_KEY].astype(np.float64)     # expected (11610, 1953)
    y_np        = raw[Y_KEY].astype(np.int64)
    subjects_np = raw[SUB_KEY]

    if X_np.ndim != 2:
        log.warning(f"Feature array has shape {X_np.shape} (ndim={X_np.ndim}); "
                    f"flattening all trailing dims into a single feature axis.")
        X_np = X_np.reshape(X_np.shape[0], -1)

    N, RAW_DIM = X_np.shape
    N_CLASSES  = int(y_np.max()) + 1
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Classes: {N_CLASSES}")
    log.info(f"Subjects in dataset: {sorted(np.unique(subjects_np).tolist())}")
    assert N_CLASSES == 2, "Sensitivity/specificity below assume binary classification."

    EXPECTED_TANG_DIM = 62 * 63 // 2  # 1953
    if RAW_DIM != EXPECTED_TANG_DIM:
        log.warning(
            f"⚠️  Feature dimension is {RAW_DIM}, but the expected EA "
            f"tangent-vector size for 62 channels is {EXPECTED_TANG_DIM}. "
            f"Double-check key '{X_KEY}' really holds tangent-space features."
        )
    else:
        log.info(f"✓ Feature dimension matches expected tangent-vector size ({EXPECTED_TANG_DIM}).")

    # Load Condition 3 per-fold accuracies for the live Δ comparison.
    condition3_per_subject_acc = {}
    if os.path.exists(CONDITION3_JSON):
        with open(CONDITION3_JSON, "r") as f:
            c3 = json.load(f)
        for rec in c3.get("fold_results", []):
            condition3_per_subject_acc[str(rec["test_subject"])] = float(rec["test_accuracy"])
        log.info(f"Loaded Condition 3 per-fold accuracies for {len(condition3_per_subject_acc)} subjects.")
    else:
        log.warning(f"{CONDITION3_JSON} not found — Δ vs Condition 3 will use the global baseline only.")

    # =========================================================================
    # SECTION 4: METRIC / CLASSIFIER UTILITIES
    # =========================================================================

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {
            "sensitivity"      : float(sensitivity),
            "specificity"      : float(specificity),
            "f1"               : float(f1),
            "confusion_matrix" : cm.tolist(),   # [[tn, fp], [fn, tp]]
        }

    def linear_predict(coef, intercept, X):
        logits = X @ coef.T + intercept
        return (logits.ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, log):
        """
        Returns (coef_final, intercept_final, best_shrink, global_acc_on_cal, local_acc_on_cal).

        global_clf : LogisticRegression fit on the 28-subject training pool (subspace).
        local_clf  : LogisticRegression fit on D_cal alone (subspace).
        shrink     : blend weight in [0, 1] selected by internal CV on D_cal ONLY.
                     final = shrink * local_coef + (1 - shrink) * global_coef
                     shrink = 0   -> pure global (group) model, ignores subject k entirely
                     shrink = 1   -> pure local (subject-specific) model
        """
        global_clf = LogisticRegression(
            C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED
        ).fit(X_train_pca, y_train)

        local_clf_full = LogisticRegression(
            C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED
        ).fit(X_cal_pca, y_cal)

        # ---- Select shrinkage weight via internal stratified CV on D_cal ONLY ----
        n_splits = min(SHRINK_CV_FOLDS, np.bincount(y_cal).min())
        n_splits = max(n_splits, 2)  # StratifiedKFold needs >= 2
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_cal_tr, X_cal_val = X_cal_pca[tr_idx], X_cal_pca[val_idx]
            y_cal_tr, y_cal_val = y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_cal_tr)) < 2:
                continue
            local_fold = LogisticRegression(
                C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED
            ).fit(X_cal_tr, y_cal_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                preds = linear_predict(coef_b, icpt_b, X_cal_val)
                shrink_scores[shrink].append((preds == y_cal_val).mean())

        mean_shrink_scores = {
            s: (np.mean(v) if len(v) > 0 else -1.0) for s, v in shrink_scores.items()
        }
        best_shrink = max(mean_shrink_scores, key=mean_shrink_scores.get)
        log.info(
            f"      Shrinkage CV scores: "
            + ", ".join(f"{s:.1f}→{mean_shrink_scores[s]:.3f}" for s in SHRINK_GRID)
        )
        log.info(f"      Selected shrinkage weight (local vs global): {best_shrink:.2f}")

        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_

        global_acc_on_cal = (global_clf.predict(X_cal_pca) == y_cal).mean()
        local_acc_on_cal  = (local_clf_full.predict(X_cal_pca) == y_cal).mean()

        return coef_final, icpt_final, best_shrink, global_clf, float(global_acc_on_cal), float(local_acc_on_cal)

    # =========================================================================
    # SECTION 5: STRICT 29-FOLD LOSO + PCA-SUBSPACE 15% CALIBRATION
    # =========================================================================

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    TEST_FOLDS      = len(unique_subjects)

    log.info(f"\nAll subjects     : {unique_subjects}")
    log.info(f"Running folds    : {TEST_FOLDS} (strict LOSO, all subjects)")
    log.info(f"PCA components   : {PCA_N_COMPONENTS}")
    log.info(f"Calibration      : {CAL_FRACTION*100:.0f}% of held-out subject, "
             f"shrinkage-blended LogisticRegression(C={LOGREG_C})")

    fold_records = []
    all_test_acc = []

    for fold_idx, test_sub in enumerate(unique_subjects):

        fold_start = time.time()
        log.info(f"\n{'═'*70}")
        log.info(f"  FOLD {fold_idx + 1}/{TEST_FOLDS}  —  Held-Out Subject: sub-{test_sub}")
        log.info(f"{'═'*70}")

        # ---- Strict split: 28 train subjects vs held-out subject k ----
        is_holdout = subjects_np == test_sub
        is_train28 = ~is_holdout

        X_train28_raw = X_np[is_train28]; y_train28 = y_np[is_train28]
        X_k_raw       = X_np[is_holdout]; y_k        = y_np[is_holdout]

        log.info(f"  28-subject pool : {X_train28_raw.shape}")
        log.info(f"  Held-out subject: {X_k_raw.shape}  ({y_k.shape[0]} epochs, subject {test_sub} ONLY)")

        # =====================================================================
        # STAGE A — Leak-Free Dimensionality Compression (Scaler + PCA)
        #   Fit strictly and exclusively on the 28-subject training pool.
        # =====================================================================
        scaler = StandardScaler()
        X_train28_z = scaler.fit_transform(X_train28_raw)
        X_k_z       = scaler.transform(X_k_raw)

        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
        X_train28_pca = pca.fit_transform(X_train28_z)
        X_k_pca       = pca.transform(X_k_z)

        n_components_used = pca.n_components_
        explained_var     = float(np.sum(pca.explained_variance_ratio_))
        log.info(
            f"  PCA subspace     : {RAW_DIM} → {n_components_used} dims "
            f"(explained variance = {explained_var*100:.1f}%)"
        )

        # =====================================================================
        # STAGE A(cont.) — 15% Stratified Calibration Split on Held-Out Subject k
        # =====================================================================
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED
        )
        cal_idx, test_idx = next(sss.split(X_k_pca, y_k))

        # Explicit leakage assertion: D_cal and D_test must be disjoint.
        assert set(cal_idx.tolist()).isdisjoint(set(test_idx.tolist())), \
            "Leakage detected: D_cal and D_test overlap!"

        X_cal_pca, y_cal   = X_k_pca[cal_idx],  y_k[cal_idx]
        X_test_pca, y_test = X_k_pca[test_idx], y_k[test_idx]

        cal_class_counts  = np.bincount(y_cal,  minlength=N_CLASSES)
        test_class_counts = np.bincount(y_test, minlength=N_CLASSES)
        log.info(f"  D_cal  : {X_cal_pca.shape}  class counts={cal_class_counts.tolist()}  "
                 f"(N_cal={X_cal_pca.shape[0]} vs P_subspace={n_components_used} "
                 f"→ ratio={X_cal_pca.shape[0]/max(n_components_used,1):.2f}x)")
        log.info(f"  D_test : {X_test_pca.shape}  class counts={test_class_counts.tolist()}")

        # =====================================================================
        # STAGE B — Global Baseline + Warm-Started Shrinkage Adaptation
        # =====================================================================
        (coef_final, icpt_final, best_shrink, global_clf,
         global_acc_on_cal, local_acc_on_cal) = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal, log
        )

        # Pre-calibration reference: pure global (28-subject) model on D_test.
        pre_cal_preds = global_clf.predict(X_test_pca)
        pre_cal_acc   = float((pre_cal_preds == y_test).mean())

        # Final shrinkage-blended prediction on D_test (touched exactly once).
        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        best_test_acc = float((final_preds == y_test).mean())

        metrics = compute_binary_metrics(y_test, final_preds)

        c3_acc = condition3_per_subject_acc.get(str(test_sub), CONDITION3_BASELINE_ACC)
        delta_vs_c3 = best_test_acc - c3_acc
        rescued = (c3_acc < RESCUE_LOW_THRESH) and (best_test_acc >= RESCUE_HIGH_THRESH)

        log.info(
            f"  └── BEST  →  pre_cal_acc={pre_cal_acc:.4f}  post_cal_acc={best_test_acc:.4f}  "
            f"(shrink={best_shrink:.2f})  sens={metrics['sensitivity']:.4f}  "
            f"spec={metrics['specificity']:.4f}  f1={metrics['f1']:.4f}"
        )
        log.info(
            f"      Δ vs Condition 3 (sub-{test_sub}: {c3_acc*100:.2f}%) "
            f"→ {delta_vs_c3*100:+.2f} pp" + ("   🟢 RESCUED" if rescued else "")
        )

        fold_records.append({
            "fold_index"           : fold_idx,
            "test_subject"         : str(test_sub),
            "n_cal_epochs"         : int(X_cal_pca.shape[0]),
            "n_test_epochs"        : int(X_test_pca.shape[0]),
            "pca_components_used"  : int(n_components_used),
            "pca_explained_variance": explained_var,
            "cal_class_counts"     : cal_class_counts.tolist(),
            "test_class_counts"    : test_class_counts.tolist(),
            "best_shrink_weight"   : float(best_shrink),
            "global_acc_on_cal"    : global_acc_on_cal,
            "local_acc_on_cal"     : local_acc_on_cal,
            "pre_calibration_acc"  : pre_cal_acc,
            "post_calibration_acc" : best_test_acc,
            "sensitivity"          : metrics["sensitivity"],
            "specificity"          : metrics["specificity"],
            "f1"                   : metrics["f1"],
            "confusion_matrix"     : metrics["confusion_matrix"],
            "condition3_acc"       : float(c3_acc),
            "delta_vs_condition3"  : float(delta_vs_c3),
            "rescued"              : bool(rescued),
            "y_true"               : y_test.tolist(),
            "y_pred"               : final_preds.tolist(),
        })
        all_test_acc.append(best_test_acc)

        elapsed = time.time() - fold_start
        log.info(f"  Fold {fold_idx+1} elapsed: {elapsed:.1f}s")

    # =========================================================================
    # SECTION 6: AGGREGATE REPORT + 4-CONDITION COMPARISON
    # =========================================================================

    mean_acc  = float(np.mean(all_test_acc))
    std_acc   = float(np.std(all_test_acc))
    mean_sens = float(np.mean([f["sensitivity"] for f in fold_records]))
    mean_spec = float(np.mean([f["specificity"] for f in fold_records]))
    mean_f1   = float(np.mean([f["f1"] for f in fold_records]))
    n_rescued = sum(1 for f in fold_records if f["rescued"])
    mean_pre_cal_acc = float(np.mean([f["pre_calibration_acc"] for f in fold_records]))
    delta_vs_c3_global = mean_acc - CONDITION3_BASELINE_ACC
    delta_vs_c2_global = mean_acc - CONDITION2_BASELINE_ACC

    log.info(f"\n{'═'*70}")
    log.info(f"  CONDITION 4 (v2) — FULL {TEST_FOLDS}-FOLD SUBSPACE CALIBRATION RESULTS")
    log.info(f"{'═'*70}")
    log.info(f"  Per-fold post-cal accuracy : {[f'{a:.4f}' for a in all_test_acc]}")
    log.info(f"  ──────────────────────────────────────────────────────────")
    log.info(f"  Mean Pre-Calibration Acc (pure global model) : {mean_pre_cal_acc:.4f}")
    log.info(f"  Mean Post-Calibration Acc (shrinkage blend)  : {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(f"  Mean Sensitivity  : {mean_sens:.4f}")
    log.info(f"  Mean Specificity  : {mean_spec:.4f}")
    log.info(f"  Mean F1           : {mean_f1:.4f}")
    log.info(f"  Subjects rescued  : {n_rescued} "
             f"(collapsed C3 <{RESCUE_LOW_THRESH*100:.0f}% → post-cal ≥{RESCUE_HIGH_THRESH*100:.0f}%)")
    log.info(f"{'═'*70}")
    log.info(f"  4-CONDITION COMPARISON SUMMARY")
    log.info(f"  ──────────────────────────────────────────────────────────")
    log.info(f"  {'Condition':<50}{'Mean Acc':>12}")
    log.info(f"  {'-'*62}")
    log.info(f"  {'Condition 2: Mamba, zero-shot, no EA':<50}{CONDITION2_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 3: SpatioMamba + EA, zero-shot':<50}{CONDITION3_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 4 (v2): PCA-Subspace + Shrinkage Calib.':<50}{mean_acc*100:>11.2f}%")
    log.info(f"  {'-'*62}")
    log.info(f"  Δ (Condition 4 − Condition 3) : {delta_vs_c3_global*100:+.2f} pp")
    log.info(f"  Δ (Condition 4 − Condition 2) : {delta_vs_c2_global*100:+.2f} pp")
    log.info(f"{'═'*70}\n")

    # =========================================================================
    # SECTION 7: SAVE RESULTS TO VOLUME + COMMIT
    # =========================================================================

    results_payload = {
        "condition"                : "Condition 4 (v2): PCA Subspace-Regularized 15% Few-Shot Calibration",
        "n_folds"                  : TEST_FOLDS,
        "hyperparameters"          : {
            "cal_fraction"      : CAL_FRACTION,
            "pca_n_components"  : PCA_N_COMPONENTS,
            "logreg_C"          : LOGREG_C,
            "shrink_grid"       : SHRINK_GRID,
            "shrink_cv_folds"   : SHRINK_CV_FOLDS,
            "rescue_low_thresh" : RESCUE_LOW_THRESH,
            "rescue_high_thresh": RESCUE_HIGH_THRESH,
            "random_seed"       : RANDOM_SEED,
        },
        "fold_results"             : fold_records,
        "mean_pre_calibration_acc" : mean_pre_cal_acc,
        "mean_accuracy"            : mean_acc,
        "std_accuracy"             : std_acc,
        "mean_sensitivity"         : mean_sens,
        "mean_specificity"         : mean_spec,
        "mean_f1"                  : mean_f1,
        "n_subjects_rescued"       : n_rescued,
        "condition2_baseline_acc"  : CONDITION2_BASELINE_ACC,
        "condition3_baseline_acc"  : CONDITION3_BASELINE_ACC,
        "delta_vs_condition3"      : delta_vs_c3_global,
        "delta_vs_condition2"      : delta_vs_c2_global,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    log.info(f"  Saved  : {OUTPUT_JSON}")

    volume.commit()
    log.info("  ✓ volume.commit() complete — results_condition4_subspace_calibrated.json")
    log.info("    is now durably available on eeg-data-vol for Phase 3 figures.")

    return {
        "mean_pre_calibration_acc" : mean_pre_cal_acc,
        "mean_accuracy"            : mean_acc,
        "std_accuracy"             : std_acc,
        "mean_sensitivity"         : mean_sens,
        "mean_specificity"         : mean_spec,
        "mean_f1"                  : mean_f1,
        "n_subjects_rescued"       : n_rescued,
        "condition2_baseline_acc"  : CONDITION2_BASELINE_ACC,
        "condition3_baseline_acc"  : CONDITION3_BASELINE_ACC,
        "delta_vs_condition3"      : delta_vs_c3_global,
        "delta_vs_condition2"      : delta_vs_c2_global,
        "output_path"              : OUTPUT_JSON,
    }


# =============================================================================
# SECTION 8: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  Condition 4 (v2) — PCA Subspace-Regularized 15% Calibration")
    print("  Strict 29-Subject LOSO Benchmark (Tangent-Space Features)")
    print("="*70 + "\n")

    results = run_condition4_subspace_calibration.remote()

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<28} : {val}")
    print("="*70)
    print(
        "\n  Full 29-fold Condition 4 (v2) run complete."
        "\n  results_condition4_subspace_calibrated.json committed to eeg-data-vol.\n"
    )