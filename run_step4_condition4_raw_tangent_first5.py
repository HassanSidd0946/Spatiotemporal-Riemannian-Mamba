# =============================================================================
# command to run : modal run run_step4_condition4_raw_tangent_first5.py::main
# run_step4_condition4_raw_tangent_first5.py
#
# *** DIAGNOSTIC: RAW TANGENT VECTORS, NO LEARNED BOTTLENECK, 5-SUBJECT SLICE ***
#
# WHY THIS SCRIPT EXISTS:
#   The spatial-only ablation (32-dim learned RiemannianSpatialBranch bottleneck,
#   dropout=0.5) scored 60.41% on subjects 01-05. The ORIGINAL Condition 4v2
#   script (raw 1953-dim tangent vector -> StandardScaler -> PCA(35) ->
#   shrinkage-blended LogReg, NO neural network at all) scored 65.99% on
#   those same 5 subjects. That's a real 5.58pp gap on identical subjects,
#   identical EA whitening, identical calibration algorithm — the only
#   difference is what feature representation feeds the calibration stage.
#
#   This script reproduces the ORIGINAL pipeline exactly (raw tangent vector,
#   no bottleneck, no dropout, no pretraining of any kind) but scoped to
#   subjects 01-05 only, so it's a byte-for-byte-fair comparison against
#   both the 65.99% reference AND the 60.41% bottlenecked-embedding result.
#   If this lands near 65.99%, the bottleneck (fixed 32-dim compression
#   before PCA even sees the data) is confirmed as the cause of the
#   regression — not dropout, not weight decay, not anything else, since
#   this script has none of those at all, matching the original exactly.
#
# WHAT'S IDENTICAL to the original Condition 4v2 script: trial_covariances,
# EA whitening (fit on 28-subject pool only), tangent_vectorize (raw
# 1953-dim, sqrt(2)-scaled upper triangle), StandardScaler -> PCA(35) ->
# global/local shrinkage-blended LogisticRegression, CAL_FRACTION=0.15,
# RANDOM_SEED=42, all leakage controls.
#
# WHAT'S DIFFERENT from the original: scoped to 5 subjects instead of 29
# (for a fast, apples-to-apples slice), and no neural network anywhere in
# this file — there is nothing to "remove," the learned branch was never
# added in the first place.
#
# Usage: modal run run_step4_condition4_raw_tangent_first5.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-raw-tangent-first5")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON   = "/data/results_condition4_raw_tangent_first5.json"
VOLUME_PATH   = "/data"

SFREQ, N_CHANNELS = 250, 62
CAL_FRACTION      = 0.15
RANDOM_SEED       = 42
COV_SHRINKAGE     = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C          = 1.0
SHRINK_GRID       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS   = 3

TARGET_SUBJECTS = ["01", "02", "03", "04", "05"]

# Reference numbers, for the printed comparison at the end
CONDITION4V2_SUBSET_MEAN_ACC   = 0.6599   # original script, same 5 subjects
SPATIAL_ONLY_ABLATION_MEAN_ACC = 0.6041   # this session's bottlenecked-embedding version
DUALBRANCH_PILOT_MEAN_ACC      = 0.5570   # this session's dual-branch (Mamba) pilot

cpu_or_light_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy<2", "scikit-learn==1.4.2", "scipy"
)


@app.function(image=cpu_or_light_image, cpu=4.0, volumes={VOLUME_PATH: volume}, timeout=1800)
def run_raw_tangent_first5():

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-raw-tangent-first5")

    np.random.seed(RANDOM_SEED)

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    log.info(f"X: {X_np.shape} | Subjects total: {len(np.unique(subjects_np))}")
    log.info(f"Running ONLY subjects: {TARGET_SUBJECTS} (no GPU needed for this diagnostic)")

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES — identical to every prior script this session
    # =========================================================================

    def trial_covariances(X, shrinkage=COV_SHRINKAGE):
        Xc = X - X.mean(axis=2, keepdims=True)
        cov = np.einsum("nct,ndt->ncd", Xc, Xc) / (X.shape[2] - 1)
        eye = np.eye(X.shape[1], dtype=cov.dtype)[None, :, :]
        tr = np.trace(cov, axis1=1, axis2=2) / X.shape[1]
        return (1 - shrinkage) * cov + shrinkage * tr[:, None, None] * eye

    def matrix_sqrt_inv_sqrt(mat, eps=1e-8):
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.clip(eigvals, eps, None)
        sv, isv = np.sqrt(eigvals), 1.0 / np.sqrt(eigvals)
        return (eigvecs * sv) @ eigvecs.T, (eigvecs * isv) @ eigvecs.T

    def fit_ea_whitening(X_train28, shrinkage=COV_SHRINKAGE):
        covs = trial_covariances(X_train28, shrinkage)
        _, W = matrix_sqrt_inv_sqrt(covs.mean(axis=0))
        return W

    def apply_ea_whitening_signal(X, W):
        return np.einsum("cd,ndt->nct", W, X)

    def tangent_vectorize(covs, eps=1e-8):
        """RAW tangent vectors — no learned compression, this is the whole point."""
        N, Cc, _ = covs.shape
        out = np.empty((N, Cc * (Cc + 1) // 2), dtype=np.float32)
        iu = np.triu_indices(Cc)
        for n in range(N):
            eigvals, eigvecs = np.linalg.eigh(covs[n])
            eigvals = np.clip(eigvals, eps, None)
            log_mat = (eigvecs * np.log(eigvals)) @ eigvecs.T
            vec = log_mat[iu].copy()
            vec[iu[0] != iu[1]] *= math.sqrt(2.0)
            out[n] = vec
        return out

    # =========================================================================
    # CALIBRATION — identical algorithm to every prior script this session
    # =========================================================================

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"sensitivity": float(sens), "specificity": float(spec),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)), "confusion_matrix": cm.tolist()}

    def linear_predict(coef, intercept, X):
        return ((X @ coef.T + intercept).ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal):
        global_clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_train_pca, y_train)
        local_clf_full = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_cal_pca, y_cal)
        n_splits = max(min(SHRINK_CV_FOLDS, np.bincount(y_cal).min()), 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        shrink_scores = {s: [] for s in SHRINK_GRID}
        for tr_idx, val_idx in skf.split(X_cal_pca, y_cal):
            X_tr, X_val, y_tr, y_val = X_cal_pca[tr_idx], X_cal_pca[val_idx], y_cal[tr_idx], y_cal[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            local_fold = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED).fit(X_tr, y_tr)
            for shrink in SHRINK_GRID:
                coef_b = shrink * local_fold.coef_ + (1 - shrink) * global_clf.coef_
                icpt_b = shrink * local_fold.intercept_ + (1 - shrink) * global_clf.intercept_
                shrink_scores[shrink].append((linear_predict(coef_b, icpt_b, X_val) == y_val).mean())
        mean_scores = {s: (np.mean(v) if v else -1.0) for s, v in shrink_scores.items()}
        best_shrink = max(mean_scores, key=mean_scores.get)
        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_
        return coef_final, icpt_final, best_shrink, global_clf

    # =========================================================================
    # LOSO over subjects 01-05 ONLY — raw tangent vectors, no neural net
    # =========================================================================

    fold_records, all_test_acc = [], []

    for fold_idx, test_sub in enumerate(TARGET_SUBJECTS):
        fold_start = time.time()
        log.info(f"\n{'='*70}\n  FOLD {fold_idx+1}/{len(TARGET_SUBJECTS)} — sub-{test_sub}\n{'='*70}")

        is_holdout = subjects_np == test_sub
        X_train28, y_train28 = X_np[~is_holdout], y_np[~is_holdout]
        X_k, y_k = X_np[is_holdout], y_np[is_holdout]

        mu = X_train28.mean(axis=(0, 2), keepdims=True)
        sd = X_train28.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train28_z = ((X_train28 - mu) / sd).astype(np.float32)
        X_k_z = ((X_k - mu) / sd).astype(np.float32)

        W = fit_ea_whitening(X_train28_z)
        X_train28_aligned = apply_ea_whitening_signal(X_train28_z, W).astype(np.float32)
        X_k_aligned = apply_ea_whitening_signal(X_k_z, W).astype(np.float32)

        tan_train28 = tangent_vectorize(trial_covariances(X_train28_aligned))   # (N, 1953) — RAW
        tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
        log.info(f"  RAW tangent dim: {tan_train28.shape[1]} (no learned compression)")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx, test_idx = next(sss.split(tan_k, y_k))
        tan_cal, y_cal = tan_k[cal_idx], y_k[cal_idx]
        tan_test, y_test = tan_k[test_idx], y_k[test_idx]

        # StandardScaler + PCA fit ONLY on the 28-subject pool — same leakage
        # boundary as every other script this session.
        scaler = StandardScaler()
        tan_train28_z = scaler.fit_transform(tan_train28)
        tan_cal_z = scaler.transform(tan_cal)
        tan_test_z = scaler.transform(tan_test)

        n_components = min(PCA_MAX_COMPONENTS, tan_train28_z.shape[1] - 1, tan_train28_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train28_pca = pca.fit_transform(tan_train28_z)
        X_cal_pca = pca.transform(tan_cal_z)
        X_test_pca = pca.transform(tan_test_z)
        log.info(f"  PCA: {tan_train28_z.shape[1]} -> {n_components} dims "
                 f"(explained var={pca.explained_variance_ratio_.sum()*100:.1f}%)")

        coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal)
        pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        best_test_acc = float((final_preds == y_test).mean())
        metrics = compute_binary_metrics(y_test, final_preds)

        log.info(f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f})")

        fold_records.append({
            "test_subject": test_sub, "raw_tangent_dim": int(tan_train28.shape[1]),
            "pca_components_used": int(n_components), "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": best_test_acc, **metrics,
        })
        all_test_acc.append(best_test_acc)
        log.info(f"  Fold elapsed: {time.time()-fold_start:.1f}s")

    mean_acc = float(np.mean(all_test_acc))
    std_acc = float(np.std(all_test_acc))

    log.info(f"\n{'='*70}\n  RAW TANGENT VECTOR DIAGNOSTIC — subjects 01-05\n{'='*70}")
    log.info(f"  Mean ± Std Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(
        "\n  COMPARISON:\n"
        f"    Original Condition 4v2 script, SAME 5 subjects : {CONDITION4V2_SUBSET_MEAN_ACC*100:.2f}%\n"
        f"    THIS RUN (raw tangent, no bottleneck)           : {mean_acc*100:.2f}%\n"
        f"    Spatial-only ablation (32-dim bottleneck)       : {SPATIAL_ONLY_ABLATION_MEAN_ACC*100:.2f}%\n"
        f"    Dual-branch (Mamba) pilot                       : {DUALBRANCH_PILOT_MEAN_ACC*100:.2f}%\n"
        "    - If THIS RUN lands near 65.99%: confirmed — the learned 32-dim\n"
        "      bottleneck (not dropout, not weight decay, since neither exists\n"
        "      in this script) is what's discarding signal before calibration.\n"
        "    - If THIS RUN is still well below 65.99%: something else differs\n"
        "      from the original script beyond the bottleneck — re-check EA/\n"
        "      tangent-vectorize code for a subtle divergence, since this script\n"
        "      was written to match the original exactly."
    )

    results_payload = {
        "condition": "Condition 4 — RAW TANGENT VECTOR diagnostic (subjects 01-05 only)",
        "is_diagnostic": True, "n_folds": len(TARGET_SUBJECTS),
        "fold_results": fold_records, "mean_accuracy": mean_acc, "std_accuracy": std_acc,
        "reference_condition4v2_subset_mean": CONDITION4V2_SUBSET_MEAN_ACC,
        "reference_spatial_only_ablation_mean": SPATIAL_ONLY_ABLATION_MEAN_ACC,
        "reference_dualbranch_pilot_mean": DUALBRANCH_PILOT_MEAN_ACC,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "output_path": OUTPUT_JSON}


@app.local_entrypoint()
def main():
    print("Condition 4 — RAW TANGENT VECTOR diagnostic (subjects 01-05, CPU-only)")
    print("Reproduces the original Condition 4v2 pipeline exactly, no neural net,")
    print("no dropout, no bottleneck — isolates whether the learned 32-dim\n"
          "compression is what caused the 5.58pp regression.\n")
    result = run_raw_tangent_first5.remote()
    print("\nRESULT:")
    for k, v in result.items():
        print(f"  {k:<20}: {v}")