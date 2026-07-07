# # =============================================================================
# # commands to run :
# # modal deploy generate_figure5_diagnostics.py
# # modal run generate_figure5_diagnostics.py::main
# # generate_figure5_diagnostics.py
# # commands to download:
# # modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.png .
# # modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.pdf .
# #
# # Phase 3.4 — Figure 5: Overall Classifier Diagnostics (29-Subject LOSO CV)
# #   Panel A — Aggregated, row-normalized confusion matrix pooled across all
# #             out-of-fold (OOF) predictions from Leave-One-Subject-Out CV.
# #   Panel B — Per-fold ROC curves (thin, translucent) + bold mean ROC curve
# #             with +/-1 SD band, chance diagonal, and mean AUC in the legend.
# #
# # ⚠️ SCIENTIFIC DISCLOSURE #1 — WHAT THE "CONDITION 4" PIPELINE MEANS HERE:
# #   This script expects /data/tangent_features_ea.npz to already contain
# #   EUCLIDEAN-ALIGNED (EA) RIEMANNIAN TANGENT-SPACE feature vectors (see
# #   Figure 4's disclosures) plus per-trial labels and a per-trial SUBJECT ID
# #   array (required to build the 29-fold Leave-One-Subject-Out splits).
# #
# #   TWO EXECUTION PATHS, CHOSEN AUTOMATICALLY, LOGGED EXPLICITLY:
# #     Path A ("precomputed_tangent_features", expected/default path) —
# #       If the archive already contains a 2D tangent-vector feature matrix,
# #       this script uses it AS-IS. No EA or tangent-mapping is recomputed.
# #     Path B ("raw_covariance_full_pipeline", fallback) —
# #       If the archive instead contains raw per-trial spatial covariance
# #       matrices (a 3D array, shape (n_trials, n_ch, n_ch)) and NOT a
# #       precomputed feature matrix, this script executes the full Condition
# #       4 preprocessing chain from scratch:
# #         1. Euclidean Alignment (He & Wu, 2020): per-subject whitening by
# #            that subject's OWN arithmetic-mean covariance R_s, i.e.
# #            C_aligned = R_s^(-1/2) @ C @ R_s^(-1/2). This uses only that
# #            subject's own trials and no label information, so it cannot
# #            leak test-subject information across the LOSO boundary.
# #         2. Riemannian tangent-space mapping via the MATRIX LOGARITHM taken
# #            at the identity reference (valid because EA drives each
# #            subject's mean covariance toward the identity; this is a
# #            documented simplification of the true Riemannian/Fréchet-mean
# #            reference and is logged as such — it is NOT re-deriving a
# #            geodesic barycenter). Off-diagonal terms are scaled by sqrt(2)
# #            (Pennec-style vectorization), matching Figure 4's convention.
# #       Whichever path runs, the console log states explicitly which one
# #       was taken and why.
# #
# #   CLASSIFIER — "Ledoit-Wolf Shrinkage + Ridge/LDA":
# #   The classifier is scikit-learn's LinearDiscriminantAnalysis with
# #   solver="lsqr" and shrinkage="auto". Per the scikit-learn documentation,
# #   shrinkage="auto" computes the analytic Ledoit-Wolf (2004) shrinkage
# #   estimate of the pooled within-class covariance matrix used by LDA. This
# #   ridge-like regularization of the covariance is why shrinkage-LDA is
# #   often referred to as "Ridge LDA" in the BCI literature (e.g. the MOABB
# #   TSclassifier default) — no separate/custom Ridge regression is fit here.
# #   If your actual Condition 4 pipeline instead specifies
# #   sklearn.linear_model.RidgeClassifier(alpha=...), swap the second step of
# #   the Pipeline built in Section 5 — the LOSO loop, OOF aggregation, and
# #   figure code below are classifier-agnostic and need no other changes.
# #   A StandardScaler is fit INSIDE each LOSO training fold only (28
# #   subjects), never on pooled or held-out data.
# #
# # ⚠️ SCIENTIFIC DISCLOSURE #2 — LOSO EVALUATION, NOT A FIXED N:
# #   The number of LOSO folds equals the number of UNIQUE subject IDs found
# #   in the archive's subject-ID array — never hardcoded to 29. If the
# #   archive contains a different subject count, this script still runs
# #   correctly and simply logs the actual count found (with a warning if it
# #   differs from the 29 expected under the manuscript's stated protocol).
# #   Likewise, pooled N (trial count) is read directly from the data, not
# #   assumed to be 600.
# #
# # LEAKAGE NOTE:
# #   For every fold, the classifier (scaler + shrinkage-LDA) is fit ONLY on
# #   the 28 non-held-out subjects' trials. All confusion-matrix cells, ROC
# #   curves, sensitivity/specificity, and AUC values in Figure 5 are computed
# #   EXCLUSIVELY from each subject's own out-of-fold (OOF) predictions/scores
# #   — a subject's trials are scored only by a model that never saw them
# #   during fitting, at any stage (EA reference, scaler, or classifier).
# #
# # Usage: modal run generate_figure5_diagnostics.py::main
# #        modal run generate_figure5_diagnostics.py::inspect
# # =============================================================================

# import modal

# # =============================================================================
# # SECTION 1: MODAL INFRASTRUCTURE
# # =============================================================================

# app    = modal.App("bci-figure5-classifier-diagnostics")
# volume = modal.Volume.from_name("eeg-data-vol")

# DATA_PATH   = "/data/tangent_features_ea.npz"
# FIGURES_DIR = "/data/figures"

# PNG_MAIN = f"{FIGURES_DIR}/figure5_classifier_diagnostics.png"
# PDF_MAIN = f"{FIGURES_DIR}/figure5_classifier_diagnostics.pdf"

# VOLUME_PATH = "/data"
# RANDOM_SEED = 42
# EXPECTED_N_SUBJECTS = 29  # documented protocol; verified (never assumed) at runtime

# CLASS_LABELS = {0: "False Memory", 1: "True Memory"}

# viz_image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .pip_install(
#         "numpy<2",
#         "scipy",
#         "matplotlib==3.8.4",
#         "scikit-learn==1.4.2",
#     )
# )


# # =============================================================================
# # SECTION 1.5: STANDALONE INSPECTOR
# #   modal run generate_figure5_diagnostics.py::inspect
# # =============================================================================

# @app.function(image=viz_image, volumes={VOLUME_PATH: volume}, timeout=300)
# def inspect_npz():
#     import numpy as np
#     raw = np.load(DATA_PATH, allow_pickle=True)
#     print(f"\nArchive: {DATA_PATH}")
#     print(f"Keys found: {raw.files}\n")
#     for k in raw.files:
#         arr = raw[k]
#         print(f"  '{k}': shape={arr.shape} dtype={arr.dtype}")
#         if arr.ndim <= 1 and arr.size <= 40:
#             print(f"      sample values: {arr[:10]}")
#     for cand in ["subject", "subjects", "subj", "subj_id", "subject_id", "group", "groups"]:
#         if cand in raw.files:
#             uniq = np.unique(raw[cand])
#             print(f"\n  Subject-ID key '{cand}' found -> {len(uniq)} unique subjects.")
#             break
#     return raw.files


# @app.local_entrypoint(name="inspect")
# def inspect_entrypoint():
#     keys = inspect_npz.remote()
#     print(f"\nKeys in {DATA_PATH}: {keys}")


# # =============================================================================
# # SECTION 2: MODAL FUNCTION
# # =============================================================================

# @app.function(
#     image=viz_image,
#     cpu=4.0,
#     volumes={VOLUME_PATH: volume},
#     timeout=3600,
#     memory=8192,
# )
# def generate_figure5():

#     import os
#     import logging

#     import numpy as np
#     from scipy.linalg import eigh, logm

#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

#     logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
#     log = logging.getLogger("figure5-diagnostics")

#     np.random.seed(RANDOM_SEED)
#     os.makedirs(FIGURES_DIR, exist_ok=True)

#     # =========================================================================
#     # SECTION 3: DATA LOADING + KEY DETECTION
#     # =========================================================================
#     log.info(f"Loading {DATA_PATH} ...")
#     raw = np.load(DATA_PATH, allow_pickle=True)
#     log.info(f"Archive keys found: {raw.files}")

#     def _pick_key(candidates, purpose, required=True):
#         for c in candidates:
#             if c in raw.files:
#                 return c
#         if required:
#             raise KeyError(
#                 f"Could not find a key for '{purpose}' in {DATA_PATH}. "
#                 f"Tried {candidates}, but archive only contains {raw.files}."
#             )
#         return None

#     Y_KEY    = _pick_key(["y", "labels", "label", "Y", "targets"], "labels")
#     SUBJ_KEY = _pick_key(
#         ["subject", "subjects", "subj", "subj_id", "subject_id", "group", "groups"],
#         "per-trial subject IDs (required for Leave-One-Subject-Out CV)",
#     )
#     COV_KEY = _pick_key(["covariances", "covariance", "cov", "X_cov"], "raw covariance", required=False)
#     has_raw_cov = COV_KEY is not None and raw[COV_KEY].ndim == 3
#     X_KEY = _pick_key(
#         ["X_ts", "X", "features", "tangent_features", "tangent_vectors", "data", "X_tangent"],
#         "precomputed tangent-space feature matrix", required=not has_raw_cov,
#     )

#     log.info(f"Using keys -> features: '{X_KEY}', labels: '{Y_KEY}', "
#              f"subject IDs: '{SUBJ_KEY}', covariance: '{COV_KEY}'")

#     y_np    = raw[Y_KEY].astype(np.int64)
#     subj_np = raw[SUBJ_KEY]
#     n_classes = int(y_np.max()) + 1
#     assert n_classes == 2, "This diagnostics pipeline assumes binary classification (True/False Memory)."

#     # =========================================================================
#     # SECTION 4: FEATURE PATH SELECTION (Disclosure #1)
#     # =========================================================================
#     if has_raw_cov:
#         log.info("Raw covariance array found -> executing FULL Condition 4 pipeline "
#                   "(Euclidean Alignment + tangent-space mapping) from scratch. "
#                   "[path = raw_covariance_full_pipeline]")
#         cov = raw[COV_KEY].astype(np.float64)
#         n_trials_raw, n_ch, _ = cov.shape

#         def sympd_power(mat, power):
#             """Symmetric-PD matrix power via eigendecomposition (numerically
#             stable for the +/-1/2 powers EA and tangent-space mapping need)."""
#             w, v = eigh(mat)
#             w = np.clip(w, 1e-12, None)
#             return (v * (w ** power)) @ v.T

#         # --- Step 1: Euclidean Alignment, per subject, using ONLY that
#         # subject's own trials (no labels involved -> no LOSO leakage). ---
#         aligned = np.empty_like(cov)
#         for s in np.unique(subj_np):
#             idx = np.where(subj_np == s)[0]
#             R_s = cov[idx].mean(axis=0)
#             R_inv_sqrt = sympd_power(R_s, -0.5)
#             for i in idx:
#                 aligned[i] = R_inv_sqrt @ cov[i] @ R_inv_sqrt.T
#         log.info("Euclidean Alignment complete (per-subject whitening by own mean covariance).")

#         # --- Step 2: tangent-space mapping at the identity reference
#         # (documented simplification; see Disclosure #1), Pennec-scaled. ---
#         iu = np.triu_indices(n_ch)
#         offdiag_mask_full = ~np.eye(n_ch, dtype=bool)
#         X_np = np.empty((n_trials_raw, len(iu[0])), dtype=np.float64)
#         for t in range(n_trials_raw):
#             L = logm(aligned[t]).real
#             L[offdiag_mask_full] *= np.sqrt(2.0)
#             X_np[t] = L[iu]
#         used_feature_path = "raw_covariance_full_pipeline"
#         log.info(f"Tangent-space vectorization complete: X shape={X_np.shape}.")
#     else:
#         log.info(f"Precomputed tangent-space feature matrix found under key '{X_KEY}' -> "
#                   f"using it AS-IS. No EA or tangent-mapping recomputed. "
#                   f"[path = precomputed_tangent_features]")
#         X_np = raw[X_KEY].astype(np.float64)
#         if X_np.ndim != 2:
#             X_np = X_np.reshape(X_np.shape[0], -1)
#         used_feature_path = "precomputed_tangent_features"

#     n_trials = X_np.shape[0]
#     assert len(y_np) == n_trials, "Label count does not match feature matrix trial count."
#     assert len(subj_np) == n_trials, "Subject-ID count does not match feature matrix trial count."
#     log.info(f"Final feature matrix for classification: X shape={X_np.shape} "
#              f"[feature path = {used_feature_path}]")

#     # =========================================================================
#     # SECTION 5: LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION (Disclosure #2)
#     # =========================================================================
#     unique_subjects = np.unique(subj_np)
#     n_subjects = len(unique_subjects)
#     if n_subjects != EXPECTED_N_SUBJECTS:
#         log.warning(f"Found {n_subjects} unique subjects; manuscript protocol "
#                      f"expects {EXPECTED_N_SUBJECTS}. Proceeding with the ACTUAL "
#                      f"count found in the data (never hardcoded).")
#     else:
#         log.info(f"Confirmed {n_subjects} unique subjects, matching the documented protocol.")

#     oof_pred  = np.full(n_trials, -1, dtype=np.int64)
#     oof_score = np.full(n_trials, np.nan, dtype=np.float64)

#     per_subject_rows = []
#     fold_fprs, fold_tprs, fold_aucs = [], [], []
#     mean_fpr_grid = np.linspace(0.0, 1.0, 200)

#     log.info(f"\nRunning {n_subjects}-fold Leave-One-Subject-Out CV "
#               f"(fold-wise scaler + shrinkage-LDA)...")

#     for fold_i, held_out in enumerate(unique_subjects, start=1):
#         test_mask  = subj_np == held_out
#         train_mask = ~test_mask

#         X_train, y_train = X_np[train_mask], y_np[train_mask]
#         X_test,  y_test  = X_np[test_mask],  y_np[test_mask]

#         # Fit ONLY on the 28 non-held-out subjects -- both the scaler and the
#         # classifier are refit from scratch every fold (zero leakage).
#         pipeline = Pipeline([
#             ("scaler", StandardScaler()),
#             ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
#         ])
#         pipeline.fit(X_train, y_train)

#         pred_k  = pipeline.predict(X_test)
#         score_k = pipeline.decision_function(X_test)

#         oof_pred[test_mask]  = pred_k
#         oof_score[test_mask] = score_k

#         acc_k = float(np.mean(pred_k == y_test))
#         cm_k = confusion_matrix(y_test, pred_k, labels=[0, 1])
#         tn_k, fp_k, fn_k, tp_k = cm_k.ravel()
#         sens_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else np.nan
#         spec_k = tn_k / (tn_k + fp_k) if (tn_k + fp_k) > 0 else np.nan

#         auc_k = np.nan
#         if len(np.unique(y_test)) == 2:
#             fpr_k, tpr_k, _ = roc_curve(y_test, score_k)
#             auc_k = auc(fpr_k, tpr_k)
#             fold_fprs.append(fpr_k)
#             fold_tprs.append(tpr_k)
#             fold_aucs.append(auc_k)
#         else:
#             log.warning(f"  Fold {fold_i:>2} (subject {held_out}): only one class present "
#                          f"in held-out trials -> ROC/AUC skipped for this fold "
#                          f"(still included in pooled confusion matrix).")

#         per_subject_rows.append({
#             "subject": held_out, "n_trials": int(test_mask.sum()),
#             "accuracy": acc_k, "sensitivity": sens_k, "specificity": spec_k,
#             "auc": auc_k,
#         })

#         log.info(f"  Fold {fold_i:>2}/{n_subjects} | subject={held_out!s:<6} "
#                   f"n={int(test_mask.sum()):>3} | acc={acc_k:.3f} | "
#                   f"sens={sens_k:.3f} | spec={spec_k:.3f} | "
#                   f"AUC={'n/a' if np.isnan(auc_k) else f'{auc_k:.3f}'}")

#     assert np.all(oof_pred >= 0), "Every trial must receive an out-of-fold prediction."
#     assert len(fold_fprs) > 0, ("No fold produced a valid ROC curve (a class was missing from "
#                                   "every held-out subject) -- cannot compute a mean ROC curve.")
#     log.info(f"\nLOSO loop complete. Pooled OOF trial count: {n_trials}.")

#     # =========================================================================
#     # SECTION 6: POOLED (OOF) CONFUSION MATRIX + OVERALL METRICS
#     # =========================================================================
#     cm_counts = confusion_matrix(y_np, oof_pred, labels=[0, 1])
#     tn, fp, fn, tp = cm_counts.ravel()

#     overall_acc  = (tp + tn) / cm_counts.sum()
#     overall_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # True Memory recall
#     overall_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # False Memory recall

#     cm_row_norm_pct = cm_counts.astype(np.float64) / cm_counts.sum(axis=1, keepdims=True) * 100.0

#     valid_auc_folds = [a for a in fold_aucs if not np.isnan(a)]
#     mean_auc = float(np.mean(valid_auc_folds))
#     std_auc  = float(np.std(valid_auc_folds))
#     # Pooled AUC (single curve over ALL OOF scores at once) is a distinct,
#     # complementary quantity -- logged separately, never conflated with the
#     # across-fold mean +/- SD reported in the figure legend.
#     pooled_auc = float(roc_auc_score(y_np, oof_score))

#     log.info("\n" + "=" * 70)
#     log.info("  POOLED OUT-OF-FOLD DIAGNOSTICS (LOSO, ALL FOLDS)")
#     log.info("=" * 70)
#     log.info(f"  Pooled N trials         : {n_trials}")
#     log.info(f"  Pooled accuracy         : {overall_acc:.4f}")
#     log.info(f"  Sensitivity (TPR, True Memory recall)  : {overall_sens:.4f}")
#     log.info(f"  Specificity (TNR, False Memory recall) : {overall_spec:.4f}")
#     log.info(f"  Mean per-fold ROC-AUC   : {mean_auc:.4f} +/- {std_auc:.4f}  "
#               f"(across {len(valid_auc_folds)}/{n_subjects} folds with both classes present)")
#     log.info(f"  Pooled ROC-AUC (single curve over all OOF scores): {pooled_auc:.4f}")
#     log.info("  Confusion matrix (rows=true, cols=pred, order=[False Memory(0), True Memory(1)]):")
#     log.info(f"    {cm_counts}")
#     log.info("=" * 70 + "\n")

#     # =========================================================================
#     # SECTION 7: MEAN ROC CURVE + +/-1 SD BAND (STANDARD FOLD-INTERPOLATION)
#     # =========================================================================
#     interp_tprs = []
#     for fpr_k, tpr_k in zip(fold_fprs, fold_tprs):
#         tpr_interp = np.interp(mean_fpr_grid, fpr_k, tpr_k)
#         tpr_interp[0] = 0.0
#         interp_tprs.append(tpr_interp)
#     interp_tprs = np.array(interp_tprs)

#     mean_tpr = interp_tprs.mean(axis=0)
#     mean_tpr[-1] = 1.0
#     std_tpr = interp_tprs.std(axis=0)
#     tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
#     tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)

#     # =========================================================================
#     # SECTION 8: FIGURE ASSEMBLY (2-PANEL, Q1 EDITORIAL LAYOUT)
#     # =========================================================================
#     log.info("Rendering 2-panel publication diagnostics figure (Matplotlib)...")
#     plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

#     fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6.5))

#     # --- Panel A: pooled, row-normalized confusion matrix ---
#     cmap_a = plt.get_cmap("Blues")
#     im_a = ax_a.imshow(cm_row_norm_pct, cmap=cmap_a, vmin=0, vmax=100, aspect="equal")

#     class_names = [f"{CLASS_LABELS[0]}\n(Class 0)", f"{CLASS_LABELS[1]}\n(Class 1)"]
#     ax_a.set_xticks([0, 1]); ax_a.set_xticklabels(class_names, fontsize=10)
#     ax_a.set_yticks([0, 1]); ax_a.set_yticklabels(class_names, fontsize=10)
#     ax_a.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
#     ax_a.set_ylabel("True Label", fontsize=11, labelpad=10)
#     ax_a.set_title("A. Aggregated Confusion Matrix\n(LOSO, Out-of-Fold, Row-Normalized)",
#                     fontsize=12, fontweight="bold", pad=14)

#     for r in range(2):
#         for c in range(2):
#             pct = cm_row_norm_pct[r, c]
#             count = cm_counts[r, c]
#             text_color = "white" if pct > 55 else "black"
#             ax_a.text(c, r - 0.12, f"{pct:.1f}%", ha="center", va="center",
#                        fontsize=17, fontweight="bold", color=text_color, zorder=5)
#             ax_a.text(c, r + 0.18, f"(N={count})", ha="center", va="center",
#                        fontsize=10, color=text_color, zorder=5)

#     ax_a.set_xlim(-0.5, 1.5); ax_a.set_ylim(1.5, -0.5)
#     for spine in ax_a.spines.values():
#         spine.set_visible(False)
#     ax_a.tick_params(length=0)

#     divider = make_axes_locatable(ax_a)
#     cax_a = divider.append_axes("right", size="5%", pad=0.12)
#     cbar_a = fig.colorbar(im_a, cax=cax_a)
#     cbar_a.set_label("Row-normalized % of true class", fontsize=9.5, labelpad=8)
#     cbar_a.ax.tick_params(labelsize=8.5)

#     # --- Panel B: ROC curves (per-fold thin, mean bold, +/-1 SD band, chance) ---
#     for fpr_k, tpr_k in zip(fold_fprs, fold_tprs):
#         ax_b.plot(fpr_k, tpr_k, color="#94A3B8", alpha=0.25, linewidth=1.0, zorder=1)

#     ax_b.fill_between(mean_fpr_grid, tpr_lower, tpr_upper, color="#1E3A8A",
#                         alpha=0.15, zorder=2, label="\u00b1 1 SD (across folds)")
#     ax_b.plot(mean_fpr_grid, mean_tpr, color="#1E3A8A", linewidth=2.8, zorder=4,
#                label=f"Mean ROC (AUC = {mean_auc:.2f} \u00b1 {std_auc:.2f})")
#     ax_b.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.2, zorder=1,
#                label="Chance (AUC = 0.50)")

#     subj_proxy = Line2D([0], [0], color="#94A3B8", alpha=0.6, linewidth=1.4,
#                           label=f"Individual subjects (n={len(fold_fprs)}, OOF)")
#     handles, labels_ = ax_b.get_legend_handles_labels()
#     handles.append(subj_proxy)
#     labels_.append(subj_proxy.get_label())

#     ax_b.set_xlim(0.0, 1.0); ax_b.set_ylim(0.0, 1.02)
#     ax_b.set_aspect("equal", adjustable="box")
#     ax_b.set_xlabel("False Positive Rate (1 \u2212 Specificity)", fontsize=11)
#     ax_b.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
#     ax_b.set_title(f"B. ROC Curve Analysis\n(Leave-One-Subject-Out, N={n_subjects})",
#                     fontsize=12, fontweight="bold", pad=14)
#     ax_b.legend(handles=handles, labels=labels_, loc="lower right", fontsize=8.8, frameon=True)
#     ax_b.grid(alpha=0.2, linewidth=0.5)

#     fig.suptitle(
#         "Figure 5: Overall Classifier Diagnostics Across 29 Subjects "
#         "(Aggregated Confusion Matrix & ROC-AUC)",
#         fontsize=14.5, fontweight="bold", y=1.04,
#     )
#     fig.text(
#         0.5, 0.975,
#         f"Pipeline: EA-aligned tangent features \u2192 fold-wise StandardScaler \u2192 "
#         f"Shrinkage-LDA (Ledoit-Wolf)  |  feature path: {used_feature_path}",
#         ha="center", va="top", fontsize=9.5, color="#333333",
#     )
#     fig.text(
#         0.5, -0.04,
#         "All confusion-matrix cells and ROC curves reflect strictly out-of-fold (OOF) predictions pooled across\n"
#         f"{n_subjects}-fold Leave-One-Subject-Out cross-validation \u2014 no held-out subject's trials were seen during model fitting at any stage.",
#         ha="center", va="top", fontsize=9.5, fontweight="normal", color="#000000",
#     )

#     fig.subplots_adjust(left=0.05, right=0.97, bottom=0.16, top=0.82, wspace=0.32)

#     log.info(f"Saving 2-panel PNG (600 DPI) -> {PNG_MAIN}")
#     fig.savefig(PNG_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
#     log.info(f"Saving 2-panel PDF -> {PDF_MAIN}")
#     fig.savefig(PDF_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
#     plt.close(fig)

#     # =========================================================================
#     # SECTION 9: PERSISTENCE + LOGGING
#     # =========================================================================
#     volume.commit()
#     log.info("volume.commit() complete.")

#     file_report = {}
#     for label, path in [("PNG_MAIN", PNG_MAIN), ("PDF_MAIN", PDF_MAIN)]:
#         exists = os.path.exists(path)
#         size_kb = os.path.getsize(path) / 1024 if exists else 0.0
#         file_report[label] = {"path": path, "exists": exists, "size_kb": round(size_kb, 1)}
#         status = "OK" if exists else "MISSING"
#         log.info(f"  [{status}] {label:<10} -> {path}  ({size_kb:.1f} KB)")

#     log.info("\n" + "=" * 70)
#     log.info("  FIGURE 5 (CLASSIFIER DIAGNOSTICS) EXPORT COMPLETE")
#     log.info("=" * 70)
#     log.info(f"  Subjects (LOSO folds)  : {n_subjects}")
#     log.info(f"  Pooled trials (N)      : {n_trials}")
#     log.info(f"  Feature path           : {used_feature_path}")
#     log.info(f"  Sensitivity            : {overall_sens:.4f}")
#     log.info(f"  Specificity            : {overall_spec:.4f}")
#     log.info(f"  Accuracy               : {overall_acc:.4f}")
#     log.info(f"  Mean ROC-AUC (per-fold): {mean_auc:.4f} +/- {std_auc:.4f}")
#     log.info(f"  Pooled ROC-AUC         : {pooled_auc:.4f}")
#     log.info("=" * 70 + "\n")

#     return {
#         "n_subjects": n_subjects,
#         "n_trials": n_trials,
#         "used_feature_path": used_feature_path,
#         "overall_accuracy": overall_acc,
#         "overall_sensitivity": overall_sens,
#         "overall_specificity": overall_spec,
#         "mean_auc": mean_auc,
#         "std_auc": std_auc,
#         "pooled_auc": pooled_auc,
#         "per_subject": per_subject_rows,
#         "files": file_report,
#     }


# # =============================================================================
# # SECTION 10: LOCAL ENTRYPOINT
# # =============================================================================

# @app.local_entrypoint()
# def main():
#     print("\n" + "=" * 70)
#     print("  Figure 5 \u2014 Classifier Diagnostics (29-Subject LOSO CV)")
#     print("  (Aggregated Confusion Matrix + Mean ROC-AUC Curve)")
#     print("=" * 70 + "\n")

#     results = generate_figure5.remote()

#     print("\n" + "=" * 70)
#     print("  FINAL RESULTS")
#     print("=" * 70)
#     for key, val in results.items():
#         if key == "per_subject":
#             print(f"  {key:<20} : ({len(val)} subjects \u2014 see fold-by-fold logs above)")
#         else:
#             print(f"  {key:<20} : {val}")
#     print("=" * 70)
#     print(
#         "\n  Figure 5 export complete."
#         "\n  2-panel PNG/PDF committed to eeg-data-vol under /figures/."
#         "\n\n  Download with:"
#         "\n    modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.png ."
#         "\n    modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.pdf .\n"
#     )






































# =============================================================================
# commands to run :
# modal deploy generate_figure5_diagnostics.py
# modal run generate_figure5_diagnostics.py::main
# generate_figure5_diagnostics.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.png .
# modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.pdf .
#
# Phase 3.4 — Figure 5: Overall Classifier Diagnostics (29-Subject LOSO CV)
#   Panel A — Aggregated, row-normalized confusion matrix pooled across all
#             out-of-fold (OOF) predictions from Leave-One-Subject-Out CV.
#   Panel B — Per-fold ROC curves (visible, slate blue) + bold mean ROC curve
#             with chance diagonal and mean AUC in the legend.
#
# ⚠️ CORRECTED DATA SOURCE (this revision):
#   A prior revision of this script RE-FIT a scratch StandardScaler ->
#   shrinkage-LDA pipeline from raw/tangent features. That pipeline is NOT
#   the manuscript's locked Condition 4 model (it lacks the subspace +
#   calibration steps), so its OOF accuracy/AUC (~0.51-0.55 acc, ~0.58 AUC)
#   silently diverged from the true, already-computed Condition 4 result.
#
#   This script no longer re-fits anything. It loads the ALREADY-COMPUTED,
#   already-locked out-of-fold predictions/scores for Condition 4 directly
#   from:
#       /data/results_condition4_subspace_calibrated.json
#   Every number in Figure 5 (confusion matrix, sensitivity, specificity,
#   per-fold ROC curves, mean AUC) is derived exclusively from that JSON's
#   `fold_results` list -- nothing is recomputed, refit, or assumed. If the
#   JSON's own summary stats (`mean_accuracy`/`std_accuracy`) disagree with
#   what this script computes from `fold_results`, that disagreement is
#   logged as an explicit WARNING rather than silently resolved either way,
#   so a stale/mismatched JSON is caught, not papered over.
#
#   CONTINUOUS SCORE SOURCING (for ROC/AUC):
#   For each fold dict, this script looks for a continuous score under,
#   in order: "y_score", "decision_scores", "probabilities", "scores".
#   If none of those keys exist for a given fold, ROC/AUC for that fold
#   falls back to using the discrete y_pred array as a score proxy, and a
#   terminal NOTICE is logged naming exactly which folds used the fallback
#   (a proxy ROC curve from binary predictions is a degenerate step
#   function, not a real trade-off curve, so this is disclosed rather than
#   silently blended in as if it were a genuine probability-based curve).
#
# LEAKAGE NOTE (inherited from Condition 4's own reported protocol):
#   This script does not fit any model and cannot introduce leakage itself.
#   It trusts that /data/results_condition4_subspace_calibrated.json was
#   itself produced under strict Leave-One-Subject-Out out-of-fold scoring
#   (per the Condition 4 pipeline that generated it, as already used for
#   Figure 3). This script's only responsibility is to pool and plot those
#   already-produced OOF numbers faithfully.
#
# Usage: modal run generate_figure5_diagnostics.py::main
#        modal run generate_figure5_diagnostics.py::inspect
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-figure5-classifier-diagnostics")
volume = modal.Volume.from_name("eeg-data-vol")

RESULTS_PATH = "/data/results_condition4_subspace_calibrated.json"
FIGURES_DIR  = "/data/figures"

PNG_MAIN = f"{FIGURES_DIR}/figure5_classifier_diagnostics.png"
PDF_MAIN = f"{FIGURES_DIR}/figure5_classifier_diagnostics.pdf"

VOLUME_PATH = "/data"
EXPECTED_N_SUBJECTS = 29  # documented protocol; verified (never assumed) at runtime

CLASS_LABELS = {0: "False Memory", 1: "True Memory"}

# Continuous-score key fallback order, checked per fold.
SCORE_KEYS = ["y_score", "decision_scores", "probabilities", "scores"]

viz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "scipy",
        "matplotlib==3.8.4",
        "scikit-learn==1.4.2",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run generate_figure5_diagnostics.py::inspect
# =============================================================================

@app.function(image=viz_image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_results():
    import json
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    print(f"\nResults file: {RESULTS_PATH}")
    print(f"Top-level keys: {list(results.keys())}\n")

    fold_results = results.get("fold_results", [])
    print(f"Number of fold_results entries: {len(fold_results)}")
    if fold_results:
        rec0 = fold_results[0]
        print(f"Keys inside a single fold dict: {list(rec0.keys())}")
        for k in SCORE_KEYS:
            if k in rec0:
                print(f"  -> continuous score key found: '{k}'")
                break
        else:
            print("  -> NO continuous score key found in first fold; "
                  "will fall back to y_pred as score proxy if this holds across folds.")

    if "mean_accuracy" in results:
        print(f"\nJSON-reported mean_accuracy: {results['mean_accuracy']}")
    if "std_accuracy" in results:
        print(f"JSON-reported std_accuracy : {results['std_accuracy']}")

    return {"n_folds": len(fold_results), "top_level_keys": list(results.keys())}


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    info = inspect_results.remote()
    print(f"\nSummary: {info}")


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================

@app.function(
    image=viz_image,
    cpu=4.0,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    memory=8192,
)
def generate_figure5():

    import os
    import json
    import logging

    import numpy as np

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("figure5-diagnostics")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =========================================================================
    # SECTION 3: LOAD LOCKED CONDITION 4 OOF RESULTS (no refitting, no recompute)
    # =========================================================================
    log.info(f"Loading locked Condition 4 OOF results from {RESULTS_PATH} ...")
    with open(RESULTS_PATH, "r") as f:
        results_json = json.load(f)

    fold_results = results_json.get("fold_results")
    if not fold_results:
        raise KeyError(
            f"'fold_results' missing or empty in {RESULTS_PATH}. "
            f"Top-level keys found: {list(results_json.keys())}"
        )

    n_subjects = len(fold_results)
    if n_subjects != EXPECTED_N_SUBJECTS:
        log.warning(f"Found {n_subjects} fold_results entries; manuscript protocol "
                     f"expects {EXPECTED_N_SUBJECTS}. Proceeding with the ACTUAL "
                     f"count found in the JSON (never hardcoded).")
    else:
        log.info(f"Confirmed {n_subjects} fold entries, matching the documented protocol.")

    # =========================================================================
    # SECTION 4: AGGREGATE OOF PREDICTIONS + SCORES ACROSS ALL LOGGED FOLDS
    # =========================================================================
    all_y_true, all_y_pred, all_y_score = [], [], []
    fold_fprs, fold_tprs, fold_aucs = [], [], []
    per_subject_rows = []
    fallback_score_folds = []
    mean_fpr_grid = np.linspace(0.0, 1.0, 200)

    for fold_i, rec in enumerate(fold_results, start=1):
        subj = rec.get("test_subject", f"fold_{fold_i}")
        y_true_k = np.asarray(rec["y_true"], dtype=np.int64)
        y_pred_k = np.asarray(rec["y_pred"], dtype=np.int64)

        score_key_used = None
        y_score_k = None
        for k in SCORE_KEYS:
            if rec.get(k) is not None:
                y_score_k = np.asarray(rec[k], dtype=np.float64)
                score_key_used = k
                break

        if y_score_k is None:
            y_score_k = y_pred_k.astype(np.float64)
            fallback_score_folds.append(str(subj))
            log.info(f"  Fold {fold_i:>2} (subject {subj}): no continuous score key found "
                      f"({SCORE_KEYS}) -> falling back to y_pred as score proxy for ROC/AUC.")
        else:
            log.info(f"  Fold {fold_i:>2} (subject {subj}): using continuous score key '{score_key_used}'.")

        all_y_true.append(y_true_k)
        all_y_pred.append(y_pred_k)
        all_y_score.append(y_score_k)

        acc_k = float(np.mean(y_pred_k == y_true_k))
        cm_k = confusion_matrix(y_true_k, y_pred_k, labels=[0, 1])
        tn_k, fp_k, fn_k, tp_k = cm_k.ravel()
        sens_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else np.nan
        spec_k = tn_k / (tn_k + fp_k) if (tn_k + fp_k) > 0 else np.nan

        auc_k = np.nan
        if len(np.unique(y_true_k)) == 2:
            fpr_k, tpr_k, _ = roc_curve(y_true_k, y_score_k)
            auc_k = auc(fpr_k, tpr_k)
            fold_fprs.append(fpr_k)
            fold_tprs.append(tpr_k)
            fold_aucs.append(auc_k)
        else:
            log.warning(f"  Fold {fold_i:>2} (subject {subj}): only one class present "
                         f"in held-out trials -> ROC/AUC skipped for this fold "
                         f"(still included in pooled confusion matrix).")

        per_subject_rows.append({
            "subject": subj, "n_trials": int(len(y_true_k)),
            "accuracy": acc_k, "sensitivity": sens_k, "specificity": spec_k,
            "auc": auc_k, "score_source": score_key_used or "y_pred_fallback",
        })

        log.info(f"  Fold {fold_i:>2}/{n_subjects} | subject={subj!s:<6} "
                  f"n={len(y_true_k):>3} | acc={acc_k:.3f} | "
                  f"sens={sens_k:.3f} | spec={spec_k:.3f} | "
                  f"AUC={'n/a' if np.isnan(auc_k) else f'{auc_k:.3f}'}")

    if fallback_score_folds:
        log.warning(f"NOTICE: {len(fallback_score_folds)}/{n_subjects} folds had no continuous "
                     f"score field and used discrete y_pred as a score proxy for ROC/AUC "
                     f"(step-function curve, not a true probability trade-off): "
                     f"subjects {fallback_score_folds}")

    y_np       = np.concatenate(all_y_true)
    oof_pred   = np.concatenate(all_y_pred)
    oof_score  = np.concatenate(all_y_score)
    n_trials   = len(y_np)

    assert len(oof_pred) == n_trials and len(oof_score) == n_trials, \
        "Mismatched lengths when pooling y_true/y_pred/scores across folds."
    assert len(fold_fprs) > 0, ("No fold produced a valid ROC curve (a class was missing from "
                                  "every held-out subject) -- cannot compute a mean ROC curve.")

    log.info(f"\nAggregation complete. Pooled OOF trial count: {n_trials} across {n_subjects} folds.")

    # =========================================================================
    # SECTION 5: POOLED (OOF) CONFUSION MATRIX + OVERALL METRICS
    # =========================================================================
    cm_counts = confusion_matrix(y_np, oof_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_counts.ravel()

    overall_acc  = (tp + tn) / cm_counts.sum()
    overall_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # True Memory recall
    overall_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # False Memory recall

    cm_row_norm_pct = cm_counts.astype(np.float64) / cm_counts.sum(axis=1, keepdims=True) * 100.0

    valid_auc_folds = [a for a in fold_aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid_auc_folds))
    std_auc  = float(np.std(valid_auc_folds))
    pooled_auc = float(roc_auc_score(y_np, oof_score))

    # Cross-check against the JSON's own self-reported summary stats, if present.
    json_mean_acc = results_json.get("mean_accuracy")
    json_std_acc  = results_json.get("std_accuracy")
    if json_mean_acc is not None:
        diff = abs(float(json_mean_acc) - overall_acc)
        if diff > 0.01:
            log.warning(f"MISMATCH: JSON-reported mean_accuracy={json_mean_acc:.4f} differs from "
                         f"accuracy computed here from fold_results ({overall_acc:.4f}) by "
                         f"{diff:.4f} (>1%). Flagging for manual review -- not auto-resolved.")
        else:
            log.info(f"JSON-reported mean_accuracy ({json_mean_acc:.4f}) matches recomputed "
                      f"accuracy ({overall_acc:.4f}) within tolerance.")

    log.info("\n" + "=" * 70)
    log.info("  POOLED OUT-OF-FOLD DIAGNOSTICS (CONDITION 4, LOCKED RESULTS)")
    log.info("=" * 70)
    log.info(f"  Pooled N trials         : {n_trials}")
    log.info(f"  Pooled accuracy         : {overall_acc:.4f}")
    log.info(f"  Sensitivity (TPR, True Memory recall)  : {overall_sens:.4f}")
    log.info(f"  Specificity (TNR, False Memory recall) : {overall_spec:.4f}")
    log.info(f"  Mean per-fold ROC-AUC   : {mean_auc:.4f} +/- {std_auc:.4f}  "
              f"(across {len(valid_auc_folds)}/{n_subjects} folds with both classes present)")
    log.info(f"  Pooled ROC-AUC (single curve over all OOF scores): {pooled_auc:.4f}")
    log.info("  Confusion matrix (rows=true, cols=pred, order=[False Memory(0), True Memory(1)]):")
    log.info(f"    {cm_counts}")
    log.info("=" * 70 + "\n")

    # =========================================================================
    # SECTION 6: MEAN ROC CURVE + +/-1 SD BAND (STANDARD FOLD-INTERPOLATION)
    # =========================================================================
    interp_tprs = []
    for fpr_k, tpr_k in zip(fold_fprs, fold_tprs):
        tpr_interp = np.interp(mean_fpr_grid, fpr_k, tpr_k)
        tpr_interp[0] = 0.0
        interp_tprs.append(tpr_interp)
    interp_tprs = np.array(interp_tprs)

    mean_tpr = interp_tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = interp_tprs.std(axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)

    # =========================================================================
    # SECTION 7: FIGURE ASSEMBLY (2-PANEL, Q1 EDITORIAL LAYOUT)
    # =========================================================================
    log.info("Rendering 2-panel publication diagnostics figure (Matplotlib)...")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- Panel A: pooled, row-normalized confusion matrix ---
    cmap_a = plt.get_cmap("Blues")
    im_a = ax_a.imshow(cm_row_norm_pct, cmap=cmap_a, vmin=0, vmax=100, aspect="equal")

    class_names = [f"{CLASS_LABELS[0]}\n(Class 0)", f"{CLASS_LABELS[1]}\n(Class 1)"]
    ax_a.set_xticks([0, 1]); ax_a.set_xticklabels(class_names, fontsize=10)
    ax_a.set_yticks([0, 1]); ax_a.set_yticklabels(class_names, fontsize=10)
    ax_a.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax_a.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax_a.set_title("A. Aggregated Confusion Matrix\n(Condition 4, Out-of-Fold, Row-Normalized)",
                    fontsize=12, fontweight="bold", pad=14)

    for r in range(2):
        for c in range(2):
            pct = cm_row_norm_pct[r, c]
            count = cm_counts[r, c]
            text_color = "white" if pct > 55 else "black"
            ax_a.text(c, r - 0.12, f"{pct:.1f}%", ha="center", va="center",
                       fontsize=17, fontweight="bold", color=text_color, zorder=5)
            ax_a.text(c, r + 0.18, f"(N={count})", ha="center", va="center",
                       fontsize=10, color=text_color, zorder=5)

    ax_a.set_xlim(-0.5, 1.5); ax_a.set_ylim(1.5, -0.5)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.tick_params(length=0)

    divider = make_axes_locatable(ax_a)
    cax_a = divider.append_axes("right", size="5%", pad=0.12)
    cbar_a = fig.colorbar(im_a, cax=cax_a)
    cbar_a.set_label("Row-normalized % of true class", fontsize=9.5, labelpad=8)
    cbar_a.ax.tick_params(labelsize=8.5)

    # --- Panel B: ROC curves (per-fold visible slate blue, mean bold navy, chance) ---
    for fpr_k, tpr_k in zip(fold_fprs, fold_tprs):
        ax_b.plot(fpr_k, tpr_k, color="#3B82F6", alpha=0.32, linewidth=1.0, zorder=1)

    ax_b.plot(mean_fpr_grid, mean_tpr, color="#1E3A8A", linewidth=3.0, zorder=4,
               label=f"Mean ROC (AUC = {mean_auc:.2f} \u00b1 {std_auc:.2f})")
    ax_b.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.2, zorder=1,
               label="Chance (AUC = 0.50)")

    subj_proxy = Line2D([0], [0], color="#3B82F6", alpha=0.6, linewidth=1.4,
                          label=f"Individual subjects (n={len(fold_fprs)}, OOF)")
    handles, labels_ = ax_b.get_legend_handles_labels()
    handles.append(subj_proxy)
    labels_.append(subj_proxy.get_label())

    ax_b.set_xlim(0.0, 1.0); ax_b.set_ylim(0.0, 1.02)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("False Positive Rate (1 \u2212 Specificity)", fontsize=11)
    ax_b.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax_b.set_title(f"B. ROC Curve Analysis\n(Condition 4, Leave-One-Subject-Out, N={n_subjects})",
                    fontsize=12, fontweight="bold", pad=14)
    ax_b.legend(handles=handles, labels=labels_, loc="lower right", fontsize=8.8, frameon=True)
    ax_b.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "Figure 5: Overall Classifier Diagnostics Across 29 Subjects "
        "(Aggregated Confusion Matrix & ROC-AUC)",
        fontsize=14.5, fontweight="bold", y=1.04,
    )
    fallback_note = (
        f" | {len(fallback_score_folds)} fold(s) used discrete y_pred as score proxy"
        if fallback_score_folds else ""
    )
    fig.text(
        0.5, 0.975,
        f"Pipeline: Condition 4 \u2014 Subspace Shrinkage Calibration (locked OOF results, "
        f"loaded from results_condition4_subspace_calibrated.json){fallback_note}",
        ha="center", va="top", fontsize=9.5, color="#333333",
    )
    fig.text(
        0.5, -0.04,
        "All confusion-matrix cells and ROC curves reflect strictly out-of-fold (OOF) predictions pooled across\n"
        f"{n_subjects}-fold Leave-One-Subject-Out cross-validation \u2014 no held-out subject's trials were seen during model fitting at any stage.",
        ha="center", va="top", fontsize=9.5, fontweight="normal", color="#000000",
    )

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.16, top=0.82, wspace=0.32)

    log.info(f"Saving 2-panel PNG (600 DPI) -> {PNG_MAIN}")
    fig.savefig(PNG_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
    log.info(f"Saving 2-panel PDF -> {PDF_MAIN}")
    fig.savefig(PDF_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # =========================================================================
    # SECTION 8: PERSISTENCE + LOGGING
    # =========================================================================
    volume.commit()
    log.info("volume.commit() complete.")

    file_report = {}
    for label, path in [("PNG_MAIN", PNG_MAIN), ("PDF_MAIN", PDF_MAIN)]:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0.0
        file_report[label] = {"path": path, "exists": exists, "size_kb": round(size_kb, 1)}
        status = "OK" if exists else "MISSING"
        log.info(f"  [{status}] {label:<10} -> {path}  ({size_kb:.1f} KB)")

    log.info("\n" + "=" * 70)
    log.info("  FIGURE 5 (CLASSIFIER DIAGNOSTICS, CONDITION 4) EXPORT COMPLETE")
    log.info("=" * 70)
    log.info(f"  Subjects (LOSO folds)  : {n_subjects}")
    log.info(f"  Pooled trials (N)      : {n_trials}")
    log.info(f"  Sensitivity            : {overall_sens:.4f}")
    log.info(f"  Specificity            : {overall_spec:.4f}")
    log.info(f"  Accuracy               : {overall_acc:.4f}")
    log.info(f"  Mean ROC-AUC (per-fold): {mean_auc:.4f} +/- {std_auc:.4f}")
    log.info(f"  Pooled ROC-AUC         : {pooled_auc:.4f}")
    if fallback_score_folds:
        log.info(f"  Folds using y_pred score proxy: {len(fallback_score_folds)}/{n_subjects}")
    log.info("=" * 70 + "\n")

    return {
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "overall_accuracy": overall_acc,
        "overall_sensitivity": overall_sens,
        "overall_specificity": overall_spec,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_auc": pooled_auc,
        "fallback_score_folds": fallback_score_folds,
        "per_subject": per_subject_rows,
        "files": file_report,
    }


# =============================================================================
# SECTION 9: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "=" * 70)
    print("  Figure 5 \u2014 Classifier Diagnostics (Condition 4, Locked OOF Results)")
    print("  (Aggregated Confusion Matrix + Mean ROC-AUC Curve)")
    print("=" * 70 + "\n")

    results = generate_figure5.remote()

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for key, val in results.items():
        if key == "per_subject":
            print(f"  {key:<20} : ({len(val)} subjects \u2014 see fold-by-fold logs above)")
        else:
            print(f"  {key:<20} : {val}")
    print("=" * 70)
    print(
        "\n  Figure 5 export complete."
        "\n  2-panel PNG/PDF committed to eeg-data-vol under /figures/."
        "\n\n  Download with:"
        "\n    modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.png ."
        "\n    modal volume get eeg-data-vol /figures/figure5_classifier_diagnostics.pdf .\n"
    )