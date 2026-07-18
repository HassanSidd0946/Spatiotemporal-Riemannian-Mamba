# =============================================================================
# command to run : modal run run_step4_condition4_spatial_only_ablation.py::main
# run_step4_condition4_spatial_only_ablation.py
#
# *** DIAGNOSTIC ABLATION — SPATIAL BRANCH ONLY, NO TEMPORAL/MAMBA BRANCH ***
#
# WHY THIS SCRIPT EXISTS:
#   The dual-branch pilot (Mamba + Riemannian, fused) landed at 55.70% over
#   5 folds — barely above EEGNet+Calib (55.64%) and ~12pp below the
#   original spatial-only Condition 4v2 result (67.79%). Two very different
#   explanations are consistent with that number:
#     (a) the temporal Mamba branch is diluting a good spatial signal once
#         fused (a fusion/architecture problem), or
#     (b) something in this NEW codebase's spatial pipeline itself
#         regressed relative to the original Condition 4v2 script (a
#         different bug, unrelated to Mamba at all).
#   This script removes the temporal branch entirely and feeds ONLY the
#   Riemannian spatial embedding to the classifier, with every other part
#   of the pipeline (EA whitening fit on the 28-subject pool, tangent-space
#   feature extraction, 15% calibration via the identical PCA + shrinkage-
#   blended LogisticRegression, lightweight dropout/regularization) held
#   IDENTICAL to the dual-branch pilot. If this lands near 67.79%, the
#   answer is (a) — dilution from fusion. If it lands near 55%, the answer
#   is (b) — something in the shared pipeline changed, and Mamba was never
#   the problem to begin with.
#
# WHAT'S IDENTICAL to the dual-branch pilot (run_step4_condition4_dualbranch
# _mamba_PILOT_v2.py): EA whitening, trial_covariances, tangent_vectorize,
# RiemannianSpatialBranch (same d_spatial, same dropout), calibration
# algorithm and hyperparameters, PRETRAIN_WD/ES_PAT, batched feature
# extraction, 5-fold pilot scope, separate output file.
#
# WHAT'S REMOVED: MambaTemporalBranch, SelectiveSSMBlock, parallel_scan —
# none of it is imported or defined here at all, not just unused, so
# there's no way for a temporal signal to leak in by accident.
#
# Usage: modal run run_step4_condition4_spatial_only_ablation.py::main
# =============================================================================

import modal

app    = modal.App("bci-condition4-spatial-only-ablation")
volume = modal.Volume.from_name("eeg-data-vol")

RAW_DATA_PATH    = "/data/processed_eeg_all_subjects.npz"
CONDITION3_JSON  = "/data/results_condition3_ea_zeroshot.json"
CONDITION1B_JSON = "/data/results_condition1b_eegnet_calibrated.json"
OUTPUT_JSON      = "/data/results_condition4_spatial_only_ablation_5_folds.json"
VOLUME_PATH      = "/data"

CONDITION1_ZEROSHOT_ACC   = 0.5214
CONDITION1B_FULL_MEAN_ACC = 0.5564   # full 29-fold EEGNet+Calib mean, for reference
CONDITION3_BASELINE_ACC   = 0.5552
CONDITION4_V2_SPATIAL_ACC = 0.6779   # original spatial-only script's full-29-fold result — the target to compare against
DUALBRANCH_PILOT_MEAN_ACC = 0.5570   # this codebase's dual-branch 5-fold pilot mean, for reference

SFREQ, N_CHANNELS = 250, 62

CAL_FRACTION      = 0.15
RANDOM_SEED       = 42
COV_SHRINKAGE     = 0.1
PCA_MAX_COMPONENTS = 35
LOGREG_C          = 1.0
SHRINK_GRID       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS   = 3

PILOT_N_FOLDS = 5

D_SPATIAL = 32     # UNCHANGED from the dual-branch pilot
DROPOUT   = 0.5    # UNCHANGED from the dual-branch pilot

calib_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.2.0", "numpy<2", "scikit-learn==1.4.2", "scipy")
)


@app.function(image=calib_image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_npz():
    import numpy as np
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    print(f"Keys: {raw.files}")
    for k in raw.files:
        print(f"  '{k}': shape={raw[k].shape} dtype={raw[k].dtype}")
    return raw.files


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    print(inspect_npz.remote())


@app.function(
    image=calib_image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=16384,
)
def run_condition4_spatial_only_pilot():

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json, copy, os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-spatial-only-ablation")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    X_np = raw["X"].astype(np.float32)
    y_np = raw["y"].astype(np.int64)
    subjects_np = raw["subjects"]
    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    assert C == N_CHANNELS and N_CLASSES == 2
    log.info(f"X: {X_np.shape} | Subjects: {len(np.unique(subjects_np))}")

    condition1b_per_subject_acc = {}
    if os.path.exists(CONDITION1B_JSON):
        with open(CONDITION1B_JSON) as f:
            c1b = json.load(f)
        for rec in c1b.get("fold_results", []):
            condition1b_per_subject_acc[str(rec["test_subject"])] = float(rec["post_calibration_acc"])

    condition3_per_subject_acc = {}
    if os.path.exists(CONDITION3_JSON):
        with open(CONDITION3_JSON) as f:
            c3 = json.load(f)
        for rec in c3.get("fold_results", []):
            condition3_per_subject_acc[str(rec["test_subject"])] = float(rec["test_accuracy"])

    # =========================================================================
    # RIEMANNIAN / EA UTILITIES — byte-for-byte identical to the dual-branch pilot
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
    # SPATIAL-ONLY MODEL — no temporal branch, no Mamba, nothing to dilute
    # =========================================================================

    class RiemannianSpatialBranch(nn.Module):
        """UNCHANGED from the dual-branch pilot — same architecture, same dims, same dropout."""
        def __init__(self, tangent_dim, d_spatial=D_SPATIAL, dropout=DROPOUT):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(tangent_dim, d_spatial * 2), nn.LayerNorm(d_spatial * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_spatial * 2, d_spatial), nn.LayerNorm(d_spatial),
            )

        def forward(self, tangent_vec):
            return self.net(tangent_vec)

    class SpatialOnlyModel(nn.Module):
        """
        Identical fusion/classifier depth to the dual-branch model (a Linear+GELU+Dropout
        "fusion" layer before the final classifier), but its input is the spatial
        embedding alone — no torch.cat with a temporal embedding, no temporal branch
        instantiated anywhere in this class. This isolates the spatial pipeline itself:
        if this model, with the same tangent features / EA whitening / calibration as
        the dual-branch pilot, doesn't come close to Condition 4v2's 67.79%, the
        regression is NOT about Mamba — it's somewhere in the shared preprocessing or
        the lighter dropout/regularization now in use.
        """
        def __init__(self, tangent_dim, n_classes=2):
            super().__init__()
            self.spatial = RiemannianSpatialBranch(tangent_dim)
            fused_dim = D_SPATIAL   # no temporal dim added
            self.fusion = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(DROPOUT))
            self.penultimate_dim = fused_dim
            self.classifier = nn.Linear(fused_dim, n_classes)

        def forward_features(self, x_tangent):
            return self.fusion(self.spatial(x_tangent))

        def forward(self, x_tangent):
            return self.classifier(self.forward_features(x_tangent))

    # =========================================================================
    # TRAIN / EVAL / CALIBRATION UTILITIES
    # =========================================================================

    def make_loader(X_tan, y, batch_size, shuffle, drop_last=False):
        ds = TensorDataset(torch.from_numpy(X_tan), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                           pin_memory=True, drop_last=drop_last)

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xt, yb in loader:
            Xt, yb = Xt.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xt)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xt, yb in loader:
            Xt, yb = Xt.to(device), yb.to(device)
            logits = model(Xt)
            total_loss += criterion(logits, yb).item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def extract_features(model, X_tan, batch_size=64):
        """Batched — same fix as the dual-branch pilot, kept here even though the
        spatial-only model is far lighter and less OOM-prone, for consistency."""
        model.eval()
        feats = []
        n = X_tan.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xt = torch.from_numpy(X_tan[start:end]).to(device)
            feats.append(model.forward_features(Xt).cpu().numpy())
        return np.concatenate(feats, axis=0)

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
        """Identical to the dual-branch pilot's calibration algorithm."""
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
    # PILOT LOSO — first PILOT_N_FOLDS subjects, spatial-only model
    # =========================================================================

    all_subjects = sorted(np.unique(subjects_np).tolist())
    unique_subjects = all_subjects[:PILOT_N_FOLDS]

    PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_LR = 30, 32, 1e-3
    PRETRAIN_WD, PRETRAIN_ES_PAT = 1e-2, 5   # UNCHANGED from the dual-branch pilot
    INTERNAL_VAL_FRAC = 0.10

    log.info(f"*** SPATIAL-ONLY ABLATION *** folds: {unique_subjects}")
    fold_records, all_test_acc = [], []

    for fold_idx, test_sub in enumerate(unique_subjects):
        fold_start = time.time()
        log.info(f"\n{'='*70}\n  ABLATION FOLD {fold_idx+1}/{len(unique_subjects)} — sub-{test_sub}\n{'='*70}")

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

        tan_train28 = tangent_vectorize(trial_covariances(X_train28_aligned))
        tan_k = tangent_vectorize(trial_covariances(X_k_aligned))
        tangent_dim = tan_train28.shape[1]

        X_pt_tr_tan, X_pt_val_tan, y_pt_tr, y_pt_val = train_test_split(
            tan_train28, y_train28, test_size=INTERNAL_VAL_FRAC, stratify=y_train28, random_state=RANDOM_SEED,
        )
        pretrain_loader = make_loader(X_pt_tr_tan, y_pt_tr, PRETRAIN_BATCH, True, drop_last=True)
        ptval_loader = make_loader(X_pt_val_tan, y_pt_val, PRETRAIN_BATCH, False)
        criterion = nn.CrossEntropyLoss()

        model = SpatialOnlyModel(tangent_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01)

        log.info(f"  Model: {n_params:,} params (spatial-only, d_spatial={D_SPATIAL}, NO temporal branch)")
        t0 = time.time()
        best_val_loss, best_state, es_ctr = math.inf, None, 0
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, pretrain_loader, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc = evaluate(model, ptval_loader, criterion)
            if epoch == 1 or epoch % 5 == 0:
                log.info(f"    Ep {epoch:02d}/{PRETRAIN_EPOCHS} | train acc={tr_acc:.3f} | "
                         f"val loss={val_loss:.4f} acc={val_acc:.3f} | {time.time()-t0:.0f}s elapsed")
            if val_loss < best_val_loss:
                best_val_loss, best_state, es_ctr = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                es_ctr += 1
                if es_ctr >= PRETRAIN_ES_PAT:
                    log.info(f"    Early stop at epoch {epoch}")
                    break
        model.load_state_dict(best_state)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        log.info(f"  Pretraining wall time: {time.time()-t0:.0f}s")

        feat_train28 = extract_features(model, tan_train28)
        feat_k = extract_features(model, tan_k)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED)
        cal_idx, test_idx = next(sss.split(feat_k, y_k))
        feat_cal, y_cal = feat_k[cal_idx], y_k[cal_idx]
        feat_test, y_test = feat_k[test_idx], y_k[test_idx]

        scaler = StandardScaler()
        feat_train28_z = scaler.fit_transform(feat_train28)
        feat_cal_z = scaler.transform(feat_cal)
        feat_test_z = scaler.transform(feat_test)

        n_components = min(PCA_MAX_COMPONENTS, model.penultimate_dim - 1, feat_train28_z.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train28_pca = pca.fit_transform(feat_train28_z)
        X_cal_pca = pca.transform(feat_cal_z)
        X_test_pca = pca.transform(feat_test_z)

        coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal)
        pre_cal_acc = float((global_clf.predict(X_test_pca) == y_test).mean())
        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        best_test_acc = float((final_preds == y_test).mean())
        metrics = compute_binary_metrics(y_test, final_preds)

        c1b_acc = condition1b_per_subject_acc.get(str(test_sub))
        log.info(f"  RESULT -> pre_cal={pre_cal_acc:.4f}  post_cal={best_test_acc:.4f} (shrink={best_shrink:.2f})")
        if c1b_acc is not None:
            log.info(f"      Δ vs EEGNet+Calib (sub-{test_sub}: {c1b_acc*100:.2f}%) -> {(best_test_acc-c1b_acc)*100:+.2f} pp")

        fold_records.append({
            "fold_index": fold_idx, "test_subject": str(test_sub),
            "tangent_dim": int(tangent_dim), "penultimate_dim": int(model.penultimate_dim),
            "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": pre_cal_acc, "post_calibration_acc": best_test_acc,
            **metrics,
            "condition1b_eegnet_calib_acc": c1b_acc,
            "condition3_zero_shot_acc": condition3_per_subject_acc.get(str(test_sub)),
        })
        all_test_acc.append(best_test_acc)
        log.info(f"  Fold elapsed: {time.time()-fold_start:.0f}s")

        del model, optimizer, scheduler, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_acc, std_acc = float(np.mean(all_test_acc)), float(np.std(all_test_acc))

    log.info(f"\n{'='*70}\n  SPATIAL-ONLY ABLATION — {len(unique_subjects)} folds\n{'='*70}")
    log.info(f"  Mean ± Std Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(
        "\n  INTERPRETATION:\n"
        f"    This codebase's dual-branch (Mamba+spatial) pilot mean : {DUALBRANCH_PILOT_MEAN_ACC*100:.2f}%\n"
        f"    This spatial-only ablation mean (THIS RUN)              : {mean_acc*100:.2f}%\n"
        f"    Original Condition 4v2 script's full-29-fold result     : {CONDITION4_V2_SPATIAL_ACC*100:.2f}%\n"
        "    - If THIS RUN is close to 67.79%: the temporal Mamba branch is diluting\n"
        "      a good spatial signal once fused — a fusion/architecture problem.\n"
        "      Try attention-pooling or gated fusion instead of concat+mean-pool.\n"
        "    - If THIS RUN is close to ~55% (like the dual-branch pilot): something\n"
        "      in the shared pipeline (whitening, tangent features, the lighter\n"
        "      dropout/regularization) changed relative to the original Condition\n"
        "      4v2 script, independent of Mamba entirely — investigate PCA_MAX_\n"
        "      COMPONENTS interaction with the smaller D_SPATIAL=32 penultimate\n"
        "      dim, and the stronger DROPOUT=0.5 / PRETRAIN_WD=1e-2 regularization,\n"
        "      before touching the temporal branch again.\n"
        "    Still only n=5 — not a manuscript number either way, just a diagnostic."
    )

    results_payload = {
        "condition": "Condition 4 — SPATIAL-ONLY ABLATION (diagnostic, no temporal/Mamba branch)",
        "is_pilot": True, "is_ablation": True, "pilot_n_folds": len(unique_subjects),
        "hyperparameters": {
            "d_spatial": D_SPATIAL, "dropout": DROPOUT,
            "pretrain_wd": PRETRAIN_WD, "pretrain_es_patience": PRETRAIN_ES_PAT,
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
        },
        "fold_results": fold_records, "mean_accuracy": mean_acc, "std_accuracy": std_acc,
        "reference_dualbranch_pilot_mean_acc": DUALBRANCH_PILOT_MEAN_ACC,
        "reference_condition4v2_full_run_acc": CONDITION4_V2_SPATIAL_ACC,
        "reference_condition1b_full_run_mean_acc": CONDITION1B_FULL_MEAN_ACC,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    volume.commit()
    log.info(f"  Saved: {OUTPUT_JSON}")

    return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "output_path": OUTPUT_JSON}


@app.local_entrypoint()
def main():
    print("Condition 4 — SPATIAL-ONLY ABLATION (diagnostic)")
    print("No temporal/Mamba branch. Same EA/tangent/calibration pipeline as the")
    print("dual-branch pilot. 5-fold sanity check only — isolates fusion dilution")
    print("from a shared-pipeline regression.\n")
    results = run_condition4_spatial_only_pilot.remote()
    print("\nSPATIAL-ONLY ABLATION RESULTS (diagnostic — do not report):")
    for k, v in results.items():
        print(f"  {k:<35}: {v}")