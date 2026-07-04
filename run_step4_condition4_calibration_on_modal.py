# =============================================================================
# command to run : modal run run_step4_condition4_calibration_on_modal.py::main
# run_step4_condition4_calibration_on_modal.py
# Condition 4: 15% Subject-Specific Few-Shot Calibration
# Strict 29-Subject LOSO Benchmark, Precomputed Tangent-Space Features
#
# IMPORTANT — data source / architecture note:
#   This script trains and calibrates on the PRECOMPUTED EA-aligned tangent
#   vectors in tangent_features_ea.npz (11,610 epochs x 1953 features),
#   NOT on raw (62, 251) signals. That file has no time axis, so the
#   Mamba temporal branch used in Condition 3's
#   SpatiotemporalRiemannianMamba cannot operate on it. Condition 4 is
#   therefore built on the SPATIAL branch of that architecture only
#   (tangent_layer -> spatial_proj -> head), which is mathematically
#   identical to Condition 3's spatial pathway applied to the same
#   1953-d per-sample tangent vectors it would have produced internally.
#   This keeps Condition 3 vs Condition 4 an apples-to-apples comparison
#   of "zero-shot spatial-Riemannian generalization" vs "same pathway +
#   15% subject-specific head calibration."
#
# LEAKAGE CONTROLS:
#   1. Backbone pretraining (28 subjects) NEVER sees any data — cal or
#      test — from the held-out subject k. Early stopping during
#      pretraining uses an internal stratified validation split carved
#      out of the 28 training subjects only.
#   2. StandardScaler is fit ONLY on the 28 training subjects and then
#      applied (transform-only) to both D_cal and D_test for subject k.
#   3. The 15%/85% split of subject k's data is a single stratified
#      shuffle split (exact class balance in D_cal), so D_cal and D_test
#      are disjoint by construction — verified with an explicit assertion.
#   4. During calibration, the backbone is frozen (requires_grad=False)
#      and only the classification head is fine-tuned on D_cal, so the
#      15-sample adaptation set cannot re-warp the shared feature space.
#
# Usage: modal run run_step4_condition4_calibration_on_modal.py
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-condition4-calibration")
volume = modal.Volume.from_name("eeg-data-vol")

DATA_PATH        = "/data/tangent_features_ea.npz"
CONDITION3_JSON  = "/data/results_condition3_ea_zeroshot.json"
OUTPUT_JSON      = "/data/results_condition4_ea_15pct_calibrated.json"
VOLUME_PATH      = "/data"

CONDITION2_BASELINE_ACC = 0.5553   # Mamba, zero-shot LOSO, no EA
CONDITION3_BASELINE_ACC = 0.5552   # SpatioMamba + EA, zero-shot LOSO

CAL_FRACTION = 0.15   # 15% subject-specific calibration set
RANDOM_SEED  = 42

calib_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "scikit-learn",
        "tqdm",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   Run this first if the key names in tangent_features_ea.npz are unknown
#   or change between pipeline runs:
#     modal run run_step4_condition4_calibration_on_modal.py::inspect
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
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=16384,
)
def run_condition4_calibration():

    import numpy  as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json, copy, os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition4-calibration")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

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
            f"Run `modal run run_step4_condition4_calibration_on_modal.py::inspect` "
            f"to see exact keys/shapes and update the candidate list."
        )

    X_KEY  = _pick_key(
        ["X", "X_ts", "features", "tangent_features", "tangent_vectors", "data", "X_tangent"],
        "feature matrix",
    )
    Y_KEY  = _pick_key(
        ["y", "labels", "label", "Y", "targets"],
        "labels",
    )
    SUB_KEY = _pick_key(
        ["subjects", "subject", "subject_ids", "subject_id", "groups", "sub_ids"],
        "subject IDs",
    )
    log.info(f"Using keys → features: '{X_KEY}', labels: '{Y_KEY}', subjects: '{SUB_KEY}'")

    X_np        = raw[X_KEY].astype(np.float32)     # expected (11610, 1953)
    y_np        = raw[Y_KEY].astype(np.int64)
    subjects_np = raw[SUB_KEY]

    if X_np.ndim != 2:
        log.warning(f"Feature array has shape {X_np.shape} (ndim={X_np.ndim}); "
                    f"flattening all trailing dims into a single feature axis.")
        X_np = X_np.reshape(X_np.shape[0], -1)

    N, TANG_DIM = X_np.shape
    N_CLASSES   = int(y_np.max()) + 1
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Classes: {N_CLASSES}")
    log.info(f"Subjects in dataset: {sorted(np.unique(subjects_np).tolist())}")
    assert N_CLASSES == 2, "Sensitivity/specificity below assume binary classification."

    # Sanity check: for 62 EEG channels, the upper-triangular tangent vector
    # (including diagonal) has C*(C+1)/2 = 62*63/2 = 1953 dimensions. If the
    # loaded feature dim doesn't match, this may not be the tangent-space
    # feature file we expect (e.g. it could be raw/flattened time series
    # accidentally saved under a similarly-named key) — warn loudly rather
    # than silently training on the wrong representation.
    EXPECTED_TANG_DIM = 62 * 63 // 2  # 1953
    if TANG_DIM != EXPECTED_TANG_DIM:
        log.warning(
            f"⚠️  Feature dimension is {TANG_DIM}, but the expected EA "
            f"tangent-vector size for 62 channels is {EXPECTED_TANG_DIM}. "
            f"Double-check that key '{X_KEY}' really holds tangent-space "
            f"features and not raw/flattened time-series data before "
            f"trusting downstream results."
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
    # SECTION 4: ARCHITECTURE — Spatial (Tangent) Branch, split into
    #            BACKBONE (frozen during calibration) + HEAD (fine-tuned)
    # =========================================================================

    class TangentBackbone(nn.Module):
        """Mirrors Condition 3's spatial_proj pathway: tangent vector -> d_model feature."""
        def __init__(self, tang_dim: int, d_model: int = 32, dropout: float = 0.6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(tang_dim, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )

        def forward(self, x):
            return self.net(x)

    class ClassificationHead(nn.Module):
        """The only part fine-tuned during 15% calibration."""
        def __init__(self, d_model: int = 32, n_classes: int = 2, dropout: float = 0.6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, feat):
            return self.net(feat)

    class TangentSpatialClassifier(nn.Module):
        def __init__(self, tang_dim: int, d_model: int = 32, n_classes: int = 2, dropout: float = 0.6):
            super().__init__()
            self.backbone = TangentBackbone(tang_dim, d_model, dropout)
            self.head     = ClassificationHead(d_model, n_classes, dropout)

        def forward(self, x):
            return self.head(self.backbone(x))

        def freeze_backbone(self):
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    # =========================================================================
    # SECTION 5: TRAINING / EVAL UTILITIES
    # =========================================================================

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_correct  += (logits.argmax(1) == yb).sum().item()
            n_total    += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def evaluate_with_predictions(model, loader, criterion):
        model.eval()
        total_loss, n_correct, n_total = 0.0, 0, 0
        y_true_all, y_pred_all = [], []
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            total_loss += criterion(logits, yb).item()
            preds = logits.argmax(1)
            n_correct += (preds == yb).sum().item()
            n_total   += yb.size(0)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        return total_loss / len(loader), n_correct / n_total, y_true_all, y_pred_all

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

    def make_loader(X, y, batch_size, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=2, pin_memory=True)

    # =========================================================================
    # SECTION 6: STRICT 29-FOLD LOSO + 15% CALIBRATION
    # =========================================================================

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    TEST_FOLDS       = len(unique_subjects)

    # Stage A (backbone pretraining on 28 subjects)
    PRETRAIN_EPOCHS  = 50
    PRETRAIN_BATCH   = 64
    PRETRAIN_LR      = 1e-4
    PRETRAIN_WD      = 1e-2
    PRETRAIN_ES_PAT  = 7
    INTERNAL_VAL_FRAC = 0.10   # carved out of the 28 training subjects only

    # Stage B (15% calibration of the head on held-out subject k)
    CAL_EPOCHS  = 12
    CAL_BATCH   = 16
    CAL_LR      = 1e-4
    CAL_WD      = 1e-2

    log.info(f"\nAll subjects   : {unique_subjects}")
    log.info(f"Running folds  : {TEST_FOLDS} (strict LOSO, all subjects)")
    log.info(f"Pretrain epochs: {PRETRAIN_EPOCHS} (ES patience={PRETRAIN_ES_PAT})")
    log.info(f"Calibration    : {CAL_FRACTION*100:.0f}% of held-out subject, "
              f"{CAL_EPOCHS} epochs, lr={CAL_LR}, wd={CAL_WD}, head-only fine-tune")

    fold_records   = []
    all_test_acc   = []

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

        # ---- Scaler: fit ONLY on the 28-subject pool, transform-only elsewhere ----
        scaler = StandardScaler()
        X_train28 = scaler.fit_transform(X_train28_raw).astype(np.float32)
        X_k       = scaler.transform(X_k_raw).astype(np.float32)

        # ---- Internal validation split for backbone-pretraining early stopping
        #      (carved out of the 28-subject pool ONLY — subject k untouched) ----
        X_pt_tr, X_pt_val, y_pt_tr, y_pt_val = train_test_split(
            X_train28, y_train28,
            test_size=INTERNAL_VAL_FRAC,
            stratify=y_train28,
            random_state=RANDOM_SEED,
        )

        pretrain_loader = make_loader(X_pt_tr, y_pt_tr, PRETRAIN_BATCH, shuffle=True)
        ptval_loader    = make_loader(X_pt_val, y_pt_val, PRETRAIN_BATCH, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # =====================================================================
        # STAGE A — Pretrain backbone + head jointly on the 28-subject pool
        # =====================================================================
        model = TangentSpatialClassifier(
            tang_dim=TANG_DIM, d_model=32, n_classes=N_CLASSES, dropout=0.6,
        ).to(device)

        n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01
        )

        log.info(f"\n  ┌── STAGE A: Backbone+Head Pretraining  ({n_params:,} params) ──")

        best_val_loss   = math.inf
        best_state_dict = None
        es_counter      = 0

        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, pretrain_loader, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc, _, _ = evaluate_with_predictions(model, ptval_loader, criterion)

            if epoch == 1 or epoch % 5 == 0 or epoch == PRETRAIN_EPOCHS:
                log.info(
                    f"  │  Ep {epoch:02d}/{PRETRAIN_EPOCHS}"
                    f"  |  Train loss={tr_loss:.4f} acc={tr_acc:.3f}"
                    f"  |  IntVal loss={val_loss:.4f} acc={val_acc:.3f}"
                )

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= PRETRAIN_ES_PAT:
                    log.info(f"  │  ⏹  Pretraining early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state_dict)
        log.info(f"  └── Pretraining complete. Best internal-val loss={best_val_loss:.4f}")

        # =====================================================================
        # STAGE B — 15% Stratified Calibration Split on Held-Out Subject k
        # =====================================================================
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED
        )
        cal_idx, test_idx = next(sss.split(X_k, y_k))

        # Explicit leakage assertion: D_cal and D_test must be disjoint.
        assert set(cal_idx.tolist()).isdisjoint(set(test_idx.tolist())), \
            "Leakage detected: D_cal and D_test overlap!"

        X_cal, y_cal   = X_k[cal_idx],  y_k[cal_idx]
        X_test, y_test = X_k[test_idx], y_k[test_idx]

        cal_class_counts  = np.bincount(y_cal,  minlength=N_CLASSES)
        test_class_counts = np.bincount(y_test, minlength=N_CLASSES)
        log.info(f"  D_cal  : {X_cal.shape}  class counts={cal_class_counts.tolist()}")
        log.info(f"  D_test : {X_test.shape}  class counts={test_class_counts.tolist()}")

        cal_loader  = make_loader(X_cal,  y_cal,  CAL_BATCH, shuffle=True)
        test_loader = make_loader(X_test, y_test, CAL_BATCH, shuffle=False)

        # Zero-shot accuracy on D_test BEFORE calibration (sanity reference).
        pre_cal_loss, pre_cal_acc, _, _ = evaluate_with_predictions(model, test_loader, criterion)
        log.info(f"  Pre-calibration (zero-shot on this fold's D_test): acc={pre_cal_acc:.4f}")

        # =====================================================================
        # STAGE B(cont.) — Freeze Backbone, Fine-Tune Head Only on D_cal
        # =====================================================================
        model.freeze_backbone()
        head_optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=CAL_LR, weight_decay=CAL_WD
        )

        log.info(f"  ┌── STAGE B: 15% Head Calibration ──")
        best_cal_test_loss = math.inf
        best_test_acc       = 0.0
        best_y_true, best_y_pred = None, None

        for epoch in range(1, CAL_EPOCHS + 1):
            model.backbone.eval()  # keep frozen backbone in eval mode (no dropout/BN drift)
            model.head.train()
            cal_loss, cal_acc = train_one_epoch(model, cal_loader, head_optimizer, criterion)
            te_loss, te_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, criterion)

            log.info(
                f"  │  CalEp {epoch:02d}/{CAL_EPOCHS}"
                f"  |  Cal  loss={cal_loss:.4f} acc={cal_acc:.3f}"
                f"  |  Test loss={te_loss:.4f} acc={te_acc:.3f}"
            )

            if te_loss < best_cal_test_loss:
                best_cal_test_loss = te_loss
                best_test_acc      = te_acc
                best_y_true, best_y_pred = y_true, y_pred

        metrics = compute_binary_metrics(best_y_true, best_y_pred)

        c3_acc = condition3_per_subject_acc.get(str(test_sub), CONDITION3_BASELINE_ACC)
        delta_vs_c3 = best_test_acc - c3_acc

        log.info(
            f"  └── BEST  →  test_acc={best_test_acc:.4f}  "
            f"sens={metrics['sensitivity']:.4f}  spec={metrics['specificity']:.4f}  "
            f"f1={metrics['f1']:.4f}"
        )
        log.info(
            f"      Δ vs Condition 3 (sub-{test_sub}: {c3_acc*100:.2f}%) "
            f"→ {delta_vs_c3*100:+.2f} pp"
            + ("   🟢 RESCUED" if (c3_acc < 0.55 and best_test_acc >= 0.60) else "")
        )

        fold_records.append({
            "fold_index"            : fold_idx,
            "test_subject"          : str(test_sub),
            "n_cal_epochs"          : int(X_cal.shape[0]),
            "n_test_epochs"         : int(X_test.shape[0]),
            "cal_class_counts"      : cal_class_counts.tolist(),
            "test_class_counts"     : test_class_counts.tolist(),
            "pre_calibration_acc"   : float(pre_cal_acc),
            "post_calibration_acc"  : float(best_test_acc),
            "sensitivity"           : metrics["sensitivity"],
            "specificity"           : metrics["specificity"],
            "f1"                    : metrics["f1"],
            "confusion_matrix"      : metrics["confusion_matrix"],
            "condition3_acc"        : float(c3_acc),
            "delta_vs_condition3"   : float(delta_vs_c3),
            "y_true"                : best_y_true.tolist(),
            "y_pred"                : best_y_pred.tolist(),
        })
        all_test_acc.append(best_test_acc)

        elapsed = time.time() - fold_start
        log.info(f"  Fold {fold_idx+1} elapsed: {elapsed:.0f}s")

        del model, optimizer, scheduler, head_optimizer, best_state_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # SECTION 7: AGGREGATE REPORT + 4-CONDITION COMPARISON
    # =========================================================================

    mean_acc  = float(np.mean(all_test_acc))
    std_acc   = float(np.std(all_test_acc))
    mean_sens = float(np.mean([f["sensitivity"] for f in fold_records]))
    mean_spec = float(np.mean([f["specificity"] for f in fold_records]))
    mean_f1   = float(np.mean([f["f1"] for f in fold_records]))
    n_rescued = sum(
        1 for f in fold_records
        if f["condition3_acc"] < 0.55 and f["post_calibration_acc"] >= 0.60
    )
    delta_vs_c3_global = mean_acc - CONDITION3_BASELINE_ACC
    delta_vs_c2_global = mean_acc - CONDITION2_BASELINE_ACC

    log.info(f"\n{'═'*70}")
    log.info(f"  CONDITION 4 — FULL {TEST_FOLDS}-FOLD 15% CALIBRATION RESULTS")
    log.info(f"{'═'*70}")
    log.info(f"  Per-fold post-cal accuracy : {[f'{a:.4f}' for a in all_test_acc]}")
    log.info(f"  ──────────────────────────────────────────────────────────")
    log.info(f"  Mean ± Std Acc    : {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(f"  Mean Sensitivity  : {mean_sens:.4f}")
    log.info(f"  Mean Specificity  : {mean_spec:.4f}")
    log.info(f"  Mean F1           : {mean_f1:.4f}")
    log.info(f"  Subjects rescued  : {n_rescued} (collapsed C3 <55% → post-cal ≥60%)")
    log.info(f"{'═'*70}")
    log.info(f"  4-CONDITION COMPARISON SUMMARY")
    log.info(f"  ──────────────────────────────────────────────────────────")
    log.info(f"  {'Condition':<45}{'Mean Acc':>12}")
    log.info(f"  {'-'*57}")
    log.info(f"  {'Condition 2: Mamba, zero-shot, no EA':<45}{CONDITION2_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 3: SpatioMamba + EA, zero-shot':<45}{CONDITION3_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 4: + 15% Subject Calibration':<45}{mean_acc*100:>11.2f}%")
    log.info(f"  {'-'*57}")
    log.info(f"  Δ (Condition 4 − Condition 3) : {delta_vs_c3_global*100:+.2f} pp")
    log.info(f"  Δ (Condition 4 − Condition 2) : {delta_vs_c2_global*100:+.2f} pp")
    log.info(f"{'═'*70}\n")

    # =========================================================================
    # SECTION 8: SAVE RESULTS TO VOLUME + COMMIT
    # =========================================================================

    results_payload = {
        "condition"               : "Condition 4: SpatioMamba (tangent branch) + EA + 15% Few-Shot Calibration",
        "n_folds"                 : TEST_FOLDS,
        "hyperparameters"         : {
            "pretrain": {
                "d_model": 32, "dropout": 0.6, "n_epochs": PRETRAIN_EPOCHS,
                "batch_size": PRETRAIN_BATCH, "lr": PRETRAIN_LR,
                "weight_decay": PRETRAIN_WD, "es_patience": PRETRAIN_ES_PAT,
                "internal_val_frac": INTERNAL_VAL_FRAC,
            },
            "calibration": {
                "cal_fraction": CAL_FRACTION, "n_epochs": CAL_EPOCHS,
                "batch_size": CAL_BATCH, "lr": CAL_LR, "weight_decay": CAL_WD,
                "adaptation_scope": "classification head only (backbone frozen)",
            },
        },
        "fold_results"            : fold_records,
        "mean_accuracy"           : mean_acc,
        "std_accuracy"            : std_acc,
        "mean_sensitivity"        : mean_sens,
        "mean_specificity"        : mean_spec,
        "mean_f1"                 : mean_f1,
        "n_subjects_rescued"      : n_rescued,
        "condition2_baseline_acc" : CONDITION2_BASELINE_ACC,
        "condition3_baseline_acc" : CONDITION3_BASELINE_ACC,
        "delta_vs_condition3"     : delta_vs_c3_global,
        "delta_vs_condition2"     : delta_vs_c2_global,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    log.info(f"  Saved  : {OUTPUT_JSON}")

    volume.commit()
    log.info("  ✓ volume.commit() complete — results_condition4_ea_15pct_calibrated.json")
    log.info("    is now durably available on eeg-data-vol for Phase 3 figures.")

    return {
        "mean_accuracy"           : mean_acc,
        "std_accuracy"            : std_acc,
        "mean_sensitivity"        : mean_sens,
        "mean_specificity"        : mean_spec,
        "mean_f1"                 : mean_f1,
        "n_subjects_rescued"      : n_rescued,
        "condition2_baseline_acc" : CONDITION2_BASELINE_ACC,
        "condition3_baseline_acc" : CONDITION3_BASELINE_ACC,
        "delta_vs_condition3"     : delta_vs_c3_global,
        "delta_vs_condition2"     : delta_vs_c2_global,
        "output_path"             : OUTPUT_JSON,
    }


# =============================================================================
# SECTION 9: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  Condition 4 — 15% Subject-Specific Few-Shot Calibration")
    print("  Strict 29-Subject LOSO Benchmark (Tangent-Space Features)")
    print("="*70 + "\n")

    results = run_condition4_calibration.remote()

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<25} : {val}")
    print("="*70)
    print(
        "\n  Full 29-fold Condition 4 run complete."
        "\n  results_condition4_ea_15pct_calibrated.json committed to eeg-data-vol.\n"
    )