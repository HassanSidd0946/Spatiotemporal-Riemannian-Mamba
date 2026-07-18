# =============================================================================
# command to run : modal run run_step4_condition1_eegnet_calibrated.py::main
# run_step4_condition1_eegnet_calibrated.py
#
# Condition 1b: Authentic EEGNet (Lawhern et al., 2018) + 15% Few-Shot
#               PCA-Subspace Shrinkage Calibration
# Strict 29-Subject LOSO Benchmark, RAW epoched EEG (NOT precomputed
# tangent-space features — EEGNet consumes (channels, time) directly).
#
# WHY THIS SCRIPT EXISTS:
#   The reviewer's core objection was that Condition 4's 67.79% cannot be
#   attributed to the Mamba/Riemannian architecture unless we show what an
#   EEGNet backbone does under the IDENTICAL calibration budget and
#   calibration ALGORITHM. This script is that missing grid cell:
#
#            zero-shot          + 15% few-shot calibration
#   EEGNet     52.14% (given)     ??? <- this script
#   SpatioMamba 55.52% (Cond 3)   67.79% (Cond 4)
#
#   If EEGNet+calibration lands well below 67.79%, that supports "Mamba
#   contributes on top of calibration." If it lands close to or above
#   67.79%, the manuscript's architecture claim is not supported and the
#   calibration step, not Mamba, is doing the work — that must be reported
#   honestly either way.
#
# CALIBRATION METHOD — deliberately matched to
#   run_step4_condition4_subspace_calibration.py (the PCA + global/local
#   shrinkage-blended LogisticRegression version), NOT the frozen-backbone
#   fine-tuned-head version. If your reported 67.79% actually came from the
#   fine-tuning script instead, this comparison is apples-to-oranges and
#   you need to tell me so I can rebuild against that method instead.
#
#   Mechanism:
#     1. Pretrain EEGNet backbone+head on the 28 training subjects (real
#        gradient training, real EEGNet architecture, no shortcuts).
#     2. Freeze the trained backbone. Extract PENULTIMATE-layer features
#        (the flattened representation immediately before the final Dense
#        classification layer) for: the 28-subject training pool, and the
#        held-out subject's calibration + test trials.
#     3. Fit StandardScaler + PCA on the 28-subject penultimate features
#        ONLY (never on subject k's data).
#     4. Fit a "global" LogisticRegression on the 28-subject PCA features,
#        and a "local" LogisticRegression on subject k's calibration-set
#        PCA features only.
#     5. Select a shrinkage blend weight between global and local via
#        internal stratified CV computed ONLY on the calibration set.
#     6. Evaluate the shrinkage-blended classifier once on subject k's
#        held-out test set.
#
# AUTHENTIC EEGNet ARCHITECTURE (Lawhern et al. 2018, J. Neural Eng.):
#   Block 1: Conv2D(1->F1, (1,kernLength), same pad) -> BN
#            -> DepthwiseConv2D(F1->F1*D, (C,1), groups=F1, max_norm=1.0)
#            -> BN -> ELU -> AvgPool2D((1,4)) -> Dropout
#   Block 2: SeparableConv2D(F1*D->F2, (1,16), same pad)
#            [depthwise (1,16) groups=F1*D, then pointwise (1,1)]
#            -> BN -> ELU -> AvgPool2D((1,8)) -> Dropout
#   Classifier: Flatten -> Dense(n_classes, max_norm=0.25)
#   Defaults: F1=8, D=2, F2=16 (=F1*D), dropoutRate=0.5
#   PyTorch has no native Keras-style kernel_constraint=max_norm, so it is
#   implemented manually via weight-norm clamping after each optimizer step
#   (see `apply_max_norm_constraints`), matching the original Keras impl.
#
# CONFIRMED CONFIG (verified against step1_2_filter_epoch.py and the
# ds005189 dataset card — no longer guesses):
#   - SFREQ=250Hz: step1_2_filter_epoch.py downsamples from the dataset's
#     native 1000Hz to SFREQ_NEW=250Hz, epoch window -0.2s to 0.8s (1.0s),
#     giving 251 samples — matches X.shape[2]=251 exactly.
#   - KERN_LENGTH=125 (SFREQ//2), per Lawhern et al.'s convention.
#   - RAW_DATA_PATH confirmed via `modal volume ls` + ::inspect: X (11610,
#     62, 251) float32, y (11610,) int32, subjects (11610,) object — raw
#     epoched signal, not tangent features.
#   - COGNITIVE FRAMING (corrected): this is NOT a true/false memory
#     paradigm. EVENT_ID groups 1x codes as Class 0 (Search/incidental
#     encoding) and 2x codes as Class 1 (Memorize/intentional encoding).
#     The task is "Incidental vs. Intentional Memory Encoding," per the
#     ds005189 dataset card (Helbing, Draschkow & Võ — Search Superiority
#     Recollection Familiarity). Do not describe this as false-memory
#     decoding anywhere in the manuscript or these comments.
#
# LEAKAGE CONTROLS:
#   1. Backbone pretraining (28 subjects) never sees any data — cal or
#      test — from the held-out subject k. Early-stopping validation split
#      is carved out of the 28 training subjects only.
#   2. StandardScaler and PCA for the calibration stage are fit ONLY on
#      the 28-subject penultimate features, transform-only elsewhere.
#   3. The 15%/85% split of subject k's data is a single stratified
#      shuffle split; D_cal and D_test disjointness is asserted explicitly.
#   4. Global LogReg is fit only on the 28-subject pool. Local LogReg and
#      the shrinkage weight are selected using D_cal ONLY (internal CV).
#      D_test is touched exactly once, for final evaluation.
#   5. EEGNet backbone weights are frozen before any subject-k data is
#      seen — subject k can never backprop into the shared feature space.
#
# Usage: modal run run_step4_condition1_eegnet_calibrated.py::main
#        modal run run_step4_condition1_eegnet_calibrated.py::inspect
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-condition1-eegnet-calibrated")
volume = modal.Volume.from_name("eeg-data-vol")

# ⚠️ VERIFY THIS PATH — must point to RAW epoched EEG, not tangent features.
# Confirmed via `modal volume ls eeg-data-vol`: processed_eeg_all_subjects.npz
# (640.1 MiB, created 2026-06-17, BEFORE tangent_features_ea.npz created
# 2026-07-04) is the only plausible raw-signal candidate on the volume — its
# size and creation order relative to the EA/tangent file are consistent
# with being the pre-tangent-mapping source, but this is still an inference,
# not a confirmed fact. The ::inspect run below is what actually confirms it
# — check that the array is 3D (n_epochs, n_channels, n_times), not 2D.
RAW_DATA_PATH   = "/data/processed_eeg_all_subjects.npz"
CONDITION3_JSON = "/data/results_condition3_ea_zeroshot.json"
CONDITION4_JSON = "/data/results_condition4_subspace_calibrated.json"
OUTPUT_JSON     = "/data/results_condition1b_eegnet_calibrated.json"
VOLUME_PATH     = "/data"

CONDITION1_ZEROSHOT_ACC = 0.5214   # EEGNet, zero-shot LOSO (given)
CONDITION2_BASELINE_ACC = 0.5553   # Mamba, zero-shot LOSO, no EA
CONDITION3_BASELINE_ACC = 0.5552   # SpatioMamba + EA, zero-shot LOSO
CONDITION4_BASELINE_ACC = 0.6779   # SpatioMamba + EA + Subspace Calib.

# ── CONFIRMED (step1_2_filter_epoch.py: SFREQ_NEW=250, epoch -0.2s:0.8s) ────
SFREQ            = 250     # Hz — confirmed, not a guess
N_CHANNELS       = 62      # confirmed via ::inspect (X.shape == (11610, 62, 251))
KERN_LENGTH      = 125     # SFREQ // 2, per Lawhern et al. convention
# ────────────────────────────────────────────────────────────────────────────

CAL_FRACTION      = 0.15   # <-- MATCHED to Condition 4 v2. Change here only
                            #     if Condition 4's true CAL_FRACTION differs.
RANDOM_SEED        = 42
PCA_MAX_COMPONENTS = 35    # ceiling; actual n_components is capped below
                            # this and (penultimate_dim - 1), see Section 6.
LOGREG_C            = 1.0
SHRINK_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHRINK_CV_FOLDS     = 3

# EEGNet hyperparameters (Lawhern et al. 2018 defaults)
EEGNET_F1            = 8
EEGNET_D             = 2
EEGNET_F2            = EEGNET_F1 * EEGNET_D   # 16
EEGNET_DROPOUT       = 0.5
EEGNET_DEPTHWISE_MAXNORM = 1.0
EEGNET_DENSE_MAXNORM     = 0.25

calib_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "scikit-learn==1.4.2",
        "scipy",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR — run this FIRST
#   modal run run_step4_condition1_eegnet_calibrated.py::inspect
# =============================================================================

@app.function(image=calib_image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_npz():
    import numpy as np
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    print(f"\nArchive: {RAW_DATA_PATH}")
    print(f"Keys found: {raw.files}\n")
    for k in raw.files:
        arr = raw[k]
        print(f"  '{k}': shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim <= 1 and arr.size <= 40:
            print(f"      sample values: {arr[:10]}")
    print(
        "\n  ⚠️  Confirm the feature array is (n_epochs, n_channels, n_times) "
        "RAW signal, not a precomputed tangent vector. Confirm SFREQ matches "
        "your true acquisition sampling rate before running ::main."
    )
    return raw.files


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    keys = inspect_npz.remote()
    print(f"\nKeys in {RAW_DATA_PATH}: {keys}")


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
def run_condition1_eegnet_calibration():

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        train_test_split, StratifiedShuffleSplit, StratifiedKFold,
    )
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json, copy, os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition1b-eegnet-calibration")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # SECTION 3: DATA LOADING (RAW epoched EEG — NOT tangent features)
    # =========================================================================
    log.info(f"Loading {RAW_DATA_PATH} ...")
    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    log.info(f"Archive keys found: {raw.files}")

    def _pick_key(candidates, purpose):
        for c in candidates:
            if c in raw.files:
                return c
        raise KeyError(
            f"Could not find a key for '{purpose}' in {RAW_DATA_PATH}. "
            f"Tried {candidates}, but archive only contains {raw.files}. "
            f"Run `modal run run_step4_condition1_eegnet_calibrated.py::inspect` "
            f"to see exact keys/shapes and update the candidate list."
        )

    X_KEY   = _pick_key(
        ["X", "X_raw", "epochs", "raw_epochs", "data", "signals"],
        "raw epoch signal array",
    )
    Y_KEY   = _pick_key(["y", "labels", "label", "Y", "targets"], "labels")
    SUB_KEY = _pick_key(
        ["subjects", "subject", "subject_ids", "subject_id", "groups", "sub_ids"],
        "subject IDs",
    )
    log.info(f"Using keys → signal: '{X_KEY}', labels: '{Y_KEY}', subjects: '{SUB_KEY}'")

    X_np        = raw[X_KEY].astype(np.float32)     # expected (N, 62, T)
    y_np        = raw[Y_KEY].astype(np.int64)
    subjects_np = raw[SUB_KEY]

    if X_np.ndim != 3:
        raise ValueError(
            f"Expected raw epoch array with shape (n_epochs, n_channels, "
            f"n_times), got ndim={X_np.ndim}, shape={X_np.shape}. This script "
            f"needs RAW signal, not precomputed tangent-space features — did "
            f"you point RAW_DATA_PATH at tangent_features_ea.npz by mistake?"
        )

    N, C, T = X_np.shape
    N_CLASSES = int(y_np.max()) + 1
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Classes: {N_CLASSES}")
    log.info(f"Subjects in dataset: {sorted(np.unique(subjects_np).tolist())}")
    assert N_CLASSES == 2, "Sensitivity/specificity below assume binary classification."

    if C != N_CHANNELS:
        log.warning(
            f"⚠️  Loaded channel count ({C}) != configured N_CHANNELS "
            f"({N_CHANNELS}). Update N_CHANNELS at the top of this script."
        )

    approx_epoch_seconds = T / SFREQ
    log.info(
        f"  Epoch length ≈ {approx_epoch_seconds:.2f}s at SFREQ={SFREQ}Hz "
        f"(T={T} samples). Sanity-check this against your true trial window."
    )

    condition3_per_subject_acc = {}
    if os.path.exists(CONDITION3_JSON):
        with open(CONDITION3_JSON, "r") as f:
            c3 = json.load(f)
        for rec in c3.get("fold_results", []):
            condition3_per_subject_acc[str(rec["test_subject"])] = float(rec["test_accuracy"])
        log.info(f"Loaded Condition 3 per-fold accuracies for {len(condition3_per_subject_acc)} subjects.")

    condition4_per_subject_acc = {}
    if os.path.exists(CONDITION4_JSON):
        with open(CONDITION4_JSON, "r") as f:
            c4 = json.load(f)
        for rec in c4.get("fold_results", []):
            condition4_per_subject_acc[str(rec["test_subject"])] = float(rec["post_calibration_acc"])
        log.info(f"Loaded Condition 4 per-fold accuracies for {len(condition4_per_subject_acc)} subjects.")

    # =========================================================================
    # SECTION 4: AUTHENTIC EEGNet ARCHITECTURE (Lawhern et al., 2018)
    # =========================================================================

    class DepthwiseConv2dMaxNorm(nn.Module):
        """Depthwise Conv2D matching Keras DepthwiseConv2D(groups=in_ch)."""
        def __init__(self, in_ch, depth_mult, kernel_size):
            super().__init__()
            self.conv = nn.Conv2d(
                in_ch, in_ch * depth_mult, kernel_size,
                groups=in_ch, bias=False,
            )

        def forward(self, x):
            return self.conv(x)

    class SeparableConv2d(nn.Module):
        """Depthwise (1,16) + Pointwise (1,1), matching Keras SeparableConv2D."""
        def __init__(self, in_ch, out_ch, kernel_size, padding):
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_ch, in_ch, kernel_size, padding=padding,
                groups=in_ch, bias=False,
            )
            self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    class EEGNet(nn.Module):
        """
        Authentic EEGNet (Lawhern et al., 2018).

        Input  : (B, 1, C=n_channels, T=n_times)
        Output : (B, n_classes) logits

        forward_features() exposes the flattened penultimate representation
        (input to the final Dense layer) for the calibration stage.
        """
        def __init__(self, n_channels, n_times, n_classes=2,
                     F1=EEGNET_F1, D=EEGNET_D, F2=EEGNET_F2,
                     kern_length=KERN_LENGTH, dropout=EEGNET_DROPOUT):
            super().__init__()

            # ---- Block 1 ----
            self.conv1 = nn.Conv2d(
                1, F1, (1, kern_length),
                padding=(0, kern_length // 2), bias=False,
            )
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = DepthwiseConv2dMaxNorm(F1, D, (n_channels, 1))
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.elu1 = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)

            # ---- Block 2 ----
            self.separable = SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8))
            self.bn3 = nn.BatchNorm2d(F2)
            self.elu2 = nn.ELU()
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            # ---- Determine flattened penultimate dim analytically ----
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                feat = self._features(dummy)
                self.penultimate_dim = feat.shape[1]

            self.classifier = nn.Linear(self.penultimate_dim, n_classes)

        def _features(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.drop1(x)

            x = self.separable(x)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            x = self.drop2(x)

            return torch.flatten(x, start_dim=1)

        def forward_features(self, x):
            """Penultimate representation, used for calibration-stage feature extraction."""
            return self._features(x)

        def forward(self, x):
            return self.classifier(self._features(x))

    def apply_max_norm_constraints(model: "EEGNet"):
        """
        Keras EEGNet applies kernel_constraint=max_norm to the depthwise
        conv (max_norm=1.0) and the final Dense layer (max_norm=0.25).
        PyTorch has no built-in equivalent, so we clamp weight norms
        in-place after every optimizer.step(), matching the original.
        """
        with torch.no_grad():
            w = model.depthwise.conv.weight
            norms = w.view(w.size(0), -1).norm(dim=1, keepdim=True)
            desired = torch.clamp(norms, max=EEGNET_DEPTHWISE_MAXNORM)
            scale = (desired / (norms + 1e-8)).view(-1, 1, 1, 1)
            w.mul_(scale)

            w2 = model.classifier.weight
            norms2 = w2.norm(dim=1, keepdim=True)
            desired2 = torch.clamp(norms2, max=EEGNET_DENSE_MAXNORM)
            scale2 = desired2 / (norms2 + 1e-8)
            w2.mul_(scale2)

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
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            apply_max_norm_constraints(model)
            total_loss += loss.item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            total_loss += criterion(logits, yb).item()
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def extract_penultimate_features(model, X_np_subset):
        """Frozen-backbone feature extraction, matching Condition 4's tangent-feature role."""
        model.eval()
        X_t = torch.from_numpy(X_np_subset).unsqueeze(1).to(device)  # (N,1,C,T)
        feats = model.forward_features(X_t)
        return feats.cpu().numpy()

    def compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
        }

    def linear_predict(coef, intercept, X):
        logits = X @ coef.T + intercept
        return (logits.ravel() > 0).astype(int)

    def fit_shrinkage_classifier(X_train_pca, y_train, X_cal_pca, y_cal, log):
        """Identical logic to Condition 4 v2's fit_shrinkage_classifier."""
        global_clf = LogisticRegression(
            C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED
        ).fit(X_train_pca, y_train)

        local_clf_full = LogisticRegression(
            C=LOGREG_C, max_iter=5000, random_state=RANDOM_SEED
        ).fit(X_cal_pca, y_cal)

        n_splits = min(SHRINK_CV_FOLDS, np.bincount(y_cal).min())
        n_splits = max(n_splits, 2)
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
        log.info(f"      Selected shrinkage weight (local vs global): {best_shrink:.2f}")

        coef_final = best_shrink * local_clf_full.coef_ + (1 - best_shrink) * global_clf.coef_
        icpt_final = best_shrink * local_clf_full.intercept_ + (1 - best_shrink) * global_clf.intercept_

        return coef_final, icpt_final, best_shrink, global_clf

    def make_loader(X, y, batch_size, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X).unsqueeze(1),  # (N,1,C,T)
            torch.from_numpy(y),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=2, pin_memory=True)

    # =========================================================================
    # SECTION 6: STRICT 29-FOLD LOSO + PRETRAIN + PCA-SHRINKAGE CALIBRATION
    # =========================================================================

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    TEST_FOLDS = len(unique_subjects)

    PRETRAIN_EPOCHS   = 100
    PRETRAIN_BATCH    = 64
    PRETRAIN_LR       = 1e-3
    PRETRAIN_WD       = 1e-4
    PRETRAIN_ES_PAT   = 10
    INTERNAL_VAL_FRAC = 0.10

    log.info(f"\nAll subjects   : {unique_subjects}")
    log.info(f"Running folds  : {TEST_FOLDS} (strict LOSO, all subjects)")
    log.info(f"Calibration    : {CAL_FRACTION*100:.0f}% of held-out subject "
             f"(MATCHED to Condition 4's CAL_FRACTION — verify this is still true)")

    fold_records = []
    all_test_acc = []

    for fold_idx, test_sub in enumerate(unique_subjects):

        fold_start = time.time()
        log.info(f"\n{'='*70}")
        log.info(f"  FOLD {fold_idx + 1}/{TEST_FOLDS}  —  Held-Out Subject: sub-{test_sub}")
        log.info(f"{'='*70}")

        is_holdout = subjects_np == test_sub
        is_train28 = ~is_holdout

        X_train28_raw = X_np[is_train28]; y_train28 = y_np[is_train28]
        X_k_raw       = X_np[is_holdout]; y_k        = y_np[is_holdout]

        log.info(f"  28-subject pool : {X_train28_raw.shape}")
        log.info(f"  Held-out subject: {X_k_raw.shape}")

        # ---- Per-channel/time z-score normalization, fit on 28-subj pool only ----
        mu  = X_train28_raw.mean(axis=(0, 2), keepdims=True)
        std = X_train28_raw.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train28 = ((X_train28_raw - mu) / std).astype(np.float32)
        X_k       = ((X_k_raw - mu) / std).astype(np.float32)

        X_pt_tr, X_pt_val, y_pt_tr, y_pt_val = train_test_split(
            X_train28, y_train28,
            test_size=INTERNAL_VAL_FRAC, stratify=y_train28,
            random_state=RANDOM_SEED,
        )

        pretrain_loader = make_loader(X_pt_tr, y_pt_tr, PRETRAIN_BATCH, shuffle=True)
        ptval_loader    = make_loader(X_pt_val, y_pt_val, PRETRAIN_BATCH, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # =====================================================================
        # STAGE A — Pretrain authentic EEGNet on 28-subject pool
        # =====================================================================
        model = EEGNet(n_channels=C, n_times=T, n_classes=N_CLASSES).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01
        )

        log.info(f"  --- STAGE A: EEGNet Pretraining ({n_params:,} params, "
                 f"penultimate_dim={model.penultimate_dim}) ---")

        best_val_loss, best_state_dict, es_counter = math.inf, None, 0

        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, pretrain_loader, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc = evaluate(model, ptval_loader, criterion)

            if epoch == 1 or epoch % 10 == 0 or epoch == PRETRAIN_EPOCHS:
                log.info(f"    Ep {epoch:03d}/{PRETRAIN_EPOCHS} | Train acc={tr_acc:.3f} "
                         f"| IntVal loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= PRETRAIN_ES_PAT:
                    log.info(f"    Early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state_dict)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        log.info(f"  Pretraining complete. Best internal-val loss={best_val_loss:.4f}")

        # =====================================================================
        # STAGE B — Extract penultimate features (28-subj pool + subject k)
        # =====================================================================
        feat_train28 = extract_penultimate_features(model, X_train28)
        feat_k       = extract_penultimate_features(model, X_k)

        # ---- 15% stratified calibration split on subject k ----
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - CAL_FRACTION), random_state=RANDOM_SEED
        )
        cal_idx, test_idx = next(sss.split(feat_k, y_k))
        assert set(cal_idx.tolist()).isdisjoint(set(test_idx.tolist())), \
            "Leakage detected: D_cal and D_test overlap!"

        feat_cal, y_cal   = feat_k[cal_idx],  y_k[cal_idx]
        feat_test, y_test = feat_k[test_idx], y_k[test_idx]

        log.info(f"  D_cal  : {feat_cal.shape}  D_test : {feat_test.shape}")

        # ---- StandardScaler + PCA fit on 28-subj penultimate features ONLY ----
        scaler = StandardScaler()
        feat_train28_z = scaler.fit_transform(feat_train28)
        feat_cal_z     = scaler.transform(feat_cal)
        feat_test_z    = scaler.transform(feat_test)

        n_components = min(PCA_MAX_COMPONENTS, model.penultimate_dim - 1, feat_train28_z.shape[0] - 1)
        if model.penultimate_dim <= PCA_MAX_COMPONENTS:
            log.warning(
                f"  ⚠️  EEGNet penultimate_dim ({model.penultimate_dim}) <= "
                f"PCA_MAX_COMPONENTS ({PCA_MAX_COMPONENTS}) — PCA is doing little "
                f"or no compression here, unlike Condition 4's 1953→35 reduction. "
                f"This is an expected architectural difference (CNN features vs. "
                f"tangent vectors), not a bug — disclose it in the manuscript."
            )
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_train28_pca = pca.fit_transform(feat_train28_z)
        X_cal_pca     = pca.transform(feat_cal_z)
        X_test_pca    = pca.transform(feat_test_z)
        log.info(f"  PCA: {model.penultimate_dim} -> {n_components} dims "
                 f"(explained var={pca.explained_variance_ratio_.sum()*100:.1f}%)")

        # =====================================================================
        # STAGE C — Shrinkage-blended calibration (identical algorithm to C4)
        # =====================================================================
        coef_final, icpt_final, best_shrink, global_clf = fit_shrinkage_classifier(
            X_train28_pca, y_train28, X_cal_pca, y_cal, log
        )

        pre_cal_preds = global_clf.predict(X_test_pca)
        pre_cal_acc   = float((pre_cal_preds == y_test).mean())

        final_preds = linear_predict(coef_final, icpt_final, X_test_pca)
        best_test_acc = float((final_preds == y_test).mean())

        metrics = compute_binary_metrics(y_test, final_preds)

        c3_acc = condition3_per_subject_acc.get(str(test_sub))
        c4_acc = condition4_per_subject_acc.get(str(test_sub))

        log.info(
            f"  RESULT -> pre_cal_acc(global-only)={pre_cal_acc:.4f}  "
            f"post_cal_acc={best_test_acc:.4f}  (shrink={best_shrink:.2f})  "
            f"sens={metrics['sensitivity']:.4f}  spec={metrics['specificity']:.4f}"
        )
        if c4_acc is not None:
            log.info(f"      Δ vs Condition 4 (sub-{test_sub}: {c4_acc*100:.2f}%) "
                      f"-> {(best_test_acc - c4_acc)*100:+.2f} pp")

        fold_records.append({
            "fold_index": fold_idx,
            "test_subject": str(test_sub),
            "n_cal_epochs": int(feat_cal.shape[0]),
            "n_test_epochs": int(feat_test.shape[0]),
            "penultimate_dim": int(model.penultimate_dim),
            "pca_components_used": int(n_components),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "best_shrink_weight": float(best_shrink),
            "pre_calibration_acc": pre_cal_acc,
            "post_calibration_acc": best_test_acc,
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "condition3_zero_shot_acc": c3_acc,
            "condition4_calibrated_acc": c4_acc,
            "y_true": y_test.tolist(),
            "y_pred": final_preds.tolist(),
        })
        all_test_acc.append(best_test_acc)

        log.info(f"  Fold {fold_idx+1} elapsed: {time.time()-fold_start:.0f}s")

        del model, optimizer, scheduler, best_state_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # SECTION 7: AGGREGATE + 5-CONDITION COMPARISON (incl. paired stats vs C4)
    # =========================================================================

    from scipy.stats import wilcoxon

    mean_acc = float(np.mean(all_test_acc))
    std_acc  = float(np.std(all_test_acc))
    mean_sens = float(np.mean([f["sensitivity"] for f in fold_records]))
    mean_spec = float(np.mean([f["specificity"] for f in fold_records]))
    mean_f1   = float(np.mean([f["f1"] for f in fold_records]))

    paired_c4 = [
        (f["post_calibration_acc"], f["condition4_calibrated_acc"])
        for f in fold_records if f["condition4_calibrated_acc"] is not None
    ]
    wilcoxon_result = None
    if len(paired_c4) == TEST_FOLDS:
        eegnet_calib = np.array([p[0] for p in paired_c4])
        mamba_calib  = np.array([p[1] for p in paired_c4])
        diffs = mamba_calib - eegnet_calib
        if np.any(diffs != 0):
            stat, pval = wilcoxon(mamba_calib, eegnet_calib)
            n = len(diffs)
            rank_biserial = (np.sum(diffs > 0) - np.sum(diffs < 0)) / n
            wilcoxon_result = {
                "statistic": float(stat), "p_value": float(pval),
                "n_subjects": int(n), "rank_biserial_effect_size": float(rank_biserial),
                "mean_diff_c4_minus_eegnet_calib": float(diffs.mean()),
            }
            log.info(f"\n  Wilcoxon (Condition 4 vs EEGNet+Calib): "
                      f"stat={stat:.2f} p={pval:.5f} rank-biserial={rank_biserial:.3f}")
    else:
        log.warning(
            f"  {CONDITION4_JSON} did not provide per-subject accuracies for all "
            f"{TEST_FOLDS} folds — skipping paired significance test. Merge in "
            f"chat once both JSONs are available."
        )

    log.info(f"\n{'='*70}")
    log.info(f"  CONDITION 1b — EEGNet + 15% PCA-SHRINKAGE CALIBRATION, FULL RESULTS")
    log.info(f"{'='*70}")
    log.info(f"  Mean ± Std Acc : {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(f"  Mean Sens/Spec/F1 : {mean_sens:.4f} / {mean_spec:.4f} / {mean_f1:.4f}")
    log.info(f"{'='*70}")
    log.info(f"  5-CONDITION COMPARISON SUMMARY")
    log.info(f"  {'Condition':<50}{'Mean Acc':>12}")
    log.info(f"  {'-'*62}")
    log.info(f"  {'Condition 1: EEGNet, zero-shot':<50}{CONDITION1_ZEROSHOT_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 1b: EEGNet + 15% Calibration (THIS RUN)':<50}{mean_acc*100:>11.2f}%")
    log.info(f"  {'Condition 2: Mamba, zero-shot, no EA':<50}{CONDITION2_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 3: SpatioMamba + EA, zero-shot':<50}{CONDITION3_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'Condition 4: SpatioMamba + EA + Calibration':<50}{CONDITION4_BASELINE_ACC*100:>11.2f}%")
    log.info(f"  {'-'*62}")
    log.info(f"  Δ (Condition 4 − Condition 1b) : {(CONDITION4_BASELINE_ACC-mean_acc)*100:+.2f} pp")
    log.info(
        "\n  INTERPRETATION GUIDE (fill in after this run completes):\n"
        "    If Condition 1b << Condition 4 (~68%): supports 'Mamba/Riemannian\n"
        "    contributes beyond calibration alone' — but remember Condition 4's\n"
        "    current codebase is spatial-only (no temporal Mamba branch active),\n"
        "    so even a clean win here does NOT yet support the paper's\n"
        "    'Dual-Branch Spatiotemporal Mamba' title until Task 2 is built and\n"
        "    run. If Condition 1b is close to or exceeds Condition 4, the\n"
        "    calibration step — not architecture — explains most of the jump,\n"
        "    and the manuscript's architectural contribution claim needs to be\n"
        "    substantially walked back regardless of which way this number lands."
    )
    log.info(f"{'='*70}\n")

    results_payload = {
        "condition": "Condition 1b: EEGNet (Lawhern et al. 2018) + 15% PCA-Shrinkage Calibration",
        "n_folds": TEST_FOLDS,
        "sfreq_used": SFREQ,
        "kern_length_used": KERN_LENGTH,
        "hyperparameters": {
            "cal_fraction": CAL_FRACTION, "pca_max_components": PCA_MAX_COMPONENTS,
            "logreg_C": LOGREG_C, "shrink_grid": SHRINK_GRID,
            "shrink_cv_folds": SHRINK_CV_FOLDS, "random_seed": RANDOM_SEED,
            "eegnet_F1": EEGNET_F1, "eegnet_D": EEGNET_D, "eegnet_F2": EEGNET_F2,
            "eegnet_dropout": EEGNET_DROPOUT,
        },
        "fold_results": fold_records,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_sensitivity": mean_sens,
        "mean_specificity": mean_spec,
        "mean_f1": mean_f1,
        "condition1_zeroshot_acc": CONDITION1_ZEROSHOT_ACC,
        "condition2_baseline_acc": CONDITION2_BASELINE_ACC,
        "condition3_baseline_acc": CONDITION3_BASELINE_ACC,
        "condition4_baseline_acc": CONDITION4_BASELINE_ACC,
        "wilcoxon_vs_condition4": wilcoxon_result,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    log.info(f"  Saved: {OUTPUT_JSON}")

    volume.commit()
    log.info("  volume.commit() complete.")

    return {
        "mean_accuracy": mean_acc, "std_accuracy": std_acc,
        "mean_sensitivity": mean_sens, "mean_specificity": mean_spec,
        "mean_f1": mean_f1, "wilcoxon_vs_condition4": wilcoxon_result,
        "output_path": OUTPUT_JSON,
    }


# =============================================================================
# SECTION 8: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  Condition 1b — EEGNet + 15% Few-Shot PCA-Shrinkage Calibration")
    print("  Strict 29-Subject LOSO Benchmark (RAW epoched EEG)")
    print("="*70)
    print(
        "  ⚠️  Before this run means anything: confirm SFREQ, RAW_DATA_PATH,\n"
        "      and that CAL_FRACTION truly matches whichever Condition 4 run\n"
        "      produced your reported 67.79%.\n"
    )

    results = run_condition1_eegnet_calibration.remote()

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<28} : {val}")
    print("="*70 + "\n")