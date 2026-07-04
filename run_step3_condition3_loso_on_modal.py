# =============================================================================
# run_step3_condition3_loso_on_modal.py
# Condition 3: SpatiotemporalRiemannianMamba + Euclidean Alignment
# Strict 29-Subject LOSO Zero-Shot Benchmark
#
# IMPORTANT — data source note:
#   This script trains on EA-aligned RAW signals (62, 251), loaded from
#   processed_eeg_all_subjects.npz, NOT on the precomputed
#   tangent_features_ea.npz. Reasons:
#     1. SpatiotemporalRiemannianMamba's temporal (Mamba SSM) branch needs
#        the raw time series; the precomputed 1953-d tangent vectors have
#        no time axis left for it to operate on.
#     2. tangent_features_ea.npz was built with a single global Frechet
#        mean across all 29 subjects (Step 2.1), which leaks test-subject
#        statistics into the headline LOSO number. EA on raw signals is
#        strictly per-subject (train and test alike) and the model's own
#        TangentSpaceLayer uses a per-sample identity-reference matrix-log
#        with no fitted global mean -- so this path is leak-free per fold.
#   tangent_features_ea.npz remains useful as an offline sanity-check
#   artifact, just not as the source of the reported Condition 3 accuracy.
#
# Usage: modal run run_step3_condition3_loso_on_modal.py
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE  (identical to run_loso_trainer_on_modal.py)
# =============================================================================

app    = modal.App("bci-condition3-loso")
volume = modal.Volume.from_name("eeg-data-vol")

DATA_PATH    = "/data/processed_eeg_all_subjects.npz"
OUTPUT_JSON  = "/data/results_condition3_ea_zeroshot.json"
VOLUME_PATH  = "/data"

CONDITION2_BASELINE_ACC = 0.5553   # Mamba, zero-shot LOSO, no EA

loso_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "packaging",
        "wheel",
        "setuptools>=68",
        "ninja",
    )
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.39.0",
        "numpy<2",
        "einops",
        "scikit-learn",
        "mne",
        "tqdm",
    )
    .run_commands(
        "CC=gcc CXX=g++ pip install --no-build-isolation causal-conv1d==1.4.0",
        "CC=gcc CXX=g++ pip install --no-build-isolation mamba-ssm==2.2.4",
    )
)


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================

@app.function(
    image=loso_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=32768,
)
def run_condition3_loso():

    import numpy  as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, f1_score
    import logging, time, math, json, copy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("condition3-loso")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # =========================================================================
    # SECTION 3: DATA LOADING (raw signals, NOT precomputed tangent features)
    # =========================================================================
    log.info(f"Loading {DATA_PATH} ...")
    raw          = np.load(DATA_PATH, allow_pickle=True)
    X_np         = raw["X"].astype(np.float32)
    y_np         = raw["y"].astype(np.int64)
    subjects_np  = raw["subjects"]

    N, N_CH, N_T = X_np.shape
    N_CLASSES    = int(y_np.max()) + 1
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Classes: {N_CLASSES}")
    log.info(f"Subjects in dataset: {sorted(np.unique(subjects_np).tolist())}")
    assert N_CLASSES == 2, "Sensitivity/specificity below assume binary classification."

    # =========================================================================
    # SECTION 3.1: EUCLIDEAN ALIGNMENT  (per subject, leak-free, pre-LOSO)
    # =========================================================================
    class EuclideanAlignment:
        def __init__(self, eps: float = 1e-6):
            self.eps = eps
            self.R_inv_sqrt_ = None

        @staticmethod
        def _reference_covariance(X):
            M, C, T = X.shape
            cov = np.einsum('mct,mdt->mcd', X, X) / (T - 1)
            return cov.mean(axis=0)

        def _inv_sqrt(self, R_bar):
            C = R_bar.shape[0]
            R_reg = R_bar + self.eps * np.eye(C, dtype=R_bar.dtype)
            eigvals, eigvecs = np.linalg.eigh(R_reg)
            eigvals = np.clip(eigvals, a_min=self.eps, a_max=None)
            R_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            return 0.5 * (R_inv_sqrt + R_inv_sqrt.T)

        def fit_transform(self, X_subject):
            R_bar = self._reference_covariance(X_subject)
            self.R_inv_sqrt_ = self._inv_sqrt(R_bar)
            return np.einsum('dc,mct->mdt', self.R_inv_sqrt_, X_subject)

    def align_dataset_per_subject(X, subjects, eps: float = 1e-6):
        X_aligned = np.empty_like(X, dtype=np.float32)
        for sub in np.unique(subjects):
            mask = subjects == sub
            ea = EuclideanAlignment(eps=eps)
            X_aligned[mask] = ea.fit_transform(X[mask]).astype(np.float32)
        return X_aligned

    log.info("Applying Euclidean Alignment (per-subject, pre-LOSO)...")
    X_np = align_dataset_per_subject(X_np, subjects_np, eps=1e-6)
    log.info("Euclidean Alignment complete.")

    # =========================================================================
    # SECTION 4: ARCHITECTURE — SpatiotemporalRiemannianMamba
    #   (unchanged from Condition 2: d_model=32, n_layers=2, dropout=0.6 —
    #    same capacity, so EA's effect is isolated cleanly.)
    # =========================================================================

    class CovarianceLayer(nn.Module):
        def __init__(self, n_channels: int, reg: float = 1e-4):
            super().__init__()
            self.reg = reg
            self.register_buffer("eye", torch.eye(n_channels))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x - x.mean(dim=-1, keepdim=True)
            n = x.shape[-1]
            cov = torch.bmm(x, x.mT) / (n - 1)
            cov = cov + self.reg * self.eye.unsqueeze(0)
            return cov

    class TangentSpaceLayer(nn.Module):
        def __init__(self, n_channels: int):
            super().__init__()
            idx = torch.triu_indices(n_channels, n_channels)
            self.register_buffer("triu_r", idx[0])
            self.register_buffer("triu_c", idx[1])

        def forward(self, S: torch.Tensor) -> torch.Tensor:
            L, V = torch.linalg.eigh(S)
            L    = torch.clamp(L, min=1e-7)
            log_S   = V @ torch.diag_embed(torch.log(L)) @ V.mT
            tangent = log_S[:, self.triu_r, self.triu_c]
            return tangent

    class SpatiotemporalRiemannianMamba(nn.Module):
        def __init__(
            self,
            n_channels : int   = 62,
            n_times    : int   = 251,
            d_model    : int   = 32,
            n_layers   : int   = 2,
            n_classes  : int   = 2,
            dropout    : float = 0.6,
        ):
            super().__init__()
            from mamba_ssm import Mamba

            tang_dim = n_channels * (n_channels + 1) // 2   # 1953 for C=62

            self.cov_layer     = CovarianceLayer(n_channels)
            self.tangent_layer = TangentSpaceLayer(n_channels)
            self.spatial_proj  = nn.Sequential(
                nn.Linear(tang_dim, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )

            self.temporal_in   = nn.Linear(n_channels, d_model)
            self.mamba_blocks  = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ])
            self.mamba_norms   = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)
            ])
            self.temp_drop     = nn.Dropout(dropout)

            self.head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cov    = self.cov_layer(x)
            tang   = self.tangent_layer(cov)
            s_feat = self.spatial_proj(tang)

            h = x.permute(0, 2, 1)
            h = self.temporal_in(h)
            for mamba, norm in zip(self.mamba_blocks, self.mamba_norms):
                h = h + mamba(norm(h))
            t_feat = self.temp_drop(h.mean(dim=1))

            return self.head(torch.cat([s_feat, t_feat], dim=-1))

    # =========================================================================
    # SECTION 5: TRAINING / EVAL UTILITIES
    # =========================================================================

    def train_one_epoch(model, loader, optimizer, criterion, scaler_amp):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(Xb)
                loss   = criterion(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()
            n_correct  += (logits.argmax(1) == yb).sum().item()
            n_total    += yb.size(0)
        return total_loss / len(loader), n_correct / n_total

    @torch.no_grad()
    def evaluate_with_predictions(model, loader, criterion):
        """Returns loss, acc, and full y_true/y_pred arrays for metrics."""
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

    # =========================================================================
    # SECTION 6: STRICT 29-FOLD LOSO CROSS-VALIDATION
    # =========================================================================

    unique_subjects = sorted(np.unique(subjects_np).tolist())
    TEST_FOLDS   = len(unique_subjects)   # full 29-fold run
    N_EPOCHS     = 50
    BATCH_SIZE   = 64
    LR           = 1e-4
    WEIGHT_DECAY = 1e-2
    ES_PATIENCE  = 7

    log.info(f"\nAll subjects  : {unique_subjects}")
    log.info(f"Running folds : {TEST_FOLDS} (strict LOSO, all subjects)")
    log.info(f"Epochs/fold   : {N_EPOCHS} (early stop patience={ES_PATIENCE})")
    log.info(f"Batch         : {BATCH_SIZE} | LR: {LR} | WD: {WEIGHT_DECAY}")

    fold_records = []   # one dict per subject, for the JSON export
    all_te_acc   = []

    for fold_idx, test_sub in enumerate(unique_subjects):

        fold_start = time.time()
        log.info(f"\n{'═'*70}")
        log.info(f"  FOLD {fold_idx + 1}/{TEST_FOLDS}  —  Left-Out Subject: sub-{test_sub}")
        log.info(f"{'═'*70}")

        # ---- Strict split: 28 train subjects, 1 test subject ----
        is_test  = subjects_np == test_sub
        is_train = ~is_test

        X_tr_raw = X_np[is_train];  y_tr = y_np[is_train]
        X_te_raw = X_np[is_test];   y_te = y_np[is_test]

        log.info(f"  Train : {X_tr_raw.shape}  ({y_tr.shape[0]} epochs, 28 subjects)")
        log.info(f"  Test  : {X_te_raw.shape}  ({y_te.shape[0]} epochs, subject {test_sub} ONLY)")

        # z-score per channel, fit on TRAIN ONLY (no leakage)
        n_tr, n_te = X_tr_raw.shape[0], X_te_raw.shape[0]
        X_tr_2d = X_tr_raw.transpose(0, 2, 1).reshape(-1, N_CH)
        X_te_2d = X_te_raw.transpose(0, 2, 1).reshape(-1, N_CH)

        sc      = StandardScaler()
        X_tr_2d = sc.fit_transform(X_tr_2d)
        X_te_2d = sc.transform(X_te_2d)

        X_tr = X_tr_2d.reshape(n_tr, N_T, N_CH).transpose(0, 2, 1).astype(np.float32)
        X_te = X_te_2d.reshape(n_te, N_T, N_CH).transpose(0, 2, 1).astype(np.float32)

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        criterion = nn.CrossEntropyLoss()

        model = SpatiotemporalRiemannianMamba(
            n_channels=N_CH, n_times=N_T,
            d_model=32, n_layers=2,
            n_classes=N_CLASSES, dropout=0.6,
        ).to(device)

        n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
        amp_scaler = torch.cuda.amp.GradScaler()

        log.info(f"\n  ┌── SpatioMamba + EA  ({n_params:,} params) ──")

        best_te_loss   = math.inf
        best_te_acc    = 0.0
        best_state_dict = None
        best_y_true, best_y_pred = None, None
        es_counter     = 0
        stopped_epoch  = N_EPOCHS

        for epoch in range(1, N_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, amp_scaler)
            scheduler.step()
            te_loss, te_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, criterion)

            log.info(
                f"  │  Ep {epoch:02d}/{N_EPOCHS}"
                f"  |  Train  loss={tr_loss:.4f}  acc={tr_acc:.3f}"
                f"  |  Test   loss={te_loss:.4f}  acc={te_acc:.3f}"
                f"  |  LR={scheduler.get_last_lr()[0]:.2e}"
            )

            if te_loss < best_te_loss:
                best_te_loss    = te_loss
                best_te_acc     = te_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                best_y_true, best_y_pred = y_true, y_pred
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= ES_PATIENCE:
                    stopped_epoch = epoch
                    log.info(f"  │  ⏹  Early stop at epoch {epoch} (patience={ES_PATIENCE})")
                    break

        metrics = compute_binary_metrics(best_y_true, best_y_pred)

        log.info(
            f"  └── BEST  →  loss={best_te_loss:.4f}  acc={best_te_acc:.4f}  "
            f"sens={metrics['sensitivity']:.4f}  spec={metrics['specificity']:.4f}  "
            f"f1={metrics['f1']:.4f}  (stopped @ ep {stopped_epoch})"
        )

        fold_records.append({
            "fold_index"      : fold_idx,
            "test_subject"    : str(test_sub),
            "n_train_epochs"  : int(n_tr),
            "n_test_epochs"   : int(n_te),
            "stopped_epoch"   : int(stopped_epoch),
            "test_loss"       : float(best_te_loss),
            "test_accuracy"   : float(best_te_acc),
            "sensitivity"     : metrics["sensitivity"],
            "specificity"     : metrics["specificity"],
            "f1"              : metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        })
        all_te_acc.append(best_te_acc)

        elapsed = time.time() - fold_start
        log.info(f"  Fold {fold_idx+1} elapsed: {elapsed:.0f}s")

        del model, optimizer, scheduler, amp_scaler, best_state_dict
        torch.cuda.empty_cache()

    # =========================================================================
    # SECTION 7: AGGREGATE REPORT + COMPARISON VS CONDITION 2
    # =========================================================================

    mean_acc = float(np.mean(all_te_acc))
    std_acc  = float(np.std(all_te_acc))
    mean_sens = float(np.mean([f["sensitivity"] for f in fold_records]))
    mean_spec = float(np.mean([f["specificity"] for f in fold_records]))
    mean_f1   = float(np.mean([f["f1"] for f in fold_records]))
    delta_vs_c2 = mean_acc - CONDITION2_BASELINE_ACC

    log.info(f"\n{'═'*70}")
    log.info(f"  CONDITION 3 — FULL {TEST_FOLDS}-FOLD LOSO RESULTS")
    log.info(f"{'═'*70}")
    log.info(f"  Per-fold accuracy : {[f'{a:.4f}' for a in all_te_acc]}")
    log.info(f"  ──────────────────────────────────────────────────────────")
    log.info(f"  Mean ± Std Acc    : {mean_acc:.4f} ± {std_acc:.4f}")
    log.info(f"  Mean Sensitivity  : {mean_sens:.4f}")
    log.info(f"  Mean Specificity  : {mean_spec:.4f}")
    log.info(f"  Mean F1           : {mean_f1:.4f}")
    log.info(f"{'═'*70}")
    log.info(f"  COMPARISON vs Condition 2 (Mamba, zero-shot, no EA)")
    log.info(f"  Condition 2 accuracy : {CONDITION2_BASELINE_ACC*100:.2f}%")
    log.info(f"  Condition 3 accuracy : {mean_acc*100:.2f}%")
    log.info(f"  Δ (Condition 3 − 2)  : {delta_vs_c2*100:+.2f} pp")
    log.info(f"{'═'*70}\n")

    # =========================================================================
    # SECTION 8: SAVE RESULTS TO VOLUME + COMMIT
    # =========================================================================

    results_payload = {
        "condition"              : "Condition 3: SpatiotemporalRiemannianMamba + Euclidean Alignment",
        "n_folds"                : TEST_FOLDS,
        "hyperparameters"        : {
            "d_model": 32, "n_layers": 2, "dropout": 0.6,
            "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "es_patience": ES_PATIENCE,
        },
        "fold_results"           : fold_records,
        "mean_accuracy"          : mean_acc,
        "std_accuracy"           : std_acc,
        "mean_sensitivity"       : mean_sens,
        "mean_specificity"       : mean_spec,
        "mean_f1"                : mean_f1,
        "condition2_baseline_acc": CONDITION2_BASELINE_ACC,
        "delta_vs_condition2"    : delta_vs_c2,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    log.info(f"  Saved  : {OUTPUT_JSON}")

    volume.commit()
    log.info("  ✓ volume.commit() complete — results_condition3_ea_zeroshot.json")
    log.info("    is now durably available on eeg-data-vol for Phase 3 figures.")

    return {
        "mean_accuracy"          : mean_acc,
        "std_accuracy"           : std_acc,
        "mean_sensitivity"       : mean_sens,
        "mean_specificity"       : mean_spec,
        "mean_f1"                : mean_f1,
        "condition2_baseline_acc": CONDITION2_BASELINE_ACC,
        "delta_vs_condition2"    : delta_vs_c2,
        "output_path"            : OUTPUT_JSON,
    }


# =============================================================================
# SECTION 9: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  Condition 3 — SpatiotemporalRiemannianMamba + Euclidean Alignment")
    print("  Strict 29-Subject LOSO Zero-Shot Benchmark")
    print("="*70 + "\n")

    results = run_condition3_loso.remote()

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<25} : {val}")
    print("="*70)
    print(
        "\n  Full 29-fold Condition 3 run complete."
        "\n  results_condition3_ea_zeroshot.json committed to eeg-data-vol.\n"
    )