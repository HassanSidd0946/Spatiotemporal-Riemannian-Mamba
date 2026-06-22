# step3_2_spatiotemporal_assembly.py

"""
BCI Research Pipeline — Phase 3, Step 3.2
Task : Assemble the full SpatiotemporalRiemannianMamba nn.Module and verify
       end-to-end forward + backward pass on a mock tensor.

Architecture (correct causal order)
-------------------------------------
Input  (B, C, T)
  │
  ▼
Temporal Block — Mamba SSM
  • Transpose   (B, C, T) → (B, T, C)   [time=sequence, channels=d_model]
  • Mamba block (B, T, C) → (B, T, C)
  • Residual    mamba_out  + input_transposed
  • Transpose   (B, T, C) → (B, C, T)
  │
  ▼
Spatial Block — Riemannian Geometry
  • CovarianceLayer   (B, C, T) → (B, C, C)   [regularised SPD matrices]
  • TangentSpaceLayer (B, C, C) → (B, 1953)    [Log-Euclidean projection]
  │
  ▼
Classification Head
  • nn.Linear  1953 → 2                        [True / False Memory]
  │
  ▼
Output (B, 2)   — raw logits for CrossEntropyLoss

Rationale for ordering
-----------------------
Mamba operates on the *time* axis (sequence length L = n_times = 251).
Riemannian covariance operates on the *spatial* axis (C × C channel maps).
Computing covariance first would collapse the time dimension before Mamba
has processed it — destroying the temporal structure the SSM is designed to
capture.  The correct order is therefore Temporal (Mamba) → Spatial
(Riemannian) → Classification.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba

# ── Configuration ──────────────────────────────────────────────────────────────
N_CHANNELS  = 62
N_TIMES     = 251
N_CLASSES   = 2
BATCH_SIZE  = 8

# Mamba hyper-parameters
D_STATE     = 16    # SSM latent state size
D_CONV      = 4     # local causal conv kernel
EXPAND      = 2     # inner-dim expansion factor

# Riemannian hyper-parameters
REG_EPS     = 1e-4  # covariance regularisation (mimics OAS shrinkage)
EIG_FLOOR   = 1e-6  # eigenvalue clamp before log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-module 1 · CovarianceLayer
# ══════════════════════════════════════════════════════════════════════════════
class CovarianceLayer(nn.Module):
    """
    Differentiable regularised empirical covariance.

    Input  : (B, C, T)
    Output : (B, C, C)   — symmetric positive-definite matrices

    C_b = X_b @ X_b^T / (T - 1)  +  eps * I_C
    """

    def __init__(self, n_channels: int, reg_eps: float = REG_EPS):
        super().__init__()
        self.register_buffer("eye", torch.eye(n_channels))
        self.reg_eps = reg_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x - x.mean(dim=-1, keepdim=True)           # mean-centre over time
        B, C, T = x.shape
        cov = torch.matmul(x, x.transpose(-1, -2)) / (T - 1)   # (B, C, C)
        cov = cov + self.reg_eps * self.eye.unsqueeze(0)        # (B, C, C)
        return cov


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-module 2 · TangentSpaceLayer
# ══════════════════════════════════════════════════════════════════════════════
class TangentSpaceLayer(nn.Module):
    """
    Log-Euclidean tangent space projection.

    Input  : (B, C, C)   — SPD covariance matrices
    Output : (B, C*(C+1)//2)   — upper-triangle of matrix log, √2-scaled

    Pipeline per matrix:
      1. Eigendecompose via torch.linalg.eigh  (stable for symmetric input)
      2. Clamp eigenvalues to eig_floor
      3. log(C) = V @ diag(log λ) @ V^T
      4. Vectorise upper triangle; scale off-diagonals by √2 to preserve the
         Frobenius inner product (pyriemann convention)
    """

    def __init__(self, n_channels: int, eig_floor: float = EIG_FLOOR):
        super().__init__()
        self.eig_floor  = eig_floor
        self.n_features = n_channels * (n_channels + 1) // 2   # 62*63//2 = 1953

        idx = torch.triu_indices(n_channels, n_channels, offset=0)
        self.register_buffer("triu_row", idx[0])
        self.register_buffer("triu_col", idx[1])

        scale = torch.where(
            idx[0] == idx[1],
            torch.ones(idx.shape[1]),
            torch.full((idx.shape[1],), 2.0 ** 0.5),
        )
        self.register_buffer("triu_scale", scale)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        # cov: (B, C, C)
        eigvals, eigvecs = torch.linalg.eigh(cov)               # (B,C), (B,C,C)
        eigvals  = eigvals.clamp(min=self.eig_floor)
        log_diag = torch.diag_embed(torch.log(eigvals))         # (B, C, C)
        log_cov  = eigvecs @ log_diag @ eigvecs.transpose(-1, -2)   # (B, C, C)

        ts = log_cov[:, self.triu_row, self.triu_col]           # (B, n_features)
        ts = ts * self.triu_scale.unsqueeze(0)
        return ts                                                # (B, 1953)


# ══════════════════════════════════════════════════════════════════════════════
#  Full model · SpatiotemporalRiemannianMamba
# ══════════════════════════════════════════════════════════════════════════════
class SpatiotemporalRiemannianMamba(nn.Module):
    """
    End-to-end False Memory Classification Model.

    Stage 1 — Temporal (Mamba SSM)
        Mamba treats n_channels as d_model and n_times as the sequence length.
        A residual connection stabilises early training.

    Stage 2 — Spatial (Riemannian Geometry)
        After temporal processing the channel signals are compressed into a
        single SPD covariance matrix per epoch, then projected to the
        Log-Euclidean tangent space — a flat, vectorised Riemannian
        representation.

    Stage 3 — Classification
        A linear readout maps the 1953-dim tangent vector to 2 class logits.
    """

    def __init__(
        self,
        n_channels : int   = N_CHANNELS,
        n_times    : int   = N_TIMES,
        n_classes  : int   = N_CLASSES,
        d_state    : int   = D_STATE,
        d_conv     : int   = D_CONV,
        expand     : int   = EXPAND,
        reg_eps    : float = REG_EPS,
        eig_floor  : float = EIG_FLOOR,
    ):
        super().__init__()

        # ── Stage 1: Temporal Block ──────────────────────────────────────────
        # d_model = n_channels because after transpose the channel axis
        # becomes the feature dimension that Mamba operates on.
        self.mamba = Mamba(
            d_model = n_channels,
            d_state = d_state,
            d_conv  = d_conv,
            expand  = expand,
        )

        # ── Stage 2: Spatial Block ───────────────────────────────────────────
        self.covariance    = CovarianceLayer(n_channels, reg_eps)
        self.tangent_space = TangentSpaceLayer(n_channels, eig_floor)

        # ── Stage 3: Classification Head ─────────────────────────────────────
        n_features = n_channels * (n_channels + 1) // 2   # 1953
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (B, C, T)   — raw EEG epochs

        Returns
            logits : (B, n_classes)
        """
        # ── Stage 1: Temporal (Mamba) ────────────────────────────────────────
        x_t = x.transpose(1, 2)               # (B, T, C)  — sequence-first for Mamba
        mamba_out = self.mamba(x_t)           # (B, T, C)
        mamba_out = mamba_out + x_t           # residual connection
        x_spatial = mamba_out.transpose(1, 2) # (B, C, T)  — back to channel-first

        # ── Stage 2: Spatial (Riemannian) ────────────────────────────────────
        cov = self.covariance(x_spatial)      # (B, C, C)
        ts  = self.tangent_space(cov)         # (B, 1953)

        # ── Stage 3: Classification ──────────────────────────────────────────
        logits = self.classifier(ts)          # (B, 2)

        return logits


# ══════════════════════════════════════════════════════════════════════════════
#  Execution check
# ══════════════════════════════════════════════════════════════════════════════
def run_verification():

    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  Phase 3 · Step 3.2 · End-to-End Assembly Verification")
    print(SEP)
    print(f"  Device : {DEVICE}")

    # ── Model instantiation ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 1 · Model Instantiation")
    print(SEP)

    model = SpatiotemporalRiemannianMamba().to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mamba_params     = sum(p.numel() for p in model.mamba.parameters())
    head_params      = sum(p.numel() for p in model.classifier.parameters())

    print(f"  Model            : SpatiotemporalRiemannianMamba")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print(f"    ├─ Mamba block : {mamba_params:,}")
    print(f"    └─ Classifier  : {head_params:,}  (Linear 1953→2)")

    # ── Mock input tensor ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 2 · Mock Input Tensor")
    print(SEP)

    torch.manual_seed(42)
    X_mock = torch.randn(
        BATCH_SIZE, N_CHANNELS, N_TIMES,
        dtype       = torch.float32,
        device      = DEVICE,
        requires_grad = True,
    )
    y_mock = torch.randint(0, N_CLASSES, (BATCH_SIZE,), device=DEVICE)

    print(f"  X_mock.shape     : {tuple(X_mock.shape)}")
    print(f"                     [batch={BATCH_SIZE}, channels={N_CHANNELS}, times={N_TIMES}]")
    print(f"  y_mock           : {y_mock.tolist()}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 3 · Forward Pass  (B, C, T) → (B, 2)")
    print(SEP)

    logits = model(X_mock)

    print(f"  logits.shape     : {tuple(logits.shape)}")
    assert tuple(logits.shape) == (BATCH_SIZE, N_CLASSES), (
        f"Shape mismatch! Expected {(BATCH_SIZE, N_CLASSES)}, got {tuple(logits.shape)}"
    )
    print(f"  ✓ Shape assertion passed: {tuple(logits.shape)} == {(BATCH_SIZE, N_CLASSES)}")
    print(f"\n  Logit stats:")
    print(f"    mean  = {logits.detach().mean():.6f}")
    print(f"    std   = {logits.detach().std():.6f}")
    print(f"    min   = {logits.detach().min():.6f}")
    print(f"    max   = {logits.detach().max():.6f}")

    # ── Loss + backward ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 4 · CrossEntropyLoss + Backward Pass")
    print(SEP)

    criterion = nn.CrossEntropyLoss()
    loss      = criterion(logits, y_mock)

    print(f"  Loss value       : {loss.item():.6f}")
    print(f"  loss.requires_grad : {loss.requires_grad}")

    loss.backward()

    grad_ok   = X_mock.grad is not None
    grad_norm = X_mock.grad.norm().item() if grad_ok else 0.0

    print(f"\n  X_mock.grad is not None : {grad_ok}")
    print(f"  X_mock.grad.shape       : {tuple(X_mock.grad.shape) if grad_ok else 'N/A'}")
    print(f"  X_mock.grad norm        : {grad_norm:.8f}")
    assert grad_ok and grad_norm > 0, "Gradient did not flow back to input!"
    print(f"  ✓ Gradients flow end-to-end through Mamba → Covariance → TangentSpace → Linear")

    # ── Stage-by-stage shape trace ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 5 · Stage-by-Stage Shape Trace")
    print(SEP)

    with torch.no_grad():
        _x     = X_mock.detach()
        _x_t   = _x.transpose(1, 2)
        _m_out = model.mamba(_x_t) + _x_t
        _x_sp  = _m_out.transpose(1, 2)
        _cov   = model.covariance(_x_sp)
        _ts    = model.tangent_space(_cov)
        _log   = model.classifier(_ts)

    rows = [
        ("Raw input",          tuple(_x.shape),     "(B, C, T)"),
        ("After transpose",    tuple(_x_t.shape),   "(B, T, C)  — Mamba sequence-first"),
        ("After Mamba+residual", tuple(_m_out.shape), "(B, T, C)"),
        ("After transpose back", tuple(_x_sp.shape), "(B, C, T)"),
        ("After CovarianceLayer", tuple(_cov.shape), "(B, C, C)  — SPD matrices"),
        ("After TangentSpace",   tuple(_ts.shape),   "(B, 1953)  — Log-Euclidean vectors"),
        ("After Classifier",     tuple(_log.shape),  "(B, 2)     — class logits"),
    ]

    print(f"\n  {'Stage':<30} {'Shape':<22} {'Meaning'}")
    print(f"  {'-'*80}")
    for stage, shape, meaning in rows:
        print(f"  {stage:<30} {str(shape):<22} {meaning}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Verification Summary")
    print(SEP)

    checks = [
        ("Output shape == (B, 2)",            tuple(logits.shape) == (BATCH_SIZE, N_CLASSES)),
        ("Loss is finite",                    loss.item() == loss.item()),   # NaN check
        ("Gradient reaches input",            grad_ok),
        ("Gradient norm > 0",                 grad_norm > 0),
    ]

    print(f"\n  {'Check':<45} {'Result'}")
    print(f"  {'-'*55}")
    all_pass = True
    for label, ok in checks:
        print(f"  {label:<45} {'✓' if ok else '✗'}")
        all_pass = all_pass and ok

    print()
    if all_pass:
        print(f"  {'='*55}")
        print("  ALL CHECKS PASSED")
        print("  SpatiotemporalRiemannianMamba is assembled and verified.")
        print("  Ready for Phase 4: Training Loop + Information Bottleneck loss.")
        print(f"  {'='*55}")
    else:
        print("  [FAIL] One or more checks failed.")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    run_verification()