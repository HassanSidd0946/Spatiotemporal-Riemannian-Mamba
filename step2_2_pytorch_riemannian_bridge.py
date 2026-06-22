# step2_2_pytorch_riemannian_bridge.py

"""
BCI Research Pipeline — Phase 2, Step 2.2
Task : PyTorch–Riemannian Differentiable Bridge
       Build nn.Module layers for Covariance + Tangent Space projection
       that are fully differentiable (autograd-compatible) and GPU-ready.

Input : PyTorch tensor  (batch_size, channels, times)
Output: Tangent vectors (batch_size, 1953)   — ready for downstream Mamba blocks
"""

import torch
import torch.nn as nn

# ── Configuration ─────────────────────────────────────────────────────────────
N_CHANNELS  = 62
N_TIMES     = 251
BATCH_SIZE  = 8
REG_EPS     = 1e-4    # regularization strength (mimics OAS shrinkage)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  Module 1 · CovarianceLayer
# ══════════════════════════════════════════════════════════════════════════════
class CovarianceLayer(nn.Module):
    """
    Compute regularised empirical spatial covariance matrices in PyTorch.

    For an input X of shape (B, C, T):

        C = (1 / (T-1)) * X @ X^T   +   eps * I_C

    The epsilon*I term ensures strict positive-definiteness across all batches,
    analogous to OAS shrinkage in pyriemann.  It is registered as a buffer so
    it moves to the correct device automatically with .to(device).

    Args:
        n_channels (int): number of EEG channels C.
        reg_eps    (float): regularisation coefficient (default 1e-4).
    """

    def __init__(self, n_channels: int, reg_eps: float = 1e-4):
        super().__init__()
        self.n_channels = n_channels
        self.reg_eps    = reg_eps
        # Register identity as a buffer — not a trainable parameter, but
        # moves with the module when .to(device) is called.
        self.register_buffer("eye", torch.eye(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, T) — raw EEG epochs, float32
        Returns:
            cov : (B, C, C) — regularised SPD covariance matrices
        """
        B, C, T = x.shape

        # Mean-centre each epoch across time (unbiased estimator pre-step)
        x_centred = x - x.mean(dim=-1, keepdim=True)   # (B, C, T)

        # Empirical covariance:  C = X X^T / (T-1)
        # torch.matmul handles the batch dimension automatically
        cov = torch.matmul(x_centred, x_centred.transpose(-1, -2)) / (T - 1)
        # cov: (B, C, C)

        # Regularise: add eps * I_C  (broadcast across batch)
        cov = cov + self.reg_eps * self.eye.unsqueeze(0)  # (1,C,C) broadcasts

        return cov  # (B, C, C)


# ══════════════════════════════════════════════════════════════════════════════
#  Module 2 · TangentSpaceLayer
# ══════════════════════════════════════════════════════════════════════════════
class TangentSpaceLayer(nn.Module):
    """
    Log-Euclidean tangent space projection for a batch of SPD matrices.

    Pipeline for each matrix C in the batch:
      1. Eigendecompose C = V diag(λ) V^T   via torch.linalg.eigh
         (eigh assumes symmetric input → numerically stable + exact)
      2. Clamp eigenvalues λ > 0 for numerical safety before log
      3. Compute matrix log:  log(C) = V diag(log λ) V^T
      4. Extract upper-triangle (including diagonal), scale off-diagonals by √2
         → flat feature vector of length C*(C+1)/2

    The √2 scaling preserves the Frobenius inner product in the vectorised
    form, matching the pyriemann convention.

    Args:
        n_channels (int): number of EEG channels C.
        eig_floor  (float): minimum eigenvalue before log (default 1e-6).
    """

    def __init__(self, n_channels: int, eig_floor: float = 1e-6):
        super().__init__()
        self.n_channels  = n_channels
        self.eig_floor   = eig_floor
        self.n_features  = n_channels * (n_channels + 1) // 2  # = 1953 for C=62

        # Pre-compute upper-triangle indices once; register as buffers
        idx = torch.triu_indices(n_channels, n_channels, offset=0)
        self.register_buffer("triu_row", idx[0])
        self.register_buffer("triu_col", idx[1])

        # Scaling mask: √2 for off-diagonal, 1 for diagonal
        scale = torch.where(idx[0] == idx[1],
                            torch.ones(idx.shape[1]),
                            torch.full((idx.shape[1],), 2.0 ** 0.5))
        self.register_buffer("triu_scale", scale)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cov : (B, C, C) — SPD covariance matrices
        Returns:
            ts  : (B, n_features) — tangent space feature vectors
        """
        # ── Matrix logarithm via eigendecomposition ──────────────────────────
        # eigh returns (eigenvalues, eigenvectors) for symmetric matrices
        # eigenvalues in ascending order, shape (B, C)
        # eigenvectors shape (B, C, C)  — columns are eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # Clamp to avoid log(0) or log(negative) from numerical noise
        eigvals = eigvals.clamp(min=self.eig_floor)  # (B, C)

        # log(C) = V @ diag(log λ) @ V^T
        log_eigvals = torch.log(eigvals)              # (B, C)

        # Build batched diagonal matrices from log eigenvalues
        # torch.diag_embed: (B, C) → (B, C, C)
        log_diag = torch.diag_embed(log_eigvals)      # (B, C, C)

        # Reconstruct: log(C) = V @ log_diag @ V^T
        log_cov = torch.matmul(eigvecs,
                  torch.matmul(log_diag, eigvecs.transpose(-1, -2)))
        # log_cov: (B, C, C)

        # ── Extract upper triangle and apply √2 scaling ──────────────────────
        # Index into each matrix in the batch along the upper triangle
        # log_cov[:, row_indices, col_indices] → (B, n_features)
        ts = log_cov[:, self.triu_row, self.triu_col]  # (B, n_features)
        ts = ts * self.triu_scale.unsqueeze(0)          # broadcast (1, n_features)

        return ts  # (B, n_features)


# ══════════════════════════════════════════════════════════════════════════════
#  Verification
# ══════════════════════════════════════════════════════════════════════════════
def run_verification():

    sep = "=" * 60

    print(f"\n{sep}")
    print("  Step 2.2 · PyTorch–Riemannian Bridge Verification")
    print(sep)
    print(f"  Device : {DEVICE}")

    # ── Step 1 · Mock tensor ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 1 · Mock Tensor Creation")
    print(sep)

    torch.manual_seed(42)
    X_pt = torch.randn(BATCH_SIZE, N_CHANNELS, N_TIMES,
                       dtype=torch.float32, device=DEVICE,
                       requires_grad=True)

    print(f"  X_pt.shape       : {tuple(X_pt.shape)}")
    print(f"  X_pt.device      : {X_pt.device}")
    print(f"  requires_grad    : {X_pt.requires_grad}")

    # ── Step 2 · CovarianceLayer ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 2 · CovarianceLayer Forward Pass")
    print(sep)

    cov_layer = CovarianceLayer(n_channels=N_CHANNELS, reg_eps=REG_EPS).to(DEVICE)
    X_cov = cov_layer(X_pt)

    print(f"  X_cov.shape      : {tuple(X_cov.shape)}")
    expected_cov = (BATCH_SIZE, N_CHANNELS, N_CHANNELS)
    assert tuple(X_cov.shape) == expected_cov, \
        f"Shape mismatch! Expected {expected_cov}, got {tuple(X_cov.shape)}"
    print(f"  ✓ Shape assertion passed: {tuple(X_cov.shape)} == {expected_cov}")

    # SPD spot-check on batch item 0
    C0       = X_cov[0].detach().cpu()
    is_sym   = torch.allclose(C0, C0.T, atol=1e-5)
    eigvals0 = torch.linalg.eigvalsh(C0)
    is_pd    = bool((eigvals0 > 0).all())
    print(f"\n  SPD check on X_cov[0]:")
    print(f"    Symmetric      : {is_sym}")
    print(f"    Positive def   : {is_pd}  (min eigval = {eigvals0.min():.6e})")

    # ── Step 3 · TangentSpaceLayer ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 3 · TangentSpaceLayer Forward Pass")
    print(sep)

    ts_layer = TangentSpaceLayer(n_channels=N_CHANNELS).to(DEVICE)
    X_ts     = ts_layer(X_cov)

    n_features_expected = N_CHANNELS * (N_CHANNELS + 1) // 2  # 1953
    expected_ts = (BATCH_SIZE, n_features_expected)
    print(f"  X_ts.shape       : {tuple(X_ts.shape)}")
    assert tuple(X_ts.shape) == expected_ts, \
        f"Shape mismatch! Expected {expected_ts}, got {tuple(X_ts.shape)}"
    print(f"  ✓ Shape assertion passed: {tuple(X_ts.shape)} == {expected_ts}")
    print(f"\n  X_ts stats (batch mean across features):")
    print(f"    mean  = {X_ts.detach().mean():.6f}")
    print(f"    std   = {X_ts.detach().std():.6f}")
    print(f"    min   = {X_ts.detach().min():.6f}")
    print(f"    max   = {X_ts.detach().max():.6f}")

    # ── Step 4 · Autograd backward test ─────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 4 · Autograd Backward Pass Test")
    print(sep)

    loss = X_ts.sum()
    print(f"  Dummy loss (sum of X_ts) : {loss.item():.4f}")
    print(f"  loss.requires_grad       : {loss.requires_grad}")

    loss.backward()

    grad_ok = X_pt.grad is not None
    print(f"  X_pt.grad is not None    : {grad_ok}")
    if grad_ok:
        print(f"  X_pt.grad.shape          : {tuple(X_pt.grad.shape)}")
        print(f"  X_pt.grad norm           : {X_pt.grad.norm().item():.6f}")
        print(f"  ✓ Gradient flows end-to-end through CovarianceLayer → TangentSpaceLayer")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Shape Transformation Summary")
    print(sep)
    print(f"\n  {'Stage':<35} {'Expected':<22} {'Actual':<22} {'Pass?'}")
    print(f"  {'-'*82}")
    rows = [
        ("Raw input tensor",        str((BATCH_SIZE, N_CHANNELS, N_TIMES)),
                                    str(tuple(X_pt.shape)),
                                    str((BATCH_SIZE, N_CHANNELS, N_TIMES)) == str(tuple(X_pt.shape))),
        ("After CovarianceLayer",   str(expected_cov),
                                    str(tuple(X_cov.shape)),
                                    tuple(X_cov.shape) == expected_cov),
        ("After TangentSpaceLayer", str(expected_ts),
                                    str(tuple(X_ts.shape)),
                                    tuple(X_ts.shape) == expected_ts),
        ("Gradient flows back",     "True",
                                    str(grad_ok),
                                    grad_ok),
    ]
    for name, exp, act, ok in rows:
        print(f"  {name:<35} {exp:<22} {act:<22} {'✓' if ok else '✗'}")

    print(f"\n{sep}")
    print("  Step 2.2 complete — differentiable Riemannian bridge verified.")
    print("  Ready for Phase 3: Mamba spatiotemporal architecture.")
    print(f"{sep}\n")


if __name__ == "__main__":
    run_verification()