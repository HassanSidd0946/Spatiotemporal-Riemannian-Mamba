# step3_1_mamba_cuda_choke_point.py

"""
BCI Research Pipeline — Phase 3, Step 3.1
Task   : Mamba CUDA/Triton Hardware Choke-Point Verification
         Isolate mamba-ssm custom kernel compilation BEFORE assembling
         the end-to-end pipeline.

Input  : Mock PyTorch tensor  (batch=8, seq_len=251, d_model=62)
Output : Verified forward + backward pass through Mamba CUDA kernels

Why this step exists
--------------------
mamba-ssm relies on custom Triton/CUDA kernels that are JIT-compiled on
first use.  In serverless GPU environments (Modal A100) the CUDA toolchain,
driver version, and PyTorch ABI must all align.  Catching a kernel
compilation failure here — before it silently corrupts a multi-hour training
run — is the entire point of the "Fail Fast" strategy.

Expected tensor flow
--------------------
  X_seq  (8, 251, 62)  →  [Mamba block]  →  Y_seq  (8, 251, 62)
  MSE loss  →  loss.backward()  →  X_seq.grad  (8, 251, 62)

Shape is identity: Mamba maps (B, L, D) → (B, L, D).
"""

import sys
import traceback

import torch
import torch.nn as nn


# ── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE = 8
SEQ_LEN    = 251      # n_times from Phase 1 epoch window
D_MODEL    = 62       # n_channels — feature dimension per time step

# Mamba internal hyper-parameters (kept small for POC)
D_STATE    = 16       # SSM latent state size (N in S4/Mamba papers)
D_CONV     = 4        # local conv kernel size inside Mamba block
EXPAND     = 2        # channel expansion factor (inner dim = D_MODEL * EXPAND)

SEP = "=" * 60


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: environment snapshot
# ══════════════════════════════════════════════════════════════════════════════
def print_env():
    print(f"\n{SEP}")
    print("  Environment Snapshot")
    print(SEP)
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"  PyTorch         : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory      : {mem_gb:.1f} GB")
    else:
        print("\n  [FATAL] No CUDA device detected.")
        print("  mamba-ssm requires a CUDA GPU.  Aborting.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 · Import mamba_ssm
# ══════════════════════════════════════════════════════════════════════════════
def import_mamba():
    """
    Import Mamba and surface any installation / ABI mismatch errors cleanly.
    Returns the Mamba class, or exits with a diagnostic on failure.
    """
    print(f"\n{SEP}")
    print("  Step 1 · Importing mamba_ssm")
    print(SEP)

    try:
        from mamba_ssm import Mamba
        import mamba_ssm
        print(f"  mamba_ssm version : {mamba_ssm.__version__}")
        print("  ✓ Import successful")
        return Mamba

    except ImportError as exc:
        print(f"\n  [FATAL] ImportError: {exc}")
        print("\n  Install with:")
        print("    pip install mamba-ssm causal-conv1d")
        print("  or inside Modal:")
        print("    image = modal.Image.debian_slim()")
        print("           .pip_install('mamba-ssm', 'causal-conv1d')")
        sys.exit(1)

    except Exception as exc:
        print(f"\n  [FATAL] Unexpected error during import:\n  {exc}")
        traceback.print_exc()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 · Mock tensor
# ══════════════════════════════════════════════════════════════════════════════
def create_mock_tensor(device: torch.device) -> torch.Tensor:
    print(f"\n{SEP}")
    print("  Step 2 · Mock Tensor Creation")
    print(SEP)

    torch.manual_seed(42)

    # Shape: (batch, sequence, features)  — Mamba's canonical (B, L, D) format
    X_seq = torch.randn(
        BATCH_SIZE, SEQ_LEN, D_MODEL,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    print(f"  X_seq.shape      : {tuple(X_seq.shape)}")
    print(f"    batch_size     = {X_seq.shape[0]}")
    print(f"    seq_len        = {X_seq.shape[1]}  (n_times from Phase 1)")
    print(f"    d_model        = {X_seq.shape[2]}  (n_channels as feature dim)")
    print(f"  dtype            : {X_seq.dtype}")
    print(f"  device           : {X_seq.device}")
    print(f"  requires_grad    : {X_seq.requires_grad}")

    return X_seq


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 · Mamba block initialisation
# ══════════════════════════════════════════════════════════════════════════════
def init_mamba_block(Mamba, device: torch.device) -> nn.Module:
    print(f"\n{SEP}")
    print("  Step 3 · Mamba Block Initialisation")
    print(SEP)
    print(f"  d_model  = {D_MODEL}")
    print(f"  d_state  = {D_STATE}   (SSM latent state size)")
    print(f"  d_conv   = {D_CONV}    (local conv kernel)")
    print(f"  expand   = {EXPAND}    (inner dim = {D_MODEL * EXPAND})")

    try:
        model = Mamba(
            d_model=D_MODEL,
            d_state=D_STATE,
            d_conv=D_CONV,
            expand=EXPAND,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  ✓ Mamba block created on {device}")
        print(f"  Trainable params : {n_params:,}")
        return model

    except Exception as exc:
        print(f"\n  [FATAL] Mamba block initialisation failed:")
        print(f"  {exc}")
        print("\n  Triton/CUDA kernel compilation may have failed.")
        print("  Full traceback:")
        traceback.print_exc()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 · Forward pass
# ══════════════════════════════════════════════════════════════════════════════
def run_forward(model: nn.Module, X_seq: torch.Tensor) -> torch.Tensor:
    print(f"\n{SEP}")
    print("  Step 4 · Forward Pass")
    print(SEP)
    print(f"  Input  : {tuple(X_seq.shape)}")

    try:
        Y_seq = model(X_seq)

        print(f"  Output : {tuple(Y_seq.shape)}")

        # Shape must be identity: Mamba (B, L, D) → (B, L, D)
        expected = (BATCH_SIZE, SEQ_LEN, D_MODEL)
        assert tuple(Y_seq.shape) == expected, (
            f"Shape mismatch!  Expected {expected}, got {tuple(Y_seq.shape)}"
        )
        print(f"  ✓ Shape assertion passed: {tuple(Y_seq.shape)} == {expected}")
        print(f"\n  Y_seq stats (sanity check):")
        print(f"    mean  = {Y_seq.detach().mean():.6f}")
        print(f"    std   = {Y_seq.detach().std():.6f}")
        print(f"    min   = {Y_seq.detach().min():.6f}")
        print(f"    max   = {Y_seq.detach().max():.6f}")

        return Y_seq

    except AssertionError as exc:
        print(f"\n  [FATAL] {exc}")
        sys.exit(1)

    except Exception as exc:
        print(f"\n  [FATAL] Forward pass raised an exception:")
        print(f"  {exc}")
        print("\n  This typically indicates a Triton kernel JIT failure.")
        print("  Possible causes:")
        print("    • CUDA / PyTorch version mismatch (check nvidia-smi vs torch.version.cuda)")
        print("    • causal-conv1d not installed  (pip install causal-conv1d)")
        print("    • Triton unavailable on this driver (try --no-use-triton flag if available)")
        print("\n  Full traceback:")
        traceback.print_exc()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 · Backward pass (gradient verification)
# ══════════════════════════════════════════════════════════════════════════════
def run_backward(Y_seq: torch.Tensor, X_seq: torch.Tensor):
    print(f"\n{SEP}")
    print("  Step 5 · Backward Pass & Gradient Verification")
    print(SEP)

    try:
        # Dummy regression target — same shape as output, detached from graph
        target = torch.zeros_like(Y_seq).detach()

        loss_fn = nn.MSELoss()
        loss    = loss_fn(Y_seq, target)

        print(f"  Dummy target shape      : {tuple(target.shape)}  (zeros)")
        print(f"  MSE loss                : {loss.item():.6f}")
        print(f"  loss.requires_grad      : {loss.requires_grad}")

        loss.backward()

        grad_ok = X_seq.grad is not None
        print(f"\n  X_seq.grad is not None  : {grad_ok}")

        if not grad_ok:
            print("  [FATAL] Gradient did not reach X_seq.")
            print("  The backward graph through the Mamba CUDA kernel is broken.")
            sys.exit(1)

        grad_norm = X_seq.grad.norm().item()
        print(f"  X_seq.grad.shape        : {tuple(X_seq.grad.shape)}")
        print(f"  X_seq.grad norm         : {grad_norm:.6f}")

        if grad_norm == 0.0:
            print("  [WARNING] Gradient norm is exactly 0 — possible vanishing gradient.")
        else:
            print(f"  ✓ Gradients flow end-to-end through the Mamba CUDA kernel")

    except Exception as exc:
        print(f"\n  [FATAL] Backward pass raised an exception:")
        print(f"  {exc}")
        print("\n  Full traceback:")
        traceback.print_exc()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6 · Summary table
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(X_seq: torch.Tensor, Y_seq: torch.Tensor):
    print(f"\n{SEP}")
    print("  Step 6 · Verification Summary")
    print(SEP)

    expected_shape = (BATCH_SIZE, SEQ_LEN, D_MODEL)

    checks = [
        ("Input tensor shape",      tuple(X_seq.shape) == expected_shape),
        ("Output shape == input",   tuple(Y_seq.shape) == expected_shape),
        ("Gradient reaches input",  X_seq.grad is not None),
        ("Gradient norm > 0",       X_seq.grad is not None and X_seq.grad.norm().item() > 0),
    ]

    print(f"\n  {'Check':<40} {'Result'}")
    print(f"  {'-'*50}")
    all_pass = True
    for label, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {label:<40} {mark}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"  {'='*50}")
        print("  ALL CHECKS PASSED")
        print("  mamba-ssm CUDA kernels compile and run correctly on this GPU.")
        print("  Phase 3 Step 3.1 complete — cleared for pipeline integration.")
        print(f"  {'='*50}")
    else:
        print("  [FAIL] One or more checks failed — see output above.")

    print(f"\n{SEP}")
    print("  Next step: Step 3.2 — Integrate CovarianceLayer + TangentSpaceLayer")
    print("             → linear projection (1953 → 62) → Mamba block")
    print("             → end-to-end differentiable spatiotemporal pipeline.")
    print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{SEP}")
    print("  Phase 3 · Step 3.1 · Mamba CUDA Choke-Point Test")
    print(f"{SEP}")

    print_env()

    device = torch.device("cuda")

    Mamba   = import_mamba()
    X_seq   = create_mock_tensor(device)
    model   = init_mamba_block(Mamba, device)
    Y_seq   = run_forward(model, X_seq)
    run_backward(Y_seq, X_seq)
    print_summary(X_seq, Y_seq)


if __name__ == "__main__":
    main()