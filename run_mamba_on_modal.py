# # run_mamba_on_modal.py

# """
# Modal runner for Phase 3, Step 3.1 — Mamba CUDA/Triton Hardware Choke-Point.

# Fixes applied vs. previous iteration
# --------------------------------------
# Fix 1 · clang++ not found
#   torch.utils.cpp_extension._check_abi() probes for clang++ as a fallback
#   C++ compiler.  build-essential gives g++ but NOT clang.  Added 'clang'
#   to the Layer 1 apt install, and set CC/CXX env-vars so torch's build
#   system uses gcc/g++ explicitly without the clang probe.

# Fix 2 · NumPy 2.x ABI break
#   Unpinned 'numpy' resolved to 2.4.6; causal-conv1d was built against the
#   NumPy 1.x C-API (_ARRAY_API not found).  Pinned to 'numpy<2'.

# Fix 3 · mamba-ssm version pulling torch 2.12
#   mamba-ssm>=2.3 requires triton>=3.5, which requires torch>=2.4 — blowing
#   up the torch==2.2.0 pin.  Pinned to mamba-ssm==2.2.4 + causal-conv1d==1.4.0,
#   the last release-pair whose dependency graph is torch-2.2-clean.

# Image layer sequence
# --------------------
#   Layer 1  · OS packages     — git, build-essential, clang
#   Layer 2  · Python runtime  — torch==2.2.0, numpy<2, einops, packaging,
#                                wheel, setuptools (build-time deps)
#   Layer 3  · CUDA extensions — causal-conv1d==1.4.0, mamba-ssm==2.2.4
#                                (no-build-isolation, CC/CXX forced to gcc/g++)
#   Layer 4  · Script mount    — step3_1_mamba_cuda_choke_point.py → /root/
# """

# import modal

# # ── Base image ────────────────────────────────────────────────────────────────
# # cuda:12.1.1-devel supplies nvcc + full CUDA toolkit headers (required for
# # causal-conv1d and mamba-ssm CUDA kernel compilation).
# BASE_IMAGE = "nvidia/cuda:12.1.1-devel-ubuntu22.04"

# # ── Image definition ──────────────────────────────────────────────────────────
# image = (
#     modal.Image.from_registry(BASE_IMAGE, add_python="3.11")

#     # ── Layer 1 · OS-level build toolchain ───────────────────────────────────
#     # FIX 1a: 'clang' added alongside build-essential.
#     # torch.utils.cpp_extension._check_abi() runs `which clang++`; without
#     # the clang package that subprocess exits non-zero and aborts the build.
#     .run_commands(
#         "apt-get update -qq",
#         "apt-get install -y --no-install-recommends "
#         "git build-essential clang",          # ← clang fixes 'which clang++' probe
#         "apt-get clean && rm -rf /var/lib/apt/lists/*",
#     )

#     # ── Layer 2 · Python runtime + build-time deps ────────────────────────────
#     # FIX 2: numpy pinned to <2 to stay on the 1.x C-ABI that
#     #         causal-conv1d was compiled against.
#     # wheel + setuptools must be here (not isolated) for Layer 3's
#     # --no-build-isolation install to find them.
#     .run_commands(
#         "pip install --upgrade pip",
#         "pip install "
#         "torch==2.2.0 "
#         "'numpy<2' "                # ← pins NumPy to 1.x ABI
#         "einops "
#         "packaging "
#         "wheel "
#         "setuptools",
#     )

#     # ── Layer 3 · CUDA extension compilation ─────────────────────────────────
#     # FIX 1b: CC=gcc CXX=g++ forces torch's cpp_extension to use g++ and
#     #          skips the clang++ ABI probe entirely.
#     # FIX 3:  causal-conv1d==1.4.0 + mamba-ssm==2.2.4 are the last pair
#     #          whose transitive deps (triton==2.2.0) are compatible with
#     #          torch==2.2.0.  mamba-ssm>=2.3 pulls triton>=3.5 → torch>=2.4
#     #          which conflicts with our pin.
#     .run_commands(
#         "CC=gcc CXX=g++ pip install --no-build-isolation "
#         "causal-conv1d==1.4.0 "     # ← pinned; last version for torch 2.2
#         "mamba-ssm==2.2.4",         # ← pinned; last version before triton>=3.5 dep
#     )

#     # ── Layer 4 · Mount choke-point test script ───────────────────────────────
#     .add_local_file(
#         local_path="step3_1_mamba_cuda_choke_point.py",
#         remote_path="/root/step3_1_mamba_cuda_choke_point.py",
#     )
# )

# # ── Modal App ─────────────────────────────────────────────────────────────────
# app = modal.App(name="mamba-cuda-choke-point", image=image)


# # ── Remote function ───────────────────────────────────────────────────────────
# @app.function(
#     gpu="A100",       # A100 required: mamba-ssm CUDA kernels need sm_80+
#     timeout=600,      # 10 min — first run includes Triton JIT compilation cache warm-up
#     memory=16384,     # 16 GB RAM — kernel compilation artefacts are large
# )
# def run_choke_point():
#     """
#     Execute step3_1_mamba_cuda_choke_point.py inside the Modal A100 container.

#     Uses importlib rather than subprocess so stdout streams back to the local
#     terminal in real time via Modal's log forwarding.
#     """
#     import importlib.util
#     import sys

#     script_path = "/root/step3_1_mamba_cuda_choke_point.py"

#     spec   = importlib.util.spec_from_file_location("choke_point", script_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules["choke_point"] = module

#     spec.loader.exec_module(module)   # executes the module body
#     module.main()                     # calls the guarded entry point


# # ── Local entry point ─────────────────────────────────────────────────────────
# @app.local_entrypoint()
# def main():
#     """
#     Invoked by:  modal run run_mamba_on_modal.py
#     """
#     print("\n" + "=" * 60)
#     print("  Dispatching Mamba CUDA choke-point test to Modal A100")
#     print("=" * 60 + "\n")

#     run_choke_point.remote()

#     print("\n" + "=" * 60)
#     print("  Remote execution complete — check output above.")
#     print("=" * 60 + "\n")















# run_mamba_on_modal.py

"""
Modal runner for Phase 3, Step 3.1 — Mamba CUDA/Triton Hardware Choke-Point.

Fix history
-----------
Iteration 1  ModuleNotFoundError: No module named 'wheel'
             → Pre-install wheel + setuptools before --no-build-isolation

Iteration 2  Three simultaneous failures:
             (a) which clang++ → non-zero exit  → apt install clang + CC/CXX env
             (b) NumPy 2.x ABI break            → numpy<2
             (c) mamba-ssm pulling torch 2.12   → pin mamba-ssm==2.2.4

Iteration 3  ImportError: cannot import name 'GreedySearchDecoderOnlyOutput'
             mamba_ssm/utils/generation.py imports symbols removed in
             transformers==4.40.0.  transformers resolved to 5.12.1.
             → pin transformers<4.40 in the same install layer
"""

import modal

BASE_IMAGE = "nvidia/cuda:12.1.1-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")

    # Layer 1 · OS toolchain
    # clang required: torch.utils.cpp_extension._check_abi() probes `which clang++`
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y --no-install-recommends git build-essential clang",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )

    # Layer 2 · Python runtime + build-time deps
    # numpy<2 : causal-conv1d was built against the NumPy 1.x C-ABI
    # wheel + setuptools : required by --no-build-isolation in Layer 3
    .run_commands(
        "pip install --upgrade pip",
        "pip install "
        "torch==2.2.0 "
        "'numpy<2' "
        "einops "
        "packaging "
        "wheel "
        "setuptools",
    )

    # Layer 3 · CUDA extension compilation
    # CC/CXX : forces torch cpp_extension to use gcc/g++, skips clang++ ABI probe
    # causal-conv1d==1.4.0 + mamba-ssm==2.2.4 : last pair compatible with torch 2.2
    # transformers<4.40 : GreedySearchDecoderOnlyOutput / SampleDecoderOnlyOutput
    #   were removed in transformers 4.40.0; mamba_ssm/utils/generation.py
    #   still imports them, so we must stay below that boundary.
    .run_commands(
        "CC=gcc CXX=g++ pip install --no-build-isolation "
        "causal-conv1d==1.4.0 "
        "mamba-ssm==2.2.4 "
        "'transformers<4.40'",      # ← fixes GreedySearchDecoderOnlyOutput import
    )

    # Layer 4 · Mount choke-point test script
    .add_local_file(
        local_path="step3_1_mamba_cuda_choke_point.py",
        remote_path="/root/step3_1_mamba_cuda_choke_point.py",
    )
)

app = modal.App(name="mamba-cuda-choke-point", image=image)


@app.function(
    gpu="A100",
    timeout=600,
    memory=16384,
)
def run_choke_point():
    import importlib.util
    import sys

    script_path = "/root/step3_1_mamba_cuda_choke_point.py"
    spec   = importlib.util.spec_from_file_location("choke_point", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["choke_point"] = module
    spec.loader.exec_module(module)
    module.main()


@app.local_entrypoint()
def main():
    print("\n" + "=" * 60)
    print("  Dispatching Mamba CUDA choke-point test to Modal A100")
    print("=" * 60 + "\n")
    run_choke_point.remote()
    print("\n" + "=" * 60)
    print("  Remote execution complete — check output above.")
    print("=" * 60 + "\n")