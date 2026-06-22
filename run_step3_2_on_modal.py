# run_step3_2_on_modal.py

"""
Modal runner for Phase 3, Step 3.2 — End-to-End Spatiotemporal Assembly.
Reuses the same verified image from Step 3.1 (no rebuild needed).
"""

import modal

BASE_IMAGE = "nvidia/cuda:12.1.1-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y --no-install-recommends git build-essential clang",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
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
    .run_commands(
        "CC=gcc CXX=g++ pip install --no-build-isolation "
        "causal-conv1d==1.4.0 "
        "mamba-ssm==2.2.4 "
        "'transformers<4.40'",
    )
    .add_local_file(
        local_path="step3_2_spatiotemporal_assembly.py",
        remote_path="/root/step3_2_spatiotemporal_assembly.py",
    )
)

app = modal.App(name="mamba-step3-2-assembly", image=image)


@app.function(
    gpu="A100",
    timeout=600,
    memory=16384,
)
def run_assembly():
    import importlib.util
    import sys

    spec   = importlib.util.spec_from_file_location(
        "step3_2", "/root/step3_2_spatiotemporal_assembly.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["step3_2"] = module
    spec.loader.exec_module(module)
    module.run_verification()


@app.local_entrypoint()
def main():
    print("\n" + "=" * 60)
    print("  Dispatching Step 3.2 Assembly to Modal A100")
    print("=" * 60 + "\n")
    run_assembly.remote()
    print("\n" + "=" * 60)
    print("  Done — check output above.")
    print("=" * 60 + "\n")