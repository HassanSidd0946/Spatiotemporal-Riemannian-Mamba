# run_step4_2_on_modal.py

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
        local_path="step4_2_mamba_benchmark.py",
        remote_path="/root/step4_2_mamba_benchmark.py",
    )
)

app = modal.App(name="mamba-step4-2-benchmark", image=image)


@app.function(gpu="A100", timeout=600, memory=16384)
def run_benchmark():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "step4_2", "/root/step4_2_mamba_benchmark.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["step4_2"] = m
    spec.loader.exec_module(m)
    m.main()


@app.local_entrypoint()
def main():
    print("\n" + "=" * 60)
    print("  Dispatching Step 4.2 Mamba Benchmark to Modal A100")
    print("=" * 60 + "\n")
    run_benchmark.remote()
    print("\n" + "=" * 60)
    print("  Done — check output above.")
    print("=" * 60 + "\n")