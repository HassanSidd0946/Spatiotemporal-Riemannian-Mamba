# Spatiotemporal-Riemannian-Mamba for Cognitive BCI and False Memory Decoding

> A dual-branch deep learning architecture fusing **Log-Euclidean Riemannian Geometry** (spatial) with **Selective State-Space Models / Mamba** (temporal) for subject-independent decoding of cognitive EEG states.

**Paper status:** 🟡 Under Review — IEEE (Q1 Journal)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Compute-Modal%20A100-7C3AED)](https://modal.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)](#)

---

## Abstract

Decoding complex cognitive states — such as the distinction between true and false memory recall — from EEG is fundamentally limited by high noise, non-stationarity, and severe inter-subject variability. This repository contains the full research pipeline for **SpatiotemporalRiemannianMamba**, a novel architecture that addresses this challenge with two specialized branches operating in parallel:

- **Spatial branch:** estimates a regularized per-epoch channel covariance matrix and projects it onto the **Log-Euclidean tangent space**, producing a Riemannian-geometric descriptor that is robust to inter-subject spatial variation.
- **Temporal branch:** processes the raw EEG sequence through a **Mamba** Selective State-Space Model, capturing long-range temporal dependencies at linear computational cost.

The two branches are fused and evaluated on the public **`ds005189`** False Memory EEG dataset under a strict **29-fold Leave-One-Subject-Out (LOSO)** cross-validation protocol — the most rigorous available test of subject-independent generalization. The model achieves a mean LOSO accuracy of **55.53% (± 5.25%)**, significantly outperforming an EEGNet-inspired `BaselineCNN` (**52.14% ± 4.21%**) by an absolute margin of 3.39 percentage points (paired t-test, **p = 0.0093**, p < 0.01).

> This repository contains the complete, reproducible pipeline: data acquisition, Riemannian feature engineering, model architecture, cloud-based LOSO training on Modal, and statistical analysis.

---

## Repository Structure

```
.
├── data/                                  # Downloaded & preprocessed ds005189 EEG data
├── Figures/                               # Generated figures for the paper (see Visual Results)
├── venv/                                  # Local Python virtual environment (not committed)
│
├── step1_1_fetch_inspect.py               # Phase 1 — Download & inspect raw ds005189 EEG data
├── step1_2_filter_epoch.py                # Phase 1 — Filtering & epoching of raw EEG
│
├── step2_1_covariance_tangentspace.py     # Phase 2 — Covariance estimation + Log-Euclidean mapping
├── step2_2_pytorch_riemannian_bridge.py   # Phase 2 — Differentiable PyTorch Riemannian layers
│
├── step3_1_mamba_cuda_choke_point.py      # Phase 3 — Mamba/CUDA kernel sanity & compatibility check
├── step3_2_spatiotemporal_assembly.py     # Phase 3 — Full model assembly + forward/backward verification
│
├── step4_1_baseline_cnn_training.py       # Phase 4 — BaselineCNN (EEGNet-inspired) training routine
├── step4_2_mamba_benchmark.py             # Phase 4 — SpatiotemporalRiemannianMamba benchmark routine
│
├── run_data_engine_on_modal.py            # Modal entrypoint — cloud-side data engine
├── run_mamba_on_modal.py                  # Modal entrypoint — standalone Mamba run
├── run_step3_2_on_modal.py                # Modal entrypoint — cloud verification of step3_2
├── run_step4_1_on_modal.py                # Modal entrypoint — cloud run of step4_1 (BaselineCNN)
├── run_step4_2_on_modal.py                # Modal entrypoint — cloud run of step4_2 (SpatioMamba)
├── run_loso_trainer_on_modal.py           # Modal entrypoint — full 29-fold LOSO benchmark (A100)
│
├── statistical_analysis.py                # Local — paired t-test, Cohen's d, 95% CI on LOSO results
├── requirements.txt
└── readme.md
```

**Naming convention:** files prefixed `stepN_M_*.py` are organized by research phase and are designed to be run **locally** for development, debugging, and verification on small/mock data. Files prefixed `run_*_on_modal.py` are **Modal cloud entrypoints** that wrap the corresponding logic (often a heavier or full-scale version of it) for execution on a remote GPU.

| Phase | Concern |
|---|---|
| `step1_*` | Data acquisition & preprocessing (local) |
| `step2_*` | Riemannian geometry feature engineering (local) |
| `step3_*` | Architecture assembly & Mamba integration sanity checks (local + cloud) |
| `step4_*` | Full model training & benchmarking (local + cloud) |

---

## Step 1: Local Setup (Environment & Requirements)

> All data preprocessing, statistical analysis, and lightweight verification steps are intended to run **locally**. Only the heavy LOSO training loop requires cloud GPU compute (see Step 2).

**1. Create a virtual environment:**

```bash
python -m venv venv
```

**2. Activate the virtual environment:**

<table>
<tr><th>Windows</th><th>Linux / macOS</th></tr>
<tr>
<td>

```powershell
venv\Scripts\activate
```

</td>
<td>

```bash
source venv/bin/activate
```

</td>
</tr>
</table>

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` covers local-only dependencies (data I/O, Riemannian geometry utilities, plotting, statistics). Heavier GPU-bound packages (`torch`, `mamba-ssm`, `causal-conv1d`) are installed **inside the Modal cloud image** at build time — see `run_loso_trainer_on_modal.py` — and do not need to be installed locally unless you intend to run GPU steps on your own machine.

---

## Step 2: Modal Cloud Setup (Crucial)

All GPU-bound model training (Mamba SSM blocks, full LOSO cross-validation) runs on [Modal](https://modal.com/), a serverless cloud GPU platform. This avoids requiring a local CUDA-capable GPU and provisions an **A100** on demand for each training run.

**1. Install the Modal client:**

```bash
pip install modal
```

**2. Authenticate the local CLI with your Modal account:**

```bash
modal setup
```

> If `modal setup` does not open a browser automatically, or you prefer to generate a token manually, use:
> ```bash
> modal token new
> ```

This links your local environment to your Modal account so that `modal run` commands can provision cloud GPUs and bill against your account.

> Once authenticated, no further manual GPU configuration is required — each `run_*_on_modal.py` script declares its own GPU type, image, and volume requirements (e.g., `gpu="A100"` in `run_loso_trainer_on_modal.py`), and Modal provisions matching hardware automatically per invocation.

---

## Step 3: Execution Pipeline (How to Run the Code)

The pipeline is designed to be run sequentially, phase by phase.

### 3.1 — Data Fetching & Preprocessing (Local)

Download and prepare the public `ds005189` dataset:

```bash
python step1_1_fetch_inspect.py
python step1_2_filter_epoch.py
```

These scripts download the raw EEG recordings into the `data/` directory, perform initial inspection, then apply filtering and epoching to produce model-ready tensors.

### 3.2 — Riemannian Geometry Processing (Local)

```bash
python step2_1_covariance_tangentspace.py
python step2_2_pytorch_riemannian_bridge.py
```

`step2_1` establishes the covariance-estimation and Log-Euclidean tangent-space mapping logic in isolation (NumPy/SciPy reference implementation). `step2_2` re-implements this pipeline as differentiable PyTorch `nn.Module` layers (`CovarianceLayer`, `TangentSpaceLayer`), bridging the Riemannian feature engineering into the trainable model graph used in later phases.

### 3.3 — Model Assembly & Verification (Local)

```bash
python step3_1_mamba_cuda_choke_point.py
python step3_2_spatiotemporal_assembly.py
```

`step3_1` sanity-checks the Mamba CUDA kernel build/compatibility in isolation before integrating it into the full model. `step3_2` assembles the complete `SpatiotemporalRiemannianMamba` model and verifies an end-to-end forward + backward pass on a mock tensor.

### 3.4 — Model Training on Modal (Cloud, A100)

The full 29-fold LOSO benchmark — `SpatiotemporalRiemannianMamba` vs. `BaselineCNN` — runs on a Modal-provisioned A100:

```bash
modal run run_loso_trainer_on_modal.py
```

> **What happens under the hood:** Modal builds the declared container image (CUDA 12.1, PyTorch 2.2.0, `mamba-ssm`, `causal-conv1d`, etc.), provisions an A100 GPU on demand, mounts the persistent `eeg-data-vol` volume containing the preprocessed dataset, and executes the decorated `@app.function(...)` remotely. Your local script and its dependencies are synced to the cloud function automatically — no manual upload step is required. Results are returned to your local terminal via the `@app.local_entrypoint()`.

Individual phases can also be run on Modal independently, for staged verification before committing to the full run:

```bash
modal run run_step3_2_on_modal.py     # cloud-side architecture verification
modal run run_step4_1_on_modal.py     # BaselineCNN training only
modal run run_step4_2_on_modal.py     # SpatioMamba training only
modal run run_mamba_on_modal.py       # standalone Mamba block benchmark
modal run run_data_engine_on_modal.py # cloud-side data engine / preprocessing at scale
```

### 3.5 — Statistical Analysis (Local)

Once LOSO results are available, reproduce the paired statistical comparison locally:

```bash
python statistical_analysis.py
```

This computes, from the 29-fold accuracy arrays:
- Mean and standard deviation per model
- Paired t-test (`scipy.stats.ttest_rel`) — t-statistic and exact p-value
- Cohen's d (paired effect size)
- 95% confidence interval on the mean difference

---

## Visual Results

> 🖼️ **Figure placeholders below — insert generated images from `Figures/` once available.**

<!--
Replace each placeholder with: ![Caption](Figures/filename.png)
-->

### Figure 1 — Architecture Block Diagram
```
[ INSERT Figures/figure1_architecture.png HERE ]
```
*Overall dual-branch architecture: spatial Riemannian-geometry branch (top) and temporal Mamba state-space branch (bottom), fused into a classification head.*

### Figure 2 — LOSO Accuracy Comparison
```
[ INSERT Figures/figure2_results.png HERE ]
```
*Mean LOSO accuracy: SpatioMamba (55.53%) vs. BaselineCNN (52.14%), with significance bracket (p < 0.01).*

### Figure 3 — Training & Validation Learning Curves
```
[ INSERT Figures/figure3_learning_curves.png HERE ]
```
*Representative-fold loss and accuracy curves, annotated with the early-stopping checkpoint.*

### Figure 4 — Aggregate Confusion Matrix
```
[ INSERT Figures/figure4_confusion_matrix.png HERE ]
```
*Aggregate True Memory vs. False Memory classification counts across all 29 LOSO folds.*

### Figure 5 — Subject-wise Performance Distribution
```
[ INSERT Figures/figure5_distribution.png HERE ]
```
*Violin + swarm plot of per-subject LOSO fold accuracies for both models.*

### Figure 6 — ROC Curve
```
[ INSERT Figures/figure6_roc_curve.png HERE ]
```
*Receiver Operating Characteristic curves and AUC for SpatioMamba vs. BaselineCNN.*

---

## Citation

If you use this codebase or build upon this work, please cite:

```bibtex
@article{author_year_spatiotemporal_riemannian_mamba,
  title     = {Spatiotemporal-Riemannian-Mamba for Cognitive BCI and False Memory Decoding},
  author    = {AUTHOR_NAME_HERE and COAUTHOR_NAME_HERE},
  journal   = {IEEE TRANSACTIONS / JOURNAL NAME HERE},
  year      = {2026},
  volume    = {XX},
  number    = {XX},
  pages     = {XX--XX},
  doi       = {10.1109/XXXX.2026.XXXXXXX},
  note      = {Under review}
}
```

> ⚠️ Update author names, journal/transactions title, volume/issue/page numbers, and DOI once the paper is accepted and assigned final publication metadata.

---

<p align="center">
<sub>Built with PyTorch, Mamba (state-space models), pyRiemann-style Riemannian geometry, and Modal serverless GPU compute.</sub>
</p>