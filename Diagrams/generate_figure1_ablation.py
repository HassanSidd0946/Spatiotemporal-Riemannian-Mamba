# =============================================================================
# command to run : modal run generate_figure1_ablation.py::main
# command to download : modal volume get eeg-data-vol figures/figure1_grand_ablation.png .
# generate_figure1_ablation.py
# Phase 3.1 — Figure 1: Grand Ablation Bar Chart
# IEEE Transactions / Nature-style publication figure, 4-condition comparison
#
# Loads exact Condition 3 / Condition 4 mean ± std accuracy from their JSON
# result files on the persistent Modal volume (eeg-data-vol), combines them
# with the hardcoded Condition 1 / Condition 2 baseline means, and renders a
# publication-grade vertical bar chart to both 600 DPI PNG and vector PDF.
#
# Usage: modal run generate_figure1_ablation.py::main
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-figure1-ablation")
volume = modal.Volume.from_name("eeg-data-vol")

VOLUME_PATH       = "/data"
CONDITION3_JSON   = "/data/results_condition3_ea_zeroshot.json"
CONDITION4_JSON   = "/data/results_condition4_subspace_calibrated.json"
FIGURES_DIR       = "/data/figures"
OUTPUT_PNG        = f"{FIGURES_DIR}/figure1_grand_ablation.png"
OUTPUT_PDF        = f"{FIGURES_DIR}/figure1_grand_ablation.pdf"

# Hardcoded, established baseline means (no fold-level std available for these).
CONDITION1_ACC = 0.5214   # Baseline CNN / EEGNet, zero-shot LOSO
CONDITION2_ACC = 0.5553   # SpatioMamba, zero-shot LOSO, no EA

CHANCE_LEVEL = 0.50

figure_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "matplotlib==3.8.4",
    )
)


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================

@app.function(
    image=figure_image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def generate_figure1():

    import json
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # =========================================================================
    # SECTION 3: LOAD EXACT METRICS FROM VOLUME
    # =========================================================================
    for path in (CONDITION3_JSON, CONDITION4_JSON):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required results file not found on volume: {path}. "
                f"Run the corresponding Condition script first and commit its output."
            )

    with open(CONDITION3_JSON, "r") as f:
        c3 = json.load(f)
    with open(CONDITION4_JSON, "r") as f:
        c4 = json.load(f)

    c3_mean = float(c3["mean_accuracy"])
    c3_std  = float(c3["std_accuracy"])
    c4_mean = float(c4["mean_accuracy"])
    c4_std  = float(c4["std_accuracy"])

    means = [CONDITION1_ACC, CONDITION2_ACC, c3_mean, c4_mean]
    stds  = [0.0,             0.0,             c3_std,  c4_std]

    labels = [
        "Condition 1\nBaseline CNN\n(EEGNet)",
        "Condition 2\nSpatioMamba\n(Zero-Shot)",
        "Condition 3\nSpatioMamba + EA\n(Zero-Shot)",
        "Condition 4\nSpatioMamba + EA\n+ Subspace Calibration",
    ]

    print("\n" + "="*70)
    print("  FIGURE 1 — VALUES BEING PLOTTED")
    print("="*70)
    print(f"  Condition 1 (hardcoded)      : {CONDITION1_ACC*100:.2f}%  (no std)")
    print(f"  Condition 2 (hardcoded)      : {CONDITION2_ACC*100:.2f}%  (no std)")
    print(f"  Condition 3 (from JSON)      : {c3_mean*100:.2f}% ± {c3_std*100:.2f}%  "
          f"[{CONDITION3_JSON}]")
    print(f"  Condition 4 (from JSON)      : {c4_mean*100:.2f}% ± {c4_std*100:.2f}%  "
          f"[{CONDITION4_JSON}]")
    print("="*70 + "\n")

    # =========================================================================
    # SECTION 4: PUBLICATION-GRADE STYLING
    # =========================================================================
    plt.rcParams.update({
        "font.family"       : "sans-serif",
        "font.sans-serif"   : ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size"         : 12,
        "axes.linewidth"    : 1.0,
        "axes.edgecolor"    : "#333333",
        "xtick.color"       : "#222222",
        "ytick.color"       : "#222222",
        "svg.fonttype"      : "none",
        "pdf.fonttype"      : 42,   # embed TrueType, not raster, in PDF/vector export
        "ps.fonttype"       : 42,
    })

    # Soft slate/grey for the two weaker zero-shot baselines, deep navy for
    # Condition 3 (the leak-free zero-shot ceiling), vibrant gold as the
    # accent color for our winning Condition 4.
    colors = [
        "#A8ADB4",   # Condition 1 — soft slate grey
        "#7C8695",   # Condition 2 — slightly deeper slate grey
        "#1F3A5F",   # Condition 3 — deep navy blue
        "#D4A017",   # Condition 4 — vibrant gold (accent / winner)
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_pos = np.arange(len(labels))
    bar_width = 0.6

    bars = ax.bar(
        x_pos, [m * 100 for m in means],
        width=bar_width,
        color=colors,
        edgecolor="#1a1a1a",
        linewidth=0.8,
        zorder=3,
    )

    # ---- Error bars (Condition 3 & 4 only) ----
    err_vals = [s * 100 for s in stds]
    ax.errorbar(
        x_pos, [m * 100 for m in means],
        yerr=err_vals,
        fmt="none",
        ecolor="#1a1a1a",
        elinewidth=1.4,
        capsize=6,
        capthick=1.4,
        zorder=4,
    )

    # ---- Bar-top percentage annotations (above bar + error bar) ----
    for i, (m, s) in enumerate(zip(means, stds)):
        top_y = m * 100 + s * 100
        label_txt = f"{m*100:.2f}%"
        ax.text(
            x_pos[i], top_y + 1.5, label_txt,
            ha="center", va="bottom",
            fontsize=12.5, fontweight="bold",
            color="#1a1a1a", zorder=5,
        )

    # ---- Chance-level reference line ----
    ax.axhline(
        y=CHANCE_LEVEL * 100,
        color="#B22222",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    ax.text(
        len(labels) - 0.42, CHANCE_LEVEL * 100 + 0.8,
        "Theoretical Chance Level (50%)",
        ha="right", va="bottom",
        fontsize=10.5, style="italic", color="#B22222",
    )

    # ---- Axes cosmetics ----
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("LOSO Test Accuracy (%)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Grand Ablation: Progressive Gains Across the 4-Condition BCI Pipeline\n"
        "(Strict 29-Subject Leave-One-Subject-Out Evaluation)",
        fontsize=13.5, fontweight="bold", pad=16,
    )

    y_max = max(m * 100 + s * 100 for m, s in zip(means, stds))
    ax.set_ylim(0, y_max + 12)

    ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", labelsize=10.5, length=4)

    fig.tight_layout()

    # =========================================================================
    # SECTION 5: EXPORT — 600 DPI PNG + VECTOR PDF
    # =========================================================================
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    png_ok = os.path.exists(OUTPUT_PNG) and os.path.getsize(OUTPUT_PNG) > 0
    pdf_ok = os.path.exists(OUTPUT_PDF) and os.path.getsize(OUTPUT_PDF) > 0

    volume.commit()

    print("="*70)
    print("  FIGURE 1 EXPORT CONFIRMATION")
    print("="*70)
    print(f"  PNG (600 DPI) : {OUTPUT_PNG}  "
          f"{'✓ written (' + str(os.path.getsize(OUTPUT_PNG)) + ' bytes)' if png_ok else '✗ MISSING'}")
    print(f"  PDF (vector)  : {OUTPUT_PDF}  "
          f"{'✓ written (' + str(os.path.getsize(OUTPUT_PDF)) + ' bytes)' if pdf_ok else '✗ MISSING'}")
    print("  ✓ volume.commit() complete — figures are now durably available")
    print("    on eeg-data-vol under /data/figures/")
    print("="*70 + "\n")

    if not (png_ok and pdf_ok):
        raise RuntimeError("Figure export verification failed — see log above.")

    return {
        "condition1_acc": CONDITION1_ACC,
        "condition2_acc": CONDITION2_ACC,
        "condition3_mean": c3_mean,
        "condition3_std": c3_std,
        "condition4_mean": c4_mean,
        "condition4_std": c4_std,
        "png_path": OUTPUT_PNG,
        "pdf_path": OUTPUT_PDF,
        "png_bytes": os.path.getsize(OUTPUT_PNG),
        "pdf_bytes": os.path.getsize(OUTPUT_PDF),
    }


# =============================================================================
# SECTION 6: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  Phase 3.1 — Figure 1: Grand Ablation Bar Chart")
    print("  4-Condition Comparison, IEEE/Nature Publication Style")
    print("="*70 + "\n")

    results = generate_figure1.remote()

    print("\n" + "="*70)
    print("  DONE")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<18} : {val}")
    print("="*70)
    print(
        "\n  Figure 1 generated and committed to eeg-data-vol.\n"
        "  Retrieve with: modal volume get eeg-data-vol figures/figure1_grand_ablation.png\n"
        "             or: modal volume get eeg-data-vol figures/figure1_grand_ablation.pdf\n"
    )