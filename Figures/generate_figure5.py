"""
generate_figure5.py
====================
Generates "Figure 5: Subject-wise Performance Distribution" for the
SpatiotemporalRiemannianMamba IEEE Q1 submission — a violin plot overlaid
with a swarm plot showing all 29 LOSO fold accuracies per model.

Run locally:
    pip install matplotlib seaborn pandas numpy
    python generate_figure5.py

Output:
    figure5_distribution.png  (300 dpi, tight bounding box)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Real 29-fold LOSO cross-validation accuracies (fractions, 0-1)
# ─────────────────────────────────────────────────────────────────────────────
mamba_per_fold = [
    0.634146, 0.56, 0.4925, 0.5675, 0.57, 0.525, 0.515, 0.4975, 0.515, 0.575,
    0.5275, 0.6025, 0.5075, 0.5725, 0.55, 0.6475, 0.495, 0.685, 0.605, 0.5775,
    0.6375, 0.51, 0.5325, 0.5, 0.5775, 0.5975, 0.5, 0.505, 0.5225,
]

cnn_per_fold = [
    0.521951, 0.54, 0.4525, 0.555, 0.5525, 0.62, 0.54, 0.485, 0.495, 0.59,
    0.545, 0.6225, 0.5, 0.5325, 0.5275, 0.5125, 0.4925, 0.465, 0.5, 0.57,
    0.465, 0.5175, 0.495, 0.4725, 0.5, 0.51, 0.53, 0.505, 0.5075,
]

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

COLOR_SPATIOMAMBA = "#2E6E72"   # teal / dark cyan, matches Fig. 1 & 2 theme
COLOR_BASELINE    = "#8C8C8C"   # neutral gray
DOT_COLOR         = "#1A1A1A"
DOT_ALPHA         = 0.6

CHANCE_LEVEL = 50.0


def main():
    # ── Build a tidy DataFrame: one row per (model, fold accuracy %) ───────
    n_folds = len(mamba_per_fold)
    assert n_folds == len(cnn_per_fold), "Fold count mismatch between models."

    df = pd.DataFrame({
        "Accuracy (%)": [v * 100 for v in mamba_per_fold] + [v * 100 for v in cnn_per_fold],
        "Model": ["SpatioMamba"] * n_folds + ["BaselineCNN"] * n_folds,
    })

    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    palette = {"SpatioMamba": COLOR_SPATIOMAMBA, "BaselineCNN": COLOR_BASELINE}
    order = ["SpatioMamba", "BaselineCNN"]

    # ── Violin plot (distribution shape) ────────────────────────────────────
    sns.violinplot(
        data=df, x="Model", y="Accuracy (%)", order=order,
        hue="Model", hue_order=order, palette=palette, legend=False,
        inner=None, cut=0, linewidth=1.1,
        edgecolor="#2B2B2B", alpha=0.55, ax=ax,
    )

    # ── Swarm plot (every individual subject/fold as a dot) ────────────────
    sns.swarmplot(
        data=df, x="Model", y="Accuracy (%)", order=order,
        color=DOT_COLOR, alpha=DOT_ALPHA, size=5, edgecolor="none", ax=ax,
        zorder=3,
    )

    # ── Median markers (explicit, since inner=None on the violin) ──────────
    for i, model in enumerate(order):
        median_val = df.loc[df["Model"] == model, "Accuracy (%)"].median()
        ax.plot([i - 0.18, i + 0.18], [median_val, median_val],
                color="#B00000", linewidth=2.0, zorder=4, solid_capstyle="round")

    # ── Chance-level reference line ─────────────────────────────────────────
    ax.axhline(CHANCE_LEVEL, color="#555555", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(
        1.48, CHANCE_LEVEL + 0.5, "Chance level", ha="right", va="bottom",
        fontsize=9.5, color="#555555", style="italic",
    )

    # ── Labels / limits ──────────────────────────────────────────────────────
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("LOSO fold accuracy (%)", fontsize=12)
    ax.set_xlim(-0.6, 1.6)
    y_min = min(min(mamba_per_fold), min(cnn_per_fold)) * 100 - 4
    y_max = max(max(mamba_per_fold), max(cnn_per_fold)) * 100 + 4
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize=10.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ax.set_title(
        "Fig. 5.  Subject-wise LOSO accuracy distribution (N = 29 subjects per model)",
        fontsize=11, pad=14,
    )

    # Small legend explaining the red median line (swarm/violin colors are
    # already labeled by the x-axis category, so only the median needs a key)
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color="#B00000", linewidth=2.0, label="Median")]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9.5, frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig("figure5_distribution.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure5_distribution.png (300 dpi)")

    # ── Printed summary for sanity-checking against the paired t-test ──────
    print(f"\nSpatioMamba : median={df.loc[df.Model=='SpatioMamba','Accuracy (%)'].median():.2f}%  "
          f"mean={df.loc[df.Model=='SpatioMamba','Accuracy (%)'].mean():.2f}%")
    print(f"BaselineCNN : median={df.loc[df.Model=='BaselineCNN','Accuracy (%)'].median():.2f}%  "
          f"mean={df.loc[df.Model=='BaselineCNN','Accuracy (%)'].mean():.2f}%")


if __name__ == "__main__":
    main()