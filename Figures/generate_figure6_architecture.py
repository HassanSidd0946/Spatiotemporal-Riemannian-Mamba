# commands to run :
# modal run generate_figure6_architecture.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure6_system_architecture.png .
# modal volume get eeg-data-vol /figures/figure6_system_architecture.pdf .
"""
generate_figure6_architecture.py

Generates "Figure 6: Proposed Spatiotemporal Riemannian Architecture with
Subspace Shrinkage Calibration" -- a publication-quality, horizontal system
architecture flowchart for the BCI EEG false-memory classification manuscript.

Built entirely with matplotlib.patches (FancyBboxPatch + annotate-based arrows)
-- no graphviz / networkx dependency, so it renders identically anywhere.

This version runs the drawing code INSIDE a Modal function so the outputs are
written directly onto the `eeg-data-vol` persistent volume at:
    /data/figures/figure6_system_architecture.png  (600 DPI)
    /data/figures/figure6_system_architecture.pdf

Usage (from your local machine, in the venv where `modal` is installed):
    modal run generate_figure6_architecture.py
"""

import modal

# ----------------------------------------------------------------------------
# Modal app / image / volume setup
# ----------------------------------------------------------------------------
app = modal.App("eeg-figure6-architecture")

image = modal.Image.debian_slim().pip_install("matplotlib")

volume = modal.Volume.from_name("eeg-data-vol", create_if_missing=True)

OUTPUT_DIR = "/data/figures"


@app.function(image=image, volumes={"/data": volume})
def generate_figure6():
    import os
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # ------------------------------------------------------------------------
    # Global style configuration
    # ------------------------------------------------------------------------
    plt.rcParams["text.usetex"] = False          # use mathtext, not a LaTeX install
    plt.rcParams["mathtext.fontset"] = "cm"       # Computer Modern-style math glyphs
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    PNG_PATH = os.path.join(OUTPUT_DIR, "figure6_system_architecture.png")
    PDF_PATH = os.path.join(OUTPUT_DIR, "figure6_system_architecture.pdf")

    # ------------------------------------------------------------------------
    # Pipeline stage definitions (title, subtext, face color, text color)
    # ------------------------------------------------------------------------
    STANDARD_FACE = "#F8FAFC"
    STANDARD_EDGE = "#64748B"
    TEAL = "#0D9488"
    DARK_TEXT = "#1E293B"
    ARROW_COLOR = "#0F172A"
    SHADOW_COLOR = "#94A3B8"
    WHITE = "#FFFFFF"

    stages = [
        {
            "title": "Raw EEG Acquisition",
            "subtitle": "62-Channel Cognitive Task Data\n" r"($X \in \mathbb{R}^{C \times T}$)",
            "face": STANDARD_FACE,
            "edge": STANDARD_EDGE,
            "text": DARK_TEXT,
            "core": False,
        },
        {
            "title": "Euclidean Alignment (EA)",
            "subtitle": "Aligns covariance matrices to subject-\nspecific reference states to eliminate\ninter-subject variability",
            "face": STANDARD_FACE,
            "edge": STANDARD_EDGE,
            "text": DARK_TEXT,
            "core": False,
        },
        {
            "title": "Tangent Space Mapping",
            "subtitle": "Computes spatial covariance " + r"($\Sigma$)" + "\nand projects onto the Log-Euclidean\nTangent Space",
            "face": STANDARD_FACE,
            "edge": STANDARD_EDGE,
            "text": DARK_TEXT,
            "core": False,
        },
        {
            "title": "Condition 4: Subspace\nShrinkage Calibration",
            "subtitle": "Ledoit-Wolf regularization and feature\nscaling to rescue collapsed subject\nmanifolds",
            "face": TEAL,
            "edge": TEAL,
            "text": WHITE,
            "core": True,
        },
        {
            "title": "LDA Classification",
            "subtitle": "Out-of-fold linear discriminant\n" r"$\rightarrow$ Output: True vs. False Memory",
            "face": STANDARD_FACE,
            "edge": STANDARD_EDGE,
            "text": DARK_TEXT,
            "core": False,
        },
    ]

    # ------------------------------------------------------------------------
    # Layout geometry
    # ------------------------------------------------------------------------
    FIG_W, FIG_H = 16, 5
    N = len(stages)

    box_w = 2.30
    box_h = 2.05
    gap = 0.95
    total_w = N * box_w + (N - 1) * gap
    x_start = (FIG_W - total_w) / 2.0
    y_center = FIG_H / 2.0 - 0.15

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    for i, stage in enumerate(stages):
        x0 = x_start + i * (box_w + gap)
        y0 = y_center - box_h / 2.0
        cx = x0 + box_w / 2.0
        cy = y_center

        is_core = stage["core"]

        # Modern drop shadow -- every card gets one, offset down-right
        shadow = FancyBboxPatch(
            (x0 + 0.05, y0 - 0.05), box_w, box_h,
            boxstyle="round,pad=0.4,rounding_size=0.16",
            linewidth=0,
            facecolor=SHADOW_COLOR,
            alpha=0.20,
            zorder=1,
        )
        ax.add_patch(shadow)

        # Main floating card
        box = FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle="round,pad=0.4,rounding_size=0.16",
            linewidth=0 if is_core else 1.5,
            edgecolor="none" if is_core else stage["edge"],
            facecolor=stage["face"],
            mutation_aspect=1,
            zorder=2,
        )
        ax.add_patch(box)

        # Title text
        ax.text(
            cx, cy + 0.42, stage["title"],
            ha="center", va="center",
            fontsize=12.5, fontweight="bold",
            color=stage["text"],
            zorder=3,
            linespacing=1.25,
        )

        # Subtitle text
        ax.text(
            cx, cy - 0.42, stage["subtitle"],
            ha="center", va="center",
            fontsize=9.2,
            color=stage["text"],
            zorder=3,
            linespacing=1.45,
        )

        # Step number badge (top-left corner of each box)
        badge_r = 0.16
        badge_x = x0 + 0.05
        badge_y = y0 + box_h - 0.05
        ax.add_patch(plt.Circle(
            (badge_x, badge_y), badge_r,
            facecolor=WHITE if is_core else DARK_TEXT,
            edgecolor=stage["text"],
            linewidth=1.1,
            zorder=4,
        ))
        # (badge fill/edge logic unchanged from before -- retained for step numbering)
        ax.text(
            badge_x, badge_y, str(i + 1),
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color=TEAL if is_core else WHITE,
            zorder=5,
        )

    # ------------------------------------------------------------------------
    # Straight, thick connecting arrows between consecutive blocks
    # ------------------------------------------------------------------------
    for i in range(N - 1):
        x_tail = x_start + i * (box_w + gap) + box_w
        x_head = x_start + (i + 1) * (box_w + gap)
        y = y_center

        ax.annotate(
            "",
            xy=(x_head, y), xycoords="data",
            xytext=(x_tail, y), textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>", lw=3.5, color=ARROW_COLOR,
                shrinkA=0, shrinkB=0, mutation_scale=25,
                joinstyle="round", capstyle="round",
            ),
            zorder=5,
        )

    # ------------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------------
    fig.suptitle(
        "Figure 6: Proposed Spatiotemporal Riemannian Architecture with\n"
        "Subspace Shrinkage Calibration",
        fontsize=14.5, fontweight="bold", color=DARK_TEXT, y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # ------------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved PNG  -> {PNG_PATH} (600 DPI)")

    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved PDF  -> {PDF_PATH}")

    plt.close(fig)

    # ------------------------------------------------------------------------
    # Persist to the Modal volume
    # ------------------------------------------------------------------------
    volume.commit()
    print("[OK] volume.commit() executed -- figures persisted to Modal volume 'eeg-data-vol'.")
    print("[DONE] Figure 6 generation complete.")


@app.local_entrypoint()
def main():
    generate_figure6.remote()