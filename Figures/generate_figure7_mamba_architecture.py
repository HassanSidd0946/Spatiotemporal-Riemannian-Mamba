# commands to run :
# modal run generate_figure7_mamba_architecture.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure7_mamba_architecture.png .
# modal volume get eeg-data-vol /figures/figure7_mamba_architecture.pdf .
"""
generate_figure7_mamba_architecture.py

Generates "Figure 7: Architecture of the Dual-Branch Spatiotemporal
Riemannian Mamba Network" -- a publication-quality, two-track flowchart for
the BCI EEG false-memory classification manuscript.

Built entirely with matplotlib.patches (FancyBboxPatch + FancyArrowPatch)
-- no graphviz / networkx dependency, so it renders identically anywhere.

LEGIBILITY PATCH (this version):
    - All in-box equation / subtext font sizes bumped up substantially
      (roughly +3.5-4pt across the board) so nothing requires zooming.
    - Global mathtext font switched from "stix" to "dejavusans", which
      renders noticeably bolder/thicker strokes than STIX and matches the
      DejaVu Sans family already used for titles/labels -- this alone
      fixes most of the "thin math text bleeding into dark backgrounds"
      problem.
    - Key symbols (Sigma, X, f_s, f_t, z, s, I) wrapped in \\mathbf{} /
      \\boldsymbol{} so they render with heavier strokes on the
      teal/amber/purple fills.
    - Card heights (CARD_H, FUSION_H, INPUT_H) increased slightly to give
      the larger text comfortable padding -- box widths, x-positions,
      colors, arrows, and branch layout are all untouched.

This version runs the drawing code INSIDE a Modal function so the outputs
are written directly onto the `eeg-data-vol` persistent volume at:
    /data/figures/figure7_mamba_architecture.png  (600 DPI)
    /data/figures/figure7_mamba_architecture.pdf

Usage (from your local machine, in the venv where `modal` is installed):
    modal run generate_figure7_mamba_architecture.py
"""

import modal

# ----------------------------------------------------------------------------
# Modal app / image / volume setup
# ----------------------------------------------------------------------------
app = modal.App("eeg-figure7-architecture")

image = modal.Image.debian_slim().pip_install("matplotlib")

volume = modal.Volume.from_name("eeg-data-vol", create_if_missing=True)

OUTPUT_DIR = "/data/figures"


@app.function(image=image, volumes={"/data": volume})
def generate_figure7():
    import os
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # ------------------------------------------------------------------------
    # Global style configuration
    # ------------------------------------------------------------------------
    plt.rcParams["text.usetex"] = False
    # Switched from "stix" -> "dejavusans": dejavusans has noticeably
    # thicker/bolder strokes than STIX, which is what was causing the thin
    # math text to visually "bleed" into the teal / amber / purple fills.
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    PNG_PATH = os.path.join(OUTPUT_DIR, "figure7_mamba_architecture.png")
    PDF_PATH = os.path.join(OUTPUT_DIR, "figure7_mamba_architecture.pdf")

    # Color palette
    COLOR_NEUTRAL_FACE = "#F1F5F9"   # light grey (input / output)
    COLOR_NEUTRAL_EDGE = "#334155"   # dark slate
    COLOR_TOP_FACE = "#0D9488"       # deep teal (spatial / Riemannian branch)
    COLOR_BOTTOM_FACE = "#D97706"    # deep amber (temporal / Mamba branch)
    COLOR_FUSION_FACE = "#7C3AED"    # muted purple (fusion / output head)
    COLOR_ARROW = "#0F172A"          # near-black slate
    COLOR_TEXT_DARK = "#0F172A"
    COLOR_TEXT_LIGHT = "#FFFFFF"

    # ------------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------------
    def draw_card(ax, x, y, w, h, title, subtitle, facecolor, edgecolor,
                  textcolor, title_fs=14.0, subtitle_fs=14.0, zorder=3):
        """Floating rounded card with a soft drop shadow, a bold title
        line, and a mathtext subtitle/formula line underneath."""
        shadow = FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0,
            facecolor="#000000",
            alpha=0.15,
            zorder=zorder - 1,
        )
        ax.add_patch(shadow)

        card = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.8,
            facecolor=facecolor,
            edgecolor=edgecolor,
            zorder=zorder,
        )
        ax.add_patch(card)

        cx = x + w / 2
        ax.text(
            cx, y + h * 0.66, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight="bold",
            color=textcolor, zorder=zorder + 1,
            family="DejaVu Sans",
        )
        ax.text(
            cx, y + h * 0.27, subtitle,
            ha="center", va="center",
            fontsize=subtitle_fs, fontweight="bold",
            color=textcolor, zorder=zorder + 1,
            math_fontfamily="dejavusans",
        )

    def draw_arrow(ax, start, end, connectionstyle="arc3,rad=0.0",
                   lw=3.0, mutation_scale=20, zorder=2):
        """Thick, bold, dark-slate arrow bridging the white-space gap
        between two cards."""
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle=connectionstyle,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=COLOR_ARROW,
            zorder=zorder,
            shrinkA=2, shrinkB=2,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(arrow)

    def draw_branch_label(ax, x, y, text, color):
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=color, style="italic",
            family="DejaVu Sans",
        )

    # ------------------------------------------------------------------------
    # Figure & axes setup
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 7.6))
    ax.set_xlim(0, 18.6)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.set_aspect("equal")

    # ------------------------------------------------------------------------
    # Layout geometry
    # ------------------------------------------------------------------------
    CARD_W = 2.35
    CARD_H = 1.45          # was 1.15 -- extra height for larger subtext
    GAP_X = 0.55

    INPUT_W = 2.35
    INPUT_H = 1.55          # was 1.35

    TOP_Y = 5.05      # spatial / Riemannian track (top)  -- unchanged
    BOTTOM_Y = 0.75   # temporal / Mamba track (bottom)   -- unchanged

    INPUT_X = 0.35
    INPUT_Y = 2.85

    track_x0 = INPUT_X + INPUT_W + 1.15   # first branch box x
    A_X = [track_x0 + i * (CARD_W + GAP_X) for i in range(3)]
    B_X = A_X

    FUSION_X0 = A_X[2] + CARD_W + 1.25
    F_X = [FUSION_X0 + i * (CARD_W + GAP_X) for i in range(3)]
    FUSION_Y = 2.85
    FUSION_H = 1.65          # was 1.35

    # ------------------------------------------------------------------------
    # INPUT CARD
    # ------------------------------------------------------------------------
    draw_card(
        ax, INPUT_X, INPUT_Y, INPUT_W, INPUT_H,
        title="Input EEG Matrix",
        subtitle=r"$\mathbf{X} \in \mathbb{R}^{B \times 62 \times T}$",
        facecolor=COLOR_NEUTRAL_FACE, edgecolor=COLOR_NEUTRAL_EDGE,
        textcolor=COLOR_TEXT_DARK, title_fs=16, subtitle_fs=15,
    )

    # ------------------------------------------------------------------------
    # TOP BRANCH -- Spatial / Riemannian (Deep Teal)
    # ------------------------------------------------------------------------
    draw_branch_label(ax, (A_X[0] + A_X[2] + CARD_W) / 2, TOP_Y + CARD_H + 0.42,
                       "Spatial Branch \u2014 Riemannian Geometry", COLOR_TOP_FACE)

    draw_card(
        ax, A_X[0], TOP_Y, CARD_W, CARD_H,
        title="Covariance +\nTikhonov Reg.",
        subtitle=r"$\boldsymbol{\Sigma}=\frac{1}{T-1}\mathbf{X}\mathbf{X}^{T}+\alpha \mathbf{I}$",
        facecolor=COLOR_TOP_FACE, edgecolor=COLOR_TOP_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=14,
    )
    draw_card(
        ax, A_X[1], TOP_Y, CARD_W, CARD_H,
        title="Log-Euclidean\nMapping",
        subtitle=r"$\log(\boldsymbol{\Sigma})\rightarrow \mathbf{s}\in\mathbb{R}^{1953}$",
        facecolor=COLOR_TOP_FACE, edgecolor=COLOR_TOP_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=14,
    )
    draw_card(
        ax, A_X[2], TOP_Y, CARD_W, CARD_H,
        title="Spatial Features",
        subtitle=r"$\mathbf{f}_{s}\in\mathbb{R}^{32}$",
        facecolor=COLOR_TOP_FACE, edgecolor=COLOR_TOP_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=15,
    )

    # ------------------------------------------------------------------------
    # BOTTOM BRANCH -- Temporal / Mamba SSM (Deep Amber)
    # ------------------------------------------------------------------------
    draw_branch_label(ax, (B_X[0] + B_X[2] + CARD_W) / 2, BOTTOM_Y - 0.42,
                       "Temporal Branch \u2014 Mamba State-Space Model", COLOR_BOTTOM_FACE)

    draw_card(
        ax, B_X[0], BOTTOM_Y, CARD_W, CARD_H,
        title="Linear\nProjection",
        subtitle=r"Channels $\mathbf{62}\rightarrow$ Hidden $\mathbf{32}$",
        facecolor=COLOR_BOTTOM_FACE, edgecolor=COLOR_BOTTOM_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=14,
    )
    draw_card(
        ax, B_X[1], BOTTOM_Y, CARD_W, CARD_H,
        title="Mamba SSM\nStack",
        subtitle=r"$\mathbf{2}$ layers, $d_{\mathrm{state}}=\mathbf{16}$",
        facecolor=COLOR_BOTTOM_FACE, edgecolor=COLOR_BOTTOM_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=14,
    )
    draw_card(
        ax, B_X[2], BOTTOM_Y, CARD_W, CARD_H,
        title="Temporal Features",
        subtitle=r"$\mathbf{f}_{t}\in\mathbb{R}^{32}$",
        facecolor=COLOR_BOTTOM_FACE, edgecolor=COLOR_BOTTOM_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=15,
    )

    # ------------------------------------------------------------------------
    # FUSION & OUTPUT (Muted Purple + final neutral output)
    # ------------------------------------------------------------------------
    draw_card(
        ax, F_X[0], FUSION_Y, CARD_W, FUSION_H,
        title="Feature\nConcatenation",
        subtitle=r"$\mathbf{z}=[\mathbf{f}_{s}\,;\,\mathbf{f}_{t}]\in\mathbb{R}^{64}$",
        facecolor=COLOR_FUSION_FACE, edgecolor=COLOR_FUSION_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=14,
    )
    draw_card(
        ax, F_X[1], FUSION_Y, CARD_W, FUSION_H,
        title="Classification\nHead",
        subtitle="Linear \u2192 GELU \u2192 Linear",
        facecolor=COLOR_FUSION_FACE, edgecolor=COLOR_FUSION_FACE,
        textcolor=COLOR_TEXT_LIGHT, title_fs=14, subtitle_fs=14,
    )
    draw_card(
        ax, F_X[2], FUSION_Y, CARD_W, FUSION_H,
        title="Binary Prediction",
        subtitle="True vs. False Memory",
        facecolor=COLOR_NEUTRAL_FACE, edgecolor=COLOR_NEUTRAL_EDGE,
        textcolor=COLOR_TEXT_DARK, title_fs=15, subtitle_fs=14,
    )

    # ------------------------------------------------------------------------
    # ARROWS
    # ------------------------------------------------------------------------
    input_right_x = INPUT_X + INPUT_W
    input_mid_y = INPUT_Y + INPUT_H / 2

    # Branching arrows: Input -> A1 (up) and Input -> B1 (down)
    draw_arrow(
        ax,
        (input_right_x, input_mid_y),
        (A_X[0], TOP_Y + CARD_H / 2),
        connectionstyle="arc3,rad=-0.28",
    )
    draw_arrow(
        ax,
        (input_right_x, input_mid_y),
        (B_X[0], BOTTOM_Y + CARD_H / 2),
        connectionstyle="arc3,rad=0.28",
    )

    # Top branch internal arrows: A1 -> A2 -> A3
    for i in range(2):
        y_mid = TOP_Y + CARD_H / 2
        draw_arrow(ax, (A_X[i] + CARD_W, y_mid), (A_X[i + 1], y_mid))

    # Bottom branch internal arrows: B1 -> B2 -> B3
    for i in range(2):
        y_mid = BOTTOM_Y + CARD_H / 2
        draw_arrow(ax, (B_X[i] + CARD_W, y_mid), (B_X[i + 1], y_mid))

    # Merging arrows: A3 -> F1 (down) and B3 -> F1 (up)
    fusion_left_x = F_X[0]
    fusion_mid_y = FUSION_Y + FUSION_H / 2

    draw_arrow(
        ax,
        (A_X[2] + CARD_W, TOP_Y + CARD_H / 2),
        (fusion_left_x, fusion_mid_y),
        connectionstyle="arc3,rad=0.28",
    )
    draw_arrow(
        ax,
        (B_X[2] + CARD_W, BOTTOM_Y + CARD_H / 2),
        (fusion_left_x, fusion_mid_y),
        connectionstyle="arc3,rad=-0.28",
    )

    # Fusion internal arrows: F1 -> F2 -> F3
    for i in range(2):
        y_mid = FUSION_Y + FUSION_H / 2
        draw_arrow(ax, (F_X[i] + CARD_W, y_mid), (F_X[i + 1], y_mid))

    # ------------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------------
    fig.suptitle(
        "Figure 7: Architecture of the Dual-Branch Spatiotemporal Riemannian Mamba Network",
        fontsize=15, fontweight="bold", color=COLOR_TEXT_DARK, y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

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
    print("[DONE] Figure 7 generation complete.")


@app.local_entrypoint()
def main():
    generate_figure7.remote()