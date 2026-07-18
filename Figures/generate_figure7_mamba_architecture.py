# commands to run :
# modal run generate_figure7_mamba_architecture.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure7_mamba_architecture.png .
# modal volume get eeg-data-vol /figures/figure7_mamba_architecture.pdf .
"""
generate_figure7_mamba_architecture.py

Generates "Figure 7: Architecture of the Dual-Branch Spatiotemporal
Riemannian Mamba Network" -- a publication-quality Modern Minimalist
architecture diagram for the BCI EEG false-memory classification
manuscript.

DESIGN NOTE (this version -- v6, "Collision Fix Pass" on top of the v5
Polish Pass):
    Two geometric placement bugs from v5 are fixed here. Everything
    else (colors, box positions, absolute coordinate system) is
    unchanged.

    1. TOP LABEL / ARROW COLLISION FIXED. The "Spatial Branch --
       Riemannian Geometry" / "Temporal Branch -- Mamba State-Space
       Model" labels sat at y=78, only 4 units above the fork point
       at y=74, and the curved fork arrows (rad=+-0.15) bulged enough
       to clip through the text. Fixed by:
         - moving both labels further from the input card AND further
           from the arrow forks: y 78 -> 81 (higher, i.e. closer to
           the Input card, farther from the fork's vertical range),
           and moved them wider apart in x (25/75 -> 22/78) so the
           diagonal fork arrows -- which travel toward x=25/75 anyway
           -- pass outside the (now wider) text bounding boxes.
         - reducing the fork curvature (rad 0.15 -> 0.08) so the
           arrows bulge less and stay clear of the label band.
       Net effect: labels and arrows now occupy clearly separated
       bands with real clearance, instead of a 4-unit margin that a
       curved arrow could eat into.

    2. BOTTOM ARROWHEAD OVERLAP FIXED. The two straight diagonal
       merge arrows (Spatial Features -> Fusion, Temporal Features ->
       Fusion) both used to target the identical point (50, join_y),
       so the two large arrowheads landed on top of each other and
       rendered as a single ugly black blob. Fixed by giving each
       arrow its own distinct landing x on the Feature Concatenation
       box's top border: the teal (spatial) arrow now targets
       fusion_cx - 8, and the orange (temporal) arrow targets
       fusion_cx + 8, both just above the box's true top edge (+
       SAFE_MARGIN). The old single shared "join point" and the
       redundant join-point -> fusion-top stub arrow are removed,
       since each merge arrow now reaches the box directly.

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
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    PNG_PATH = os.path.join(OUTPUT_DIR, "figure7_mamba_architecture.png")
    PDF_PATH = os.path.join(OUTPUT_DIR, "figure7_mamba_architecture.pdf")

    # Color palette -- premium, softened tones
    COLOR_NEUTRAL_FACE = "#F8F9F9"   # light grey (input / final output)
    COLOR_TOP_FACE = "#16A085"       # soft sea-green/teal (spatial branch)
    COLOR_BOTTOM_FACE = "#E67E22"    # muted professional orange (temporal branch)
    COLOR_FUSION_FACE = "#8E44AD"    # rich muted purple (fusion / class. head)
    COLOR_SOFT_BORDER = "#CBD5E1"    # soft grey outline, no heavy black
    COLOR_ARROW = "#334155"          # soft slate
    COLOR_TEXT_DARK = "#0F172A"
    COLOR_TEXT_LIGHT = "#FFFFFF"

    SAFE_MARGIN = 1.5          # fixed margin nudged off every box edge for arrows
    ARROW_LW = 2.2
    ARROW_MUTATION = 18

    # ------------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------------
    def draw_card(ax, x, y, w, h, title, subtitle, face, text,
                   title_fs=13, sub_fs=12, zorder=3):
        """Draws a single standalone floating rounded card at the exact
        absolute (x, y) bottom-left coordinate given, with soft grey
        outline and generous rounding -- no heavy black border, no
        drop shadow."""
        card = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.5,rounding_size=2",
            facecolor=face, edgecolor=COLOR_SOFT_BORDER, linewidth=1.5,
            zorder=zorder,
        )
        ax.add_patch(card)

        cx = x + w / 2
        ax.text(
            cx, y + h * 0.62, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight="bold",
            color=text, zorder=zorder + 1,
            family="DejaVu Sans",
        )
        ax.text(
            cx, y + h * 0.28, subtitle,
            ha="center", va="center",
            fontsize=sub_fs, fontweight="bold",
            color=text, zorder=zorder + 1,
            math_fontfamily="dejavusans",
        )

    def box_top_center(x, y, w, h):
        return (x + w / 2, y + h)

    def box_bottom_center(x, y, w, h):
        return (x + w / 2, y)

    def draw_arrow(ax, start_pt, end_pt, rad=0.0, zorder=10,
                    shrinkA=0, shrinkB=0):
        """Draws an arrow between two points. By default both points
        are assumed ALREADY-MARGIN-ADJUSTED and shrinkA/shrinkB stay
        at 0 (the margin was already baked into start_pt / end_pt by
        the caller). Pass non-zero shrinkA/shrinkB for the rare case
        where raw box-edge points are supplied instead and the gap
        should be handled by matplotlib's own shrink mechanism."""
        arrow = FancyArrowPatch(
            start_pt, end_pt,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=ARROW_MUTATION,
            linewidth=ARROW_LW,
            color=COLOR_ARROW,
            zorder=zorder,
            shrinkA=shrinkA, shrinkB=shrinkB,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(arrow)

    def connect_down(ax, box_a, box_b, rad=0.0):
        """Straight-down (or gently curved) connector from the bottom
        of box_a to the top of box_b, each point nudged inward by
        SAFE_MARGIN so the arrow starts/ends just short of the
        rounded edge. Generic over whatever box tuples are passed in,
        so re-spacing a stack is just a matter of updating the box
        constants -- no separate arrow code to touch."""
        ax_, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        sx, sy = box_bottom_center(ax_, ay, aw, ah)
        ex, ey = box_top_center(bx, by, bw, bh)
        start_pt = (sx, sy - SAFE_MARGIN)
        end_pt = (ex, ey + SAFE_MARGIN)
        draw_arrow(ax, start_pt, end_pt, rad=rad)

    def draw_branch_label(ax, x, y, text, color):
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=13.5, fontweight="bold",
            color=color, style="italic",
            family="DejaVu Sans",
        )

    # ------------------------------------------------------------------------
    # Figure & axes setup -- fixed absolute grid (extended downward to fit
    # the re-spaced fusion stack, which now reaches y = -13)
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 15))
    ax.set_xlim(0, 100)
    ax.set_ylim(-18, 100)
    ax.axis("off")

    # ------------------------------------------------------------------------
    # ABSOLUTE COORDINATE SPECIFICATIONS (all literal, no derived math)
    # ------------------------------------------------------------------------
    # -- Input card --
    INPUT_BOX = (30, 85, 40, 10)  # x, y, w, h

    # -- Branch labels -- FIX #1: moved higher (78 -> 81, closer to the
    # Input card / farther from the arrow-fork region below) and wider
    # apart (25/75 -> 22/78), so the curved fork arrows can no longer
    # clip through the label text.
    SPATIAL_LABEL = (22, 81)
    TEMPORAL_LABEL = (78, 81)

    # -- Spatial branch stack (teal) --
    SPATIAL_BOX_1 = (10, 62, 30, 9)
    SPATIAL_BOX_2 = (10, 47, 30, 9)
    SPATIAL_BOX_3 = (10, 32, 30, 9)

    # -- Temporal branch stack (orange) --
    TEMPORAL_BOX_1 = (60, 62, 30, 9)
    TEMPORAL_BOX_2 = (60, 47, 30, 9)
    TEMPORAL_BOX_3 = (60, 32, 30, 9)

    # -- Fusion stack (purple, purple, grey) --
    FUSION_BOX_1 = (30, 17, 40, 10)
    FUSION_BOX_2 = (30, 2, 40, 10)
    FUSION_BOX_3 = (30, -13, 40, 10)

    # ------------------------------------------------------------------------
    # INPUT CARD
    # ------------------------------------------------------------------------
    draw_card(
        ax, *INPUT_BOX,
        title="Input EEG Matrix",
        subtitle=r"$\mathbf{X} \in \mathbb{R}^{B \times 62 \times T}$",
        face=COLOR_NEUTRAL_FACE, text=COLOR_TEXT_DARK,
        title_fs=17, sub_fs=15,
    )

    # ------------------------------------------------------------------------
    # BRANCH LABELS -- FIX #1: repositioned (see SPATIAL_LABEL /
    # TEMPORAL_LABEL above) so they sit clear of the fork arrows below.
    # ------------------------------------------------------------------------
    draw_branch_label(ax, *SPATIAL_LABEL,
                       "Spatial Branch \u2014 Riemannian Geometry",
                       COLOR_TOP_FACE)
    draw_branch_label(ax, *TEMPORAL_LABEL,
                       "Temporal Branch \u2014 Mamba State-Space Model",
                       COLOR_BOTTOM_FACE)

    # ------------------------------------------------------------------------
    # SPATIAL / RIEMANNIAN BRANCH (soft teal)
    # ------------------------------------------------------------------------
    draw_card(ax, *SPATIAL_BOX_1,
              title="Covariance + Tikhonov Reg.",
              subtitle=r"$\boldsymbol{\Sigma}=\frac{1}{T-1}\mathbf{X}\mathbf{X}^{T}+\alpha \mathbf{I}$",
              face=COLOR_TOP_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=13, sub_fs=13)
    draw_card(ax, *SPATIAL_BOX_2,
              title="Log-Euclidean Mapping",
              subtitle=r"$\log(\boldsymbol{\Sigma})\rightarrow \mathbf{s}\in\mathbb{R}^{1953}$",
              face=COLOR_TOP_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=13, sub_fs=13)
    draw_card(ax, *SPATIAL_BOX_3,
              title="Spatial Features",
              subtitle=r"$\mathbf{f}_{s}\in\mathbb{R}^{32}$",
              face=COLOR_TOP_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=13, sub_fs=14)

    connect_down(ax, SPATIAL_BOX_1, SPATIAL_BOX_2)
    connect_down(ax, SPATIAL_BOX_2, SPATIAL_BOX_3)

    # ------------------------------------------------------------------------
    # TEMPORAL / MAMBA SSM BRANCH (muted orange)
    # ------------------------------------------------------------------------
    draw_card(ax, *TEMPORAL_BOX_1,
              title="Linear Projection",
              subtitle=r"Channels $\mathbf{62} \rightarrow$ Hidden $\mathbf{32}$",
              face=COLOR_BOTTOM_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=13, sub_fs=13)
    draw_card(ax, *TEMPORAL_BOX_2,
              title="Mamba SSM Stack",
              subtitle=r"$\mathbf{2}$ layers, $d_{\mathrm{state}}=\mathbf{16}$",
              face=COLOR_BOTTOM_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=13, sub_fs=13)
    draw_card(ax, *TEMPORAL_BOX_3,
              title="Temporal Features",
              subtitle=r"$\mathbf{f}_{t}\in\mathbb{R}^{32}$",
              face=COLOR_BOTTOM_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=13, sub_fs=14)

    connect_down(ax, TEMPORAL_BOX_1, TEMPORAL_BOX_2)
    connect_down(ax, TEMPORAL_BOX_2, TEMPORAL_BOX_3)

    # ------------------------------------------------------------------------
    # FUSION + CLASSIFICATION HEAD + BINARY PREDICTION
    # ------------------------------------------------------------------------
    draw_card(ax, *FUSION_BOX_1,
              title="Feature Concatenation",
              subtitle=r"$\mathbf{z}=[\mathbf{f}_{s}\,;\,\mathbf{f}_{t}]\in\mathbb{R}^{64}$",
              face=COLOR_FUSION_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=14, sub_fs=13)
    draw_card(ax, *FUSION_BOX_2,
              title="Classification Head",
              subtitle="Linear \u2192 GELU \u2192 Linear",
              face=COLOR_FUSION_FACE, text=COLOR_TEXT_LIGHT,
              title_fs=14, sub_fs=12)
    draw_card(ax, *FUSION_BOX_3,
              title="Binary Prediction",
              subtitle="True vs. False Memory",
              face=COLOR_NEUTRAL_FACE, text=COLOR_TEXT_DARK,
              title_fs=17, sub_fs=12)

    # Feature Concatenation -> Classification Head
    connect_down(ax, FUSION_BOX_1, FUSION_BOX_2)
    # Classification Head -> Binary Prediction
    connect_down(ax, FUSION_BOX_2, FUSION_BOX_3)

    # ------------------------------------------------------------------------
    # ARROWS -- branching (Input -> Spatial / Temporal) and merging
    # (Spatial / Temporal -> Fusion). Every point is the box's true
    # top/bottom-center, manually nudged by a fixed margin, per spec.
    # ------------------------------------------------------------------------
    input_bottom = box_bottom_center(*INPUT_BOX)          # (50, 85)
    spatial_top = box_top_center(*SPATIAL_BOX_1)          # (25, 71)
    temporal_top = box_top_center(*TEMPORAL_BOX_1)        # (75, 71)

    trunk_start = (input_bottom[0], input_bottom[1] - SAFE_MARGIN)
    trunk_split_y = 74.0  # strictly below the label band (labels now at y=81)

    # Straight trunk stub: Input -> split point (centered under Input,
    # horizontally clear of both branch labels).
    draw_arrow(ax, trunk_start, (50, trunk_split_y), rad=0.0)

    # Diagonal branching arrows -- leave from two points 2 units apart
    # (48 / 52) for a clean visual fork. FIX #1: curvature reduced
    # (0.15 -> 0.08) so the bulge stays well clear of the (now higher,
    # wider) branch labels above.
    FORK_LEFT_X, FORK_RIGHT_X = 48.0, 52.0
    draw_arrow(ax, (FORK_LEFT_X, trunk_split_y),
               (spatial_top[0], spatial_top[1] + SAFE_MARGIN), rad=0.08)
    draw_arrow(ax, (FORK_RIGHT_X, trunk_split_y),
               (temporal_top[0], temporal_top[1] + SAFE_MARGIN), rad=-0.08)

    spatial_bottom = box_bottom_center(*SPATIAL_BOX_3)    # (25, 32)
    temporal_bottom = box_bottom_center(*TEMPORAL_BOX_3)  # (75, 32)
    fusion_top = box_top_center(*FUSION_BOX_1)            # (50, 27)
    fusion_cx = fusion_top[0]                             # 50

    # FIX #2: the two merge arrows used to both target the identical
    # point (50, join_y), causing their arrowheads to overlap into a
    # black blob. Each now targets its own distinct x on the Feature
    # Concatenation box's top border: teal -> fusion_cx - 8, orange ->
    # fusion_cx + 8, both just above the box's true top edge
    # (+ SAFE_MARGIN) so neither arrowhead pierces the purple card.
    spatial_merge_target = (fusion_cx - 8, fusion_top[1] + SAFE_MARGIN)
    temporal_merge_target = (fusion_cx + 8, fusion_top[1] + SAFE_MARGIN)

    draw_arrow(ax, spatial_bottom, spatial_merge_target, rad=0.0,
               shrinkA=2, shrinkB=5)
    draw_arrow(ax, temporal_bottom, temporal_merge_target, rad=0.0,
               shrinkA=2, shrinkB=5)

    # ------------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------------
    fig.suptitle(
        "Figure 7: Architecture of the Dual-Branch Spatiotemporal Riemannian Mamba Network",
        x=0.5, ha="center", fontweight="bold", fontsize=16,
        color=COLOR_TEXT_DARK, y=0.95,
    )

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