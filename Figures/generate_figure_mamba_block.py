"""
generate_figure_mamba_block.py
================================
Generates "Figure: Internal Architecture of the Mamba / State-Space Model
(SSM) Block" for the SpatiotemporalRiemannianMamba IEEE Q1 submission.

Box widths are computed from actual rendered text extents (measured via
matplotlib), so labels cannot overflow or collide regardless of font or
DPI — the same approach used for the main architecture diagram.

Run locally:
    pip install matplotlib
    python generate_figure_mamba_block.py

Output:
    figure_mamba_block.png  (300 dpi, tight bounding box, no title)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

COLOR_TEAL   = {"face": "#D9EDED", "edge": "#1A7A7A", "text": "#0E4444"}
COLOR_BLUE   = {"face": "#D8E3EC", "edge": "#0B3C5D", "text": "#0B3C5D"}
COLOR_GRAY   = {"face": "#EBEBEB", "edge": "#888888", "text": "#3A3A3A"}
COLOR_AMBER  = {"face": "#F7E9C7", "edge": "#B8860B", "text": "#6B4F08"}
COLOR_NODE   = {"face": "#F0F0F0", "edge": "#444444", "text": "#222222"}

ARROW_COLOR = "#3A3A3A"
FONT_TITLE    = 9.6
FONT_SUBTITLE = 8.4
BOX_HEIGHT    = 0.85
BOX_LW        = 1.15
PAD_X         = 0.30
MIN_WIDTH     = 1.5


def text_width_data(fig, ax, s, fontsize, fontweight="normal"):
    t = ax.text(0, 0, s, fontsize=fontsize, fontweight=fontweight)
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x0, _ = inv.transform((bbox.x0, bbox.y0))
    x1, _ = inv.transform((bbox.x1, bbox.y1))
    t.remove()
    return abs(x1 - x0)


class Node:
    def __init__(self, title, subtitle, colors, width_override=None):
        self.title = title
        self.subtitle = subtitle
        self.colors = colors
        self.width = MIN_WIDTH
        self.height = BOX_HEIGHT
        self.width_override = width_override

    def measure(self, fig, ax):
        if self.width_override is not None:
            self.width = self.width_override
            return self.width
        w_title = text_width_data(fig, ax, self.title, FONT_TITLE, "bold")
        w_sub = text_width_data(fig, ax, self.subtitle, FONT_SUBTITLE) if self.subtitle else 0
        self.width = max(w_title, w_sub) + 2 * PAD_X
        self.width = max(self.width, MIN_WIDTH)
        return self.width

    def draw(self, ax, x, y):
        box = FancyBboxPatch(
            (x, y), self.width, self.height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=BOX_LW, edgecolor=self.colors["edge"],
            facecolor=self.colors["face"], zorder=3,
        )
        ax.add_patch(box)
        cx, cy = x + self.width / 2, y + self.height / 2
        if self.subtitle:
            ax.text(cx, cy + self.height * 0.20, self.title, ha="center", va="center",
                    fontsize=FONT_TITLE, fontweight="bold", color=self.colors["text"], zorder=4)
            ax.text(cx, cy - self.height * 0.24, self.subtitle, ha="center", va="center",
                    fontsize=FONT_SUBTITLE, color=self.colors["text"], zorder=4)
        else:
            ax.text(cx, cy, self.title, ha="center", va="center",
                    fontsize=FONT_TITLE, fontweight="bold", color=self.colors["text"], zorder=4)
        return (x, cy), (x + self.width, cy), (cx, cy), (cx, y + self.height), (cx, y)


def draw_arrow(ax, start, end, color=ARROW_COLOR, lw=1.2, connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
        zorder=2, shrinkA=1, shrinkB=1,
    )
    ax.add_patch(arrow)


def main():
    GAP = 0.65

    # ── Define nodes ─────────────────────────────────────────────────────
    n_input = Node("Input sequence", r"$X \in \mathbb{R}^{T \times D}$", COLOR_NODE)

    n_proj = Node("Linear projection", "splits into two branches", COLOR_BLUE)

    # Top / Main branch (strictly sequential)
    n_conv  = Node("Causal 1-D convolution", "depthwise, kernel size 4", COLOR_TEAL)
    n_silu1 = Node("SiLU activation", None, COLOR_TEAL, width_override=2.0)
    n_param = Node("Parameterization", r"$A,\ B,\ C,\ \Delta$ from input", COLOR_AMBER)
    n_disc  = Node("Discretization",
                   r"$\bar{A}=\exp(\Delta A),\ \bar{B}=(\Delta A)^{-1}(\exp(\Delta A)-I)\Delta B$",
                   COLOR_AMBER)
    n_ssm   = Node("Selective SSM scan",
                   r"$h_t=\bar{A}h_{t-1}+\bar{B}x_t,\ \ y_t=Ch_t$",
                   COLOR_AMBER)

    # Bottom / Gating branch
    n_gate_silu = Node("SiLU activation", "gating branch", COLOR_GRAY)

    # Merge + output
    n_mul     = Node(r"$\otimes$", "element-wise gating", COLOR_BLUE, width_override=1.6)
    n_outproj = Node("Linear projection", None, COLOR_BLUE)
    n_output  = Node("Output sequence", r"$Y \in \mathbb{R}^{T \times D}$", COLOR_NODE)

    all_nodes = [n_input, n_proj, n_conv, n_silu1, n_param, n_disc, n_ssm,
                 n_gate_silu, n_mul, n_outproj, n_output]

    # ── Pass 1: measure on a scratch figure ──────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    for n in all_nodes:
        n.measure(fig, ax)
    plt.close(fig)

    # ── Layout ───────────────────────────────────────────────────────────
    #
    # Two-row top branch:
    #   Row 1 (y_main):    [conv]  [silu1]
    #   Row 2 (y_ssm_row):         [param] [disc] [ssm]
    #
    # silu1 and param share the same column (col-2), so the
    # silu1 → param arrow is a perfectly straight vertical drop.
    #
    H = BOX_HEIGHT
    y_main    = 8.6    # row 1 of top branch  (conv, silu1)
    y_gate    = 2.2    # bottom / gating branch lane
    y_center  = (y_main + y_gate) / 2   # centre y for input/proj/⊗/outproj/output boxes
    y_ssm_row = y_main - 2.9            # row 2 of top branch (param, disc, ssm)
    #                                     chosen so top of row-2 boxes stays above
    #                                     the centre-lane boxes (no overlap)

    x_input = 0.4
    x_proj  = x_input + n_input.width + 1.0
    x_split = x_proj  + n_proj.width  + 0.9

    # Col-1 (row 1 only): conv
    x_s1 = x_split + 0.7

    # Col-2 (rows 1 & 2): silu1 on top, param below — centred on the same x
    #   column width = max of the two box widths so neither overflows
    w_col2  = max(n_silu1.width, n_param.width)
    x_s2    = x_s1 + n_conv.width + GAP          # left edge of col-2

    # Col-3 (row 2 only): disc
    x_s3    = x_s2 + w_col2 + GAP

    # Col-4 (row 2 only): ssm
    x_s4    = x_s3 + n_disc.width + GAP

    # Merge + output columns
    x_mul     = x_s4 + n_ssm.width + 1.1
    x_outproj = x_mul + n_mul.width + GAP
    x_output  = x_outproj + n_outproj.width + GAP

    fig_w = x_output + n_output.width + 0.6
    fig_h = y_main   + H + 1.4

    fig, ax = plt.subplots(figsize=(fig_w * 0.62, fig_h * 0.62))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(-0.4, fig_h - 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Input ────────────────────────────────────────────────────────────
    in_L, in_R, in_C, in_T, in_B = n_input.draw(ax, x_input, y_center - H / 2)

    # ── Linear projection (pre-split) ───────────────────────────────────
    p_L, p_R, p_C, p_T, p_B = n_proj.draw(ax, x_proj, y_center - H / 2)
    draw_arrow(ax, in_R, p_L)

    # ── Top branch — row 1: conv → silu1 ────────────────────────────────
    # Centre each box within its column so the vertical arrow is plumb.
    conv_draw_x  = x_s1                                         # col-1 has only conv
    silu1_draw_x = x_s2 + (w_col2 - n_silu1.width) / 2         # centred in col-2

    conv_L,  conv_R,  conv_C,  _,  _ = n_conv.draw(ax,  conv_draw_x,  y_main)
    silu1_L, silu1_R, silu1_C, _,  _ = n_silu1.draw(ax, silu1_draw_x, y_main)
    draw_arrow(ax, conv_R, silu1_L)   # ① conv → silu1 (horizontal)

    # ── Bottom / gating branch ───────────────────────────────────────────
    # Centre under col-2 so the layout reads as two symmetric branches.
    gate_draw_x = x_s2 + (w_col2 - n_gate_silu.width) / 2
    gate_L, gate_R, gate_C, _, _ = n_gate_silu.draw(ax, gate_draw_x, y_gate)

    # ── Split node → exactly TWO branches ───────────────────────────────
    split_pt = (x_split, p_C[1])
    ax.plot(*split_pt, marker="o", markersize=4.5, color=ARROW_COLOR, zorder=5)
    draw_arrow(ax, p_R,       split_pt)                                         # proj  → split
    draw_arrow(ax, split_pt,  conv_L,  connectionstyle="arc3,rad=0.18")         # ② split → conv  (top)
    draw_arrow(ax, split_pt,  gate_L,  connectionstyle="arc3,rad=-0.18")        # ③ split → gate  (bottom)

    # ── Top branch — row 2: param → disc → ssm ──────────────────────────
    # param is centred in col-2, directly below silu1.
    param_draw_x = x_s2 + (w_col2 - n_param.width) / 2         # centred in col-2

    param_L, param_R, param_C, param_T, param_B = n_param.draw(ax, param_draw_x, y_ssm_row)
    disc_L,  disc_R,  disc_C,  disc_T,  disc_B  = n_disc.draw(ax,  x_s3,         y_ssm_row)
    ssm_L,   ssm_R,   ssm_C,   ssm_T,   ssm_B   = n_ssm.draw(ax,   x_s4,         y_ssm_row)

    # ④ silu1 → param  — straight vertical drop (centres share the same x)
    silu1_bottom_pt = (silu1_C[0], y_main)          # bottom-centre of silu1
    draw_arrow(ax, silu1_bottom_pt, param_T)

    # ⑤ param → disc → ssm  (sequential horizontal)
    draw_arrow(ax, param_R, disc_L)
    draw_arrow(ax, disc_R,  ssm_L)

    # SSM-core bracket label
    ax.text(
        (param_L[0] + ssm_R[0]) / 2, y_ssm_row - 0.55,
        "Selective state-space model (SSM) core",
        ha="center", va="center", fontsize=9.3, style="italic", color=COLOR_AMBER["edge"],
    )

    # ── Merge: element-wise multiplication ⊗ ────────────────────────────
    mul_draw_y = y_center - n_mul.height / 2
    mul_L, mul_R, mul_C, _, _ = n_mul.draw(ax, x_mul, mul_draw_y)

    # ⑥ ssm → ⊗  (arc upward from the right side of ssm)
    draw_arrow(ax, ssm_R, mul_L, connectionstyle="arc3,rad=-0.35")

    # ⑦ gate silu → ⊗  (keep original two-segment routing, unchanged)
    draw_arrow(ax, gate_R, (x_mul, y_gate),  connectionstyle="arc3,rad=0.0")
    draw_arrow(ax, (x_mul, y_gate), mul_L,   connectionstyle="arc3,rad=0.22")

    # ── Output projection + output sequence ─────────────────────────────
    outp_L, outp_R, outp_C, _, _ = n_outproj.draw(ax, x_outproj, y_center - H / 2)
    draw_arrow(ax, mul_R, outp_L)

    out_L, out_R, out_C, _, _ = n_output.draw(ax, x_output, y_center - H / 2)
    draw_arrow(ax, outp_R, out_L)

    plt.tight_layout()
    fig.savefig("figure_mamba_block.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure_mamba_block.png (300 dpi)")


if __name__ == "__main__":
    main()