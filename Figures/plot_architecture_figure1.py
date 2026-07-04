# """
# generate_figure1.py
# ====================
# Generates "Figure 1: Overall Architecture Block Diagram" for
# SpatiotemporalRiemannianMamba (IEEE Q1 submission, ds005189 False Memory task).

# Box widths are computed automatically from the actual rendered text size
# (via matplotlib's text extent measurement), so titles/subtitles cannot
# overflow or collide with neighboring boxes regardless of font or DPI.

# Run locally:
#     pip install matplotlib
#     python generate_figure1.py

# Output:
#     figure1_architecture.png  (300 dpi, tight bounding box)
# """

# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
# plt.rcParams["mathtext.fontset"] = "stix"

# COLOR_INPUT_OUTPUT = {"face": "#EDEDED", "edge": "#5A5A5A", "text": "#2B2B2B"}
# COLOR_SPATIAL       = {"face": "#DCEEEF", "edge": "#2E6E72", "text": "#1B4144"}
# COLOR_TEMPORAL      = {"face": "#FBEAD8", "edge": "#A8632E", "text": "#5E370F"}
# COLOR_FUSION        = {"face": "#E6E2F1", "edge": "#5A4B85", "text": "#33294D"}

# ARROW_COLOR = "#444444"
# FONT_TITLE    = 9.5
# FONT_SUBTITLE = 8.5
# BOX_HEIGHT    = 1.0
# BOX_LW        = 1.1
# PAD_X         = 0.34     # horizontal padding inside box, each side
# MIN_WIDTH     = 1.9


# def text_width_data(fig, ax, s, fontsize, fontweight="normal"):
#     """Render a throwaway text object and measure its width in data units,
#     given the axes' CURRENT transform. Caller must set xlim/ylim first if
#     they need stable units, or call this after final limits are fixed and
#     re-measure once (we do a two-pass layout below)."""
#     t = ax.text(0, 0, s, fontsize=fontsize, fontweight=fontweight)
#     fig.canvas.draw()
#     bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
#     inv = ax.transData.inverted()
#     x0, _ = inv.transform((bbox.x0, bbox.y0))
#     x1, _ = inv.transform((bbox.x1, bbox.y1))
#     t.remove()
#     return abs(x1 - x0)


# class Node:
#     """A box with title/subtitle whose width is computed from rendered text."""
#     def __init__(self, title, subtitle, colors):
#         self.title = title
#         self.subtitle = subtitle
#         self.colors = colors
#         self.width = MIN_WIDTH  # placeholder, set by measure()
#         self.height = BOX_HEIGHT

#     def measure(self, fig, ax):
#         w_title = text_width_data(fig, ax, self.title, FONT_TITLE, "bold")
#         w_sub = text_width_data(fig, ax, self.subtitle, FONT_SUBTITLE) if self.subtitle else 0
#         self.width = max(w_title, w_sub) + 2 * PAD_X
#         self.width = max(self.width, MIN_WIDTH)
#         return self.width

#     def draw(self, ax, x, y):
#         """x, y = bottom-left corner. Returns (left_mid, right_mid, center)."""
#         box = FancyBboxPatch(
#             (x, y), self.width, self.height,
#             boxstyle="round,pad=0.02,rounding_size=0.07",
#             linewidth=BOX_LW, edgecolor=self.colors["edge"],
#             facecolor=self.colors["face"], zorder=3,
#         )
#         ax.add_patch(box)
#         cx, cy = x + self.width / 2, y + self.height / 2
#         if self.subtitle:
#             ax.text(cx, cy + self.height * 0.18, self.title, ha="center", va="center",
#                     fontsize=FONT_TITLE, fontweight="bold", color=self.colors["text"], zorder=4)
#             ax.text(cx, cy - self.height * 0.22, self.subtitle, ha="center", va="center",
#                     fontsize=FONT_SUBTITLE, color=self.colors["text"], zorder=4)
#         else:
#             ax.text(cx, cy, self.title, ha="center", va="center",
#                     fontsize=FONT_TITLE, fontweight="bold", color=self.colors["text"], zorder=4)
#         return (x, cy), (x + self.width, cy), (cx, cy)


# def draw_arrow(ax, start, end, color=ARROW_COLOR, lw=1.2, connectionstyle="arc3,rad=0.0"):
#     arrow = FancyArrowPatch(
#         start, end, arrowstyle="-|>", mutation_scale=12,
#         linewidth=lw, color=color, connectionstyle=connectionstyle,
#         zorder=2, shrinkA=1, shrinkB=1,
#     )
#     ax.add_patch(arrow)


# def draw_branch_label(ax, x, y, text, color):
#     ax.text(x, y, text, ha="left", va="center", fontsize=9.5, style="italic",
#             color=color, zorder=4)


# def main():
#     GAP = 0.7              # horizontal gap between boxes within a lane
#     LANE_GAP = 3.8          # vertical gap between spatial and temporal lanes
#     y_temporal = 1.8
#     y_spatial = y_temporal + LANE_GAP
#     y_center = (y_spatial + y_temporal) / 2

#     # ── Define all nodes ──────────────────────────────────────────────────
#     n_input = Node("Input EEG matrix", r"$X\in\mathbb{R}^{B\times 62\times 251}$", COLOR_INPUT_OUTPUT)

#     s1 = Node("Covariance + Tikhonov reg.", r"$\Sigma=\frac{1}{T-1}XX^{\top}+10^{-4}I$", COLOR_SPATIAL)
#     s2 = Node("Log-Euclidean mapping", r"$\log(\Sigma)\to s\in\mathbb{R}^{1953}$", COLOR_SPATIAL)
#     s3 = Node("2-layer MLP projection", "dropout = 0.6", COLOR_SPATIAL)
#     s4 = Node("Spatial features", r"$f_s \in \mathbb{R}^{32}$", COLOR_SPATIAL)

#     t1 = Node("Linear projection", "channels (62) \u2192 hidden (32)", COLOR_TEMPORAL)
#     t2 = Node("Mamba SSM stack", r"2 layers, $d_{state}=16$", COLOR_TEMPORAL)
#     t3 = Node("Temporal mean pooling", "dropout = 0.6", COLOR_TEMPORAL)
#     t4 = Node("Temporal features", r"$f_t \in \mathbb{R}^{32}$", COLOR_TEMPORAL)

#     f1 = Node("Feature concatenation", r"$z=[f_s \,\Vert\, f_t]\in\mathbb{R}^{64}$", COLOR_FUSION)
#     f2 = Node("Classification head", "Linear \u2192 GELU \u2192 Linear", COLOR_FUSION)
#     n_out = Node("Binary prediction", "True memory  /  False memory", COLOR_INPUT_OUTPUT)

#     all_nodes = [n_input, s1, s2, s3, s4, t1, t2, t3, t4, f1, f2, n_out]

#     # ── Pass 1: create a generously-sized scratch figure to measure text ───
#     fig, ax = plt.subplots(figsize=(20, 9))
#     ax.set_xlim(0, 30)
#     ax.set_ylim(0, 12)
#     ax.set_aspect("equal")
#     ax.axis("off")
#     fig.canvas.draw()
#     for n in all_nodes:
#         n.measure(fig, ax)
#     plt.close(fig)

#     # ── Per-column width = max of the spatial/temporal box at that stage ──
#     w_col1 = max(s1.width, t1.width)
#     w_col2 = max(s2.width, t2.width)
#     w_col3 = max(s3.width, t3.width)
#     w_col4 = max(s4.width, t4.width)
#     w_fuse1 = f1.width
#     w_fuse2 = f2.width

#     # ── Pass 2: lay out final x-positions using measured widths ────────────
#     x_input = 0.5
#     x_split = x_input + n_input.width + 0.9

#     x_col1 = x_split + 0.7
#     x_col2 = x_col1 + w_col1 + GAP
#     x_col3 = x_col2 + w_col2 + GAP
#     x_col4 = x_col3 + w_col3 + GAP

#     x_merge = x_col4 + w_col4 + 0.9
#     x_fuse1 = x_merge + 0.7
#     x_fuse2 = x_fuse1 + w_fuse1 + GAP
#     x_out = x_fuse2 + w_fuse2 + GAP

#     fig_w = x_out + n_out.width + 0.6
#     fig_h = y_spatial + BOX_HEIGHT + 1.6

#     fig, ax = plt.subplots(figsize=(fig_w * 0.62, fig_h * 0.85))
#     ax.set_xlim(0, fig_w)
#     ax.set_ylim(-1.3, fig_h - 1.0)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     H = BOX_HEIGHT

#     # ── Draw input ───────────────────────────────────────────────────────
#     in_L, in_R, in_C = n_input.draw(ax, x_input, y_center - H / 2)

#     ax.plot(x_split, in_C[1], marker="o", markersize=4.5, color=ARROW_COLOR, zorder=5)
#     draw_arrow(ax, in_R, (x_split, in_C[1]))
#     draw_arrow(ax, (x_split, in_C[1]), (x_col1, y_spatial + H / 2), connectionstyle="arc3,rad=0.18")
#     draw_arrow(ax, (x_split, in_C[1]), (x_col1, y_temporal + H / 2), connectionstyle="arc3,rad=-0.18")

#     # ── Spatial lane ─────────────────────────────────────────────────────
#     s1_L, s1_R, s1_C = s1.draw(ax, x_col1, y_spatial)
#     s2_L, s2_R, s2_C = s2.draw(ax, x_col2, y_spatial)
#     s3_L, s3_R, s3_C = s3.draw(ax, x_col3, y_spatial)
#     s4_L, s4_R, s4_C = s4.draw(ax, x_col4, y_spatial)
#     draw_arrow(ax, s1_R, s2_L)
#     draw_arrow(ax, s2_R, s3_L)
#     draw_arrow(ax, s3_R, s4_L)
#     draw_branch_label(ax, x_col1, y_spatial + H + 0.35,
#                        "Spatial branch \u2014 Riemannian geometry", COLOR_SPATIAL["edge"])

#     # ── Temporal lane ────────────────────────────────────────────────────
#     t1_L, t1_R, t1_C = t1.draw(ax, x_col1, y_temporal)
#     t2_L, t2_R, t2_C = t2.draw(ax, x_col2, y_temporal)
#     t3_L, t3_R, t3_C = t3.draw(ax, x_col3, y_temporal)
#     t4_L, t4_R, t4_C = t4.draw(ax, x_col4, y_temporal)
#     draw_arrow(ax, t1_R, t2_L)
#     draw_arrow(ax, t2_R, t3_L)
#     draw_arrow(ax, t3_R, t4_L)
#     draw_branch_label(ax, x_col1, y_temporal - 0.45,
#                        "Temporal branch \u2014 Mamba state-space model", COLOR_TEMPORAL["edge"])

#     # ── Merge ────────────────────────────────────────────────────────────
#     draw_arrow(ax, s4_R, (x_merge, y_center), connectionstyle="arc3,rad=-0.18")
#     draw_arrow(ax, t4_R, (x_merge, y_center), connectionstyle="arc3,rad=0.18")
#     ax.plot(x_merge, y_center, marker="o", markersize=4.5, color=ARROW_COLOR, zorder=5)

#     # ── Fusion + classification + output ────────────────────────────────
#     f1_L, f1_R, f1_C = f1.draw(ax, x_fuse1, y_center - H / 2)
#     draw_arrow(ax, (x_merge, y_center), f1_L)

#     f2_L, f2_R, f2_C = f2.draw(ax, x_fuse2, y_center - H / 2)
#     draw_arrow(ax, f1_R, f2_L)

#     o_L, o_R, o_C = n_out.draw(ax, x_out, y_center - H / 2)
#     draw_arrow(ax, f2_R, o_L)

#     # ── Caption ──────────────────────────────────────────────────────────
#     ax.text(
#         x_input, -1.05,
#         "Fig. 1.  Overall architecture of SpatiotemporalRiemannianMamba. The spatial branch (teal) projects the\n"
#         "regularized covariance matrix onto the Log-Euclidean tangent space; the temporal branch (amber) processes\n"
#         "the raw sequence through a Mamba state-space stack. Both 32-dimensional embeddings are concatenated\n"
#         "and classified.",
#         ha="left", va="top", fontsize=9, color="#333333",
#     )

#     fig.savefig("figure1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
#     print("Saved: figure1_architecture.png (300 dpi)")


# if __name__ == "__main__":
#     main()























"""
generate_figure1.py
====================
Generates "Figure 1: Overall Architecture Block Diagram" for
SpatiotemporalRiemannianMamba (IEEE Q1 submission, ds005189 False Memory task).

Box widths are computed automatically from the actual rendered text size
(via matplotlib's text extent measurement), so titles/subtitles cannot
overflow or collide with neighboring boxes regardless of font or DPI.

Run locally:
    pip install matplotlib
    python generate_figure1.py

Output:
    figure1_architecture.png  (300 dpi, tight bounding box)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

COLOR_INPUT_OUTPUT = {"face": "#EDEDED", "edge": "#5A5A5A", "text": "#2B2B2B"}
COLOR_SPATIAL       = {"face": "#DCEEEF", "edge": "#2E6E72", "text": "#1B4144"}
COLOR_TEMPORAL      = {"face": "#FBEAD8", "edge": "#A8632E", "text": "#5E370F"}
COLOR_FUSION        = {"face": "#E6E2F1", "edge": "#5A4B85", "text": "#33294D"}

ARROW_COLOR = "#444444"
FONT_TITLE    = 9.5
FONT_SUBTITLE = 8.5
BOX_HEIGHT    = 1.0
BOX_LW        = 1.1
PAD_X         = 0.34     # horizontal padding inside box, each side
MIN_WIDTH     = 1.9


def text_width_data(fig, ax, s, fontsize, fontweight="normal"):
    """Render a throwaway text object and measure its width in data units,
    given the axes' CURRENT transform. Caller must set xlim/ylim first if
    they need stable units, or call this after final limits are fixed and
    re-measure once (we do a two-pass layout below)."""
    t = ax.text(0, 0, s, fontsize=fontsize, fontweight=fontweight)
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x0, _ = inv.transform((bbox.x0, bbox.y0))
    x1, _ = inv.transform((bbox.x1, bbox.y1))
    t.remove()
    return abs(x1 - x0)


class Node:
    """A box with title/subtitle whose width is computed from rendered text."""
    def __init__(self, title, subtitle, colors):
        self.title = title
        self.subtitle = subtitle
        self.colors = colors
        self.width = MIN_WIDTH  # placeholder, set by measure()
        self.height = BOX_HEIGHT

    def measure(self, fig, ax):
        w_title = text_width_data(fig, ax, self.title, FONT_TITLE, "bold")
        w_sub = text_width_data(fig, ax, self.subtitle, FONT_SUBTITLE) if self.subtitle else 0
        self.width = max(w_title, w_sub) + 2 * PAD_X
        self.width = max(self.width, MIN_WIDTH)
        return self.width

    def draw(self, ax, x, y):
        """x, y = bottom-left corner. Returns (left_mid, right_mid, center)."""
        box = FancyBboxPatch(
            (x, y), self.width, self.height,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=BOX_LW, edgecolor=self.colors["edge"],
            facecolor=self.colors["face"], zorder=3,
        )
        ax.add_patch(box)
        cx, cy = x + self.width / 2, y + self.height / 2
        if self.subtitle:
            ax.text(cx, cy + self.height * 0.18, self.title, ha="center", va="center",
                    fontsize=FONT_TITLE, fontweight="bold", color=self.colors["text"], zorder=4)
            ax.text(cx, cy - self.height * 0.22, self.subtitle, ha="center", va="center",
                    fontsize=FONT_SUBTITLE, color=self.colors["text"], zorder=4)
        else:
            ax.text(cx, cy, self.title, ha="center", va="center",
                    fontsize=FONT_TITLE, fontweight="bold", color=self.colors["text"], zorder=4)
        return (x, cy), (x + self.width, cy), (cx, cy)


def draw_arrow(ax, start, end, color=ARROW_COLOR, lw=1.2, connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
        zorder=2, shrinkA=1, shrinkB=1,
    )
    ax.add_patch(arrow)


def draw_branch_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=9.5, style="italic",
            color=color, zorder=4)


def main():
    GAP = 0.7              # horizontal gap between boxes within a lane
    LANE_GAP = 3.8          # vertical gap between spatial and temporal lanes
    y_temporal = 1.8
    y_spatial = y_temporal + LANE_GAP
    y_center = (y_spatial + y_temporal) / 2

    # ── Define all nodes ──────────────────────────────────────────────────
    n_input = Node("Input EEG matrix", r"$X\in\mathbb{R}^{B\times 62\times 251}$", COLOR_INPUT_OUTPUT)

    s1 = Node("Covariance + Tikhonov reg.", r"$\Sigma=\frac{1}{T-1}XX^{\top}+10^{-4}I$", COLOR_SPATIAL)
    s2 = Node("Log-Euclidean mapping", r"$\log(\Sigma)\to s\in\mathbb{R}^{1953}$", COLOR_SPATIAL)
    s3 = Node("2-layer MLP projection", "dropout = 0.6", COLOR_SPATIAL)
    s4 = Node("Spatial features", r"$f_s \in \mathbb{R}^{32}$", COLOR_SPATIAL)

    t1 = Node("Linear projection", "channels (62) \u2192 hidden (32)", COLOR_TEMPORAL)
    t2 = Node("Mamba SSM stack", r"2 layers, $d_{state}=16$", COLOR_TEMPORAL)
    t3 = Node("Temporal mean pooling", "dropout = 0.6", COLOR_TEMPORAL)
    t4 = Node("Temporal features", r"$f_t \in \mathbb{R}^{32}$", COLOR_TEMPORAL)

    f1 = Node("Feature concatenation", r"$z=[f_s \,\Vert\, f_t]\in\mathbb{R}^{64}$", COLOR_FUSION)
    f2 = Node("Classification head", "Linear \u2192 GELU \u2192 Linear", COLOR_FUSION)
    n_out = Node("Binary prediction", "True memory  /  False memory", COLOR_INPUT_OUTPUT)

    all_nodes = [n_input, s1, s2, s3, s4, t1, t2, t3, t4, f1, f2, n_out]

    # ── Pass 1: create a generously-sized scratch figure to measure text ───
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    for n in all_nodes:
        n.measure(fig, ax)
    plt.close(fig)

    # ── Per-column width = max of the spatial/temporal box at that stage ──
    w_col1 = max(s1.width, t1.width)
    w_col2 = max(s2.width, t2.width)
    w_col3 = max(s3.width, t3.width)
    w_col4 = max(s4.width, t4.width)
    w_fuse1 = f1.width
    w_fuse2 = f2.width

    # ── Pass 2: lay out final x-positions using measured widths ────────────
    x_input = 0.5
    x_split = x_input + n_input.width + 0.9

    x_col1 = x_split + 0.7
    x_col2 = x_col1 + w_col1 + GAP
    x_col3 = x_col2 + w_col2 + GAP
    x_col4 = x_col3 + w_col3 + GAP

    x_merge = x_col4 + w_col4 + 0.9
    x_fuse1 = x_merge + 0.7
    x_fuse2 = x_fuse1 + w_fuse1 + GAP
    x_out = x_fuse2 + w_fuse2 + GAP

    fig_w = x_out + n_out.width + 0.6
    fig_h = y_spatial + BOX_HEIGHT + 1.6

    fig, ax = plt.subplots(figsize=(fig_w * 0.62, fig_h * 0.85))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(-1.3, fig_h - 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    H = BOX_HEIGHT

    # ── Draw input ───────────────────────────────────────────────────────
    in_L, in_R, in_C = n_input.draw(ax, x_input, y_center - H / 2)

    ax.plot(x_split, in_C[1], marker="o", markersize=4.5, color=ARROW_COLOR, zorder=5)
    draw_arrow(ax, in_R, (x_split, in_C[1]))
    draw_arrow(ax, (x_split, in_C[1]), (x_col1, y_spatial + H / 2), connectionstyle="arc3,rad=0.18")
    draw_arrow(ax, (x_split, in_C[1]), (x_col1, y_temporal + H / 2), connectionstyle="arc3,rad=-0.18")

    # ── Spatial lane ─────────────────────────────────────────────────────
    s1_L, s1_R, s1_C = s1.draw(ax, x_col1, y_spatial)
    s2_L, s2_R, s2_C = s2.draw(ax, x_col2, y_spatial)
    s3_L, s3_R, s3_C = s3.draw(ax, x_col3, y_spatial)
    s4_L, s4_R, s4_C = s4.draw(ax, x_col4, y_spatial)
    draw_arrow(ax, s1_R, s2_L)
    draw_arrow(ax, s2_R, s3_L)
    draw_arrow(ax, s3_R, s4_L)
    draw_branch_label(ax, x_col1, y_spatial + H + 0.35,
                       "Spatial branch \u2014 Riemannian geometry", COLOR_SPATIAL["edge"])

    # ── Temporal lane ────────────────────────────────────────────────────
    t1_L, t1_R, t1_C = t1.draw(ax, x_col1, y_temporal)
    t2_L, t2_R, t2_C = t2.draw(ax, x_col2, y_temporal)
    t3_L, t3_R, t3_C = t3.draw(ax, x_col3, y_temporal)
    t4_L, t4_R, t4_C = t4.draw(ax, x_col4, y_temporal)
    draw_arrow(ax, t1_R, t2_L)
    draw_arrow(ax, t2_R, t3_L)
    draw_arrow(ax, t3_R, t4_L)
    draw_branch_label(ax, x_col1, y_temporal - 0.45,
                       "Temporal branch \u2014 Mamba state-space model", COLOR_TEMPORAL["edge"])

    # ── Merge ────────────────────────────────────────────────────────────
    draw_arrow(ax, s4_R, (x_merge, y_center), connectionstyle="arc3,rad=-0.18")
    draw_arrow(ax, t4_R, (x_merge, y_center), connectionstyle="arc3,rad=0.18")
    ax.plot(x_merge, y_center, marker="o", markersize=4.5, color=ARROW_COLOR, zorder=5)

    # ── Fusion + classification + output ────────────────────────────────
    f1_L, f1_R, f1_C = f1.draw(ax, x_fuse1, y_center - H / 2)
    draw_arrow(ax, (x_merge, y_center), f1_L)

    f2_L, f2_R, f2_C = f2.draw(ax, x_fuse2, y_center - H / 2)
    draw_arrow(ax, f1_R, f2_L)

    o_L, o_R, o_C = n_out.draw(ax, x_out, y_center - H / 2)
    draw_arrow(ax, f2_R, o_L)

    # ── Caption ──────────────────────────────────────────────────────────
    ax.text(
        x_input, -1.05,
        "Fig. 1.  Overall architecture of SpatiotemporalRiemannianMamba. The spatial branch (teal) projects the\n"
        "regularized covariance matrix onto the Log-Euclidean tangent space; the temporal branch (amber) processes\n"
        "the raw sequence through a Mamba state-space stack. Both 32-dimensional embeddings are concatenated\n"
        "and classified.",
        ha="left", va="top", fontsize=9, color="#333333",
    )

    fig.savefig("figure1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure1_architecture.png (300 dpi)")


if __name__ == "__main__":
    main()
