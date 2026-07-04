# """
# generate_figure2.py
# ====================
# Generates "Figure 2: Performance Comparison" for the
# SpatiotemporalRiemannianMamba IEEE Q1 submission (29-fold LOSO results).

# Run locally:
#     pip install matplotlib seaborn
#     python generate_figure2.py

# Output:
#     figure2_results.png  (300 dpi, tight bounding box)
# """

# import matplotlib.pyplot as plt
# import seaborn as sns


# # ─────────────────────────────────────────────────────────────────────────────
# # Global style
# # ─────────────────────────────────────────────────────────────────────────────
# plt.style.use("seaborn-v0_8-whitegrid")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
# plt.rcParams["mathtext.fontset"] = "stix"

# # ─────────────────────────────────────────────────────────────────────────────
# # Empirical data (29-fold LOSO cross-validation)
# # ─────────────────────────────────────────────────────────────────────────────
# MODELS    = ["SpatioMamba", "BaselineCNN"]
# MEANS     = [55.53, 52.14]      # %
# STD_DEVS  = [5.25, 4.21]        # %
# P_VALUE   = 0.0093

# # Colors — teal matches the architecture diagram's spatial-branch color;
# # gray is neutral for the baseline.
# COLOR_SPATIOMAMBA = "#2E6E72"   # teal (same edge tone as Fig. 1 spatial branch)
# COLOR_BASELINE    = "#8C8C8C"   # neutral gray
# BAR_COLORS = [COLOR_SPATIOMAMBA, COLOR_BASELINE]

# CHANCE_LEVEL = 50.0


# def significance_stars(p):
#     """Map a p-value to conventional significance asterisks."""
#     if p < 0.001:
#         return "***"
#     elif p < 0.01:
#         return "**"
#     elif p < 0.05:
#         return "*"
#     return "n.s."


# def draw_significance_bracket(ax, x1, x2, y, h, label, color="#222222", lw=1.3, fontsize=10.5):
#     """
#     Draw a horizontal significance bracket between x1 and x2 at height y,
#     with small downward ticks at each end, and a centered label above it.

#     x1, x2 : x-coordinates of the two bar centers being compared
#     y      : y-coordinate of the horizontal bracket line
#     h      : length of the downward ticks at each end
#     label  : text to place above the bracket (e.g. "** (p < 0.01)")
#     """
#     # Horizontal connecting line
#     ax.plot([x1, x2], [y, y], color=color, linewidth=lw, zorder=5)
#     # Downward ticks at each end
#     ax.plot([x1, x1], [y, y - h], color=color, linewidth=lw, zorder=5)
#     ax.plot([x2, x2], [y, y - h], color=color, linewidth=lw, zorder=5)
#     # Centered label above the bracket
#     ax.text((x1 + x2) / 2, y + 0.35, label, ha="center", va="bottom",
#             fontsize=fontsize, color=color, zorder=5)


# def main():
#     fig, ax = plt.subplots(figsize=(6.2, 5.6))

#     x_pos = [0, 1]
#     bar_width = 0.55

#     bars = ax.bar(
#         x_pos, MEANS, width=bar_width, color=BAR_COLORS,
#         edgecolor="#2B2B2B", linewidth=1.0, zorder=3,
#     )

#     # Error bars (standard deviation), capsize=5 as required
#     ax.errorbar(
#         x_pos, MEANS, yerr=STD_DEVS, fmt="none",
#         ecolor="#2B2B2B", elinewidth=1.3, capsize=5, capthick=1.3,
#         zorder=4,
#     )

#     # ── Axis labels, limits, ticks ──────────────────────────────────────────
#     ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
#     ax.set_ylim(45, 65)
#     ax.set_xlim(-0.65, 1.65)
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(MODELS, fontsize=11.5)
#     ax.tick_params(axis="y", labelsize=10.5)

#     # ── Chance-level reference line ──────────────────────────────────────────
#     ax.axhline(CHANCE_LEVEL, color="#555555", linestyle="--", linewidth=1.1, zorder=2)
#     ax.text(
#         1.63, CHANCE_LEVEL + 0.35, "Chance level", ha="right", va="bottom",
#         fontsize=9.5, color="#555555", style="italic",
#     )

#     # ── Value labels on top of each bar (just above its error bar cap) ─────
#     for xi, mean, std in zip(x_pos, MEANS, STD_DEVS):
#         ax.text(xi, mean + std + 0.5, f"{mean:.2f}%", ha="center", va="bottom",
#                  fontsize=10, color="#2B2B2B")

#     # ── Significance bracket connecting the tops of the two error bars ─────
#     top1 = MEANS[0] + STD_DEVS[0]
#     top2 = MEANS[1] + STD_DEVS[1]
#     bracket_y = max(top1, top2) + 2.2     # sits above both error-bar caps + labels
#     tick_h = 0.5
#     sig_label = f"{significance_stars(P_VALUE)} (p < 0.01)"
#     draw_significance_bracket(ax, x_pos[0], x_pos[1], bracket_y, tick_h, sig_label)

#     # Headroom so the bracket + its label fit inside the axes
#     ax.set_ylim(45, max(65, bracket_y + 2.0))

#     # ── Spines / grid cleanup ───────────────────────────────────────────────
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(axis="x", visible=False)
#     ax.set_axisbelow(True)

#     ax.set_title(
#         "Fig. 2.  LOSO cross-validation accuracy: SpatioMamba vs. BaselineCNN",
#         fontsize=11, pad=14,
#     )

#     plt.tight_layout()
#     fig.savefig("figure2_results.png", dpi=300, bbox_inches="tight", facecolor="white")
#     print("Saved: figure2_results.png (300 dpi)")


# if __name__ == "__main__":
#     main()





























"""
generate_figure2.py
====================
Generates "Figure 2: Performance Comparison" for the
SpatiotemporalRiemannianMamba IEEE Q1 submission (29-fold LOSO results).

Run locally:
    pip install matplotlib seaborn
    python generate_figure2.py

Output:
    figure2_results.png  (300 dpi, tight bounding box)
"""

import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# ─────────────────────────────────────────────────────────────────────────────
# Empirical data (29-fold LOSO cross-validation)
# ─────────────────────────────────────────────────────────────────────────────
MODELS    = ["SpatioMamba", "BaselineCNN"]
MEANS     = [55.53, 52.14]      # %
STD_DEVS  = [5.25, 4.21]        # %
P_VALUE   = 0.0093

# Colors — teal matches the architecture diagram's spatial-branch color;
# gray is neutral for the baseline.
COLOR_SPATIOMAMBA = "#2E6E72"   # teal (same edge tone as Fig. 1 spatial branch)
COLOR_BASELINE    = "#8C8C8C"   # neutral gray
BAR_COLORS = [COLOR_SPATIOMAMBA, COLOR_BASELINE]

CHANCE_LEVEL = 50.0


def significance_stars(p):
    """Map a p-value to conventional significance asterisks."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def draw_significance_bracket(ax, x1, x2, y, h, label, color="#222222", lw=1.3, fontsize=10.5):
    """
    Draw a horizontal significance bracket between x1 and x2 at height y,
    with small downward ticks at each end, and a centered label above it.

    x1, x2 : x-coordinates of the two bar centers being compared
    y      : y-coordinate of the horizontal bracket line
    h      : length of the downward ticks at each end
    label  : text to place above the bracket (e.g. "** (p < 0.01)")
    """
    # Horizontal connecting line
    ax.plot([x1, x2], [y, y], color=color, linewidth=lw, zorder=5)
    # Downward ticks at each end
    ax.plot([x1, x1], [y, y - h], color=color, linewidth=lw, zorder=5)
    ax.plot([x2, x2], [y, y - h], color=color, linewidth=lw, zorder=5)
    # Centered label above the bracket
    ax.text((x1 + x2) / 2, y + 0.35, label, ha="center", va="bottom",
            fontsize=fontsize, color=color, zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    x_pos = [0, 1]
    bar_width = 0.55

    bars = ax.bar(
        x_pos, MEANS, width=bar_width, color=BAR_COLORS,
        edgecolor="#2B2B2B", linewidth=1.0, zorder=3,
    )

    # Error bars (standard deviation), capsize=5 as required
    ax.errorbar(
        x_pos, MEANS, yerr=STD_DEVS, fmt="none",
        ecolor="#2B2B2B", elinewidth=1.3, capsize=5, capthick=1.3,
        zorder=4,
    )

    # ── Axis labels, limits, ticks ──────────────────────────────────────────
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax.set_ylim(45, 65)
    ax.set_xlim(-0.65, 1.65)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODELS, fontsize=11.5)
    ax.tick_params(axis="y", labelsize=10.5)

    # ── Chance-level reference line ──────────────────────────────────────────
    ax.axhline(CHANCE_LEVEL, color="#555555", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(
        1.63, CHANCE_LEVEL + 0.35, "Chance level", ha="right", va="bottom",
        fontsize=9.5, color="#555555", style="italic",
    )

    # ── Value labels on top of each bar (just above its error bar cap) ─────
    for xi, mean, std in zip(x_pos, MEANS, STD_DEVS):
        ax.text(xi, mean + std + 0.5, f"{mean:.2f}%", ha="center", va="bottom",
                 fontsize=10, color="#2B2B2B")

    # ── Significance bracket connecting the tops of the two error bars ─────
    top1 = MEANS[0] + STD_DEVS[0]
    top2 = MEANS[1] + STD_DEVS[1]
    bracket_y = max(top1, top2) + 2.2     # sits above both error-bar caps + labels
    tick_h = 0.5
    sig_label = f"{significance_stars(P_VALUE)} (p < 0.01)"
    draw_significance_bracket(ax, x_pos[0], x_pos[1], bracket_y, tick_h, sig_label)

    # Headroom so the bracket + its label fit inside the axes
    ax.set_ylim(45, max(65, bracket_y + 2.0))

    # ── Spines / grid cleanup ───────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.set_title(
        "Fig. 2.  LOSO cross-validation accuracy: SpatioMamba vs. BaselineCNN",
        fontsize=11, pad=14,
    )

    plt.tight_layout()
    fig.savefig("figure2_results.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure2_results.png (300 dpi)")


if __name__ == "__main__":
    main()
