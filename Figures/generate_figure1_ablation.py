# =============================================================================
# command to run : modal run generate_figure1_ablation.py::main
# command to download : modal volume get eeg-data-vol figures/figure1_grand_ablation.png .
# generate_figure1_ablation.py
# Phase 3.1 — Figure 1: Grand Ablation Bar Chart (v4, chance-label relocated)
# IEEE Transactions / Nature Neuroscience-style publication figure
#
# v4 fixes over v3 (per Senior Scientific Illustrator sign-off):
#   - The ONLY change from v3: the red "Theoretical Chance Level" label was
#     sitting at x≈2.5 (between Condition 3 and Condition 4), creating a
#     white-sticker collision effect across bars/dots. It has been moved to
#     the empty margin space on the far LEFT, before Condition 1.
#   - Horizontal xlim expanded slightly (-0.6, 3.8) to guarantee open
#     whitespace on both flanks for label placement.
#   - All other geometry, brackets, y-limits, and data are unchanged from v3.
#
# v3 fixes over v2 (all text-collision issues from the reviewed v2 render):
#   1. Vertical headroom expanded to ax.set_ylim(0, 101) — the previous 0-85
#      ceiling left no room for the bracket + subtitle above the tallest
#      Condition 4 scatter dots (~79%).
#   2. Percentage labels are no longer placed at a static offset above the
#      bar. Each label's Y position is computed dynamically as
#      peak_y = max(bar_mean + bar_std, max(subject_dot_y)) + 2.2, so a
#      label can never collide with its own error cap or an outlier dot.
#   3. The significance bracket connecting Condition 3 and Condition 4 is
#      now pinned at exact, fixed data coordinates (horizontal bar at
#      Y=87.0, label at Y=88.8) rather than being derived relative to bar
#      heights, so it can never drift down into the scatter cloud.
#   4. Title uses pad=32 to sit fully above the axes box; the italic
#      subtitle is placed at a fixed data-coordinate Y=93.5, which leaves
#      >4 percentage points of clearance above the bracket label at Y=88.8
#      and is unambiguously below the title (which lives outside the axes
#      entirely).
#   5. The chance-level label has been moved to the far-left, above the
#      dashed line, left-aligned — it no longer cuts across the Condition
#      4 bar and its scatter dots.
#
# Usage: modal run generate_figure1_ablation.py::main
#        modal run generate_figure1_ablation.py::inspect   (sanity-check inputs)
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

# Hardcoded, established baseline means (no fold-level data available for these).
CONDITION1_ACC = 0.5214   # Baseline CNN / EEGNet, zero-shot LOSO
CONDITION2_ACC = 0.5553   # SpatioMamba, zero-shot LOSO, no EA

CHANCE_LEVEL = 0.50

# Fixed layout constants (data coordinates, in percentage-point units).
Y_AXIS_MAX          = 101.0
LABEL_OFFSET        = 2.2     # % points above each bar's true data peak
BRACKET_Y           = 87.0    # horizontal bar of the significance bracket
BRACKET_LABEL_Y     = 88.8    # "+X.XX pp Absolute Gain" text
BRACKET_TICK_MARGIN = 4.0     # gap between bracket tick foot and bar's % label
SUBTITLE_Y          = 93.5    # italic subtitle, data coordinates

# Chance-level label placement (v4: relocated to far-left empty margin).
CHANCE_LABEL_X      = -0.55   # far-left, in the expanded xlim margin
CHANCE_LABEL_Y      = 51.0    # just above the dashed chance line

figure_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "matplotlib==3.8.4",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run generate_figure1_ablation.py::inspect
# =============================================================================

@app.function(image=figure_image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_inputs():
    import json, os
    for path in (CONDITION3_JSON, CONDITION4_JSON):
        print(f"\n{path}")
        if not os.path.exists(path):
            print("  ✗ NOT FOUND")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        n_folds = len(d.get("fold_results", []))
        print(f"  ✓ mean_accuracy={d.get('mean_accuracy')}  std_accuracy={d.get('std_accuracy')}")
        print(f"  ✓ fold_results entries: {n_folds}")
        if n_folds:
            sample = d["fold_results"][0]
            print(f"  sample fold keys: {list(sample.keys())}")
    return "ok"


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    inspect_inputs.remote()


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

    # =========================================================================
    # SECTION 3: LOAD EXACT METRICS + PER-SUBJECT FOLD DATA FROM VOLUME
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

    # Per-subject fold accuracies (the scientific overlay).
    c3_fold_accs = [float(rec["test_accuracy"]) for rec in c3.get("fold_results", [])]
    c4_fold_accs = [float(rec["post_calibration_acc"]) for rec in c4.get("fold_results", [])]

    if len(c3_fold_accs) == 0 or len(c4_fold_accs) == 0:
        raise ValueError(
            "One or both JSON files have no 'fold_results' entries — "
            "cannot overlay per-subject scatter points without them."
        )

    means = [CONDITION1_ACC, CONDITION2_ACC, c3_mean, c4_mean]
    stds  = [0.0,             0.0,             c3_std,  c4_std]
    fold_data = [None, None, c3_fold_accs, c4_fold_accs]

    delta_c3_to_c4 = (c4_mean - c3_mean) * 100

    print("\n" + "="*70)
    print("  FIGURE 1 — VALUES BEING PLOTTED")
    print("="*70)
    print(f"  Condition 1 (hardcoded)      : {CONDITION1_ACC*100:.2f}%  (no fold data)")
    print(f"  Condition 2 (hardcoded)      : {CONDITION2_ACC*100:.2f}%  (no fold data)")
    print(f"  Condition 3 (from JSON)      : {c3_mean*100:.2f}% ± {c3_std*100:.2f}%  "
          f"across {len(c3_fold_accs)} subjects  [{CONDITION3_JSON}]")
    print(f"  Condition 4 (from JSON)      : {c4_mean*100:.2f}% ± {c4_std*100:.2f}%  "
          f"across {len(c4_fold_accs)} subjects  [{CONDITION4_JSON}]")
    print(f"  Absolute gain (C4 − C3)      : +{delta_c3_to_c4:.2f} pp")
    print("="*70 + "\n")

    # =========================================================================
    # SECTION 4: EDITORIAL TYPOGRAPHY + STYLE
    # =========================================================================
    plt.rcParams.update({
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size"        : 11.5,
        "axes.linewidth"   : 0.9,
        "axes.edgecolor"   : "#3F3F46",
        "xtick.color"      : "#27272A",
        "ytick.color"      : "#27272A",
        "text.color"       : "#18181B",
        "pdf.fonttype"     : 42,   # embed real (non-rasterized) fonts in vector PDF
        "ps.fonttype"      : 42,
    })

    labels = [
        "Condition 1\nBaseline CNN\n(EEGNet)",
        "Condition 2\nSpatioMamba\n(Zero-Shot)",
        "Condition 3\nSpatioMamba + EA\n(Zero-Shot)",
        "Condition 4\nSpatioMamba + EA\n+ Subspace Calibration",
    ]

    # Sophisticated modern scientific palette.
    colors = [
        "#94A3B8",   # Condition 1 — muted slate grey
        "#64748B",   # Condition 2 — deep slate
        "#1E40AF",   # Condition 3 — classic navy / indigo
        "#059669",   # Condition 4 — rich emerald (editorial winner accent)
    ]

    fig, ax = plt.subplots(figsize=(9.2, 6.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_pos = np.arange(len(labels))
    bar_width = 0.55

    # ---- Light y-grid, drawn first and pushed behind everything ----
    ax.yaxis.grid(True, color="#E4E4E7", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ---- Bars, subtle top-edge shading via slightly darker edgecolor ----
    bars = ax.bar(
        x_pos, [m * 100 for m in means],
        width=bar_width,
        color=colors,
        edgecolor="none",
        linewidth=0,
        zorder=2,
        alpha=0.95,
    )
    for bar in bars:
        bar.set_linewidth(0.6)
        bar.set_edgecolor("#1F1F23")

    # ---- Sleek, thin charcoal error bars (Condition 3 & 4 only carry std) ----
    err_vals = [s * 100 for s in stds]
    ax.errorbar(
        x_pos, [m * 100 for m in means],
        yerr=err_vals,
        fmt="none",
        ecolor="#3F3F46",
        elinewidth=1.2,
        capsize=4,
        capthick=1.1,
        zorder=4,
    )

    # ---- Jittered per-subject scatter overlay (Condition 3 & 4 only) ----
    rng = np.random.default_rng(42)
    scatter_half_width = bar_width * 0.32
    for i, accs in enumerate(fold_data):
        if accs is None:
            continue
        jitter = rng.uniform(-scatter_half_width, scatter_half_width, size=len(accs))
        ax.scatter(
            x_pos[i] + jitter, [a * 100 for a in accs],
            s=22, facecolor="white", edgecolor="#27272A",
            linewidth=0.5, alpha=0.45, zorder=3,
        )

    # =========================================================================
    # DYNAMIC, COLLISION-FREE PEAK CALCULATION (per bar)
    #   peak_y = max(bar_mean + bar_std, max(subject_dot_y))
    #   Percentage label is placed at peak_y + LABEL_OFFSET, zorder=10, so it
    #   is guaranteed to clear the error cap AND the tallest scatter dot.
    # =========================================================================
    peak_y = []
    for i in range(len(means)):
        bar_top = means[i] * 100 + stds[i] * 100
        if fold_data[i] is not None:
            dot_top = max(fold_data[i]) * 100
            peak_y.append(max(bar_top, dot_top))
        else:
            peak_y.append(bar_top)

    for i, m in enumerate(means):
        label_y = peak_y[i] + LABEL_OFFSET
        ax.text(
            x_pos[i], label_y, f"{m*100:.2f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color="#18181B", zorder=10,
        )

    # ---- Expanded horizontal margins so the chance-level label has
    #      guaranteed open whitespace on the far left, clear of any bar
    #      or scatter dot (per illustrator sign-off, v4). ----
    ax.set_xlim(-0.6, 3.8)

    # ---- Chance-level reference line — label relocated to the far-LEFT
    #      empty margin (v4), left-aligned, sitting just above the dashed
    #      line with a white bbox as a hard guarantee against any
    #      bar-border or scatter-dot collision. ----
    ax.axhline(
        y=CHANCE_LEVEL * 100,
        color="#EF4444", alpha=0.6,
        linestyle="--", linewidth=1.1,
        zorder=1,
    )
    ax.text(
        CHANCE_LABEL_X, CHANCE_LABEL_Y,
        "Theoretical Chance Level (50.0%)",
        ha="left", va="bottom",
        fontsize=9.5, style="italic", color="#B91C1C", alpha=0.85,
        zorder=10,
        bbox=dict(facecolor="white", alpha=0.95, edgecolor="none", pad=2),
    )

    # =========================================================================
    # SIGNIFICANCE BRACKET — FIXED, ABSOLUTE DATA COORDINATES (with safety net)
    #   Defaults: horizontal bar at Y=87.0, label at Y=88.8, exactly as
    #   specified. If a fold's data ever pushes a bar's peak (mean+std, or
    #   an individual subject dot) high enough that the bracket's vertical
    #   tick would need to rise above the fixed BRACKET_Y, every downstream
    #   anchor (bracket, label, subtitle, y-limit) is shifted upward by the
    #   same amount and a warning is logged — rather than silently drawing
    #   an inverted/overlapping bracket or crashing on an assertion.
    # =========================================================================
    x3, x4 = x_pos[2], x_pos[3]
    foot3 = peak_y[2] + LABEL_OFFSET + BRACKET_TICK_MARGIN
    foot4 = peak_y[3] + LABEL_OFFSET + BRACKET_TICK_MARGIN

    bracket_y       = BRACKET_Y
    bracket_label_y = BRACKET_LABEL_Y
    subtitle_y      = SUBTITLE_Y
    y_axis_max      = Y_AXIS_MAX

    required_bracket_y = max(foot3, foot4) + 0.5
    if required_bracket_y > bracket_y:
        shift = required_bracket_y - bracket_y
        bracket_y       += shift
        bracket_label_y += shift
        subtitle_y      += shift
        y_axis_max       = max(y_axis_max, subtitle_y + 7.0)
        print(
            f"  ⚠️  Data peak exceeded the fixed bracket geometry by {shift:.2f} pp "
            f"— auto-shifted bracket_y→{bracket_y:.1f}, label_y→{bracket_label_y:.1f}, "
            f"subtitle_y→{subtitle_y:.1f}, y_max→{y_axis_max:.1f} to avoid collision."
        )

    ax.plot(
        [x3, x3, x4, x4],
        [foot3, bracket_y, bracket_y, foot4],
        color="#18181B", linewidth=1.0, zorder=6,
    )
    ax.text(
        (x3 + x4) / 2, bracket_label_y,
        f"+{delta_c3_to_c4:.2f} pp Absolute Gain",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#059669", zorder=10,
    )

    # =========================================================================
    # TITLE / SUBTITLE — STRICTLY SEPARATED, NO COLLISION WITH BRACKET
    #   Title: pad=32 keeps it fully outside/above the axes box.
    #   Subtitle: data-coordinate Y (auto-shifted above if needed), always
    #   kept >=4 pp above the bracket label.
    # =========================================================================
    ax.set_title(
        "Grand Ablation: Progressive Gains Across the 4-Condition BCI Pipeline",
        fontsize=14, fontweight="bold", pad=32, loc="center",
    )
    x_center = (x_pos[0] + x_pos[-1]) / 2
    ax.text(
        x_center, subtitle_y,
        "Strict 29-Subject Leave-One-Subject-Out Evaluation — points show individual held-out subjects",
        ha="center", va="bottom",
        fontsize=9.5, color="#52525B", style="italic", zorder=10,
    )

    # ---- Axes cosmetics ----
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("LOSO Test Accuracy (%)", fontsize=12.5, fontweight="bold", labelpad=10)

    ax.set_ylim(0, y_axis_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["left"].set_color("#3F3F46")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#3F3F46")

    ax.tick_params(axis="both", which="major", labelsize=10, length=3.5)

    fig.tight_layout()

    # =========================================================================
    # LAYOUT SELF-CHECK — verify no collisions before export
    # =========================================================================
    assert bracket_label_y > max(foot3, foot4) - 0.01, (
        "Bracket label Y is not above its own tick feet after adjustment — "
        "this should be unreachable; check the shift logic above."
    )
    assert subtitle_y >= bracket_label_y + 4.0, (
        f"Subtitle (Y={subtitle_y:.1f}) does not clear the bracket label "
        f"(Y={bracket_label_y:.1f}) by the required 4 pp margin."
    )
    assert y_axis_max >= subtitle_y + 2.0, (
        "Y-axis ceiling too low to fit the subtitle without clipping."
    )
    print(f"  Layout self-check passed: bracket feet=({foot3:.1f}, {foot4:.1f}), "
          f"bracket_y={bracket_y:.1f}, label_y={bracket_label_y:.1f}, "
          f"subtitle_y={subtitle_y:.1f}, y_max={y_axis_max:.1f}")
    print(f"  Chance-level label anchored at x={CHANCE_LABEL_X}, y={CHANCE_LABEL_Y} "
          f"(far-left margin, xlim={ax.get_xlim()})")

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
    print(f"  Per-subject overlay: {len(c3_fold_accs)} pts (Cond 3), "
          f"{len(c4_fold_accs)} pts (Cond 4)")
    print("  ✓ volume.commit() complete — figures are now durably available")
    print("    on eeg-data-vol under /data/figures/")
    print("="*70 + "\n")

    if not (png_ok and pdf_ok):
        raise RuntimeError("Figure export verification failed — see log above.")

    return {
        "condition1_acc"    : CONDITION1_ACC,
        "condition2_acc"    : CONDITION2_ACC,
        "condition3_mean"   : c3_mean,
        "condition3_std"    : c3_std,
        "condition4_mean"   : c4_mean,
        "condition4_std"    : c4_std,
        "n_subjects_c3"     : len(c3_fold_accs),
        "n_subjects_c4"     : len(c4_fold_accs),
        "absolute_gain_pp"  : delta_c3_to_c4,
        "png_path"          : OUTPUT_PNG,
        "pdf_path"          : OUTPUT_PDF,
        "png_bytes"         : os.path.getsize(OUTPUT_PNG),
        "pdf_bytes"         : os.path.getsize(OUTPUT_PDF),
    }


# =============================================================================
# SECTION 6: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("  Phase 3.1 — Figure 1: Grand Ablation Bar Chart (v4, chance-label relocated)")
    print("  4-Condition Comparison + Per-Subject Overlay, Nature/IEEE Style")
    print("="*70 + "\n")

    results = generate_figure1.remote()

    print("\n" + "="*70)
    print("  DONE")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<20} : {val}")
    print("="*70)
    print(
        "\n  Figure 1 (v4) generated and committed to eeg-data-vol.\n"
        "  Retrieve with: modal volume get eeg-data-vol figures/figure1_grand_ablation.png .\n"
        "             or: modal volume get eeg-data-vol figures/figure1_grand_ablation.pdf .\n"
    )