# =============================================================================
# command to run : modal run generate_figure2_subject_rescue.py::main
# command to download : modal volume get eeg-data-vol figures/figure2_subject_wise_rescue.png .
# generate_figure2_subject_rescue.py
# Phase 3.1 — Figure 2: 29-Subject Fold-by-Fold Rescue Comparison
# IEEE Transactions / Nature Neuroscience-style publication figure
#
# v2 fixes over v1 (per Senior Scientific Illustrator editorial audit):
#   1. Rescue annotations ("+XX pp ↑") now carry an explicit white bbox
#      (alpha=0.92, pad=1.5) and zorder=15, so they cleanly mask out the
#      Condition 4 mean reference line (or any gridline) whenever a
#      rescued subject's bar top lands near y=67.79%, instead of the text
#      visually merging into the dash-dot line.
#   2. The legend has been moved completely outside the plot area — a
#      horizontal 2-column legend anchored above the axes
#      (loc='lower center', bbox_to_anchor=(0.5, 1.02)) — so it no longer
#      crowds the bars for sub-01..sub-05 in the upper-left of the plot.
#   3. Y-axis headroom widened to (25, 105) to give the tallest rescue
#      annotations (e.g. near sub-11's ~87% bar) more breathing room below
#      the new top boundary.
#
# Narrative: shows, subject by subject, how PCA Subspace Shrinkage Calibration
# (Condition 4) rescues the "task-negative / collapsed" subjects that fail
# under unsupervised zero-shot Euclidean Alignment (Condition 3).
#
# Design notes:
#   - Subject IDs and per-fold accuracies are read dynamically from the two
#     JSON result files on the Modal volume — nothing about which subjects
#     were "rescued" is hardcoded. Gains are computed as
#     (Condition 4 accuracy − Condition 3 accuracy) per subject, and any
#     subject whose gain exceeds RESCUE_THRESHOLD_PP is flagged and annotated.
#   - Subject-ID field names vary across pipelines, so a small set of common
#     key names is tried (subject_id, subject, test_subject,
#     left_out_subject, sub_id, held_out_subject). If none is present, the
#     script falls back to positional sub-01..sub-N numbering and prints an
#     explicit warning so this assumption is never silent.
#   - Condition 3 and Condition 4 fold records are joined on subject ID
#     (not on list position), so the two files can be in different subject
#     order without corrupting the comparison. Any subject present in one
#     file but not the other is logged and excluded from the plot.
#   - Reference lines (chance level, Condition 4 overall mean) are labeled
#     via the legend rather than inline text, since with 29 grouped-bar
#     pairs there is no guaranteed empty whitespace region for inline labels.
#
# Usage: modal run generate_figure2_subject_rescue.py::main
#        modal run generate_figure2_subject_rescue.py::inspect   (sanity-check inputs)
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-figure2-subject-rescue")
volume = modal.Volume.from_name("eeg-data-vol")

VOLUME_PATH       = "/data"
CONDITION3_JSON   = "/data/results_condition3_ea_zeroshot.json"
CONDITION4_JSON   = "/data/results_condition4_subspace_calibrated.json"
FIGURES_DIR       = "/data/figures"
OUTPUT_PNG        = f"{FIGURES_DIR}/figure2_subject_wise_rescue.png"
OUTPUT_PDF        = f"{FIGURES_DIR}/figure2_subject_wise_rescue.pdf"

CHANCE_LEVEL         = 0.50
COND4_OVERALL_MEAN   = 0.6779   # Condition 4 headline LOSO mean, 29 subjects

# A subject counts as "rescued" if Condition 4 beats Condition 3 by more
# than this many percentage points. Purely a labeling threshold for the
# annotation layer — does not affect which subjects are plotted.
RESCUE_THRESHOLD_PP  = 15.0

# Known accuracy field names in each results file (established in the
# Condition 3 / Condition 4 training scripts).
C3_ACC_KEY = "test_accuracy"
C4_ACC_KEY = "post_calibration_acc"

# Candidate field names for the per-fold subject identifier — tried in order.
SUBJECT_ID_KEYS = [
    "subject_id", "subject", "test_subject",
    "left_out_subject", "held_out_subject", "sub_id",
]

figure_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "matplotlib==3.8.4",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run generate_figure2_subject_rescue.py::inspect
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
        folds = d.get("fold_results", [])
        print(f"  ✓ fold_results entries: {len(folds)}")
        if folds:
            print(f"  sample fold keys: {list(folds[0].keys())}")
            print(f"  sample fold record: {folds[0]}")
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
def generate_figure2():

    import json
    import os
    import re
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # =========================================================================
    # SECTION 3: LOAD + JOIN PER-SUBJECT FOLD DATA FROM VOLUME
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

    c3_folds = c3.get("fold_results", [])
    c4_folds = c4.get("fold_results", [])

    if len(c3_folds) == 0 or len(c4_folds) == 0:
        raise ValueError(
            "One or both JSON files have no 'fold_results' entries — "
            "cannot build the subject-wise comparison without them."
        )

    def normalize_label(raw_value):
        """Turn '9', 'sub-09', 'subject_9', etc. into a canonical 'sub-09'."""
        s = str(raw_value)
        m = re.search(r"(\d+)", s)
        if m:
            return f"sub-{int(m.group(1)):02d}"
        return s

    def extract_subject_label(record, fallback_index, used_fallback_flag):
        for key in SUBJECT_ID_KEYS:
            if key in record and record[key] is not None:
                return normalize_label(record[key]), False
        # No recognized subject-ID field — fall back to positional numbering.
        return f"sub-{fallback_index + 1:02d}", True

    def numeric_key(label):
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) if m else 0

    def build_subject_map(fold_records, acc_key, condition_name):
        subj_map = {}
        any_fallback = False
        for i, rec in enumerate(fold_records):
            if acc_key not in rec:
                raise KeyError(
                    f"Expected accuracy field '{acc_key}' missing from a "
                    f"{condition_name} fold record: {rec}"
                )
            label, used_fallback = extract_subject_label(rec, i, any_fallback)
            any_fallback = any_fallback or used_fallback
            subj_map[label] = float(rec[acc_key])
        if any_fallback:
            print(
                f"  ⚠️  {condition_name}: no recognized subject-ID field found "
                f"in fold_results (checked {SUBJECT_ID_KEYS}); fell back to "
                f"positional sub-01..sub-N numbering. Verify this matches the "
                f"actual LOSO subject order before trusting subject labels."
            )
        return subj_map

    c3_map = build_subject_map(c3_folds, C3_ACC_KEY, "Condition 3")
    c4_map = build_subject_map(c4_folds, C4_ACC_KEY, "Condition 4")

    common_subjects = sorted(set(c3_map) & set(c4_map), key=numeric_key)
    only_in_c4 = sorted(set(c4_map) - set(c3_map), key=numeric_key)
    only_in_c3 = sorted(set(c3_map) - set(c4_map), key=numeric_key)

    if only_in_c3 or only_in_c4:
        print(
            f"  ⚠️  Subject-ID mismatch between conditions — "
            f"only in Condition 3: {only_in_c3}; only in Condition 4: {only_in_c4}. "
            f"These subjects are excluded from the comparison plot."
        )

    if len(common_subjects) == 0:
        raise ValueError(
            "No overlapping subject IDs between Condition 3 and Condition 4 "
            "fold results — cannot build a paired comparison."
        )

    c3_vals = np.array([c3_map[s] for s in common_subjects]) * 100.0
    c4_vals = np.array([c4_map[s] for s in common_subjects]) * 100.0
    gains   = c4_vals - c3_vals

    rescued_idx = [i for i, g in enumerate(gains) if g > RESCUE_THRESHOLD_PP]

    print("\n" + "="*70)
    print("  FIGURE 2 — SUBJECT-WISE RESCUE SUMMARY")
    print("="*70)
    print(f"  Subjects plotted (paired across C3 & C4)    : {len(common_subjects)}")
    print(f"  Condition 3 mean (recomputed from folds)    : {c3_vals.mean():.2f}%")
    print(f"  Condition 4 mean (recomputed from folds)    : {c4_vals.mean():.2f}%")
    print(f"  Mean per-subject gain (C4 − C3)              : {gains.mean():+.2f} pp")
    print(f"  Rescued subjects (gain > {RESCUE_THRESHOLD_PP:.0f} pp)          : {len(rescued_idx)}")
    for i in rescued_idx:
        print(f"    {common_subjects[i]:<8} : {c3_vals[i]:6.2f}% -> {c4_vals[i]:6.2f}%   ({gains[i]:+.2f} pp)")
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
        "pdf.fonttype"     : 42,
        "ps.fonttype"      : 42,
    })

    n = len(common_subjects)
    x = np.arange(n)
    width = 0.40

    fig, ax = plt.subplots(figsize=(18, 7), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Grid behind everything ----
    ax.yaxis.grid(True, color="#E4E4E7", alpha=0.25, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ---- Grouped bars ----
    bars3 = ax.bar(
        x - width / 2, c3_vals, width=width,
        color="#64748B", edgecolor="#3F3F46", linewidth=0.5,
        alpha=0.95, zorder=2,
        label="Condition 3: Zero-Shot LOSO + EA",
    )
    bars4 = ax.bar(
        x + width / 2, c4_vals, width=width,
        color="#059669", edgecolor="#065F46", linewidth=0.5,
        alpha=0.95, zorder=2,
        label="Condition 4: PCA Subspace Shrinkage Calibration",
    )

    # ---- Reference lines (labeled via legend — see rationale in header) ----
    ax.axhline(
        y=CHANCE_LEVEL * 100, color="#EF4444", alpha=0.7,
        linestyle="--", linewidth=1.1, zorder=1,
        label="Theoretical Chance Level (50.0%)",
    )
    ax.axhline(
        y=COND4_OVERALL_MEAN * 100, color="#0D9488", alpha=0.8,
        linestyle="-.", linewidth=1.3, zorder=1,
        label=f"Condition 4 Overall Mean ({COND4_OVERALL_MEAN*100:.2f}%)",
    )

    # ---- Rescue annotations: arrow + delta label, combined into one text
    #      block with an explicit white mask so it stays crisp even when it
    #      lands directly on top of the Condition 4 mean line or a gridline
    #      (zorder=15 puts it above every plot element, including the
    #      reference lines at zorder=1). ----
    for i in rescued_idx:
        bar_top_x = x[i] + width / 2
        ax.annotate(
            f"+{gains[i]:.0f} pp ↑",
            xy=(bar_top_x, c4_vals[i]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color="#047857", fontweight="bold", zorder=15,
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=1.5),
        )

    # ---- Axes cosmetics ----
    ax.set_xticks(x)
    ax.set_xticklabels(common_subjects, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.7, n - 1 + 0.7)
    ax.set_ylim(25, 105)

    ax.set_ylabel("LOSO Test Accuracy (%)", fontsize=12.5, fontweight="bold", labelpad=10)
    ax.set_xlabel("Held-Out Subject (Leave-One-Subject-Out)", fontsize=11.5, labelpad=10)
    ax.set_title(
        "Subject-Wise Rescue: PCA Subspace Shrinkage Calibration vs. Zero-Shot Alignment",
        fontsize=14, fontweight="bold", pad=52, loc="center",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["left"].set_color("#3F3F46")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#3F3F46")

    ax.tick_params(axis="both", which="major", labelsize=9.5, length=3.5)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
        fontsize=9, frameon=True, facecolor="white",
        edgecolor="#E2E8F0", framealpha=0.95,
    )

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
    print("  FIGURE 2 EXPORT CONFIRMATION")
    print("="*70)
    print(f"  PNG (600 DPI) : {OUTPUT_PNG}  "
          f"{'✓ written (' + str(os.path.getsize(OUTPUT_PNG)) + ' bytes)' if png_ok else '✗ MISSING'}")
    print(f"  PDF (vector)  : {OUTPUT_PDF}  "
          f"{'✓ written (' + str(os.path.getsize(OUTPUT_PDF)) + ' bytes)' if pdf_ok else '✗ MISSING'}")
    print(f"  Subjects plotted : {n}")
    print(f"  Rescued subjects : {len(rescued_idx)}")
    print("  ✓ volume.commit() complete — figures are now durably available")
    print("    on eeg-data-vol under /data/figures/")
    print("="*70 + "\n")

    if not (png_ok and pdf_ok):
        raise RuntimeError("Figure export verification failed — see log above.")

    return {
        "n_subjects"        : n,
        "condition3_mean"   : float(c3_vals.mean() / 100.0),
        "condition4_mean"   : float(c4_vals.mean() / 100.0),
        "mean_gain_pp"      : float(gains.mean()),
        "rescued_subjects"  : [common_subjects[i] for i in rescued_idx],
        "rescued_count"     : len(rescued_idx),
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
    print("  Phase 3.1 — Figure 2: Subject-Wise Rescue Comparison")
    print("  29-Subject Fold-by-Fold Grouped Bar Chart, Nature/IEEE Style")
    print("="*70 + "\n")

    results = generate_figure2.remote()

    print("\n" + "="*70)
    print("  DONE")
    print("="*70)
    for key, val in results.items():
        print(f"  {key:<20} : {val}")
    print("="*70)
    print(
        "\n  Figure 2 generated and committed to eeg-data-vol.\n"
        "  Retrieve with: modal volume get eeg-data-vol figures/figure2_subject_wise_rescue.png .\n"
        "             or: modal volume get eeg-data-vol figures/figure2_subject_wise_rescue.pdf .\n"
    )