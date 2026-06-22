"""
generate_figure4.py
====================
Generates "Figure 4: Confusion Matrix" (aggregate across all 29 LOSO folds)
for the SpatiotemporalRiemannianMamba IEEE Q1 submission.

*** IMPORTANT — READ BEFORE SUBMITTING TO YOUR PAPER ***
This script ships with PLACEHOLDER cell counts (clearly marked below) so you
can preview layout, colors, and annotation style. A confusion matrix in a
Results section is read as the literal, measured tally of your model's
predictions — not an illustration. Submitting fabricated counts (even ones
that average to your real accuracy) misrepresents what the model actually
did. Replace the placeholder counts with real aggregated predictions before
using this figure in the paper.

To use REAL data:
  Option A — you already have aggregate counts:
      Set USE_REAL_COUNTS = True and fill in TN, FP, FN, TP below.
  Option B — you have raw labels/predictions concatenated across all folds:
      Set USE_RAW_PREDICTIONS = True and point RAW_DATA_CSV_PATH at a CSV
      with columns: true_label,pred_label (one row per test epoch, all
      29 folds concatenated). The script computes the confusion matrix
      from sklearn.metrics.confusion_matrix.

  To actually collect raw predictions during training, your evaluate()
  function in run_loso_trainer_on_modal.py needs to append
  logits.argmax(1) and yb (moved to CPU/numpy) to running lists across
  all folds, then write them out to RAW_DATA_CSV_PATH at the end of the run.

Run locally:
    pip install matplotlib seaborn numpy pandas scikit-learn
    python generate_figure4.py

Output:
    figure4_confusion_matrix.png  (300 dpi, tight bounding box)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Data source switches — exactly ONE of these should be True for real use
# ─────────────────────────────────────────────────────────────────────────────
USE_REAL_COUNTS      = False   # set True if you already have aggregate TN/FP/FN/TP
USE_RAW_PREDICTIONS  = False   # set True if you have per-epoch labels/predictions

RAW_DATA_CSV_PATH = "aggregate_predictions.csv"   # columns: true_label,pred_label

# ── Real aggregate counts (fill these in if USE_REAL_COUNTS = True) ─────────
REAL_TN = None   # True Memory correctly classified
REAL_FP = None   # True Memory misclassified as False Memory
REAL_FN = None   # False Memory misclassified as True Memory
REAL_TP = None   # False Memory correctly classified

# ── Placeholder counts (used only when both switches above are False) ──────
PLACEHOLDER_TN = 3170
PLACEHOLDER_FP = 2600
PLACEHOLDER_FN = 2559
PLACEHOLDER_TP = 3271

CLASS_LABELS = ["True memory", "False memory"]   # index 0 = negative, 1 = positive

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"


def get_confusion_matrix():
    """
    Returns (cm, data_note) where cm is a 2x2 numpy array
    [[TN, FP], [FN, TP]] and data_note is None for real data or a
    warning string for placeholder data.
    """
    if USE_REAL_COUNTS:
        for name, val in [("REAL_TN", REAL_TN), ("REAL_FP", REAL_FP),
                           ("REAL_FN", REAL_FN), ("REAL_TP", REAL_TP)]:
            if val is None:
                raise ValueError(
                    f"USE_REAL_COUNTS is True but {name} is not set. "
                    f"Fill in all four real counts before running."
                )
        cm = np.array([[REAL_TN, REAL_FP], [REAL_FN, REAL_TP]])
        return cm, None

    if USE_RAW_PREDICTIONS:
        import pandas as pd
        from sklearn.metrics import confusion_matrix

        df = pd.read_csv(RAW_DATA_CSV_PATH)
        required = {"true_label", "pred_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        cm = confusion_matrix(df["true_label"], df["pred_label"], labels=[0, 1])
        return cm, None

    # Fallback: placeholder data
    cm = np.array([[PLACEHOLDER_TN, PLACEHOLDER_FP], [PLACEHOLDER_FN, PLACEHOLDER_TP]])
    note = ("PLACEHOLDER COUNTS — not measured data. "
            "Replace with real aggregated predictions before submission.")
    return cm, note


def main():
    cm, data_note = get_confusion_matrix()
    total = cm.sum()
    cm_pct = cm / total * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        cbar=True, cbar_kws={"label": "Count"},
        linewidths=0.6, linecolor="white",
        annot_kws={"fontsize": 13, "fontweight": "normal"},
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        square=True, ax=ax,
        vmin=0,
    )

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.tick_params(axis="both", labelsize=10.5, length=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    overall_acc = (cm[0, 0] + cm[1, 1]) / total * 100
    ax.set_title(
        f"Fig. 4.  Aggregate confusion matrix across 29 LOSO folds\n"
        f"(N = {total:,} test epochs, overall accuracy = {overall_acc:.2f}%)",
        fontsize=10.5, pad=14,
    )

    if data_note:
        fig.text(0.5, -0.06, data_note, ha="center", va="top",
                  fontsize=8.5, color="#B00000", style="italic", wrap=True)

    plt.tight_layout()
    fig.savefig("figure4_confusion_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure4_confusion_matrix.png (300 dpi)")
    print(f"\nConfusion matrix (TN, FP, FN, TP): "
          f"{cm[0,0]}, {cm[0,1]}, {cm[1,0]}, {cm[1,1]}")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    if data_note:
        print("\nNOTE:", data_note)


if __name__ == "__main__":
    main()