"""
generate_figure6.py
====================
Generates "Figure 6: ROC Curve" for the SpatiotemporalRiemannianMamba
IEEE Q1 submission.

*** IMPORTANT — READ BEFORE SUBMITTING TO YOUR PAPER ***
This script ships with a SIMULATED ROC curve (clearly marked below) built
by sampling a smooth parametric curve that hits a target AUC. An ROC curve
is read by reviewers as direct evidence that you swept classification
thresholds over real predicted probabilities on held-out data — it is one
of the most scrutinized empirical artifacts in a classification paper.
A simulated curve shaped to "look like" a given AUC is not a stylized
version of that evidence; it is a different, fabricated claim. Do not
submit the placeholder curve generated here as a real result.

To use REAL data:
  You need, for each model, the concatenated arrays across all 29 LOSO
  folds of:
    y_true  — ground-truth binary labels (0 = True memory, 1 = False memory)
    y_score — predicted probability of the positive class (label 1),
              i.e. softmax(logits, dim=1)[:, 1] from your forward pass

  Save these to CSVs (one row per test epoch, all folds concatenated):
    columns: true_label, pred_prob

  Set USE_REAL_DATA = True and point the two CSV paths below at your files.
  The script will then compute the ROC curve and AUC directly via
  sklearn.metrics.roc_curve / roc_auc_score — no simulation involved.

  Your evaluate() function in run_loso_trainer_on_modal.py currently only
  accumulates loss and accuracy. To get real predicted probabilities, add
  inside the evaluation loop:
      probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
  and append probs + yb.cpu().numpy() to running lists across all folds,
  then save them to CSV at the end of the run.

Run locally:
    pip install matplotlib seaborn numpy pandas scikit-learn
    python generate_figure6.py

Output:
    figure6_roc_curve.png  (300 dpi, tight bounding box)
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Data source switch
# ─────────────────────────────────────────────────────────────────────────────
USE_REAL_DATA = False

REAL_DATA_CSV_SPATIOMAMBA = "spatiomamba_predictions.csv"   # columns: true_label, pred_prob
REAL_DATA_CSV_BASELINE    = "baselinecnn_predictions.csv"   # columns: true_label, pred_prob

TARGET_AUC_SPATIOMAMBA = 0.59
TARGET_AUC_BASELINE    = 0.53

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

COLOR_SPATIOMAMBA = "#2E6E72"   # teal / dark cyan, matches the established theme
COLOR_BASELINE    = "#8C8C8C"   # neutral gray
COLOR_CHANCE      = "#1A1A1A"


def simulate_roc_curve(target_auc, n_points=200, seed=0):
    """
    Build a SYNTHETIC, smooth ROC curve using a single-parameter family
        TPR = FPR^gamma   (0 < gamma <= 1)
    whose analytic AUC is 1 / (gamma + 1). We solve for gamma that gives
    the requested target_auc, then sample it on a dense FPR grid.
    This is for layout preview only — see module docstring. It does NOT
    correspond to any model's real predictions.
    """
    gamma = max(1e-3, (1.0 / target_auc) - 1.0)
    fpr = np.linspace(0, 1, n_points)
    tpr = np.power(fpr, gamma)
    tpr[0], tpr[-1] = 0.0, 1.0  # anchor endpoints exactly
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    achieved_auc = trapezoid_fn(tpr, fpr)
    return fpr, tpr, achieved_auc


def load_real_roc(csv_path):
    import pandas as pd
    from sklearn.metrics import roc_curve, roc_auc_score

    df = pd.read_csv(csv_path)
    required = {"true_label", "pred_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} is missing required columns: {missing}")

    fpr, tpr, _ = roc_curve(df["true_label"], df["pred_prob"])
    auc = roc_auc_score(df["true_label"], df["pred_prob"])
    return fpr, tpr, auc


def main():
    if USE_REAL_DATA:
        fpr_mamba, tpr_mamba, auc_mamba = load_real_roc(REAL_DATA_CSV_SPATIOMAMBA)
        fpr_cnn, tpr_cnn, auc_cnn = load_real_roc(REAL_DATA_CSV_BASELINE)
        data_note = None
    else:
        fpr_mamba, tpr_mamba, auc_mamba = simulate_roc_curve(TARGET_AUC_SPATIOMAMBA, seed=1)
        fpr_cnn, tpr_cnn, auc_cnn = simulate_roc_curve(TARGET_AUC_BASELINE, seed=2)
        data_note = ("SIMULATED CURVES — not measured ROC data. "
                     "Replace with real predicted probabilities before submission.")

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    ax.plot(fpr_mamba, tpr_mamba, color=COLOR_SPATIOMAMBA, linewidth=2.0,
            label=f"SpatioMamba (AUC = {auc_mamba:.2f})", zorder=3)
    ax.plot(fpr_cnn, tpr_cnn, color=COLOR_BASELINE, linewidth=2.0,
            label=f"BaselineCNN (AUC = {auc_cnn:.2f})", zorder=3)

    # Chance line
    ax.plot([0, 1], [0, 1], color=COLOR_CHANCE, linestyle="--", linewidth=1.2,
            label="Chance (AUC = 0.50)", zorder=2)

    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=10.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)
    ax.set_title("Fig. 6.  ROC curve: SpatioMamba vs. BaselineCNN", fontsize=11, pad=12)

    if data_note:
        fig.text(0.5, -0.04, data_note, ha="center", va="top",
                  fontsize=8.5, color="#B00000", style="italic")

    plt.tight_layout()
    fig.savefig("figure6_roc_curve.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure6_roc_curve.png (300 dpi)")
    print(f"\nSpatioMamba achieved AUC : {auc_mamba:.4f}")
    print(f"BaselineCNN achieved AUC : {auc_cnn:.4f}")
    if data_note:
        print("\nNOTE:", data_note)


if __name__ == "__main__":
    main()