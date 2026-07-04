# """
# generate_figure3.py
# ====================
# Generates "Figure 3: Training and Validation Learning Curves" for the
# SpatiotemporalRiemannianMamba IEEE Q1 submission.

# *** IMPORTANT — READ BEFORE SUBMITTING TO YOUR PAPER ***
# This script ships with SYNTHETIC, illustrative data (clearly marked below)
# so you can preview the exact layout, colors, and early-stopping annotation
# before wiring in real numbers. A learning-curve figure in a Results section
# is read by reviewers as an empirical record of an actual training run.
# Submitting simulated curves as if they were measured data is a research
# integrity issue, not a styling choice — independent of how plausible the
# numbers look.

# To use REAL data instead of the mock curves:
#   1. In your training loop (see run_loso_trainer_on_modal.py), log
#      tr_loss, tr_acc, te_loss, te_acc for every epoch of one representative
#      fold to a CSV with columns: epoch,train_loss,val_loss,train_acc,val_acc
#   2. Set USE_REAL_DATA = True and REAL_DATA_CSV_PATH below.
#   3. Re-run this script. The mock-data block is skipped entirely.

# Run locally:
#     pip install matplotlib seaborn numpy pandas
#     python generate_figure3.py

# Output:
#     figure3_learning_curves.png  (300 dpi, tight bounding box)
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# # ─────────────────────────────────────────────────────────────────────────────
# # Data source switch
# # ─────────────────────────────────────────────────────────────────────────────
# USE_REAL_DATA = False                       # set True once you have real per-epoch logs
# REAL_DATA_CSV_PATH = "fold_epoch_log.csv"   # columns: epoch,train_loss,val_loss,train_acc,val_acc

# N_EPOCHS = 35
# EARLY_STOP_EPOCH = 28        # epoch index (1-based) where val_loss is minimum

# # ─────────────────────────────────────────────────────────────────────────────
# # Style
# # ─────────────────────────────────────────────────────────────────────────────
# plt.style.use("seaborn-v0_8-whitegrid")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
# plt.rcParams["mathtext.fontset"] = "stix"

# COLOR_TRAIN = "#0072B2"   # teal/blue
# COLOR_VAL   = "#D55E00"   # amber/orange
# COLOR_ES    = "#CC0000"   # red, early-stopping marker


# def generate_mock_curves(n_epochs, es_epoch, seed=42):
#     """
#     Build SYNTHETIC, illustrative learning curves with the requested shape:
#       - train loss: smooth monotonic decrease
#       - val loss: decreases, bottoms out near es_epoch, then flattens/ticks up
#       - train acc: smooth increase toward ~75-80%
#       - val acc: increases then plateaus near ~55-58%
#     This is for layout preview only — see module docstring.
#     """
#     rng = np.random.default_rng(seed)
#     epochs = np.arange(1, n_epochs + 1)

#     # Training loss: smooth exponential decay + tiny noise
#     train_loss = 0.95 * np.exp(-epochs / 9.0) + 0.08 + rng.normal(0, 0.006, n_epochs)
#     train_loss = np.clip(train_loss, 0.05, None)

#     # Validation loss: decreases to a minimum at es_epoch, then flattens/rises slightly
#     val_loss = np.empty(n_epochs)
#     for i, ep in enumerate(epochs):
#         if ep <= es_epoch:
#             val_loss[i] = 0.85 * np.exp(-ep / 11.0) + 0.42
#         else:
#             overshoot = (ep - es_epoch) * 0.006
#             val_loss[i] = (0.85 * np.exp(-es_epoch / 11.0) + 0.42) + overshoot
#     val_loss += rng.normal(0, 0.012, n_epochs)
#     val_loss = np.clip(val_loss, 0.30, None)

#     # Training accuracy: smooth rise toward ~75-80%
#     train_acc = 78.0 - 60.0 * np.exp(-epochs / 8.0) + rng.normal(0, 0.6, n_epochs)
#     train_acc = np.clip(train_acc, 0, 100)

#     # Validation accuracy: rises then plateaus near 55-58%
#     val_acc = np.empty(n_epochs)
#     for i, ep in enumerate(epochs):
#         plateau = 56.5
#         val_acc[i] = plateau - 22.0 * np.exp(-ep / 7.0)
#     val_acc += rng.normal(0, 0.9, n_epochs)
#     val_acc = np.clip(val_acc, 0, 100)

#     return epochs, train_loss, val_loss, train_acc, val_acc


# def load_real_curves(csv_path):
#     import pandas as pd
#     df = pd.read_csv(csv_path)
#     required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"CSV is missing required columns: {missing}")
#     es_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
#     return (
#         df["epoch"].to_numpy(),
#         df["train_loss"].to_numpy(),
#         df["val_loss"].to_numpy(),
#         df["train_acc"].to_numpy(),
#         df["val_acc"].to_numpy(),
#         es_epoch,
#     )


# def annotate_early_stop(ax, es_epoch, xytext_offset=(1.4, 0), y_text_frac=0.5,
#                          label="Early stopping\ncheckpoint"):
#     ax.axvline(es_epoch, color=COLOR_ES, linestyle="--", linewidth=1.3, zorder=4)
#     ylim = ax.get_ylim()
#     y_anchor = ylim[0] + y_text_frac * (ylim[1] - ylim[0])
#     y_text = y_anchor + xytext_offset[1]
#     ax.annotate(
#         label,
#         xy=(es_epoch, y_anchor),
#         xytext=(es_epoch + xytext_offset[0], y_text),
#         fontsize=8.5, color=COLOR_ES, ha="left", va="center",
#         arrowprops=dict(arrowstyle="->", color=COLOR_ES, lw=1.0),
#         zorder=6,
#     )


# def main():
#     if USE_REAL_DATA:
#         epochs, train_loss, val_loss, train_acc, val_acc, es_epoch = load_real_curves(REAL_DATA_CSV_PATH)
#         data_note = None
#     else:
#         epochs, train_loss, val_loss, train_acc, val_acc = generate_mock_curves(N_EPOCHS, EARLY_STOP_EPOCH)
#         es_epoch = EARLY_STOP_EPOCH
#         data_note = ("SYNTHETIC DATA — for layout preview only. "
#                      "Replace with real per-epoch logs before submission.")

#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))

#     # ── Left: Loss ───────────────────────────────────────────────────────
#     ax = axes[0]
#     ax.plot(epochs, train_loss, color=COLOR_TRAIN, linestyle="-", linewidth=1.6, label="Training loss")
#     ax.plot(epochs, val_loss, color=COLOR_VAL, linestyle="--", linewidth=1.6, label="Validation loss")
#     ax.set_xlabel("Epochs", fontsize=11)
#     ax.set_ylabel("Loss", fontsize=11)
#     ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
#     ax.set_title("(a) Loss curves", fontsize=10.5, pad=10)
#     ax.set_ylim(bottom=ax.get_ylim()[0] - 0.05)
#     annotate_early_stop(ax, es_epoch, xytext_offset=(1.6, 0), y_text_frac=0.30)

#     # ── Right: Accuracy ──────────────────────────────────────────────────
#     ax = axes[1]
#     ax.plot(epochs, train_acc, color=COLOR_TRAIN, linestyle="-", linewidth=1.6, label="Training accuracy")
#     ax.plot(epochs, val_acc, color=COLOR_VAL, linestyle="--", linewidth=1.6, label="Validation accuracy")
#     ax.set_xlabel("Epochs", fontsize=11)
#     ax.set_ylabel("Accuracy (%)", fontsize=11)
#     ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
#     ax.set_title("(b) Accuracy curves", fontsize=10.5, pad=10)
#     annotate_early_stop(ax, es_epoch, xytext_offset=(-7.5, 8), y_text_frac=0.78)

#     for a in axes:
#         a.spines["top"].set_visible(False)
#         a.spines["right"].set_visible(False)
#         a.set_axisbelow(True)

#     if data_note:
#         fig.text(0.5, -0.06, data_note, ha="center", va="top",
#                   fontsize=8, color="#B00000", style="italic")

#     plt.tight_layout()
#     fig.savefig("figure3_learning_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
#     print("Saved: figure3_learning_curves.png (300 dpi)")
#     if data_note:
#         print("\nNOTE:", data_note)


# if __name__ == "__main__":
#     main()
















"""
generate_figure3.py
====================
Generates "Figure 3: Training and Validation Learning Curves" for the
SpatiotemporalRiemannianMamba IEEE Q1 submission.

*** IMPORTANT — READ BEFORE SUBMITTING TO YOUR PAPER ***
This script ships with SYNTHETIC, illustrative data (clearly marked below)
so you can preview the exact layout, colors, and early-stopping annotation
before wiring in real numbers. A learning-curve figure in a Results section
is read by reviewers as an empirical record of an actual training run.
Submitting simulated curves as if they were measured data is a research
integrity issue, not a styling choice — independent of how plausible the
numbers look.

To use REAL data instead of the mock curves:
  1. In your training loop (see run_loso_trainer_on_modal.py), log
     tr_loss, tr_acc, te_loss, te_acc for every epoch of one representative
     fold to a CSV with columns: epoch,train_loss,val_loss,train_acc,val_acc
  2. Set USE_REAL_DATA = True and REAL_DATA_CSV_PATH below.
  3. Re-run this script. The mock-data block is skipped entirely.

Run locally:
    pip install matplotlib seaborn numpy pandas
    python generate_figure3.py

Output:
    figure3_learning_curves.png  (300 dpi, tight bounding box)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Data source switch
# ─────────────────────────────────────────────────────────────────────────────
USE_REAL_DATA = False                       # set True once you have real per-epoch logs
REAL_DATA_CSV_PATH = "fold_epoch_log.csv"   # columns: epoch,train_loss,val_loss,train_acc,val_acc

N_EPOCHS = 35
EARLY_STOP_EPOCH = 28        # epoch index (1-based) where val_loss is minimum

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

COLOR_TRAIN = "#0072B2"   # teal/blue
COLOR_VAL   = "#D55E00"   # amber/orange
COLOR_ES    = "#CC0000"   # red, early-stopping marker


def generate_mock_curves(n_epochs, es_epoch, seed=42):
    """
    Build SYNTHETIC, illustrative learning curves with the requested shape:
      - train loss: smooth monotonic decrease
      - val loss: decreases, bottoms out near es_epoch, then flattens/ticks up
      - train acc: smooth increase toward ~75-80%
      - val acc: increases then plateaus near ~55-58%
    This is for layout preview only — see module docstring.
    """
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n_epochs + 1)

    # Training loss: smooth exponential decay + tiny noise
    train_loss = 0.95 * np.exp(-epochs / 9.0) + 0.08 + rng.normal(0, 0.006, n_epochs)
    train_loss = np.clip(train_loss, 0.05, None)

    # Validation loss: decreases to a minimum at es_epoch, then flattens/rises slightly
    val_loss = np.empty(n_epochs)
    for i, ep in enumerate(epochs):
        if ep <= es_epoch:
            val_loss[i] = 0.85 * np.exp(-ep / 11.0) + 0.42
        else:
            overshoot = (ep - es_epoch) * 0.006
            val_loss[i] = (0.85 * np.exp(-es_epoch / 11.0) + 0.42) + overshoot
    val_loss += rng.normal(0, 0.012, n_epochs)
    val_loss = np.clip(val_loss, 0.30, None)

    # Training accuracy: smooth rise toward ~75-80%
    train_acc = 78.0 - 60.0 * np.exp(-epochs / 8.0) + rng.normal(0, 0.6, n_epochs)
    train_acc = np.clip(train_acc, 0, 100)

    # Validation accuracy: rises then plateaus near 55-58%
    val_acc = np.empty(n_epochs)
    for i, ep in enumerate(epochs):
        plateau = 56.5
        val_acc[i] = plateau - 22.0 * np.exp(-ep / 7.0)
    val_acc += rng.normal(0, 0.9, n_epochs)
    val_acc = np.clip(val_acc, 0, 100)

    return epochs, train_loss, val_loss, train_acc, val_acc


def load_real_curves(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    es_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
    return (
        df["epoch"].to_numpy(),
        df["train_loss"].to_numpy(),
        df["val_loss"].to_numpy(),
        df["train_acc"].to_numpy(),
        df["val_acc"].to_numpy(),
        es_epoch,
    )


def annotate_early_stop(ax, es_epoch, xytext_offset=(1.4, 0), y_text_frac=0.5,
                         label="Early stopping\ncheckpoint"):
    ax.axvline(es_epoch, color=COLOR_ES, linestyle="--", linewidth=1.3, zorder=4)
    ylim = ax.get_ylim()
    y_anchor = ylim[0] + y_text_frac * (ylim[1] - ylim[0])
    y_text = y_anchor + xytext_offset[1]
    ax.annotate(
        label,
        xy=(es_epoch, y_anchor),
        xytext=(es_epoch + xytext_offset[0], y_text),
        fontsize=8.5, color=COLOR_ES, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COLOR_ES, lw=1.0),
        zorder=6,
    )


def main():
    if USE_REAL_DATA:
        epochs, train_loss, val_loss, train_acc, val_acc, es_epoch = load_real_curves(REAL_DATA_CSV_PATH)
        data_note = None
    else:
        epochs, train_loss, val_loss, train_acc, val_acc = generate_mock_curves(N_EPOCHS, EARLY_STOP_EPOCH)
        es_epoch = EARLY_STOP_EPOCH
        data_note = ("SYNTHETIC DATA — for layout preview only. "
                     "Replace with real per-epoch logs before submission.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: Loss ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, color=COLOR_TRAIN, linestyle="-", linewidth=1.6, label="Training loss")
    ax.plot(epochs, val_loss, color=COLOR_VAL, linestyle="--", linewidth=1.6, label="Validation loss")
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    ax.set_title("(a) Loss curves", fontsize=10.5, pad=10)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.05)
    annotate_early_stop(ax, es_epoch, xytext_offset=(1.6, 0), y_text_frac=0.30)

    # ── Right: Accuracy ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, train_acc, color=COLOR_TRAIN, linestyle="-", linewidth=1.6, label="Training accuracy")
    ax.plot(epochs, val_acc, color=COLOR_VAL, linestyle="--", linewidth=1.6, label="Validation accuracy")
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    ax.set_title("(b) Accuracy curves", fontsize=10.5, pad=10)
    annotate_early_stop(ax, es_epoch, xytext_offset=(-7.5, 8), y_text_frac=0.78)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_axisbelow(True)

    if data_note:
        fig.text(0.5, -0.06, data_note, ha="center", va="top",
                  fontsize=8, color="#B00000", style="italic")

    plt.tight_layout()
    fig.savefig("figure3_learning_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure3_learning_curves.png (300 dpi)")
    if data_note:
        print("\nNOTE:", data_note)


if __name__ == "__main__":
    main()
