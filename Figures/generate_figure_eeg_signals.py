# """
# generate_figure_eeg_signals.py
# ================================
# Generates "Figure: Raw vs. Preprocessed EEG Signals" — an illustrative
# schematic comparing a simulated raw EEG epoch against its preprocessed
# counterpart (bandpass filtering + baseline correction), for the
# SpatiotemporalRiemannianMamba IEEE Q1 submission.

# NOTE ON FIGURE TYPE: unlike a learning curve, confusion matrix, or ROC
# curve, this figure is a SCHEMATIC/EXPLANATORY illustration of what the
# preprocessing pipeline does to a representative epoch — it is not a claim
# about any single specific measured trial's exact values. This is a
# standard and accepted figure type in EEG/BCI papers. The caption below
# explicitly states the signals are a representative simulated example;
# keep that framing in your paper's figure caption so it isn't mistaken for
# a specific subject's raw recording.

# Run locally:
#     pip install matplotlib numpy
#     python generate_figure_eeg_signals.py

# Output:
#     figure_eeg_signals.png  (300 dpi, tight bounding box)
# """

# import numpy as np
# import matplotlib.pyplot as plt


# plt.style.use("seaborn-v0_8-whitegrid")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
# plt.rcParams["mathtext.fontset"] = "stix"

# FS = 250            # sampling rate, Hz
# DURATION = 2.0       # seconds
# N_POINTS = int(FS * DURATION)  # 500

# CHANNEL_NAMES = ["Fz", "Cz", "Pz"]
# CHANNEL_COLORS = ["#1B3A5C", "#2E6E72", "#7A7A7A"]   # dark blue, teal, gray
# OFFSET_STEP = 6.0     # vertical spacing between stacked channels (uV)

# SEED = 7


# def simulate_clean_signal(t, rng, phase_offset=0.0, amp=1.6):
#     """Underlying 'true' neural-like oscillatory signal (the eventual
#     preprocessed output), built from a few EEG-plausible frequency bands."""
#     alpha = 0.9 * np.sin(2 * np.pi * 10 * t + phase_offset)
#     beta  = 0.35 * np.sin(2 * np.pi * 20 * t + phase_offset * 1.4 + 0.5)
#     theta = 0.5 * np.sin(2 * np.pi * 5 * t + phase_offset * 0.6)
#     jitter = rng.normal(0, 0.06, size=t.shape)
#     return amp * (alpha + beta + theta) + jitter


# def add_raw_artifacts(clean, t, rng):
#     """Corrupt a clean signal with baseline drift + high-frequency noise
#     to simulate an unprocessed raw EEG channel."""
#     drift = 1.8 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, np.pi)) \
#             + 0.9 * rng.uniform(-1, 1) * t / DURATION
#     line_noise = 0.5 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, np.pi))
#     emg_noise = rng.normal(0, 0.45, size=t.shape)
#     burst_idx = rng.integers(0, len(t) - 25)
#     burst = np.zeros_like(t)
#     burst[burst_idx:burst_idx + 25] += rng.normal(0, 1.4, size=25)
#     return clean + drift + line_noise + emg_noise + burst


# def main():
#     rng = np.random.default_rng(SEED)
#     t = np.linspace(0, DURATION, N_POINTS, endpoint=False)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     offsets = [OFFSET_STEP * (len(CHANNEL_NAMES) - 1 - i) for i in range(len(CHANNEL_NAMES))]

#     clean_signals = []
#     raw_signals = []
#     for i, name in enumerate(CHANNEL_NAMES):
#         clean = simulate_clean_signal(t, rng, phase_offset=i * 0.8, amp=1.6 - 0.15 * i)
#         raw = add_raw_artifacts(clean, t, rng)
#         clean_signals.append(clean)
#         raw_signals.append(raw)

#     # ── Left panel: Raw EEG ──────────────────────────────────────────────
#     ax = axes[0]
#     for i, name in enumerate(CHANNEL_NAMES):
#         ax.plot(t, raw_signals[i] + offsets[i], color=CHANNEL_COLORS[i],
#                 linewidth=0.9, label=name, zorder=3)
#     ax.set_title("(a) Raw EEG signals", fontsize=12, pad=10)
#     ax.set_xlabel("Time (s)", fontsize=11.5)
#     ax.set_ylabel(r"Amplitude ($\mu$V)", fontsize=11.5)
#     ax.set_xlim(0, DURATION)
#     ax.set_yticks(offsets)
#     ax.set_yticklabels(CHANNEL_NAMES, fontsize=11)
#     ax.tick_params(axis="y", length=0)
#     ax.tick_params(axis="x", labelsize=10)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.grid(axis="y", visible=False)
#     ax.set_axisbelow(True)

#     # ── Right panel: Preprocessed EEG ────────────────────────────────────
#     ax = axes[1]
#     for i, name in enumerate(CHANNEL_NAMES):
#         ax.plot(t, clean_signals[i] + offsets[i], color=CHANNEL_COLORS[i],
#                 linewidth=1.1, label=name, zorder=3)
#     ax.set_title("(b) Preprocessed EEG epoch", fontsize=12, pad=10)
#     ax.set_xlabel("Time (s)", fontsize=11.5)
#     ax.set_ylabel(r"Amplitude ($\mu$V)", fontsize=11.5)
#     ax.set_xlim(0, DURATION)
#     ax.set_yticks(offsets)
#     ax.set_yticklabels(CHANNEL_NAMES, fontsize=11)
#     ax.tick_params(axis="y", length=0)
#     ax.tick_params(axis="x", labelsize=10)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.grid(axis="y", visible=False)
#     ax.set_axisbelow(True)

#     # Shared y-limits so both panels are visually comparable
#     raw_stack = np.array([r + o for r, o in zip(raw_signals, offsets)])
#     clean_stack = np.array([c + o for c, o in zip(clean_signals, offsets)])
#     y_min = min(raw_stack.min(), clean_stack.min()) - 2.0
#     y_max = max(raw_stack.max(), clean_stack.max()) + 2.0
#     axes[0].set_ylim(y_min, y_max)
#     axes[1].set_ylim(y_min, y_max)

#     fig.suptitle(
#         "Fig. 1.  Representative simulated EEG epoch before and after preprocessing\n"
#         "(1\u201340 Hz bandpass filter + baseline correction)",
#         fontsize=10.5, y=1.04,
#     )

#     fig.text(
#         0.5, -0.04,
#         "Note: signals are a representative simulated example for illustrative purposes, not a specific measured trial.",
#         ha="center", va="top", fontsize=8.3, color="#555555", style="italic",
#     )

#     plt.tight_layout()
#     fig.savefig("figure_eeg_signals.png", dpi=300, bbox_inches="tight", facecolor="white")
#     print("Saved: figure_eeg_signals.png (300 dpi)")


# if __name__ == "__main__":
#     main()
























"""
generate_figure_eeg_signals.py
================================
Generates "Figure: Raw vs. Preprocessed EEG Signals" — an illustrative
schematic comparing a simulated raw EEG epoch against its preprocessed
counterpart (bandpass filtering + baseline correction), for the
SpatiotemporalRiemannianMamba IEEE Q1 submission.

NOTE ON FIGURE TYPE: unlike a learning curve, confusion matrix, or ROC
curve, this figure is a SCHEMATIC/EXPLANATORY illustration of what the
preprocessing pipeline does to a representative epoch — it is not a claim
about any single specific measured trial's exact values. This is a
standard and accepted figure type in EEG/BCI papers. The caption below
explicitly states the signals are a representative simulated example;
keep that framing in your paper's figure caption so it isn't mistaken for
a specific subject's raw recording.

Run locally:
    pip install matplotlib numpy
    python generate_figure_eeg_signals.py

Output:
    figure_eeg_signals.png  (300 dpi, tight bounding box)
"""

import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

FS = 250            # sampling rate, Hz
DURATION = 2.0       # seconds
N_POINTS = int(FS * DURATION)  # 500

CHANNEL_NAMES = ["Fz", "Cz", "Pz"]
CHANNEL_COLORS = ["#1B3A5C", "#2E6E72", "#7A7A7A"]   # dark blue, teal, gray
OFFSET_STEP = 6.0     # vertical spacing between stacked channels (uV)

SEED = 7


def simulate_clean_signal(t, rng, phase_offset=0.0, amp=1.6):
    """Underlying 'true' neural-like oscillatory signal (the eventual
    preprocessed output), built from a few EEG-plausible frequency bands."""
    alpha = 0.9 * np.sin(2 * np.pi * 10 * t + phase_offset)
    beta  = 0.35 * np.sin(2 * np.pi * 20 * t + phase_offset * 1.4 + 0.5)
    theta = 0.5 * np.sin(2 * np.pi * 5 * t + phase_offset * 0.6)
    jitter = rng.normal(0, 0.06, size=t.shape)
    return amp * (alpha + beta + theta) + jitter


def add_raw_artifacts(clean, t, rng):
    """Corrupt a clean signal with baseline drift + high-frequency noise
    to simulate an unprocessed raw EEG channel."""
    drift = 1.8 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, np.pi)) \
            + 0.9 * rng.uniform(-1, 1) * t / DURATION
    line_noise = 0.5 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, np.pi))
    emg_noise = rng.normal(0, 0.45, size=t.shape)
    burst_idx = rng.integers(0, len(t) - 25)
    burst = np.zeros_like(t)
    burst[burst_idx:burst_idx + 25] += rng.normal(0, 1.4, size=25)
    return clean + drift + line_noise + emg_noise + burst


def main():
    rng = np.random.default_rng(SEED)
    t = np.linspace(0, DURATION, N_POINTS, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    offsets = [OFFSET_STEP * (len(CHANNEL_NAMES) - 1 - i) for i in range(len(CHANNEL_NAMES))]

    clean_signals = []
    raw_signals = []
    for i, name in enumerate(CHANNEL_NAMES):
        clean = simulate_clean_signal(t, rng, phase_offset=i * 0.8, amp=1.6 - 0.15 * i)
        raw = add_raw_artifacts(clean, t, rng)
        clean_signals.append(clean)
        raw_signals.append(raw)

    # ── Left panel: Raw EEG ──────────────────────────────────────────────
    ax = axes[0]
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(t, raw_signals[i] + offsets[i], color=CHANNEL_COLORS[i],
                linewidth=0.9, label=name, zorder=3)
    ax.set_title("(a) Raw EEG signals", fontsize=12, pad=10)
    ax.set_xlabel("Time (s)", fontsize=11.5)
    ax.set_ylabel(r"Amplitude ($\mu$V)", fontsize=11.5)
    ax.set_xlim(0, DURATION)
    ax.set_yticks(offsets)
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    # ── Right panel: Preprocessed EEG ────────────────────────────────────
    ax = axes[1]
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(t, clean_signals[i] + offsets[i], color=CHANNEL_COLORS[i],
                linewidth=1.1, label=name, zorder=3)
    ax.set_title("(b) Preprocessed EEG epoch", fontsize=12, pad=10)
    ax.set_xlabel("Time (s)", fontsize=11.5)
    ax.set_ylabel(r"Amplitude ($\mu$V)", fontsize=11.5)
    ax.set_xlim(0, DURATION)
    ax.set_yticks(offsets)
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    # Shared y-limits so both panels are visually comparable
    raw_stack = np.array([r + o for r, o in zip(raw_signals, offsets)])
    clean_stack = np.array([c + o for c, o in zip(clean_signals, offsets)])
    y_min = min(raw_stack.min(), clean_stack.min()) - 2.0
    y_max = max(raw_stack.max(), clean_stack.max()) + 2.0
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    fig.suptitle(
        "Fig. 1.  Representative simulated EEG epoch before and after preprocessing\n"
        "(1\u201340 Hz bandpass filter + baseline correction)",
        fontsize=10.5, y=1.04,
    )

    fig.text(
        0.5, -0.04,
        "Note: signals are a representative simulated example for illustrative purposes, not a specific measured trial.",
        ha="center", va="top", fontsize=8.3, color="#555555", style="italic",
    )

    plt.tight_layout()
    fig.savefig("figure_eeg_signals.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved: figure_eeg_signals.png (300 dpi)")


if __name__ == "__main__":
    main()
