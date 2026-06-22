# step1_2_filter_epoch.py

"""
BCI Research Pipeline — Phase 1, Step 1.2
Task : Bandpass filter → Downsample → Epoch → Verify tensor shape
Input : sub-01 .vhdr file (already downloaded in Step 1.1)
Output: NumPy array shape [n_epochs, n_channels, n_times]
"""

from pathlib import Path
import mne
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
VHDR_PATH   = Path("./data/ds005189/sub-01/eeg/sub-01_task-SearchSupRecFam_eeg.vhdr")
SFREQ_NEW   = 250          # downsample target (Hz)
L_FREQ      = 1.0          # bandpass low  (Hz)
H_FREQ      = 40.0         # bandpass high (Hz)
TMIN        = -0.2         # epoch start (s), pre-stimulus baseline
TMAX        =  0.8         # epoch end   (s), post-stimulus

# Binary label mapping:
#   1x codes → Class 0  (Search/Sup condition responses)
#   2x codes → Class 1  (Rec/Fam condition responses)
EVENT_ID = {
    'Stimulus/ 10': 0,
    'Stimulus/ 11': 0,
    'Stimulus/ 12': 0,
    'Stimulus/ 13': 0,
    'Stimulus/ 20': 1,
    'Stimulus/ 21': 1,
    'Stimulus/ 22': 1,
    'Stimulus/ 23': 1,
}

# ── Step 1 · Load raw data into memory ────────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 1 · Loading raw EEG into memory")
print(f"{'='*60}")

if not VHDR_PATH.exists():
    raise FileNotFoundError(
        f"VHdr file not found: {VHDR_PATH}\n"
        "Make sure Step 1.1 completed successfully and you are running\n"
        "this script from the D:\\EEG\\ directory."
    )

raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose="WARNING")
print(f"  Loaded  : {VHDR_PATH.name}")
print(f"  Channels: {len(raw.ch_names)}  |  sfreq: {raw.info['sfreq']} Hz")

# ── Step 2 · Bandpass filter ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Step 2 · Bandpass filter  [{L_FREQ} – {H_FREQ} Hz]")
print(f"{'='*60}")

raw.filter(l_freq=L_FREQ, h_freq=H_FREQ,
           method="fir", fir_window="hamming",
           verbose="WARNING")
print("  Filter applied.")

# ── Step 3 · Downsample ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Step 3 · Downsample  {int(raw.info['sfreq'])} Hz → {SFREQ_NEW} Hz")
print(f"{'='*60}")

raw.resample(SFREQ_NEW, verbose="WARNING")
print(f"  New sfreq: {raw.info['sfreq']} Hz")

# ── Step 4 · Extract events from annotations ──────────────────────────────────
print(f"\n{'='*60}")
print("  Step 4 · Extracting events from annotations")
print(f"{'='*60}")

# Build a reverse lookup: MNE integer code → our binary label
# mne.events_from_annotations maps annotation strings → sequential integers;
# we need to intercept that mapping to assign our own class labels.
events, event_id_auto = mne.events_from_annotations(raw, verbose="WARNING")

# event_id_auto looks like: {'Stimulus/ 10': 1, 'Stimulus/ 11': 2, ...}
# We remap to binary 0 / 1 using EVENT_ID defined above.
# Step A: build mne_code → binary_label lookup
mne_to_binary = {}
for annot_str, binary_label in EVENT_ID.items():
    if annot_str in event_id_auto:
        mne_code = event_id_auto[annot_str]
        mne_to_binary[mne_code] = binary_label

# Step B: filter events array to only the 8 stimulus codes we care about
mask = np.isin(events[:, 2], list(mne_to_binary.keys()))
events_filtered = events[mask].copy()

# Step C: replace MNE codes with binary labels (0 or 1)
for mne_code, binary_label in mne_to_binary.items():
    events_filtered[events_filtered[:, 2] == mne_code, 2] = binary_label

print(f"  Total events after filtering : {len(events_filtered)}")
print(f"  Class 0 (1x stimuli)         : {(events_filtered[:, 2] == 0).sum()}")
print(f"  Class 1 (2x stimuli)         : {(events_filtered[:, 2] == 1).sum()}")

epoch_event_id = {"Class_0": 0, "Class_1": 1}

# ── Step 5 · Epoch ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Step 5 · Epoching  [tmin={TMIN}s, tmax={TMAX}s]")
print(f"{'='*60}")

epochs = mne.Epochs(
    raw,
    events_filtered,
    event_id=epoch_event_id,
    tmin=TMIN,
    tmax=TMAX,
    baseline=(None, 0),   # baseline correct using pre-stimulus interval
    preload=True,
    reject=None,          # no amplitude rejection yet (Step 2.x)
    verbose="WARNING",
)

print(f"  Epochs created : {len(epochs)}")
print(f"  Class_0        : {len(epochs['Class_0'])}")
print(f"  Class_1        : {len(epochs['Class_1'])}")

# ── Step 6 · Extract tensor & print shape ─────────────────────────────────────
print(f"\n{'='*60}")
print("  Step 6 · Final tensor shape")
print(f"{'='*60}")

X = epochs.get_data()          # shape: [n_epochs, n_channels, n_times]
y = epochs.events[:, 2]        # labels: 0 or 1

print(f"\n  X.shape  = {X.shape}")
print(f"             [n_epochs={X.shape[0]}, n_channels={X.shape[1]}, n_times={X.shape[2]}]")
print(f"  y.shape  = {y.shape}")
print(f"  y values = {np.unique(y, return_counts=True)}")

expected_times = int((TMAX - TMIN) * SFREQ_NEW) + 1
print(f"\n  Expected n_times @ {SFREQ_NEW} Hz over {TMAX-TMIN}s window : {expected_times}")
print(f"  Actual   n_times                                         : {X.shape[2]}")

print(f"\n{'='*60}")
print("  Step 1.2 complete — tensor verified, ready for Step 2 (artifact rejection / feature extraction).")
print(f"{'='*60}\n")