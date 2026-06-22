# step1_1_fetch_inspect.py

import sys
from pathlib import Path
import openneuro
import mne

DATASET_ID  = "ds005189"
SUBJECT     = "01"
TARGET_DIR  = Path("./data") / DATASET_ID

print(f"\n{'='*60}")
print(f"  Downloading sub-{SUBJECT} from OpenNeuro {DATASET_ID}")
print(f"{'='*60}\n")

openneuro.download(
    dataset=DATASET_ID,
    target_dir=TARGET_DIR,
    include=[f"sub-{SUBJECT}", "/*.json"],
    verify_hash=True,
)

print("\nDownload complete.\n")

SUPPORTED_EXT = (".set", ".fif", ".vhdr", ".edf", ".bdf")
eeg_files = sorted(p for p in TARGET_DIR.rglob(f"sub-{SUBJECT}/**/*") if p.suffix in SUPPORTED_EXT)

if not eeg_files:
    sys.exit(f"[ERROR] No EEG file found for sub-{SUBJECT} under {TARGET_DIR}.")

eeg_path = eeg_files[0]
print(f"EEG file: {eeg_path}\n")

try:
    ext = eeg_path.suffix.lower()
    if ext == ".set":
        raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose="WARNING")
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(eeg_path, preload=False, verbose="WARNING")
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose="WARNING")
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(eeg_path, preload=False, verbose="WARNING")
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(eeg_path, preload=False, verbose="WARNING")
    else:
        raise ValueError(f"Unhandled extension: {ext}")
except Exception as exc:
    sys.exit(f"[ERROR] Could not load EEG file.\n  {exc}")

print(raw.info)
print(f"\n  Channel count : {len(raw.ch_names)}")
print(f"  Sampling rate : {raw.info['sfreq']} Hz")
print(f"  Duration      : {raw.times[-1]:.1f} s\n")

if raw.annotations and len(raw.annotations) > 0:
    all_descs = [str(d) for d in raw.annotations.description]
    unique = sorted(set(all_descs))
    print(f"  Total annotations : {len(raw.annotations)}")
    print(f"  Unique labels ({len(unique)}):\n")
    for d in unique:
        print(f"    {d!r:<40}  (n={all_descs.count(d)})")
else:
    print("  [WARNING] No annotations found.")
    for ef in sorted(TARGET_DIR.rglob(f"sub-{SUBJECT}/**/*_events.tsv")):
        print(f"  Found events file: {ef}")
