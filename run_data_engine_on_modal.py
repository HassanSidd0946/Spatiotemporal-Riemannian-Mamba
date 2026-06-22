# run_data_engine_on_modal.py

# # =============================================================================
# # bci_data_engine.py
# # Full-Scale EEG Data Engine for False Memory Classification (LOSO Pipeline)
# # Dataset: OpenNeuro ds005189 | Architecture: Spatiotemporal Riemannian Mamba
# #
# # Usage:
# #   modal run bci_data_engine.py
# # =============================================================================

# import modal
# import shutil

# # =============================================================================
# # SECTION 1: MODAL APP & INFRASTRUCTURE SETUP
# # =============================================================================

# app = modal.App("bci-data-engine")

# # --- Docker Image Definition ---
# # We use debian_slim as a minimal, fast base image.
# # All scientific packages are installed via pip.
# # pinning mne to a stable version to avoid API surprises mid-run.
# eeg_image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .pip_install(
#         "mne==1.7.1",
#         "openneuro-py==2024.2.0",
#         "numpy>=1.26.0",
#         "scipy==1.14.1",   # ✅ Last version jisme sph_harm exist karta tha
#         "tqdm",
#     )
# )

# # --- Persistent Volume ---
# # This volume persists across runs. Raw downloads are stored here
# # temporarily, and the final .npz is saved here permanently.
# # The volume must already exist; create it once via:
# #   modal volume create eeg-data-vol
# volume = modal.Volume.from_name("eeg-data-vol")
# VOLUME_MOUNT_PATH = "/data"

# # --- Subject Configuration ---
# # ds005189 contains 30 subjects: sub-01 through sub-30
# NUM_SUBJECTS = 30

# # --- Event ID Mapping (from BIDS annotations in ds005189) ---
# # Class 0 → True Memory  (Stimulus codes 10–13)
# # Class 1 → False Memory (Stimulus codes 20–23)
# # These string keys must match MNE's annotation descriptions exactly.
# EVENT_ID = {
#     "Stimulus/ 10": 0,
#     "Stimulus/ 11": 0,
#     "Stimulus/ 12": 0,
#     "Stimulus/ 13": 0,
#     "Stimulus/ 20": 1,
#     "Stimulus/ 21": 1,
#     "Stimulus/ 22": 1,
#     "Stimulus/ 23": 1,
# }

# # --- Epoching Parameters ---
# TMIN        = -0.2   # 200ms pre-stimulus baseline
# TMAX        =  0.8   # 800ms post-stimulus
# BASELINE    = (None, 0)  # Correct using pre-stimulus period
# L_FREQ      =  1.0   # Bandpass lower bound (Hz)
# H_FREQ      = 40.0   # Bandpass upper bound (Hz) — removes muscle noise
# RESAMPLE_HZ = 250    # Target sample rate (reduces data size ~4x from 1000 Hz)


# # =============================================================================
# # SECTION 2: MODAL FUNCTION DEFINITION
# # =============================================================================

# @app.function(
#     image=eeg_image,
#     volumes={VOLUME_MOUNT_PATH: volume},  # Mount the persistent volume at /data
#     timeout=10800,                        # 3-hour timeout — full 30-subject run
#     memory=16384,                         # 16 GB RAM — MNE loads raw EEG fully
#     cpu=4.0,                              # 4 vCPUs for parallel MNE operations
# )
# def run_full_pipeline():
#     """
#     Master pipeline function. For each of the 30 subjects:
#       1. Downloads their raw BIDS folder from OpenNeuro into /data
#       2. Loads & preprocesses the .vhdr EEG recording via MNE
#       3. Extracts numpy epochs (X) and binary labels (y)
#       4. Immediately deletes the raw folder to reclaim Volume space
#       5. After all subjects, concatenates and saves a single .npz file
#     """

#     # Deferred imports — these run inside the Modal container, not locally
#     import os
#     import glob
#     import shutil
#     import logging
#     import traceback

#     import numpy as np
#     import mne
#     import openneuro

#     # Suppress MNE's verbose INFO logs to keep Modal output readable.
#     # Set to WARNING so real errors still surface.
#     mne.set_log_level("WARNING")
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
#     log = logging.getLogger("bci-engine")

#     # =========================================================================
#     # ACCUMULATORS — grow across all subjects before final concatenation
#     # =========================================================================
#     all_X        = []   # List of 2-D arrays: (n_epochs, n_channels * n_times)
#     all_y        = []   # List of 1-D label arrays
#     all_subjects = []   # List of string subject IDs, one per epoch

#     failed_subjects = []  # Track which subjects errored for the final report

#     # =========================================================================
#     # SUBJECT LOOP
#     # =========================================================================
#     for sub_idx in range(1, NUM_SUBJECTS + 1):

#         sub_id  = f"{sub_idx:02d}"           # Zero-padded string: "01" … "30"
#         sub_tag = f"sub-{sub_id}"            # BIDS format: "sub-01" … "sub-30"
#         raw_dir = os.path.join(VOLUME_MOUNT_PATH, "openneuro", sub_tag)

#         log.info(f"{'='*60}")
#         log.info(f"Processing {sub_tag} ({sub_idx}/{NUM_SUBJECTS})")
#         log.info(f"{'='*60}")

#         # ---------------------------------------------------------------------
#         # GUARD: Wrap each subject in try-except so a single bad subject
#         # cannot abort the entire 3-hour job. Errors are logged and collected.
#         # ---------------------------------------------------------------------
#         try:

#             # =================================================================
#             # STEP 1: DOWNLOAD — pull only this subject's BIDS folder
#             # =================================================================
#             log.info(f"[{sub_tag}] Downloading from OpenNeuro ds005189 ...")

#             # openneuro.download fetches only the paths matching `include`.
#             # This avoids pulling the entire 45 GB dataset for all subjects.
#             openneuro.download(
#                 dataset="ds005189",
#                 target_dir=os.path.join(VOLUME_MOUNT_PATH, "openneuro"),
#                 include=[sub_tag],   # Only this subject's folder
#             )
#             log.info(f"[{sub_tag}] Download complete.")

#             # =================================================================
#             # STEP 2: LOCATE .vhdr FILE
#             # The BIDS layout places EEG files under:
#             #   sub-XX/eeg/sub-XX_task-*_eeg.vhdr
#             # =================================================================
#             vhdr_pattern = os.path.join(raw_dir, "**", "*.vhdr")
#             vhdr_files   = glob.glob(vhdr_pattern, recursive=True)

#             if not vhdr_files:
#                 raise FileNotFoundError(
#                     f"No .vhdr file found under {raw_dir}. "
#                     f"Download may have been incomplete or the BIDS structure differs."
#                 )

#             # Take the first match (ds005189 has one EEG session per subject)
#             vhdr_path = vhdr_files[0]
#             log.info(f"[{sub_tag}] Found EEG file: {vhdr_path}")

#             # =================================================================
#             # STEP 3: LOAD RAW EEG (BrainVision format)
#             # preload=True loads the full signal into RAM for filtering.
#             # =================================================================
#             raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
#             log.info(
#                 f"[{sub_tag}] Raw loaded | "
#                 f"Channels: {raw.info['nchan']} | "
#                 f"Sfreq: {raw.info['sfreq']} Hz | "
#                 f"Duration: {raw.times[-1]:.1f}s"
#             )

#             # =================================================================
#             # STEP 4: BANDPASS FILTER — 1–40 Hz
#             # Removes DC drift (< 1 Hz) and high-frequency muscle artifacts.
#             # method='iir' is faster than FIR for online-style pipelines.
#             # =================================================================
#             raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="iir", verbose=False)
#             log.info(f"[{sub_tag}] Bandpass filtered: {L_FREQ}–{H_FREQ} Hz")

#             # =================================================================
#             # STEP 5: RESAMPLE TO 250 Hz
#             # Original sfreq in ds005189 is typically 1000 Hz.
#             # 250 Hz retains all relevant ERP components up to 125 Hz (Nyquist).
#             # =================================================================
#             raw.resample(sfreq=RESAMPLE_HZ, verbose=False)
#             log.info(f"[{sub_tag}] Resampled to {RESAMPLE_HZ} Hz")

#             # =================================================================
#             # STEP 6: EXTRACT EVENTS FROM ANNOTATIONS
#             # MNE converts BrainVision markers (stored as Annotations) into
#             # a standard (n_events, 3) integer array via events_from_annotations.
#             # We filter to only the stimulus codes defined in EVENT_ID.
#             # =================================================================
#             events, event_id_found = mne.events_from_annotations(
#                 raw,
#                 event_id=EVENT_ID,   # Only extract the 8 stimulus codes we care about
#                 verbose=False,
#             )
#             log.info(
#                 f"[{sub_tag}] Events extracted: {len(events)} total | "
#                 f"Classes found: {event_id_found}"
#             )

#             if len(events) == 0:
#                 raise ValueError(
#                     f"Zero events found for {sub_tag}. "
#                     f"Check annotation labels in the .vhdr/.vmrk file."
#                 )

#             # =================================================================
#             # STEP 7: EPOCH THE DATA
#             # Creates an Epochs object: shape (n_epochs, n_channels, n_times)
#             # baseline correction subtracts the mean of the pre-stimulus window.
#             # reject=None — we skip automatic rejection here; artifacts can be
#             # handled during model training or with a separate ICA step.
#             # =================================================================
#             epochs = mne.Epochs(
#                 raw,
#                 events,
#                 event_id=event_id_found,   # Use only events actually present
#                 tmin=TMIN,
#                 tmax=TMAX,
#                 baseline=BASELINE,
#                 preload=True,
#                 reject=None,               # No hard rejection threshold
#                 verbose=False,
#             )
#             log.info(
#                 f"[{sub_tag}] Epochs created: {len(epochs)} epochs | "
#                 f"Shape: {epochs.get_data().shape}"
#             )

#             if len(epochs) == 0:
#                 raise ValueError(f"Zero epochs survived for {sub_tag}.")

#             # =================================================================
#             # STEP 8: EXTRACT NUMPY ARRAYS
#             # X shape: (n_epochs, n_channels, n_times)
#             # y: integer labels (0 = true memory, 1 = false memory)
#             # =================================================================
#             X_sub = epochs.get_data(copy=True).astype(np.float32)
#             # Map the original integer event codes back to our binary labels.
#             # epochs.events[:, 2] holds the event code for each epoch.
#             # We reconstruct the binary label by checking which class each
#             # code belongs to in EVENT_ID.
#             raw_codes = epochs.events[:, 2]

#             # Build a lookup: MNE internal event code → binary class
#             # event_id_found maps "Stimulus/ 10" → 1, "Stimulus/ 20" → 2, etc.
#             # We need to reverse this to get MNE code → our label.
#             inv_event_map = {}
#             for annotation_str, binary_label in EVENT_ID.items():
#                 if annotation_str in event_id_found:
#                     mne_code = event_id_found[annotation_str]
#                     inv_event_map[mne_code] = binary_label

#             y_sub = np.array(
#                 [inv_event_map[code] for code in raw_codes],
#                 dtype=np.int32,
#             )

#             # subject_ids: one string per epoch (for LOSO splitting later)
#             subject_ids_sub = np.array([sub_id] * len(y_sub), dtype=object)

#             log.info(
#                 f"[{sub_tag}] Extracted | "
#                 f"X: {X_sub.shape} | "
#                 f"y distribution: class0={np.sum(y_sub==0)}, class1={np.sum(y_sub==1)}"
#             )

#             # =================================================================
#             # STEP 9: APPEND TO GLOBAL ACCUMULATORS
#             # =================================================================
#             all_X.append(X_sub)
#             all_y.append(y_sub)
#             all_subjects.append(subject_ids_sub)

#         # ---------------------------------------------------------------------
#         # ERROR HANDLING — log but do not re-raise; continue to next subject
#         # ---------------------------------------------------------------------
#         except Exception as e:
#             log.error(f"[{sub_tag}] FAILED — {type(e).__name__}: {e}")
#             log.error(traceback.format_exc())
#             failed_subjects.append(sub_tag)

#         # ---------------------------------------------------------------------
#         # CLEANUP — delete raw folder regardless of success/failure
#         # This is the critical memory-management step: we keep only the
#         # extracted numpy arrays, not the raw BrainVision files (~1.5 GB/subject).
#         # ---------------------------------------------------------------------
#         finally:
#             if os.path.exists(raw_dir):
#                 log.info(f"[{sub_tag}] Cleaning up raw folder: {raw_dir}")
#                 shutil.rmtree(raw_dir)
#                 log.info(f"[{sub_tag}] Raw folder deleted. Volume space reclaimed.")

#     # =========================================================================
#     # SECTION 3: CONCATENATE & SAVE
#     # =========================================================================
#     if not all_X:
#         raise RuntimeError(
#             "No subjects were processed successfully. "
#             f"All failed: {failed_subjects}"
#         )

#     log.info("Concatenating data from all successful subjects ...")

#     # np.concatenate along axis=0 stacks epochs from all subjects
#     X_all        = np.concatenate(all_X,        axis=0)  # (total_epochs, channels, times)
#     y_all        = np.concatenate(all_y,        axis=0)  # (total_epochs,)
#     subjects_all = np.concatenate(all_subjects, axis=0)  # (total_epochs,)

#     log.info(
#         f"Final dataset | "
#         f"X: {X_all.shape} | "
#         f"y: {y_all.shape} | "
#         f"Subjects array: {subjects_all.shape} | "
#         f"Dtype X: {X_all.dtype}"
#     )
#     log.info(
#         f"Label distribution across all subjects — "
#         f"Class 0 (True Memory): {np.sum(y_all==0)} | "
#         f"Class 1 (False Memory): {np.sum(y_all==1)}"
#     )

#     # --- Save as compressed .npz ---
#     # npz stores multiple arrays in a single zip archive.
#     # float32 + compression typically achieves ~4–6x size reduction vs. raw.
#     output_path = os.path.join(VOLUME_MOUNT_PATH, "processed_eeg_all_subjects.npz")
#     log.info(f"Saving compressed dataset to {output_path} ...")

#     np.savez_compressed(
#         output_path,
#         X=X_all,               # EEG epochs: float32 (epochs, channels, times)
#         y=y_all,               # Binary labels: int32 (epochs,)
#         subjects=subjects_all, # Subject IDs: str array (epochs,) — for LOSO splits
#     )

#     log.info(f"Dataset saved. File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

#     # =========================================================================
#     # SECTION 4: COMMIT VOLUME
#     # volume.commit() flushes all writes and makes them visible to future
#     # Modal runs or local `modal volume get` commands.
#     # Without this call, data written in a function is NOT guaranteed to persist.
#     # =========================================================================
#     log.info("Committing Modal Volume to persist the output ...")
#     volume.commit()
#     log.info("Volume committed. All data is now permanently stored.")

#     # =========================================================================
#     # FINAL REPORT
#     # =========================================================================
#     successful = NUM_SUBJECTS - len(failed_subjects)
#     log.info(f"{'='*60}")
#     log.info(f"PIPELINE COMPLETE")
#     log.info(f"  Subjects succeeded : {successful}/{NUM_SUBJECTS}")
#     log.info(f"  Subjects failed    : {len(failed_subjects)} → {failed_subjects}")
#     log.info(f"  Output             : {output_path}")
#     log.info(f"  Total epochs       : {len(y_all)}")
#     log.info(f"{'='*60}")

#     return {
#         "output_path"       : output_path,
#         "shape_X"           : X_all.shape,
#         "total_epochs"      : int(len(y_all)),
#         "class_0_count"     : int(np.sum(y_all == 0)),
#         "class_1_count"     : int(np.sum(y_all == 1)),
#         "successful_subjects": successful,
#         "failed_subjects"   : failed_subjects,
#     }


# # =============================================================================
# # SECTION 5: LOCAL ENTRYPOINT
# # Called when you run: modal run bci_data_engine.py
# # =============================================================================

# @app.local_entrypoint()
# def main():
#     print("Launching BCI Data Engine on Modal cloud ...")
#     print("This will process all 30 subjects from OpenNeuro ds005189.")
#     print("Estimated runtime: 90–180 minutes depending on download speed.\n")

#     result = run_full_pipeline.remote()

#     print("\n" + "="*60)
#     print("REMOTE RUN COMPLETE")
#     print("="*60)
#     for key, val in result.items():
#         print(f"  {key:<25}: {val}")
#     print("="*60)
#     print(
#         "\nTo download the output locally, run:\n"
#         "  modal volume get eeg-data-vol processed_eeg_all_subjects.npz .\n"
#     )






























































































# =============================================================================
# bci_data_engine.py  — v2: Preemption-Safe Checkpoint Strategy
# =============================================================================

import modal

app = modal.App("bci-data-engine")

eeg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mne==1.7.1",
        "openneuro-py==2024.2.0",
        "numpy>=1.26.0",
        "scipy==1.14.1",
        "tqdm",
    )
)

volume = modal.Volume.from_name("eeg-data-vol")
VOLUME_MOUNT_PATH = "/data"
CHECKPOINT_DIR   = "/data/checkpoints"   # ← Har subject ka .npz yahan save hoga

NUM_SUBJECTS = 30

EVENT_ID = {
    "Stimulus/ 10": 0, "Stimulus/ 11": 0,
    "Stimulus/ 12": 0, "Stimulus/ 13": 0,
    "Stimulus/ 20": 1, "Stimulus/ 21": 1,
    "Stimulus/ 22": 1, "Stimulus/ 23": 1,
}

TMIN        = -0.2
TMAX        =  0.8
BASELINE    = (None, 0)
L_FREQ      =  1.0
H_FREQ      = 40.0
RESAMPLE_HZ = 250


@app.function(
    image=eeg_image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=10800,
    memory=16384,
    cpu=4.0,
)
def run_full_pipeline():
    import os, glob, shutil, logging, traceback
    import numpy as np
    import mne
    import openneuro

    mne.set_log_level("WARNING")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    log = logging.getLogger("bci-engine")

    # =========================================================================
    # Checkpoint directory banana — yahan har subject ka .npz save hoga
    # =========================================================================
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    failed_subjects = []

    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id  = f"{sub_idx:02d}"
        sub_tag = f"sub-{sub_id}"
        raw_dir = os.path.join(VOLUME_MOUNT_PATH, "openneuro", sub_tag)

        # =====================================================================
        # CHECKPOINT CHECK — agar yeh subject pehle se done hai toh skip karo
        # Yahi preemption-safety ka core hai
        # =====================================================================
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{sub_tag}.npz")
        if os.path.exists(checkpoint_path):
            log.info(f"[{sub_tag}] ALREADY DONE — checkpoint found, skipping.")
            continue

        log.info(f"{'='*60}")
        log.info(f"Processing {sub_tag} ({sub_idx}/{NUM_SUBJECTS})")
        log.info(f"{'='*60}")

        try:
            # STEP 1: Download
            log.info(f"[{sub_tag}] Downloading ...")
            openneuro.download(
                dataset="ds005189",
                target_dir=os.path.join(VOLUME_MOUNT_PATH, "openneuro"),
                include=[sub_tag],
            )

            # STEP 2: Find .vhdr
            vhdr_files = glob.glob(
                os.path.join(raw_dir, "**", "*.vhdr"), recursive=True
            )
            if not vhdr_files:
                raise FileNotFoundError(f"No .vhdr found under {raw_dir}")
            vhdr_path = vhdr_files[0]
            log.info(f"[{sub_tag}] EEG file: {vhdr_path}")

            # STEP 3: Load
            raw = mne.io.read_raw_brainvision(
                vhdr_path, preload=True, verbose=False
            )
            log.info(
                f"[{sub_tag}] Loaded | Ch: {raw.info['nchan']} | "
                f"Sfreq: {raw.info['sfreq']} Hz | Dur: {raw.times[-1]:.1f}s"
            )

            # STEP 4: Filter
            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="iir", verbose=False)

            # STEP 5: Resample
            raw.resample(sfreq=RESAMPLE_HZ, verbose=False)

            # STEP 6: Events
            events, event_id_found = mne.events_from_annotations(
                raw, event_id=EVENT_ID, verbose=False
            )
            if len(events) == 0:
                raise ValueError(f"Zero events found for {sub_tag}")

            # STEP 7: Epochs
            epochs = mne.Epochs(
                raw, events, event_id=event_id_found,
                tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                preload=True, reject=None, verbose=False,
            )
            if len(epochs) == 0:
                raise ValueError(f"Zero epochs survived for {sub_tag}")

            # STEP 8: Extract numpy arrays
            X_sub = epochs.get_data(copy=True).astype(np.float32)

            inv_event_map = {
                event_id_found[k]: v
                for k, v in EVENT_ID.items()
                if k in event_id_found
            }
            y_sub = np.array(
                [inv_event_map[c] for c in epochs.events[:, 2]],
                dtype=np.int32,
            )
            subject_ids_sub = np.array([sub_id] * len(y_sub), dtype=object)

            log.info(
                f"[{sub_tag}] Extracted | X: {X_sub.shape} | "
                f"class0={np.sum(y_sub==0)}, class1={np.sum(y_sub==1)}"
            )

            # =================================================================
            # STEP 9: CHECKPOINT SAVE — volume pe likh do ABHI
            # Yeh line preemption ke against protection hai
            # =================================================================
            np.savez_compressed(
                checkpoint_path,
                X=X_sub,
                y=y_sub,
                subjects=subject_ids_sub,
            )
            # Har subject ke baad commit — data secure karo
            volume.commit()
            log.info(f"[{sub_tag}] ✅ Checkpoint saved & committed.")

        except Exception as e:
            log.error(f"[{sub_tag}] FAILED — {type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            failed_subjects.append(sub_tag)

        finally:
            # Cleanup raw folder — space bachao
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir)
                log.info(f"[{sub_tag}] Raw folder deleted.")

    # =========================================================================
    # FINAL MERGE — saare checkpoint .npz files ek mein merge karo
    # =========================================================================
    log.info("All subjects done. Merging checkpoints ...")

    all_X, all_y, all_subjects = [], [], []

    for sub_idx in range(1, NUM_SUBJECTS + 1):
        sub_id = f"{sub_idx:02d}"
        cp = os.path.join(CHECKPOINT_DIR, f"sub-{sub_id}.npz")
        if os.path.exists(cp):
            data = np.load(cp, allow_pickle=True)
            all_X.append(data["X"])
            all_y.append(data["y"])
            all_subjects.append(data["subjects"])
            log.info(f"[sub-{sub_id}] Loaded from checkpoint: {data['X'].shape}")
        else:
            log.warning(f"[sub-{sub_id}] No checkpoint found — was failed/skipped.")

    if not all_X:
        raise RuntimeError("No checkpoints found to merge!")

    X_all        = np.concatenate(all_X,        axis=0)
    y_all        = np.concatenate(all_y,        axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    output_path = os.path.join(VOLUME_MOUNT_PATH, "processed_eeg_all_subjects.npz")
    np.savez_compressed(output_path, X=X_all, y=y_all, subjects=subjects_all)

    log.info(f"Final dataset saved: {output_path}")
    log.info(f"Shape X: {X_all.shape} | Total epochs: {len(y_all)}")
    log.info(
        f"class0={np.sum(y_all==0)} | class1={np.sum(y_all==1)}"
    )
    log.info(f"Failed: {failed_subjects}")

    volume.commit()
    log.info("Volume committed. DONE!")

    return {
        "output_path"        : output_path,
        "shape_X"            : X_all.shape,
        "total_epochs"       : int(len(y_all)),
        "class_0_count"      : int(np.sum(y_all == 0)),
        "class_1_count"      : int(np.sum(y_all == 1)),
        "failed_subjects"    : failed_subjects,
    }


@app.local_entrypoint()
def main():
    print("Launching BCI Data Engine v2 (Preemption-Safe) ...")
    result = run_full_pipeline.remote()
    print("\n" + "="*60)
    for k, v in result.items():
        print(f"  {k:<25}: {v}")
    print("="*60)
    print("\nDownload:\n  modal volume get eeg-data-vol processed_eeg_all_subjects.npz .")