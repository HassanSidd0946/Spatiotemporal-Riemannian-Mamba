# run_data_engine_on_modal.py

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