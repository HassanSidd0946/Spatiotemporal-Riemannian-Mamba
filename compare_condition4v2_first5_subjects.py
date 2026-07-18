# =============================================================================
# command to run : modal run compare_condition4v2_first5_subjects.py::main
# compare_condition4v2_first5_subjects.py
#
# Zero-cost check: pulls per-subject accuracies for sub-01..sub-05 out of
# results_condition4_subspace_calibrated.json (already on the volume from
# the original Condition 4v2 run) and compares that 5-subject subset mean
# against this session's spatial-only ablation (60.41%) and the full
# dual-branch pilot (55.70%).
#
# No GPU, no training, no model code at all — just JSON parsing on CPU.
# This isolates "is 60.41% low because of subject-selection variance, or
# because of a real regression from the heavier regularization / smaller
# embedding dim" before spending any more Modal budget chasing the wrong one.
#
# Usage: modal run compare_condition4v2_first5_subjects.py::main
# =============================================================================

import modal

app = modal.App("bci-compare-first5-subjects")
volume = modal.Volume.from_name("eeg-data-vol")

CONDITION4V2_JSON = "/data/results_condition4_subspace_calibrated.json"
VOLUME_PATH = "/data"

TARGET_SUBJECTS = ["01", "02", "03", "04", "05"]

# Reference numbers from this session, for the printed comparison
DUALBRANCH_PILOT_MEAN_ACC = 0.5570
SPATIAL_ONLY_ABLATION_MEAN_ACC = 0.6041
CONDITION4V2_FULL_29_MEAN_ACC = 0.6779

cpu_image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume}, timeout=60)
def compare_first5():
    import json
    import statistics

    with open(CONDITION4V2_JSON) as f:
        data = json.load(f)

    fold_results = data.get("fold_results", [])
    print(f"Total folds in {CONDITION4V2_JSON}: {len(fold_results)}")

    # Condition 4v2's own fold_results use "test_subject" as the key (per its
    # own script's fold_records.append(...) — same field name used throughout
    # this session's other scripts), and "post_calibration_acc" for the
    # calibrated accuracy. Confirm both are present before trusting the pull.
    sample = fold_results[0] if fold_results else {}
    if "test_subject" not in sample or "post_calibration_acc" not in sample:
        print(f"⚠️  Unexpected fold_result schema. Keys found: {list(sample.keys())}")
        print("    Update TEST_SUBJECT_KEY / ACC_KEY below to match, then rerun.")
        return {"error": "schema_mismatch", "sample_keys": list(sample.keys())}

    subject_accs = {}
    for rec in fold_results:
        sub = str(rec["test_subject"])
        subject_accs[sub] = float(rec["post_calibration_acc"])

    print(f"\nAll subjects found in Condition 4v2 results: {sorted(subject_accs.keys())}\n")

    matched = {}
    missing = []
    for sub in TARGET_SUBJECTS:
        if sub in subject_accs:
            matched[sub] = subject_accs[sub]
        else:
            missing.append(sub)

    if missing:
        print(f"⚠️  Subjects not found under these exact keys: {missing}")
        print("    Check for zero-padding / naming differences (e.g. '1' vs '01') "
              "in the printed subject list above and adjust TARGET_SUBJECTS if needed.")

    print(f"{'Subject':<10}{'Condition 4v2 Acc':>20}")
    print("-" * 30)
    for sub in TARGET_SUBJECTS:
        if sub in matched:
            print(f"{sub:<10}{matched[sub]*100:>19.2f}%")

    if matched:
        subset_mean = statistics.mean(matched.values())
        subset_std = statistics.pstdev(matched.values()) if len(matched) > 1 else 0.0

        print(f"\n{'='*60}")
        print(f"  Condition 4v2, THIS 5-subject subset only : {subset_mean*100:6.2f}% (± {subset_std*100:.2f})")
        print(f"  Condition 4v2, full 29-subject mean       : {CONDITION4V2_FULL_29_MEAN_ACC*100:6.2f}%")
        print(f"  This session's spatial-only ablation (n=5): {SPATIAL_ONLY_ABLATION_MEAN_ACC*100:6.2f}%")
        print(f"  This session's dual-branch pilot (n=5)    : {DUALBRANCH_PILOT_MEAN_ACC*100:6.2f}%")
        print(f"{'='*60}")

        gap_vs_subset = subset_mean - SPATIAL_ONLY_ABLATION_MEAN_ACC
        gap_vs_full = CONDITION4V2_FULL_29_MEAN_ACC - subset_mean

        print(f"\n  Condition 4v2 on THIS SAME 5-subject subset vs. this session's "
              f"spatial-only ablation: {gap_vs_subset*100:+.2f} pp")
        print(f"  Condition 4v2's full-29 mean vs. its OWN 5-subject subset mean: "
              f"{gap_vs_full*100:+.2f} pp (this is how much of the '7.4pp gap' is "
              f"just subject-selection variance, independent of any code change)")

        return {
            "condition4v2_subset_mean": subset_mean,
            "condition4v2_subset_per_subject": matched,
            "condition4v2_full_29_mean": CONDITION4V2_FULL_29_MEAN_ACC,
            "spatial_only_ablation_mean": SPATIAL_ONLY_ABLATION_MEAN_ACC,
            "dualbranch_pilot_mean": DUALBRANCH_PILOT_MEAN_ACC,
            "gap_subset_vs_spatial_only_ablation_pp": gap_vs_subset * 100,
            "gap_full29_vs_own_subset_pp": gap_vs_full * 100,
        }
    else:
        return {"error": "no_subjects_matched"}


@app.local_entrypoint()
def main():
    result = compare_first5.remote()
    print("\n(Returned dict, for reference:)")
    print(result)