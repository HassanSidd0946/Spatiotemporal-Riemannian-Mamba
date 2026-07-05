# =============================================================================
# command to run : modal run generate_figure3_3d_manifold.py::main
# generate_figure3_3d_manifold.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure3_3d_riemannian_manifold.png .
# modal volume get eeg-data-vol /figures/figure3_3d_riemannian_manifold.pdf .
# modal volume get eeg-data-vol /figures/figure3_interactive_manifold.html .
# modal volume get eeg-data-vol /figures/figure3a_singlesubject_manifold.png .
# modal volume get eeg-data-vol /figures/figure3a_singlesubject_manifold.pdf .
#
# Phase 3.2 (v3) — Figure 3: Three-Panel Manifold Visualization
#   Panel A — Best-responder single-subject manifold (unsupervised PCA axes
#             inherited from that subject's real Condition-4 LOSO fold,
#             t-SNE for 3D layout only; points shaped by ACTUAL stored
#             prediction correctness from that fold's calibrated classifier)
#   Panel B — Multi-subject manifold, colored by TRUE CLASS
#   Panel C — The SAME multi-subject coordinates, colored by SUBJECT ID
#             (Panel B vs C together are the actual evidence for the
#             "subject islands" diagnosis, not merely an assertion of it)
#
# WHY THIS VERSION EXISTS:
#   v2 showed that a label-informed (LDA+t-SNE) embedding of pooled,
#   pre-calibration tangent features produces ~15-20 geometric islands that
#   correspond to SUBJECT IDENTITY, not cognitive state — i.e. inter-subject
#   variance dominates raw tangent-space geometry. That is a genuine,
#   important finding (it is exactly why Condition 4's per-subject
#   calibration exists), but it means v2's figure could not honestly show
#   "clean class separation" without either (a) misrepresenting the pooled
#   geometry, or (b) laundering group-level pooling through a single fold's
#   subspace (Option 1 from the request — see caveat below).
#
#   v3 resolves this the way high-impact neuroscience papers actually do:
#   show a genuine best-case single-subject example (Panel A) AND the honest
#   population-level picture that explains why calibration is necessary
#   (Panels B/C), rather than picking one and hiding the other.
#
# ⚠️ WHY OPTION 1 ("GLOBAL CALIBRATED LATENT SPACE") WAS NOT IMPLEMENTED:
#   Condition 4 fits a DIFFERENT PCA subspace per LOSO fold (each fit on a
#   different set of 28 training subjects). There is no single shared
#   "calibrated latent space" to project all 29 subjects into. Projecting
#   every subject through any one fold's subspace would silently put 28 of
#   29 subjects in-sample (they were in that fold's training pool) and only
#   1 genuinely held out — producing an artificially clean plot that
#   misrepresents held-out generalization. If you need a true global
#   calibrated view, it requires cross-fold subspace alignment (e.g.
#   Procrustes registration of each fold's 35-dim PCA basis) which is a
#   separate, nontrivial analysis — flag this explicitly if you want it
#   built next, rather than approximating it here.
#
# ⚠️ PANEL A SCIENTIFIC DISCLOSURE:
#   The subject shown is selected as the TOP performer by real, stored
#   Condition-4 post-calibration LOSO accuracy (read from
#   results_condition4_subspace_calibrated.json at runtime — never
#   hardcoded), and is explicitly captioned as a best-case example, not a
#   representative one. Its PCA axes are fit ONLY on the other 28 subjects
#   (exactly as in the real Condition-4 pipeline), so no label or feature
#   information from this subject leaks into the axes themselves. t-SNE is
#   applied afterward purely as an unsupervised 3D layout tool on those
#   already-computed, leakage-free PCA coordinates. Marker shape (correct /
#   incorrect) comes directly from the ACTUAL stored predictions of that
#   fold's shrinkage-blended classifier — it is not re-fit or reinterpreted
#   here. Population-level separability claims must still cite the
#   Condition-4 aggregate mean/std accuracy, not this single-subject panel.
#
# ⚠️ PANEL B/C SCIENTIFIC DISCLOSURE (unchanged from v2):
#   Panel B's axes come from a label-informed pipeline (LDA + PCA
#   pre-compression → t-SNE) fit on the pooled, PRE-calibration multi-
#   subject subsample. This is disclosed because LDA explicitly optimizes
#   for between-class separation using the same labels shown in the plot.
#   The finding these panels support is precisely that even a label-
#   informed axis fails to produce clean class separation once you look at
#   what's actually structuring the geometry (Panel C: subject identity) —
#   that failure is the evidence, not a limitation being hidden.
#
# LEAKAGE NOTE:
#   This is a visualization-only script. No component built here is fit
#   using any held-out subject's own D_test, and this script's outputs must
#   never be substituted into, or reported as, the strict LOSO calibration
#   pipeline (Condition 3/4). No classification accuracy is computed here
#   except by faithfully reproducing the exact Condition-4 split and then
#   using the numbers already reported in results_condition4_subspace_calibrated.json.
#
# Usage: modal run generate_figure3_3d_manifold.py::main
#        modal run generate_figure3_3d_manifold.py::inspect
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-figure3-3d-manifold")
volume = modal.Volume.from_name("eeg-data-vol")

DATA_PATH        = "/data/tangent_features_ea.npz"
CONDITION4_JSON  = "/data/results_condition4_subspace_calibrated.json"
FIGURES_DIR      = "/data/figures"

PNG_MAIN   = f"{FIGURES_DIR}/figure3_3d_riemannian_manifold.png"     # 3-panel A+B+C
PDF_MAIN   = f"{FIGURES_DIR}/figure3_3d_riemannian_manifold.pdf"
HTML_MAIN  = f"{FIGURES_DIR}/figure3_interactive_manifold.html"      # 3-scene interactive

PNG_A      = f"{FIGURES_DIR}/figure3a_singlesubject_manifold.png"    # Panel A standalone
PDF_A      = f"{FIGURES_DIR}/figure3a_singlesubject_manifold.pdf"

VOLUME_PATH = "/data"

RANDOM_SEED       = 42

# --- Panel B/C (multi-subject) hyperparameters, unchanged from v2 ---
N_TOTAL_POINTS    = 600
N_PER_CLASS       = N_TOTAL_POINTS // 2
LDA_N_COMPONENTS  = 1
PCA_PRECOMP_DIMS  = 30
TSNE_N_COMPONENTS = 3
TSNE_PERPLEXITY   = 30
METHOD_NAME_BC    = "Supervised LDA+PCA Pre-Compression -> t-SNE (label-informed)"

# --- Panel A hyperparameters: MUST match Condition 4's real pipeline exactly ---
C4_CAL_FRACTION      = 0.15
C4_PCA_N_COMPONENTS  = 35
C4_RANDOM_SEED       = 42
METHOD_NAME_A        = "Condition-4 PCA subspace (unsupervised, fit on other 28 subjects) -> t-SNE (layout only)"

CLASS_COLORS = {
    0: "#BE123C",  # Deep Crimson — Class 0: False Memory
    1: "#0D9488",  # Rich Emerald Teal — Class 1: True Memory
}
CLASS_LABELS = {
    0: "False Memory",
    1: "True Memory",
}

viz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "scikit-learn==1.4.2",
        "scipy",
        "matplotlib==3.8.4",
        "plotly==5.22.0",
        "pandas",
        "kaleido==0.2.1",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run generate_figure3_3d_manifold.py::inspect
# =============================================================================

@app.function(image=viz_image, volumes={VOLUME_PATH: volume}, timeout=300)
def inspect_npz():
    import numpy as np
    raw = np.load(DATA_PATH, allow_pickle=True)
    print(f"\nArchive: {DATA_PATH}")
    print(f"Keys found: {raw.files}\n")
    for k in raw.files:
        arr = raw[k]
        print(f"  '{k}': shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim <= 1 and arr.size <= 40:
            print(f"      sample values: {arr[:10]}")
    return raw.files


@app.local_entrypoint(name="inspect")
def inspect_entrypoint():
    keys = inspect_npz.remote()
    print(f"\nKeys in {DATA_PATH}: {keys}")


# =============================================================================
# SECTION 2: MODAL FUNCTION
# =============================================================================

@app.function(
    image=viz_image,
    cpu=4.0,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    memory=16384,
)
def generate_figure3():

    import os
    import json
    import time
    import logging

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import TSNE
    from sklearn.model_selection import StratifiedShuffleSplit

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("figure3-manifold")

    np.random.seed(RANDOM_SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =========================================================================
    # SECTION 3: DATA LOADING
    # =========================================================================
    log.info(f"Loading {DATA_PATH} ...")
    raw = np.load(DATA_PATH, allow_pickle=True)
    log.info(f"Archive keys found: {raw.files}")

    def _pick_key(candidates, purpose):
        for c in candidates:
            if c in raw.files:
                return c
        raise KeyError(
            f"Could not find a key for '{purpose}' in {DATA_PATH}. "
            f"Tried {candidates}, but archive only contains {raw.files}."
        )

    X_KEY   = _pick_key(["X_ts", "X", "features", "tangent_features", "tangent_vectors", "data", "X_tangent"], "feature matrix")
    Y_KEY   = _pick_key(["y", "labels", "label", "Y", "targets"], "labels")
    SUB_KEY = _pick_key(["subjects", "subject", "subject_ids", "subject_id", "groups", "sub_ids"], "subject IDs")
    log.info(f"Using keys -> features: '{X_KEY}', labels: '{Y_KEY}', subjects: '{SUB_KEY}'")

    X_np        = raw[X_KEY].astype(np.float64)
    y_np        = raw[Y_KEY].astype(np.int64)
    subjects_np = np.asarray(raw[SUB_KEY])

    if X_np.ndim != 2:
        X_np = X_np.reshape(X_np.shape[0], -1)

    N_CLASSES = int(y_np.max()) + 1
    assert N_CLASSES == 2, "This figure pipeline assumes binary classification (True/False Memory)."
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Subjects: {len(np.unique(subjects_np))}")

    # =========================================================================
    # SECTION 3.5: LOAD CONDITION-4 RESULTS, PICK BEST RESPONDER DYNAMICALLY
    #   The subject used in Panel A is NEVER hardcoded — it is selected as
    #   the argmax of post_calibration_acc across the real 29-fold results.
    # =========================================================================
    if not os.path.exists(CONDITION4_JSON):
        raise FileNotFoundError(
            f"{CONDITION4_JSON} not found. Panel A requires the real Condition-4 "
            f"fold results to select a subject and mark prediction correctness. "
            f"Run run_step4_condition4_subspace_calibration.py first."
        )

    with open(CONDITION4_JSON, "r") as f:
        c4 = json.load(f)

    fold_records = c4["fold_results"]
    accs = [(rec["test_subject"], rec["post_calibration_acc"]) for rec in fold_records]
    accs_sorted = sorted(accs, key=lambda t: t[1], reverse=True)
    best_subject, best_acc = accs_sorted[0]
    median_acc = float(np.median([a for _, a in accs]))
    worst_subject, worst_acc = accs_sorted[-1]
    group_mean = float(c4["mean_accuracy"])
    group_std  = float(c4["std_accuracy"])

    log.info(f"Condition-4 accuracy distribution across {len(accs)} folds:")
    log.info(f"  BEST   : sub-{best_subject}  = {best_acc*100:.2f}%  (rank 1 of {len(accs)})")
    log.info(f"  MEDIAN : {median_acc*100:.2f}%")
    log.info(f"  WORST  : sub-{worst_subject} = {worst_acc*100:.2f}%")
    log.info(f"  GROUP MEAN ± STD : {group_mean*100:.2f}% ± {group_std*100:.2f}%")
    log.info(f"  -> Panel A will show sub-{best_subject} (objectively selected, best-case, NOT typical).")

    best_fold_record = next(rec for rec in fold_records if rec["test_subject"] == best_subject)
    stored_y_true = np.array(best_fold_record["y_true"], dtype=np.int64)
    stored_y_pred = np.array(best_fold_record["y_pred"], dtype=np.int64)

    # =========================================================================
    # SECTION 4: PANEL A — FAITHFUL RECONSTRUCTION OF THE BEST SUBJECT'S
    #   REAL CONDITION-4 LOSO FOLD (unsupervised PCA axes, no label leakage)
    # =========================================================================
    log.info(f"\nReconstructing Condition-4 LOSO fold for sub-{best_subject} "
             f"(StandardScaler + PCA fit on the other 28 subjects only)...")

    is_holdout = subjects_np == best_subject
    is_train28 = ~is_holdout
    X_train28_raw = X_np[is_train28]
    X_k_raw       = X_np[is_holdout]
    y_k           = y_np[is_holdout]

    scaler_a = StandardScaler()
    X_train28_z = scaler_a.fit_transform(X_train28_raw)
    X_k_z       = scaler_a.transform(X_k_raw)

    pca_a = PCA(n_components=C4_PCA_N_COMPONENTS, random_state=C4_RANDOM_SEED)
    pca_a.fit(X_train28_z)
    X_k_pca = pca_a.transform(X_k_z)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - C4_CAL_FRACTION), random_state=C4_RANDOM_SEED)
    cal_idx, test_idx = next(sss.split(X_k_pca, y_k))

    X_cal_pca, y_cal   = X_k_pca[cal_idx],  y_k[cal_idx]
    X_test_pca, y_test = X_k_pca[test_idx], y_k[test_idx]

    # ---- Sanity check: our reconstructed split must match the stored fold exactly ----
    if len(y_test) == len(stored_y_true) and np.array_equal(y_test, stored_y_true):
        log.info(f"  Verified: reconstructed D_test ({len(y_test)} epochs) matches "
                 f"stored y_true from {CONDITION4_JSON} exactly.")
        y_pred_for_test = stored_y_pred
    else:
        log.warning(
            f"  Reconstructed D_test does not exactly match stored y_true "
            f"(len {len(y_test)} vs {len(stored_y_true)}, or value mismatch). "
            f"Falling back to marking all D_test points as 'unverified' (no "
            f"correctness overlay) rather than risk mislabeling predictions."
        )
        y_pred_for_test = None

    n_cal, n_test = len(y_cal), len(y_test)
    log.info(f"  D_cal={n_cal} epochs, D_test={n_test} epochs (subject sub-{best_subject} only)")

    X_a_combined = np.vstack([X_cal_pca, X_test_pca])
    y_a_combined = np.concatenate([y_cal, y_test])
    is_test_a    = np.concatenate([np.zeros(n_cal, dtype=bool), np.ones(n_test, dtype=bool)])

    log.info(f"  Fitting unsupervised t-SNE(n_components=3) purely as a 3D layout tool "
             f"on the {C4_PCA_N_COMPONENTS}-dim leakage-free PCA coordinates...")
    perplexity_a = max(5, min(30, (n_cal + n_test) // 4))
    tsne_a = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=perplexity_a,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_SEED,
    )
    X_a_plot = tsne_a.fit_transform(X_a_combined)

    if y_pred_for_test is not None:
        correct_a = (y_pred_for_test == y_test)
        log.info(f"  Panel A held-out accuracy check: "
                 f"{correct_a.mean()*100:.2f}% (should equal stored post_calibration_acc "
                 f"{best_acc*100:.2f}%)")

    # =========================================================================
    # SECTION 5: PANEL B/C — MULTI-SUBJECT MANIFOLD (unchanged pipeline from v2)
    # =========================================================================
    log.info(f"\nDrawing multi-subject stratified balanced subsample: "
             f"{N_PER_CLASS} Class 0 + {N_PER_CLASS} Class 1...")

    rng = np.random.RandomState(RANDOM_SEED)
    sampled_idx_parts = []
    for cls in (0, 1):
        cls_idx = np.where(y_np == cls)[0]
        chosen = rng.choice(cls_idx, size=N_PER_CLASS, replace=False)
        sampled_idx_parts.append(chosen)
    sample_idx = np.concatenate(sampled_idx_parts)
    rng.shuffle(sample_idx)
    assert len(set(sample_idx.tolist())) == N_TOTAL_POINTS

    X_sub    = X_np[sample_idx]
    y_bc     = y_np[sample_idx]
    sub_bc   = subjects_np[sample_idx]

    scaler_bc = StandardScaler()
    X_bc_z = scaler_bc.fit_transform(X_sub)

    lda = LinearDiscriminantAnalysis(n_components=LDA_N_COMPONENTS)
    lda_embedding = lda.fit_transform(X_bc_z, y_bc)

    pca_dims = min(PCA_PRECOMP_DIMS, X_bc_z.shape[0] - 1, X_bc_z.shape[1])
    pca_bc = PCA(n_components=pca_dims, random_state=RANDOM_SEED)
    pca_embedding = pca_bc.fit_transform(X_bc_z)

    combined_bc = np.concatenate([lda_embedding, pca_embedding], axis=1)

    log.info(f"  Fitting TSNE(n_components=3, perplexity={TSNE_PERPLEXITY}) on the "
             f"label-informed LDA+PCA embedding (see disclosure in file header)...")
    tsne_bc = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=TSNE_PERPLEXITY,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_SEED,
    )
    X_bc_plot = tsne_bc.fit_transform(combined_bc)

    centroid0 = X_bc_plot[y_bc == 0].mean(axis=0)
    centroid1 = X_bc_plot[y_bc == 1].mean(axis=0)
    centroid_gap = np.linalg.norm(centroid1 - centroid0)
    within0 = np.linalg.norm(X_bc_plot[y_bc == 0] - centroid0, axis=1).mean()
    within1 = np.linalg.norm(X_bc_plot[y_bc == 1] - centroid1, axis=1).mean()

    unique_subjects_bc = sorted(np.unique(sub_bc).tolist())
    n_islands_subjects = len(unique_subjects_bc)
    log.info(f"  Class centroid separation: {centroid_gap:.3f} "
             f"(within-class spread: C0={within0:.3f}, C1={within1:.3f})")
    log.info(f"  Unique subjects represented in multi-subject sample: {n_islands_subjects}")
    log.info("  NOTE: class-colored separation here is label-informed (LDA-driven); "
             "compare Panel B vs Panel C below to see what actually structures the geometry.")

    # =========================================================================
    # SECTION 6: STATIC 3-PANEL FIGURE (MATPLOTLIB)
    # =========================================================================
    log.info("\nRendering 3-panel static publication figure (Matplotlib)...")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8})

    fig = plt.figure(figsize=(23, 8.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04])

    def _style_3d_axes(ax):
        pane_color = (0.96, 0.96, 0.97, 1.0)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0.4
            axis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.5)
        ax.set_box_aspect((1, 1, 1))
        try:
            ax.dist = 8.5  # tighter camera distance -> larger, more prominent panels
        except AttributeError:
            pass  # newer Matplotlib versions may not expose .dist; box_aspect still applies

    # ---- Panel A: single best-responder subject ----
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    cal_mask = ~is_test_a
    ax_a.scatter(
        X_a_plot[cal_mask, 0], X_a_plot[cal_mask, 1], X_a_plot[cal_mask, 2],
        c="lightgrey", alpha=0.35, s=18, linewidths=0, label="D_cal (calibration only)",
    )
    test_mask = is_test_a
    for cls in (0, 1):
        cls_mask = test_mask & (y_a_combined == cls)
        if y_pred_for_test is not None:
            correct_mask = cls_mask.copy()
            correct_mask[test_mask] &= correct_a
            incorrect_mask = cls_mask & ~correct_mask
            ax_a.scatter(
                X_a_plot[correct_mask, 0], X_a_plot[correct_mask, 1], X_a_plot[correct_mask, 2],
                c=CLASS_COLORS[cls], alpha=0.85, s=55, marker="o",
                linewidths=0.4, edgecolors="dimgrey",
            )
            ax_a.scatter(
                X_a_plot[incorrect_mask, 0], X_a_plot[incorrect_mask, 1], X_a_plot[incorrect_mask, 2],
                c=CLASS_COLORS[cls], alpha=0.9, s=70, marker="x", linewidths=1.6,
            )
        else:
            ax_a.scatter(
                X_a_plot[cls_mask, 0], X_a_plot[cls_mask, 1], X_a_plot[cls_mask, 2],
                c=CLASS_COLORS[cls], alpha=0.85, s=55, marker="o",
                linewidths=0.4, edgecolors="dimgrey",
            )
    _style_3d_axes(ax_a)
    ax_a.set_xlabel("t-SNE Axis 1", labelpad=8, fontsize=10)
    ax_a.set_ylabel("t-SNE Axis 2", labelpad=8, fontsize=10)
    ax_a.set_zlabel("t-SNE Axis 3", labelpad=8, fontsize=10)
    ax_a.set_title(
        f"A. Best-Responder Example: sub-{best_subject}\n"
        f"(rank 1/{len(accs)}, held-out acc={best_acc*100:.1f}%; "
        f"group mean={group_mean*100:.1f}%±{group_std*100:.1f}%)",
        fontsize=11, fontweight="bold", pad=26,
    )
    legend_a_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[0], markeredgecolor="dimgrey", markersize=9, label="False Memory (correct)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[1], markeredgecolor="dimgrey", markersize=9, label="True Memory (correct)"),
        Line2D([0], [0], marker="x", color="black", markersize=9, label="Misclassified (D_test)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey", markersize=9, label="D_cal (context only)"),
    ]
    ax_a.legend(
        handles=legend_a_elems, loc="upper left", bbox_to_anchor=(0.0, 0.84),
        fontsize=7.5, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1",
    )

    sep_vec_a = (X_a_plot[test_mask & (y_a_combined == 1)].mean(axis=0) -
                 X_a_plot[test_mask & (y_a_combined == 0)].mean(axis=0))
    sep_norm_a = sep_vec_a / (np.linalg.norm(sep_vec_a) + 1e-9)
    azim_a = float(np.degrees(np.arctan2(sep_norm_a[1], sep_norm_a[0]))) + 30.0
    elev_a = float(np.degrees(np.arcsin(np.clip(sep_norm_a[2], -1.0, 1.0)))) + 20.0
    ax_a.view_init(elev=elev_a, azim=azim_a)

    # ---- Panel B: multi-subject, colored by TRUE CLASS ----
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    for cls in (0, 1):
        mask = y_bc == cls
        ax_b.scatter(
            X_bc_plot[mask, 0], X_bc_plot[mask, 1], X_bc_plot[mask, 2],
            c=CLASS_COLORS[cls], label=CLASS_LABELS[cls],
            alpha=0.75, s=40, linewidths=0.4, edgecolors="dimgrey",
        )
    _style_3d_axes(ax_b)
    sep_vec_bc = centroid1 - centroid0
    sep_norm_bc = sep_vec_bc / (np.linalg.norm(sep_vec_bc) + 1e-9)
    azim_bc = float(np.degrees(np.arctan2(sep_norm_bc[1], sep_norm_bc[0]))) + 30.0
    elev_bc = float(np.degrees(np.arcsin(np.clip(sep_norm_bc[2], -1.0, 1.0)))) + 20.0
    ax_b.view_init(elev=elev_bc, azim=azim_bc)
    ax_b.set_xlabel("t-SNE Axis 1", labelpad=8, fontsize=10)
    ax_b.set_ylabel("t-SNE Axis 2", labelpad=8, fontsize=10)
    ax_b.set_zlabel("t-SNE Axis 3", labelpad=8, fontsize=10)
    ax_b.set_title(
        f"B. Multi-Subject (N={N_TOTAL_POINTS}), Colored by Class\n"
        f"(label-informed axes; centroid gap={centroid_gap:.2f})",
        fontsize=11, fontweight="bold", pad=26,
    )
    ax_b.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.84),
        fontsize=8, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1",
    )

    # ---- Panel C: SAME coordinates, colored by SUBJECT ID ----
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")
    cmap = plt.get_cmap("turbo", n_islands_subjects)
    subj_to_color_idx = {s: i for i, s in enumerate(unique_subjects_bc)}
    color_idx_arr = np.array([subj_to_color_idx[s] for s in sub_bc])
    sc = ax_c.scatter(
        X_bc_plot[:, 0], X_bc_plot[:, 1], X_bc_plot[:, 2],
        c=color_idx_arr, cmap=cmap, alpha=0.85, s=40, linewidths=0.3, edgecolors="black",
    )
    _style_3d_axes(ax_c)
    ax_c.view_init(elev=elev_bc, azim=azim_bc)
    ax_c.set_xlabel("t-SNE Axis 1", labelpad=8, fontsize=10)
    ax_c.set_ylabel("t-SNE Axis 2", labelpad=8, fontsize=10)
    ax_c.set_zlabel("t-SNE Axis 3", labelpad=8, fontsize=10)
    ax_c.set_title(
        f"C. Same Coordinates, Colored by Subject ID\n"
        f"({n_islands_subjects} subjects -> geometry tracks identity, not class)",
        fontsize=11, fontweight="bold",
    )
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Subject (arbitrary index)", fontsize=8)
    cbar.set_ticks([])

    fig.suptitle(
        "Figure 3: Best-Case Single-Subject Separability vs. Population-Level Subject-Islands Effect",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "A: Unsupervised PCA axes (fit on 28 subjects only) + t-SNE layout; markers show real held-out prediction correctness.\n"
        "B/C: Label-informed LDA+PCA → t-SNE axes on pooled pre-calibration features, demonstrating geometry tracks subject identity prior to calibration.",
        ha="center", va="top", fontsize=9.5, fontweight="normal", color="#000000", wrap=True,
    )
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.14, top=0.85, wspace=0.02)

    log.info(f"Saving 3-panel PNG (600 DPI) -> {PNG_MAIN}")
    fig.savefig(PNG_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
    log.info(f"Saving 3-panel PDF -> {PDF_MAIN}")
    fig.savefig(PDF_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Standalone Panel A (supplementary / talks) ----
    fig_a = plt.figure(figsize=(10, 8.5))
    ax_a2 = fig_a.add_subplot(111, projection="3d")
    ax_a2.scatter(
        X_a_plot[cal_mask, 0], X_a_plot[cal_mask, 1], X_a_plot[cal_mask, 2],
        c="lightgrey", alpha=0.35, s=20, linewidths=0,
    )
    for cls in (0, 1):
        cls_mask = test_mask & (y_a_combined == cls)
        if y_pred_for_test is not None:
            correct_mask = cls_mask.copy()
            correct_mask[test_mask] &= correct_a
            incorrect_mask = cls_mask & ~correct_mask
            ax_a2.scatter(X_a_plot[correct_mask, 0], X_a_plot[correct_mask, 1], X_a_plot[correct_mask, 2],
                          c=CLASS_COLORS[cls], alpha=0.85, s=60, marker="o", linewidths=0.4, edgecolors="dimgrey",
                          label=f"{CLASS_LABELS[cls]} (correct)")
            ax_a2.scatter(X_a_plot[incorrect_mask, 0], X_a_plot[incorrect_mask, 1], X_a_plot[incorrect_mask, 2],
                          c=CLASS_COLORS[cls], alpha=0.9, s=75, marker="x", linewidths=1.6)
        else:
            ax_a2.scatter(X_a_plot[cls_mask, 0], X_a_plot[cls_mask, 1], X_a_plot[cls_mask, 2],
                          c=CLASS_COLORS[cls], alpha=0.85, s=60, marker="o", linewidths=0.4,
                          edgecolors="dimgrey", label=CLASS_LABELS[cls])
    _style_3d_axes(ax_a2)
    ax_a2.view_init(elev=elev_a, azim=azim_a)
    ax_a2.set_xlabel("t-SNE Axis 1", labelpad=10)
    ax_a2.set_ylabel("t-SNE Axis 2", labelpad=10)
    ax_a2.set_zlabel("t-SNE Axis 3", labelpad=10)
    ax_a2.set_title(
        f"Best-Responder Single-Subject Manifold: sub-{best_subject}\n"
        f"Held-out LOSO accuracy = {best_acc*100:.2f}% (rank 1/{len(accs)}; "
        f"group mean = {group_mean*100:.2f}%±{group_std*100:.2f}%)",
        fontsize=12, fontweight="bold", pad=26,
    )
    fig_a.text(0.5, 0.01,
        "Axes: PCA fit on the other 28 subjects only (no leakage); t-SNE used solely for 3D layout. "
        "Marker shape reflects real held-out prediction outcomes. Best-case example — not representative of the population mean.",
        ha="center", fontsize=9.5, fontweight="normal", color="#000000")
    ax_a2.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.84),
        fontsize=8, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1",
    )
    fig_a.tight_layout()
    log.info(f"Saving standalone Panel A PNG -> {PNG_A}")
    fig_a.savefig(PNG_A, dpi=600, bbox_inches="tight", facecolor="white")
    log.info(f"Saving standalone Panel A PDF -> {PDF_A}")
    fig_a.savefig(PDF_A, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig_a)

    # =========================================================================
    # SECTION 7: INTERACTIVE 3-SCENE PLOTLY HTML
    # =========================================================================
    log.info("\nRendering interactive 3-scene Plotly HTML...")

    fig_ly = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"A. sub-{best_subject} (best responder, {best_acc*100:.1f}%)",
            "B. Multi-Subject — Colored by Class",
            "C. Multi-Subject — Colored by Subject ID",
        ),
    )

    # --- Panel A traces ---
    fig_ly.add_trace(go.Scatter3d(
        x=X_a_plot[cal_mask, 0], y=X_a_plot[cal_mask, 1], z=X_a_plot[cal_mask, 2],
        mode="markers", name="D_cal (context)",
        marker=dict(size=3, color="lightgrey", opacity=0.4),
        hovertemplate="D_cal (calibration only)<extra></extra>",
    ), row=1, col=1)

    for cls in (0, 1):
        cls_mask = test_mask & (y_a_combined == cls)
        if y_pred_for_test is not None:
            correct_mask = cls_mask.copy()
            correct_mask[test_mask] &= correct_a
            incorrect_mask = cls_mask & ~correct_mask
        else:
            correct_mask = cls_mask
            incorrect_mask = np.zeros_like(cls_mask)

        fig_ly.add_trace(go.Scatter3d(
            x=X_a_plot[correct_mask, 0], y=X_a_plot[correct_mask, 1], z=X_a_plot[correct_mask, 2],
            mode="markers", name=f"{CLASS_LABELS[cls]} (correct)",
            marker=dict(size=5, color=CLASS_COLORS[cls], opacity=0.85, line=dict(width=0.4, color="dimgrey")),
            hovertemplate=f"<b>sub-{best_subject}</b><br><b>State:</b> {CLASS_LABELS[cls]}<br><b>Prediction:</b> correct<extra></extra>",
        ), row=1, col=1)

        if incorrect_mask.any():
            fig_ly.add_trace(go.Scatter3d(
                x=X_a_plot[incorrect_mask, 0], y=X_a_plot[incorrect_mask, 1], z=X_a_plot[incorrect_mask, 2],
                mode="markers", name=f"{CLASS_LABELS[cls]} (misclassified)",
                marker=dict(size=6, color=CLASS_COLORS[cls], opacity=0.9, symbol="x"),
                hovertemplate=f"<b>sub-{best_subject}</b><br><b>State:</b> {CLASS_LABELS[cls]}<br><b>Prediction:</b> INCORRECT<extra></extra>",
            ), row=1, col=1)

    # --- Panel B traces (colored by class) ---
    for cls in (0, 1):
        mask = y_bc == cls
        fig_ly.add_trace(go.Scatter3d(
            x=X_bc_plot[mask, 0], y=X_bc_plot[mask, 1], z=X_bc_plot[mask, 2],
            mode="markers", name=CLASS_LABELS[cls],
            marker=dict(size=4, color=CLASS_COLORS[cls], opacity=0.75, line=dict(width=0.3, color="dimgrey")),
            customdata=[f"sub-{s}" for s in sub_bc[mask]],
            hovertemplate="<b>Subject:</b> %{customdata}<br><b>State:</b> " + CLASS_LABELS[cls] + "<extra></extra>",
            legendgroup="panelB",
        ), row=1, col=2)

    # --- Panel C trace (colored by subject) ---
    fig_ly.add_trace(go.Scatter3d(
        x=X_bc_plot[:, 0], y=X_bc_plot[:, 1], z=X_bc_plot[:, 2],
        mode="markers", name="By Subject",
        marker=dict(
            size=4, color=color_idx_arr, colorscale="Turbo", opacity=0.85,
            line=dict(width=0.3, color="black"),
            colorbar=dict(title="Subject idx", x=1.02, len=0.5),
        ),
        customdata=[f"sub-{s}" for s in sub_bc],
        hovertemplate="<b>Subject:</b> %{customdata}<extra></extra>",
    ), row=1, col=3)

    fig_ly.update_layout(
        title=dict(
            text=(
                "Figure 3 (Interactive): Best-Case Single-Subject Separability vs. Subject-Islands Effect<br>"
                "<sup>Panel A axes are unsupervised (PCA fit on other 28 subjects); Panels B/C are label-informed (LDA-based) — see figure notes</sup>"
            ),
            x=0.5,
        ),
        scene=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"),
        scene2=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"),
        scene3=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"),
        template="plotly_white",
        width=1900, height=750,
        margin=dict(l=0, r=0, b=60, t=100),
        annotations=[
            dict(
                text=("A: unsupervised axes, real held-out predictions shown. "
                      "B/C: label-informed axes on pooled pre-calibration features — shown to demonstrate the subject-islands finding, not to claim raw separability."),
                showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.06,
                font=dict(size=10, color="dimgrey"),
            )
        ],
    )

    config = {"displayModeBar": True, "scrollZoom": True, "responsive": True}
    log.info(f"Saving interactive HTML -> {HTML_MAIN}")
    fig_ly.write_html(HTML_MAIN, config=config, include_plotlyjs="cdn", full_html=True)

    # =========================================================================
    # SECTION 8: PERSISTENCE + LOGGING
    # =========================================================================
    volume.commit()
    log.info("volume.commit() complete.")

    file_report = {}
    for label, path in [("PNG_MAIN", PNG_MAIN), ("PDF_MAIN", PDF_MAIN), ("HTML_MAIN", HTML_MAIN),
                         ("PNG_A", PNG_A), ("PDF_A", PDF_A)]:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0.0
        file_report[label] = {"path": path, "exists": exists, "size_kb": round(size_kb, 1)}
        status = "OK" if exists else "MISSING"
        log.info(f"  [{status}] {label:<10} -> {path}  ({size_kb:.1f} KB)")

    log.info("\n" + "=" * 70)
    log.info("  FIGURE 3 (v3, 3-PANEL) EXPORT COMPLETE")
    log.info("=" * 70)
    log.info(f"  Panel A subject       : sub-{best_subject} (best of {len(accs)}, acc={best_acc*100:.2f}%)")
    log.info(f"  Group mean accuracy   : {group_mean*100:.2f}% ± {group_std*100:.2f}%")
    log.info(f"  Panel B/C sample size : {N_TOTAL_POINTS} ({n_islands_subjects} subjects)")
    log.info(f"  Panel B centroid gap  : {centroid_gap:.3f} (label-informed; see disclosure)")
    log.info("=" * 70 + "\n")

    return {
        "best_subject": str(best_subject),
        "best_subject_acc": best_acc,
        "median_acc": median_acc,
        "worst_subject": str(worst_subject),
        "worst_subject_acc": worst_acc,
        "group_mean_acc": group_mean,
        "group_std_acc": group_std,
        "panel_bc_n_points": N_TOTAL_POINTS,
        "panel_bc_n_subjects": n_islands_subjects,
        "panel_bc_centroid_gap": float(centroid_gap),
        "files": file_report,
    }


# =============================================================================
# SECTION 9: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "=" * 70)
    print("  Figure 3 (v3) — 3-Panel Manifold: Best Responder + Subject Islands")
    print("=" * 70 + "\n")

    results = generate_figure3.remote()

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for key, val in results.items():
        print(f"  {key:<28} : {val}")
    print("=" * 70)
    print(
        "\n  Figure 3 (v3) export complete."
        "\n  3-panel PNG/PDF + interactive HTML + standalone Panel A committed to eeg-data-vol.\n"
    )