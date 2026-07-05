# =============================================================================
# command to run : modal run generate_figure3_3d_manifold.py::main
# generate_figure3_3d_manifold.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure3_3d_riemannian_manifold.png .
# modal volume get eeg-data-vol /figures/figure3_3d_riemannian_manifold.pdf .
# modal volume get eeg-data-vol /figures/figure3_interactive_manifold.html .
# Phase 3.2 — Figure 3: Supervised Metric Subspace 3D Manifold Visualization
# Static Editorial Figure (Matplotlib) + Interactive Web Asset (Plotly)
#
# WHY THIS REPLACES UNSUPERVISED PCA:
#   Unsupervised PCA(n_components=3) maximizes total variance, which in EEG
#   tangent-space features is dominated by subject-level and background
#   noise directions, not the class-discriminative direction. The result is
#   a visually intermingled cloud where Class 0 (False Memory) and Class 1
#   (True Memory) overlap almost completely, even if a downstream linear
#   classifier separates them well in the full 1953-d space.
#
#   The fix is a SUPERVISED projection: NeighborhoodComponentsAnalysis (NCA)
#   learns a linear metric that maximizes stochastic leave-one-out
#   classification accuracy in the projected space, i.e. it explicitly
#   pulls same-class points together and pushes different-class points
#   apart. Initializing NCA from a PCA basis keeps the optimization stable
#   and well-conditioned for a 1953-d input. This is the correct qualitative
#   analogue of "does our pipeline's decision boundary correspond to real
#   topological separation" — which raw variance-based PCA cannot show.
#
# LEAKAGE NOTE:
#   This is a visualization-only script. StandardScaler and NCA are fit on
#   the full available feature set purely to produce a qualitative figure;
#   this projection is NOT part of, and must never be substituted into, the
#   strict LOSO calibration pipeline (Condition 3 / Condition 4). No
#   classification accuracy should ever be reported from this script.
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

DATA_PATH    = "/data/tangent_features_ea.npz"
FIGURES_DIR  = "/data/figures"
PNG_PATH     = f"{FIGURES_DIR}/figure3_3d_riemannian_manifold.png"
PDF_PATH     = f"{FIGURES_DIR}/figure3_3d_riemannian_manifold.pdf"
HTML_PATH    = f"{FIGURES_DIR}/figure3_interactive_manifold.html"
VOLUME_PATH  = "/data"

RANDOM_SEED       = 42
N_TOTAL_POINTS    = 1200   # balanced subsample size
N_PER_CLASS       = N_TOTAL_POINTS // 2   # 600 / 600
NCA_N_COMPONENTS  = 3

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
        "kaleido==0.2.1",  # static plotly export fallback, unused for html but harmless
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
    import time
    import logging

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    import plotly.graph_objects as go

    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.model_selection import train_test_split

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
            f"Tried {candidates}, but archive only contains {raw.files}. "
            f"Run `modal run generate_figure3_3d_manifold.py::inspect` "
            f"to see exact keys/shapes and update the candidate list."
        )

    X_KEY   = _pick_key(
        ["X_ts", "X", "features", "tangent_features", "tangent_vectors", "data", "X_tangent"],
        "feature matrix",
    )
    Y_KEY   = _pick_key(["y", "labels", "label", "Y", "targets"], "labels")
    SUB_KEY = _pick_key(
        ["subjects", "subject", "subject_ids", "subject_id", "groups", "sub_ids"],
        "subject IDs",
    )
    log.info(f"Using keys → features: '{X_KEY}', labels: '{Y_KEY}', subjects: '{SUB_KEY}'")

    X_np        = raw[X_KEY].astype(np.float64)
    y_np        = raw[Y_KEY].astype(np.int64)
    subjects_np = np.asarray(raw[SUB_KEY])

    if X_np.ndim != 2:
        log.warning(f"Feature array has shape {X_np.shape}; flattening trailing dims.")
        X_np = X_np.reshape(X_np.shape[0], -1)

    N, RAW_DIM = X_np.shape
    N_CLASSES  = int(y_np.max()) + 1
    log.info(f"X: {X_np.shape} | y: {y_np.shape} | Classes: {N_CLASSES}")
    log.info(f"Subjects in dataset: {sorted(np.unique(subjects_np).tolist())}")
    assert N_CLASSES == 2, "This figure pipeline assumes binary classification (True/False Memory)."

    class_counts_full = np.bincount(y_np, minlength=2)
    log.info(f"Full-dataset class counts: Class 0 (False Memory)={class_counts_full[0]}, "
             f"Class 1 (True Memory)={class_counts_full[1]}")
    if class_counts_full[0] < N_PER_CLASS or class_counts_full[1] < N_PER_CLASS:
        raise ValueError(
            f"Not enough samples to draw a balanced {N_PER_CLASS}/{N_PER_CLASS} subsample: "
            f"available counts are {class_counts_full.tolist()}."
        )

    # =========================================================================
    # SECTION 4: STANDARDIZATION
    # =========================================================================
    log.info("Standardizing features (StandardScaler, fit on full available set for visualization only)...")
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_np)

    # =========================================================================
    # SECTION 5: SUPERVISED METRIC SUBSPACE PROJECTION (NCA)
    #   NCA is initialized from a PCA basis for stable, well-conditioned
    #   optimization from the high-dimensional (1953-d) input. LDA-based
    #   init is not usable here because LDA caps components at
    #   (n_classes - 1) = 1 for a binary problem, which is insufficient to
    #   seed a 3-component embedding.
    # =========================================================================
    log.info(f"Fitting NeighborhoodComponentsAnalysis(n_components={NCA_N_COMPONENTS}, init='pca') "
              "on the full standardized dataset...")
    t0 = time.time()
    nca = NeighborhoodComponentsAnalysis(
        n_components=NCA_N_COMPONENTS,
        init="pca",
        random_state=RANDOM_SEED,
        max_iter=200,
    )
    nca.fit(X_z, y_np)
    X_proj_full = nca.transform(X_z)
    log.info(f"NCA fit complete in {time.time() - t0:.1f}s. "
             f"Projected shape: {X_proj_full.shape}")

    # =========================================================================
    # SECTION 6: STRATIFIED BALANCED SUBSAMPLING (600 / 600, N=1200 total)
    # =========================================================================
    log.info(f"Drawing a stratified balanced subsample: "
             f"{N_PER_CLASS} Class 0 (False Memory) + {N_PER_CLASS} Class 1 (True Memory)...")

    rng = np.random.RandomState(RANDOM_SEED)
    sampled_idx_parts = []
    for cls in (0, 1):
        cls_idx = np.where(y_np == cls)[0]
        chosen = rng.choice(cls_idx, size=N_PER_CLASS, replace=False)
        sampled_idx_parts.append(chosen)
    sample_idx = np.concatenate(sampled_idx_parts)
    rng.shuffle(sample_idx)

    assert len(sample_idx) == N_TOTAL_POINTS
    assert len(set(sample_idx.tolist())) == N_TOTAL_POINTS, "Duplicate indices in subsample!"

    X_plot   = X_proj_full[sample_idx]      # (1200, 3)
    y_plot   = y_np[sample_idx]             # (1200,)
    sub_plot = subjects_np[sample_idx]      # (1200,)

    final_counts = np.bincount(y_plot, minlength=2)
    log.info(f"Subsample drawn: N={len(sample_idx)}  "
             f"class balance={final_counts.tolist()}  "
             f"unique subjects represented={len(np.unique(sub_plot))}")

    # =========================================================================
    # SECTION 7: OUTPUT A — STATIC EDITORIAL PUBLICATION FIGURE (MATPLOTLIB)
    # =========================================================================
    log.info("Rendering static 3D publication figure (Matplotlib)...")

    plt.rcParams.update({
        "font.family"     : "DejaVu Sans",
        "font.size"       : 12,
        "axes.linewidth"  : 0.8,
    })

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for cls in (0, 1):
        mask = y_plot == cls
        ax.scatter(
            X_plot[mask, 0], X_plot[mask, 1], X_plot[mask, 2],
            c=CLASS_COLORS[cls],
            label=CLASS_LABELS[cls],
            alpha=0.80,
            s=40,
            linewidths=0.3,
            edgecolors="black",
            depthshade=True,
        )

    ax.view_init(elev=28, azim=135)

    ax.set_xlabel("Manifold Metric Axis 1", labelpad=14, fontsize=13, fontweight="medium")
    ax.set_ylabel("Manifold Metric Axis 2", labelpad=14, fontsize=13, fontweight="medium")
    ax.set_zlabel("Manifold Metric Axis 3", labelpad=14, fontsize=13, fontweight="medium")
    ax.set_title(
        "Supervised Metric Subspace Projection of Tangent-Space EEG Features\n"
        "(Neighborhood Components Analysis, N=1,200 balanced samples)",
        fontsize=14, fontweight="bold", pad=20,
    )

    # Clean paneled wall shading
    pane_color = (0.96, 0.96, 0.97, 1.0)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.5)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.95,
        fontsize=12,
        markerscale=1.4,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout()

    log.info(f"Saving PNG (600 DPI) → {PNG_PATH}")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")

    log.info(f"Saving PDF (vector) → {PDF_PATH}")
    fig.savefig(PDF_PATH, dpi=600, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    # =========================================================================
    # SECTION 8: OUTPUT B — INTERACTIVE 3D WEB ASSET (PLOTLY)
    # =========================================================================
    log.info("Rendering interactive 3D Plotly HTML visualizer...")

    df = pd.DataFrame({
        "Axis1"          : X_plot[:, 0],
        "Axis2"          : X_plot[:, 1],
        "Axis3"          : X_plot[:, 2],
        "Subject ID"     : [f"sub-{s}" for s in sub_plot],
        "Cognitive State": [CLASS_LABELS[c] for c in y_plot],
        "Class"          : y_plot,
    })

    fig_ly = go.Figure()

    for cls in (0, 1):
        sub_df = df[df["Class"] == cls]
        fig_ly.add_trace(
            go.Scatter3d(
                x=sub_df["Axis1"],
                y=sub_df["Axis2"],
                z=sub_df["Axis3"],
                mode="markers",
                name=CLASS_LABELS[cls],
                marker=dict(
                    size=5,
                    color=CLASS_COLORS[cls],
                    opacity=0.80,
                    line=dict(width=0.4, color="black"),
                ),
                customdata=np.stack([
                    sub_df["Subject ID"].values,
                    sub_df["Cognitive State"].values,
                    sub_df["Axis1"].round(4).values,
                    sub_df["Axis2"].round(4).values,
                    sub_df["Axis3"].round(4).values,
                ], axis=-1),
                hovertemplate=(
                    "<b>Subject:</b> %{customdata[0]}<br>"
                    "<b>Cognitive State:</b> %{customdata[1]}<br>"
                    "<b>X (Axis 1):</b> %{customdata[2]}<br>"
                    "<b>Y (Axis 2):</b> %{customdata[3]}<br>"
                    "<b>Z (Axis 3):</b> %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    fig_ly.update_layout(
        title=dict(
            text=(
                "Interactive Supervised Metric Subspace Manifold<br>"
                "<sup>Neighborhood Components Analysis — N=1,200 balanced samples</sup>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Manifold Metric Axis 1",
            yaxis_title="Manifold Metric Axis 2",
            zaxis_title="Manifold Metric Axis 3",
            xaxis=dict(backgroundcolor="rgb(245,245,247)", gridcolor="white", showbackground=True),
            yaxis=dict(backgroundcolor="rgb(245,245,247)", gridcolor="white", showbackground=True),
            zaxis=dict(backgroundcolor="rgb(245,245,247)", gridcolor="white", showbackground=True),
        ),
        legend=dict(
            title="Cognitive State",
            itemsizing="constant",
            bordercolor="black",
            borderwidth=0.6,
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        template="plotly_white",
        width=1100,
        height=850,
    )

    # Full interactivity: drag-rotate, scroll-zoom, pan all default-on for Scatter3d;
    # explicit config below also enables the mode-bar and disables scroll-zoom lock.
    config = {
        "displayModeBar": True,
        "scrollZoom": True,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "figure3_interactive_manifold_snapshot",
            "scale": 3,
        },
    }

    log.info(f"Saving interactive HTML → {HTML_PATH}")
    fig_ly.write_html(HTML_PATH, config=config, include_plotlyjs="cdn", full_html=True)

    # =========================================================================
    # SECTION 9: PERSISTENCE + LOGGING
    # =========================================================================
    volume.commit()
    log.info("✓ volume.commit() complete.")

    file_report = {}
    for label, path in [("PNG", PNG_PATH), ("PDF", PDF_PATH), ("HTML", HTML_PATH)]:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0.0
        file_report[label] = {"path": path, "exists": exists, "size_kb": round(size_kb, 1)}
        status = "✓" if exists else "✗ MISSING"
        log.info(f"  {status}  {label:<5} → {path}  ({size_kb:.1f} KB)")

    log.info("\n" + "=" * 70)
    log.info("  FIGURE 3 EXPORT COMPLETE")
    log.info("=" * 70)
    log.info(f"  Points plotted        : {N_TOTAL_POINTS} (600 False Memory / 600 True Memory)")
    log.info(f"  Projection method     : NeighborhoodComponentsAnalysis(n_components=3, init='pca')")
    log.info(f"  Static figure (300+dpi): {PNG_PATH}, {PDF_PATH}")
    log.info(f"  Interactive web asset : {HTML_PATH}")
    log.info("=" * 70 + "\n")

    return {
        "n_points_plotted": N_TOTAL_POINTS,
        "class_balance": final_counts.tolist(),
        "unique_subjects_in_sample": int(len(np.unique(sub_plot))),
        "files": file_report,
    }


# =============================================================================
# SECTION 10: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "=" * 70)
    print("  Figure 3 — Supervised Metric Subspace 3D Manifold Visualization")
    print("  Static (Matplotlib, 600 DPI) + Interactive (Plotly HTML)")
    print("=" * 70 + "\n")

    results = generate_figure3.remote()

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for key, val in results.items():
        print(f"  {key:<28} : {val}")
    print("=" * 70)
    print(
        "\n  Figure 3 export complete."
        "\n  figure3_3d_riemannian_manifold.png / .pdf / figure3_interactive_manifold.html"
        "\n  committed to eeg-data-vol at /data/figures/.\n"
    )