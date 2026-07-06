# =============================================================================
# commands to run :
# modal deploy generate_figure4_3d_brain_topography.py
# modal run generate_figure4_3d_brain_topography.py::main
# generate_figure4_3d_brain_topography.py
# commands to download:
# modal volume get eeg-data-vol /figures/figure4_cortical_topography.png .
# modal volume get eeg-data-vol /figures/figure4_cortical_topography.pdf .
# modal volume get eeg-data-vol /figures/figure4_interactive_topography.html .
#
# Phase 3.3 — Figure 4: Cortical Scalp Topography & Discriminative Spatial Map
#   Panel A — True Memory  (Class 1) scalp topography of spatial log-power
#   Panel B — False Memory (Class 0) scalp topography of spatial log-power
#   Panel C — Discriminative difference topography (Class 1 - Class 0),
#             annotated with the top-3 most discriminative electrodes
#             (selected dynamically from the data — never hardcoded)
#
#   PLUS an interactive 3D Plotly scalp asset (figure4_interactive_topography.html)
#   with the same three panels rendered as rotatable 3D scenes.
#
# ⚠️ SCIENTIFIC DISCLOSURE #1 — WHAT "SPATIAL POWER" MEANS HERE:
#   tangent_features_ea.npz stores EUCLIDEAN-ALIGNED TANGENT-SPACE feature
#   vectors, not raw per-channel band power. Each trial's tangent vector is
#   the upper-triangular (Pennec-style, off-diagonal terms scaled by sqrt(2))
#   vectorization of that trial's log-mapped spatial covariance matrix at the
#   post-EA reference point. The DIAGONAL entries of that vectorization are
#   untouched by the sqrt(2) off-diagonal scaling, so they can be read
#   directly out of the tangent vector at known index positions (see
#   `_diagonal_positions()` below) without needing to reconstruct the full
#   NxN matrix. Those diagonal entries are each channel's LOG-POWER RELATIVE
#   TO THE EA REFERENCE MATRIX (approximately log-variance) — this is the
#   closest honest, non-fabricated proxy for "per-channel spatial power"
#   available from this feature representation.
#
#   If the archive instead contains a genuine raw-covariance array (a key
#   containing "cov" with shape (n_trials, n_ch, n_ch)), this script uses
#   its true diagonal directly and skips the tangent-space approximation
#   entirely — the console log states explicitly which path was taken.
#
#   This is NOT raw sensor-space band power: no re-referencing to pre-EA
#   covariance and no frequency-band decomposition happens here. Report
#   figure values as "spatial log-power (post-EA, tangent-space diagonal)",
#   never as raw microvolt-squared power.
#
# ⚠️ SCIENTIFIC DISCLOSURE #2 — ELECTRODE LAYOUT:
#   If the .npz contains a channel-name array (a key containing "ch"), those
#   names are used and matched, in order, to the tangent-vector columns. If
#   no channel names are stored, this script falls back to a documented,
#   published 10-10-style 62-channel layout (ROW_LAYOUT below) and ASSUMES
#   the tangent-vector column order matches that same anterior-to-posterior,
#   left-to-right channel order. That assumption is logged explicitly at
#   runtime and should be verified against the upstream feature-extraction
#   script before citing exact electrode names (e.g. "Fz", "AF3") in the
#   manuscript text — the topographic PATTERN is trustworthy either way
#   (it is derived from the real per-column statistics), but the specific
#   NAME attached to a given cortical location depends on this assumption.
#   Electrode (x, y) coordinates are a standard-system polar approximation
#   (anterior-posterior row + left-right symmetric spacing constrained to
#   the unit head circle), not digitized 3-D positions from this dataset's
#   own recording montage. The 3D (x, y, z) positions used for the Plotly
#   asset are obtained by projecting these same 2D coordinates onto the
#   upper hemisphere of a unit sphere (z = sqrt(1 - x^2 - y^2)) — a standard,
#   commonly used approximation for turning a 10-10 topomap layout into a
#   3D scalp render. It is NOT a digitized 3D montage either.
#
# LEAKAGE NOTE:
#   This is a visualization-only, descriptive-statistics script. It computes
#   per-channel means/variances from pooled features and involves no
#   classifier fitting, no cross-validation, and no reuse of any held-out
#   test split from the Condition 3/4 pipeline. Nothing here should be
#   substituted into, or reported as, classification performance.
#
# Usage: modal run generate_figure4_3d_brain_topography.py::main
#        modal run generate_figure4_3d_brain_topography.py::inspect
# =============================================================================

import modal

# =============================================================================
# SECTION 1: MODAL INFRASTRUCTURE
# =============================================================================

app    = modal.App("bci-figure4-brain-topography")
volume = modal.Volume.from_name("eeg-data-vol")

DATA_PATH   = "/data/tangent_features_ea.npz"
FIGURES_DIR = "/data/figures"

PNG_MAIN  = f"{FIGURES_DIR}/figure4_cortical_topography.png"
PDF_MAIN  = f"{FIGURES_DIR}/figure4_cortical_topography.pdf"
HTML_3D   = f"{FIGURES_DIR}/figure4_interactive_topography.html"

VOLUME_PATH = "/data"
RANDOM_SEED = 42

CLASS_LABELS = {0: "False Memory", 1: "True Memory"}

# --- Published, documented 62-channel 10-10-style row layout -----------------
# Rows ordered anterior -> posterior. Each row lists channel names left ->
# right exactly as they would appear scalp-view (nose at top, y=+1).
ROW_LAYOUT = [
    (0.92,  ["Fp1", "Fpz", "Fp2"]),
    (0.74,  ["AF7", "AF3", "AFz", "AF4", "AF8"]),
    (0.55,  ["F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"]),
    (0.34,  ["FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8"]),
    (0.12,  ["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"]),
    (-0.10, ["TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8"]),
    (-0.32, ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]),
    (-0.55, ["PO7", "PO3", "POz", "PO4", "PO8"]),
    (-0.75, ["O1", "Oz", "O2", "Iz"]),
]
N_CHANNELS_DEFAULT = sum(len(names) for _, names in ROW_LAYOUT)  # 62

viz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "scipy",
        "matplotlib==3.8.4",
        "pandas",
        "plotly==5.22.0",
    )
)


# =============================================================================
# SECTION 1.5: STANDALONE INSPECTOR
#   modal run generate_figure4_3d_brain_topography.py::inspect
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
    timeout=1800,
    memory=8192,
)
def generate_figure4():

    import os
    import logging

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Circle, Ellipse
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import griddata

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("figure4-topography")

    np.random.seed(RANDOM_SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =========================================================================
    # SECTION 3: ELECTRODE LAYOUT CONSTRUCTION
    # =========================================================================
    def build_default_positions():
        """Assign (x, y) in a unit head circle for each row, symmetric about
        the midline, width constrained so outer electrodes sit near — but
        inside — the head boundary (circle of radius 1)."""
        names, xs, ys = [], [], []
        for y, row_names in ROW_LAYOUT:
            n = len(row_names)
            max_half_width = np.sqrt(max(1e-6, 1.0 - y ** 2)) * 0.95
            if n == 1:
                row_x = [0.0]
            else:
                row_x = list(np.linspace(-max_half_width, max_half_width, n))
            for name, x in zip(row_names, row_x):
                names.append(name)
                xs.append(x)
                ys.append(y)
        return names, np.array(xs), np.array(ys)

    default_names, default_x, default_y = build_default_positions()
    default_pos = {n: (x, y) for n, x, y in zip(default_names, default_x, default_y)}
    log.info(f"Default documented 62-channel layout built: {len(default_names)} channels.")

    # =========================================================================
    # SECTION 4: DATA LOADING
    # =========================================================================
    log.info(f"Loading {DATA_PATH} ...")
    raw = np.load(DATA_PATH, allow_pickle=True)
    log.info(f"Archive keys found: {raw.files}")

    def _pick_key(candidates, purpose, required=True):
        for c in candidates:
            if c in raw.files:
                return c
        if required:
            raise KeyError(
                f"Could not find a key for '{purpose}' in {DATA_PATH}. "
                f"Tried {candidates}, but archive only contains {raw.files}."
            )
        return None

    X_KEY   = _pick_key(["X_ts", "X", "features", "tangent_features", "tangent_vectors", "data", "X_tangent"], "feature matrix")
    Y_KEY   = _pick_key(["y", "labels", "label", "Y", "targets"], "labels")
    COV_KEY = _pick_key(["covariances", "covariance", "cov", "X_cov"], "raw covariance", required=False)
    CH_KEY  = _pick_key(["channels", "ch_names", "channel_names", "chs"], "channel names", required=False)

    log.info(f"Using keys -> features: '{X_KEY}', labels: '{Y_KEY}', "
             f"covariance: '{COV_KEY}', channel names: '{CH_KEY}'")

    y_np = raw[Y_KEY].astype(np.int64)
    N_CLASSES = int(y_np.max()) + 1
    assert N_CLASSES == 2, "This figure pipeline assumes binary classification (True/False Memory)."

    # =========================================================================
    # SECTION 5: PER-CHANNEL SPATIAL LOG-POWER RECONSTRUCTION
    # =========================================================================
    if COV_KEY is not None and raw[COV_KEY].ndim == 3:
        log.info(f"Raw covariance array '{COV_KEY}' found — using its TRUE diagonal "
                  f"directly. No tangent-space approximation needed.")
        cov = raw[COV_KEY].astype(np.float64)
        n_channels = cov.shape[1]
        channel_power = np.diagonal(cov, axis1=1, axis2=2)  # (n_trials, n_channels)
        used_path = "raw_covariance_diagonal"
    else:
        log.info("No raw covariance array found — reconstructing per-channel spatial "
                  "log-power from the tangent-space feature vectors (see Disclosure #1).")
        X_np = raw[X_KEY].astype(np.float64)
        if X_np.ndim != 2:
            X_np = X_np.reshape(X_np.shape[0], -1)
        L = X_np.shape[1]

        # Solve n(n+1)/2 = L for n (upper-triangular vectorization length).
        n_channels = int(round((-1 + np.sqrt(1 + 8 * L)) / 2))
        if n_channels * (n_channels + 1) // 2 != L:
            raise ValueError(
                f"Feature vector length {L} is not a valid triangular number "
                f"(n(n+1)/2); cannot infer channel count. Check '{X_KEY}' shape."
            )
        log.info(f"Inferred n_channels={n_channels} from tangent-vector length L={L}.")

        # Diagonal entry (i, i) is the FIRST element of row i in a row-major
        # np.triu_indices ordering (matches the standard Pennec/pyriemann
        # tangent_space vectorization convention): its flat index is
        # i*n - i*(i-1)/2.
        diag_positions = np.array(
            [i * n_channels - i * (i - 1) // 2 for i in range(n_channels)], dtype=int
        )
        channel_power = X_np[:, diag_positions]  # (n_trials, n_channels)
        used_path = "tangent_space_diagonal_proxy"

    n_trials = channel_power.shape[0]
    assert len(y_np) == n_trials, "Label count does not match reconstructed channel-power trial count."
    log.info(f"Reconstructed channel_power: shape={channel_power.shape} via '{used_path}'.")

    # =========================================================================
    # SECTION 6: CHANNEL NAMES + POSITIONS
    # =========================================================================
    if CH_KEY is not None:
        ch_names = [str(c) for c in raw[CH_KEY]]
        if len(ch_names) != n_channels:
            log.warning(
                f"Stored channel-name array has length {len(ch_names)} but "
                f"reconstructed channel_power has {n_channels} columns. "
                f"Falling back to generic channel names — do NOT trust "
                f"electrode names in the resulting figure until this is resolved."
            )
            ch_names = [f"Ch{i+1}" for i in range(n_channels)]
            name_source = "generic_fallback_length_mismatch"
        else:
            name_source = f"npz_key:'{CH_KEY}'"
    elif n_channels == N_CHANNELS_DEFAULT:
        ch_names = default_names
        name_source = "documented_default_62ch_layout (column-order ASSUMED, see Disclosure #2)"
    else:
        ch_names = [f"Ch{i+1}" for i in range(n_channels)]
        name_source = "generic_fallback_channel_count_mismatch"
        log.warning(
            f"Reconstructed channel count ({n_channels}) does not match the "
            f"documented default layout ({N_CHANNELS_DEFAULT}). Using generic "
            f"channel labels; electrode-name annotations will read 'Ch#' "
            f"rather than 10-10 names."
        )
    log.info(f"Channel-name source: {name_source}")

    if all(name in default_pos for name in ch_names):
        pos_x = np.array([default_pos[name][0] for name in ch_names])
        pos_y = np.array([default_pos[name][1] for name in ch_names])
    else:
        # Generic channels: distribute on a simple spiral inside the unit
        # circle so the topomap still renders sensibly.
        log.warning("Channel names do not all match the documented layout — "
                     "placing electrodes on a generic spiral instead of named 10-10 sites.")
        idx = np.arange(n_channels)
        golden_angle = np.pi * (3 - np.sqrt(5))
        r = np.sqrt((idx + 0.5) / n_channels) * 0.95
        theta = idx * golden_angle
        pos_x = r * np.cos(theta)
        pos_y = r * np.sin(theta)

    # =========================================================================
    # SECTION 7: PER-CLASS SPATIAL POWER + DISCRIMINATIVE CONTRAST
    # =========================================================================
    mask1 = y_np == 1
    mask0 = y_np == 0
    log.info(f"Class counts -> True Memory (1): {mask1.sum()} trials | "
              f"False Memory (0): {mask0.sum()} trials")

    power1_mean = channel_power[mask1].mean(axis=0)
    power0_mean = channel_power[mask0].mean(axis=0)
    power1_var  = channel_power[mask1].var(axis=0)
    power0_var  = channel_power[mask0].var(axis=0)

    diff = power1_mean - power0_mean
    fisher = (power1_mean - power0_mean) ** 2 / (power1_var + power0_var + 1e-12)

    order_diff = np.argsort(-np.abs(diff))
    top5_idx = order_diff[:5]
    top3_idx = order_diff[:3]

    log.info("Top-5 most discriminative electrodes (by |Class1 - Class0| spatial log-power):")
    for rank, i in enumerate(top5_idx, start=1):
        log.info(f"  #{rank}: {ch_names[i]:<6} diff={diff[i]:+.4f}  "
                  f"(True={power1_mean[i]:.4f}, False={power0_mean[i]:.4f}, "
                  f"Fisher={fisher[i]:.4f})")

    # =========================================================================
    # SECTION 8: TOPOMAP INTERPOLATION HELPER (2D Matplotlib)
    # =========================================================================
    grid_n = 300
    gx, gy = np.meshgrid(
        np.linspace(-1.05, 1.05, grid_n),
        np.linspace(-1.05, 1.05, grid_n),
    )
    head_mask = (gx ** 2 + gy ** 2) <= 1.0

    def interpolate_topomap(values):
        grid_z = griddata(
            (pos_x, pos_y), values, (gx, gy), method="cubic"
        )
        grid_z_nearest = griddata(
            (pos_x, pos_y), values, (gx, gy), method="nearest"
        )
        grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
        grid_z = np.where(head_mask, grid_z, np.nan)
        return grid_z

    def draw_head_outline(ax):
        ax.add_patch(Circle((0, 0), 1.0, fill=False, linewidth=1.6, edgecolor="black", zorder=5))
        nose = plt.Polygon(
            [(-0.09, 0.99), (0.0, 1.14), (0.09, 0.99)],
            closed=True, fill=False, linewidth=1.6, edgecolor="black", zorder=5,
        )
        ax.add_patch(nose)
        for side in (-1, 1):
            ear = Ellipse(
                (side * 1.02, 0.0), width=0.09, height=0.28,
                fill=False, linewidth=1.6, edgecolor="black", zorder=5,
            )
            ax.add_patch(ear)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        ax.axis("off")

    def render_panel(ax, values, cmap, vmin, vmax, title, n_contours=6, annotate_idx=None):
        grid_z = interpolate_topomap(values)
        im = ax.contourf(
            gx, gy, grid_z, levels=np.linspace(vmin, vmax, 40),
            cmap=cmap, vmin=vmin, vmax=vmax, extend="both", zorder=1,
        )
        ax.contour(
            gx, gy, grid_z, levels=n_contours, colors="black",
            linewidths=0.4, alpha=0.5, zorder=2,
        )
        ax.scatter(
            pos_x, pos_y, s=16, facecolor="white", edgecolor="black",
            linewidths=0.6, zorder=6,
        )
        draw_head_outline(ax)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=14)

        if annotate_idx is not None:
            for i in annotate_idx:
                ax.annotate(
                    ch_names[i],
                    xy=(pos_x[i], pos_y[i]),
                    xytext=(pos_x[i] + 0.28 * np.sign(pos_x[i] if pos_x[i] != 0 else 1),
                            pos_y[i] + 0.22),
                    fontsize=9.5, fontweight="bold", color="black", zorder=7,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
                    ha="center",
                )
        return im

    # =========================================================================
    # SECTION 9: FIGURE ASSEMBLY (3-PANEL, Q1 EDITORIAL LAYOUT)
    # =========================================================================
    log.info("\nRendering 3-panel static publication figure (Matplotlib)...")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    shared_vmin = float(min(power1_mean.min(), power0_mean.min()))
    shared_vmax = float(max(power1_mean.max(), power0_mean.max()))
    diff_abs_max = float(np.max(np.abs(diff)))

    im_a = render_panel(
        axes[0], power1_mean, cmap="viridis", vmin=shared_vmin, vmax=shared_vmax,
        title="A. True Memory (Class 1)\nSpatial Log-Power",
    )
    im_b = render_panel(
        axes[1], power0_mean, cmap="viridis", vmin=shared_vmin, vmax=shared_vmax,
        title="B. False Memory (Class 0)\nSpatial Log-Power",
    )
    im_c = render_panel(
        axes[2], diff, cmap="RdBu_r", vmin=-diff_abs_max, vmax=diff_abs_max,
        title="C. Discriminative Difference\n(True \u2212 False Memory)",
        annotate_idx=top3_idx,
    )

    # --- SURGICAL FIX: strict 4-tick, non-overlapping colorbar formatting ---
    colorbar_specs = [
        (axes[0], im_a, shared_vmin, shared_vmax, "log-power (a.u.)"),
        (axes[1], im_b, shared_vmin, shared_vmax, "log-power (a.u.)"),
        (axes[2], im_c, -diff_abs_max, diff_abs_max, "\u0394 log-power (True \u2212 False)"),
    ]
    for ax, im, cb_vmin, cb_vmax, cb_label in colorbar_specs:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.35)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

        # Exactly 4 evenly spaced ticks spanning [vmin, vmax], each rounded
        # to 3 decimal places -> guarantees fixed-width, non-colliding labels.
        tick_values = np.round(np.linspace(cb_vmin, cb_vmax, 4), 3)
        cbar.set_ticks(tick_values)
        cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        cbar.ax.tick_params(labelsize=8.5, pad=2)
        cbar.ax.set_xlabel(cb_label, fontsize=8.5, labelpad=4)

    fig.suptitle(
        "Figure 4: Spatial Cortical Topography and Discriminative Electrode Map across 62 Channels",
        fontsize=14.5, fontweight="bold", y=1.03,
    )
    fig.text(
        0.5, 0.965,
        f"Reconstruction path: {used_path}  |  Channel-name source: {name_source}",
        ha="center", va="top", fontsize=9.5, color="#333333",
    )
    fig.text(
        0.5, -0.03,
        "Values are per-channel spatial log-power relative to the post-EA tangent-space reference (not raw \u03bcV\u00b2 band power).\n"
        "Panel C annotates the top-3 most discriminative electrodes by |Class 1 \u2212 Class 0| difference, selected dynamically from the data.",
        ha="center", va="top", fontsize=9, fontweight="normal", color="#000000",
    )

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.14, top=0.86, wspace=0.25)

    log.info(f"Saving 3-panel PNG (600 DPI) -> {PNG_MAIN}")
    fig.savefig(PNG_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
    log.info(f"Saving 3-panel PDF -> {PDF_MAIN}")
    fig.savefig(PDF_MAIN, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # =========================================================================
    # SECTION 9.5: INTERACTIVE 3D PLOTLY SCALP ASSET
    # =========================================================================
    log.info("\nBuilding interactive 3D Plotly scalp topography (HTML)...")

    # --- 3D electrode positions: project the same 2D 10-10 layout onto the
    # upper hemisphere of a unit sphere. z = sqrt(1 - x^2 - y^2). This is a
    # standard topomap -> 3D-scalp approximation, NOT a digitized montage
    # (see Disclosure #2). ---
    r2 = pos_x ** 2 + pos_y ** 2
    r2_clipped = np.clip(r2, 0.0, 0.999)
    elec_x = pos_x
    elec_y = pos_y
    elec_z = np.sqrt(1.0 - r2_clipped)

    # --- 3D head surface: parametric hemisphere (unit sphere, upper half) ---
    n_u, n_v = 48, 24
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi / 2, n_v)  # v=0 -> vertex (top of head), v=pi/2 -> equator
    uu, vv = np.meshgrid(u, v)
    head_x = np.cos(uu) * np.sin(vv)
    head_y = np.sin(uu) * np.sin(vv)
    head_z = np.cos(vv)

    def make_head_surface():
        return go.Surface(
            x=head_x, y=head_y, z=head_z,
            colorscale=[[0, "rgb(235,225,210)"], [1, "rgb(235,225,210)"]],
            showscale=False, opacity=0.18,
            hoverinfo="skip", name="scalp",
        )

    def hover_text(idx_array, power_vals, diff_vals, fisher_vals):
        texts = []
        for i in idx_array:
            texts.append(
                f"<b>{ch_names[i]}</b><br>"
                f"Log-power: {power_vals[i]:.4f}<br>"
                f"\u0394 (True\u2212False): {diff_vals[i]:+.4f}<br>"
                f"Fisher score: {fisher_vals[i]:.4f}"
            )
        return texts

    all_idx = np.arange(n_channels)

    fig3d = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "True Memory (Class 1)<br>Spatial Log-Power",
            "False Memory (Class 0)<br>Spatial Log-Power",
            "Discriminative Difference<br>(True \u2212 False Memory)",
        ),
        horizontal_spacing=0.02,
    )

    # --- Scene 1: True Memory ---
    fig3d.add_trace(make_head_surface(), row=1, col=1)
    fig3d.add_trace(
        go.Scatter3d(
            x=elec_x, y=elec_y, z=elec_z, mode="markers",
            marker=dict(
                size=6, color=power1_mean, colorscale="Viridis",
                cmin=shared_vmin, cmax=shared_vmax,
                colorbar=dict(title="log-power", x=0.28, len=0.6, thickness=14),
                line=dict(color="black", width=0.5),
            ),
            text=hover_text(all_idx, power1_mean, diff, fisher),
            hoverinfo="text", name="True Memory",
        ),
        row=1, col=1,
    )

    # --- Scene 2: False Memory ---
    fig3d.add_trace(make_head_surface(), row=1, col=2)
    fig3d.add_trace(
        go.Scatter3d(
            x=elec_x, y=elec_y, z=elec_z, mode="markers",
            marker=dict(
                size=6, color=power0_mean, colorscale="Viridis",
                cmin=shared_vmin, cmax=shared_vmax,
                colorbar=dict(title="log-power", x=0.635, len=0.6, thickness=14),
                line=dict(color="black", width=0.5),
            ),
            text=hover_text(all_idx, power0_mean, diff, fisher),
            hoverinfo="text", name="False Memory",
        ),
        row=1, col=2,
    )

    # --- Scene 3: Discriminative Difference (size + color dynamic) ---
    fisher_norm = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-12)
    marker_sizes = 5 + fisher_norm * 14  # dynamic size range ~5-19

    fig3d.add_trace(make_head_surface(), row=1, col=3)
    fig3d.add_trace(
        go.Scatter3d(
            x=elec_x, y=elec_y, z=elec_z, mode="markers",
            marker=dict(
                size=marker_sizes, color=diff, colorscale="RdBu_r",
                cmin=-diff_abs_max, cmax=diff_abs_max,
                colorbar=dict(title="\u0394 log-power", x=0.99, len=0.6, thickness=14),
                line=dict(color="black", width=0.5),
            ),
            text=hover_text(all_idx, power1_mean, diff, fisher),
            hoverinfo="text", name="Difference",
        ),
        row=1, col=3,
    )

    # Highlight + label the top-3 discriminative electrodes in Scene 3
    fig3d.add_trace(
        go.Scatter3d(
            x=elec_x[top3_idx], y=elec_y[top3_idx], z=elec_z[top3_idx] + 0.03,
            mode="markers+text",
            marker=dict(size=marker_sizes[top3_idx] + 4, color="rgba(0,0,0,0)",
                        line=dict(color="black", width=2.5)),
            text=[ch_names[i] for i in top3_idx],
            textposition="top center",
            textfont=dict(size=13, color="black", family="Arial Black"),
            hoverinfo="text",
            hovertext=hover_text(top3_idx, power1_mean, diff, fisher),
            name="Top-3 discriminative",
            showlegend=False,
        ),
        row=1, col=3,
    )

    scene_axis_common = dict(
        visible=False, showbackground=False, showticklabels=False,
        showspikes=False, range=[-1.2, 1.2],
    )
    scene_common = dict(
        xaxis=scene_axis_common, yaxis=scene_axis_common, zaxis=scene_axis_common,
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        aspectmode="cube",
    )
    fig3d.update_layout(
        scene=scene_common, scene2=scene_common, scene3=scene_common,
        title=dict(
            text="Figure 4 (Interactive): 3D Cortical Scalp Topography and Discriminative Electrode Map",
            x=0.5, xanchor="center", font=dict(size=18),
        ),
        annotations=[
            dict(
                text=(
                    f"Reconstruction path: {used_path} | Channel-name source: {name_source}<br>"
                    "3D electrode positions are a spherical projection of the 10-10 topomap layout "
                    "(not a digitized montage). Drag to rotate each panel; hover markers for values."
                ),
                x=0.5, y=1.06, xref="paper", yref="paper",
                showarrow=False, font=dict(size=11, color="#333333"),
            )
        ],
        margin=dict(l=10, r=10, t=110, b=10),
        height=650, width=1650,
        paper_bgcolor="white",
    )

    log.info(f"Saving interactive 3D HTML -> {HTML_3D}")
    fig3d.write_html(HTML_3D, include_plotlyjs="cdn", full_html=True)

    # =========================================================================
    # SECTION 10: PERSISTENCE + LOGGING
    # =========================================================================
    volume.commit()
    log.info("volume.commit() complete.")

    file_report = {}
    for label, path in [("PNG_MAIN", PNG_MAIN), ("PDF_MAIN", PDF_MAIN), ("HTML_3D", HTML_3D)]:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0.0
        file_report[label] = {"path": path, "exists": exists, "size_kb": round(size_kb, 1)}
        status = "OK" if exists else "MISSING"
        log.info(f"  [{status}] {label:<10} -> {path}  ({size_kb:.1f} KB)")

    log.info("\n" + "=" * 70)
    log.info("  FIGURE 4 (CORTICAL TOPOGRAPHY, 2D + 3D) EXPORT COMPLETE")
    log.info("=" * 70)
    log.info(f"  Channels               : {n_channels}")
    log.info(f"  Reconstruction path    : {used_path}")
    log.info(f"  Channel-name source    : {name_source}")
    log.info(f"  Class counts           : True={int(mask1.sum())}, False={int(mask0.sum())}")
    log.info("  Top-5 discriminative electrodes:")
    for rank, i in enumerate(top5_idx, start=1):
        log.info(f"    #{rank}: {ch_names[i]:<6} diff={diff[i]:+.4f}")
    log.info("=" * 70 + "\n")

    return {
        "n_channels": n_channels,
        "used_path": used_path,
        "name_source": name_source,
        "top5_electrodes": [
            {"rank": r + 1, "channel": ch_names[i], "diff": float(diff[i]),
             "true_memory_power": float(power1_mean[i]), "false_memory_power": float(power0_mean[i]),
             "fisher_score": float(fisher[i])}
            for r, i in enumerate(top5_idx)
        ],
        "files": file_report,
    }


# =============================================================================
# SECTION 11: LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("\n" + "=" * 70)
    print("  Figure 4 — Cortical Scalp Topography & Discriminative Spatial Map")
    print("  (2D static Matplotlib + interactive 3D Plotly asset)")
    print("=" * 70 + "\n")

    results = generate_figure4.remote()

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for key, val in results.items():
        print(f"  {key:<20} : {val}")
    print("=" * 70)
    print(
        "\n  Figure 4 export complete."
        "\n  Static PNG/PDF + interactive 3D HTML committed to eeg-data-vol under /figures/."
        "\n\n  Download with:"
        "\n    modal volume get eeg-data-vol /figures/figure4_cortical_topography.png ."
        "\n    modal volume get eeg-data-vol /figures/figure4_cortical_topography.pdf ."
        "\n    modal volume get eeg-data-vol /figures/figure4_interactive_topography.html .\n"
    )