#!/usr/bin/env python3
"""
Generate composite multi-panel PDF figures for the robustness audit paper.

Creates 6 PDF figures by stacking individual source panels vertically.
Source panels are either pre-existing PNGs (in source_panels/) or
regenerated from CSV data by generate_source_panels.py.

Layout:
  - Each panel gets the full figure width (7.5 in) for readability.
  - Height is derived from each source image's native aspect ratio.
  - Panel labels (A, B, C ...) are placed at a consistent top-left position.

Run:
    python scripts/generate_figures.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to repository root)
# ---------------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parent.parent
PANELS = ROOT / "source_panels"
OUT    = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_WIDTH = 7.5   # inches; Bioinformatics single-column width
PANEL_GAP = 0.12  # vertical gap between panels (fraction of max panel height)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_img(path):
    """Load image as numpy array; return None if not found."""
    p = Path(path)
    if p.exists():
        return mpimg.imread(str(p))
    print(f"  WARNING: missing {p}")
    return None


def img_aspect(img):
    """Return width/height aspect ratio of an image array."""
    if img is None:
        return 1.6
    h, w = img.shape[:2]
    return w / h


def add_panel_label(ax, label):
    """Place a bold panel label (A, B, C ...) at top-left of axes."""
    ax.text(
        -0.02, 1.03, label,
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        va="bottom", ha="left", zorder=10,
    )


def show_img(ax, img):
    """Display image on axes with no ticks/spines, or show placeholder."""
    if img is not None:
        ax.imshow(img, aspect="auto")
    else:
        ax.text(0.5, 0.5, "Panel not available",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
    ax.axis("off")


def build_vertical_figure(panels, out_path):
    """
    Stack panels vertically into a composite figure.

    Parameters
    ----------
    panels : list of (image_array_or_None, label_str)
    out_path : Path
    """
    height_ratios = [FIG_WIDTH / img_aspect(img) for img, _ in panels]
    total_height = (sum(height_ratios)
                    + PANEL_GAP * (len(panels) - 1) * max(height_ratios))

    fig = plt.figure(figsize=(FIG_WIDTH, total_height))
    gs = GridSpec(len(panels), 1, figure=fig,
                  height_ratios=height_ratios, hspace=PANEL_GAP)

    for idx, (img, label) in enumerate(panels):
        ax = fig.add_subplot(gs[idx, 0])
        show_img(ax, img)
        add_panel_label(ax, label)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================
# Figure 1: Cell-count subsampling (2 panels)
# ============================================================
def fig1_cell_count():
    print("  Fig 1: Cell-count subsampling ...")
    panels = [
        (load_img(PANELS / "fig1a_mvcc.png"), "A"),
        (load_img(PANELS / "fig1b_edge_support.png"), "B"),
    ]
    build_vertical_figure(panels, OUT / "fig1_cell_count.pdf")


# ============================================================
# Figure 2: Resolution sensitivity + dual-null calibration
# ============================================================
def fig2_resolution():
    print("  Fig 2: Resolution sensitivity ...")
    panels = [
        (load_img(PANELS / "fig1_pooled_resolution_sensitivity.png"), "A"),
        (load_img(PANELS / "fig2b_null_calibration.png"), "B"),
    ]
    build_vertical_figure(panels, OUT / "fig2_resolution.pdf")


# ============================================================
# Figure 3: Rare cell-type stress test (3 panels)
# ============================================================
def fig3_rare():
    print("  Fig 3: Rare-cell-type reliability ...")
    panels = [
        (load_img(PANELS / "fig3a_null_auc_scatter.png"), "A"),
        (load_img(PANELS / "fig3b_rarity_boxplots.png"), "B"),
        (load_img(PANELS / "fig3c_min_cell_size.png"), "C"),
    ]
    build_vertical_figure(panels, OUT / "fig3_rare.pdf")


# ============================================================
# Figure 4: Donor generalization (2 panels)
# ============================================================
def fig4_donor():
    print("  Fig 4: Donor generalization ...")
    panels = [
        (load_img(PANELS / "adversarial_immune_top12_fixed_holdout_curves.png"), "A"),
        (load_img(PANELS / "fig4b_composition_control.png"), "B"),
    ]
    build_vertical_figure(panels, OUT / "fig4_donor.pdf")


# ============================================================
# Figure 5: Integration-method sensitivity (3 panels)
# ============================================================
def fig5_integration():
    print("  Fig 5: Integration-method sensitivity ...")
    panels = [
        (load_img(PANELS / "fig5a_rank_heatmap.png"), "A"),
        (load_img(PANELS / "fig5b_precision.png"), "B"),
        (load_img(PANELS / "fig5c_pcs_sweep.png"), "C"),
    ]
    build_vertical_figure(panels, OUT / "fig5_integration.pdf")


# ============================================================
# Figure 6: Cross-axis fragility summary (generated from data)
# ============================================================
def fig6_cross_axis():
    print("  Fig 6: Cross-axis fragility summary ...")

    datasets = ["Kidney", "Lung", "Immune", "Immune inv.", "Ext. lung"]
    axes_labels = [
        "Cell count\n(MVCC > 3 000)",
        "Resolution\n(dual-null)",
        "Rare types\n(reliability gate)",
        "Donor\n(6\u20138 needed)",
        "Integration\n(no pert. gain)",
    ]

    # Fragility matrix: 0 = Pass, 1 = Marginal, 2 = Fail
    fragility = np.array([
        [2, 2, 1, 1, 1],   # Kidney
        [1, 2, 2, 2, 2],   # Lung
        [1, 2, 2, 1, 1],   # Immune
        [1, 2, 2, 1, 1],   # Immune inv.
        [1, 2, 2, 1, 1],   # External lung
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))
    cmap = ListedColormap(["#4CAF50", "#FFC107", "#F44336"])
    ax.imshow(fragility, cmap=cmap, aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(len(axes_labels)))
    ax.set_xticklabels(axes_labels, fontsize=10, ha="center")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=11)

    status_labels = {0: "Pass", 1: "Marginal", 2: "Fail"}
    for i in range(len(datasets)):
        for j in range(len(axes_labels)):
            val = int(fragility[i, j])
            color = "white" if val == 2 else "black"
            ax.text(j, i, status_labels[val],
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("Cross-Axis Fragility Profile",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Robustness Axis", fontsize=11, labelpad=10)
    ax.set_ylabel("Dataset", fontsize=11)

    legend_elements = [
        Patch(facecolor="#4CAF50", label="Pass"),
        Patch(facecolor="#FFC107", label="Marginal"),
        Patch(facecolor="#F44336", label="Fail"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=10, frameon=True)

    fig.savefig(OUT / "fig6_cross_axis.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating composite figures ...\n")
    fig1_cell_count()
    fig2_resolution()
    fig3_rare()
    fig4_donor()
    fig5_integration()
    fig6_cross_axis()
    print("\nAll 6 figures saved to paper/figures/")
