#!/usr/bin/env python3
"""
Regenerate problematic source panel PNGs from their CSV data files.

These replace the original PNGs that had overlapping text, tiny subplots,
truncated titles, raw underscore labels, or zero-value ghost bars.
The composite figure script (generate_figures.py) will then assemble
them into the final multi-panel PDFs.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent  # biodyn-work root

P31 = BASE / "subproject_38_cell_count_subsampling_robustness"
P34 = BASE / "subproject_38_cluster_resolution_sensitivity"
P36 = BASE / "subproject_38_rare_cell_type_stress_test"
P38 = BASE / "subproject_38_leave_one_donor_out_generalization"
P40 = BASE / "subproject_38_integration_method_sensitivity"

# Style: clean, publication-quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------------
# Human-readable label mappings (no underscores, no abbreviations)
# ---------------------------------------------------------------------------
POLICY_LABEL = {
    "unconstrained": "Unconstrained",
    "tf_source": "TF-guided (source)",
    "tf_source_target": "TF-guided (source + target)",
}

TIER_LABEL = {"large": "Large", "medium": "Medium", "small": "Small"}

DATASET_LABEL = {
    "kidney": "Kidney",
    "Kidney": "Kidney",
    "lung": "Lung",
    "Lung": "Lung",
    "immune": "Immune",
    "Immune": "Immune",
    "immune_invariant": "Immune (invariant)",
    "external_lung": "External lung",
}

METHOD_LABEL = {
    "baseline": "No correction",
    "harmony_batch": "Harmony (batch)",
    "harmony_donor": "Harmony (donor)",
    "harmony_method": "Harmony (method)",
}

RARITY_LABEL = {
    "rare": "Rare",
    "intermediate": "Intermediate",
    "abundant": "Abundant",
}


# ============================================================
# Fig 1A: MVCC bar chart
# ============================================================
def fix_fig1a():
    """
    Regenerate MVCC bar chart.
    Only plot policies that actually exist in the data (no ghost bars).
    """
    print("Fixing fig1_A (MVCC estimates)...")
    csv = P31 / "implementation/outputs_adversarial/mvcc_summary.csv"
    df = pd.read_csv(csv)

    # Cap the conservative MVCC estimate at 3500 for display
    df["mvcc"] = df["mvcc_conservative_cells"].clip(upper=3500)
    df["policy_label"] = df["policy"].map(POLICY_LABEL)
    df["tier_label"] = df["tier"].map(TIER_LABEL)

    # Only use policies present in the data
    policies_present = [POLICY_LABEL[p] for p in df["policy"].unique()
                        if p in POLICY_LABEL]
    # Keep a consistent order
    policy_order = ["Unconstrained", "TF-guided (source)", "TF-guided (source + target)"]
    policies = [p for p in policy_order if p in policies_present]

    tiers = ["Large", "Medium", "Small"]
    tier_colors = {"Large": "#2196F3", "Medium": "#FF9800", "Small": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_width = 0.25
    x = np.arange(len(policies))

    for i, tier in enumerate(tiers):
        vals = []
        for policy in policies:
            row = df[(df["policy_label"] == policy) & (df["tier_label"] == tier)]
            vals.append(row["mvcc"].values[0] if len(row) > 0 else 0)
        offset = (i - 1) * bar_width
        ax.bar(x + offset, vals, bar_width, label=tier,
               color=tier_colors[tier], edgecolor="white", linewidth=0.5)

    # MVCC threshold line
    ax.axhline(y=3000, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.3, 3050, "MVCC threshold (3,000)",
            fontsize=10, ha="left", va="bottom", style="italic")

    ax.set_xlabel("Edge-ranking policy")
    ax.set_ylabel("MVCC (cells)")
    ax.set_title("Conservative MVCC estimates after adversarial repairs")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.legend(title="Dataset size tier", frameon=True, loc="upper right")
    ax.set_ylim(0, 3800)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = P31 / "reports/figures/fig15_round4_mvcc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 1B: High-cell-only edge support
# ============================================================
def fix_fig1b():
    """
    Regenerate edge support bar chart.
    Only plot policies that actually exist in the data.
    """
    print("Fixing fig1_B (high-cell edge support)...")
    csv = P31 / "implementation/outputs_adversarial/emergence_summary.csv"
    df = pd.read_csv(csv)

    # Filter to high_cell_only category
    hc = df[df["category"] == "high_cell_only"].copy()
    hc["policy_label"] = hc["policy"].map(POLICY_LABEL)
    hc["tier_label"] = hc["tier"].map(TIER_LABEL)

    # Only use policies that have non-negligible values (drop Unconstrained ~0)
    policies_present = [POLICY_LABEL[p] for p in hc["policy"].unique()
                        if p in POLICY_LABEL]
    policy_order = ["TF-guided (source)", "TF-guided (source + target)"]
    policies = [p for p in policy_order if p in policies_present]

    tiers = ["Large", "Medium", "Small"]
    tier_colors = {"Large": "#2196F3", "Medium": "#FF9800", "Small": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.25
    x = np.arange(len(policies))

    for i, tier in enumerate(tiers):
        vals = []
        for policy in policies:
            row = hc[(hc["policy_label"] == policy) & (hc["tier_label"] == tier)]
            vals.append(row["dorothea_rate"].values[0] if len(row) > 0 else 0)
        offset = (i - 1) * bar_width
        ax.bar(x + offset, vals, bar_width, label=tier,
               color=tier_colors[tier], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Edge-ranking policy")
    ax.set_ylabel("DoRothEA support rate")
    ax.set_title("High-cell-only edge support by TF-guided policy")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend(title="Dataset size tier", frameon=True, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = P31 / "reports/figures/fig16_round4_high_cell_support.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 2B: Null calibration (overlapping labels)
# ============================================================
def fix_fig2b():
    """
    Regenerate dual-null calibration figure.
    Two subplots stacked vertically with clean dataset labels.
    """
    print("Fixing fig2_B (null calibration)...")
    csv = P34 / "implementation/results/null_calibration.csv"
    df = pd.read_csv(csv)

    datasets = df["dataset"].values
    ds_labels = [DATASET_LABEL.get(d, d) for d in datasets]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    x = np.arange(len(datasets))
    bar_w = 0.28

    # Subplot 1: Top-k overlap vs null
    ax1.bar(x - bar_w, df["global_observed_topk_overlap"], bar_w,
            label="Observed", color="#2196F3")
    ax1.bar(x, df["global_null_mean_topk_overlap"], bar_w,
            label="Global-shuffle null", color="#FF9800")
    ax1.bar(x + bar_w, df["constrained_null_mean_topk_overlap"], bar_w,
            label="Within-coarse null", color="#4CAF50")
    ax1.errorbar(x, df["global_null_mean_topk_overlap"],
                 yerr=df["global_null_std_topk_overlap"],
                 fmt="none", color="black", capsize=3, linewidth=1)
    ax1.errorbar(x + bar_w, df["constrained_null_mean_topk_overlap"],
                 yerr=df["constrained_null_std_topk_overlap"],
                 fmt="none", color="black", capsize=3, linewidth=1)

    ax1.set_ylabel("Top-k overlap")
    ax1.set_title("Top-k overlap vs null distributions")
    ax1.legend(frameon=True, loc="upper right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot 2: Resolution sensitivity vs null
    ax2.bar(x - bar_w, df["global_observed_resolution_sensitivity"], bar_w,
            label="Observed", color="#2196F3")
    ax2.bar(x, df["global_null_mean_resolution_sensitivity"], bar_w,
            label="Global-shuffle null", color="#FF9800")
    ax2.bar(x + bar_w, df["constrained_null_mean_resolution_sensitivity"], bar_w,
            label="Within-coarse null", color="#4CAF50")
    ax2.errorbar(x, df["global_null_mean_resolution_sensitivity"],
                 yerr=df["global_null_std_resolution_sensitivity"],
                 fmt="none", color="black", capsize=3, linewidth=1)
    ax2.errorbar(x + bar_w, df["constrained_null_mean_resolution_sensitivity"],
                 yerr=df["constrained_null_std_resolution_sensitivity"],
                 fmt="none", color="black", capsize=3, linewidth=1)

    ax2.set_ylabel("Resolution sensitivity")
    ax2.set_title("Sensitivity score vs null distributions")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ds_labels)
    ax2.set_xlabel("Dataset")
    ax2.legend(frameon=True, loc="upper right", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    out = P34 / "reports/figures/fig5_null_calibration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 3A: Null AUC vs cell count scatter
# (original had raw "external_lung", "dataset", "rarity_group")
# ============================================================
def fix_fig3a():
    """
    Regenerate null-separation AUC vs cell-count scatter plot
    with clean labels (no underscores in legend).
    """
    print("Fixing fig3_A (null AUC vs cell count)...")
    csv = P36 / "implementation/results/cell_type_metrics.csv"
    df = pd.read_csv(csv)

    ds_colors = {"kidney": "#2196F3", "lung": "#FF9800",
                 "immune": "#4CAF50", "external_lung": "#F44336"}
    rarity_markers = {"abundant": "o", "intermediate": "X", "rare": "s"}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for ds in ["kidney", "lung", "immune", "external_lung"]:
        for rg in ["abundant", "intermediate", "rare"]:
            sub = df[(df["dataset"] == ds) & (df["rarity_group"] == rg)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["cell_count"], sub["null_auc_mean"],
                       c=ds_colors[ds], marker=rarity_markers[rg],
                       s=50, alpha=0.7, edgecolors="none",
                       label=f"{DATASET_LABEL[ds]}, {RARITY_LABEL[rg]}")

    ax.axhline(y=0.65, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Cell count per cell type (log scale)")
    ax.set_ylabel("Observed-vs-null separation (AUC)")
    ax.set_title("Null separation improves with cell-type size")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create a cleaner two-part legend: datasets (color) + rarity (shape)
    ds_handles = [Patch(facecolor=ds_colors[d], alpha=0.7,
                        label=DATASET_LABEL[d])
                  for d in ["kidney", "lung", "immune", "external_lung"]]
    import matplotlib.lines as mlines
    rarity_handles = [
        mlines.Line2D([], [], color="gray", marker=rarity_markers[r],
                      linestyle="None", markersize=7,
                      label=RARITY_LABEL[r])
        for r in ["abundant", "intermediate", "rare"]
    ]
    leg1 = ax.legend(handles=ds_handles, title="Dataset",
                     loc="upper left", frameon=True, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=rarity_handles, title="Rarity group",
              loc="center left", frameon=True, fontsize=9,
              bbox_to_anchor=(0.0, 0.55))

    fig.tight_layout()
    out = P36 / "reports/figures/fig2_null_auc_vs_cell_count.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 3B: Rare vs abundant boxplots
# ============================================================
def fix_fig3b():
    """
    Regenerate boxplots stacked vertically with clean labels.
    """
    print("Fixing fig3_B (rare vs abundant boxplots)...")
    csv = P36 / "implementation/results/cell_type_metrics.csv"
    df = pd.read_csv(csv)

    groups = ["rare", "intermediate", "abundant"]
    ds_colors = {"kidney": "#2196F3", "lung": "#4CAF50",
                 "immune": "#FF9800", "external_lung": "#F44336"}
    datasets = [d for d in df["dataset"].unique() if d in ds_colors]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Subplot 1: Bootstrap rank stability
    positions = []
    bp_data = []
    tick_positions = []
    tick_labels = []
    pos = 0
    for gi, grp in enumerate(groups):
        for di, ds in enumerate(datasets):
            subset = df[(df["rarity_group"] == grp) & (df["dataset"] == ds)]
            vals = subset["stability_spearman_mean"].dropna().values
            if len(vals) > 0:
                bp_data.append(vals)
                positions.append(pos)
            pos += 1
        tick_positions.append(pos - len(datasets) / 2 - 0.5)
        tick_labels.append(RARITY_LABEL[grp])
        pos += 1  # gap between groups

    bp1 = ax1.boxplot(bp_data, positions=positions, widths=0.7,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker="o", markersize=3, alpha=0.5))
    # Color by dataset
    color_idx = 0
    for gi, grp in enumerate(groups):
        for di, ds in enumerate(datasets):
            subset = df[(df["rarity_group"] == grp) & (df["dataset"] == ds)]
            if len(subset["stability_spearman_mean"].dropna()) > 0:
                bp1["boxes"][color_idx].set_facecolor(ds_colors[ds])
                bp1["boxes"][color_idx].set_alpha(0.7)
                color_idx += 1

    ax1.axhline(y=0.75, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_ylabel("Rank stability (Spearman)")
    ax1.set_title("Stability by rarity group")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_ylim(0, 1.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    legend_patches = [Patch(facecolor=ds_colors[d], alpha=0.7,
                            label=DATASET_LABEL.get(d, d)) for d in datasets]
    ax1.legend(handles=legend_patches, loc="lower left", frameon=True, fontsize=9)

    # Subplot 2: Null separation AUC
    positions2 = []
    bp_data2 = []
    tick_positions2 = []
    tick_labels2 = []
    pos = 0
    for gi, grp in enumerate(groups):
        for di, ds in enumerate(datasets):
            subset = df[(df["rarity_group"] == grp) & (df["dataset"] == ds)]
            vals = subset["null_auc_mean"].dropna().values
            if len(vals) > 0:
                bp_data2.append(vals)
                positions2.append(pos)
            pos += 1
        tick_positions2.append(pos - len(datasets) / 2 - 0.5)
        tick_labels2.append(RARITY_LABEL[grp])
        pos += 1

    bp2 = ax2.boxplot(bp_data2, positions=positions2, widths=0.7,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker="o", markersize=3, alpha=0.5))
    color_idx = 0
    for gi, grp in enumerate(groups):
        for di, ds in enumerate(datasets):
            subset = df[(df["rarity_group"] == grp) & (df["dataset"] == ds)]
            if len(subset["null_auc_mean"].dropna()) > 0:
                bp2["boxes"][color_idx].set_facecolor(ds_colors[ds])
                bp2["boxes"][color_idx].set_alpha(0.7)
                color_idx += 1

    ax2.axhline(y=0.65, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Observed-vs-null separation (AUC)")
    ax2.set_title("Null separation by rarity group")
    ax2.set_xticks(tick_positions2)
    ax2.set_xticklabels(tick_labels2)
    ax2.set_xlabel("Cell-type rarity group")
    ax2.set_ylim(0, 1.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(handles=legend_patches, loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    out = P36 / "reports/figures/fig3_rare_vs_abundant_boxplots.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 3C: Minimum cell size guideline
# ============================================================
def fix_fig3c():
    """
    Regenerate with clean dataset labels.
    """
    print("Fixing fig3_C (minimum cell-size guideline)...")
    csv = P36 / "implementation/results/minimum_cell_size_guidelines.csv"
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(8, 5))

    ds_labels = [DATASET_LABEL.get(d, d) for d in df["dataset"]]
    colors = ["#90CAF9", "#FFAB91", "#A5D6A7", "#F48FB1"]
    bars = ax.bar(ds_labels, df["combined_guideline"], color=colors[:len(df)],
                  edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, df["combined_guideline"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{int(val):,}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.axhline(y=250, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="Typical rare-type threshold (250)")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Estimated minimum cells")
    ax.set_title("Estimated minimum cell-type size for reliable inference")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = P36 / "reports/figures/fig5_minimum_cell_size_guideline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 4B: Composition control boxplots
# (original had "top8_unbalanced", "top8_celltype_matched")
# ============================================================
def fix_fig4b():
    """
    Regenerate composition control comparison boxplot
    with clean, human-readable x-axis labels.
    """
    print("Fixing fig4_B (composition control boxplots)...")
    csv = P38 / "reports/artifacts/adversarial_composition_control_top8_split_metrics.csv"
    df = pd.read_csv(csv)

    variant_label = {
        "top8_unbalanced": "Natural composition",
        "top8_celltype_matched": "Cell-type-matched composition",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    variants = ["top8_unbalanced", "top8_celltype_matched"]
    bp_data = [df[df["panel_variant"] == v]["spearman_rho"].values for v in variants]

    bp = ax.boxplot(bp_data, widths=0.5, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))

    colors_box = ["#2196F3", "#FF9800"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, v in enumerate(variants):
        vals = df[df["panel_variant"] == v]["spearman_rho"].values
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full_like(vals, i + 1) + jitter, vals,
                   c="gray", alpha=0.4, s=20, zorder=3)

    ax.set_xticklabels([variant_label[v] for v in variants])
    ax.set_ylabel("Spearman rank correlation (top-8 panel)")
    ax.set_title("Effect of donor composition on reproducibility")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = P38 / "reports/figures/adversarial_composition_control_top8_spearman.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 5A: Rank Spearman heatmap
# (original had "harmony_batch", "harmony_donor", "harmony_method"
#  and y-axis label "dataset")
# ============================================================
def fix_fig5a():
    """
    Regenerate integration rank-correlation heatmap with clean labels.
    """
    print("Fixing fig5_A (rank Spearman heatmap)...")
    csv = P40 / "data/integration_summary.csv"
    df = pd.read_csv(csv)

    methods_order = ["harmony_batch", "harmony_donor", "harmony_method"]
    datasets_order = ["Immune", "Kidney", "Lung"]

    # Build matrix (datasets x methods)
    matrix = np.full((len(datasets_order), len(methods_order)), np.nan)
    for di, ds in enumerate(datasets_order):
        for mi, meth in enumerate(methods_order):
            row = df[(df["dataset"] == ds) & (df["method"] == meth)]
            if len(row) > 0:
                matrix[di, mi] = row["rank_spearman"].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use a masked array to handle NaN (missing Kidney/harmony_donor)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Clean labels
    method_display = [METHOD_LABEL[m] for m in methods_order]
    dataset_display = datasets_order

    ax.set_xticks(np.arange(len(methods_order)))
    ax.set_xticklabels(method_display)
    ax.set_yticks(np.arange(len(datasets_order)))
    ax.set_yticklabels(dataset_display)

    ax.set_xlabel("Integration covariate")
    # No "dataset" label on y-axis; the dataset names themselves are clear

    ax.set_title("Rank correlation (Spearman) vs no-correction baseline")

    # Annotate cells with values
    for di in range(len(datasets_order)):
        for mi in range(len(methods_order)):
            val = matrix[di, mi]
            if not np.isnan(val):
                text_color = "white" if val > 0.5 else "black"
                ax.text(mi, di, f"{val:.3f}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=text_color)
            else:
                ax.text(mi, di, "N/A", ha="center", va="center",
                        fontsize=12, color="gray", style="italic")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman correlation")

    fig.tight_layout()
    out = P40 / "reports/figures/fig2_rank_spearman_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 5B: Reference precision@500
# ============================================================
def fix_fig5b():
    """
    Regenerate precision chart with clean method labels.
    """
    print("Fixing fig5_B (reference precision@500)...")
    csv = P40 / "data/reference_precision.csv"
    df = pd.read_csv(csv)

    # Filter to k=500 and the most informative references
    df500 = df[df["k"] == 500].copy()
    key_refs = ["dorothea_ab", "dorothea_abcd", "trrust"]
    df500 = df500[df500["reference"].isin(key_refs)]

    datasets = ["Immune", "Lung", "Kidney"]
    ref_colors = {
        "dorothea_ab": "#2196F3",
        "dorothea_abcd": "#FF9800",
        "trrust": "#4CAF50",
    }
    ref_labels = {
        "dorothea_ab": "DoRothEA (AB)",
        "dorothea_abcd": "DoRothEA (ABCD)",
        "trrust": "TRRUST",
    }

    fig, axes = plt.subplots(len(datasets), 1, figsize=(9, 10), sharex=True)

    for di, ds in enumerate(datasets):
        ax = axes[di]
        ds_data = df500[df500["dataset"] == ds]
        method_list = sorted(ds_data["method"].unique())
        x = np.arange(len(method_list))
        bar_w = 0.25

        for ri, ref in enumerate(key_refs):
            vals = []
            for m in method_list:
                row = ds_data[(ds_data["method"] == m) & (ds_data["reference"] == ref)]
                vals.append(row["precision"].values[0] if len(row) > 0 else 0)
            offset = (ri - 1) * bar_w
            ax.bar(x + offset, vals, bar_w, label=ref_labels[ref],
                   color=ref_colors[ref], edgecolor="white", linewidth=0.5)

        # Clean method names using the global lookup
        method_display = [METHOD_LABEL.get(m, m) for m in method_list]
        ax.set_xticks(x)
        ax.set_xticklabels(method_display, rotation=15, ha="right")
        ax.set_ylabel("Precision@500")
        ax.set_title(f"{ds}", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if di == 0:
            ax.legend(frameon=True, loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Integration covariate")
    fig.suptitle("Precision@500 vs external references", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    out = P40 / "reports/figures/fig5_reference_precision_k500.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Fig 5C: Harmony PCs sweep
# ============================================================
def fix_fig5c():
    """
    Regenerate PCs sweep with stacked vertical subplots and clean titles.
    """
    print("Fixing fig5_C (Harmony PCs sweep)...")
    csv = P40 / "data/harmony_batch_pcs_sweep.csv"
    df = pd.read_csv(csv)

    ds_colors = {"Immune": "#2196F3", "Lung": "#FF9800", "Kidney": "#4CAF50"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for ds in ["Immune", "Lung", "Kidney"]:
        sub = df[df["dataset"] == ds].sort_values("n_pcs")
        ax1.plot(sub["n_pcs"], sub["top500_overlap_fraction"], "o-",
                 color=ds_colors[ds], label=ds, linewidth=2, markersize=6)
        ax2.plot(sub["n_pcs"], sub["precision500_dorothea_abcd"], "s-",
                 color=ds_colors[ds], label=ds, linewidth=2, markersize=6)

    ax1.set_ylabel("Top-500 overlap fraction")
    ax1.set_title("Network overlap vs number of PCs (Harmony, batch covariate)")
    ax1.legend(frameon=True, loc="lower right")
    ax1.set_ylim(0, 1.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_ylabel("DoRothEA (ABCD) precision@500")
    ax2.set_title("External validation vs number of PCs")
    ax2.set_xlabel("Number of principal components")
    ax2.legend(frameon=True, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    out = P40 / "reports/figures/fig7_harmony_batch_pcs_sweep.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Regenerating all source panels...\n")
    fix_fig1a()
    fix_fig1b()
    fix_fig2b()
    fix_fig3a()   # NEW: scatter with clean legend
    fix_fig3b()
    fix_fig3c()
    fix_fig4b()   # NEW: composition control with clean labels
    fix_fig5a()   # NEW: heatmap with clean axis labels
    fix_fig5b()
    fix_fig5c()
    print("\nAll source panels regenerated.")
    print("\nNow re-run generate_figures.py to rebuild composite figures.")
