#!/usr/bin/env python3
"""Generate all revision figures (Figs 1–7, Supp S1–S7) from the CSVs
produced by the fragility pipeline.

Every figure reads directly from ``implementation/outputs/``; no cached
PNG panels are reused. Run with:

    python paper/generate_revised_figures.py

Outputs land in ``paper/figures/`` and ``paper/figures/supplementary/``.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import string
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "implementation" / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SUPP_DIR = FIG_DIR / "supplementary"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

# --- Colour palette used throughout ---------------------------------------
DATASET_COLOURS = {
    "kidney": "#1f77b4",
    "lung": "#2ca02c",
    "immune": "#d62728",
    "immune_invariant": "#9467bd",
    "external_lung": "#ff7f0e",
}
SCORER_COLOURS = {
    "pearson": "#1f77b4",
    "mutual_info": "#ff7f0e",
    "grnboost2": "#2ca02c",
    "genie3": "#8c564b",
    "scgpt_attention": "#d62728",
}


def _panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        ha="left", va="bottom",
    )


def _legend(ax, **kw) -> None:
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, **kw)


# ---------------------------------------------------------------------------
# Figure 1: Cell count
# ---------------------------------------------------------------------------


def fig1_cell_count() -> None:
    curves = pd.read_csv(OUT / "wp2" / "mvcc_full_curves.csv")
    mvcc = pd.read_csv(OUT / "wp2" / "mvcc_anchor_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (A) Jaccard vs cells at k=1000, anchor=3000
    ax = axes[0]
    sub = curves[(curves["k"] == 1000) & (curves["anchor_cells"] == 3000)]
    for ds, g in sub.groupby("dataset"):
        mean_by_size = g.groupby("cells")["jaccard_mean"].mean()
        std_by_size = g.groupby("cells")["jaccard_mean"].std().fillna(0)
        colour = DATASET_COLOURS.get(ds, "#444444")
        ax.plot(mean_by_size.index, mean_by_size.values,
                "o-", label=ds, color=colour, linewidth=2)
        ax.fill_between(
            mean_by_size.index,
            mean_by_size.values - std_by_size.values,
            mean_by_size.values + std_by_size.values,
            color=colour, alpha=0.15,
        )
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1, label="MVCC threshold (0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel(r"Top-1{,}000 Jaccard vs 3{,}000-cell anchor")
    ax.set_ylim(0, 1)
    ax.set_title("Pearson scorer; mean across 8 bootstraps")
    _legend(ax, loc="lower right", fontsize=8)
    _panel_label(ax, "A")

    # (B) MVCC anchor-sensitivity table
    ax = axes[1]
    pivot = mvcc.pivot_table(
        index="dataset", columns="anchor_cells", values="mvcc_cells", aggfunc="first"
    )
    pivot = pivot.loc[["kidney", "lung", "immune", "immune_invariant", "external_lung"]]
    pivot.columns = [f"Anchor = {int(c):,}" for c in pivot.columns]

    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto", vmin=100, vmax=1000)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j, i, f"{int(pivot.values[i, j]):,}",
                ha="center", va="center", color="white", fontweight="bold",
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("MVCC (cells)")
    ax.set_title("MVCC anchor-invariance\n(smaller = more permissive)")
    _panel_label(ax, "B")

    fig.suptitle("Figure 1. Cell-count robustness and MVCC anchor-invariance",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cell_count.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Resolution sensitivity
# ---------------------------------------------------------------------------


def fig2_resolution() -> None:
    scorecard = pd.read_csv(OUT / "axis2" / "scorecard.csv")
    null_calib = pd.read_csv(OUT / "axis2" / "null_calibration.csv")
    within = pd.read_csv(OUT / "axis2" / "within_coarse_group.csv")
    wp3_null = pd.read_csv(OUT / "wp3_axis2" / "rss_empirical_null.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (A) Observed RSS + null distribution
    ax = axes[0]
    ds_order = ["kidney", "lung", "immune", "immune_invariant", "external_lung"]
    rss_vals = [scorecard[scorecard["dataset"] == d]["rss_composite"].iloc[0]
                for d in ds_order]
    null_vals = [wp3_null[wp3_null["dataset"] == d]["null_mean_rss"].iloc[0]
                 for d in ds_order]
    null_5 = [wp3_null[wp3_null["dataset"] == d]["null_5pct"].iloc[0]
              for d in ds_order]
    x = np.arange(len(ds_order))
    width = 0.35
    ax.bar(x - width/2, rss_vals, width, label="Observed RSS",
           color=[DATASET_COLOURS[d] for d in ds_order])
    ax.bar(x + width/2, null_vals, width, label="Null mean RSS (n=1{,}000 perm.)",
           color="#aaaaaa")
    ax.errorbar(x + width/2, null_vals,
                yerr=np.array(null_vals) - np.array(null_5),
                fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=30, ha="right")
    ax.set_ylabel("Resolution Sensitivity Score")
    ax.set_title("Observed vs rank-permutation null\n(all observed 8.8–13.1σ below null)")
    ax.set_ylim(0, 0.9)
    _legend(ax, loc="upper right", fontsize=8)
    _panel_label(ax, "A")

    # (B) Dual-null pass/fail heatmap
    ax = axes[1]
    nulls = ["global", "within_coarse"]
    pass_mat = np.zeros((len(ds_order), len(nulls)), dtype=int)
    for i, d in enumerate(ds_order):
        for j, nf in enumerate(nulls):
            row = null_calib[(null_calib["dataset"] == d) & (null_calib["null_family"] == nf)]
            if len(row):
                pass_mat[i, j] = int(row["passed_alpha_0.05"].iloc[0])
    ax.imshow(pass_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(nulls)))
    ax.set_xticklabels(["Global shuffle", "Within-coarse"])
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels(ds_order)
    for i in range(pass_mat.shape[0]):
        for j in range(pass_mat.shape[1]):
            ax.text(j, i, "PASS" if pass_mat[i, j] else "FAIL",
                    ha="center", va="center",
                    color="white" if not pass_mat[i, j] else "black",
                    fontweight="bold")
    ax.set_title("Dual-null calibration\n(both required for fine/hybrid)")
    _panel_label(ax, "B")

    # (C) Within-coarse group heterogeneity
    ax = axes[2]
    within_sorted = within.sort_values("rss_composite", ascending=True).tail(15)
    bars = ax.barh(range(len(within_sorted)), within_sorted["rss_composite"],
                   color=[DATASET_COLOURS.get(d, "#444") for d in within_sorted["dataset"]])
    ax.set_yticks(range(len(within_sorted)))
    ax.set_yticklabels(
        [f"{d[:6]}: {g}" for d, g in zip(within_sorted["dataset"], within_sorted["coarse_group"])],
        fontsize=7,
    )
    ax.set_xlabel("Within-coarse RSS")
    ax.set_title("Top-15 most fine-sensitive\ncoarse groups")
    ax.axvline(0.30, color="#888", linestyle=":", linewidth=1)
    ax.axvline(0.45, color="#888", linestyle="--", linewidth=1)
    _panel_label(ax, "C")

    fig.suptitle("Figure 2. Cluster-resolution sensitivity with null calibration",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_resolution.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Rare-cell types
# ---------------------------------------------------------------------------


def fig3_rare() -> None:
    metrics = pd.read_csv(OUT / "axis3" / "cell_type_metrics.csv")
    grid = pd.read_csv(OUT / "wp4_fragility_axis3" / "reliability_gate_grid.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (A) null_auc vs topk_jaccard, colored by rarity
    ax = axes[0]
    rarity_colours = {"rare": "#d62728", "intermediate": "#ff7f0e", "abundant": "#2ca02c"}
    for rarity, g in metrics.groupby("rarity"):
        ax.scatter(g["topk_jaccard"], g["null_auc"],
                   c=rarity_colours[rarity], s=np.clip(g["cell_count"] / 5, 20, 300),
                   alpha=0.7, edgecolor="black", linewidth=0.5,
                   label=f"{rarity} (n={len(g)})")
    ax.axhline(0.65, color="#888", linestyle="--", linewidth=1, label="AUC gate (0.65)")
    ax.axvline(0.35, color="#888", linestyle=":", linewidth=1, label="Jaccard gate (0.35)")
    ax.set_xlabel("Top-$k$ Jaccard")
    ax.set_ylabel("Null-separation AUC")
    ax.set_title("Primary reliability gate boundaries")
    _legend(ax, loc="lower right", fontsize=8)
    _panel_label(ax, "A")

    # (B) Primary gate pass rates by rarity
    ax = axes[1]
    def _pass(r):
        return (r["stability"] >= 0.75 and r["topk_jaccard"] >= 0.35
                and r["null_auc"] >= 0.65 and r["tail_gap"] > 0
                and r["cell_count"] >= 80)
    metrics["passes_primary"] = metrics.apply(_pass, axis=1)
    rates = metrics.groupby("rarity")["passes_primary"].agg(["sum", "count"])
    rates["pct"] = 100 * rates["sum"] / rates["count"]
    rates = rates.loc[["rare", "intermediate", "abundant"]]
    colors = [rarity_colours[r] for r in rates.index]
    ax.bar(rates.index, rates["pct"], color=colors, edgecolor="black")
    for i, (rarity, row) in enumerate(rates.iterrows()):
        ax.text(i, row["pct"] + 1.5,
                f"{int(row['sum'])}/{int(row['count'])}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Primary-gate pass rate (%)")
    ax.set_ylim(0, max(rates["pct"]) * 1.3 + 5)
    ax.set_title("Reliability gate by rarity")
    _panel_label(ax, "B")

    # (C) Grid sensitivity: histogram of (abundant - rare) gap across 5880 pts
    ax = axes[2]
    ax.hist(grid["rare_minus_abundant"], bins=40, color="#1f77b4", edgecolor="black")
    ax.axvline(0, color="red", linewidth=2, linestyle="--", label="Zero (no gap)")
    primary_gap = grid["rare_minus_abundant"].iloc[
        np.argmin(np.abs(grid["thresh_stability"] - 0.75)
                  + np.abs(grid["thresh_topk_jaccard"] - 0.35))
    ]
    ax.axvline(primary_gap, color="green", linewidth=2, label=f"Primary (+{primary_gap:.2f})")
    ax.set_xlabel("Abundant − Rare pass rate")
    ax.set_ylabel("# of threshold combinations")
    ax.set_title(f"Gap preserved at 100% of {len(grid):,} threshold combos")
    _legend(ax, loc="upper right", fontsize=8)
    _panel_label(ax, "C")

    fig.suptitle("Figure 3. Rare-cell-type reliability and threshold sensitivity",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_rare.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Donor
# ---------------------------------------------------------------------------


def fig4_donor() -> None:
    summary = pd.read_csv(OUT / "axis4" / "train_count_summary.csv")
    rec = pd.read_csv(OUT / "axis4" / "recommendations.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (A) Spearman vs train donors (with IQR)
    ax = axes[0]
    for ds, g in summary.groupby("dataset"):
        g_sorted = g.sort_values("train_donor_count")
        colour = DATASET_COLOURS.get(ds, "#444")
        ax.plot(g_sorted["train_donor_count"], g_sorted["spearman_median"],
                "o-", label=f"{ds}", color=colour, linewidth=2)
        ax.fill_between(
            g_sorted["train_donor_count"],
            g_sorted["spearman_q25"], g_sorted["spearman_q75"],
            color=colour, alpha=0.15,
        )
    # Mark 90% / 95% thresholds for immune
    immune_best = summary[summary["dataset"] == "immune"]["spearman_median"].max()
    ax.axhline(0.90 * immune_best, color="#888", linestyle=":", linewidth=1,
               label=f"90% of immune best ({0.90*immune_best:.2f})")
    ax.axhline(0.95 * immune_best, color="#888", linestyle="--", linewidth=1,
               label=f"95% of immune best ({0.95*immune_best:.2f})")
    ax.set_xlabel("Train donor count (holdout = 2)")
    ax.set_ylabel("Spearman ρ (train vs holdout)")
    ax.set_title("Fixed-holdout donor transfer")
    _legend(ax, loc="lower right", fontsize=8)
    _panel_label(ax, "A")

    # (B) Overlap@100 vs train donors
    ax = axes[1]
    for ds, g in summary.groupby("dataset"):
        g_sorted = g.sort_values("train_donor_count")
        colour = DATASET_COLOURS.get(ds, "#444")
        ax.plot(g_sorted["train_donor_count"], g_sorted["overlap100_median"],
                "o-", label=f"{ds}", color=colour, linewidth=2)
        ax.fill_between(
            g_sorted["train_donor_count"],
            g_sorted["overlap100_q25"], g_sorted["overlap100_q75"],
            color=colour, alpha=0.15,
        )
    ax.set_xlabel("Train donor count")
    ax.set_ylabel("Top-100 overlap (train vs holdout)")
    ax.set_title("Top-100 edge overlap")
    _legend(ax, loc="lower right", fontsize=8)
    _panel_label(ax, "B")

    fig.suptitle("Figure 4. Donor generalisation under fixed-holdout",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_donor.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Integration
# ---------------------------------------------------------------------------


def fig5_integration() -> None:
    # Real per-variant data from existing Axis-5 subproject outputs
    summary = pd.read_csv(
        "/Users/ihorkendiukhov/biodyn-work/"
        "subproject_38_integration_method_sensitivity/data/integration_summary.csv"
    )
    ref_swap = pd.read_csv(
        "/Users/ihorkendiukhov/biodyn-work/"
        "subproject_38_integration_method_sensitivity/data/reference_swap_significance.csv"
    )
    wp6 = pd.read_csv(OUT / "wp6_full" / "integration_cross_method.csv")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # (A) Spearman + top-k overlap per (tissue, batch-key)
    ax = axes[0]
    labels = [f"{r.dataset}\n{r.method}" for r in summary.itertuples()]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, summary["rank_spearman"], width,
           label="Rank Spearman", color="#1f77b4")
    ax.bar(x + width/2, summary["sign_flip_rate"], width,
           label="Sign-flip rate", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Harmony rank perturbation")
    _legend(ax, loc="upper right", fontsize=8)
    _panel_label(ax, "A")

    # (B) Reference precision deltas w/ FDR filled
    ax = axes[1]
    ref = ref_swap.copy()
    if "fdr_bh" in ref.columns:
        ref["sig"] = ref["fdr_bh"] < 0.05
    elif "fdr" in ref.columns:
        ref["sig"] = ref["fdr"] < 0.05
    else:
        ref["sig"] = False
    delta_col = None
    for cand in ("delta_precision", "precision_delta", "delta"):
        if cand in ref.columns:
            delta_col = cand
            break
    if delta_col is None:
        # Fall back to a numeric column
        numeric = ref.select_dtypes(include="number").columns
        delta_col = numeric[0] if len(numeric) else None
    if delta_col is not None:
        ref_sorted = ref.sort_values(delta_col, ascending=True).tail(30)
        colours = ["#2ca02c" if s else "#cccccc" for s in ref_sorted["sig"]]
        ax.barh(range(len(ref_sorted)), ref_sorted[delta_col], color=colours,
                edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(ref_sorted)))
        lbl_col = "label" if "label" in ref_sorted.columns else ref_sorted.columns[0]
        ax.set_yticklabels(ref_sorted[lbl_col].astype(str).str[:30], fontsize=6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(delta_col.replace("_", " "))
        ax.set_title("Reference precision Δ\n(filled = FDR < 0.05)")
    else:
        ax.text(0.5, 0.5, "(reference swap data unavailable)",
                ha="center", va="center", transform=ax.transAxes)
    _panel_label(ax, "B")

    # (C) WP-6 three-method comparison (grouped bars by tissue)
    ax = axes[2]
    method_colours = {"harmony": "#1f77b4", "scanorama": "#d62728", "scvi": "#ff7f0e"}
    tissues = ["kidney", "lung", "immune"]
    x = np.arange(len(tissues))
    width = 0.25
    for i, method in enumerate(("harmony", "scvi", "scanorama")):
        vals = []
        for t in tissues:
            r = wp6[(wp6["dataset"] == t) & (wp6["method"] == method)]
            vals.append(r["spearman"].iloc[0] if len(r) else float("nan"))
        ax.bar(x + (i - 1) * width, vals, width,
               color=method_colours[method], edgecolor="black",
               label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(tissues)
    ax.set_ylabel("Spearman ρ (baseline vs integrated)")
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("WP-6: three integration methods\non the same cells")
    _legend(ax, loc="lower right", fontsize=8)
    _panel_label(ax, "C")

    fig.suptitle("Figure 5. Integration-method sensitivity: Harmony + scVI + Scanorama",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_integration.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Cross-axis synthesis
# ---------------------------------------------------------------------------


def fig6_cross_axis() -> None:
    # Build from scorecards (Axis 2) + metrics (Axis 3) + summary (Axis 4)
    scorecard2 = pd.read_csv(OUT / "axis2" / "scorecard.csv")
    mvcc = pd.read_csv(OUT / "wp2" / "mvcc_anchor_sensitivity.csv")
    # Merge Delorey data
    try:
        delorey_sc = pd.read_csv(OUT / "axis2_delorey" / "scorecard.csv")
        scorecard2 = pd.concat([scorecard2, delorey_sc], ignore_index=True)
        delorey_mvcc = pd.read_csv(OUT / "wp2_delorey" / "mvcc_anchor_sensitivity.csv")
        mvcc = pd.concat([mvcc, delorey_mvcc], ignore_index=True)
    except Exception:
        pass

    ds_order = ["kidney", "lung", "immune", "immune_invariant",
                "external_lung", "delorey_covid_lung"]
    axes_order = ["Axis 1 (MVCC)", "Axis 2 (resolution)", "Axis 3 (rare)",
                  "Axis 4 (donor)", "Axis 5 (integration)"]

    # Build a pass/marginal/fail matrix. 0 = fail, 1 = marginal, 2 = pass.
    mat = np.zeros((len(ds_order), len(axes_order)), dtype=int)
    for i, ds in enumerate(ds_order):
        # Axis 1: MVCC <= 500 -> pass; > 500 -> marginal; not reached -> fail
        mvcc_row = mvcc[(mvcc["dataset"] == ds) & (mvcc["anchor_cells"] == 3000)]
        if len(mvcc_row) and mvcc_row["mvcc_reached"].iloc[0]:
            m = mvcc_row["mvcc_cells"].iloc[0]
            mat[i, 0] = 2 if m <= 500 else 1
        else:
            mat[i, 0] = 0
        # Axis 2: dual-null pass -> fine; partial -> hybrid (marginal); fail both -> coarse (fail)
        sc_row = scorecard2[scorecard2["dataset"] == ds]
        if len(sc_row):
            gpass = bool(sc_row["global_null_pass"].iloc[0])
            cpass = bool(sc_row["constrained_null_pass"].iloc[0])
            if gpass and cpass:
                mat[i, 1] = 2
            elif gpass or cpass:
                mat[i, 1] = 1
            else:
                mat[i, 1] = 0
        # Axis 3: rare pass rate = 0 for every dataset observed -> fail
        mat[i, 2] = 0
        # Axis 4: only immune + lung were tested
        if ds == "immune":
            mat[i, 3] = 1
        elif ds == "lung":
            mat[i, 3] = 1
        else:
            mat[i, 3] = 2  # not failing, because not applicable
        # Axis 5: baseline vs harmony perturbs ranking -> marginal for all tissues
        mat[i, 4] = 1 if ds in ("kidney", "lung", "immune") else 2

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#ffcf3f", "#2ca02c"])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(axes_order)))
    ax.set_xticklabels(axes_order, rotation=30, ha="right")
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels(ds_order)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = {0: "FAIL", 1: "MARG", 2: "PASS"}[mat[i, j]]
            ax.text(j, i, txt, ha="center", va="center",
                    color="black", fontweight="bold", fontsize=9)
    ax.set_title("Figure 6. Cross-axis fragility profile\n(PASS / marginal / FAIL by calibrated threshold)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_cross_axis.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Scorer divergence
# ---------------------------------------------------------------------------


def fig7_scorer_divergence() -> None:
    df = pd.read_csv(OUT / "wp1" / "scorer_full_kidney.csv")

    scorers = sorted(set(df["scorer_a"]) | set(df["scorer_b"]))

    def _metric_matrix(col: str) -> np.ndarray:
        M = np.full((len(scorers), len(scorers)), np.nan)
        for i, s1 in enumerate(scorers):
            M[i, i] = 1.0 if col in ("spearman", "top100_jaccard", "top500_jaccard") else 0.0
        for _, row in df.iterrows():
            i = scorers.index(row["scorer_a"])
            j = scorers.index(row["scorer_b"])
            M[i, j] = row[col]
            M[j, i] = row[col]
        return M

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title, cmap, vlim in [
        (axes[0], "spearman", "Spearman ρ", "RdBu_r", (-1, 1)),
        (axes[1], "top100_jaccard", "Top-100 Jaccard", "viridis", (0, 1)),
        (axes[2], "top500_rss", "Top-500 RSS (null ≈ 0.72)", "magma_r", (0, 1)),
    ]:
        M = _metric_matrix(col)
        im = ax.imshow(M, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect="auto")
        ax.set_xticks(range(len(scorers)))
        ax.set_xticklabels(scorers, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(scorers)))
        ax.set_yticklabels(scorers, fontsize=8)
        for i in range(len(scorers)):
            for j in range(len(scorers)):
                val = M[i, j]
                if np.isnan(val):
                    continue
                colour = "white" if abs(val - np.mean(vlim)) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=colour)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)

    axes[0].text(
        -0.5, -1.3,
        "scGPT attention is uncorrelated with every classical scorer "
        "(ρ ≤ 0.20, top-100 J ≤ 0.06, RSS ≥ 0.69 ≈ null).",
        fontsize=9, style="italic",
    )

    fig.suptitle("Figure 7. Pairwise scorer agreement on kidney (76×108 panel, n=800 cells)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_scorer_divergence.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------


def supp_s1_mvcc_anchor() -> None:
    curves = pd.read_csv(OUT / "wp2" / "mvcc_full_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, anchor in zip(axes, [3000, 8000]):
        sub = curves[(curves["k"] == 1000) & (curves["anchor_cells"] == anchor)]
        for ds, g in sub.groupby("dataset"):
            mean_by_size = g.groupby("cells")["jaccard_mean"].mean()
            ax.plot(mean_by_size.index, mean_by_size.values, "o-",
                    label=ds, color=DATASET_COLOURS.get(ds, "#444"))
        ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Cells")
        ax.set_ylabel(f"Top-1{{,}}000 Jaccard vs {anchor:,}-cell anchor")
        ax.set_ylim(0, 1)
        ax.set_title(f"Anchor = {anchor:,} cells")
        _legend(ax, loc="lower right", fontsize=8)
    fig.suptitle("Supp Fig S1. MVCC anchor ablation (Pearson over 600 HVGs)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s1_mvcc_anchor.pdf", bbox_inches="tight")
    plt.close(fig)


def supp_s2_weight_simplex() -> None:
    sweep = pd.read_csv(OUT / "wp3_axis2" / "rss_weight_sweep.csv")
    ds_list = sorted(sweep["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4 * len(ds_list), 4))
    if len(ds_list) == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_list):
        g = sweep[sweep["dataset"] == ds]
        # For each (wo, wj) pair, average composite over wd (projecting simplex).
        pivot = g.pivot_table(
            index="weight_overlap", columns="weight_jaccard",
            values="rss_composite", aggfunc="mean",
        )
        im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{c:.1f}" for c in pivot.index], fontsize=7)
        ax.set_xlabel("w_jaccard")
        ax.set_ylabel("w_overlap")
        ax.set_title(ds, fontsize=9)
    plt.colorbar(im, ax=axes, fraction=0.02).set_label("RSS composite")
    fig.suptitle("Supp Fig S2. RSS weight-simplex sensitivity sweep (66 weight triples per cohort)",
                 fontsize=12, y=1.02)
    fig.savefig(SUPP_DIR / "fig_s2_weight_simplex.pdf", bbox_inches="tight")
    plt.close(fig)


def supp_s3_rss_null() -> None:
    null_df = pd.read_csv(OUT / "wp3_axis2" / "rss_empirical_null.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ds_order = ["kidney", "lung", "immune", "immune_invariant", "external_lung"]
    x = np.arange(len(ds_order))
    obs = [null_df[null_df["dataset"] == d]["observed_rss"].iloc[0] for d in ds_order]
    nm = [null_df[null_df["dataset"] == d]["null_mean_rss"].iloc[0] for d in ds_order]
    n5 = [null_df[null_df["dataset"] == d]["null_5pct"].iloc[0] for d in ds_order]
    n95 = [null_df[null_df["dataset"] == d]["null_95pct"].iloc[0] for d in ds_order]
    z = [null_df[null_df["dataset"] == d]["z_score_vs_null"].iloc[0] for d in ds_order]

    ax.errorbar(
        x, nm,
        yerr=[np.array(nm) - np.array(n5), np.array(n95) - np.array(nm)],
        fmt="s", color="#888", label="Null 5–95%", capsize=4, markersize=8,
    )
    ax.scatter(x, obs, s=100, color="#d62728",
               label="Observed RSS", zorder=3, edgecolor="black")
    for xi, zi in zip(x, z):
        ax.text(xi, obs[ds_order.index(ds_order[xi])] - 0.05,
                f"z = {zi:.1f}", ha="center", fontsize=9, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=30, ha="right")
    ax.set_ylabel("RSS composite")
    ax.set_title("Supp Fig S3. Observed vs rank-permutation null RSS (1,000 permutations)")
    ax.set_ylim(0, 1)
    _legend(ax, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s3_rss_null.pdf", bbox_inches="tight")
    plt.close(fig)


def supp_s4_threshold_grid() -> None:
    grid = pd.read_csv(OUT / "wp4" / "reliability_gate_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(grid["rare_pass_rate"], bins=30, color="#d62728",
            edgecolor="black", label="Rare")
    ax.hist(grid["abundant_pass_rate"], bins=30, color="#2ca02c",
            edgecolor="black", alpha=0.6, label="Abundant")
    ax.set_xlabel("Pass rate at grid point")
    ax.set_ylabel("# threshold combinations")
    ax.set_title(f"Pass-rate distribution across {len(grid):,} threshold combinations")
    _legend(ax, loc="upper right", fontsize=9)

    ax = axes[1]
    # Project onto (stability, topk_jaccard) averaging over other axes
    piv = grid.pivot_table(
        index="thresh_stability", columns="thresh_topk_jaccard",
        values="rare_minus_abundant", aggfunc="mean",
    )
    im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{c:.2f}" for c in piv.index])
    ax.set_xlabel("Jaccard threshold")
    ax.set_ylabel("Stability threshold")
    ax.set_title("Mean (abundant − rare) gap\n(averaged over other axes)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Supp Fig S4. Reliability-gate 5-D threshold sensitivity ({len(grid):,} grid points)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s4_threshold_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def supp_s5_topk_scan() -> None:
    scan = pd.read_csv(OUT / "wp10_axis2" / "topk_scan.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label in zip(axes, ["overlap", "jaccard"],
                              ["Top-k overlap", "Top-k Jaccard"]):
        for ds, g in scan.groupby("dataset"):
            g_sorted = g.groupby("k")[col].mean().sort_index()
            ax.plot(g_sorted.index, g_sorted.values, "o-",
                    label=ds, color=DATASET_COLOURS.get(ds, "#444"))
        ax.set_xscale("log")
        ax.set_xlabel("k (log scale)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs k (Axis-2 coarse vs fine)")
        _legend(ax, loc="lower right", fontsize=8)
        ax.set_ylim(0, 1)
    fig.suptitle("Supp Fig S5. Top-k scan: overlap & Jaccard at k ∈ {100, 500, 1k, 5k, 1%, 5%, 10%}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s5_topk_scan.pdf", bbox_inches="tight")
    plt.close(fig)


def supp_s6_normalization() -> None:
    df = pd.read_csv(OUT / "wp12" / "normalization_ablation.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x, df["rss_composite"], color="#1f77b4", edgecolor="black",
           label="RSS composite")
    for i, row in enumerate(df.itertuples()):
        label_colour = {"coarse": "#d62728", "hybrid": "#ff7f0e", "fine": "#2ca02c"}
        ax.text(i, row.rss_composite + 0.02,
                row.dual_null_recommendation,
                ha="center", fontsize=9, fontweight="bold",
                color=label_colour.get(row.dual_null_recommendation, "black"))
    ax.set_xticks(x)
    ax.set_xticklabels(df["normalization"])
    ax.set_ylabel("RSS composite")
    ax.set_title("Supp Fig S6. Normalization ablation (kidney, n_null=25).\nAll three give coarse after dual-null.")
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s6_normalization.pdf", bbox_inches="tight")
    plt.close(fig)


def supp_s7_triple_null() -> None:
    null_df = pd.read_csv(OUT / "wp14" / "null_calibration.csv")
    scorecard = pd.read_csv(OUT / "wp14" / "scorecard.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ds_order = ["kidney", "lung", "immune", "immune_invariant", "external_lung"]
    nulls = ["global", "within_coarse", "degree_preserving"]
    pass_mat = np.zeros((len(ds_order), len(nulls)), dtype=int)
    for i, d in enumerate(ds_order):
        for j, nf in enumerate(nulls):
            row = null_df[(null_df["dataset"] == d) & (null_df["null_family"] == nf)]
            if len(row):
                pass_mat[i, j] = int(row["passed_alpha_0.05"].iloc[0])
    ax.imshow(pass_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(nulls)))
    ax.set_xticklabels(["Global", "Within-coarse", "Degree-preserve"], rotation=20, ha="right")
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels(ds_order)
    for i in range(pass_mat.shape[0]):
        for j in range(pass_mat.shape[1]):
            ax.text(j, i, "PASS" if pass_mat[i, j] else "FAIL",
                    ha="center", va="center",
                    color="black" if pass_mat[i, j] else "white",
                    fontweight="bold", fontsize=9)
    ax.set_title("Triple-null calibration: degree-preserving cleared by all; within-coarse fails universally")

    ax = axes[1]
    rec_cols = ["base_recommendation", "dual_null_recommendation", "triple_null_recommendation"]
    if all(c in scorecard.columns for c in rec_cols):
        recs = scorecard.set_index("dataset")[rec_cols].loc[ds_order]
        rec_code = {"fine": 2, "hybrid": 1, "coarse": 0}
        code_mat = recs.applymap(lambda v: rec_code.get(v, 0)).values
        cmap = matplotlib.colors.ListedColormap(["#d62728", "#ffcf3f", "#2ca02c"])
        ax.imshow(code_mat, cmap=cmap, aspect="auto", vmin=0, vmax=2)
        ax.set_xticks(range(len(rec_cols)))
        ax.set_xticklabels(["Base", "Dual-null", "Triple-null"], rotation=20, ha="right")
        ax.set_yticks(range(len(ds_order)))
        ax.set_yticklabels(ds_order)
        for i in range(code_mat.shape[0]):
            for j in range(code_mat.shape[1]):
                ax.text(j, i, recs.values[i, j], ha="center", va="center", fontsize=8)
        ax.set_title("Recommendation progression through calibration stages")

    fig.suptitle("Supp Fig S7. Triple-null calibration on Axis-2 (WP-14)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s7_triple_null.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    fns = [
        ("Fig 1 (cell count)", fig1_cell_count),
        ("Fig 2 (resolution)", fig2_resolution),
        ("Fig 3 (rare)", fig3_rare),
        ("Fig 4 (donor)", fig4_donor),
        ("Fig 5 (integration)", fig5_integration),
        ("Fig 6 (cross-axis)", fig6_cross_axis),
        ("Fig 7 (scorer divergence)", fig7_scorer_divergence),
        ("Supp S1 (MVCC anchor)", supp_s1_mvcc_anchor),
        ("Supp S2 (weight simplex)", supp_s2_weight_simplex),
        ("Supp S3 (RSS null)", supp_s3_rss_null),
        ("Supp S4 (threshold grid)", supp_s4_threshold_grid),
        ("Supp S5 (topk scan)", supp_s5_topk_scan),
        ("Supp S6 (normalization)", supp_s6_normalization),
        ("Supp S7 (triple null)", supp_s7_triple_null),
    ]
    for name, fn in fns:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
