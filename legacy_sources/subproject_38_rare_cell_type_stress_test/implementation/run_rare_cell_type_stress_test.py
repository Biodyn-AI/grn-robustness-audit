#!/usr/bin/env python3
"""
Rare Cell-Type Stress Test.

Project goal:
- quantify whether mechanistic edge inference degrades in rare cell types,
- measure stability and null separation by cell type,
- produce a practical minimum cell-type size guideline.

The pipeline uses existing processed single-cell datasets and TRRUST priors.
All outputs are written under this subproject folder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import mannwhitneyu, rankdata, spearmanr


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/Users/ihorkendiukhov/biodyn-work")
PROJECT_ROOT = REPO_ROOT / "subproject_38_rare_cell_type_stress_test"
IMPLEMENTATION_DIR = PROJECT_ROOT / "implementation"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = IMPLEMENTATION_DIR / "results"

DATASETS = {
    "kidney": REPO_ROOT / "single_cell_mechinterp" / "outputs" / "tabula_sapiens_processed.h5ad",
    "lung": REPO_ROOT
    / "single_cell_mechinterp"
    / "outputs"
    / "probe_benchmark"
    / "lung_hvg5000_processed.h5ad",
    "immune": REPO_ROOT
    / "single_cell_mechinterp"
    / "outputs"
    / "tabula_sapiens_immune_subset_hpn_processed.h5ad",
    "external_lung": REPO_ROOT
    / "single_cell_mechinterp"
    / "outputs"
    / "invariant_causal_edges"
    / "external_lung"
    / "processed.h5ad",
}

TRRUST_PATH = REPO_ROOT / "single_cell_mechinterp" / "external" / "networks" / "trrust_human.tsv"


# ---------------------------------------------------------------------------
# Analysis thresholds
# ---------------------------------------------------------------------------
STABILITY_THRESHOLD = 0.75
TOPK_JACCARD_THRESHOLD = 0.35
NULL_AUC_THRESHOLD = 0.65
MIN_CELLTYPE_SIZE_THRESHOLD = 80


@dataclass
class Panel:
    """Edge panel and matrix mappings for a dataset."""

    selected_genes: List[str]
    tf_genes: List[str]
    target_genes: List[str]
    edges: pd.DataFrame
    tf_cols: np.ndarray
    target_cols: np.ndarray
    edge_tf_map: np.ndarray
    edge_target_map: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rare cell-type stress test pipeline.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-cells",
        type=int,
        default=40,
        help="Minimum cells per cell type required for analysis.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=8,
        help="Bootstrap replicates per cell type.",
    )
    parser.add_argument(
        "--subsample-frac",
        type=float,
        default=0.8,
        help="Fraction of cells sampled per bootstrap replicate.",
    )
    parser.add_argument(
        "--max-tfs",
        type=int,
        default=80,
        help="Maximum TF count retained from TRRUST panel in each dataset.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=320,
        help="Maximum high-variance targets retained from TRRUST panel in each dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-k edges for overlap and tail-gap metrics.",
    )
    parser.add_argument(
        "--exclude-external-lung",
        action="store_true",
        help="Exclude the external lung dataset from analysis.",
    )
    parser.add_argument(
        "--skip-shared-panel-sensitivity",
        action="store_true",
        help="Skip shared-panel sensitivity analysis (adversarial robustness check).",
    )
    parser.add_argument(
        "--shared-max-tfs",
        type=int,
        default=70,
        help="Maximum TFs in the shared-panel sensitivity edge set.",
    )
    parser.add_argument(
        "--shared-max-targets",
        type=int,
        default=260,
        help="Maximum targets in the shared-panel sensitivity edge set.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_trrust(path: Path) -> pd.DataFrame:
    """
    Load TRRUST edge table with a stable schema and remove duplicate TF-target pairs.
    """
    trrust = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["tf", "target", "regulation", "pmid"],
        usecols=[0, 1, 2, 3],
    )
    trrust["tf"] = trrust["tf"].astype(str)
    trrust["target"] = trrust["target"].astype(str)
    trrust = trrust.drop_duplicates(subset=["tf", "target"]).reset_index(drop=True)
    return trrust


def resolve_var_names(adata: ad.AnnData) -> None:
    """
    Convert Ensembl-like var names to gene symbols if feature_name exists.
    """
    if adata.n_vars == 0:
        return

    first_name = str(adata.var_names[0])
    if first_name.startswith("ENSG") and "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).to_numpy()
        adata.var_names_make_unique()


def extract_gene_names_for_path(path: Path) -> pd.Index:
    """
    Read gene names from a dataset path without loading full matrices into memory.
    """
    adata = ad.read_h5ad(path, backed="r")
    try:
        var_names = pd.Index(adata.var_names.astype(str))
        if len(var_names) > 0 and str(var_names[0]).startswith("ENSG") and "feature_name" in adata.var.columns:
            var_names = pd.Index(adata.var["feature_name"].astype(str).to_numpy())
        return var_names
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


def collect_dataset_gene_sets(dataset_map: Dict[str, Path]) -> Dict[str, set]:
    """
    Collect per-dataset gene symbol sets for shared-panel construction.
    """
    gene_sets: Dict[str, set] = {}
    for name, path in dataset_map.items():
        genes = extract_gene_names_for_path(path)
        gene_sets[name] = set(genes.tolist())
    return gene_sets


def choose_cell_type_key(adata: ad.AnnData) -> str:
    if "cell_type" in adata.obs.columns:
        return "cell_type"
    if "broad_cell_class" in adata.obs.columns:
        return "broad_cell_class"
    raise ValueError("No usable cell-type column found (expected cell_type or broad_cell_class).")


def dense_slice(adata: ad.AnnData, genes: List[str]) -> np.ndarray:
    """
    Materialize selected gene matrix as dense float32.
    """
    X = adata[:, genes].X
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def variance_by_gene(X: np.ndarray) -> np.ndarray:
    """
    Compute per-gene variance from dense matrix.
    """
    return np.var(X, axis=0, ddof=1)


def build_panel(
    adata: ad.AnnData,
    trrust: pd.DataFrame,
    max_tfs: int,
    max_targets: int,
) -> Panel:
    """
    Build a compact TRRUST-derived edge panel for a dataset.

    Design:
    - retain only genes present in the dataset,
    - prioritize TFs with stronger prior connectivity,
    - prioritize targets with higher variance for stronger signal contrast.
    """
    genes = pd.Index(adata.var_names.astype(str))
    gene_set = set(genes)

    edges = trrust[(trrust["tf"].isin(gene_set)) & (trrust["target"].isin(gene_set))].copy()
    edges = edges[edges["tf"] != edges["target"]].copy()
    if edges.empty:
        raise ValueError("No TRRUST edges overlap with dataset genes.")

    # TF prioritization by prior degree.
    tf_degree = edges["tf"].value_counts()
    selected_tfs = tf_degree.head(max_tfs).index.tolist()
    edges = edges[edges["tf"].isin(selected_tfs)].copy()

    # Target prioritization by expression variance.
    target_candidates = sorted(edges["target"].unique())
    X_targets = dense_slice(adata, target_candidates)
    target_var = pd.Series(variance_by_gene(X_targets), index=target_candidates)
    selected_targets = target_var.sort_values(ascending=False).head(max_targets).index.tolist()
    edges = edges[edges["target"].isin(selected_targets)].copy()

    # Re-trim TFs after target filter so panel remains compact and balanced.
    selected_tfs = edges["tf"].value_counts().head(max_tfs).index.tolist()
    edges = edges[edges["tf"].isin(selected_tfs)].copy()
    edges = edges.drop_duplicates(subset=["tf", "target"]).reset_index(drop=True)
    if edges.empty:
        raise ValueError("Filtered panel became empty after TF/target trimming.")

    tf_genes = sorted(edges["tf"].unique())
    target_genes = sorted(edges["target"].unique())
    selected_genes = tf_genes + [g for g in target_genes if g not in tf_genes]

    gene_to_col = {g: i for i, g in enumerate(selected_genes)}
    tf_to_local = {g: i for i, g in enumerate(tf_genes)}
    target_to_local = {g: i for i, g in enumerate(target_genes)}

    tf_cols = np.array([gene_to_col[g] for g in tf_genes], dtype=np.int32)
    target_cols = np.array([gene_to_col[g] for g in target_genes], dtype=np.int32)
    edge_tf_map = np.array([tf_to_local[g] for g in edges["tf"]], dtype=np.int32)
    edge_target_map = np.array([target_to_local[g] for g in edges["target"]], dtype=np.int32)

    return Panel(
        selected_genes=selected_genes,
        tf_genes=tf_genes,
        target_genes=target_genes,
        edges=edges,
        tf_cols=tf_cols,
        target_cols=target_cols,
        edge_tf_map=edge_tf_map,
        edge_target_map=edge_target_map,
    )


def build_fixed_edges_from_gene_set(
    gene_set: set,
    trrust: pd.DataFrame,
    max_tfs: int,
    max_targets: int,
) -> pd.DataFrame:
    """
    Build a shared fixed edge table from a provided gene universe.
    """
    edges = trrust[(trrust["tf"].isin(gene_set)) & (trrust["target"].isin(gene_set))].copy()
    edges = edges[edges["tf"] != edges["target"]].copy()
    if edges.empty:
        raise ValueError("No shared TRRUST edges in the supplied gene universe.")

    tf_degree = edges["tf"].value_counts()
    selected_tfs = tf_degree.head(max_tfs).index.tolist()
    edges = edges[edges["tf"].isin(selected_tfs)].copy()

    target_degree = edges["target"].value_counts()
    selected_targets = target_degree.head(max_targets).index.tolist()
    edges = edges[edges["target"].isin(selected_targets)].copy()

    selected_tfs = edges["tf"].value_counts().head(max_tfs).index.tolist()
    edges = edges[edges["tf"].isin(selected_tfs)].copy()
    edges = edges.drop_duplicates(subset=["tf", "target"]).reset_index(drop=True)
    if edges.empty:
        raise ValueError("Shared edge table is empty after TF/target truncation.")
    return edges


def build_panel_from_edges(adata: ad.AnnData, fixed_edges: pd.DataFrame) -> Panel:
    """
    Build panel mappings from an externally supplied fixed edge table.
    """
    genes = set(pd.Index(adata.var_names.astype(str)).tolist())
    edges = fixed_edges[(fixed_edges["tf"].isin(genes)) & (fixed_edges["target"].isin(genes))].copy()
    edges = edges[edges["tf"] != edges["target"]].drop_duplicates(subset=["tf", "target"]).reset_index(drop=True)
    if edges.empty:
        raise ValueError("Fixed edge table has zero overlap with dataset genes.")

    tf_genes = sorted(edges["tf"].unique())
    target_genes = sorted(edges["target"].unique())
    selected_genes = tf_genes + [g for g in target_genes if g not in tf_genes]

    gene_to_col = {g: i for i, g in enumerate(selected_genes)}
    tf_to_local = {g: i for i, g in enumerate(tf_genes)}
    target_to_local = {g: i for i, g in enumerate(target_genes)}

    tf_cols = np.array([gene_to_col[g] for g in tf_genes], dtype=np.int32)
    target_cols = np.array([gene_to_col[g] for g in target_genes], dtype=np.int32)
    edge_tf_map = np.array([tf_to_local[g] for g in edges["tf"]], dtype=np.int32)
    edge_target_map = np.array([target_to_local[g] for g in edges["target"]], dtype=np.int32)

    return Panel(
        selected_genes=selected_genes,
        tf_genes=tf_genes,
        target_genes=target_genes,
        edges=edges,
        tf_cols=tf_cols,
        target_cols=target_cols,
        edge_tf_map=edge_tf_map,
        edge_target_map=edge_target_map,
    )


def zscore_columns(X: np.ndarray) -> np.ndarray:
    """
    Column-wise z-score with safe handling for near-constant columns.
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    valid = stds > 1e-8
    Z = np.zeros_like(X, dtype=np.float32)
    if np.any(valid):
        Z[:, valid] = (X[:, valid] - means[valid]) / stds[valid]
    return Z


def score_edges(tf_expr: np.ndarray, target_expr: np.ndarray, panel: Panel) -> np.ndarray:
    """
    Compute Pearson correlation-based edge scores for the panel.
    """
    n_cells = tf_expr.shape[0]
    if n_cells < 3:
        return np.zeros(panel.edges.shape[0], dtype=np.float32)

    Z_tf = zscore_columns(tf_expr)
    Z_target = zscore_columns(target_expr)

    corr = (Z_tf.T @ Z_target) / float(n_cells - 1)
    scores = corr[panel.edge_tf_map, panel.edge_target_map]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scores.astype(np.float32), -1.0, 1.0)


def jaccard_index(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(a)
    sb = set(b)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return len(sa & sb) / union


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Fast top-k indices by absolute score.
    """
    k = max(1, min(k, scores.shape[0]))
    idx = np.argpartition(np.abs(scores), -k)[-k:]
    return idx[np.argsort(np.abs(scores[idx]))[::-1]]


def empirical_auc(observed: np.ndarray, null: np.ndarray) -> float:
    """
    AUC-style separation metric: P(observed > null) on |score|.
    """
    obs = np.abs(observed).astype(np.float64)
    nul = np.abs(null).astype(np.float64)
    n_obs = obs.size
    n_null = nul.size
    if n_obs == 0 or n_null == 0:
        return float("nan")

    merged = np.concatenate([obs, nul])
    ranks = rankdata(merged, method="average")
    rank_sum_obs = np.sum(ranks[:n_obs])
    u_stat = rank_sum_obs - (n_obs * (n_obs + 1) / 2.0)
    return float(u_stat / (n_obs * n_null))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Effect size for two independent samples.
    """
    if x.size == 0 or y.size == 0:
        return float("nan")
    comp = np.subtract.outer(x, y)
    return float((np.sum(comp > 0) - np.sum(comp < 0)) / (x.size * y.size))


def summarize_pairwise(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def classify_failure_modes(celltype_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign interpretable failure categories for unstable cell types.
    """
    df = celltype_df.copy()
    tf_det_q25 = df["tf_detection_rate"].quantile(0.25)
    gene_det_q25 = df["mean_detected_genes"].quantile(0.25)

    def classify(row: pd.Series) -> str:
        if row["is_reliable"]:
            return "Reliable"
        if row["null_auc_mean"] < NULL_AUC_THRESHOLD:
            return "Null-overlap (weak signal)"
        if row["stability_spearman_mean"] < STABILITY_THRESHOLD and row["tf_detection_rate"] <= tf_det_q25:
            return "TF-dropout instability"
        if row["stability_spearman_mean"] < STABILITY_THRESHOLD and row["mean_detected_genes"] <= gene_det_q25:
            return "Low-complexity instability"
        if row["topk_jaccard_mean"] < TOPK_JACCARD_THRESHOLD:
            return "Top-k volatility"
        return "Mixed instability"

    df["failure_mode"] = df.apply(classify, axis=1)
    return df


def fit_min_cells_for_metric(metric_df: pd.DataFrame, metric_col: str, target: float) -> pd.DataFrame:
    """
    Linear fit in log10(cell_count) space to estimate minimum required cells.
    """
    rows: List[Dict[str, object]] = []
    for dataset, ddf in metric_df.groupby("dataset"):
        x = np.log10(ddf["cell_count"].astype(float).to_numpy())
        y = ddf[metric_col].astype(float).to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        observed_min_cells = float(np.min(ddf["cell_count"].astype(float)))
        observed_max_cells = float(np.max(ddf["cell_count"].astype(float)))
        if x.size < 3 or np.allclose(x, x[0]):
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric_col,
                    "target": target,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "estimate_status": "insufficient_variation",
                    "raw_min_cells_estimate": np.nan,
                    "min_cells_estimate": np.nan,
                    "observed_min_cells": observed_min_cells,
                    "observed_max_cells": observed_max_cells,
                }
            )
            continue

        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        pred_min = float(slope * x_min + intercept)
        pred_max = float(slope * x_max + intercept)
        pred_lo = min(pred_min, pred_max)
        pred_hi = max(pred_min, pred_max)

        raw_min_cells_est = np.nan
        if abs(slope) > 1e-12:
            raw_min_cells_est = float(10 ** ((target - intercept) / slope))
        status = "within_observed_range"
        min_cells_est = raw_min_cells_est

        if target < pred_lo:
            status = "threshold_below_observed_prediction_range"
            min_cells_est = observed_min_cells
        elif target > pred_hi:
            status = "above_observed_range_unresolved"
            min_cells_est = np.nan

        if np.isfinite(min_cells_est):
            min_cells_est = max(1.0, float(min_cells_est))

        rows.append(
            {
                "dataset": dataset,
                "metric": metric_col,
                "target": target,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "estimate_status": status,
                "raw_min_cells_estimate": raw_min_cells_est,
                "min_cells_estimate": min_cells_est,
                "observed_min_cells": observed_min_cells,
                "observed_max_cells": observed_max_cells,
            }
        )

    return pd.DataFrame(rows)


def plot_stability_curve(celltype_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=celltype_df,
        x="cell_count",
        y="stability_spearman_mean",
        hue="dataset",
        style="rarity_group",
        s=70,
        alpha=0.9,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.axhline(STABILITY_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="stability threshold")
    ax.set_title("Edge Stability Increases with Cell-Type Size")
    ax.set_xlabel("Cell count per cell type (log scale)")
    ax.set_ylabel("Bootstrap rank stability (Spearman)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_stability_vs_cell_count.png", dpi=220)
    plt.close(fig)


def plot_null_separation_curve(celltype_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=celltype_df,
        x="cell_count",
        y="null_auc_mean",
        hue="dataset",
        style="rarity_group",
        s=70,
        alpha=0.9,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.axhline(NULL_AUC_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="null threshold")
    ax.set_title("Null Separation Improves with Cell-Type Size")
    ax.set_xlabel("Cell count per cell type (log scale)")
    ax.set_ylabel("Observed-vs-null separation (AUC)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_null_auc_vs_cell_count.png", dpi=220)
    plt.close(fig)


def plot_group_boxplots(celltype_df: pd.DataFrame) -> None:
    order = ["rare", "intermediate", "abundant"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(
        data=celltype_df,
        x="rarity_group",
        y="stability_spearman_mean",
        order=order,
        hue="dataset",
        ax=axes[0],
    )
    axes[0].axhline(STABILITY_THRESHOLD, linestyle="--", color="black", linewidth=1.0)
    axes[0].set_title("Stability by rarity group")
    axes[0].set_xlabel("Cell-type rarity group")
    axes[0].set_ylabel("Bootstrap rank stability")
    axes[0].set_ylim(0.0, 1.02)

    sns.boxplot(
        data=celltype_df,
        x="rarity_group",
        y="null_auc_mean",
        order=order,
        hue="dataset",
        ax=axes[1],
    )
    axes[1].axhline(NULL_AUC_THRESHOLD, linestyle="--", color="black", linewidth=1.0)
    axes[1].set_title("Null separation by rarity group")
    axes[1].set_xlabel("Cell-type rarity group")
    axes[1].set_ylabel("Observed-vs-null AUC")
    axes[1].set_ylim(0.0, 1.02)

    # Keep one legend for clean layout.
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "fig3_rare_vs_abundant_boxplots.png", dpi=220)
    plt.close(fig)


def plot_failure_modes(celltype_df: pd.DataFrame) -> None:
    subset = celltype_df[
        (celltype_df["rarity_group"] == "rare") & (~celltype_df["is_reliable"])
    ].copy()
    if subset.empty:
        return
    table = pd.crosstab(subset["dataset"], subset["failure_mode"])
    table = table.reindex(sorted(table.columns), axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    table.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Failure Modes in Rare, Unreliable Cell Types")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Cell-type count")
    ax.legend(title="Failure mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_failure_modes_rare_types.png", dpi=220)
    plt.close(fig)


def plot_guidelines(guideline_df: pd.DataFrame) -> None:
    if guideline_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = guideline_df.copy()
    sns.barplot(
        data=plot_df,
        x="dataset",
        y="combined_guideline",
        hue="dataset",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.axhline(MIN_CELLTYPE_SIZE_THRESHOLD, linestyle="--", color="black", linewidth=1.0)
    ax.set_title("Estimated Minimum Cell-Type Size for Reliable Inference")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Estimated minimum cells")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_minimum_cell_size_guideline.png", dpi=220)
    plt.close(fig)


def analyze_dataset(
    dataset_name: str,
    dataset_path: Path,
    trrust: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    analysis_tag: str,
    fixed_edges: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze one dataset and return:
    - per-cell-type metrics,
    - top-edge table per cell type,
    - dataset-level metadata summary.
    """
    print(f"[{dataset_name}] loading {dataset_path}")
    adata = ad.read_h5ad(dataset_path)
    resolve_var_names(adata)
    cell_key = choose_cell_type_key(adata)
    adata.obs[cell_key] = adata.obs[cell_key].astype(str)

    counts = adata.obs[cell_key].value_counts()
    eligible_counts = counts[counts >= args.min_cells].sort_values(ascending=False)
    if eligible_counts.empty:
        raise ValueError(f"[{dataset_name}] no eligible cell types with >= {args.min_cells} cells.")

    q25 = float(eligible_counts.quantile(0.25))
    q75 = float(eligible_counts.quantile(0.75))

    def rarity_group(n: int) -> str:
        if n <= q25:
            return "rare"
        if n >= q75:
            return "abundant"
        return "intermediate"

    if fixed_edges is None:
        panel = build_panel(adata, trrust, max_tfs=args.max_tfs, max_targets=args.max_targets)
        panel_mode = "adaptive_panel"
    else:
        panel = build_panel_from_edges(adata, fixed_edges)
        panel_mode = "shared_panel"
    print(
        f"[{dataset_name}][{analysis_tag}] panel: edges={panel.edges.shape[0]}, "
        f"TFs={len(panel.tf_genes)}, targets={len(panel.target_genes)}"
    )

    X = dense_slice(adata, panel.selected_genes)
    obs_cell_types = adata.obs[cell_key].to_numpy()

    celltype_rows: List[Dict[str, object]] = []
    top_edge_rows: List[Dict[str, object]] = []

    for cell_type, n_cells in eligible_counts.items():
        idx = np.where(obs_cell_types == cell_type)[0]
        if idx.size < args.min_cells:
            continue

        sample_size = int(round(args.subsample_frac * idx.size))
        sample_size = max(args.min_cells, sample_size)
        sample_size = min(sample_size, idx.size)

        obs_scores_list: List[np.ndarray] = []
        null_auc_cellperm_values: List[float] = []
        null_auc_geneshuffle_values: List[float] = []
        null_auc_conservative_values: List[float] = []
        tail_gap_cellperm_values: List[float] = []
        tail_gap_geneshuffle_values: List[float] = []
        tail_gap_conservative_values: List[float] = []

        for _ in range(args.replicates):
            if sample_size == idx.size:
                rep_idx = idx
            else:
                rep_idx = rng.choice(idx, size=sample_size, replace=False)
            X_rep = X[rep_idx, :]
            tf_rep = X_rep[:, panel.tf_cols]
            target_rep = X_rep[:, panel.target_cols]

            obs_scores = score_edges(tf_rep, target_rep, panel)
            perm_cells = rng.permutation(sample_size)
            null_scores_cellperm = score_edges(tf_rep, target_rep[perm_cells], panel)
            perm_targets = rng.permutation(target_rep.shape[1])
            null_scores_geneshuffle = score_edges(tf_rep, target_rep[:, perm_targets], panel)

            obs_scores_list.append(obs_scores)
            auc_cell = empirical_auc(obs_scores, null_scores_cellperm)
            auc_gene = empirical_auc(obs_scores, null_scores_geneshuffle)
            null_auc_cellperm_values.append(auc_cell)
            null_auc_geneshuffle_values.append(auc_gene)
            null_auc_conservative_values.append(min(auc_cell, auc_gene))

            k = max(1, min(args.top_k, obs_scores.shape[0]))
            top_obs = np.abs(obs_scores[topk_indices(obs_scores, k)])
            null_q99_cell = float(np.quantile(np.abs(null_scores_cellperm), 0.99))
            null_q99_gene = float(np.quantile(np.abs(null_scores_geneshuffle), 0.99))
            tail_gap_cellperm_values.append(float(np.mean(top_obs) - null_q99_cell))
            tail_gap_geneshuffle_values.append(float(np.mean(top_obs) - null_q99_gene))
            tail_gap_conservative_values.append(float(np.mean(top_obs) - max(null_q99_cell, null_q99_gene)))

        pairwise_rho: List[float] = []
        pairwise_jaccard: List[float] = []
        k = max(1, min(args.top_k, panel.edges.shape[0]))
        for i, j in combinations(range(args.replicates), 2):
            s1 = np.abs(obs_scores_list[i])
            s2 = np.abs(obs_scores_list[j])
            rho = spearmanr(s1, s2).statistic
            if not np.isfinite(rho):
                rho = 0.0
            pairwise_rho.append(float(rho))

            top_i = topk_indices(obs_scores_list[i], k)
            top_j = topk_indices(obs_scores_list[j], k)
            pairwise_jaccard.append(jaccard_index(top_i, top_j))

        stability_mean, stability_std = summarize_pairwise(pairwise_rho)
        jaccard_mean, jaccard_std = summarize_pairwise(pairwise_jaccard)
        null_auc_cellperm_mean, null_auc_cellperm_std = summarize_pairwise(null_auc_cellperm_values)
        null_auc_geneshuffle_mean, null_auc_geneshuffle_std = summarize_pairwise(null_auc_geneshuffle_values)
        null_auc_conservative_mean, null_auc_conservative_std = summarize_pairwise(null_auc_conservative_values)
        tail_gap_cellperm_mean, tail_gap_cellperm_std = summarize_pairwise(tail_gap_cellperm_values)
        tail_gap_geneshuffle_mean, tail_gap_geneshuffle_std = summarize_pairwise(tail_gap_geneshuffle_values)
        tail_gap_conservative_mean, tail_gap_conservative_std = summarize_pairwise(tail_gap_conservative_values)

        # Score panel using the full cell type to report representative top edges.
        X_full = X[idx, :]
        full_scores = score_edges(X_full[:, panel.tf_cols], X_full[:, panel.target_cols], panel)
        top_idx = topk_indices(full_scores, min(20, panel.edges.shape[0]))
        for rank_pos, edge_idx in enumerate(top_idx, start=1):
            edge = panel.edges.iloc[int(edge_idx)]
            top_edge_rows.append(
                {
                    "analysis_tag": analysis_tag,
                    "dataset": dataset_name,
                    "cell_type": cell_type,
                    "rank": rank_pos,
                    "tf": edge["tf"],
                    "target": edge["target"],
                    "score": float(full_scores[int(edge_idx)]),
                    "abs_score": float(abs(full_scores[int(edge_idx)])),
                }
            )

        # Basic quality diagnostics used to interpret rare-type failures.
        X_cell = X_full
        mean_detected_genes = float(np.mean(np.sum(X_cell > 0, axis=1)))
        tf_detection_rate = float(np.mean(X_cell[:, panel.tf_cols] > 0))
        target_detection_rate = float(np.mean(X_cell[:, panel.target_cols] > 0))
        median_tf_var = float(np.median(np.var(X_cell[:, panel.tf_cols], axis=0, ddof=1)))

        is_reliable = (
            (stability_mean >= STABILITY_THRESHOLD)
            and (jaccard_mean >= TOPK_JACCARD_THRESHOLD)
            and (null_auc_cellperm_mean >= NULL_AUC_THRESHOLD)
            and (tail_gap_cellperm_mean > 0.0)
            and (idx.size >= MIN_CELLTYPE_SIZE_THRESHOLD)
        )

        celltype_rows.append(
            {
                "analysis_tag": analysis_tag,
                "dataset": dataset_name,
                "cell_type": cell_type,
                "cell_count": int(idx.size),
                "rarity_group": rarity_group(int(idx.size)),
                "panel_mode": panel_mode,
                "sample_size": int(sample_size),
                "n_edges": int(panel.edges.shape[0]),
                "n_tfs": int(len(panel.tf_genes)),
                "n_targets": int(len(panel.target_genes)),
                "stability_spearman_mean": stability_mean,
                "stability_spearman_std": stability_std,
                "topk_jaccard_mean": jaccard_mean,
                "topk_jaccard_std": jaccard_std,
                "null_auc_mean": null_auc_cellperm_mean,
                "null_auc_std": null_auc_cellperm_std,
                "null_auc_cellperm_mean": null_auc_cellperm_mean,
                "null_auc_cellperm_std": null_auc_cellperm_std,
                "null_auc_geneshuffle_mean": null_auc_geneshuffle_mean,
                "null_auc_geneshuffle_std": null_auc_geneshuffle_std,
                "null_auc_conservative_mean": null_auc_conservative_mean,
                "null_auc_conservative_std": null_auc_conservative_std,
                "null_tail_gap_mean": tail_gap_cellperm_mean,
                "null_tail_gap_std": tail_gap_cellperm_std,
                "null_tail_gap_cellperm_mean": tail_gap_cellperm_mean,
                "null_tail_gap_cellperm_std": tail_gap_cellperm_std,
                "null_tail_gap_geneshuffle_mean": tail_gap_geneshuffle_mean,
                "null_tail_gap_geneshuffle_std": tail_gap_geneshuffle_std,
                "null_tail_gap_conservative_mean": tail_gap_conservative_mean,
                "null_tail_gap_conservative_std": tail_gap_conservative_std,
                "mean_detected_genes": mean_detected_genes,
                "tf_detection_rate": tf_detection_rate,
                "target_detection_rate": target_detection_rate,
                "median_tf_variance": median_tf_var,
                "is_reliable": bool(is_reliable),
            }
        )

    per_celltype = pd.DataFrame(celltype_rows)
    top_edges = pd.DataFrame(top_edge_rows)
    dataset_summary = pd.DataFrame(
        [
            {
                "analysis_tag": analysis_tag,
                "dataset": dataset_name,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "celltype_key": cell_key,
                "panel_mode": panel_mode,
                "eligible_cell_types": int(eligible_counts.shape[0]),
                "q25_count": q25,
                "q75_count": q75,
                "panel_edges": int(panel.edges.shape[0]),
                "panel_tfs": int(len(panel.tf_genes)),
                "panel_targets": int(len(panel.target_genes)),
            }
        ]
    )

    return per_celltype, top_edges, dataset_summary


def build_group_comparison(celltype_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset, ddf in celltype_df.groupby("dataset"):
        rare = ddf[ddf["rarity_group"] == "rare"]
        abundant = ddf[ddf["rarity_group"] == "abundant"]
        for metric in ["stability_spearman_mean", "topk_jaccard_mean", "null_auc_mean"]:
            x = rare[metric].to_numpy(dtype=float)
            y = abundant[metric].to_numpy(dtype=float)
            if x.size > 0 and y.size > 0:
                test = mannwhitneyu(x, y, alternative="less")
                p_value = float(test.pvalue)
                delta = cliffs_delta(x, y)
                effect = float(np.nanmean(y) - np.nanmean(x))
            else:
                p_value = np.nan
                delta = np.nan
                effect = np.nan

            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "rare_n": int(x.size),
                    "abundant_n": int(y.size),
                    "rare_mean": float(np.nanmean(x)) if x.size else np.nan,
                    "abundant_mean": float(np.nanmean(y)) if y.size else np.nan,
                    "abundant_minus_rare": effect,
                    "mannwhitney_p_less": p_value,
                    "cliffs_delta_rare_vs_abundant": delta,
                }
            )
    return pd.DataFrame(rows)


def build_global_comparison(
    celltype_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Pool rare vs abundant groups across tissues for an overall effect estimate.
    """
    rare = celltype_df[celltype_df["rarity_group"] == "rare"]
    abundant = celltype_df[celltype_df["rarity_group"] == "abundant"]

    if metrics is None:
        metrics = [
            "stability_spearman_mean",
            "topk_jaccard_mean",
            "null_auc_mean",
            "null_tail_gap_mean",
        ]

    rows: List[Dict[str, object]] = []
    for metric in metrics:
        x = rare[metric].to_numpy(dtype=float)
        y = abundant[metric].to_numpy(dtype=float)
        test = mannwhitneyu(x, y, alternative="less")
        rows.append(
            {
                "metric": metric,
                "rare_n": int(x.size),
                "abundant_n": int(y.size),
                "rare_mean": float(np.mean(x)),
                "abundant_mean": float(np.mean(y)),
                "abundant_minus_rare": float(np.mean(y) - np.mean(x)),
                "mannwhitney_p_less": float(test.pvalue),
                "cliffs_delta_rare_vs_abundant": cliffs_delta(x, y),
            }
        )
    return pd.DataFrame(rows)


def build_size_bin_summary(celltype_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize reliability as a function of absolute cell-type size bins.
    """
    df = celltype_df.copy()
    bins = [40, 100, 300, 1000, 3000, 200000]
    labels = ["40-99", "100-299", "300-999", "1000-2999", "3000+"]
    df["size_bin"] = pd.cut(df["cell_count"], bins=bins, labels=labels, right=False)
    summary = (
        df.groupby("size_bin", observed=False)
        .agg(
            n_cell_types=("cell_type", "count"),
            reliability_rate=("is_reliable", "mean"),
            mean_null_auc=("null_auc_mean", "mean"),
            mean_tail_gap=("null_tail_gap_mean", "mean"),
            mean_topk_jaccard=("topk_jaccard_mean", "mean"),
        )
        .reset_index()
    )
    return summary


def build_panel_sensitivity_summary(
    primary_df: pd.DataFrame,
    shared_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantify robustness of conclusions to adaptive vs shared edge panels.
    """
    if shared_df.empty:
        return pd.DataFrame()

    key_cols = ["dataset", "cell_type", "cell_count", "rarity_group"]
    merged = primary_df.merge(
        shared_df,
        on=key_cols,
        suffixes=("_adaptive", "_shared"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3 or np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
            return np.nan
        rho = spearmanr(a, b).statistic
        return float(rho) if np.isfinite(rho) else np.nan

    rows: List[Dict[str, object]] = []
    for dataset, ddf in merged.groupby("dataset"):
        rho_null = safe_spearman(
            ddf["null_auc_mean_adaptive"].to_numpy(dtype=float),
            ddf["null_auc_mean_shared"].to_numpy(dtype=float),
        )
        rho_jacc = safe_spearman(
            ddf["topk_jaccard_mean_adaptive"].to_numpy(dtype=float),
            ddf["topk_jaccard_mean_shared"].to_numpy(dtype=float),
        )
        rho_stab = safe_spearman(
            ddf["stability_spearman_mean_adaptive"].to_numpy(dtype=float),
            ddf["stability_spearman_mean_shared"].to_numpy(dtype=float),
        )
        reliability_agreement = float(np.mean(ddf["is_reliable_adaptive"] == ddf["is_reliable_shared"]))
        rows.append(
            {
                "dataset": dataset,
                "n_cell_types_overlap": int(ddf.shape[0]),
                "rho_null_auc_adaptive_vs_shared": rho_null,
                "rho_topk_jaccard_adaptive_vs_shared": rho_jacc,
                "rho_stability_adaptive_vs_shared": rho_stab,
                "reliability_agreement_rate": reliability_agreement,
                "reliability_rate_adaptive": float(np.mean(ddf["is_reliable_adaptive"])),
                "reliability_rate_shared": float(np.mean(ddf["is_reliable_shared"])),
            }
        )
    return pd.DataFrame(rows)


def build_null_model_summary(primary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize discrepancy between cell-permutation and gene-shuffle nulls.
    """
    df = primary_df.copy()
    df["auc_delta_gene_minus_cell"] = df["null_auc_geneshuffle_mean"] - df["null_auc_cellperm_mean"]
    df["tail_gap_delta_gene_minus_cell"] = (
        df["null_tail_gap_geneshuffle_mean"] - df["null_tail_gap_cellperm_mean"]
    )
    summary = (
        df.groupby(["dataset", "rarity_group"], observed=False)
        .agg(
            n_cell_types=("cell_type", "count"),
            mean_auc_cellperm=("null_auc_cellperm_mean", "mean"),
            mean_auc_geneshuffle=("null_auc_geneshuffle_mean", "mean"),
            mean_auc_conservative=("null_auc_conservative_mean", "mean"),
            mean_auc_delta_gene_minus_cell=("auc_delta_gene_minus_cell", "mean"),
            mean_tail_gap_cellperm=("null_tail_gap_cellperm_mean", "mean"),
            mean_tail_gap_geneshuffle=("null_tail_gap_geneshuffle_mean", "mean"),
            mean_tail_gap_conservative=("null_tail_gap_conservative_mean", "mean"),
            mean_tail_gap_delta_gene_minus_cell=("tail_gap_delta_gene_minus_cell", "mean"),
        )
        .reset_index()
    )
    return summary


def build_tissue_summary(celltype_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset, ddf in celltype_df.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "n_cell_types": int(ddf.shape[0]),
                "rare_cell_types": int((ddf["rarity_group"] == "rare").sum()),
                "abundant_cell_types": int((ddf["rarity_group"] == "abundant").sum()),
                "reliable_fraction_all": float(np.mean(ddf["is_reliable"])),
                "reliable_fraction_rare": float(np.mean(ddf.loc[ddf["rarity_group"] == "rare", "is_reliable"])),
                "reliable_fraction_abundant": float(
                    np.mean(ddf.loc[ddf["rarity_group"] == "abundant", "is_reliable"])
                ),
                "mean_stability_all": float(ddf["stability_spearman_mean"].mean()),
                "mean_stability_rare": float(ddf.loc[ddf["rarity_group"] == "rare", "stability_spearman_mean"].mean()),
                "mean_stability_abundant": float(
                    ddf.loc[ddf["rarity_group"] == "abundant", "stability_spearman_mean"].mean()
                ),
                "mean_null_auc_all": float(ddf["null_auc_mean"].mean()),
                "mean_null_auc_rare": float(ddf.loc[ddf["rarity_group"] == "rare", "null_auc_mean"].mean()),
                "mean_null_auc_abundant": float(ddf.loc[ddf["rarity_group"] == "abundant", "null_auc_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_guideline_table(celltype_df: pd.DataFrame) -> pd.DataFrame:
    stability_fit = fit_min_cells_for_metric(celltype_df, "stability_spearman_mean", STABILITY_THRESHOLD)
    null_fit = fit_min_cells_for_metric(celltype_df, "null_auc_mean", NULL_AUC_THRESHOLD)
    empirical_min = (
        celltype_df[celltype_df["is_reliable"]]
        .groupby("dataset")["cell_count"]
        .min()
        .rename("empirical_min_reliable_cells")
    )

    merged = stability_fit.merge(
        null_fit[
            [
                "dataset",
                "min_cells_estimate",
                "estimate_status",
                "raw_min_cells_estimate",
            ]
        ].rename(
            columns={
                "min_cells_estimate": "null_min_cells_estimate",
                "estimate_status": "null_estimate_status",
                "raw_min_cells_estimate": "null_raw_min_cells_estimate",
            }
        ),
        on="dataset",
        how="left",
    )
    merged = merged.rename(
        columns={
            "min_cells_estimate": "stability_min_cells_estimate",
            "estimate_status": "stability_estimate_status",
            "raw_min_cells_estimate": "stability_raw_min_cells_estimate",
        }
    )
    merged = merged.merge(empirical_min, on="dataset", how="left")
    merged["combined_guideline_model"] = merged[
        ["stability_min_cells_estimate", "null_min_cells_estimate"]
    ].max(axis=1, skipna=True)
    merged["combined_guideline"] = merged[
        ["combined_guideline_model", "empirical_min_reliable_cells"]
    ].max(axis=1, skipna=True)
    merged["combined_guideline"] = merged["combined_guideline"].round(0)
    merged["guideline_status"] = np.where(
        np.isfinite(merged["empirical_min_reliable_cells"]),
        "empirical_reliable_min",
        np.where(
            np.isfinite(merged["combined_guideline_model"]),
            "model_based",
            "unresolved_no_threshold_crossing",
        ),
    )
    return merged


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    rng = np.random.default_rng(args.seed)
    trrust = load_trrust(TRRUST_PATH)
    dataset_map = dict(DATASETS)
    if args.exclude_external_lung:
        dataset_map.pop("external_lung", None)

    all_celltypes: List[pd.DataFrame] = []
    all_top_edges: List[pd.DataFrame] = []
    all_dataset_meta: List[pd.DataFrame] = []

    # Primary analysis: adaptive panel per dataset.
    for dataset_name, dataset_path in dataset_map.items():
        per_celltype, top_edges, dataset_meta = analyze_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            trrust=trrust,
            args=args,
            rng=rng,
            analysis_tag="adaptive_panel",
        )
        all_celltypes.append(per_celltype)
        all_top_edges.append(top_edges)
        all_dataset_meta.append(dataset_meta)

    # Adversarial sensitivity analysis: shared fixed panel across datasets.
    shared_panel_rows = pd.DataFrame()
    shared_tissue_summary_df = pd.DataFrame()
    panel_sensitivity_df = pd.DataFrame()
    if not args.skip_shared_panel_sensitivity:
        gene_sets = collect_dataset_gene_sets(dataset_map)
        shared_genes = set.intersection(*gene_sets.values())
        shared_edges = build_fixed_edges_from_gene_set(
            shared_genes,
            trrust,
            max_tfs=args.shared_max_tfs,
            max_targets=args.shared_max_targets,
        )
        shared_edges.to_csv(RESULTS_DIR / "shared_panel_edges.csv", index=False)

        for dataset_name, dataset_path in dataset_map.items():
            per_celltype, top_edges, dataset_meta = analyze_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                trrust=trrust,
                args=args,
                rng=rng,
                analysis_tag="shared_panel",
                fixed_edges=shared_edges,
            )
            all_celltypes.append(per_celltype)
            all_top_edges.append(top_edges)
            all_dataset_meta.append(dataset_meta)

    celltype_all_df = pd.concat(all_celltypes, ignore_index=True)
    top_edges_all_df = pd.concat(all_top_edges, ignore_index=True)
    dataset_meta_all_df = pd.concat(all_dataset_meta, ignore_index=True)

    # Classify failure modes separately per analysis tag so quantiles are not mixed.
    classified_parts = []
    for _, part in celltype_all_df.groupby("analysis_tag"):
        classified_parts.append(classify_failure_modes(part))
    celltype_all_df = pd.concat(classified_parts, ignore_index=True)

    primary_df = celltype_all_df[celltype_all_df["analysis_tag"] == "adaptive_panel"].copy()
    top_edges_df = top_edges_all_df[top_edges_all_df["analysis_tag"] == "adaptive_panel"].copy()
    dataset_meta_df = dataset_meta_all_df[dataset_meta_all_df["analysis_tag"] == "adaptive_panel"].copy()

    group_df = build_group_comparison(primary_df)
    global_df = build_global_comparison(primary_df)
    global_conservative_df = build_global_comparison(
        primary_df,
        metrics=["null_auc_conservative_mean", "null_tail_gap_conservative_mean"],
    )
    size_bin_df = build_size_bin_summary(primary_df)
    tissue_summary_df = build_tissue_summary(primary_df)
    guideline_df = build_guideline_table(primary_df)
    null_model_df = build_null_model_summary(primary_df)

    # Shared-panel sensitivity outputs.
    if not args.skip_shared_panel_sensitivity:
        shared_panel_rows = celltype_all_df[celltype_all_df["analysis_tag"] == "shared_panel"].copy()
        if not shared_panel_rows.empty:
            shared_tissue_summary_df = build_tissue_summary(shared_panel_rows)
            panel_sensitivity_df = build_panel_sensitivity_summary(primary_df, shared_panel_rows)

    # Persist tabular artifacts (primary outputs keep canonical filenames).
    celltype_all_df.to_csv(RESULTS_DIR / "cell_type_metrics_all.csv", index=False)
    top_edges_all_df.to_csv(RESULTS_DIR / "top_edges_by_cell_type_all.csv", index=False)
    dataset_meta_all_df.to_csv(RESULTS_DIR / "dataset_metadata_all.csv", index=False)
    primary_df.to_csv(RESULTS_DIR / "cell_type_metrics.csv", index=False)
    top_edges_df.to_csv(RESULTS_DIR / "top_edges_by_cell_type.csv", index=False)
    dataset_meta_df.to_csv(RESULTS_DIR / "dataset_metadata.csv", index=False)
    group_df.to_csv(RESULTS_DIR / "rare_vs_abundant_comparisons.csv", index=False)
    global_df.to_csv(RESULTS_DIR / "global_rare_vs_abundant_comparison.csv", index=False)
    global_conservative_df.to_csv(
        RESULTS_DIR / "global_rare_vs_abundant_conservative_comparison.csv",
        index=False,
    )
    size_bin_df.to_csv(RESULTS_DIR / "size_bin_summary.csv", index=False)
    tissue_summary_df.to_csv(RESULTS_DIR / "tissue_summary.csv", index=False)
    guideline_df.to_csv(RESULTS_DIR / "minimum_cell_size_guidelines.csv", index=False)
    null_model_df.to_csv(RESULTS_DIR / "null_model_summary.csv", index=False)
    if not shared_panel_rows.empty:
        shared_panel_rows.to_csv(RESULTS_DIR / "shared_panel_cell_type_metrics.csv", index=False)
    if not shared_tissue_summary_df.empty:
        shared_tissue_summary_df.to_csv(RESULTS_DIR / "shared_panel_tissue_summary.csv", index=False)
    if not panel_sensitivity_df.empty:
        panel_sensitivity_df.to_csv(RESULTS_DIR / "adaptive_vs_shared_panel_sensitivity.csv", index=False)

    # Aggregate summary JSON for paper writing and quick audit (primary + sensitivities).
    summary = {
        "config": {
            "seed": args.seed,
            "min_cells": args.min_cells,
            "replicates": args.replicates,
            "subsample_frac": args.subsample_frac,
            "max_tfs": args.max_tfs,
            "max_targets": args.max_targets,
            "top_k": args.top_k,
            "exclude_external_lung": args.exclude_external_lung,
            "run_shared_panel_sensitivity": (not args.skip_shared_panel_sensitivity),
            "shared_max_tfs": args.shared_max_tfs,
            "shared_max_targets": args.shared_max_targets,
            "stability_threshold": STABILITY_THRESHOLD,
            "jaccard_threshold": TOPK_JACCARD_THRESHOLD,
            "null_auc_threshold": NULL_AUC_THRESHOLD,
            "min_celltype_size_threshold": MIN_CELLTYPE_SIZE_THRESHOLD,
        },
        "datasets_analyzed": list(dataset_map.keys()),
        "n_cell_types_total": int(primary_df.shape[0]),
        "n_reliable_total": int(primary_df["is_reliable"].sum()),
        "reliable_fraction_total": float(primary_df["is_reliable"].mean()),
        "tissues": tissue_summary_df.to_dict(orient="records"),
        "global_rare_vs_abundant": global_df.to_dict(orient="records"),
        "global_rare_vs_abundant_conservative": global_conservative_df.to_dict(orient="records"),
        "size_bin_summary": size_bin_df.to_dict(orient="records"),
        "null_model_summary": null_model_df.to_dict(orient="records"),
        "guidelines": guideline_df.to_dict(orient="records"),
        "shared_panel_tissues": shared_tissue_summary_df.to_dict(orient="records"),
        "adaptive_vs_shared_panel_sensitivity": panel_sensitivity_df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figures used in the paper (primary/adaptive analysis).
    sns.set_theme(style="whitegrid", context="talk")
    plot_stability_curve(primary_df)
    plot_null_separation_curve(primary_df)
    plot_group_boxplots(primary_df)
    plot_failure_modes(primary_df)
    plot_guidelines(guideline_df)

    print("Analysis complete.")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
