#!/usr/bin/env python3
"""
Adversarial follow-up for subproject 38.

This script addresses methodological weaknesses identified after the initial
LODO generalization draft:
1) train-donor and holdout-donor counts were coupled,
2) donor selection was limited to top-count donors,
3) split comparisons did not control for donor-specific cell-type composition.

The script executes three corrective experiments:
- Fixed-holdout donor protocol on Immune top-12 donors.
- Fixed-holdout donor protocol on an expanded Immune top-20 donor panel.
- Composition-matched vs unbalanced comparison under identical split geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from run_leave_one_donor_out_generalization import (
    TF_LIST,
    TARGET_POOL,
    _build_gene_column_map,
    _extract_gene_name_vector,
    compute_edge_scores,
)


TOP_KS = (50, 100, 250, 500)


@dataclass
class Panel:
    name: str
    X: np.ndarray
    donor_labels: np.ndarray
    cell_types: np.ndarray
    donors: List[str]
    donor_counts_original: Dict[str, int]
    edge_names: List[str]
    edge_tf_idx: np.ndarray
    edge_tgt_idx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial follow-up experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/ihorkendiukhov/biodyn-work/subproject_38_leave_one_donor_out_generalization/reports/artifacts"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("/Users/ihorkendiukhov/biodyn-work/subproject_38_leave_one_donor_out_generalization/reports/figures"),
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def build_edge_index(gene_names: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Build directed TF->target edge indices for the selected gene panel."""
    gene_to_local_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    tf_names = [gene for gene in TF_LIST if gene in gene_to_local_idx]
    target_names = [gene for gene in TARGET_POOL if gene in gene_to_local_idx]

    edge_names: List[str] = []
    edge_tf_idx: List[int] = []
    edge_tgt_idx: List[int] = []
    for tf_name in tf_names:
        tf_idx = gene_to_local_idx[tf_name]
        for target_name in target_names:
            if tf_name == target_name:
                continue
            edge_names.append(f"{tf_name}->{target_name}")
            edge_tf_idx.append(tf_idx)
            edge_tgt_idx.append(gene_to_local_idx[target_name])

    return edge_names, np.asarray(edge_tf_idx, dtype=int), np.asarray(edge_tgt_idx, dtype=int)


def load_balanced_immune_panel(
    max_donors: int,
    cells_per_donor: int,
    rng: np.random.Generator,
) -> Panel:
    """
    Load immune data and build a balanced donor panel by uniform per-donor sampling.

    The panel keeps expression, donor labels, and cell types so that later
    experiments can either run on the unbalanced-by-celltype sample or enforce
    explicit donor-wise cell-type matching.
    """
    path = Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad")
    adata = ad.read_h5ad(path, backed="r")

    donor_series = adata.obs["donor_id"].astype(str)
    # Use broad classes for composition control to avoid sparsity from highly
    # granular labels that are not shared across all donors.
    if "broad_cell_class" in adata.obs.columns:
        celltype_series = adata.obs["broad_cell_class"].astype(str)
    else:
        celltype_series = adata.obs["cell_type"].astype(str)

    donor_counts = donor_series.value_counts()
    eligible = donor_counts[donor_counts >= cells_per_donor]
    donors = eligible.head(max_donors).index.tolist()
    if len(donors) < 4:
        raise ValueError("Need at least 4 donors for fixed-holdout follow-up.")

    gene_names_full = _extract_gene_name_vector(adata)
    gene_pool = list(dict.fromkeys(TF_LIST + TARGET_POOL))
    gene_to_col = _build_gene_column_map(gene_names_full, gene_pool)

    selected_gene_items = sorted([(col, gene) for gene, col in gene_to_col.items()], key=lambda pair: pair[0])
    selected_gene_cols = np.array([item[0] for item in selected_gene_items], dtype=int)
    selected_gene_names = [item[1] for item in selected_gene_items]

    sampled_rows: List[int] = []
    sampled_donors: List[str] = []
    sampled_celltypes: List[str] = []

    donor_values = donor_series.values
    celltype_values = celltype_series.values
    for donor in donors:
        donor_rows = np.where(donor_values == donor)[0]
        chosen_rows = rng.choice(donor_rows, size=cells_per_donor, replace=False)
        sampled_rows.extend(chosen_rows.tolist())
        sampled_donors.extend([donor] * len(chosen_rows))
        sampled_celltypes.extend(celltype_values[chosen_rows].tolist())

    # Keep deterministic row order for reproducibility.
    sample_df = pd.DataFrame(
        {"row": sampled_rows, "donor": sampled_donors, "cell_type": sampled_celltypes}
    ).sort_values("row")

    panel_rows = sample_df["row"].to_numpy(dtype=int)
    panel_adata = adata[panel_rows, selected_gene_cols].to_memory()

    X = panel_adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    edge_names, edge_tf_idx, edge_tgt_idx = build_edge_index(selected_gene_names)

    return Panel(
        name=f"immune_top{max_donors}_cells{cells_per_donor}",
        X=X,
        donor_labels=sample_df["donor"].to_numpy(dtype=str),
        cell_types=sample_df["cell_type"].to_numpy(dtype=str),
        donors=donors,
        donor_counts_original=donor_counts[donors].to_dict(),
        edge_names=edge_names,
        edge_tf_idx=edge_tf_idx,
        edge_tgt_idx=edge_tgt_idx,
    )


def build_celltype_matched_panel(
    panel: Panel,
    rng: np.random.Generator,
    min_count_per_donor_type: int = 8,
    max_count_per_donor_type: int = 20,
) -> Tuple[Panel, Dict[str, object]]:
    """
    Create a donor-balanced and cell-type-matched panel.

    For each cell type present in every donor with at least
    min_count_per_donor_type cells, we sample the same count per donor.
    """
    df = pd.DataFrame(
        {
            "idx": np.arange(panel.X.shape[0]),
            "donor": panel.donor_labels,
            "cell_type": panel.cell_types,
        }
    )

    count_table = (
        df.groupby(["donor", "cell_type"]).size().reset_index(name="count")
    )
    pivot = count_table.pivot(index="cell_type", columns="donor", values="count").fillna(0)

    # Adaptive thresholding: start strict, then relax if needed so the run
    # remains executable on heterogeneous donor panels.
    common_celltypes: List[str] = []
    per_type_counts: Dict[str, int] = {}
    used_min_threshold = min_count_per_donor_type
    for threshold in [min_count_per_donor_type, 6, 4, 2, 1]:
        tmp_common: List[str] = []
        tmp_counts: Dict[str, int] = {}
        for cell_type, row in pivot.iterrows():
            min_count = int(row.min())
            if min_count >= threshold:
                sampled_count = min(min_count, max_count_per_donor_type)
                tmp_common.append(cell_type)
                tmp_counts[cell_type] = sampled_count
        if len(tmp_common) >= 2:
            common_celltypes = tmp_common
            per_type_counts = tmp_counts
            used_min_threshold = threshold
            break

    if len(common_celltypes) < 2:
        raise ValueError("Composition matching failed: fewer than 2 common cell classes even at relaxed thresholds.")

    selected_indices: List[int] = []
    for donor in panel.donors:
        donor_df = df[df["donor"] == donor]
        for cell_type in common_celltypes:
            needed = per_type_counts[cell_type]
            candidates = donor_df.loc[donor_df["cell_type"] == cell_type, "idx"].to_numpy(dtype=int)
            chosen = rng.choice(candidates, size=needed, replace=False)
            selected_indices.extend(chosen.tolist())

    selected_indices = np.asarray(sorted(selected_indices), dtype=int)

    matched_panel = Panel(
        name=f"{panel.name}_celltype_matched",
        X=panel.X[selected_indices],
        donor_labels=panel.donor_labels[selected_indices],
        cell_types=panel.cell_types[selected_indices],
        donors=panel.donors,
        donor_counts_original=panel.donor_counts_original,
        edge_names=panel.edge_names,
        edge_tf_idx=panel.edge_tf_idx,
        edge_tgt_idx=panel.edge_tgt_idx,
    )

    metadata = {
        "n_common_celltypes": len(common_celltypes),
        "common_celltypes": common_celltypes,
        "per_type_sample_count": per_type_counts,
        "used_min_count_threshold": used_min_threshold,
        "cells_total": int(matched_panel.X.shape[0]),
        "cells_per_donor": int(matched_panel.X.shape[0] // len(panel.donors)),
    }
    return matched_panel, metadata


def topk_overlap_metrics(train_scores: np.ndarray, holdout_scores: np.ndarray) -> Dict[str, float]:
    """Compute overlap and null-calibrated lift for several top-k settings."""
    abs_train = np.abs(train_scores)
    abs_holdout = np.abs(holdout_scores)
    n_edges = len(train_scores)

    metrics: Dict[str, float] = {}
    for k in TOP_KS:
        k_eff = min(k, n_edges)
        train_idx = np.argpartition(abs_train, -k_eff)[-k_eff:]
        holdout_idx = np.argpartition(abs_holdout, -k_eff)[-k_eff:]

        overlap = len(set(train_idx.tolist()) & set(holdout_idx.tolist())) / float(k_eff)
        null_overlap = k_eff / float(n_edges)
        lift = overlap / null_overlap if null_overlap > 0 else np.nan

        metrics[f"overlap_at_{k}"] = overlap
        metrics[f"null_overlap_at_{k}"] = null_overlap
        metrics[f"lift_at_{k}"] = lift

    return metrics


def evaluate_explicit_split(
    panel: Panel,
    train_donors: Sequence[str],
    holdout_donors: Sequence[str],
) -> Dict[str, float]:
    """Evaluate a split with explicitly chosen train and holdout donor sets."""
    train_mask = np.isin(panel.donor_labels, np.asarray(train_donors, dtype=str))
    holdout_mask = np.isin(panel.donor_labels, np.asarray(holdout_donors, dtype=str))

    if np.any(train_mask & holdout_mask):
        raise ValueError("Train and holdout donors overlap in explicit split.")
    if train_mask.sum() < 10 or holdout_mask.sum() < 10:
        raise ValueError("Split has too few cells to score correlations robustly.")

    train_scores = compute_edge_scores(panel.X[train_mask], panel.edge_tf_idx, panel.edge_tgt_idx)
    holdout_scores = compute_edge_scores(panel.X[holdout_mask], panel.edge_tf_idx, panel.edge_tgt_idx)

    metrics: Dict[str, float] = {
        "n_train_cells": int(train_mask.sum()),
        "n_holdout_cells": int(holdout_mask.sum()),
        "n_train_donors": int(len(set(train_donors))),
        "n_holdout_donors": int(len(set(holdout_donors))),
        "pearson_r": float(pearsonr(train_scores, holdout_scores).statistic),
        "spearman_rho": float(spearmanr(train_scores, holdout_scores).statistic),
        "mean_abs_score_delta": float(np.mean(np.abs(train_scores - holdout_scores))),
        "sign_agreement": float(np.mean(np.sign(train_scores) == np.sign(holdout_scores))),
    }
    metrics.update(topk_overlap_metrics(train_scores, holdout_scores))
    return metrics


def sample_fixed_holdout_splits(
    donors: Sequence[str],
    train_count: int,
    holdout_count: int,
    n_splits: int,
    rng: np.random.Generator,
) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Sample unique (train_set, holdout_set) tuples under fixed holdout size.

    This decouples train-count scaling from the pathological 1-donor holdout case.
    """
    donors = sorted(donors)
    max_unique = math.comb(len(donors), holdout_count) * math.comb(len(donors) - holdout_count, train_count)
    target_splits = min(int(n_splits), int(max_unique))
    max_attempts = max(4000, target_splits * 80)

    splits: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    seen = set()

    for _ in range(max_attempts):
        holdout = tuple(sorted(rng.choice(donors, size=holdout_count, replace=False).tolist()))
        remaining = [donor for donor in donors if donor not in holdout]
        if train_count > len(remaining):
            continue
        train = tuple(sorted(rng.choice(remaining, size=train_count, replace=False).tolist()))

        key = (train, holdout)
        if key in seen:
            continue
        seen.add(key)
        splits.append(key)
        if len(splits) >= target_splits:
            break

    if len(splits) == 0:
        raise ValueError("No valid fixed-holdout splits could be sampled.")
    return splits


def run_fixed_holdout_experiment(
    panel: Panel,
    train_counts: Sequence[int],
    holdout_count: int,
    splits_per_count: int,
    rng: np.random.Generator,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated fixed-holdout splits and return split metrics + summary."""
    rows: List[Dict[str, object]] = []

    for train_count in train_counts:
        splits = sample_fixed_holdout_splits(
            donors=panel.donors,
            train_count=train_count,
            holdout_count=holdout_count,
            n_splits=splits_per_count,
            rng=rng,
        )

        for split_idx, (train_set, holdout_set) in enumerate(splits):
            metrics = evaluate_explicit_split(panel=panel, train_donors=train_set, holdout_donors=holdout_set)
            row: Dict[str, object] = {
                "experiment": label,
                "panel": panel.name,
                "train_donor_count": train_count,
                "holdout_donor_count": holdout_count,
                "train_donors": ";".join(train_set),
                "holdout_donors": ";".join(holdout_set),
                "split_idx": split_idx,
            }
            row.update(metrics)
            rows.append(row)

    split_df = pd.DataFrame(rows)

    summary_rows: List[Dict[str, object]] = []
    summary_cols = [
        "pearson_r",
        "spearman_rho",
        "mean_abs_score_delta",
        "sign_agreement",
        "overlap_at_100",
        "lift_at_100",
    ]
    for train_count, group in split_df.groupby("train_donor_count"):
        row: Dict[str, object] = {
            "experiment": label,
            "panel": panel.name,
            "train_donor_count": int(train_count),
            "holdout_donor_count": int(holdout_count),
            "n_splits": int(len(group)),
        }
        for col in summary_cols:
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_q25"] = float(group[col].quantile(0.25))
            row[f"{col}_q75"] = float(group[col].quantile(0.75))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("train_donor_count")
    return split_df, summary_df


def plot_fixed_holdout_curve(summary_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Plot Spearman and top-100 overlap curves for one fixed-holdout experiment."""
    sns.set_style("whitegrid")
    x_vals = summary_df["train_donor_count"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_vals, summary_df["spearman_rho_median"], marker="o", color="#2c7fb8")
    axes[0].fill_between(
        x_vals,
        summary_df["spearman_rho_q25"],
        summary_df["spearman_rho_q75"],
        color="#2c7fb8",
        alpha=0.2,
    )
    axes[0].set_xlabel("Train donors")
    axes[0].set_ylabel("Spearman rho")
    axes[0].set_title("Rank agreement")

    axes[1].plot(x_vals, summary_df["overlap_at_100_median"], marker="o", color="#238b45")
    axes[1].fill_between(
        x_vals,
        summary_df["overlap_at_100_q25"],
        summary_df["overlap_at_100_q75"],
        color="#238b45",
        alpha=0.2,
    )
    axes[1].set_xlabel("Train donors")
    axes[1].set_ylabel("Top-100 overlap")
    axes[1].set_title("Top-100 reproducibility")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_balance_comparison(
    combined_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Compare metric distributions between unbalanced and composition-matched panels."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=combined_df, x="panel_variant", y=metric, ax=ax)
    sns.stripplot(data=combined_df, x="panel_variant", y=metric, color="black", alpha=0.35, size=2, ax=ax)
    ax.set_xlabel("Panel variant")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Composition control comparison: {ylabel}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Experiment 1: fixed holdout with original top-12 donor panel.
    immune_top12 = load_balanced_immune_panel(max_donors=12, cells_per_donor=300, rng=rng)
    top12_split_df, top12_summary_df = run_fixed_holdout_experiment(
        panel=immune_top12,
        train_counts=[2, 4, 6, 8, 10],
        holdout_count=2,
        splits_per_count=30,
        rng=rng,
        label="immune_top12_fixed_holdout2",
    )
    top12_split_df.to_csv(args.output_dir / "adversarial_immune_top12_fixed_holdout_split_metrics.csv", index=False)
    top12_summary_df.to_csv(args.output_dir / "adversarial_immune_top12_fixed_holdout_summary.csv", index=False)

    plot_fixed_holdout_curve(
        top12_summary_df,
        title="Immune top-12 donors: fixed holdout=2",
        out_path=args.figure_dir / "adversarial_immune_top12_fixed_holdout_curves.png",
    )

    # Experiment 2: expanded donor panel to stress donor-selection bias.
    immune_top20 = load_balanced_immune_panel(max_donors=20, cells_per_donor=150, rng=rng)
    top20_split_df, top20_summary_df = run_fixed_holdout_experiment(
        panel=immune_top20,
        train_counts=[4, 8, 12, 16],
        holdout_count=2,
        splits_per_count=40,
        rng=rng,
        label="immune_top20_fixed_holdout2",
    )
    top20_split_df.to_csv(args.output_dir / "adversarial_immune_top20_fixed_holdout_split_metrics.csv", index=False)
    top20_summary_df.to_csv(args.output_dir / "adversarial_immune_top20_fixed_holdout_summary.csv", index=False)

    plot_fixed_holdout_curve(
        top20_summary_df,
        title="Immune top-20 donors: fixed holdout=2",
        out_path=args.figure_dir / "adversarial_immune_top20_fixed_holdout_curves.png",
    )

    # Experiment 3: composition-matched panel versus unbalanced baseline.
    matched_panel, matched_meta = build_celltype_matched_panel(
        panel=immune_top12,
        rng=rng,
        min_count_per_donor_type=8,
        max_count_per_donor_type=20,
    )

    unbalanced_cmp_df, unbalanced_cmp_summary = run_fixed_holdout_experiment(
        panel=immune_top12,
        train_counts=[6],
        holdout_count=2,
        splits_per_count=40,
        rng=rng,
        label="immune_top12_unbalanced_train6_holdout2",
    )
    matched_cmp_df, matched_cmp_summary = run_fixed_holdout_experiment(
        panel=matched_panel,
        train_counts=[6],
        holdout_count=2,
        splits_per_count=40,
        rng=rng,
        label="immune_top12_celltype_matched_train6_holdout2",
    )

    unbalanced_cmp_df = unbalanced_cmp_df.copy()
    matched_cmp_df = matched_cmp_df.copy()
    unbalanced_cmp_df["panel_variant"] = "unbalanced"
    matched_cmp_df["panel_variant"] = "celltype_matched"

    composition_compare_df = pd.concat([unbalanced_cmp_df, matched_cmp_df], ignore_index=True)
    composition_compare_df.to_csv(args.output_dir / "adversarial_composition_control_split_metrics.csv", index=False)

    composition_compare_summary = pd.concat(
        [unbalanced_cmp_summary.assign(panel_variant="unbalanced"), matched_cmp_summary.assign(panel_variant="celltype_matched")],
        ignore_index=True,
    )
    composition_compare_summary.to_csv(
        args.output_dir / "adversarial_composition_control_summary.csv", index=False
    )

    # Secondary composition-control experiment on a smaller donor cohort.
    # This avoids severe sample collapse when requiring common cell classes
    # across all 12 donors.
    immune_top8 = load_balanced_immune_panel(max_donors=8, cells_per_donor=300, rng=rng)
    matched_top8_panel, matched_top8_meta = build_celltype_matched_panel(
        panel=immune_top8,
        rng=rng,
        min_count_per_donor_type=8,
        max_count_per_donor_type=25,
    )

    top8_unbalanced_df, top8_unbalanced_summary = run_fixed_holdout_experiment(
        panel=immune_top8,
        train_counts=[6],
        holdout_count=2,
        splits_per_count=40,
        rng=rng,
        label="immune_top8_unbalanced_train6_holdout2",
    )
    top8_matched_df, top8_matched_summary = run_fixed_holdout_experiment(
        panel=matched_top8_panel,
        train_counts=[6],
        holdout_count=2,
        splits_per_count=40,
        rng=rng,
        label="immune_top8_celltype_matched_train6_holdout2",
    )

    top8_unbalanced_df = top8_unbalanced_df.copy()
    top8_matched_df = top8_matched_df.copy()
    top8_unbalanced_df["panel_variant"] = "top8_unbalanced"
    top8_matched_df["panel_variant"] = "top8_celltype_matched"

    top8_composition_compare_df = pd.concat([top8_unbalanced_df, top8_matched_df], ignore_index=True)
    top8_composition_compare_df.to_csv(
        args.output_dir / "adversarial_composition_control_top8_split_metrics.csv", index=False
    )

    top8_composition_compare_summary = pd.concat(
        [
            top8_unbalanced_summary.assign(panel_variant="top8_unbalanced"),
            top8_matched_summary.assign(panel_variant="top8_celltype_matched"),
        ],
        ignore_index=True,
    )
    top8_composition_compare_summary.to_csv(
        args.output_dir / "adversarial_composition_control_top8_summary.csv", index=False
    )

    plot_balance_comparison(
        composition_compare_df,
        metric="spearman_rho",
        ylabel="Spearman rho",
        out_path=args.figure_dir / "adversarial_composition_control_spearman.png",
    )
    plot_balance_comparison(
        composition_compare_df,
        metric="overlap_at_100",
        ylabel="Top-100 overlap",
        out_path=args.figure_dir / "adversarial_composition_control_overlap100.png",
    )
    plot_balance_comparison(
        top8_composition_compare_df,
        metric="spearman_rho",
        ylabel="Spearman rho (top-8 panel)",
        out_path=args.figure_dir / "adversarial_composition_control_top8_spearman.png",
    )
    plot_balance_comparison(
        top8_composition_compare_df,
        metric="overlap_at_100",
        ylabel="Top-100 overlap (top-8 panel)",
        out_path=args.figure_dir / "adversarial_composition_control_top8_overlap100.png",
    )

    # Cross-experiment summary for manuscript tables.
    combined_summary = pd.concat(
        [top12_summary_df, top20_summary_df, composition_compare_summary, top8_composition_compare_summary],
        ignore_index=True,
        sort=False,
    )
    combined_summary.to_csv(args.output_dir / "adversarial_followup_combined_summary.csv", index=False)

    metadata = {
        "top12_panel": {
            "name": immune_top12.name,
            "n_donors": len(immune_top12.donors),
            "cells_total": int(immune_top12.X.shape[0]),
            "donors": immune_top12.donors,
            "donor_counts_original": immune_top12.donor_counts_original,
        },
        "top20_panel": {
            "name": immune_top20.name,
            "n_donors": len(immune_top20.donors),
            "cells_total": int(immune_top20.X.shape[0]),
            "donors": immune_top20.donors,
            "donor_counts_original": immune_top20.donor_counts_original,
        },
        "top8_panel": {
            "name": immune_top8.name,
            "n_donors": len(immune_top8.donors),
            "cells_total": int(immune_top8.X.shape[0]),
            "donors": immune_top8.donors,
            "donor_counts_original": immune_top8.donor_counts_original,
        },
        "celltype_matched": matched_meta,
        "celltype_matched_top8": matched_top8_meta,
    }
    with (args.output_dir / "adversarial_followup_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
