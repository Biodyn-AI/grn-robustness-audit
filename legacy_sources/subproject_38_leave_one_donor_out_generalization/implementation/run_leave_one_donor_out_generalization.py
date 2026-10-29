#!/usr/bin/env python3
"""
Project 7 (2026-02-17): Leave-One-Donor-Out Generalization.

This script builds balanced donor panels from Tabula Sapiens tissue datasets,
runs repeated donor train/holdout splits, and quantifies how well edge rankings
generalize to held-out donors.
"""

from __future__ import annotations

import argparse
import itertools
import json
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


TF_LIST = [
    "STAT1",
    "STAT3",
    "STAT4",
    "STAT5A",
    "STAT5B",
    "STAT6",
    "IRF1",
    "IRF4",
    "IRF7",
    "IRF8",
    "NFKB1",
    "NFKB2",
    "RELA",
    "REL",
    "RELB",
    "TBX21",
    "GATA3",
    "RORC",
    "FOXP3",
    "BCL6",
    "SPI1",
    "RUNX1",
    "RUNX3",
    "ETS1",
    "ETS2",
    "MYC",
    "MAX",
    "JUN",
    "FOS",
    "JUNB",
    "JUND",
    "FOSL1",
    "FOSL2",
    "HIF1A",
    "EPAS1",
    "ARNT",
    "TP53",
    "RB1",
    "E2F1",
    "E2F4",
    "PAX5",
    "EBF1",
    "TCF7",
    "LEF1",
    "CEBPA",
    "CEBPB",
    "CEBPD",
    "SOX2",
    "SOX4",
    "SOX9",
    "KLF2",
    "KLF4",
    "KLF6",
    "NR4A1",
    "NR4A2",
    "NR4A3",
    "BATF",
    "BATF3",
    "MAF",
    "MAFB",
    "EOMES",
    "PRDM1",
    "ID2",
    "ID3",
    "NFE2L2",
    "BACH1",
    "BACH2",
    "ZEB1",
    "ZEB2",
    "SNAI1",
    "TWIST1",
    "FOXO1",
    "FOXO3",
    "AHR",
    "IKZF1",
    "IKZF3",
]

TARGET_POOL = [
    "CD3D",
    "CD3E",
    "CD4",
    "CD8A",
    "CD8B",
    "CD19",
    "MS4A1",
    "CD79A",
    "CD79B",
    "CD14",
    "CD68",
    "ITGAM",
    "ITGAX",
    "FCGR3A",
    "CSF1R",
    "IL2",
    "IL2RA",
    "IL7R",
    "IL6",
    "IL10",
    "IL17A",
    "IL21",
    "IL4",
    "IL13",
    "IFNG",
    "TNF",
    "TNFRSF9",
    "TNFRSF18",
    "GZMB",
    "GZMA",
    "GZMK",
    "PRF1",
    "GNLY",
    "NKG7",
    "CCL5",
    "CCR7",
    "CXCR4",
    "CXCL13",
    "CCL2",
    "CXCL10",
    "HLA-DRA",
    "HLA-DRB1",
    "HLA-A",
    "HLA-B",
    "HLA-C",
    "B2M",
    "LYZ",
    "S100A8",
    "S100A9",
    "S100A12",
    "SELL",
    "ITGA4",
    "ITGB1",
    "ICAM1",
    "VCAM1",
    "MKI67",
    "TOP2A",
    "CDK1",
    "PCNA",
    "BCL2",
    "BCL2L1",
    "MCL1",
    "BAX",
    "BAK1",
    "CTLA4",
    "PDCD1",
    "LAG3",
    "HAVCR2",
    "TIGIT",
    "TOX",
    "FN1",
    "COL1A1",
    "COL3A1",
    "VIM",
    "ACTA2",
    "CDH1",
    "EPCAM",
    "SFTPC",
    "SFTPB",
    "SFTPA1",
    "SCGB1A1",
    "MUC5AC",
    "FOXJ1",
    "PECAM1",
    "CDH5",
    "VWF",
    "KDR",
    "NOTCH1",
    "DLL4",
    "JAG1",
    "WNT5A",
    "CTNNB1",
    "TGFB1",
    "TGFBR1",
    "SMAD3",
    "SMAD7",
    "SOD2",
    "HMOX1",
    "NQO1",
    "GSTP1",
    "GAPDH",
    "ACTB",
    "RPL13A",
    "RPS18",
    "PTPRC",
    "THY1",
    "ENG",
    "NT5E",
]

TOP_KS = (50, 100, 250, 500)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    donor_col: str
    max_donors: int
    cells_per_donor: int
    train_donor_counts: Tuple[int, ...]
    random_splits_per_count: int


@dataclass
class PanelData:
    dataset: str
    donors: List[str]
    donor_counts_original: Dict[str, int]
    donor_labels_balanced: np.ndarray
    gene_names: List[str]
    tf_names: List[str]
    target_names: List[str]
    edge_names: List[str]
    edge_tf_idx: np.ndarray
    edge_tgt_idx: np.ndarray
    X: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run donor holdout generalization study.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/ihorkendiukhov/biodyn-work/subproject_38_leave_one_donor_out_generalization/reports/artifacts"),
        help="Directory for CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("/Users/ihorkendiukhov/biodyn-work/subproject_38_leave_one_donor_out_generalization/reports/figures"),
        help="Directory for generated figures.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _extract_gene_name_vector(adata: ad.AnnData) -> np.ndarray:
    if "feature_name" in adata.var.columns and str(adata.var_names[0]).startswith("ENSG"):
        return adata.var["feature_name"].astype(str).values
    return adata.var_names.astype(str).values


def _build_gene_column_map(gene_names: Sequence[str], target_genes: Iterable[str]) -> Dict[str, int]:
    target_set = set(target_genes)
    mapping: Dict[str, int] = {}
    for idx, gene in enumerate(gene_names):
        if gene in target_set and gene not in mapping:
            mapping[gene] = idx
    return mapping


def load_balanced_panel(config: DatasetConfig, rng: np.random.Generator) -> PanelData:
    adata = ad.read_h5ad(config.path, backed="r")
    donor_series = adata.obs[config.donor_col].astype(str)
    donor_counts = donor_series.value_counts()

    eligible = donor_counts[donor_counts >= config.cells_per_donor]
    selected_donors = eligible.head(config.max_donors).index.tolist()
    if len(selected_donors) < 2:
        raise ValueError(f"{config.name}: fewer than 2 eligible donors at cells_per_donor={config.cells_per_donor}.")

    gene_names_full = _extract_gene_name_vector(adata)
    gene_pool = list(dict.fromkeys(TF_LIST + TARGET_POOL))
    gene_to_col = _build_gene_column_map(gene_names_full, gene_pool)

    selected_gene_items = sorted([(col, gene) for gene, col in gene_to_col.items()], key=lambda pair: pair[0])
    selected_gene_cols = np.array([item[0] for item in selected_gene_items], dtype=int)
    selected_gene_names = [item[1] for item in selected_gene_items]

    tf_names = [gene for gene in TF_LIST if gene in gene_to_col]
    target_names = [gene for gene in TARGET_POOL if gene in gene_to_col]
    if not tf_names or not target_names:
        raise ValueError(f"{config.name}: no overlap between gene panel and expression matrix.")

    sampled_rows: List[int] = []
    sampled_donors: List[str] = []
    donor_values = donor_series.values
    for donor in selected_donors:
        donor_rows = np.where(donor_values == donor)[0]
        chosen_rows = rng.choice(donor_rows, size=config.cells_per_donor, replace=False)
        sampled_rows.extend(chosen_rows.tolist())
        sampled_donors.extend([donor] * len(chosen_rows))

    sample_df = pd.DataFrame({"row": sampled_rows, "donor": sampled_donors}).sort_values("row").reset_index(drop=True)
    balanced_rows = sample_df["row"].to_numpy(dtype=int)
    balanced_donor_labels = sample_df["donor"].to_numpy(dtype=str)

    panel_adata = adata[balanced_rows, selected_gene_cols].to_memory()
    X = panel_adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    local_gene_to_idx = {gene: idx for idx, gene in enumerate(selected_gene_names)}
    edge_tf_idx: List[int] = []
    edge_tgt_idx: List[int] = []
    edge_names: List[str] = []
    for tf_name in tf_names:
        tf_idx = local_gene_to_idx[tf_name]
        for target_name in target_names:
            if tf_name == target_name:
                continue
            target_idx = local_gene_to_idx[target_name]
            edge_tf_idx.append(tf_idx)
            edge_tgt_idx.append(target_idx)
            edge_names.append(f"{tf_name}->{target_name}")

    return PanelData(
        dataset=config.name,
        donors=selected_donors,
        donor_counts_original=donor_counts[selected_donors].to_dict(),
        donor_labels_balanced=balanced_donor_labels,
        gene_names=selected_gene_names,
        tf_names=tf_names,
        target_names=target_names,
        edge_names=edge_names,
        edge_tf_idx=np.asarray(edge_tf_idx, dtype=int),
        edge_tgt_idx=np.asarray(edge_tgt_idx, dtype=int),
        X=X,
    )


def compute_edge_scores(X: np.ndarray, edge_tf_idx: np.ndarray, edge_tgt_idx: np.ndarray) -> np.ndarray:
    n_cells = X.shape[0]
    if n_cells < 3:
        return np.zeros(len(edge_tf_idx), dtype=np.float32)

    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True, ddof=1)
    valid = (stds.ravel() > 1e-8)

    Z = np.zeros_like(X, dtype=np.float32)
    Z[:, valid] = (X[:, valid] - means[:, valid]) / stds[:, valid]
    corr = (Z.T @ Z) / float(n_cells - 1)

    scores = corr[edge_tf_idx, edge_tgt_idx]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def _topk_overlap_metrics(train_scores: np.ndarray, holdout_scores: np.ndarray, top_ks: Sequence[int]) -> Dict[str, float]:
    abs_train = np.abs(train_scores)
    abs_holdout = np.abs(holdout_scores)
    n_edges = len(train_scores)

    output: Dict[str, float] = {}
    for k in top_ks:
        k_eff = min(k, n_edges)
        train_idx = np.argpartition(abs_train, -k_eff)[-k_eff:]
        holdout_idx = np.argpartition(abs_holdout, -k_eff)[-k_eff:]

        overlap = len(set(train_idx.tolist()) & set(holdout_idx.tolist())) / float(k_eff)
        null_overlap = k_eff / float(n_edges)
        lift = overlap / null_overlap if null_overlap > 0 else np.nan

        output[f"overlap_at_{k}"] = overlap
        output[f"null_overlap_at_{k}"] = null_overlap
        output[f"lift_at_{k}"] = lift

    return output


def evaluate_split(
    X: np.ndarray,
    donor_labels: np.ndarray,
    donors_train: Sequence[str],
    edge_tf_idx: np.ndarray,
    edge_tgt_idx: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.isin(donor_labels, np.asarray(list(donors_train), dtype=str))
    holdout_mask = ~train_mask

    X_train = X[train_mask]
    X_holdout = X[holdout_mask]

    train_scores = compute_edge_scores(X_train, edge_tf_idx, edge_tgt_idx)
    holdout_scores = compute_edge_scores(X_holdout, edge_tf_idx, edge_tgt_idx)

    pearson_val = pearsonr(train_scores, holdout_scores).statistic
    spearman_val = spearmanr(train_scores, holdout_scores).statistic

    metrics: Dict[str, float] = {
        "n_train_cells": int(train_mask.sum()),
        "n_holdout_cells": int(holdout_mask.sum()),
        "n_train_donors": len(set(donors_train)),
        "n_holdout_donors": len(set(donor_labels[holdout_mask])),
        "pearson_r": float(pearson_val),
        "spearman_rho": float(spearman_val),
        "mean_abs_score_delta": float(np.mean(np.abs(train_scores - holdout_scores))),
        "sign_agreement": float(np.mean(np.sign(train_scores) == np.sign(holdout_scores))),
    }
    metrics.update(_topk_overlap_metrics(train_scores, holdout_scores, TOP_KS))

    return metrics, train_scores, holdout_scores, train_mask, holdout_mask


def sample_train_donor_sets(
    donors: Sequence[str], train_count: int, n_splits: int, rng: np.random.Generator
) -> List[Tuple[str, ...]]:
    all_sets = list(itertools.combinations(sorted(donors), train_count))
    if len(all_sets) <= n_splits:
        return all_sets
    indices = rng.choice(len(all_sets), size=n_splits, replace=False)
    return [all_sets[idx] for idx in sorted(indices)]


def summarize_by_train_count(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    summary_cols = [
        "pearson_r",
        "spearman_rho",
        "mean_abs_score_delta",
        "sign_agreement",
        "overlap_at_50",
        "overlap_at_100",
        "overlap_at_250",
        "overlap_at_500",
        "lift_at_50",
        "lift_at_100",
        "lift_at_250",
        "lift_at_500",
    ]
    for train_count, group in metrics_df.groupby("train_donor_count"):
        row: Dict[str, float] = {
            "dataset": str(group["dataset"].iloc[0]),
            "train_donor_count": int(train_count),
            "n_splits": int(len(group)),
        }
        for column in summary_cols:
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_q25"] = float(group[column].quantile(0.25))
            row[f"{column}_q75"] = float(group[column].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("train_donor_count").reset_index(drop=True)


def recommend_minimum_train_donors(summary_df: pd.DataFrame) -> Dict[str, float]:
    best_spearman = float(summary_df["spearman_rho_median"].max())
    best_overlap = float(summary_df["overlap_at_100_median"].max())

    target_rel_spearman = 0.9 * best_spearman
    target_rel_overlap = 0.9 * best_overlap

    rec_rel_spearman = int(
        summary_df.loc[summary_df["spearman_rho_median"] >= target_rel_spearman, "train_donor_count"].min()
    )
    rec_rel_overlap = int(
        summary_df.loc[summary_df["overlap_at_100_median"] >= target_rel_overlap, "train_donor_count"].min()
    )
    rec_rel_final = int(max(rec_rel_spearman, rec_rel_overlap))

    abs_target_spearman = 0.70
    abs_target_overlap = 0.55
    abs_ok = summary_df[
        (summary_df["spearman_rho_median"] >= abs_target_spearman)
        & (summary_df["overlap_at_100_median"] >= abs_target_overlap)
    ]
    if abs_ok.empty:
        rec_abs_final = int(summary_df.loc[summary_df["spearman_rho_median"].idxmax(), "train_donor_count"])
    else:
        rec_abs_final = int(abs_ok["train_donor_count"].min())

    return {
        "best_spearman": best_spearman,
        "best_overlap_at_100": best_overlap,
        "target_rel_spearman": target_rel_spearman,
        "target_rel_overlap_at_100": target_rel_overlap,
        "recommended_train_donors_rel_spearman": rec_rel_spearman,
        "recommended_train_donors_rel_overlap100": rec_rel_overlap,
        "recommended_train_donors_rel_final": rec_rel_final,
        "target_abs_spearman": abs_target_spearman,
        "target_abs_overlap_at_100": abs_target_overlap,
        "recommended_train_donors_abs_final": rec_abs_final,
        "recommended_train_donors_final": int(max(rec_rel_final, rec_abs_final)),
    }


def run_dataset(
    config: DatasetConfig,
    output_dir: Path,
    figure_dir: Path,
    rng: np.random.Generator,
) -> Dict[str, pd.DataFrame]:
    panel = load_balanced_panel(config, rng)

    split_rows: List[Dict[str, object]] = []
    edge_abs_deltas: List[np.ndarray] = []
    edge_sign_matches: List[np.ndarray] = []

    for train_count in config.train_donor_counts:
        donor_sets = sample_train_donor_sets(
            donors=panel.donors,
            train_count=train_count,
            n_splits=config.random_splits_per_count,
            rng=rng,
        )

        for split_idx, donors_train in enumerate(donor_sets):
            metrics, train_scores, holdout_scores, train_mask, holdout_mask = evaluate_split(
                X=panel.X,
                donor_labels=panel.donor_labels_balanced,
                donors_train=donors_train,
                edge_tf_idx=panel.edge_tf_idx,
                edge_tgt_idx=panel.edge_tgt_idx,
            )

            split_row: Dict[str, object] = {
                "dataset": panel.dataset,
                "train_donor_count": train_count,
                "train_donors": ";".join(sorted(donors_train)),
                "holdout_donors": ";".join(sorted(set(panel.donor_labels_balanced[holdout_mask].tolist()))),
                "split_idx": split_idx,
            }
            split_row.update(metrics)
            split_rows.append(split_row)

            edge_abs_deltas.append(np.abs(train_scores - holdout_scores))
            edge_sign_matches.append((np.sign(train_scores) == np.sign(holdout_scores)).astype(np.float32))

    split_metrics_df = pd.DataFrame(split_rows)
    split_metrics_path = output_dir / f"{panel.dataset.lower()}_split_metrics.csv"
    split_metrics_df.to_csv(split_metrics_path, index=False)

    edge_abs_delta_matrix = np.vstack(edge_abs_deltas)
    edge_sign_match_matrix = np.vstack(edge_sign_matches)
    edge_sensitivity_df = pd.DataFrame(
        {
            "dataset": panel.dataset,
            "edge": panel.edge_names,
            "mean_abs_delta": edge_abs_delta_matrix.mean(axis=0),
            "median_abs_delta": np.median(edge_abs_delta_matrix, axis=0),
            "q90_abs_delta": np.quantile(edge_abs_delta_matrix, 0.90, axis=0),
            "sign_agreement": edge_sign_match_matrix.mean(axis=0),
        }
    ).sort_values(["mean_abs_delta", "sign_agreement"], ascending=[False, True])
    edge_sensitivity_path = output_dir / f"{panel.dataset.lower()}_edge_sensitivity.csv"
    edge_sensitivity_df.to_csv(edge_sensitivity_path, index=False)

    lodo_rows: List[Dict[str, object]] = []
    all_donors = sorted(panel.donors)
    for holdout_donor in all_donors:
        donors_train = [donor for donor in all_donors if donor != holdout_donor]
        metrics, *_ = evaluate_split(
            X=panel.X,
            donor_labels=panel.donor_labels_balanced,
            donors_train=donors_train,
            edge_tf_idx=panel.edge_tf_idx,
            edge_tgt_idx=panel.edge_tgt_idx,
        )
        row: Dict[str, object] = {
            "dataset": panel.dataset,
            "holdout_donor": holdout_donor,
            "train_donors": ";".join(sorted(donors_train)),
        }
        row.update(metrics)
        lodo_rows.append(row)

    lodo_df = pd.DataFrame(lodo_rows).sort_values("holdout_donor").reset_index(drop=True)
    lodo_path = output_dir / f"{panel.dataset.lower()}_lodo_metrics.csv"
    lodo_df.to_csv(lodo_path, index=False)

    summary_df = summarize_by_train_count(split_metrics_df)
    summary_path = output_dir / f"{panel.dataset.lower()}_train_count_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    recommendation = recommend_minimum_train_donors(summary_df)
    recommendation_row = {
        "dataset": panel.dataset,
        "n_selected_donors": len(panel.donors),
        "cells_per_donor": config.cells_per_donor,
        "n_edges": len(panel.edge_names),
        "n_genes": len(panel.gene_names),
    }
    recommendation_row.update(recommendation)
    recommendation_df = pd.DataFrame([recommendation_row])
    recommendation_path = output_dir / f"{panel.dataset.lower()}_recommendation.csv"
    recommendation_df.to_csv(recommendation_path, index=False)

    metadata = {
        "dataset": panel.dataset,
        "source_path": str(config.path),
        "donor_col": config.donor_col,
        "selected_donors": panel.donors,
        "donor_counts_original": panel.donor_counts_original,
        "cells_per_donor": config.cells_per_donor,
        "balanced_cells_total": int(panel.X.shape[0]),
        "n_genes": len(panel.gene_names),
        "n_tfs_found": len(panel.tf_names),
        "n_targets_found": len(panel.target_names),
        "n_edges": len(panel.edge_names),
    }
    with (output_dir / f"{panel.dataset.lower()}_panel_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    plot_dataset_curves(panel.dataset, summary_df, figure_dir)
    plot_lodo_bars(panel.dataset, lodo_df, figure_dir)
    plot_edge_sensitivity(panel.dataset, edge_sensitivity_df, figure_dir)

    return {
        "split_metrics": split_metrics_df,
        "summary": summary_df,
        "lodo": lodo_df,
        "edge_sensitivity": edge_sensitivity_df,
        "recommendation": recommendation_df,
    }


def plot_dataset_curves(dataset: str, summary_df: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x_vals = summary_df["train_donor_count"].to_numpy()

    axes[0].plot(x_vals, summary_df["spearman_rho_median"], marker="o", color="#2c7fb8", label="Spearman")
    axes[0].fill_between(
        x_vals,
        summary_df["spearman_rho_q25"],
        summary_df["spearman_rho_q75"],
        color="#2c7fb8",
        alpha=0.2,
    )
    axes[0].set_title("Rank Generalization")
    axes[0].set_xlabel("Train donors")
    axes[0].set_ylabel("Spearman rho")

    axes[1].plot(x_vals, summary_df["overlap_at_100_median"], marker="o", color="#238b45", label="Top-100 overlap")
    axes[1].fill_between(
        x_vals,
        summary_df["overlap_at_100_q25"],
        summary_df["overlap_at_100_q75"],
        color="#238b45",
        alpha=0.2,
    )
    axes[1].set_title("Top-100 Reproducibility")
    axes[1].set_xlabel("Train donors")
    axes[1].set_ylabel("Overlap fraction")

    axes[2].plot(x_vals, summary_df["lift_at_100_median"], marker="o", color="#b30000", label="Null lift")
    axes[2].fill_between(
        x_vals,
        summary_df["lift_at_100_q25"],
        summary_df["lift_at_100_q75"],
        color="#b30000",
        alpha=0.2,
    )
    axes[2].set_title("Signal vs Random")
    axes[2].set_xlabel("Train donors")
    axes[2].set_ylabel("Lift over null")

    fig.suptitle(f"{dataset}: donor-split generalization curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{dataset.lower()}_generalization_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lodo_bars(dataset: str, lodo_df: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_df = lodo_df.sort_values("spearman_rho", ascending=True)
    axes[0].barh(plot_df["holdout_donor"], plot_df["spearman_rho"], color="#2c7fb8")
    axes[0].set_title("LODO rank correlation")
    axes[0].set_xlabel("Spearman rho")

    axes[1].barh(plot_df["holdout_donor"], plot_df["overlap_at_100"], color="#238b45")
    axes[1].set_title("LODO top-100 overlap")
    axes[1].set_xlabel("Overlap fraction")

    fig.suptitle(f"{dataset}: donor-wise leave-one-donor-out performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{dataset.lower()}_lodo_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_edge_sensitivity(dataset: str, edge_df: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_style("whitegrid")
    top_edges = edge_df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        edge_df["mean_abs_delta"],
        edge_df["sign_agreement"],
        s=10,
        alpha=0.25,
        color="#6a51a3",
    )
    _ = sc
    ax.scatter(
        top_edges["mean_abs_delta"],
        top_edges["sign_agreement"],
        s=35,
        color="#de2d26",
        label="Top 20 unstable edges",
    )
    for _, row in top_edges.iterrows():
        ax.text(row["mean_abs_delta"], row["sign_agreement"], row["edge"], fontsize=7)

    ax.set_xlabel("Mean |train-holdout score delta|")
    ax.set_ylabel("Sign agreement fraction")
    ax.set_title(f"{dataset}: edge-level donor sensitivity")
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(figure_dir / f"{dataset.lower()}_edge_sensitivity_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_summary(combined_summary: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_style("whitegrid")
    plot_df = combined_summary.copy()
    plot_df["train_fraction"] = plot_df["train_donor_count"] / plot_df["n_selected_donors"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.lineplot(
        data=plot_df,
        x="train_fraction",
        y="spearman_rho_median",
        hue="dataset",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_xlabel("Train donor fraction")
    axes[0].set_ylabel("Median Spearman rho")
    axes[0].set_title("Rank generalization by donor fraction")

    sns.lineplot(
        data=plot_df,
        x="train_fraction",
        y="overlap_at_100_median",
        hue="dataset",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_xlabel("Train donor fraction")
    axes[1].set_ylabel("Median top-100 overlap")
    axes[1].set_title("Top-100 overlap by donor fraction")

    fig.tight_layout()
    fig.savefig(figure_dir / "cross_dataset_generalization_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    datasets = [
        DatasetConfig(
            name="Lung",
            path=Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"),
            donor_col="donor_id",
            max_donors=4,
            cells_per_donor=800,
            train_donor_counts=(1, 2, 3),
            random_splits_per_count=40,
        ),
        DatasetConfig(
            name="Immune",
            path=Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"),
            donor_col="donor_id",
            max_donors=12,
            cells_per_donor=300,
            train_donor_counts=(2, 4, 6, 8, 10, 11),
            random_splits_per_count=40,
        ),
    ]

    combined_split_metrics: List[pd.DataFrame] = []
    combined_summary: List[pd.DataFrame] = []
    combined_lodo: List[pd.DataFrame] = []
    combined_recommendations: List[pd.DataFrame] = []

    for config in datasets:
        outputs = run_dataset(config=config, output_dir=args.output_dir, figure_dir=args.figure_dir, rng=rng)
        summary_df = outputs["summary"].copy()
        recommendation_df = outputs["recommendation"].copy()
        summary_df["n_selected_donors"] = int(recommendation_df["n_selected_donors"].iloc[0])

        combined_split_metrics.append(outputs["split_metrics"])
        combined_summary.append(summary_df)
        combined_lodo.append(outputs["lodo"])
        combined_recommendations.append(recommendation_df)

    combined_split_metrics_df = pd.concat(combined_split_metrics, ignore_index=True)
    combined_summary_df = pd.concat(combined_summary, ignore_index=True)
    combined_lodo_df = pd.concat(combined_lodo, ignore_index=True)
    combined_recommendations_df = pd.concat(combined_recommendations, ignore_index=True)

    combined_split_metrics_df.to_csv(args.output_dir / "combined_split_metrics.csv", index=False)
    combined_summary_df.to_csv(args.output_dir / "combined_train_count_summary.csv", index=False)
    combined_lodo_df.to_csv(args.output_dir / "combined_lodo_metrics.csv", index=False)
    combined_recommendations_df.to_csv(args.output_dir / "combined_recommendations.csv", index=False)

    plot_cross_dataset_summary(combined_summary_df, args.figure_dir)

    high_level = (
        combined_summary_df[["dataset", "train_donor_count", "spearman_rho_median", "overlap_at_100_median", "lift_at_100_median"]]
        .sort_values(["dataset", "train_donor_count"])
        .to_dict(orient="records")
    )
    with (args.output_dir / "combined_high_level_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(high_level, handle, indent=2)


if __name__ == "__main__":
    main()
