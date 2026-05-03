"""Axis-4 core — fixed-holdout donor transfer using shared infrastructure."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from ...metrics import topk_jaccard, topk_overlap, sign_flip_rate
from ...panels import load_panel
from ...scorers import get_scorer
from ...utils import rng_for


@dataclass
class Axis4DatasetSpec:
    name: str
    path: Path
    donor_col: str
    max_donors: int = 12
    cells_per_donor: int = 300


@dataclass
class Axis4Config:
    panel: str = "hematopoiesis_76x108"
    scorer: str = "pearson"
    train_donor_counts: Tuple[int, ...] = (2, 4, 6, 8, 10)
    holdout_donors: int = 2
    n_splits_per_count: int = 20
    top_ks: Tuple[int, ...] = (50, 100, 250, 500)
    rel_threshold: float = 0.90
    rel_threshold_conservative: float = 0.95
    seed_namespace: str = "axis4:default"


def _balanced_panel(
    spec: Axis4DatasetSpec, panel_name: str, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Build a (cells, genes) matrix balanced across donors.

    Returns ``(X, donor_labels, gene_names, tf_idx, target_idx)``.
    """

    adata = ad.read_h5ad(spec.path, backed="r")
    donor_series = adata.obs[spec.donor_col].astype(str)
    counts = donor_series.value_counts()
    eligible = counts[counts >= spec.cells_per_donor]
    selected = eligible.head(spec.max_donors).index.tolist()
    if len(selected) < 2:
        raise ValueError(
            f"{spec.name}: fewer than 2 eligible donors (cells/donor >= {spec.cells_per_donor})."
        )

    # Upper-case gene names for panel matching
    gene_names_raw = [str(g).upper() for g in adata.var_names]
    panel = load_panel(panel_name, gene_universe=gene_names_raw)
    if panel.empty:
        raise ValueError(f"{spec.name}: panel '{panel_name}' has no genes in common with dataset.")
    tfs = sorted(panel["source"].unique())
    targets = sorted(panel["target"].unique())
    required = sorted(set(tfs) | set(targets))
    # Map each gene -> first occurrence index
    gene_to_col = {}
    for i, g in enumerate(gene_names_raw):
        if g in required and g not in gene_to_col:
            gene_to_col[g] = i
    selected_cols = np.array([gene_to_col[g] for g in required], dtype=int)
    local_to_col = {g: i for i, g in enumerate(required)}
    tf_idx = np.array([local_to_col[g] for g in tfs])
    target_idx = np.array([local_to_col[g] for g in targets])

    # Balanced cell sampling per donor
    donor_values = donor_series.values
    rows: List[int] = []
    labels: List[str] = []
    for donor in selected:
        donor_rows = np.where(donor_values == donor)[0]
        chosen = rng.choice(donor_rows, size=spec.cells_per_donor, replace=False)
        rows.extend(sorted(chosen.tolist()))
        labels.extend([donor] * len(chosen))
    rows_arr = np.array(rows, dtype=int)
    labels_arr = np.array(labels, dtype=str)

    # Materialise X and take only the required gene columns
    sub = adata[rows_arr, selected_cols].to_memory()
    X = sub.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    return X, labels_arr, list(required), tf_idx, target_idx


def _split_scores(
    X: np.ndarray,
    gene_names: List[str],
    tf_idx: np.ndarray,
    target_idx: np.ndarray,
    mask: np.ndarray,
    scorer_name: str,
) -> np.ndarray:
    """Score edges on a subset of cells."""

    X_sub = X[mask]
    scorer_kwargs = {}
    if scorer_name in ("grnboost2", "genie3", "scgpt_attention"):
        scorer_kwargs["gene_names"] = gene_names
    scorer = get_scorer(scorer_name, **scorer_kwargs)
    out = scorer.score(X_sub, tf_idx, target_idx, gene_names)
    return out.scores, out.signs


def run(
    datasets: Sequence[Axis4DatasetSpec],
    out_dir: Path,
    config: Optional[Axis4Config] = None,
) -> Dict[str, Path]:
    cfg = config or Axis4Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    recommendation_rows: List[Dict] = []

    for spec in datasets:
        rng = rng_for(f"{cfg.seed_namespace}:{spec.name}")
        X, donor_labels, gene_names, tf_idx, target_idx = _balanced_panel(
            spec, cfg.panel, rng,
        )
        donors = sorted(set(donor_labels.tolist()))
        if len(donors) < cfg.holdout_donors + min(cfg.train_donor_counts):
            continue

        # Fixed-holdout strategy: pick one holdout set of ``holdout_donors``,
        # then sample train donor sets of varying size from the remaining.
        holdout_set = tuple(rng.choice(donors, size=cfg.holdout_donors, replace=False))
        candidate_train = sorted(set(donors) - set(holdout_set))

        holdout_mask = np.isin(donor_labels, np.asarray(holdout_set))
        holdout_scores, holdout_signs = _split_scores(
            X, gene_names, tf_idx, target_idx, holdout_mask, cfg.scorer,
        )

        for train_count in cfg.train_donor_counts:
            if train_count > len(candidate_train):
                continue
            all_subsets = list(itertools.combinations(candidate_train, train_count))
            if len(all_subsets) > cfg.n_splits_per_count:
                idx = rng.choice(len(all_subsets), size=cfg.n_splits_per_count, replace=False)
                subsets = [all_subsets[i] for i in sorted(idx)]
            else:
                subsets = all_subsets

            for train_set in subsets:
                train_mask = np.isin(donor_labels, np.asarray(train_set))
                train_scores, train_signs = _split_scores(
                    X, gene_names, tf_idx, target_idx, train_mask, cfg.scorer,
                )
                rho = spearmanr(train_scores, holdout_scores).statistic
                r = pearsonr(train_scores, holdout_scores).statistic
                entry = {
                    "dataset": spec.name,
                    "scorer": cfg.scorer,
                    "panel": cfg.panel,
                    "train_donor_count": int(train_count),
                    "n_train_cells": int(train_mask.sum()),
                    "n_holdout_cells": int(holdout_mask.sum()),
                    "pearson_r": float(r) if not np.isnan(r) else float("nan"),
                    "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
                    "mean_abs_score_delta": float(np.mean(np.abs(train_scores - holdout_scores))),
                }
                for k in cfg.top_ks:
                    entry[f"overlap_at_{k}"] = topk_overlap(train_scores, holdout_scores, k)
                    entry[f"jaccard_at_{k}"] = topk_jaccard(train_scores, holdout_scores, k)
                if train_signs is not None and holdout_signs is not None:
                    entry["sign_flip_rate"] = sign_flip_rate(train_signs, holdout_signs)
                split_rows.append(entry)

        # Summaries + recommendations
        split_df = pd.DataFrame([r for r in split_rows if r["dataset"] == spec.name])
        for tc, grp in split_df.groupby("train_donor_count"):
            summary_rows.append({
                "dataset": spec.name,
                "scorer": cfg.scorer,
                "train_donor_count": int(tc),
                "n_splits": int(len(grp)),
                "spearman_median": float(grp["spearman_rho"].median()),
                "spearman_q25": float(grp["spearman_rho"].quantile(0.25)),
                "spearman_q75": float(grp["spearman_rho"].quantile(0.75)),
                "overlap100_median": float(grp["overlap_at_100"].median()),
                "overlap100_q25": float(grp["overlap_at_100"].quantile(0.25)),
                "overlap100_q75": float(grp["overlap_at_100"].quantile(0.75)),
            })
        if not split_df.empty:
            summary_df = pd.DataFrame([r for r in summary_rows if r["dataset"] == spec.name])
            best_rho = summary_df["spearman_median"].max()
            threshold_rel = cfg.rel_threshold * best_rho
            threshold_conserv = cfg.rel_threshold_conservative * best_rho
            passing = summary_df[summary_df["spearman_median"] >= threshold_rel]
            passing_c = summary_df[summary_df["spearman_median"] >= threshold_conserv]
            recommendation_rows.append({
                "dataset": spec.name,
                "scorer": cfg.scorer,
                "best_spearman_median": float(best_rho),
                "threshold_90pct": float(threshold_rel),
                "threshold_95pct": float(threshold_conserv),
                "min_train_donors_90pct": int(passing["train_donor_count"].min()) if not passing.empty else -1,
                "min_train_donors_95pct": int(passing_c["train_donor_count"].min()) if not passing_c.empty else -1,
                "holdout_donors": list(holdout_set),
            })

    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame(summary_rows)
    rec_df = pd.DataFrame(recommendation_rows)
    if "holdout_donors" in rec_df.columns:
        rec_df["holdout_donors"] = rec_df["holdout_donors"].apply(lambda xs: ",".join(map(str, xs)))

    paths = {
        "split_metrics": out_dir / "split_metrics.csv",
        "train_count_summary": out_dir / "train_count_summary.csv",
        "recommendations": out_dir / "recommendations.csv",
    }
    split_df.to_csv(paths["split_metrics"], index=False)
    summary_df.to_csv(paths["train_count_summary"], index=False)
    rec_df.to_csv(paths["recommendations"], index=False)
    return paths
