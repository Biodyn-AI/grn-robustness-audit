"""Axis-2 end-to-end runner.

Produces:

* ``scored_pairs.csv`` — long-format CSV consumed by WP-3 (RSS redesign)
  and WP-10 (top-k scan). One row per edge.
* ``scorecard.csv`` — one row per (dataset, scorer, pair_id) with
  component metrics + composite RSS + base + dual-null-calibrated
  recommendation + (optionally) triple-null recommendation.
* ``within_coarse_group.csv`` — per-coarse-group within-group RSS.
* ``null_calibration.csv`` — one row per (dataset, null_family) with the
  empirical p-value for the observed RSS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ...metrics import (
    rss_with_weights,
    rss_components,
    spearman_rank_stability,
    topk_jaccard,
    topk_overlap,
)
from ...utils import rng_for
from .scoring import (
    RankingResult,
    infer_rankings,
    rank_high_variance_genes,
)
from .calibration import (
    calibrate_dual_null,
    calibrate_triple_null,
    metric_to_recommendation,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Axis2DatasetSpec:
    name: str
    path: Path
    coarse_key: str
    fine_key: str


@dataclass
class LoadedDataset:
    name: str
    x: sparse.csr_matrix
    gene_ids: np.ndarray
    coarse_labels: np.ndarray
    fine_labels: np.ndarray


@dataclass
class Axis2Config:
    n_top_genes: int = 600
    global_weight: float = 0.6
    top_k: int = 1000
    min_cells_group: int = 40
    min_detect_frac: float = 0.05
    n_null_permutations: int = 49
    rss_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    run_triple_null: bool = False
    seed_namespace: str = "axis2:default"


# ---------------------------------------------------------------------------
# h5ad loading (mirrors the original script's direct-h5py reader so we do
# not rely on the anndata representation of scvi_leiden_* being strings)
# ---------------------------------------------------------------------------


def _decode_bytes(arr) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind == "O":
        out = np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in out])
    elif out.dtype.kind == "S":
        out = np.array([x.decode("utf-8") for x in out])
    else:
        out = out.astype(str)
    return out


def _read_obs_column(f: h5py.File, column: str) -> np.ndarray:
    obs = f["obs"]
    if column not in obs:
        raise KeyError(f"obs column '{column}' not found in {f.filename}")
    entry = obs[column]
    # AnnData-style categorical: dataset group with 'codes' + 'categories'
    if isinstance(entry, h5py.Group) and "codes" in entry and "categories" in entry:
        codes = np.asarray(entry["codes"])
        categories = _decode_bytes(entry["categories"])
        out = np.asarray([
            categories[int(c)] if 0 <= int(c) < len(categories) else "nan"
            for c in codes
        ])
        return out
    return _decode_bytes(entry)


def _read_gene_names(f: h5py.File) -> np.ndarray:
    """Return symbolic gene names, preferring feature_name if the index is Ensembl IDs.

    Handles both plain datasets (h5py.Dataset) and AnnData-style
    categoricals (h5py.Group with 'codes' + 'categories').
    """

    if "var" not in f:
        raise KeyError("h5ad missing /var")
    var = f["var"]

    def _decode(entry):
        if isinstance(entry, h5py.Group) and "codes" in entry and "categories" in entry:
            codes = np.asarray(entry["codes"])
            cats = _decode_bytes(entry["categories"])
            return np.asarray([cats[int(c)] if 0 <= int(c) < len(cats) else "nan"
                               for c in codes])
        return _decode_bytes(entry)

    candidates = []
    for key in ("_index", "feature_name", "gene_symbols", "gene_ids"):
        if key in var:
            candidates.append((key, _decode(var[key])))
    if not candidates:
        raise KeyError("no recognisable gene name column in /var")
    primary = candidates[0]
    if primary[0] == "_index" and len(primary[1]) > 0 and str(primary[1][0]).startswith("ENSG"):
        for key, arr in candidates[1:]:
            if key == "feature_name":
                return arr
    return primary[1]


def _read_x(f: h5py.File) -> sparse.csr_matrix:
    X = f["X"]
    if isinstance(X, h5py.Group) and {"data", "indices", "indptr"}.issubset(X.keys()):
        data = np.asarray(X["data"])
        indices = np.asarray(X["indices"])
        indptr = np.asarray(X["indptr"])
        shape = tuple(X.attrs.get("shape", (len(indptr) - 1, int(indices.max()) + 1)))
        return sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))
    data = np.asarray(X)
    return sparse.csr_matrix(data)


def load_h5ad(spec: Axis2DatasetSpec) -> LoadedDataset:
    """Load a single processed h5ad using direct h5py for deterministic obs parsing."""

    with h5py.File(spec.path, "r") as f:
        x = _read_x(f)
        genes = _read_gene_names(f)
        coarse = _read_obs_column(f, spec.coarse_key)
        fine = _read_obs_column(f, spec.fine_key)
    return LoadedDataset(
        name=spec.name, x=x, gene_ids=genes,
        coarse_labels=coarse, fine_labels=fine,
    )


def make_hierarchical_fine_labels(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    return np.asarray([f"{c}:::{fi}" for c, fi in zip(coarse, fine)])


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def _score_ranking(
    x_dense: np.ndarray,
    labels: np.ndarray,
    cfg: Axis2Config,
) -> RankingResult:
    return infer_rankings(
        x_dense=x_dense,
        labels=labels,
        min_cells_group=cfg.min_cells_group,
        global_weight=cfg.global_weight,
    )


def _prep_dense(ds: LoadedDataset, cfg: Axis2Config) -> Tuple[np.ndarray, np.ndarray]:
    """Select top-N high-variance genes and densify expression."""

    ranked = rank_high_variance_genes(ds.x, min_detect_frac=cfg.min_detect_frac)
    top_idx = ranked[: cfg.n_top_genes]
    sub = ds.x[:, top_idx]
    return sub.toarray().astype(np.float32), top_idx


def _permute_fine_labels(
    fine_labels: np.ndarray,
    coarse_labels: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "global":
        return fine_labels[rng.permutation(fine_labels.shape[0])]
    if mode == "within_coarse":
        out = fine_labels.copy()
        for c in np.unique(coarse_labels):
            idx = np.where(coarse_labels == c)[0]
            if idx.size <= 1:
                continue
            out[idx] = fine_labels[idx][rng.permutation(idx.size)]
        return out
    raise ValueError(f"unknown null mode {mode!r}")


def _null_pass(
    observed_rss: float,
    null_rss_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """Return (passed, empirical_p) where pass means observed is more
    stable (lower RSS) than the null tail."""

    # Null hypothesis: observed RSS comes from the null distribution. We
    # test a lower tail (observed << null -> reject null, i.e. pass).
    count = int(np.sum(null_rss_values <= observed_rss))
    p = (count + 1.0) / (len(null_rss_values) + 1.0)
    return p <= alpha, p


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run(
    datasets: Sequence[Axis2DatasetSpec],
    out_dir: Path,
    config: Optional[Axis2Config] = None,
) -> Dict[str, Path]:
    """Run Axis-2 on a list of processed h5ads and emit the tidy CSVs."""

    cfg = config or Axis2Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_rows: List[Dict] = []
    scorecard_rows: List[Dict] = []
    null_rows: List[Dict] = []
    within_rows: List[Dict] = []

    rng = rng_for(cfg.seed_namespace)

    for spec in datasets:
        ds = load_h5ad(spec)
        x_dense, gene_idx = _prep_dense(ds, cfg)
        genes = ds.gene_ids[gene_idx]

        coarse = ds.coarse_labels
        fine = make_hierarchical_fine_labels(coarse, ds.fine_labels)

        # Observed coarse and fine rankings
        coarse_ranking = _score_ranking(x_dense, coarse, cfg)
        fine_ranking = _score_ranking(x_dense, fine, cfg)

        # Edge list: upper-triangular HVG×HVG
        iu = np.triu_indices(x_dense.shape[1], 1)
        edge_scores_coarse = coarse_ranking.edge_scores
        edge_scores_fine = fine_ranking.edge_scores

        # Long-format scored pairs CSV (consumed by WP-3 / WP-10)
        for e in range(len(edge_scores_coarse)):
            g1, g2 = int(iu[0][e]), int(iu[1][e])
            scored_rows.append({
                "dataset": ds.name,
                "scorer": "pearson_global_plus_grouped",
                "pair_id": "coarse_vs_fine",
                "edge_id": f"{ds.name}_e{e}",
                "target_id": str(genes[g2]),
                "source_id": str(genes[g1]),
                "score_coarse": float(edge_scores_coarse[e]),
                "score_fine": float(edge_scores_fine[e]),
            })

        # Scorecard (composite + components)
        rss_comp = rss_with_weights(
            edge_scores_coarse, edge_scores_fine,
            k=cfg.top_k, weights=cfg.rss_weights,
        )
        spearman = spearman_rank_stability(edge_scores_coarse, edge_scores_fine)
        base_rec, base_narrative = metric_to_recommendation(rss_comp.composite)

        # Dual null: permute fine labels under (a) global shuffle, (b) within-coarse shuffle
        null_stats = {"global": [], "within_coarse": []}
        for mode in ("global", "within_coarse"):
            for p in range(cfg.n_null_permutations):
                perm_fine = _permute_fine_labels(fine, coarse, mode, rng)
                null_ranking = _score_ranking(x_dense, perm_fine, cfg)
                null_rss = rss_with_weights(
                    edge_scores_coarse, null_ranking.edge_scores,
                    k=cfg.top_k, weights=cfg.rss_weights,
                ).composite
                null_stats[mode].append(null_rss)
            stats = np.asarray(null_stats[mode])
            passed, p_value = _null_pass(rss_comp.composite, stats)
            null_rows.append({
                "dataset": ds.name,
                "null_family": mode,
                "observed_rss": float(rss_comp.composite),
                "null_mean_rss": float(stats.mean()),
                "null_std_rss": float(stats.std(ddof=1)) if len(stats) > 1 else 0.0,
                "null_5pct": float(np.quantile(stats, 0.05)),
                "null_95pct": float(np.quantile(stats, 0.95)),
                "empirical_p": float(p_value),
                "passed_alpha_0.05": bool(passed),
                "n_permutations": int(cfg.n_null_permutations),
            })
        global_pass, _ = _null_pass(rss_comp.composite, np.asarray(null_stats["global"]))
        constrained_pass, _ = _null_pass(rss_comp.composite, np.asarray(null_stats["within_coarse"]))
        dual_rec, dual_narrative = calibrate_dual_null(base_rec, global_pass, constrained_pass)

        row = {
            "dataset": ds.name,
            "scorer": "pearson_global_plus_grouped",
            "pair_id": "coarse_vs_fine",
            "k": int(cfg.top_k),
            "n_genes": int(x_dense.shape[1]),
            "n_cells": int(x_dense.shape[0]),
            "overlap": 1.0 - rss_comp.overlap_loss,
            "jaccard": 1.0 - rss_comp.jaccard_loss,
            "drift_norm": rss_comp.drift_norm,
            "rss_composite": rss_comp.composite,
            "spearman": spearman,
            "global_null_pass": bool(global_pass),
            "constrained_null_pass": bool(constrained_pass),
            "base_recommendation": base_rec,
            "dual_null_recommendation": dual_rec,
            "base_narrative": base_narrative,
            "dual_null_narrative": dual_narrative,
        }

        # Optional triple-null (WP-14): degree-preserving rewire on the
        # (coarse, fine) score matrices.
        if cfg.run_triple_null:
            from ...nulls import DegreePreservingNull
            dp = DegreePreservingNull(top_k=cfg.top_k)
            dp_stats = []
            for _ in range(cfg.n_null_permutations):
                perm = dp.permute(coarse_ranking.score_matrix, None, rng).X
                perm_edges = perm[np.triu_indices(perm.shape[0], 1)]
                dp_rss = rss_with_weights(
                    edge_scores_coarse, perm_edges,
                    k=cfg.top_k, weights=cfg.rss_weights,
                ).composite
                dp_stats.append(dp_rss)
            dp_stats = np.asarray(dp_stats)
            dp_pass, dp_p = _null_pass(rss_comp.composite, dp_stats)
            null_rows.append({
                "dataset": ds.name,
                "null_family": "degree_preserving",
                "observed_rss": float(rss_comp.composite),
                "null_mean_rss": float(dp_stats.mean()),
                "null_std_rss": float(dp_stats.std(ddof=1)) if len(dp_stats) > 1 else 0.0,
                "null_5pct": float(np.quantile(dp_stats, 0.05)),
                "null_95pct": float(np.quantile(dp_stats, 0.95)),
                "empirical_p": float(dp_p),
                "passed_alpha_0.05": bool(dp_pass),
                "n_permutations": int(cfg.n_null_permutations),
            })
            triple_rec, triple_narrative = calibrate_triple_null(
                base_rec, global_pass, constrained_pass, dp_pass,
            )
            row["degree_null_pass"] = bool(dp_pass)
            row["triple_null_recommendation"] = triple_rec
            row["triple_null_narrative"] = triple_narrative

        scorecard_rows.append(row)

        # Within-coarse-group RSS
        for c in np.unique(coarse):
            mask = coarse == c
            if mask.sum() < cfg.min_cells_group:
                continue
            sub_dense = x_dense[mask]
            sub_coarse = np.full(mask.sum(), c)
            sub_fine = fine[mask]
            if len(np.unique(sub_fine)) < 2:
                continue
            sub_coarse_rank = _score_ranking(sub_dense, sub_coarse, cfg)
            sub_fine_rank = _score_ranking(sub_dense, sub_fine, cfg)
            sub_rss = rss_with_weights(
                sub_coarse_rank.edge_scores, sub_fine_rank.edge_scores,
                k=cfg.top_k, weights=cfg.rss_weights,
            )
            within_rows.append({
                "dataset": ds.name,
                "coarse_group": str(c),
                "n_cells": int(mask.sum()),
                "overlap": 1.0 - sub_rss.overlap_loss,
                "jaccard": 1.0 - sub_rss.jaccard_loss,
                "drift_norm": sub_rss.drift_norm,
                "rss_composite": sub_rss.composite,
            })

    # Write CSVs
    scored_df = pd.DataFrame(scored_rows)
    scorecard_df = pd.DataFrame(scorecard_rows)
    null_df = pd.DataFrame(null_rows)
    within_df = pd.DataFrame(within_rows)

    paths = {
        "scored_pairs": out_dir / "scored_pairs.csv",
        "scorecard": out_dir / "scorecard.csv",
        "null_calibration": out_dir / "null_calibration.csv",
        "within_coarse": out_dir / "within_coarse_group.csv",
    }
    scored_df.to_csv(paths["scored_pairs"], index=False)
    scorecard_df.to_csv(paths["scorecard"], index=False)
    null_df.to_csv(paths["null_calibration"], index=False)
    within_df.to_csv(paths["within_coarse"], index=False)
    return paths
