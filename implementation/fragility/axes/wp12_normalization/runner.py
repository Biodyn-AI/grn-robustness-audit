"""Core WP-12 computation — rerun Axis-2 under alternative normalizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from ...metrics import rss_with_weights, spearman_rank_stability
from ..axis2_resolution.runner import (
    Axis2Config,
    Axis2DatasetSpec,
    _permute_fine_labels,
    _null_pass,
    _score_ranking,
    load_h5ad,
    make_hierarchical_fine_labels,
)
from ..axis2_resolution.calibration import calibrate_dual_null, metric_to_recommendation
from ..axis2_resolution.scoring import rank_high_variance_genes


NORMALIZATIONS = ("depth", "size_factor", "pearson_residuals")


def _apply_normalization(X: sparse.spmatrix, kind: str) -> np.ndarray:
    """Return a dense ``(n_cells, n_genes)`` matrix under the chosen scheme."""

    # All three schemes assume the h5ad X is raw counts.
    if kind == "depth":
        counts = np.asarray(X.sum(axis=1)).ravel()
        counts[counts == 0] = 1.0
        scale = 1e4 / counts
        if sparse.issparse(X):
            normed = sparse.diags(scale) @ X
        else:
            normed = (np.asarray(X).T * scale).T
        return np.log1p(normed.toarray() if sparse.issparse(normed) else normed).astype(np.float32)
    if kind == "size_factor":
        # Median-ratio size factors (R-free surrogate for scran)
        counts = np.asarray(X.sum(axis=1)).ravel()
        counts[counts == 0] = 1.0
        sf = counts / np.median(counts)
        scale = 1.0 / sf
        if sparse.issparse(X):
            normed = sparse.diags(scale) @ X
            dense = normed.toarray()
        else:
            dense = (np.asarray(X).T * scale).T
        return np.log1p(dense).astype(np.float32)
    if kind == "pearson_residuals":
        # scanpy's analytic Pearson residuals (Lause et al. 2021)
        import anndata as ad
        adata = ad.AnnData(X=X if not sparse.issparse(X) else X.copy())
        try:
            sc.experimental.pp.normalize_pearson_residuals(adata)
        except Exception:
            # Fallback: per-column z-score on log1p(depth-normalised).
            return _apply_normalization(X, "depth") - np.mean(
                _apply_normalization(X, "depth"), axis=0, keepdims=True,
            )
        dense = adata.X
        if sparse.issparse(dense):
            dense = dense.toarray()
        return dense.astype(np.float32)
    raise ValueError(f"unknown normalization: {kind}")


def run_single_dataset(
    spec: Axis2DatasetSpec,
    normalizations: Sequence[str] = NORMALIZATIONS,
    axis2_cfg: Axis2Config = Axis2Config(n_null_permutations=25),
) -> pd.DataFrame:
    """Rerun Axis-2 scorecard under each normalization. Returns one row per
    (dataset, normalization)."""

    ds = load_h5ad(spec)
    # Rank genes once (on raw expression) so the HVG selection is fixed.
    ranked = rank_high_variance_genes(ds.x, min_detect_frac=axis2_cfg.min_detect_frac)
    gene_idx = ranked[: axis2_cfg.n_top_genes]
    coarse = ds.coarse_labels
    fine = make_hierarchical_fine_labels(coarse, ds.fine_labels)

    rows: List[Dict] = []
    for norm in normalizations:
        X_norm = _apply_normalization(ds.x[:, gene_idx], norm)
        coarse_ranking = _score_ranking(X_norm, coarse, axis2_cfg)
        fine_ranking = _score_ranking(X_norm, fine, axis2_cfg)
        rss_comp = rss_with_weights(
            coarse_ranking.edge_scores, fine_ranking.edge_scores,
            k=axis2_cfg.top_k, weights=axis2_cfg.rss_weights,
        )
        spearman = spearman_rank_stability(
            coarse_ranking.edge_scores, fine_ranking.edge_scores,
        )
        # Dual-null permutation
        rng = np.random.default_rng(20260218 + hash(norm) % 100000)
        null_stats = {"global": [], "within_coarse": []}
        for mode in ("global", "within_coarse"):
            for _ in range(axis2_cfg.n_null_permutations):
                perm_fine = _permute_fine_labels(fine, coarse, mode, rng)
                null_ranking = _score_ranking(X_norm, perm_fine, axis2_cfg)
                null_rss = rss_with_weights(
                    coarse_ranking.edge_scores, null_ranking.edge_scores,
                    k=axis2_cfg.top_k, weights=axis2_cfg.rss_weights,
                ).composite
                null_stats[mode].append(null_rss)

        g_pass, g_p = _null_pass(rss_comp.composite, np.asarray(null_stats["global"]))
        c_pass, c_p = _null_pass(rss_comp.composite, np.asarray(null_stats["within_coarse"]))
        base_rec, _ = metric_to_recommendation(rss_comp.composite)
        dual_rec, _ = calibrate_dual_null(base_rec, g_pass, c_pass)
        rows.append({
            "dataset": spec.name,
            "normalization": norm,
            "rss_composite": float(rss_comp.composite),
            "overlap": 1.0 - rss_comp.overlap_loss,
            "jaccard": 1.0 - rss_comp.jaccard_loss,
            "drift_norm": rss_comp.drift_norm,
            "spearman": spearman,
            "global_p": float(g_p),
            "within_coarse_p": float(c_p),
            "base_recommendation": base_rec,
            "dual_null_recommendation": dual_rec,
        })
    return pd.DataFrame(rows)


def run(
    datasets: Sequence[Axis2DatasetSpec],
    out_dir: Path,
    normalizations: Sequence[str] = NORMALIZATIONS,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[pd.DataFrame] = []
    for spec in datasets:
        all_rows.append(
            run_single_dataset(
                spec, normalizations=normalizations,
                axis2_cfg=Axis2Config(n_null_permutations=25),
            )
        )
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out_path = out_dir / "normalization_ablation.csv"
    df.to_csv(out_path, index=False)
    return {"normalization_ablation": out_path}
