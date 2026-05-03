"""Core computation for WP-2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...metrics import topk_jaccard, topk_overlap
from ...utils import rng_for
from ..axis2_resolution.runner import (
    Axis2DatasetSpec,
    LoadedDataset,
    load_h5ad,
)
from ..axis2_resolution.scoring import (
    abs_correlation,
    rank_high_variance_genes,
)


@dataclass
class MVCCConfig:
    n_top_genes: int = 600
    min_detect_frac: float = 0.05
    cell_sizes: Tuple[int, ...] = (100, 200, 500, 1000, 2000, 3000, 5000, 8000, 15000)
    anchors: Tuple[int, ...] = (3000, 8000, 15000)    # clipped to dataset max
    ks: Tuple[int, ...] = (100, 500, 1000, 5000)
    n_subsamples: int = 16
    mvcc_threshold_jaccard: float = 0.5
    mvcc_threshold_k: int = 1000
    seed_namespace: str = "wp2:default"


def _subsample_score(
    ds: LoadedDataset,
    gene_idx: np.ndarray,
    cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the flattened upper-triangular score vector for a random subsample."""

    total = ds.x.shape[0]
    if cells >= total:
        idx = np.arange(total)
    else:
        idx = rng.choice(total, size=cells, replace=False)
    sub = ds.x[idx][:, gene_idx]
    dense = sub.toarray().astype(np.float32)
    corr = abs_correlation(dense)
    iu = np.triu_indices(corr.shape[0], 1)
    return corr[iu]


def _emergence_curve(
    ds: LoadedDataset,
    gene_idx: np.ndarray,
    anchor: int,
    cfg: MVCCConfig,
    rng_ns: str,
) -> pd.DataFrame:
    """Return per-(cells, k) Jaccard + overlap relative to ``anchor`` cells.

    Multiple independent draws at ``anchor`` cells give a bootstrap CI on
    the anchor itself; multiple draws at each smaller size give a CI on
    the observation.
    """

    total = ds.x.shape[0]
    if anchor > total:
        return pd.DataFrame()

    rng = rng_for(f"{rng_ns}:{ds.name}:anchor{anchor}")
    # Compute ``cfg.n_subsamples`` independent anchor draws so the curves
    # have a proper CI on the reference.
    anchors = [_subsample_score(ds, gene_idx, anchor, rng) for _ in range(min(cfg.n_subsamples, 4))]

    rows: List[Dict] = []
    for cells in cfg.cell_sizes:
        if cells > total:
            continue
        for draw in range(cfg.n_subsamples):
            rng_draw = rng_for(f"{rng_ns}:{ds.name}:anchor{anchor}:size{cells}:draw{draw}")
            scores = _subsample_score(ds, gene_idx, cells, rng_draw)
            for k in cfg.ks:
                # Average across anchor draws
                jac_samples = [topk_jaccard(anchor_scores, scores, k=k) for anchor_scores in anchors]
                ovl_samples = [topk_overlap(anchor_scores, scores, k=k) for anchor_scores in anchors]
                rows.append({
                    "dataset": ds.name,
                    "scorer": "pearson",
                    "anchor_cells": int(anchor),
                    "cells": int(cells),
                    "k": int(k),
                    "draw": int(draw),
                    "jaccard_mean": float(np.mean(jac_samples)),
                    "jaccard_std": float(np.std(jac_samples, ddof=1)) if len(jac_samples) > 1 else 0.0,
                    "overlap_mean": float(np.mean(ovl_samples)),
                    "overlap_std": float(np.std(ovl_samples, ddof=1)) if len(ovl_samples) > 1 else 0.0,
                })
    return pd.DataFrame(rows)


def _mvcc_estimate(
    curve: pd.DataFrame,
    k: int,
    threshold: float,
) -> Optional[int]:
    """Smallest cell count whose mean Jaccard at ``k`` exceeds ``threshold``."""

    subset = curve[curve["k"] == k]
    # Use mean Jaccard across draws
    grouped = (
        subset.groupby("cells", as_index=False)["jaccard_mean"].mean()
        .sort_values("cells")
    )
    passing = grouped[grouped["jaccard_mean"] >= threshold]
    if passing.empty:
        return None
    return int(passing.iloc[0]["cells"])


def run(
    datasets: Sequence[Axis2DatasetSpec],
    out_dir: Path,
    config: Optional[MVCCConfig] = None,
) -> Dict[str, Path]:
    """Run WP-2 end-to-end, emitting curve CSVs + MVCC summary + anchor ablation."""

    cfg = config or MVCCConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: List[pd.DataFrame] = []
    mvcc_rows: List[Dict] = []

    for spec in datasets:
        ds = load_h5ad(spec)
        ranked = rank_high_variance_genes(ds.x, min_detect_frac=cfg.min_detect_frac)
        gene_idx = ranked[: cfg.n_top_genes]

        total_cells = ds.x.shape[0]
        usable_anchors = tuple(a for a in cfg.anchors if a <= total_cells)

        for anchor in usable_anchors:
            curve = _emergence_curve(ds, gene_idx, anchor, cfg, rng_ns=cfg.seed_namespace)
            if curve.empty:
                continue
            curve_rows.append(curve)
            mvcc = _mvcc_estimate(
                curve, k=cfg.mvcc_threshold_k, threshold=cfg.mvcc_threshold_jaccard,
            )
            mvcc_rows.append({
                "dataset": ds.name,
                "scorer": "pearson",
                "anchor_cells": int(anchor),
                "n_top_genes": int(cfg.n_top_genes),
                "mvcc_threshold_k": int(cfg.mvcc_threshold_k),
                "mvcc_threshold_jaccard": float(cfg.mvcc_threshold_jaccard),
                "mvcc_cells": mvcc if mvcc is not None else -1,
                "mvcc_reached": mvcc is not None,
                "total_cells_available": int(total_cells),
            })

    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    mvcc_df = pd.DataFrame(mvcc_rows)

    curves_path = out_dir / "mvcc_full_curves.csv"
    mvcc_path = out_dir / "mvcc_anchor_sensitivity.csv"
    curves.to_csv(curves_path, index=False)
    mvcc_df.to_csv(mvcc_path, index=False)
    return {"mvcc_full_curves": curves_path, "mvcc_anchor_sensitivity": mvcc_path}
