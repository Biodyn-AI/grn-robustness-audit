"""Axis-3 core computation — pluggable scorer, per-cell-type bootstraps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...metrics import (
    null_separation_auc,
    spearman_rank_stability,
    tail_gap,
    topk_jaccard,
)
from ...panels import load_panel
from ...scorers import EdgeScorer, get_scorer
from ...utils import rng_for
from ..axis2_resolution.runner import Axis2DatasetSpec, load_h5ad


@dataclass
class Axis3Config:
    panel: str = "primary"
    scorer: str = "pearson"
    top_k: int = 100
    n_bootstraps: int = 8
    subsample_fraction: float = 0.8
    min_cells: int = 40
    rarity_quartile: float = 0.25   # bottom quartile = rare, top = abundant
    n_null_gene_shuffles: int = 20
    n_null_cell_perms: int = 20
    seed_namespace: str = "axis3:default"


def _rarity_label(counts: Dict[str, int]) -> Dict[str, str]:
    """Assign rare/intermediate/abundant by within-dataset count quartiles."""

    vals = np.array(list(counts.values()))
    q25 = np.quantile(vals, 0.25)
    q75 = np.quantile(vals, 0.75)
    out = {}
    for name, c in counts.items():
        if c <= q25:
            out[name] = "rare"
        elif c >= q75:
            out[name] = "abundant"
        else:
            out[name] = "intermediate"
    return out


def _score_cells(
    X: np.ndarray,
    gene_names: Sequence[str],
    tf_idx: np.ndarray,
    target_idx: np.ndarray,
    scorer: EdgeScorer,
) -> np.ndarray:
    out = scorer.score(X, tf_idx, target_idx, gene_names)
    return out.scores


def _cell_type_metrics(
    X: np.ndarray,
    gene_names: Sequence[str],
    tf_idx: np.ndarray,
    target_idx: np.ndarray,
    cfg: Axis3Config,
    scorer: EdgeScorer,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute stability + Jaccard + null-AUC + tail-gap for one cell type."""

    n_cells = X.shape[0]
    # Bootstrap replicates
    boot_scores: List[np.ndarray] = []
    for _ in range(cfg.n_bootstraps):
        size = max(1, int(cfg.subsample_fraction * n_cells))
        idx = rng.choice(n_cells, size=size, replace=False)
        boot_scores.append(_score_cells(X[idx], gene_names, tf_idx, target_idx, scorer))
    boot_scores = np.stack(boot_scores, axis=0)  # (B, n_edges)

    # Stability: mean pairwise Spearman across bootstrap replicates
    rho_vals: List[float] = []
    jaccards: List[float] = []
    for i in range(len(boot_scores)):
        for j in range(i + 1, len(boot_scores)):
            rho = spearman_rank_stability(boot_scores[i], boot_scores[j])
            if not np.isnan(rho):
                rho_vals.append(abs(rho))
            jaccards.append(topk_jaccard(boot_scores[i], boot_scores[j], k=cfg.top_k))
    stability = float(np.mean(rho_vals)) if rho_vals else float("nan")
    tk_jaccard = float(np.nanmean(jaccards)) if jaccards else float("nan")

    # Null: cell-permutation null (shuffle gene-cell correspondence).
    # Only the TF + target columns matter for edge scoring, so we restrict
    # the shuffle to those and use vectorised per-column permutation
    # (``apply_along_axis`` is 20–50x faster than a Python gene loop).
    observed = _score_cells(X, gene_names, tf_idx, target_idx, scorer)
    used_cols = np.unique(np.concatenate([tf_idx, target_idx]))
    n_cells_loc = X.shape[0]
    null_cell = []
    for _ in range(cfg.n_null_cell_perms):
        X_perm = X.copy()
        # generate one permutation matrix (n_cells, n_used_cols) and gather.
        perms = np.argsort(rng.random((n_cells_loc, used_cols.size)), axis=0)
        X_perm[:, used_cols] = np.take_along_axis(
            X[:, used_cols], perms, axis=0
        )
        null_cell.append(_score_cells(X_perm, gene_names, tf_idx, target_idx, scorer))
    null_cell = np.concatenate(null_cell)
    auc = null_separation_auc(observed, null_cell)
    tg = tail_gap(observed, null_cell, tail_fraction=0.05)

    return {
        "stability": stability,
        "topk_jaccard": tk_jaccard,
        "null_auc": auc,
        "tail_gap": tg,
    }


def run(
    datasets: Sequence[Axis2DatasetSpec],
    out_dir: Path,
    config: Optional[Axis3Config] = None,
) -> Dict[str, Path]:
    """Run Axis-3 on a list of h5ads and emit the per-cell-type CSV."""

    cfg = config or Axis3Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for spec in datasets:
        ds = load_h5ad(spec)
        gene_names = [str(g).upper() for g in ds.gene_ids]

        panel = load_panel(cfg.panel, gene_universe=gene_names)
        if panel.empty:
            continue
        tfs = sorted(panel["source"].unique())
        targets = sorted(panel["target"].unique())
        tf_idx = np.array([gene_names.index(t) for t in tfs])
        target_idx = np.array([gene_names.index(t) for t in targets])

        # Pre-compute the scorer once (stateful-init fine, score(...) is stateless).
        scorer_kwargs = {}
        if cfg.scorer in ("grnboost2", "genie3", "scgpt_attention"):
            scorer_kwargs["gene_names"] = gene_names
        scorer = get_scorer(cfg.scorer, **scorer_kwargs)

        # Counts per cell type
        cts = pd.Series(ds.coarse_labels)
        counts = cts.value_counts().to_dict()
        eligible = {k: v for k, v in counts.items() if v >= cfg.min_cells}
        if not eligible:
            continue
        rarity = _rarity_label(eligible)

        for cell_type, n in eligible.items():
            mask = ds.coarse_labels == cell_type
            X_ct = ds.x[mask].toarray().astype(np.float32)
            rng = rng_for(f"{cfg.seed_namespace}:{spec.name}:{cell_type}")
            metrics = _cell_type_metrics(
                X_ct, gene_names, tf_idx, target_idx, cfg, scorer, rng,
            )
            rows.append({
                "dataset": spec.name,
                "cell_type": str(cell_type),
                "rarity": rarity[cell_type],
                "cell_count": int(n),
                **metrics,
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / "cell_type_metrics.csv"
    df.to_csv(out_path, index=False)
    return {"cell_type_metrics": out_path}
