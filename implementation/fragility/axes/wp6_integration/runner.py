"""Core WP-6 computation — multi-method integration comparison.

For each (tissue, integration method) combination, we:

1. Load the h5ad.
2. Compute a baseline edge score matrix on the uncorrected expression.
3. Apply the integration method (Harmony / Scanorama) which returns an
   integrated embedding (and re-projects to the expression space for
   correlation-based scoring).
4. Recompute edge scores on the integrated expression.
5. Report rank-stability metrics (Spearman, top-$k$ overlap, RSS).

scVI is stubbed because the current anaconda environment has a
torchvision/lightning incompatibility; enabling it requires a
clean torch+lightning install which is recorded as a future-work item.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# Make sure user-site (where pip --user installs land) is on the path.
import site
for p in [site.getusersitepackages(), site.getusersitepackages().replace("3.11", "3.10")]:
    if p and p not in sys.path:
        sys.path.insert(0, p)


from ...metrics import rss_with_weights, sign_flip_rate, spearman_rank_stability, topk_jaccard, topk_overlap
from ...panels import load_panel
from ...scorers import get_scorer
from ...utils import rng_for
from ..axis2_resolution.runner import Axis2DatasetSpec, load_h5ad


@dataclass
class WP6Config:
    panel: str = "hematopoiesis_76x108"
    scorer: str = "pearson"
    n_cells: int = 2000
    n_pcs: int = 50
    seed_namespace: str = "wp6:default"


def _score(X: np.ndarray, gene_names: list, tf_idx, target_idx, scorer_name: str):
    scorer = get_scorer(scorer_name)
    out = scorer.score(X, tf_idx, target_idx, gene_names)
    return out


def _apply_harmony(X: np.ndarray, batch: np.ndarray, n_pcs: int) -> np.ndarray:
    """Apply Harmony integration in PCA space, then project back."""

    import harmonypy
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_pcs, X.shape[1] - 1), random_state=0)
    Z = pca.fit_transform(X)
    batch_df = pd.DataFrame({"batch": batch.astype(str)})
    ho = harmonypy.run_harmony(
        Z, batch_df, "batch", theta=1.0, max_iter_harmony=20,
    )
    Z_harmony = ho.Z_corr.T if hasattr(ho, "Z_corr") and ho.Z_corr.shape[0] == Z.shape[1] else ho.Z_corr
    # harmonypy.Z_corr is shape (n_components, n_cells); transpose if needed
    if Z_harmony.shape[0] == Z.shape[1]:
        Z_harmony = Z_harmony.T
    X_corr = Z_harmony @ pca.components_
    return X_corr.astype(np.float32)


def _apply_scanorama(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Apply Scanorama integration; returns the dense integrated expression."""

    import scanorama

    unique_batches = np.unique(batch)
    datasets = [X[batch == b] for b in unique_batches]
    gene_lists = [[f"g{i}" for i in range(X.shape[1])] for _ in datasets]
    corrected, _ = scanorama.correct(datasets, gene_lists)
    X_out = np.zeros_like(X)
    for b, corr in zip(unique_batches, corrected):
        mask = batch == b
        X_out[mask] = (
            corr.toarray() if sparse.issparse(corr) else np.asarray(corr)
        ).astype(np.float32)
    return X_out


def _apply_scvi(X: np.ndarray, batch: np.ndarray, n_latent: int = 10,
                max_epochs: int = 50) -> np.ndarray:
    """Apply scVI-tools integration, returning latent-space re-projection.

    scVI learns a variational latent representation that corrects for batch
    via a batch-conditioned decoder. We train a short model (50 epochs with
    early stopping) and project back through the decoder's gene-mean output.
    """

    import anndata as ad
    import scvi

    # scVI expects raw counts; we pass the log-normalized expression as
    # already processed and treat it as a continuous value (Gaussian latent).
    adata = ad.AnnData(X=X.copy())
    adata.obs["batch"] = batch.astype(str)
    # Reset model state each call
    scvi.settings.seed = 20260218
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    model = scvi.model.SCVI(adata, n_latent=n_latent, n_hidden=64, n_layers=1)
    model.train(max_epochs=max_epochs, early_stopping=True,
                check_val_every_n_epoch=10,
                train_size=0.9)
    # Get the mean of the decoder's reconstruction -- this is the "integrated"
    # expression. Use the "expression" normalization mode.
    integrated = model.get_normalized_expression(
        adata, return_numpy=True, library_size=1e4,
    )
    return np.asarray(integrated, dtype=np.float32)


def run(
    datasets: Sequence[Axis2DatasetSpec],
    out_dir: Path,
    config: Optional[WP6Config] = None,
    batch_key_candidates: Sequence[str] = ("_batch", "donor_id", "method"),
) -> Dict[str, Path]:
    cfg = config or WP6Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    scored_rows: List[Dict] = []
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

        rng = rng_for(f"{cfg.seed_namespace}:{spec.name}")
        n_cells_total = ds.x.shape[0]
        n_cells_use = min(cfg.n_cells, n_cells_total)
        cell_idx = rng.choice(n_cells_total, size=n_cells_use, replace=False)
        X = ds.x[cell_idx].toarray().astype(np.float32)

        # Find a batch vector
        batch = None
        batch_key_used = None
        for key in batch_key_candidates:
            if key in ("_batch", "donor_id", "method"):
                # Load batch vector from anndata obs
                try:
                    import anndata as ad
                    a = ad.read_h5ad(spec.path, backed="r")
                    if key in a.obs.columns:
                        b = a.obs[key].astype(str).values[cell_idx]
                        if len(np.unique(b)) >= 2:
                            batch = b
                            batch_key_used = key
                            break
                except Exception:
                    continue
        if batch is None:
            continue

        # Baseline scoring
        base_out = _score(X, gene_names, tf_idx, target_idx, cfg.scorer)
        base_scores = base_out.scores
        base_signs = base_out.signs

        for method in ("harmony", "scanorama", "scvi"):
            try:
                if method == "harmony":
                    X_corr = _apply_harmony(X, batch, cfg.n_pcs)
                elif method == "scanorama":
                    X_corr = _apply_scanorama(X, batch)
                else:  # scvi
                    X_corr = _apply_scvi(X, batch)
            except Exception as e:
                rows.append({
                    "dataset": spec.name, "method": method,
                    "batch_key": batch_key_used,
                    "status": "FAILED",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                })
                continue

            corr_out = _score(X_corr, gene_names, tf_idx, target_idx, cfg.scorer)
            corr_scores = corr_out.scores
            corr_signs = corr_out.signs

            rho = spearman_rank_stability(base_scores, corr_scores)
            j500 = topk_jaccard(base_scores, corr_scores, k=500)
            j1000 = topk_jaccard(base_scores, corr_scores, k=1000)
            rss500 = rss_with_weights(base_scores, corr_scores, k=500)
            rss1000 = rss_with_weights(base_scores, corr_scores, k=1000)
            sf = float("nan")
            if base_signs is not None and corr_signs is not None:
                sf = sign_flip_rate(base_signs, corr_signs)

            rows.append({
                "dataset": spec.name,
                "method": method,
                "batch_key": batch_key_used,
                "status": "OK",
                "n_cells": int(X.shape[0]),
                "n_batches": int(len(np.unique(batch))),
                "spearman": rho,
                "overlap_500": 1.0 - rss500.overlap_loss,
                "jaccard_500": j500,
                "rss_500": rss500.composite,
                "overlap_1000": 1.0 - rss1000.overlap_loss,
                "jaccard_1000": j1000,
                "rss_1000": rss1000.composite,
                "sign_flip_rate": sf,
            })
            # Long-format scored_pairs CSV (compatible with WP-3/WP-10)
            for e_id, (s_base, s_corr) in enumerate(zip(base_scores, corr_scores)):
                scored_rows.append({
                    "dataset": spec.name,
                    "scorer": cfg.scorer,
                    "pair_id": f"baseline_vs_{method}",
                    "edge_id": f"{spec.name}_{method}_{e_id}",
                    "source_id": base_out.tf_names[e_id],
                    "target_id": base_out.target_names[e_id],
                    "score_coarse": float(s_base),
                    "score_fine": float(s_corr),
                })

    summary_df = pd.DataFrame(rows)
    scored_df = pd.DataFrame(scored_rows)
    paths = {
        "integration_cross_method": out_dir / "integration_cross_method.csv",
        "scored_pairs": out_dir / "scored_pairs.csv",
    }
    summary_df.to_csv(paths["integration_cross_method"], index=False)
    scored_df.to_csv(paths["scored_pairs"], index=False)
    return paths
