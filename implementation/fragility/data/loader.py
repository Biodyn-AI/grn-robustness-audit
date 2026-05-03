"""AnnData loading with a shared, documented entry point.

All per-axis pipelines must load via :func:`load_anndata`. This keeps
gene-symbol normalization, minimum-expression filtering, and duplicate
gene handling identical across axes so that comparisons across pipelines
are apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .registry import DatasetSpec, resolve


def _normalize_var_names(adata: ad.AnnData) -> ad.AnnData:
    """Uppercase gene symbols and drop duplicates, keeping first occurrence."""

    var = adata.var.copy()
    var.index = [str(g).upper() for g in var.index]
    # Deduplicate gene symbols (first-occurrence wins) before further processing.
    keep = ~pd.Index(var.index).duplicated(keep="first")
    if keep.sum() < len(keep):
        adata = adata[:, keep].copy()
        var = adata.var.copy()
        var.index = [str(g).upper() for g in var.index]
        adata.var_names = var.index
    else:
        adata.var_names = var.index
    return adata


def _drop_low_expression_genes(
    adata: ad.AnnData, min_cells: int
) -> ad.AnnData:
    if min_cells <= 0:
        return adata
    X = adata.X
    if sp.issparse(X):
        per_gene_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        per_gene_counts = (np.asarray(X) > 0).sum(axis=0)
    keep = per_gene_counts >= min_cells
    if keep.sum() < keep.size:
        adata = adata[:, keep].copy()
    return adata


def load_anndata(
    dataset: str | DatasetSpec,
    min_cells_per_gene: int = 10,
    backed: Optional[str] = None,
) -> ad.AnnData:
    """Load an ``AnnData`` by registry name or explicit spec.

    Parameters
    ----------
    dataset
        Registry name (e.g. ``"kidney"``) or a :class:`DatasetSpec`.
    min_cells_per_gene
        Drop genes expressed in fewer than this many cells. ``0`` disables.
    backed
        Passed straight through to :func:`anndata.read_h5ad`.
    """

    spec = resolve(dataset) if isinstance(dataset, str) else dataset

    adata = ad.read_h5ad(spec.path, backed=backed)
    adata = _normalize_var_names(adata)
    adata = _drop_low_expression_genes(adata, min_cells_per_gene)

    # Record provenance on the AnnData itself so downstream code can assert.
    adata.uns["fragility_dataset"] = {
        "name": spec.name,
        "path": str(spec.path),
        "cell_type_key": spec.cell_type_key,
        "donor_key": spec.donor_key or "",
        "batch_key": spec.batch_key or "",
        "method_key": spec.method_key or "",
        "min_cells_per_gene": int(min_cells_per_gene),
    }
    return adata
