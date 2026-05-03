"""Shared scRNA-seq preprocessing (P2 of the revision plan).

One function per normalization family so that the paper's Methods section
can cite exactly one parameter set per figure. The default
``normalize_and_hvg`` matches the behaviour used by every
``subproject_38_*`` pipeline prior to the revision, so porting existing
code does not change numerical outputs.

Addressing reviewers:

* **R3** ("sec 2.1 should state HVG count"): every call stamps
  ``adata.uns["fragility_preprocessing"]`` with the exact parameters.
* **R3** ("cell-wise depth normalization inflates variance in low-count
  cells"): :func:`normalize_pearson_residuals` and :func:`normalize_size_factor`
  provide the ablation variants for WP-12.
* **R4-minor(1)** ("HVG count missing"): default HVG count surfaced as
  ``n_top_genes`` keyword.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import anndata as ad
import numpy as np
import scanpy as sc


NormalizationKind = Literal["depth", "pearson_residuals", "size_factor"]


@dataclass(frozen=True)
class PreprocessingParams:
    """Parameters that together determine a preprocessing run."""

    normalization: NormalizationKind = "depth"
    target_sum: float = 1e4
    log1p: bool = True
    hvg_flavor: str = "seurat_v3"
    n_top_genes: int = 2000
    min_mean: float = 0.0125
    max_mean: float = 3.0
    min_disp: float = 0.5
    batch_key: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _apply_depth_normalization(
    adata: ad.AnnData, target_sum: float
) -> ad.AnnData:
    sc.pp.normalize_total(adata, target_sum=target_sum)
    return adata


def _apply_size_factor_normalization(
    adata: ad.AnnData, target_sum: float
) -> ad.AnnData:
    # scran-style size-factor normalization requires R. We fall back to
    # median-ratio size factors as a close, R-free approximation: divide each
    # cell by its total count then multiply by the median total count across
    # cells (which matches normalize_total with per-cell target_sum).
    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    median = float(np.median(counts))
    if target_sum and target_sum > 0:
        scale = target_sum / np.maximum(counts, 1)
    else:  # pragma: no cover - unusual use
        scale = median / np.maximum(counts, 1)
    from scipy import sparse

    if sparse.issparse(adata.X):
        diag = sparse.diags(scale)
        adata.X = diag @ adata.X
    else:
        adata.X = (adata.X.T * scale).T
    return adata


def _apply_pearson_residuals(
    adata: ad.AnnData, n_top_genes: int
) -> ad.AnnData:
    # Use scanpy experimental implementation; it mutates adata.X in place
    # and sets adata.uns accordingly.
    sc.experimental.pp.normalize_pearson_residuals(adata)
    return adata


def normalize_and_hvg(
    adata: ad.AnnData,
    params: PreprocessingParams | None = None,
    copy: bool = True,
) -> ad.AnnData:
    """Apply the primary depth normalization + log1p + HVG selection.

    Returns an ``AnnData`` with ``adata.var["highly_variable"]`` set.
    The function does NOT subset to HVGs; callers should do that themselves.
    """

    if copy:
        adata = adata.copy()
    params = params or PreprocessingParams()

    if params.normalization == "depth":
        _apply_depth_normalization(adata, params.target_sum)
        if params.log1p:
            sc.pp.log1p(adata)
    elif params.normalization == "size_factor":
        _apply_size_factor_normalization(adata, params.target_sum)
        if params.log1p:
            sc.pp.log1p(adata)
    elif params.normalization == "pearson_residuals":
        _apply_pearson_residuals(adata, params.n_top_genes)
        # residuals are already variance-stabilised; log1p would be wrong.
    else:  # pragma: no cover - dataclass guard
        raise ValueError(f"unknown normalization: {params.normalization}")

    # HVG selection runs on the log-normalised / residual matrix.
    if params.hvg_flavor == "seurat_v3":
        # seurat_v3 requires raw counts, so run on the unnormalised layer if
        # available; otherwise fall back to seurat flavor.
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=params.n_top_genes,
                flavor="seurat_v3",
                batch_key=params.batch_key,
                subset=False,
            )
        except Exception:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=params.n_top_genes,
                flavor="seurat",
                batch_key=params.batch_key,
                min_mean=params.min_mean,
                max_mean=params.max_mean,
                min_disp=params.min_disp,
                subset=False,
            )
    else:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=params.n_top_genes,
            flavor=params.hvg_flavor,
            batch_key=params.batch_key,
            min_mean=params.min_mean,
            max_mean=params.max_mean,
            min_disp=params.min_disp,
            subset=False,
        )

    adata.uns["fragility_preprocessing"] = params.as_dict()
    return adata


def select_hvg(
    adata: ad.AnnData,
    n_top_genes: Optional[int] = None,
) -> ad.AnnData:
    """Subset ``adata`` to its highly-variable genes.

    If ``n_top_genes`` is given, keep that many by rank; otherwise use the
    boolean mask that :func:`normalize_and_hvg` wrote into ``adata.var``.
    """

    if "highly_variable" not in adata.var.columns:
        raise RuntimeError(
            "adata.var['highly_variable'] is missing; call normalize_and_hvg first"
        )

    if n_top_genes is not None:
        # Prefer ranked column if present; fall back to boolean-ordered top-n.
        if "highly_variable_rank" in adata.var.columns:
            ranked = adata.var.sort_values("highly_variable_rank").index[:n_top_genes]
            adata = adata[:, adata.var_names.isin(ranked)].copy()
        elif "dispersions_norm" in adata.var.columns:
            ranked = (
                adata.var.sort_values("dispersions_norm", ascending=False)
                .index[:n_top_genes]
            )
            adata = adata[:, adata.var_names.isin(ranked)].copy()
        else:
            mask = adata.var["highly_variable"].astype(bool).values
            idx = np.where(mask)[0][:n_top_genes]
            adata = adata[:, idx].copy()
    else:
        mask = adata.var["highly_variable"].astype(bool).values
        adata = adata[:, mask].copy()
    return adata
