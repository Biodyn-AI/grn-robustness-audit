"""Axis-2 edge-scoring primitives.

The scoring rule used by Axis 2 blends a global and a per-group absolute
Pearson correlation over the high-variance-gene × high-variance-gene
matrix. This differs from the other axes, which score only over a
panel's TF × target pairs; Axis 2's panel is "all HVG pairs" by design,
so that the coarse-vs-fine question stays inside one shared universe of
edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class RankingResult:
    score_matrix: np.ndarray     # (G, G) absolute blend scores
    edge_scores: np.ndarray      # upper-triangular flatten
    edge_order: np.ndarray       # indices that sort edge_scores descending
    rank_vector: np.ndarray      # position (1-indexed) of each edge
    top_group_summary: List[Tuple[str, int]]


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (x - mu) / sd


def abs_correlation(x_dense: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation matrix over columns of ``x_dense``."""

    if x_dense.shape[0] < 3:
        out = np.zeros((x_dense.shape[1], x_dense.shape[1]), dtype=np.float32)
        return out
    z = _standardize_columns(x_dense.astype(np.float32, copy=False))
    corr = (z.T @ z) / float(max(z.shape[0] - 1, 1))
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def grouped_max_abs_correlation(
    x_dense: np.ndarray,
    labels: np.ndarray,
    min_cells: int,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Element-wise max absolute correlation across label groups."""

    unique_labels, counts = np.unique(labels, return_counts=True)
    summary = sorted(
        [(str(label), int(count)) for label, count in zip(unique_labels, counts)],
        key=lambda t: t[1],
        reverse=True,
    )
    agg = np.zeros((x_dense.shape[1], x_dense.shape[1]), dtype=np.float32)
    used_groups: List[Tuple[str, int]] = []
    for label, count in summary:
        if count < min_cells:
            continue
        mask = labels == label
        corr = abs_correlation(x_dense[mask])
        agg = np.maximum(agg, corr)
        used_groups.append((label, count))
    np.fill_diagonal(agg, 0.0)
    return agg, used_groups


def rank_vector_from_score_matrix(score: np.ndarray) -> np.ndarray:
    """Return a 1-indexed rank vector over the upper-triangular edges."""

    iu = np.triu_indices(score.shape[0], 1)
    edge_scores = score[iu]
    edge_order = np.argsort(-edge_scores, kind="mergesort")
    rank_vector = np.empty(edge_order.shape[0], dtype=np.int64)
    rank_vector[edge_order] = np.arange(1, edge_order.shape[0] + 1, dtype=np.int64)
    return rank_vector


def infer_rankings(
    x_dense: np.ndarray,
    labels: np.ndarray,
    min_cells_group: int,
    global_weight: float,
) -> RankingResult:
    """Global-plus-grouped-max absolute correlation blended score."""

    global_corr = abs_correlation(x_dense)
    group_corr, used_groups = grouped_max_abs_correlation(
        x_dense, labels, min_cells=min_cells_group
    )
    score = (global_weight * global_corr) + ((1.0 - global_weight) * group_corr)
    np.fill_diagonal(score, 0.0)

    iu = np.triu_indices(score.shape[0], 1)
    edge_scores = score[iu]
    edge_order = np.argsort(-edge_scores, kind="mergesort")
    rank_vector = rank_vector_from_score_matrix(score)
    return RankingResult(
        score_matrix=score,
        edge_scores=edge_scores,
        edge_order=edge_order,
        rank_vector=rank_vector,
        top_group_summary=used_groups,
    )


def rank_high_variance_genes(
    x: sp.csr_matrix,
    min_detect_frac: float,
) -> np.ndarray:
    """Return gene indices in descending-variance order, filtered by detection."""

    n_cells = x.shape[0]
    detect_counts = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float64)
    detect_frac = detect_counts / max(float(n_cells), 1.0)
    mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float64)
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel().astype(np.float64)
    var = np.maximum(mean_sq - np.square(mean), 0.0)
    keep = np.isfinite(var) & (detect_frac >= min_detect_frac)
    candidate_idx = np.where(keep)[0]
    if candidate_idx.size == 0:
        raise ValueError("No genes passed detection/variance filters.")
    ranked = candidate_idx[np.argsort(var[candidate_idx])[::-1]]
    return ranked
