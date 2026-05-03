"""Base ``EdgeScorer`` contract.

All five axes consume scorers through this interface. The contract
intentionally accepts *dense-ish* input: a ``(n_cells, n_genes)`` matrix
plus integer column indices for TFs/targets. Scorers can be stateful but
must be safe to re-score with fresh data (no cached coupling between
runs).

The contract returns an ``EdgeScores`` object rather than a plain array
so that downstream metrics can separate "score" from "rank" and from
"sign" without further introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EdgeScores:
    """Container for scored TF–target edges."""

    tf_names: Sequence[str]
    target_names: Sequence[str]
    scores: np.ndarray           # shape (n_edges,), float
    ranks: np.ndarray            # shape (n_edges,), int (0 = best)
    signs: Optional[np.ndarray] = None  # shape (n_edges,), -1/0/+1 when meaningful
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def edges(self) -> List[tuple]:
        return list(zip(self.tf_names, self.target_names))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class EdgeScorer(ABC):
    """Minimal interface every scorer implements.

    Concrete subclasses must implement :meth:`_score`. The public entry
    point :meth:`score` adds the rank/sign bookkeeping so all scorers emit
    an identical ``EdgeScores`` object.
    """

    #: short, lower_snake_case identifier used in config files and output CSVs
    name: str = ""
    #: human-readable label for figure legends
    label: str = ""
    #: ``True`` if the scorer's output is naturally signed (positive vs
    #: negative correlations). Otherwise callers skip sign-based metrics.
    supports_sign: bool = False

    def __init__(self, **kwargs):
        self.options: Dict[str, object] = dict(kwargs)

    # ---- concrete public API -----------------------------------------

    def score(
        self,
        X,
        tf_idx: Sequence[int],
        target_idx: Sequence[int],
        gene_names: Sequence[str],
    ) -> EdgeScores:
        """Score every TF × target pair in the Cartesian product.

        Parameters
        ----------
        X
            Either a dense ``np.ndarray`` or ``scipy.sparse`` of shape
            ``(n_cells, n_genes)`` with already-normalised expression.
        tf_idx, target_idx
            Integer column indices for TFs / targets in ``X``.
        gene_names
            All gene names in ``X``, used for provenance.
        """

        tf_idx = np.asarray(tf_idx, dtype=int)
        target_idx = np.asarray(target_idx, dtype=int)
        if tf_idx.size == 0 or target_idx.size == 0:
            raise ValueError("tf_idx and target_idx must be non-empty")

        X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)

        raw, signs = self._score(X_dense, tf_idx, target_idx)

        # Flatten to per-edge representation (row-major: TF-major order).
        if raw.ndim != 2:
            raise RuntimeError(
                f"{type(self).__name__}._score returned ndim={raw.ndim}; expected 2"
            )
        if raw.shape != (len(tf_idx), len(target_idx)):
            raise RuntimeError(
                f"{type(self).__name__}._score returned shape {raw.shape}; "
                f"expected ({len(tf_idx)}, {len(target_idx)})"
            )

        scores = raw.ravel(order="C")
        # Ranks: largest score -> rank 0. Ties broken deterministically by
        # original edge order (stable mergesort on -score).
        order = np.argsort(-scores, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))

        tf_names_arr = np.asarray(gene_names)[tf_idx]
        target_names_arr = np.asarray(gene_names)[target_idx]
        tf_names = np.repeat(tf_names_arr, len(target_idx))
        target_names = np.tile(target_names_arr, len(tf_idx))

        signs_flat: Optional[np.ndarray] = None
        if signs is not None:
            signs_flat = signs.ravel(order="C")

        return EdgeScores(
            tf_names=list(tf_names),
            target_names=list(target_names),
            scores=scores,
            ranks=ranks,
            signs=signs_flat,
            meta={
                "scorer": self.name,
                "n_tfs": int(len(tf_idx)),
                "n_targets": int(len(target_idx)),
                "n_cells": int(X_dense.shape[0]),
                "n_genes": int(X_dense.shape[1]),
                **dict(self.options),
            },
        )

    # ---- subclass hook ------------------------------------------------

    @abstractmethod
    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return a ``(scores[n_tfs, n_targets], signs_or_None)`` pair."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, Type[EdgeScorer]] = {}


def register_scorer(cls: Type[EdgeScorer]) -> Type[EdgeScorer]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a non-empty .name")
    _REGISTRY[cls.name] = cls
    return cls


def list_scorers() -> List[str]:
    return sorted(_REGISTRY)


def get_scorer(name: str, **kwargs) -> EdgeScorer:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown scorer '{name}'. Known: {list_scorers()}"
        )
    return _REGISTRY[name](**kwargs)
