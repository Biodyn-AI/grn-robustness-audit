"""Rank-permutation null used by WP-3 to build the empirical RSS reference.

Unlike the data-level nulls, this one operates directly on pre-computed
ranks — it generates two unrelated random rankings of the same length
and measures what an RSS (or any two-list comparison statistic) would
produce on truly independent rankings. WP-3 uses this to give readers a
calibrated scale for RSS values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import NullFamily, NullResult, register_null


@register_null
class RankPermutationNull(NullFamily):
    """Permute ranks uniformly at random. Used for composite-metric nulls."""

    name = "rank_permutation"
    label = "Independent-ranking null"
    clear_threshold = 0.05

    def permute(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> NullResult:
        # Here ``X`` is interpreted as an ``(n_edges,)`` rank vector.
        n = len(X)
        perm = rng.permutation(n)
        return NullResult(X=perm, labels=labels, meta={"null": self.name})
