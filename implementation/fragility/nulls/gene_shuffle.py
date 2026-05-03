"""Gene-wise shuffle: permute expression across cells, independently per gene."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import NullFamily, NullResult, register_null


@register_null
class GeneShuffleNull(NullFamily):
    """Per-gene cell-axis shuffle; destroys co-expression structure."""

    name = "gene_shuffle"
    label = "Gene-wise permutation"
    clear_threshold = 0.05

    def permute(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> NullResult:
        X_perm = X.copy()
        for j in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, j])
        return NullResult(X=X_perm, labels=labels, meta={"null": self.name})
