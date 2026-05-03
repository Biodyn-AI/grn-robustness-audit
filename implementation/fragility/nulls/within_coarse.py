"""Within-coarse-group label shuffle.

Used in Axis 2 to test whether fine-level clustering carries signal
beyond the coarse partition. Cell labels are permuted only among cells
that share the same coarse label, so across-group structure is preserved
but within-group fine structure is destroyed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import NullFamily, NullResult, register_null


@register_null
class WithinCoarseShuffleNull(NullFamily):
    """Permute fine labels within each coarse group independently."""

    name = "within_coarse_shuffle"
    label = "Within-coarse label shuffle"
    clear_threshold = 0.05

    def permute(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> NullResult:
        if labels is None:
            raise ValueError("WithinCoarseShuffleNull needs ``labels``")

        # labels is expected to be a (n_cells, 2) array with columns
        # [coarse, fine]. The permutation shuffles fine within each
        # coarse group.
        if labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError(
                "within_coarse_shuffle expects labels shape (n_cells, 2) "
                "with columns [coarse, fine]"
            )
        coarse = labels[:, 0]
        fine = labels[:, 1].copy()
        for group in np.unique(coarse):
            idx = np.where(coarse == group)[0]
            perm = rng.permutation(idx)
            fine[idx] = fine[perm]
        perm_labels = np.stack([coarse, fine], axis=1)
        return NullResult(X=X, labels=perm_labels, meta={"null": self.name})
