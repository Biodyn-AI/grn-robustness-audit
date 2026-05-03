"""Global cell-label shuffle."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import NullFamily, NullResult, register_null


@register_null
class GlobalShuffleNull(NullFamily):
    """Permute cell labels globally (ignores any grouping structure)."""

    name = "global_shuffle"
    label = "Global cell-label shuffle"
    clear_threshold = 0.05

    def permute(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> NullResult:
        if labels is None:
            raise ValueError("GlobalShuffleNull needs a ``labels`` array")
        perm_labels = labels.copy()
        rng.shuffle(perm_labels)
        return NullResult(X=X, labels=perm_labels, meta={"null": self.name})
