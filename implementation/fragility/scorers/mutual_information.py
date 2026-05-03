"""Mutual-information edge scorer (ARACNe/CLR family)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .base import EdgeScorer, register_scorer


@register_scorer
class MutualInfoScorer(EdgeScorer):
    """k-nearest-neighbour mutual information between TF and target expression.

    Uses scikit-learn's ``mutual_info_regression`` with the default
    ``n_neighbors=3`` (Kraskov estimator). Cost scales as
    ``O(n_cells * n_tfs * n_targets)``; callers should subsample cells for
    very large datasets.
    """

    name = "mutual_info"
    label = "Mutual information"
    supports_sign = False

    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        n_tfs = len(tf_idx)
        n_targets = len(target_idx)
        scores = np.zeros((n_tfs, n_targets), dtype=float)

        # mutual_info_regression vectorises over predictors but not over targets.
        tfs = X[:, tf_idx]
        n_neighbors = int(self.options.get("n_neighbors", 3))
        random_state = int(self.options.get("random_state", 20260218))
        for j, t in enumerate(target_idx):
            y = X[:, t]
            scores[:, j] = mutual_info_regression(
                tfs,
                y,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
        return scores, None
