"""Tree-based scorers: GRNBoost2 and GENIE3.

Uses sklearn directly rather than arboreto to avoid the dask/multiprocessing
fork issues when running inside this conda environment. The per-target
regression recipe matches Moerman et al. 2019 (GRNBoost2) and Huynh-Thu
et al. 2010 (GENIE3).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import EdgeScorer, register_scorer


@register_scorer
class GRNBoost2Scorer(EdgeScorer):
    """Gradient-boosted regression importance (Moerman et al. 2019).

    For each target ``g``, fit a ``GradientBoostingRegressor`` with the TFs
    as predictors and use the feature-importance vector as the edge-score
    row. Single-threaded sklearn to avoid multiprocessing issues in this
    conda environment.
    """

    name = "grnboost2"
    label = "GRNBoost2"
    supports_sign = False

    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        from sklearn.ensemble import GradientBoostingRegressor

        random_state = int(self.options.get("random_state", 20260218))
        n_estimators = int(self.options.get("n_estimators", 500))
        max_features = self.options.get("max_features", "sqrt")
        tfs = X[:, tf_idx]
        n_tfs = tfs.shape[1]
        scores = np.zeros((n_tfs, len(target_idx)), dtype=float)

        for j, t in enumerate(target_idx):
            y = X[:, t]
            if np.var(y) < 1e-12:
                continue
            # When a TF is also the target gene, exclude it to avoid trivial
            # self-importance (identity regressor).
            keep = np.ones(n_tfs, dtype=bool)
            for i, ti in enumerate(tf_idx):
                if ti == t:
                    keep[i] = False
            if not keep.any():
                continue
            Xk = tfs[:, keep]
            reg = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=random_state,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.9,
            )
            reg.fit(Xk, y)
            # importances normalise per-target (Moerman et al. protocol).
            importances = reg.feature_importances_
            # Place importances back into full TF slots
            full = np.zeros(n_tfs, dtype=float)
            full[keep] = importances
            scores[:, j] = full
        return scores, None


@register_scorer
class GENIE3Scorer(EdgeScorer):
    """Random-forest feature importance (Huynh-Thu et al. 2010)."""

    name = "genie3"
    label = "GENIE3"
    supports_sign = False

    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        from sklearn.ensemble import RandomForestRegressor

        random_state = int(self.options.get("random_state", 20260218))
        n_estimators = int(self.options.get("n_estimators", 500))
        max_features = self.options.get("max_features", "sqrt")
        n_jobs = int(self.options.get("n_jobs", 1))  # single-threaded for safety
        tfs = X[:, tf_idx]
        n_tfs = tfs.shape[1]
        scores = np.zeros((n_tfs, len(target_idx)), dtype=float)

        for j, t in enumerate(target_idx):
            y = X[:, t]
            if np.var(y) < 1e-12:
                continue
            keep = np.ones(n_tfs, dtype=bool)
            for i, ti in enumerate(tf_idx):
                if ti == t:
                    keep[i] = False
            if not keep.any():
                continue
            Xk = tfs[:, keep]
            reg = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            reg.fit(Xk, y)
            importances = reg.feature_importances_
            full = np.zeros(n_tfs, dtype=float)
            full[keep] = importances
            scores[:, j] = full
        return scores, None
