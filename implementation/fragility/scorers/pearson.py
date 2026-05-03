"""Pearson-correlation edge scorer (the paper's original baseline)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import EdgeScorer, register_scorer


@register_scorer
class PearsonScorer(EdgeScorer):
    """Absolute Pearson correlation as edge score, sign preserved separately."""

    name = "pearson"
    label = "Pearson correlation"
    supports_sign = True

    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        # Column-wise normalisation: (X - mean) / (std * sqrt(n-1))
        n = X.shape[0]
        X_centered = X - X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        X_norm = X_centered / std
        # Correlation for each (tf, target) pair via dot products.
        Xt = X_norm[:, tf_idx]        # (n, Ktf)
        Xg = X_norm[:, target_idx]    # (n, Kt)
        corr = (Xt.T @ Xg) / (n - 1)  # (Ktf, Kt)
        # Zero-variance TFs/targets propagate to NaN -> set to 0.
        corr = np.nan_to_num(corr, nan=0.0)
        signs = np.sign(corr)
        return np.abs(corr), signs
