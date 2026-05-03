"""Rank-stability metrics.

Spearman rank stability is the Spearman ρ between two edge-score rank
vectors produced by the same scorer on two perturbed inputs (subsamples,
resolutions, donor splits, ...). ρ ∈ [-1, 1]; the paper reports ρ by
convention in [0, 1] via absolute value only where a sign flip is not
meaningful (see :func:`sign_flip_rate` for the complementary statistic).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def spearman_rank_stability(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    absolute: bool = False,
) -> float:
    """Spearman ρ between two edge-score vectors.

    Parameters
    ----------
    scores_a, scores_b
        Aligned edge-score vectors of identical length.
    absolute
        If ``True``, return ``|ρ|``.

    Returns
    -------
    float
        Spearman's rank correlation. Returns NaN when either input is
        constant.
    """

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have identical lengths")
    if np.var(a) == 0 or np.var(b) == 0:
        return float("nan")
    rho, _ = spearmanr(a, b)
    if absolute:
        rho = abs(rho)
    return float(rho)
