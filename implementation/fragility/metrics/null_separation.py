"""Null-separation AUC and tail-gap (Axis 3 reliability gate).

Formal definitions:

* ``null_separation_auc``: ROC AUC where "positives" are the observed
  edge scores and "negatives" are scores from null permutations. AUC = 1
  means every observed score exceeds every null score; AUC = 0.5 is
  random.
* ``tail_gap``: the difference between the k-th order statistic of the
  observed scores and the k-th order statistic of the pooled null
  scores; measures whether the extreme tail of the observed distribution
  clears the null tail.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def null_separation_auc(
    observed: Sequence[float],
    null: Sequence[float],
) -> float:
    """Return the AUROC separating ``observed`` from ``null``."""

    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    y = np.concatenate([np.ones_like(observed), np.zeros_like(null)])
    s = np.concatenate([observed, null])
    if np.var(s) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tail_gap(
    observed: Sequence[float],
    null: Sequence[float],
    tail_fraction: float = 0.01,
) -> float:
    """Gap between the top ``tail_fraction`` quantile of observed and null."""

    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    q = 1.0 - tail_fraction
    return float(np.quantile(observed, q) - np.quantile(null, q))
