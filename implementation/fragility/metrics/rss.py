"""Resolution Sensitivity Score (RSS) — redesigned per WP-3.

Reviewer concerns addressed:

* **R1a(iii)** — "unequal weights": :func:`rss_with_weights` accepts any
  non-negative weight triple summing to 1; WP-3 sweeps the simplex.
* **R1a(iv)** — "overlap and Jaccard are redundant": :func:`rss_components`
  returns the three components separately so the paper can quote both
  (with their empirical correlation) rather than collapse them silently.
* **R1a(v), R4(4)** — "median drift is unbounded": :func:`drift_norm`
  divides the median absolute rank change by ``k``, yielding a bounded
  statistic in ``[0, 1]`` (worst-case rank change within the top-k is
  exactly ``k``).
* **R4(4)** — "no empirical null": :func:`rss_with_weights` is a pure
  function so WP-3 can pass in rank-permutation nulls and build the
  observed-vs-null distribution.

Definitions
-----------
Let ``r_a``, ``r_b`` be the edge ranks from scorer runs A and B over a
shared edge universe of size ``N``. Let ``T_a`` and ``T_b`` be the top-k
index sets.

* ``overlap_loss = 1 - |T_a ∩ T_b| / k``  (∈ [0, 1])
* ``jaccard_loss = 1 - |T_a ∩ T_b| / |T_a ∪ T_b|``  (∈ [0, 1])
* ``drift_norm = min(1, median_{i ∈ T_a ∩ T_b} |r_a_i - r_b_i| / k)``

The composite score is

    RSS = w_o · overlap_loss + w_j · jaccard_loss + w_d · drift_norm

with default weights ``(0.4, 0.3, 0.3)`` preserved from the pre-revision
definition (so numerical outputs remain comparable) and exposed as
arguments so the weight-sensitivity sweep can override them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .topk import topk_overlap, topk_jaccard


@dataclass(frozen=True)
class RSSComponents:
    """Container for RSS components + composite + metadata."""

    overlap_loss: float
    jaccard_loss: float
    drift_norm: float
    composite: float
    weights: Tuple[float, float, float]
    k: int

    def as_dict(self) -> Dict[str, float]:
        w_o, w_j, w_d = self.weights
        return {
            "rss_overlap_loss": self.overlap_loss,
            "rss_jaccard_loss": self.jaccard_loss,
            "rss_drift_norm": self.drift_norm,
            "rss_composite": self.composite,
            "rss_weight_overlap": w_o,
            "rss_weight_jaccard": w_j,
            "rss_weight_drift": w_d,
            "rss_k": self.k,
        }


def _topk_indices(scores: Sequence[float], k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if k >= scores.size:
        return np.argsort(-scores, kind="mergesort")
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part], kind="mergesort")]


def drift_norm(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    k: int,
) -> float:
    """Bounded median-drift in ``[0, 1]``.

    Computes ``min(1, median |r_a_i - r_b_i| / k)`` over the intersection
    of the top-k sets of ``a`` and ``b``. Returns ``1.0`` when the
    intersection is empty (worst possible drift).
    """

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    top_a = _topk_indices(a, k)
    top_b = _topk_indices(b, k)
    common = np.intersect1d(top_a, top_b, assume_unique=True)
    if len(common) == 0:
        return 1.0
    # Ranks: position in the full ranking (0 = best).
    ranks_a = np.argsort(np.argsort(-a, kind="mergesort"), kind="mergesort")
    ranks_b = np.argsort(np.argsort(-b, kind="mergesort"), kind="mergesort")
    diffs = np.abs(ranks_a[common] - ranks_b[common])
    return float(min(1.0, np.median(diffs) / k))


def rss_components(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    k: int = 1000,
) -> Tuple[float, float, float]:
    """Return (overlap_loss, jaccard_loss, drift_norm) without weighting."""

    overlap = topk_overlap(scores_a, scores_b, k)
    jaccard = topk_jaccard(scores_a, scores_b, k)
    return 1.0 - overlap, 1.0 - jaccard, drift_norm(scores_a, scores_b, k)


def rss_with_weights(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    k: int = 1000,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> RSSComponents:
    """Compute RSS with explicit weights. Weights must be non-negative."""

    w_o, w_j, w_d = map(float, weights)
    if min(w_o, w_j, w_d) < 0 or not np.isclose(w_o + w_j + w_d, 1.0, atol=1e-6):
        raise ValueError(
            "weights must be non-negative and sum to 1.0; got "
            f"({w_o}, {w_j}, {w_d})"
        )
    overlap_loss, jaccard_loss, drift = rss_components(scores_a, scores_b, k)
    composite = w_o * overlap_loss + w_j * jaccard_loss + w_d * drift
    return RSSComponents(
        overlap_loss=overlap_loss,
        jaccard_loss=jaccard_loss,
        drift_norm=drift,
        composite=composite,
        weights=(w_o, w_j, w_d),
        k=int(k),
    )


def rss(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    k: int = 1000,
) -> float:
    """Default-weight RSS scalar (preserves pre-revision numerical output)."""

    return rss_with_weights(scores_a, scores_b, k=k).composite
