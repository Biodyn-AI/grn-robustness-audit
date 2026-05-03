"""Top-k overlap / Jaccard metrics.

Formal definitions:

* ``topk_overlap``:  |A_k ∩ B_k| / k
* ``topk_jaccard``:  |A_k ∩ B_k| / |A_k ∪ B_k|

where A_k, B_k are the top-k edge sets of two rankings. If the two lists
share no duplicates (a single ranking over a shared edge universe), the
two quantities are related by ``overlap = 2·jaccard / (1 + jaccard)``,
which is why R1a(iv) asks whether both need to appear in RSS — see WP-3
(:mod:`fragility.metrics.rss`) for the empirical correlation check.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _topk_indices(scores: Sequence[float], k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if k >= scores.size:
        return np.argsort(-scores, kind="mergesort")
    # argpartition gets top-k in O(n); then sort only the top-k slice.
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part], kind="mergesort")]


def topk_intersection(
    scores_a: Sequence[float], scores_b: Sequence[float], k: int
) -> int:
    a = set(_topk_indices(scores_a, k).tolist())
    b = set(_topk_indices(scores_b, k).tolist())
    return len(a & b)


def topk_overlap(
    scores_a: Sequence[float], scores_b: Sequence[float], k: int
) -> float:
    """Return |A_k ∩ B_k| / k."""

    inter = topk_intersection(scores_a, scores_b, k)
    return inter / k


def topk_jaccard(
    scores_a: Sequence[float], scores_b: Sequence[float], k: int
) -> float:
    """Return |A_k ∩ B_k| / |A_k ∪ B_k|."""

    a = set(_topk_indices(scores_a, k).tolist())
    b = set(_topk_indices(scores_b, k).tolist())
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def topk_per_target_overlap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    target_ids: Sequence[int],
    k_per_target: int,
) -> float:
    """Per-target top-k overlap (R3 request).

    Groups edges by ``target_ids`` and, for each target, takes the top
    ``k_per_target`` edges by score. Returns the mean Jaccard of these
    per-target sets across all targets.
    """

    target_ids = np.asarray(target_ids)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    jaccards = []
    for t in np.unique(target_ids):
        idx = np.where(target_ids == t)[0]
        if len(idx) == 0:
            continue
        k = min(k_per_target, len(idx))
        local_a = a[idx]
        local_b = b[idx]
        top_a = set(idx[_topk_indices(local_a, k)].tolist())
        top_b = set(idx[_topk_indices(local_b, k)].tolist())
        if not (top_a or top_b):
            continue
        jaccards.append(len(top_a & top_b) / len(top_a | top_b))
    return float(np.mean(jaccards)) if jaccards else float("nan")


def topk_scan(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    ks: Sequence[int],
) -> Dict[str, Dict[int, float]]:
    """Return overlap and Jaccard at every k in ``ks`` (WP-10 helper)."""

    out: Dict[str, Dict[int, float]] = {"overlap": {}, "jaccard": {}}
    for k in ks:
        out["overlap"][k] = topk_overlap(scores_a, scores_b, k)
        out["jaccard"][k] = topk_jaccard(scores_a, scores_b, k)
    return out
