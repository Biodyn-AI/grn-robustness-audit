"""Metric definitions (formal, self-contained).

This module addresses two reviewer demands simultaneously:

* **R3** (metric definitions): every metric the paper cites lives here
  with a one-paragraph docstring and a formula. Methods section cites
  this module.
* **R1a(iii–v), R4(4)** (RSS redesign): the new :func:`rss` uses a
  bounded ``drift_norm`` term, reports components separately, and exposes
  weights as arguments so WP-3's weight-sensitivity sweep can call it in
  a loop.
"""

from .stability import spearman_rank_stability
from .topk import (
    topk_jaccard,
    topk_overlap,
    topk_intersection,
    topk_per_target_overlap,
)
from .null_separation import null_separation_auc, tail_gap
from .rank_shift import mean_absolute_rank_shift, sign_flip_rate
from .rss import (
    rss,
    rss_components,
    rss_with_weights,
    drift_norm,
)

__all__ = [
    "spearman_rank_stability",
    "topk_jaccard",
    "topk_overlap",
    "topk_intersection",
    "topk_per_target_overlap",
    "null_separation_auc",
    "tail_gap",
    "mean_absolute_rank_shift",
    "sign_flip_rate",
    "rss",
    "rss_components",
    "rss_with_weights",
    "drift_norm",
]
