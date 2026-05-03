"""Shared null-model library (P5 of the revision plan).

Null models are deliberately kept independent of the scoring engine: a
null takes in an expression matrix plus labels and returns a *permuted*
version that the scorer can re-score. This contract ensures every axis
uses the same null implementations so permutation tests are directly
comparable across axes.

Addressing reviewers:

* **R3** ("what are null families and what does 'cleared' mean"):
  :class:`NullFamily` carries its own ``clear_threshold`` so the term is
  defined in code, not in prose.
* **R3** ("constrained shuffles may be too strict"): WP-14 adds the
  degree-preserving rewiring null as a third family.
"""

from .base import NullFamily, NullResult, apply_null
from .global_shuffle import GlobalShuffleNull
from .within_coarse import WithinCoarseShuffleNull
from .gene_shuffle import GeneShuffleNull
from .rank_permutation import RankPermutationNull
from .degree_preserving import DegreePreservingNull


__all__ = [
    "NullFamily",
    "NullResult",
    "apply_null",
    "GlobalShuffleNull",
    "WithinCoarseShuffleNull",
    "GeneShuffleNull",
    "RankPermutationNull",
    "DegreePreservingNull",
]
