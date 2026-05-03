"""Edge-scoring engine (P4 of the revision plan).

Every scorer implements the ``EdgeScorer`` contract below. Axes iterate
over scorers without knowing which is active. This is the infrastructure
layer that resolves **R1a(i), R2(2), R3(title), R4(1), R4(7)** by
enabling the same pipeline to run with Pearson, GRNBoost2, GENIE3,
mutual information, and scGPT attention, on the same datasets and
panels.
"""

from .base import EdgeScorer, EdgeScores, list_scorers, get_scorer, register_scorer
from .pearson import PearsonScorer
from .mutual_information import MutualInfoScorer

# Heavy scorers are imported lazily so missing optional dependencies
# (arboreto, torch, scGPT) do not break the light-weight Pearson path.
try:  # pragma: no cover - optional dep
    from .tree import GRNBoost2Scorer, GENIE3Scorer  # noqa: F401
except ImportError:
    pass

try:  # pragma: no cover - optional dep
    from .scgpt_attention import ScGPTAttentionScorer  # noqa: F401
except ImportError:
    pass

try:  # working scGPT attention scorer (bypasses torchtext)
    from .scgpt_scorer import ScGPTAttentionRealScorer  # noqa: F401
except Exception:
    pass


__all__ = [
    "EdgeScorer",
    "EdgeScores",
    "PearsonScorer",
    "MutualInfoScorer",
    "list_scorers",
    "get_scorer",
    "register_scorer",
]
