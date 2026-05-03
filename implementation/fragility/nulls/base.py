"""Base class and helpers for null-model generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np


@dataclass
class NullResult:
    """Result of a single null permutation.

    ``X`` and ``labels`` may be views onto the original arrays; callers
    must not mutate them.
    """

    X: np.ndarray
    labels: Optional[np.ndarray] = None
    meta: Dict[str, object] = field(default_factory=dict)


class NullFamily(ABC):
    """Contract for null-permutation generators."""

    #: short identifier used in configs and output CSVs
    name: str = ""
    #: human-readable label for figures
    label: str = ""
    #: empirical p-value threshold below which the null is "cleared"
    clear_threshold: float = 0.05

    def __init__(self, **kwargs):
        self.options: Dict[str, object] = dict(kwargs)

    def permutations(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        n: int,
        rng: np.random.Generator,
    ) -> List[NullResult]:
        return [self.permute(X, labels, rng) for _ in range(n)]

    @abstractmethod
    def permute(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> NullResult:
        """Return one null permutation of the inputs."""


def empirical_p_value(
    observed: float,
    null_stats: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    """Empirical p-value with a +1 smoothing correction.

    ``alternative``:
        * ``"two-sided"`` (default): how extreme on either side
        * ``"greater"``: null >= observed
        * ``"less"``: null <= observed
    """

    null_stats = np.asarray(null_stats)
    n = len(null_stats)
    if alternative == "greater":
        count = int(np.sum(null_stats >= observed))
    elif alternative == "less":
        count = int(np.sum(null_stats <= observed))
    elif alternative == "two-sided":
        count = int(
            np.sum(np.abs(null_stats - np.mean(null_stats))
                   >= np.abs(observed - np.mean(null_stats)))
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown alternative: {alternative}")
    return (count + 1.0) / (n + 1.0)


def apply_null(
    family: NullFamily,
    X: np.ndarray,
    labels: Optional[np.ndarray],
    n_permutations: int,
    rng: np.random.Generator,
) -> List[NullResult]:
    """Run ``family`` for ``n_permutations`` iterations."""

    if n_permutations < int(np.ceil(1 / family.clear_threshold)) - 1:
        raise ValueError(
            f"n_permutations={n_permutations} too low for clear_threshold="
            f"{family.clear_threshold} (floor of empirical p would exceed "
            f"threshold)."
        )
    return family.permutations(X, labels, n_permutations, rng)


#: registry used by YAML configs
_REGISTRY: Dict[str, type] = {}


def register_null(cls: type) -> type:
    if not getattr(cls, "name", None):  # pragma: no cover
        raise ValueError(f"{cls.__name__} must define .name")
    _REGISTRY[cls.name] = cls
    return cls


def get_null(name: str, **kwargs) -> NullFamily:
    if name not in _REGISTRY:
        raise KeyError(f"unknown null '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_nulls() -> list:
    return sorted(_REGISTRY)
