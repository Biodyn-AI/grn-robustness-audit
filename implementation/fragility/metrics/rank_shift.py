"""Rank-shift metrics (Axis 5 integration sensitivity)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_absolute_rank_shift(
    ranks_a: Sequence[int],
    ranks_b: Sequence[int],
) -> float:
    """Mean absolute difference between two aligned rank vectors."""

    a = np.asarray(ranks_a)
    b = np.asarray(ranks_b)
    if len(a) != len(b):
        raise ValueError("ranks_a and ranks_b must align")
    return float(np.mean(np.abs(a - b)))


def sign_flip_rate(
    signs_a: Sequence[float],
    signs_b: Sequence[float],
) -> float:
    """Fraction of edges whose sign of score flips between ``a`` and ``b``."""

    a = np.asarray(signs_a)
    b = np.asarray(signs_b)
    if len(a) != len(b):
        raise ValueError("signs_a and signs_b must align")
    flipped = (a * b) < 0
    return float(flipped.mean())
