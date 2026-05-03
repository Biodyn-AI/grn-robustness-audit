"""Deterministic seeding helpers.

Uses a single canonical seed and derives sub-seeds for per-component RNGs,
so that logging one project-wide seed in provenance is sufficient to
reproduce every random draw.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Iterable, Optional

import numpy as np


CANONICAL_SEED = 20260218  # original subproject-38 seed; kept for cross-compat


def _derive_subseed(name: str, base_seed: int) -> int:
    """Stable name-based sub-seed derivation.

    Same ``(name, base_seed)`` -> same integer, independent of call order.
    """

    h = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    # keep 32 low-order bits so it fits numpy legacy RandomState as well
    return int.from_bytes(h[:4], "big")


def seed_everything(base_seed: Optional[int] = None, components: Iterable[str] = ()) -> dict:
    """Seed Python ``random``, ``numpy``, and (if importable) ``torch``.

    Returns a dict of ``{name: derived_seed}`` for every name in ``components``
    plus the special key ``"__base__"`` with the base seed itself. The caller
    can later use these to spawn isolated ``np.random.Generator``s.
    """

    if base_seed is None:
        base_seed = CANONICAL_SEED

    random.seed(base_seed)
    np.random.seed(base_seed)
    try:  # optional torch
        import torch  # type: ignore

        torch.manual_seed(base_seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.manual_seed_all(base_seed)
    except Exception:
        pass

    os.environ.setdefault("PYTHONHASHSEED", str(base_seed))

    derived = {"__base__": base_seed}
    for name in components:
        derived[name] = _derive_subseed(name, base_seed)
    return derived


def rng_for(name: str, base_seed: int = CANONICAL_SEED) -> np.random.Generator:
    """Return a deterministic ``np.random.Generator`` for the named component."""

    return np.random.default_rng(_derive_subseed(name, base_seed))
