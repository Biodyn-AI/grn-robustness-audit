"""Core computation for WP-3."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...metrics import (
    rss_with_weights,
    rss_components,
    drift_norm,
    topk_overlap,
    topk_jaccard,
    spearman_rank_stability,
)
from ...utils import rng_for


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class ScoredPair:
    """One (dataset, scorer, resolution-pair) scored edge list pair."""

    dataset: str
    scorer: str
    pair_id: str                     # e.g. "kidney_coarse_vs_fine"
    scores_coarse: np.ndarray
    scores_fine: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Weight simplex grid (step 0.1)
# ---------------------------------------------------------------------------


def simplex_weights(step: float = 0.1) -> List[Tuple[float, float, float]]:
    """Enumerate non-negative weight triples (w_o, w_j, w_d) summing to 1."""

    out: List[Tuple[float, float, float]] = []
    n = int(round(1.0 / step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            w = (a / n, b / n, c / n)
            out.append(w)
    # Deduplicate by 6-digit float representation (avoids near-equal doubles).
    seen: set = set()
    dedup: List[Tuple[float, float, float]] = []
    for w in out:
        key = tuple(round(x, 6) for x in w)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(tuple(float(x) for x in w))
    return dedup


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_components(
    pair: ScoredPair,
    k: int,
) -> Dict[str, float]:
    """Return the three RSS components + Spearman + absolute scores."""

    overlap_loss, jaccard_loss, drift = rss_components(
        pair.scores_coarse, pair.scores_fine, k=k
    )
    return {
        "dataset": pair.dataset,
        "scorer": pair.scorer,
        "pair_id": pair.pair_id,
        "k": int(k),
        "overlap": 1.0 - overlap_loss,
        "jaccard": 1.0 - jaccard_loss,
        "overlap_loss": overlap_loss,
        "jaccard_loss": jaccard_loss,
        "drift_norm": drift,
        "spearman": spearman_rank_stability(pair.scores_coarse, pair.scores_fine),
    }


def compute_weight_sweep(
    pair: ScoredPair,
    k: int,
    weights: Sequence[Tuple[float, float, float]],
) -> List[Dict[str, float]]:
    """Return one row per weight triple with the composite RSS."""

    rows: List[Dict[str, float]] = []
    for w in weights:
        comp = rss_with_weights(pair.scores_coarse, pair.scores_fine, k=k, weights=w)
        rows.append({
            "dataset": pair.dataset,
            "scorer": pair.scorer,
            "pair_id": pair.pair_id,
            "k": int(k),
            "weight_overlap": float(w[0]),
            "weight_jaccard": float(w[1]),
            "weight_drift": float(w[2]),
            "rss_composite": float(comp.composite),
        })
    return rows


def compute_empirical_null(
    pair: ScoredPair,
    k: int,
    n_permutations: int,
    seed_namespace: str,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> Dict[str, float]:
    """Build an empirical null RSS distribution by permuting scores.

    Null model: for each permutation, replace ``scores_fine`` with a
    uniform random permutation of the same values. This preserves the
    marginal score distribution but breaks the coarse/fine correspondence.
    """

    rng = rng_for(f"wp3:null:{pair.dataset}:{pair.scorer}:{pair.pair_id}:{seed_namespace}")
    null_stats = np.empty(n_permutations, dtype=float)
    a = np.asarray(pair.scores_coarse, dtype=float)
    b = np.asarray(pair.scores_fine, dtype=float)
    for i in range(n_permutations):
        b_perm = rng.permutation(b)
        comp = rss_with_weights(a, b_perm, k=k, weights=weights)
        null_stats[i] = comp.composite
    observed = rss_with_weights(a, b, k=k, weights=weights).composite
    null_mean = float(null_stats.mean())
    null_std = float(null_stats.std(ddof=1) or 1e-9)
    z = (observed - null_mean) / null_std
    # Empirical two-sided p with +1 smoothing.
    count = int(np.sum(np.abs(null_stats - null_mean) >= abs(observed - null_mean)))
    p_value = (count + 1.0) / (n_permutations + 1.0)
    return {
        "dataset": pair.dataset,
        "scorer": pair.scorer,
        "pair_id": pair.pair_id,
        "k": int(k),
        "weights": weights,
        "n_permutations": int(n_permutations),
        "observed_rss": float(observed),
        "null_mean_rss": null_mean,
        "null_std_rss": null_std,
        "null_5pct": float(np.quantile(null_stats, 0.05)),
        "null_95pct": float(np.quantile(null_stats, 0.95)),
        "z_score_vs_null": float(z),
        "empirical_p": float(p_value),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run(
    pairs: Iterable[ScoredPair],
    out_dir: Path,
    k: int = 1000,
    n_null: int = 1000,
    weights_step: float = 0.1,
    default_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> Dict[str, Path]:
    """Run WP-3 over an iterable of scored pairs.

    Emits three CSVs:

    * ``rss_components.csv``: one row per (dataset, scorer, pair_id, k).
    * ``rss_weight_sweep.csv``: one row per weight triple × pair.
    * ``rss_empirical_null.csv``: one row per pair × (k) with z-score
      vs the rank-permutation null.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_grid = simplex_weights(step=weights_step)

    component_rows: List[dict] = []
    sweep_rows: List[dict] = []
    null_rows: List[dict] = []

    for pair in pairs:
        component_rows.append(compute_components(pair, k=k))
        sweep_rows.extend(compute_weight_sweep(pair, k=k, weights=weights_grid))
        null_rows.append(compute_empirical_null(
            pair, k=k, n_permutations=n_null,
            seed_namespace="default", weights=default_weights,
        ))

    comp_df = pd.DataFrame(component_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    null_df = pd.DataFrame(null_rows)
    # Serialize tuple-valued "weights" column as a string for CSV.
    if "weights" in null_df.columns:
        null_df["weights"] = null_df["weights"].apply(
            lambda t: ",".join(f"{x:.3f}" for x in t)
        )

    comp_path = out_dir / "rss_components.csv"
    sweep_path = out_dir / "rss_weight_sweep.csv"
    null_path = out_dir / "rss_empirical_null.csv"
    comp_df.to_csv(comp_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    null_df.to_csv(null_path, index=False)
    return {
        "rss_components": comp_path,
        "rss_weight_sweep": sweep_path,
        "rss_empirical_null": null_path,
    }
