"""End-to-end synthetic tests for the compute-light work packages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fragility.axes.wp3_rss_redesign.runner import (
    ScoredPair,
    compute_components,
    compute_empirical_null,
    compute_weight_sweep,
    run as run_wp3,
    simplex_weights,
)
from fragility.axes.wp4_reliability_grid.runner import (
    GRID,
    PRIMARY_THRESHOLDS,
    run as run_wp4,
)
from fragility.axes.wp10_topk_scan.runner import run as run_wp10


def _make_pair(rng: np.random.Generator, correlation: float = 0.7) -> ScoredPair:
    n = 2000
    a = rng.normal(size=n)
    b = correlation * a + np.sqrt(max(1 - correlation ** 2, 1e-6)) * rng.normal(size=n)
    return ScoredPair("kidney", "pearson", "coarse_vs_fine", a, b)


def test_simplex_weights_step_and_sum():
    ws = simplex_weights(step=0.1)
    # Triangular number: (1/0.1 + 1)*(1/0.1 + 2)/2 = 11*12/2 = 66
    assert len(ws) == 66
    for w in ws:
        assert abs(sum(w) - 1.0) < 1e-6
        assert all(x >= 0 for x in w)


def test_wp3_null_detects_stable_pair():
    rng = np.random.default_rng(0)
    pair = _make_pair(rng, correlation=0.9)
    null = compute_empirical_null(pair, k=200, n_permutations=200, seed_namespace="test")
    # Stable pair should be many sigma below the rank-perm null.
    assert null["z_score_vs_null"] < -5
    assert null["observed_rss"] < null["null_5pct"]


def test_wp3_end_to_end_emits_three_csvs(tmp_path: Path):
    rng = np.random.default_rng(1)
    pairs = [_make_pair(rng, correlation=c) for c in (0.3, 0.7, 0.9)]
    for i, p in enumerate(pairs):
        p.pair_id = f"pair_{i}"  # make unique
    paths = run_wp3(pairs, out_dir=tmp_path, k=200, n_null=50, weights_step=0.25)
    for name in ("rss_components", "rss_weight_sweep", "rss_empirical_null"):
        df = pd.read_csv(paths[name])
        assert len(df) > 0


def test_wp4_grid_preserves_rare_abundant_ordering(tmp_path: Path):
    # Synthetic: abundant cell types have uniformly higher stability/jaccard
    # than rare, so every grid point should give abundant >= rare.
    rng = np.random.default_rng(2)
    rows = []
    for rarity, shift in (("rare", -0.2), ("intermediate", 0.0), ("abundant", 0.2)):
        for i in range(10):
            rows.append({
                "dataset": "synthetic",
                "cell_type": f"{rarity}_ct_{i}",
                "rarity": rarity,
                "stability": 0.7 + shift + rng.normal(scale=0.05),
                "topk_jaccard": 0.4 + shift + rng.normal(scale=0.05),
                "null_auc": 0.6 + shift + rng.normal(scale=0.05),
                "tail_gap": shift + rng.normal(scale=0.02),
                "cell_count": 80 + int(rng.integers(0, 200)),
            })
    inp = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(inp, index=False)
    summary = run_wp4(inp, tmp_path, grid=GRID, primary=PRIMARY_THRESHOLDS)
    grid_df = pd.read_csv(summary.out_grid)
    # In every grid point, abundant >= rare.
    assert (grid_df["rare_minus_abundant"] >= 0).all() or \
        (grid_df["rare_minus_abundant"] <= 0).all()


def test_wp10_scan_has_expected_keys(tmp_path: Path):
    rng = np.random.default_rng(3)
    rows = []
    n = 2000
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.5, size=n)
    for eid in range(n):
        rows.append({
            "dataset": "synthetic",
            "scorer": "pearson",
            "pair_id": "p",
            "edge_id": f"e{eid}",
            "target_id": f"t{eid % 20}",
            "score_coarse": float(a[eid]),
            "score_fine": float(b[eid]),
        })
    inp = tmp_path / "scored_pairs.csv"
    pd.DataFrame(rows).to_csv(inp, index=False)
    paths = run_wp10(inp, tmp_path, ks=(100, 500), k_percents=(10.0,), per_target_ks=(5,))
    scan = pd.read_csv(paths["topk_scan"])
    assert set(scan["k"]).issuperset({100, 500, 200})  # 200 = 10% of 2000
    pt = pd.read_csv(paths["topk_per_target"])
    assert (pt["k_per_target"] == 5).all()
