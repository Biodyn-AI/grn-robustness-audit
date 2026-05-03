"""Unit tests for fragility.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from fragility.metrics import (
    drift_norm,
    mean_absolute_rank_shift,
    null_separation_auc,
    rss,
    rss_components,
    rss_with_weights,
    sign_flip_rate,
    spearman_rank_stability,
    tail_gap,
    topk_jaccard,
    topk_overlap,
)


def test_rss_identical_is_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    assert rss(a, a, k=100) == 0.0


def test_rss_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    c = rss_with_weights(a, b, k=50)
    for x in (c.overlap_loss, c.jaccard_loss, c.drift_norm, c.composite):
        assert 0.0 <= x <= 1.0


def test_rss_components_match_topk():
    rng = np.random.default_rng(2)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    ol, jl, _ = rss_components(a, b, k=50)
    assert pytest.approx(topk_overlap(a, b, k=50), abs=1e-9) == 1.0 - ol
    assert pytest.approx(topk_jaccard(a, b, k=50), abs=1e-9) == 1.0 - jl


def test_rss_weight_validation():
    rng = np.random.default_rng(3)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    with pytest.raises(ValueError):
        rss_with_weights(a, b, k=10, weights=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        rss_with_weights(a, b, k=10, weights=(-0.1, 0.6, 0.5))


def test_drift_norm_bounded():
    a = np.arange(100, dtype=float)
    # Reverse the order -> max possible drift within top-k
    assert drift_norm(a, -a, k=20) == 1.0
    # Identical -> zero drift
    assert drift_norm(a, a, k=20) == 0.0


def test_spearman_stability_bounds():
    rng = np.random.default_rng(4)
    a = rng.normal(size=50)
    assert spearman_rank_stability(a, a) == pytest.approx(1.0)
    assert spearman_rank_stability(a, -a) == pytest.approx(-1.0)


def test_null_separation_auc():
    obs = np.array([0.8, 0.9, 0.95])
    null = np.array([0.1, 0.2, 0.3])
    assert null_separation_auc(obs, null) == 1.0


def test_tail_gap_sign():
    obs = np.array([1.0, 2.0, 3.0, 10.0])
    null = np.array([-1.0, 0.0, 1.0, 2.0])
    # Observed extreme > null extreme -> positive gap
    assert tail_gap(obs, null, tail_fraction=0.25) > 0


def test_sign_flip_rate_boundary():
    signs_a = np.array([1, -1, 1, 1, -1])
    signs_b = np.array([1, -1, -1, -1, 1])
    # Edges 3 and 4 flip (+/- -> -/+) and edge 5 flips (-/+ -> +/-)
    assert sign_flip_rate(signs_a, signs_b) == 3 / 5


def test_mean_abs_rank_shift_zero():
    r = np.arange(100)
    assert mean_absolute_rank_shift(r, r) == 0.0
