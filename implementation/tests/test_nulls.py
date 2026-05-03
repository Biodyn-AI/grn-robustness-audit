"""Unit tests for fragility.nulls."""

from __future__ import annotations

import numpy as np
import pytest

from fragility.nulls import (
    GlobalShuffleNull,
    GeneShuffleNull,
    WithinCoarseShuffleNull,
    RankPermutationNull,
    DegreePreservingNull,
)
from fragility.nulls.base import apply_null, empirical_p_value


def test_global_shuffle_preserves_label_set():
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3])
    X = np.zeros((len(labels), 5))
    null = GlobalShuffleNull()
    out = null.permute(X, labels, rng)
    assert sorted(out.labels.tolist()) == sorted(labels.tolist())


def test_within_coarse_preserves_coarse():
    rng = np.random.default_rng(1)
    coarse = np.array([0, 0, 0, 1, 1, 1])
    fine = np.array([0, 1, 2, 3, 4, 5])
    labels = np.stack([coarse, fine], axis=1)
    X = np.zeros((6, 3))
    null = WithinCoarseShuffleNull()
    out = null.permute(X, labels, rng)
    # coarse column preserved exactly
    assert np.array_equal(out.labels[:, 0], coarse)
    # fine shuffled within coarse groups -> permutation of original set
    assert sorted(out.labels[:, 1].tolist()) == sorted(fine.tolist())


def test_gene_shuffle_changes_structure():
    rng = np.random.default_rng(2)
    X = np.arange(200).reshape(10, 20).astype(float)
    null = GeneShuffleNull()
    out = null.permute(X, None, rng)
    # Column sums preserved; row sums need not be.
    assert np.allclose(out.X.sum(axis=0), X.sum(axis=0))


def test_rank_permutation_is_permutation():
    rng = np.random.default_rng(3)
    r = np.arange(50)
    null = RankPermutationNull()
    out = null.permute(r, None, rng)
    assert sorted(out.X.tolist()) == sorted(r.tolist())


def test_degree_preserving_is_bounded():
    rng = np.random.default_rng(4)
    scores = rng.random(size=(10, 15))
    null = DegreePreservingNull(top_k=30)
    out = null.permute(scores, None, rng)
    assert out.X.shape == scores.shape
    assert (out.X >= 0).all()


def test_apply_null_enforces_permutation_floor():
    rng = np.random.default_rng(5)
    labels = np.arange(10)
    X = np.zeros((10, 3))
    null = GlobalShuffleNull()
    # floor for alpha=0.05 is 19 permutations: fewer should raise.
    with pytest.raises(ValueError):
        apply_null(null, X, labels, n_permutations=5, rng=rng)
    apply_null(null, X, labels, n_permutations=25, rng=rng)


def test_empirical_p_value_bounds():
    null_stats = np.arange(100)
    p = empirical_p_value(1000.0, null_stats, alternative="greater")
    assert p == 1.0 / 101.0
    p = empirical_p_value(-1000.0, null_stats, alternative="greater")
    assert p == 101.0 / 101.0
