"""Unit tests for fragility.scorers."""

from __future__ import annotations

import numpy as np
import pytest

from fragility.scorers import get_scorer, list_scorers


def _planted_matrix(rng: np.random.Generator) -> tuple:
    n_cells, n_genes = 200, 40
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    # Plant strong coexpression
    X[:, 10] = 0.9 * X[:, 2] + 0.1 * rng.normal(size=n_cells)
    X[:, 20] = 0.95 * X[:, 3] + 0.05 * rng.normal(size=n_cells)
    gene_names = np.array([f"G{i:02d}" for i in range(n_genes)])
    tf_idx = np.array([2, 3, 5])
    target_idx = np.array([10, 20, 30])
    return X, gene_names, tf_idx, target_idx


def test_pearson_recovers_planted_edges():
    rng = np.random.default_rng(0)
    X, genes, tf, tg = _planted_matrix(rng)
    scorer = get_scorer("pearson")
    out = scorer.score(X, tf, tg, genes)
    top = np.argsort(out.ranks)[:2]
    edges_top = set(zip(np.asarray(out.tf_names)[top], np.asarray(out.target_names)[top]))
    assert edges_top == {("G02", "G10"), ("G03", "G20")}
    assert out.signs is not None


def test_mutual_info_recovers_planted_edges():
    rng = np.random.default_rng(0)
    X, genes, tf, tg = _planted_matrix(rng)
    scorer = get_scorer("mutual_info", n_neighbors=3)
    out = scorer.score(X, tf, tg, genes)
    top = np.argsort(out.ranks)[:2]
    edges_top = set(zip(np.asarray(out.tf_names)[top], np.asarray(out.target_names)[top]))
    assert edges_top == {("G02", "G10"), ("G03", "G20")}


def test_scorer_registry_contains_at_least_pearson_mi():
    names = set(list_scorers())
    assert {"pearson", "mutual_info"}.issubset(names)


def test_pearson_shape_and_meta():
    rng = np.random.default_rng(0)
    X, genes, tf, tg = _planted_matrix(rng)
    out = get_scorer("pearson").score(X, tf, tg, genes)
    assert len(out.scores) == len(tf) * len(tg)
    assert out.meta["n_tfs"] == len(tf)
    assert out.meta["n_targets"] == len(tg)
    assert out.meta["n_cells"] == X.shape[0]


def test_scorer_unknown_raises():
    with pytest.raises(KeyError):
        get_scorer("made_up_scorer")
