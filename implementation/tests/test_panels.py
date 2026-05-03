"""Unit tests for fragility.panels."""

from __future__ import annotations

import pytest

from fragility.panels import PANEL_REGISTRY, load_panel, list_panels


def test_registry_contains_expected_names():
    expected = {
        "primary",
        "dorothea_ab",
        "dorothea_abcd",
        "trrust",
        "hematopoiesis_76x108",
        "shared_36",
    }
    assert expected.issubset(set(list_panels()))


def test_hematopoiesis_panel_exact_count():
    df = load_panel("hematopoiesis_76x108")
    assert df["source"].nunique() == 76
    assert df["target"].nunique() == 108
    assert len(df) == 76 * 108


def test_primary_intersection_nonempty():
    df = load_panel("primary")
    assert len(df) > 100
    assert set(df.columns).issuperset({"source", "target", "panel"})


def test_gene_universe_restriction():
    uni = {"STAT1", "STAT3", "CD3D", "CD4", "MYC"}
    df = load_panel("hematopoiesis_76x108", gene_universe=uni)
    assert set(df["source"].unique()).issubset({g.upper() for g in uni})
    assert set(df["target"].unique()).issubset({g.upper() for g in uni})


def test_unknown_panel_raises():
    with pytest.raises(KeyError):
        load_panel("does_not_exist")
