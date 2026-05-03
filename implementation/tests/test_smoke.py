"""Smoke test that packages import cleanly and configs parse."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_import_fragility():
    import fragility  # noqa: F401
    from fragility import preprocessing, panels, scorers, nulls, metrics, utils, data
    assert hasattr(fragility, "__version__")


def test_config_loading(tmp_path: Path):
    from fragility.utils import load_config, dump_config

    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.safe_dump({"nested": {"y": 99}, "new": 7}))
    merged = load_config(override_path, default=base)
    assert merged["a"] == 1
    assert merged["nested"]["x"] == 10
    assert merged["nested"]["y"] == 99
    assert merged["new"] == 7

    dump_config(merged, tmp_path / "resolved.yaml")
    assert (tmp_path / "resolved.yaml").exists()


def test_seeds_deterministic():
    from fragility.utils import rng_for

    g1 = rng_for("component_a")
    g2 = rng_for("component_a")
    import numpy as np

    assert np.array_equal(g1.integers(0, 10**6, size=5), g2.integers(0, 10**6, size=5))
