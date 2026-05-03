"""CLI wrapper for Axis 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...utils import (
    dump_config,
    load_config,
    seed_everything,
    write_provenance,
)
from ..axis2_resolution.cli import DEFAULT_CONFIG as AXIS2_DEFAULTS
from ..axis2_resolution.runner import Axis2DatasetSpec
from .runner import Axis3Config, run as _run_runner


DEFAULT_CONFIG = {
    "datasets": AXIS2_DEFAULTS["datasets"],
    "config": {
        "panel": "primary",
        "scorer": "pearson",
        "top_k": 100,
        "n_bootstraps": 8,
        "subsample_fraction": 0.8,
        "min_cells": 40,
        "rarity_quartile": 0.25,
        "n_null_gene_shuffles": 20,
        "n_null_cell_perms": 20,
        "seed_namespace": "axis3:default",
    },
    "base_seed": 20260218,
}


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="fragility axis3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args(argv)

    config = load_config(args.config, default=DEFAULT_CONFIG)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.resolved.yaml")

    derived = seed_everything(
        base_seed=int(config["base_seed"]),
        components=(config["config"]["seed_namespace"],),
    )

    dataset_dicts = config["datasets"]
    if args.only:
        allowed = set(args.only)
        dataset_dicts = [d for d in dataset_dicts if d["name"] in allowed]

    specs = [
        Axis2DatasetSpec(
            name=d["name"],
            path=Path(d["path"]),
            coarse_key=d["coarse_key"],
            fine_key=d["fine_key"],
        )
        for d in dataset_dicts
    ]
    cfg = Axis3Config(**dict(config["config"]))
    paths = _run_runner(specs, out_dir=out_dir, config=cfg)

    write_provenance(
        out_dir=out_dir,
        pipeline="axis3_rare",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[s.path for s in specs],
        extra={"output_csvs": {k: str(v) for k, v in paths.items()}},
    )
    print(f"Axis-3 complete on {len(specs)} datasets.")
    return 0
