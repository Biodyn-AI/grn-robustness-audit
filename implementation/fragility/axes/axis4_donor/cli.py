"""CLI wrapper for Axis 4."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...utils import (
    dump_config,
    load_config,
    seed_everything,
    write_provenance,
)
from .runner import Axis4Config, Axis4DatasetSpec, run as _run_runner


DEFAULT_CONFIG = {
    "datasets": [
        {
            "name": "immune",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad",
            "donor_col": "donor_id",
            "max_donors": 12,
            "cells_per_donor": 300,
        },
        {
            "name": "lung",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad",
            "donor_col": "donor_id",
            "max_donors": 4,
            "cells_per_donor": 800,
        },
    ],
    "config": {
        "panel": "hematopoiesis_76x108",
        "scorer": "pearson",
        "train_donor_counts": [2, 4, 6, 8, 10],
        "holdout_donors": 2,
        "n_splits_per_count": 20,
        "top_ks": [50, 100, 250, 500],
        "rel_threshold": 0.90,
        "rel_threshold_conservative": 0.95,
        "seed_namespace": "axis4:default",
    },
    "base_seed": 20260218,
}


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="fragility axis4")
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
        Axis4DatasetSpec(
            name=d["name"],
            path=Path(d["path"]),
            donor_col=d["donor_col"],
            max_donors=int(d.get("max_donors", 12)),
            cells_per_donor=int(d.get("cells_per_donor", 300)),
        )
        for d in dataset_dicts
    ]
    cfg_dict = dict(config["config"])
    cfg_dict["train_donor_counts"] = tuple(cfg_dict["train_donor_counts"])
    cfg_dict["top_ks"] = tuple(cfg_dict["top_ks"])
    cfg = Axis4Config(**cfg_dict)

    paths = _run_runner(specs, out_dir=out_dir, config=cfg)

    write_provenance(
        out_dir=out_dir,
        pipeline="axis4_donor",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[s.path for s in specs],
        extra={"output_csvs": {k: str(v) for k, v in paths.items()}},
    )
    print(f"Axis-4 complete on {len(specs)} datasets.")
    return 0
