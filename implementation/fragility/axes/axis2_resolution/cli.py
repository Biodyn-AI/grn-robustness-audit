"""CLI wrapper for Axis 2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from ...utils import (
    dump_config,
    load_config,
    seed_everything,
    write_provenance,
)
from .runner import Axis2Config, Axis2DatasetSpec, run as _run_runner


DEFAULT_CONFIG = {
    "datasets": [
        {
            "name": "kidney",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/kidney/processed.h5ad",
            "coarse_key": "broad_cell_class",
            "fine_key": "scvi_leiden_res05_tissue",
        },
        {
            "name": "lung",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad",
            "coarse_key": "broad_cell_class",
            "fine_key": "scvi_leiden_res05_tissue",
        },
        {
            "name": "immune",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad",
            "coarse_key": "broad_cell_class",
            "fine_key": "scvi_leiden_res05_compartment",
        },
        {
            "name": "immune_invariant",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/immune/processed.h5ad",
            "coarse_key": "broad_cell_class",
            "fine_key": "scvi_leiden_res05_compartment",
        },
        {
            "name": "external_lung",
            "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/external_lung/processed.h5ad",
            "coarse_key": "compartment",
            "fine_key": "cell_type",
        },
    ],
    "config": {
        "n_top_genes": 600,
        "global_weight": 0.6,
        "top_k": 1000,
        "min_cells_group": 40,
        "min_detect_frac": 0.05,
        "n_null_permutations": 49,
        "rss_weights": [0.4, 0.3, 0.3],
        "run_triple_null": False,
        "seed_namespace": "axis2:default",
    },
    "base_seed": 20260218,
}


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="fragility axis2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--only",
        nargs="+",
        help="Restrict to a subset of dataset names (e.g. kidney external_lung)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, default=DEFAULT_CONFIG)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.resolved.yaml")

    derived = seed_everything(
        base_seed=int(config["base_seed"]),
        components=(config["config"].get("seed_namespace", "axis2:default"),),
    )

    dataset_dicts = config["datasets"]
    if args.only:
        allowed = set(args.only)
        dataset_dicts = [d for d in dataset_dicts if d["name"] in allowed]
        if not dataset_dicts:
            raise ValueError(f"no datasets match --only={args.only}")

    specs = [
        Axis2DatasetSpec(
            name=d["name"],
            path=Path(d["path"]),
            coarse_key=d["coarse_key"],
            fine_key=d["fine_key"],
        )
        for d in dataset_dicts
    ]

    cfg_dict = dict(config["config"])
    cfg_dict["rss_weights"] = tuple(cfg_dict.get("rss_weights", (0.4, 0.3, 0.3)))
    cfg = Axis2Config(**cfg_dict)

    paths = _run_runner(specs, out_dir=out_dir, config=cfg)

    write_provenance(
        out_dir=out_dir,
        pipeline="axis2_resolution",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[s.path for s in specs],
        extra={
            "datasets": [s.name for s in specs],
            "output_csvs": {k: str(v) for k, v in paths.items()},
        },
    )
    print(f"Axis-2 complete on {len(specs)} datasets. Outputs: {list(paths.keys())}")
    return 0
