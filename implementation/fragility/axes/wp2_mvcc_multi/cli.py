"""CLI wrapper for WP-2."""

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
from .runner import MVCCConfig, run as _run_runner


DEFAULT_CONFIG = {
    # Reuse the Axis-2 dataset list verbatim so the paper's Axes 1 and 2
    # always speak about the same cohorts.
    "datasets": AXIS2_DEFAULTS["datasets"],
    "config": {
        "n_top_genes": 600,
        "min_detect_frac": 0.05,
        "cell_sizes": [100, 200, 500, 1000, 2000, 3000, 5000, 8000, 15000],
        "anchors": [3000, 8000, 15000],
        "ks": [100, 500, 1000, 5000],
        "n_subsamples": 16,
        "mvcc_threshold_jaccard": 0.5,
        "mvcc_threshold_k": 1000,
        "seed_namespace": "wp2:default",
    },
    "base_seed": 20260218,
}


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="fragility wp2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="+", help="restrict dataset names")
    args = parser.parse_args(argv)

    config = load_config(args.config, default=DEFAULT_CONFIG)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.resolved.yaml")

    derived = seed_everything(
        base_seed=int(config["base_seed"]),
        components=(config["config"].get("seed_namespace", "wp2:default"),),
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
    cfg_dict = dict(config["config"])
    cfg_dict["cell_sizes"] = tuple(cfg_dict["cell_sizes"])
    cfg_dict["anchors"] = tuple(cfg_dict["anchors"])
    cfg_dict["ks"] = tuple(cfg_dict["ks"])
    cfg = MVCCConfig(**cfg_dict)

    paths = _run_runner(specs, out_dir=out_dir, config=cfg)

    write_provenance(
        out_dir=out_dir,
        pipeline="wp2_mvcc_multi",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[s.path for s in specs],
        extra={"output_csvs": {k: str(v) for k, v in paths.items()}},
    )
    print(f"WP-2 complete on {len(specs)} datasets. Outputs: {list(paths.keys())}")
    return 0
