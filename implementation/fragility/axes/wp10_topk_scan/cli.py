"""CLI wrapper for WP-10."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...utils import (
    dump_config,
    load_config,
    seed_everything,
    write_provenance,
)
from .runner import run as _run_runner


DEFAULT_CONFIG = {
    "scored_pairs_csv": "outputs/axis2/scored_pairs.csv",
    "ks": [100, 500, 1000, 5000],
    "k_percents": [1.0, 5.0, 10.0],
    "per_target_ks": [5, 10],
    "base_seed": 20260218,
}


def run(argv: list[str]) -> int:  # noqa: F811
    parser = argparse.ArgumentParser(prog="fragility wp10")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config, default=DEFAULT_CONFIG)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.resolved.yaml")

    derived = seed_everything(base_seed=int(config["base_seed"]))

    scored_pairs = Path(config["scored_pairs_csv"])
    if not scored_pairs.exists():
        raise FileNotFoundError(
            f"WP-10 needs a scored-pairs CSV at {scored_pairs}"
        )

    paths = _run_runner(
        scored_pairs,
        out_dir=out_dir,
        ks=tuple(config["ks"]),
        k_percents=tuple(config["k_percents"]),
        per_target_ks=tuple(config["per_target_ks"]),
    )

    write_provenance(
        out_dir=out_dir,
        pipeline="wp10_topk_scan",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[scored_pairs],
        extra={"output_csvs": {k: str(v) for k, v in paths.items()}},
    )
    print(f"WP-10 complete. Outputs: {[str(p) for p in paths.values()]}")
    return 0
