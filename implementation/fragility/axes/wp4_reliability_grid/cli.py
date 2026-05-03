"""CLI wrapper for WP-4 (reliability-gate threshold sensitivity)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...utils import (
    dump_config,
    load_config,
    seed_everything,
    write_provenance,
)
from .runner import run, GRID, PRIMARY_THRESHOLDS


DEFAULT_CONFIG = {
    "metrics_csv": "outputs/axis3/cell_type_metrics.csv",
    "base_seed": 20260218,
    # Explicit grid (defaults to module-level GRID) — included so a reviewer
    # reading the config knows exactly what axis values were swept.
    "grid": {k: list(v) for k, v in GRID.items()},
    "primary": dict(PRIMARY_THRESHOLDS),
}


def run(argv: list[str]) -> int:  # noqa: F811
    parser = argparse.ArgumentParser(prog="fragility wp4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config, default=DEFAULT_CONFIG)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.resolved.yaml")

    derived = seed_everything(base_seed=int(config["base_seed"]))

    metrics_csv = Path(config["metrics_csv"])
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"WP-4 requires a per-cell-type metrics CSV; not found at {metrics_csv}"
        )

    from .runner import run as _run_runner

    summary = _run_runner(
        metrics_csv=metrics_csv,
        out_dir=out_dir,
        grid={k: tuple(v) for k, v in config["grid"].items()},
        primary=dict(config["primary"]),
    )

    write_provenance(
        out_dir=out_dir,
        pipeline="wp4_reliability_grid",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[metrics_csv],
        extra={
            "output_csvs": {
                "reliability_gate_grid": str(summary.out_grid),
                "reliability_gate_primary": str(summary.out_primary),
                "reliability_gate_per_cell_type": str(summary.out_per_cell_type),
            }
        },
    )
    grid_size = 1
    for key, values in config["grid"].items():
        grid_size *= len(values)
    print(f"WP-4 complete. Grid size: {grid_size}")
    return 0
