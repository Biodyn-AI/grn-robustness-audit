"""CLI wrapper for WP-3.

Usage::

    python -m fragility wp3 --config configs/wp3_rss_redesign.yaml \
                            --out outputs/wp3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils import (
    dump_config,
    load_config,
    seed_everything,
    write_provenance,
)
from .runner import ScoredPair, run


DEFAULT_CONFIG = {
    "k": 1000,
    "n_permutations": 1000,
    "weights_step": 0.1,
    "default_weights": [0.4, 0.3, 0.3],
    "input_scores_csv": "",       # required: long CSV with scored pairs
    "base_seed": 20260218,
}


def _load_pairs_from_csv(path: Path) -> list[ScoredPair]:
    """Load scored pairs from a long-format CSV.

    Expected columns: ``dataset``, ``scorer``, ``pair_id``, ``edge_id``,
    ``score_coarse``, ``score_fine``. All edges within the same
    (dataset, scorer, pair_id) group form one :class:`ScoredPair`.
    """

    df = pd.read_csv(path)
    required = {"dataset", "scorer", "pair_id", "edge_id", "score_coarse", "score_fine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {missing}")
    pairs: list[ScoredPair] = []
    for (ds, sc, pid), group in df.groupby(["dataset", "scorer", "pair_id"]):
        group = group.sort_values("edge_id")
        pairs.append(ScoredPair(
            dataset=str(ds),
            scorer=str(sc),
            pair_id=str(pid),
            scores_coarse=group["score_coarse"].to_numpy(dtype=float),
            scores_fine=group["score_fine"].to_numpy(dtype=float),
        ))
    return pairs


def run(argv: list[str]) -> int:  # noqa: F811 (export as module-level run)
    parser = argparse.ArgumentParser(prog="fragility wp3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config, default=DEFAULT_CONFIG)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.resolved.yaml")

    derived = seed_everything(
        base_seed=int(config["base_seed"]),
        components=("wp3:null:default",),
    )

    input_csv = Path(config["input_scores_csv"])
    if not input_csv.exists():
        raise FileNotFoundError(
            f"WP-3 requires --config to point at a CSV of scored pairs; "
            f"got path {input_csv} (missing)."
        )
    pairs = _load_pairs_from_csv(input_csv)
    if not pairs:
        raise RuntimeError(f"no scored pairs found in {input_csv}")

    from .runner import run as _run_runner

    paths = _run_runner(
        pairs,
        out_dir=out_dir,
        k=int(config["k"]),
        n_null=int(config["n_permutations"]),
        weights_step=float(config["weights_step"]),
        default_weights=tuple(config["default_weights"]),
    )

    write_provenance(
        out_dir=out_dir,
        pipeline="wp3_rss_redesign",
        config=config,
        base_seed=int(config["base_seed"]),
        derived_seeds=derived,
        input_files=[input_csv],
        extra={"output_csvs": {k: str(v) for k, v in paths.items()}},
    )
    print(f"WP-3 complete. Outputs: {[str(p) for p in paths.values()]}")
    return 0
