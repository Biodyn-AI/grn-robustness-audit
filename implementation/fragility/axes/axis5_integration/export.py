"""Convert per-tissue Axis-5 edge CSVs to a canonical scored_pairs.csv.

The subproject-38 integration outputs are one CSV per (tissue, variant)
with columns ``edge, tf, target, score, abs_score, rank``. This helper
merges them into a single long-format table with one row per
(dataset, scorer, pair_id, edge_id, source_id, target_id, score_coarse,
score_fine) so that WP-3 and WP-10 can reuse the same consumer.

* ``score_coarse``  := baseline (uncorrected) edge score
* ``score_fine``    := Harmony-integrated edge score
* ``pair_id``        := ``baseline_vs_{integration_variant}`` (e.g.
  ``baseline_vs_harmony_batch``, ``baseline_vs_harmony_donor``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


_DEFAULT_TISSUES = ("kidney", "lung", "immune")
_DEFAULT_VARIANTS = ("harmony_batch", "harmony_donor", "harmony_method")


def _load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"edge", "tf", "target", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return df[["edge", "tf", "target", "score"]]


def export_scored_pairs(
    input_dir: Path,
    output_csv: Path,
    tissues: Sequence[str] = _DEFAULT_TISSUES,
    variants: Sequence[str] = _DEFAULT_VARIANTS,
    scorer_name: str = "pearson_panel",
) -> pd.DataFrame:
    """Merge baseline + variant edge CSVs into one scored_pairs.csv.

    Missing variant files are skipped silently (some tissues lack donor-
    or method-keyed Harmony variants). At least one variant per tissue
    must be present or the function raises.
    """

    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for tissue in tissues:
        baseline_path = input_dir / f"{tissue}_edges_baseline.csv"
        if not baseline_path.exists():
            continue
        baseline = _load_edges(baseline_path).rename(columns={"score": "score_coarse"})
        baseline_map = dict(zip(baseline["edge"], baseline["score_coarse"]))
        found_variants = 0
        for variant in variants:
            var_path = input_dir / f"{tissue}_edges_{variant}.csv"
            if not var_path.exists():
                continue
            variant_df = _load_edges(var_path).rename(columns={"score": "score_fine"})
            # Inner join on edge (panel should be identical).
            merged = baseline.merge(
                variant_df[["edge", "score_fine"]],
                on="edge",
                how="inner",
            )
            for i, row in enumerate(merged.itertuples(index=False)):
                rows.append({
                    "dataset": tissue,
                    "scorer": scorer_name,
                    "pair_id": f"baseline_vs_{variant}",
                    "edge_id": f"{tissue}_{variant}_{i}",
                    "source_id": row.tf,
                    "target_id": row.target,
                    "score_coarse": float(row.score_coarse),
                    "score_fine": float(row.score_fine),
                })
            found_variants += 1
        if found_variants == 0:
            raise RuntimeError(
                f"no variant CSVs found for tissue '{tissue}' under {input_dir}"
            )

    if not rows:
        raise RuntimeError(f"no baseline edge CSVs found under {input_dir}")
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df
