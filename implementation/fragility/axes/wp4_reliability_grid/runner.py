"""Core computation for WP-4.

Input: a tidy CSV with per-cell-type metrics, one row per cell type with
columns {dataset, cell_type, rarity, stability, topk_jaccard, null_auc,
tail_gap, cell_count}. Produced by Axis 3's pipeline.

Output: a grid CSV with one row per (threshold combination, rarity_group)
giving pass rate. A second CSV records the per-rare-cell-type outcome so
the paper can show that no single cell type passes in a wide neighbourhood
of the primary thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import itertools
import numpy as np
import pandas as pd


# Pre-registered primary thresholds, frozen in code before data inspection.
PRIMARY_THRESHOLDS: Dict[str, float] = {
    "stability": 0.75,
    "topk_jaccard": 0.35,
    "null_auc": 0.65,
    "tail_gap": 0.0,   # strict positivity
    "cell_count": 80,
}


# Grid axes, step sizes pre-registered per the revision plan.
GRID: Dict[str, Sequence[float]] = {
    "stability":    (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    "topk_jaccard": (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    "null_auc":     (0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
    "tail_gap":     (-0.02, 0.00, 0.02, 0.05),
    "cell_count":   (40, 60, 80, 100, 120),
}

RARITY_ORDER = ("rare", "intermediate", "abundant")


@dataclass
class GridSummary:
    out_grid: Path
    out_primary: Path
    out_per_cell_type: Path


def _passes(row: Dict[str, float], thresh: Dict[str, float]) -> bool:
    return (
        row["stability"] >= thresh["stability"]
        and row["topk_jaccard"] >= thresh["topk_jaccard"]
        and row["null_auc"] >= thresh["null_auc"]
        and row["tail_gap"] > thresh["tail_gap"]
        and row["cell_count"] >= thresh["cell_count"]
    )


def _count_pass_rates(
    df: pd.DataFrame, thresh: Dict[str, float]
) -> Dict[str, float]:
    passes = df.apply(lambda r: _passes(r, thresh), axis=1)
    out: Dict[str, float] = {}
    for rarity in RARITY_ORDER:
        mask = df["rarity"] == rarity
        n = int(mask.sum())
        out[f"{rarity}_pass_rate"] = float(passes[mask].mean()) if n else float("nan")
        out[f"{rarity}_n"] = n
    out["rare_minus_abundant"] = out["abundant_pass_rate"] - out["rare_pass_rate"]
    out["total_pass_rate"] = float(passes.mean())
    return out


def run(
    metrics_csv: Path,
    out_dir: Path,
    grid: Dict[str, Sequence[float]] = GRID,
    primary: Dict[str, float] = PRIMARY_THRESHOLDS,
) -> GridSummary:
    """Sweep the 5-D threshold grid and emit CSVs.

    The input CSV must have columns: dataset, cell_type, rarity (str),
    stability, topk_jaccard, null_auc, tail_gap, cell_count.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)
    required = {
        "dataset", "cell_type", "rarity",
        "stability", "topk_jaccard", "null_auc", "tail_gap", "cell_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {missing}")

    keys = list(grid.keys())
    rows: List[Dict[str, float]] = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        thresh = dict(zip(keys, combo))
        stats = _count_pass_rates(df, thresh)
        rows.append({
            **{f"thresh_{k}": v for k, v in thresh.items()},
            **stats,
        })
    grid_df = pd.DataFrame(rows)
    grid_path = out_dir / "reliability_gate_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    primary_stats = _count_pass_rates(df, primary)
    primary_df = pd.DataFrame([{
        **{f"thresh_{k}": v for k, v in primary.items()},
        **primary_stats,
    }])
    primary_path = out_dir / "reliability_gate_primary.csv"
    primary_df.to_csv(primary_path, index=False)

    # Per-cell-type: under what fraction of grid points does each cell
    # type pass? Useful for showing specific rare types never clear the
    # gate under any reasonable setting.
    per_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        pass_count = 0
        total = 0
        for combo in itertools.product(*[grid[k] for k in keys]):
            thresh = dict(zip(keys, combo))
            if _passes(row.to_dict(), thresh):
                pass_count += 1
            total += 1
        per_rows.append({
            **row.to_dict(),
            "grid_passes": pass_count,
            "grid_total": total,
            "grid_pass_fraction": pass_count / total if total else float("nan"),
        })
    per_df = pd.DataFrame(per_rows)
    per_path = out_dir / "reliability_gate_per_cell_type.csv"
    per_df.to_csv(per_path, index=False)

    return GridSummary(
        out_grid=grid_path,
        out_primary=primary_path,
        out_per_cell_type=per_path,
    )
