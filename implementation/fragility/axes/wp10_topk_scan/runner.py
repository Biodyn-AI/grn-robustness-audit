"""Core computation for WP-10."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...metrics import topk_overlap, topk_jaccard, topk_per_target_overlap, drift_norm


def _resolve_ks(
    ks: Sequence[int],
    percents: Sequence[float],
    total: int,
) -> List[int]:
    """Combine explicit k values with percentages of the edge universe."""

    out: List[int] = [int(k) for k in ks if k <= total]
    for p in percents:
        kk = max(1, int(round(total * p / 100.0)))
        if kk <= total:
            out.append(kk)
    seen: set = set()
    dedup: List[int] = []
    for k in sorted(out):
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return dedup


def run(
    scored_pairs_csv: Path,
    out_dir: Path,
    ks: Sequence[int] = (100, 500, 1000, 5000),
    k_percents: Sequence[float] = (1.0, 5.0, 10.0),
    per_target_ks: Sequence[int] = (5, 10),
) -> Dict[str, Path]:
    """Emit per-k and per-target-k metric scans."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scored_pairs_csv)
    required = {"dataset", "scorer", "pair_id", "edge_id",
                "score_coarse", "score_fine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {missing}")

    has_target = "target_id" in df.columns

    rows: List[dict] = []
    per_target_rows: List[dict] = []

    for (ds, sc, pid), group in df.groupby(["dataset", "scorer", "pair_id"]):
        group = group.sort_values("edge_id")
        a = group["score_coarse"].to_numpy(dtype=float)
        b = group["score_fine"].to_numpy(dtype=float)
        total = len(a)
        resolved_ks = _resolve_ks(ks, k_percents, total=total)
        for k in resolved_ks:
            rows.append({
                "dataset": ds,
                "scorer": sc,
                "pair_id": pid,
                "edge_universe_size": total,
                "k": int(k),
                "k_fraction": k / total,
                "overlap": topk_overlap(a, b, k=k),
                "jaccard": topk_jaccard(a, b, k=k),
                "drift_norm": drift_norm(a, b, k=k),
            })
        if has_target:
            tids = group["target_id"].to_numpy()
            for k_pt in per_target_ks:
                if k_pt <= 0:
                    continue
                per_target_rows.append({
                    "dataset": ds,
                    "scorer": sc,
                    "pair_id": pid,
                    "edge_universe_size": total,
                    "k_per_target": int(k_pt),
                    "per_target_jaccard": topk_per_target_overlap(a, b, tids, k_pt),
                })

    scan_df = pd.DataFrame(rows)
    scan_path = out_dir / "topk_scan.csv"
    scan_df.to_csv(scan_path, index=False)

    per_target_path = out_dir / "topk_per_target.csv"
    if per_target_rows:
        pd.DataFrame(per_target_rows).to_csv(per_target_path, index=False)
    else:
        # Emit an empty-but-well-formed CSV for reproducibility.
        pd.DataFrame(
            columns=[
                "dataset", "scorer", "pair_id",
                "edge_universe_size", "k_per_target", "per_target_jaccard",
            ]
        ).to_csv(per_target_path, index=False)

    return {
        "topk_scan": scan_path,
        "topk_per_target": per_target_path,
    }
