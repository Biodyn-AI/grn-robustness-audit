#!/usr/bin/env python3
"""Cell-count subsampling robustness analysis for project 1 (2026-02-17).

This script builds a standalone evidence package from existing scaling artifacts.
It estimates edge-ranking stability across cell counts, derives operational MVCC
(minimum viable cell count) thresholds, and extracts high-cell-only edge sets.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SUBPROJECT_ROOT = ROOT / "subproject_38_cell_count_subsampling_robustness"
OUTPUT_DIR = SUBPROJECT_ROOT / "implementation" / "outputs"
FIG_DIR = SUBPROJECT_ROOT / "reports" / "figures"

SINGLE_CELL_ROOT = ROOT / "single_cell_mechinterp"
if str(SINGLE_CELL_ROOT) not in sys.path:
    sys.path.insert(0, str(SINGLE_CELL_ROOT))

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges


TOP_K_WINDOWS = [100, 500, 1000, 5000]
PRIMARY_TOP_K = 1000
REFERENCE_CELLS = 3000
REFERENCE_SEED = 1
CELL_ORDER = [200, 1000, 3000]
TIER_ORDER = ["small", "medium", "large"]

# Operational thresholds chosen for this sparse benchmark regime.
REF_JACCARD_THRESHOLD = 0.20
SEED_JACCARD_THRESHOLD = 0.05
REF_JACCARD_STRICT = 0.30
SEED_JACCARD_STRICT = 0.10


@dataclass(frozen=True)
class RunKey:
    tier: str
    cells: int
    seed: int


@dataclass
class RankedEdges:
    """Ranked edge table with fast lookup helpers."""

    table: pd.DataFrame
    edges: np.ndarray
    rank_map: Dict[str, int]


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    return (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y"])
    )


def load_run_inventory() -> pd.DataFrame:
    manifest_path = SINGLE_CELL_ROOT / "configs" / "scaling_workshop_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    manifest["candidate_sources_from_dorothea"] = _normalize_bool_col(
        manifest, "candidate_sources_from_dorothea"
    )
    manifest["candidate_targets_from_dorothea"] = _normalize_bool_col(
        manifest, "candidate_targets_from_dorothea"
    )

    canonical = manifest[
        (manifest["tissue"] == "kidney")
        & (manifest["model_tier"].isin(TIER_ORDER))
        & (manifest["gene_subset"] == "full_hvg")
        & (manifest["preprocess_hvg"] == 2000)
        & (manifest["max_genes"] == 1200)
        & (manifest["edge_method"] == "top_k")
        & (manifest["top_k"] == 100)
        & (manifest["max_cells"].isin(CELL_ORDER))
        & (~manifest["candidate_sources_from_dorothea"])
        & (~manifest["candidate_targets_from_dorothea"])
    ].copy()

    canonical["network_edges_abs"] = canonical["network_edges"].map(lambda p: ROOT / "single_cell_mechinterp" / p)
    canonical = canonical[canonical["network_edges_abs"].map(lambda p: Path(p).exists())].copy()
    canonical.sort_values(["model_tier", "max_cells", "seed"], inplace=True)
    canonical.reset_index(drop=True, inplace=True)
    return canonical


def load_alias_and_truth_sets() -> tuple[
    dict[str, str],
    set[tuple[str, str]],
    set[tuple[str, str]],
    set[str],
    set[str],
]:
    alias_path = SINGLE_CELL_ROOT / "external" / "hgnc_complete_set.txt"
    trrust_path = SINGLE_CELL_ROOT / "external" / "networks" / "trrust_human.tsv"
    dorothea_path = SINGLE_CELL_ROOT / "external" / "networks" / "dorothea_human.tsv"

    alias_map = load_hgnc_alias_map(alias_path)

    # Reuse existing normalization/evaluation utilities to keep symbol handling consistent.
    trrust_edges = load_dorothea(trrust_path, confidence_levels=["A", "B", "C", "D"])
    trrust_edges = normalize_edges(trrust_edges, alias_map)
    trrust_set = set(map(tuple, trrust_edges[["source", "target"]].drop_duplicates().itertuples(index=False, name=None)))

    dorothea_edges = load_dorothea(dorothea_path, confidence_levels=["A", "B", "C", "D"])
    dorothea_edges = normalize_edges(dorothea_edges, alias_map)
    dorothea_set = set(
        map(tuple, dorothea_edges[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))
    )

    dorothea_sources = set(dorothea_edges["source"].astype(str).tolist())
    dorothea_targets = set(dorothea_edges["target"].astype(str).tolist())

    return alias_map, trrust_set, dorothea_set, dorothea_sources, dorothea_targets


def _normalize_symbol(symbol: str, alias_map: dict[str, str]) -> str:
    value = str(symbol).strip().upper()
    if not value:
        return value
    return alias_map.get(value, value)


def load_ranked_edges(path: Path, alias_map: dict[str, str]) -> RankedEdges:
    table = pd.read_csv(path, sep="\t", usecols=["source", "target", "score"])
    table["source"] = table["source"].map(lambda x: _normalize_symbol(x, alias_map))
    table["target"] = table["target"].map(lambda x: _normalize_symbol(x, alias_map))
    table = table[(table["source"] != "") & (table["target"] != "")].copy()

    table["edge"] = table["source"] + "->" + table["target"]
    table.sort_values("score", ascending=False, inplace=True)
    table = table.drop_duplicates(subset=["edge"], keep="first").reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    edges = table["edge"].to_numpy(dtype=object)
    rank_map = dict(zip(edges, table["rank"].astype(int)))
    return RankedEdges(table=table, edges=edges, rank_map=rank_map)


def top_set(ranked: RankedEdges, top_k: int) -> set[str]:
    k = min(top_k, len(ranked.edges))
    return set(ranked.edges[:k])


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = len(set_a | set_b)
    if union == 0:
        return float("nan")
    return len(set_a & set_b) / union


def overlap_coeff(set_a: set[str], set_b: set[str]) -> float:
    denom = min(len(set_a), len(set_b))
    if denom == 0:
        return float("nan")
    return len(set_a & set_b) / denom


def union_rank_spearman(ranked_a: RankedEdges, ranked_b: RankedEdges, limit: int | None = None) -> float:
    if limit is None:
        union_edges = set(ranked_a.edges) | set(ranked_b.edges)
        max_rank = max(len(ranked_a.edges), len(ranked_b.edges)) + 1
    else:
        union_edges = set(ranked_a.edges[: min(limit, len(ranked_a.edges))])
        union_edges |= set(ranked_b.edges[: min(limit, len(ranked_b.edges))])
        max_rank = limit + 1

    if len(union_edges) < 2:
        return float("nan")

    edges = list(union_edges)
    ranks_a = np.array([ranked_a.rank_map.get(e, max_rank) for e in edges], dtype=float)
    ranks_b = np.array([ranked_b.rank_map.get(e, max_rank) for e in edges], dtype=float)
    corr, _ = spearmanr(ranks_a, ranks_b)
    return float(corr)


def build_rank_cache(runs: pd.DataFrame, alias_map: dict[str, str]) -> dict[RunKey, RankedEdges]:
    cache: dict[RunKey, RankedEdges] = {}
    for row in runs.itertuples(index=False):
        key = RunKey(row.model_tier, int(row.max_cells), int(row.seed))
        cache[key] = load_ranked_edges(Path(row.network_edges_abs), alias_map)
    return cache


def compute_reference_stability(rank_cache: dict[RunKey, RankedEdges]) -> tuple[pd.DataFrame, pd.DataFrame]:
    topk_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []

    for tier in TIER_ORDER:
        ref_key = RunKey(tier, REFERENCE_CELLS, REFERENCE_SEED)
        ref_ranked = rank_cache.get(ref_key)
        if ref_ranked is None:
            continue

        candidate_keys = sorted(
            [key for key in rank_cache if key.tier == tier and key.seed == REFERENCE_SEED],
            key=lambda key: key.cells,
        )
        for key in candidate_keys:
            ranked = rank_cache[key]
            for top_k in TOP_K_WINDOWS:
                edges_a = top_set(ranked, top_k)
                edges_b = top_set(ref_ranked, top_k)
                topk_rows.append(
                    {
                        "tier": tier,
                        "cells": key.cells,
                        "seed": key.seed,
                        "reference_cells": REFERENCE_CELLS,
                        "top_k": top_k,
                        "jaccard": jaccard(edges_a, edges_b),
                        "overlap_coeff": overlap_coeff(edges_a, edges_b),
                    }
                )

            rank_rows.append(
                {
                    "tier": tier,
                    "cells": key.cells,
                    "seed": key.seed,
                    "rank_scope": "all",
                    "spearman": union_rank_spearman(ranked, ref_ranked, limit=None),
                }
            )
            rank_rows.append(
                {
                    "tier": tier,
                    "cells": key.cells,
                    "seed": key.seed,
                    "rank_scope": "top20000",
                    "spearman": union_rank_spearman(ranked, ref_ranked, limit=20000),
                }
            )

    return pd.DataFrame(topk_rows), pd.DataFrame(rank_rows)


def compute_seed_stability(rank_cache: dict[RunKey, RankedEdges]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    rank_pair_rows: list[dict[str, object]] = []

    by_group: dict[tuple[str, int], list[RunKey]] = {}
    for key in rank_cache:
        by_group.setdefault((key.tier, key.cells), []).append(key)

    for (tier, cells), keys in by_group.items():
        keys = sorted(keys, key=lambda key: key.seed)
        if len(keys) < 2:
            continue

        for key_a, key_b in combinations(keys, 2):
            ranked_a = rank_cache[key_a]
            ranked_b = rank_cache[key_b]
            for top_k in TOP_K_WINDOWS:
                edges_a = top_set(ranked_a, top_k)
                edges_b = top_set(ranked_b, top_k)
                pair_rows.append(
                    {
                        "tier": tier,
                        "cells": cells,
                        "seed_a": key_a.seed,
                        "seed_b": key_b.seed,
                        "top_k": top_k,
                        "jaccard": jaccard(edges_a, edges_b),
                        "overlap_coeff": overlap_coeff(edges_a, edges_b),
                    }
                )

            rank_pair_rows.append(
                {
                    "tier": tier,
                    "cells": cells,
                    "seed_a": key_a.seed,
                    "seed_b": key_b.seed,
                    "rank_scope": "all",
                    "spearman": union_rank_spearman(ranked_a, ranked_b, limit=None),
                }
            )
            rank_pair_rows.append(
                {
                    "tier": tier,
                    "cells": cells,
                    "seed_a": key_a.seed,
                    "seed_b": key_b.seed,
                    "rank_scope": "top20000",
                    "spearman": union_rank_spearman(ranked_a, ranked_b, limit=20000),
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    rank_pair_df = pd.DataFrame(rank_pair_rows)

    summary_df = (
        pair_df.groupby(["tier", "cells", "top_k"], as_index=False)
        .agg(
            jaccard_mean=("jaccard", "mean"),
            jaccard_std=("jaccard", "std"),
            overlap_mean=("overlap_coeff", "mean"),
            overlap_std=("overlap_coeff", "std"),
            n_pairs=("jaccard", "count"),
        )
        .sort_values(["tier", "cells", "top_k"])
    )

    rank_summary_df = (
        rank_pair_df.groupby(["tier", "cells", "rank_scope"], as_index=False)
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            n_pairs=("spearman", "count"),
        )
        .sort_values(["tier", "cells", "rank_scope"])
    )

    return pair_df, summary_df, rank_summary_df


def monotone_envelope(values: list[float]) -> list[float]:
    running = -float("inf")
    out: list[float] = []
    for value in values:
        if math.isnan(value):
            out.append(value)
            continue
        running = max(running, value)
        out.append(running)
    return out


def interpolate_threshold(cells: list[int], values: list[float], threshold: float) -> float:
    points = [(c, v) for c, v in zip(cells, values) if not math.isnan(v)]
    if not points:
        return float("inf")

    points.sort(key=lambda item: item[0])
    cells_sorted = [p[0] for p in points]
    vals_sorted = [p[1] for p in points]

    if vals_sorted[0] >= threshold:
        return float(cells_sorted[0])
    if max(vals_sorted) < threshold:
        return float("inf")

    for i in range(1, len(points)):
        c0, v0 = points[i - 1]
        c1, v1 = points[i]
        if v0 < threshold <= v1:
            if v1 == v0:
                return float(c1)
            x0 = math.log10(c0)
            x1 = math.log10(c1)
            x = x0 + (threshold - v0) * (x1 - x0) / (v1 - v0)
            return float(10 ** x)

    return float("inf")


def estimate_mvcc(reference_topk: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    max_observed_cells = max(CELL_ORDER)

    for tier in TIER_ORDER:
        ref_curve = (
            reference_topk[
                (reference_topk["tier"] == tier)
                & (reference_topk["top_k"] == PRIMARY_TOP_K)
            ]
            .groupby("cells", as_index=False)["jaccard"]
            .mean()
            .sort_values("cells")
        )
        seed_curve = (
            seed_summary[
                (seed_summary["tier"] == tier)
                & (seed_summary["top_k"] == PRIMARY_TOP_K)
            ]
            .groupby("cells", as_index=False)["jaccard_mean"]
            .mean()
            .sort_values("cells")
        )

        ref_cells = ref_curve["cells"].astype(int).tolist()
        ref_values = monotone_envelope(ref_curve["jaccard"].astype(float).tolist())

        seed_cells = seed_curve["cells"].astype(int).tolist()
        seed_values = monotone_envelope(seed_curve["jaccard_mean"].astype(float).tolist())

        ref_mvcc = interpolate_threshold(ref_cells, ref_values, REF_JACCARD_THRESHOLD)
        seed_mvcc = interpolate_threshold(seed_cells, seed_values, SEED_JACCARD_THRESHOLD)
        operational_mvcc = max(ref_mvcc, seed_mvcc)

        ref_mvcc_strict = interpolate_threshold(ref_cells, ref_values, REF_JACCARD_STRICT)
        seed_mvcc_strict = interpolate_threshold(seed_cells, seed_values, SEED_JACCARD_STRICT)
        strict_mvcc = max(ref_mvcc_strict, seed_mvcc_strict)

        rows.append(
            {
                "tier": tier,
                "primary_top_k": PRIMARY_TOP_K,
                "reference_threshold": REF_JACCARD_THRESHOLD,
                "seed_threshold": SEED_JACCARD_THRESHOLD,
                "reference_mvcc_cells": ref_mvcc,
                "seed_mvcc_cells": seed_mvcc,
                "operational_mvcc_cells": operational_mvcc,
                "operational_mvcc_label": f">{max_observed_cells}" if math.isinf(operational_mvcc) else f"{operational_mvcc:.1f}",
                "strict_reference_threshold": REF_JACCARD_STRICT,
                "strict_seed_threshold": SEED_JACCARD_STRICT,
                "strict_mvcc_cells": strict_mvcc,
                "strict_mvcc_label": f">{max_observed_cells}" if math.isinf(strict_mvcc) else f"{strict_mvcc:.1f}",
            }
        )

    return pd.DataFrame(rows)


def split_edge(edge: str) -> tuple[str, str]:
    source, target = edge.split("->", 1)
    return source, target


def compute_edge_emergence(
    rank_cache: dict[RunKey, RankedEdges],
    trrust_set: set[tuple[str, str]],
    dorothea_set: set[tuple[str, str]],
    dorothea_sources: set[str],
    dorothea_targets: set[str],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for tier in TIER_ORDER:
        key_200 = RunKey(tier, 200, REFERENCE_SEED)
        key_1000 = RunKey(tier, 1000, REFERENCE_SEED)
        key_3000 = RunKey(tier, 3000, REFERENCE_SEED)
        if key_200 not in rank_cache or key_1000 not in rank_cache or key_3000 not in rank_cache:
            continue

        edges_200 = top_set(rank_cache[key_200], top_k)
        edges_1000 = top_set(rank_cache[key_1000], top_k)
        edges_3000 = top_set(rank_cache[key_3000], top_k)

        persistent = edges_200 & edges_1000 & edges_3000
        high_only = edges_3000 - (edges_200 | edges_1000)
        gained_1000 = edges_1000 - edges_200

        categories = {
            "persistent_all_counts": persistent,
            "high_cell_only": high_only,
            "gained_at_1000_vs_200": gained_1000,
        }

        for category, edge_set in categories.items():
            trrust_hits = 0
            dorothea_hits = 0
            source_tf_hits = 0
            target_known_hits = 0
            for edge in edge_set:
                source, target = split_edge(edge)
                pair = (source, target)
                trrust_hits += int(pair in trrust_set)
                dorothea_hits += int(pair in dorothea_set)
                source_tf_hits += int(source in dorothea_sources)
                target_known_hits += int(target in dorothea_targets)

            n_edges = len(edge_set)
            summary_rows.append(
                {
                    "tier": tier,
                    "top_k": top_k,
                    "category": category,
                    "n_edges": n_edges,
                    "trrust_hits": trrust_hits,
                    "trrust_rate": (trrust_hits / n_edges) if n_edges else float("nan"),
                    "dorothea_hits": dorothea_hits,
                    "dorothea_rate": (dorothea_hits / n_edges) if n_edges else float("nan"),
                    "source_tf_hits": source_tf_hits,
                    "source_tf_rate": (source_tf_hits / n_edges) if n_edges else float("nan"),
                    "target_known_hits": target_known_hits,
                    "target_known_rate": (target_known_hits / n_edges) if n_edges else float("nan"),
                }
            )

        ranked_3000 = rank_cache[key_3000].table.set_index("edge")
        for edge in sorted(high_only, key=lambda e: ranked_3000.at[e, "score"], reverse=True):
            source, target = split_edge(edge)
            pair = (source, target)
            detail_rows.append(
                {
                    "tier": tier,
                    "category": "high_cell_only",
                    "top_k": top_k,
                    "source": source,
                    "target": target,
                    "edge": edge,
                    "score_3000": float(ranked_3000.at[edge, "score"]),
                    "rank_3000": int(ranked_3000.at[edge, "rank"]),
                    "present_200": edge in edges_200,
                    "present_1000": edge in edges_1000,
                    "supported_trrust": pair in trrust_set,
                    "supported_dorothea": pair in dorothea_set,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df.sort_values(["tier", "rank_3000"], inplace=True)
    return pd.DataFrame(summary_rows), detail_df


def plot_reference_curves(reference_topk: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for tier in TIER_ORDER:
        subset = reference_topk[(reference_topk["tier"] == tier) & (reference_topk["top_k"] == PRIMARY_TOP_K)]
        if subset.empty:
            continue
        subset = subset.sort_values("cells")
        ax.plot(subset["cells"], subset["jaccard"], marker="o", linewidth=2, label=tier)

    ax.set_xscale("log")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel(f"Top-{PRIMARY_TOP_K} Jaccard vs {REFERENCE_CELLS}-cell reference")
    ax.set_title("Cell-count stability curves (seed 1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_seed_robustness(seed_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    subset = seed_summary[seed_summary["top_k"] == PRIMARY_TOP_K].copy()

    for tier in TIER_ORDER:
        tier_df = subset[subset["tier"] == tier].sort_values("cells")
        if tier_df.empty:
            continue
        yerr = tier_df["jaccard_std"].fillna(0.0)
        ax.errorbar(
            tier_df["cells"],
            tier_df["jaccard_mean"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=tier,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel(f"Seed-pair Jaccard (top-{PRIMARY_TOP_K})")
    ax.set_title("Within-cell-count seed robustness")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Model tier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_topk_sensitivity(reference_topk: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), sharey=True)

    for idx, tier in enumerate(TIER_ORDER):
        ax = axes[idx]
        tier_df = reference_topk[reference_topk["tier"] == tier]
        for cells, style in [(200, "--"), (1000, "-")]:
            sub = tier_df[tier_df["cells"] == cells].sort_values("top_k")
            if sub.empty:
                continue
            ax.plot(sub["top_k"], sub["jaccard"], marker="o", linestyle=style, label=f"cells={cells}")

        ax.set_xscale("log")
        ax.set_title(tier)
        ax.set_xlabel("Global top-k")
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.set_ylabel(f"Jaccard vs {REFERENCE_CELLS}-cell reference")
        ax.legend(fontsize=8)

    fig.suptitle("Top-k sensitivity of stability estimates", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_edge_support(summary_df: pd.DataFrame, out_path: Path) -> None:
    data = summary_df[summary_df["category"].isin(["persistent_all_counts", "high_cell_only"])].copy()
    if data.empty:
        return

    labels = []
    trrust_rates = []
    dorothea_rates = []

    for tier in TIER_ORDER:
        for category in ["persistent_all_counts", "high_cell_only"]:
            row = data[(data["tier"] == tier) & (data["category"] == category)]
            if row.empty:
                continue
            labels.append(f"{tier}\n{category.replace('_', ' ')}")
            trrust_rates.append(float(row.iloc[0]["trrust_rate"]))
            dorothea_rates.append(float(row.iloc[0]["dorothea_rate"]))

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12.0, 4.5))
    ax.bar(x - width / 2, trrust_rates, width=width, label="TRRUST support rate")
    ax.bar(x + width / 2, dorothea_rates, width=width, label="DoRothEA support rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Support rate")
    ax.set_ylim(0, max(0.02, np.nanmax(trrust_rates + dorothea_rates) * 1.2))
    ax.set_title("External support for persistent vs high-cell-only edges")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def write_summary_markdown(
    run_inventory: pd.DataFrame,
    reference_topk: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mvcc_df: pd.DataFrame,
    emergence_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Cell-Count Subsampling Robustness: Analysis Summary")
    lines.append("")
    lines.append("## Data Scope")
    lines.append(f"- Canonical runs analyzed: {len(run_inventory)}")
    lines.append("- Tissue: kidney")
    lines.append("- Model tiers: small, medium, large")
    lines.append("- Cell counts: 200, 1000, 3000 (available per manifest + artifacts)")
    lines.append("- Seed policy: seed-1 anchored reference analyses; all available seeds for within-count robustness")
    lines.append("")

    lines.append("## Key Stability Metrics (top-1000)")
    topk_subset = reference_topk[reference_topk["top_k"] == PRIMARY_TOP_K].sort_values(["tier", "cells"])
    for row in topk_subset.itertuples(index=False):
        lines.append(
            f"- {row.tier} @ {int(row.cells)} cells: Jaccard vs 3000-cell reference = {row.jaccard:.3f}, overlap coeff = {row.overlap_coeff:.3f}"
        )
    lines.append("")

    lines.append("## Seed Robustness (top-1000)")
    seed_subset = seed_summary[seed_summary["top_k"] == PRIMARY_TOP_K].sort_values(["tier", "cells"])
    for row in seed_subset.itertuples(index=False):
        lines.append(
            f"- {row.tier} @ {int(row.cells)} cells: mean seed-pair Jaccard = {row.jaccard_mean:.3f} (std={0.0 if pd.isna(row.jaccard_std) else row.jaccard_std:.3f}, n={int(row.n_pairs)})"
        )
    lines.append("")

    lines.append("## MVCC Estimates")
    for row in mvcc_df.itertuples(index=False):
        lines.append(
            f"- {row.tier}: operational MVCC={row.operational_mvcc_label} cells; strict MVCC={row.strict_mvcc_label} cells"
        )
    lines.append("")

    lines.append("## Edge Emergence")
    for row in emergence_summary.sort_values(["tier", "category"]).itertuples(index=False):
        lines.append(
            f"- {row.tier} / {row.category}: n={int(row.n_edges)}, TRRUST rate={0.0 if pd.isna(row.trrust_rate) else row.trrust_rate:.4f}, DoRothEA rate={0.0 if pd.isna(row.dorothea_rate) else row.dorothea_rate:.4f}, TF-source rate={0.0 if pd.isna(row.source_tf_rate) else row.source_tf_rate:.4f}, known-target rate={0.0 if pd.isna(row.target_known_rate) else row.target_known_rate:.4f}"
        )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    run_inventory = load_run_inventory()
    (
        alias_map,
        trrust_set,
        dorothea_set,
        dorothea_sources,
        dorothea_targets,
    ) = load_alias_and_truth_sets()
    rank_cache = build_rank_cache(run_inventory, alias_map)

    reference_topk, reference_rank = compute_reference_stability(rank_cache)
    seed_pairwise, seed_summary, seed_rank_summary = compute_seed_stability(rank_cache)
    mvcc_df = estimate_mvcc(reference_topk, seed_summary)
    emergence_summary, emergence_detail = compute_edge_emergence(
        rank_cache=rank_cache,
        trrust_set=trrust_set,
        dorothea_set=dorothea_set,
        dorothea_sources=dorothea_sources,
        dorothea_targets=dorothea_targets,
        top_k=5000,
    )

    run_inventory.to_csv(OUTPUT_DIR / "run_inventory.csv", index=False)
    reference_topk.to_csv(OUTPUT_DIR / "reference_topk_stability.csv", index=False)
    reference_rank.to_csv(OUTPUT_DIR / "reference_rank_stability.csv", index=False)
    seed_pairwise.to_csv(OUTPUT_DIR / "seed_pairwise_stability.csv", index=False)
    seed_summary.to_csv(OUTPUT_DIR / "seed_stability_summary.csv", index=False)
    seed_rank_summary.to_csv(OUTPUT_DIR / "seed_rank_summary.csv", index=False)
    mvcc_df.to_csv(OUTPUT_DIR / "mvcc_estimates.csv", index=False)
    emergence_summary.to_csv(OUTPUT_DIR / "edge_emergence_summary.csv", index=False)
    emergence_detail.to_csv(OUTPUT_DIR / "high_cell_only_edges_top5000.csv", index=False)

    plot_reference_curves(reference_topk, FIG_DIR / "fig1_reference_stability_curves.png")
    plot_seed_robustness(seed_summary, FIG_DIR / "fig2_seed_robustness.png")
    plot_topk_sensitivity(reference_topk, FIG_DIR / "fig3_topk_sensitivity.png")
    plot_edge_support(emergence_summary, FIG_DIR / "fig4_edge_support_by_emergence_class.png")

    write_summary_markdown(
        run_inventory=run_inventory,
        reference_topk=reference_topk,
        seed_summary=seed_summary,
        mvcc_df=mvcc_df,
        emergence_summary=emergence_summary,
        out_path=OUTPUT_DIR / "analysis_summary.md",
    )

    print("Analysis complete.")
    print(f"Runs analyzed: {len(run_inventory)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Figure dir: {FIG_DIR}")


if __name__ == "__main__":
    main()
