#!/usr/bin/env python3
"""Adversarial repair pipeline for subproject 38.

Fixes applied versus the initial analysis:
1. Shared-gene-universe filtering for all cross-cell/seed comparisons.
2. Multi-anchor reference evaluation (all available 3000-cell seeds).
3. Null-calibrated overlap enrichment against random overlap expectation.
4. New constrained-policy experiments (TF-source and TF-source+target).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

ROOT = Path(__file__).resolve().parents[2]
SUBPROJECT_ROOT = ROOT / "subproject_38_cell_count_subsampling_robustness"
OUTPUT_DIR = SUBPROJECT_ROOT / "implementation" / "outputs_adversarial"
FIG_DIR = SUBPROJECT_ROOT / "reports" / "figures"
EDGE_CACHE_DIR = OUTPUT_DIR / "policy_edges"

SINGLE_CELL_ROOT = ROOT / "single_cell_mechinterp"
if str(SINGLE_CELL_ROOT) not in sys.path:
    sys.path.insert(0, str(SINGLE_CELL_ROOT))

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges


TIER_ORDER = ["small", "medium", "large"]
CELL_ORDER = [200, 1000, 3000]
TOP_K_WINDOWS = [100, 500, 1000, 5000]
PRIMARY_TOP_K = 1000
REFERENCE_CELLS = 3000

REF_THRESHOLD = 0.20
SEED_THRESHOLD = 0.05
REF_THRESHOLD_STRICT = 0.30
SEED_THRESHOLD_STRICT = 0.10

POLICY_UNCONSTRAINED = "unconstrained"
POLICY_TF_SOURCE = "tf_source"
POLICY_TF_SOURCE_TARGET = "tf_source_target"
POLICIES = [POLICY_UNCONSTRAINED, POLICY_TF_SOURCE, POLICY_TF_SOURCE_TARGET]


@dataclass(frozen=True)
class RunKey:
    tier: str
    cells: int
    seed: int


@dataclass
class RankedEdges:
    table: pd.DataFrame
    edges: np.ndarray
    rank_map: Dict[str, int]


@dataclass
class RunRecord:
    key: RunKey
    experiment_id: str
    network_edges: Path
    attention_scores: Path
    processed_h5ad: Path


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])


def load_inventory() -> list[RunRecord]:
    manifest = pd.read_csv(SINGLE_CELL_ROOT / "configs" / "scaling_workshop_manifest.csv")
    manifest["candidate_sources_from_dorothea"] = normalize_bool_col(manifest, "candidate_sources_from_dorothea")
    manifest["candidate_targets_from_dorothea"] = normalize_bool_col(manifest, "candidate_targets_from_dorothea")

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

    runs: list[RunRecord] = []
    for row in canonical.itertuples(index=False):
        network_edges = SINGLE_CELL_ROOT / str(row.network_edges)
        attention_scores = SINGLE_CELL_ROOT / str(row.attention_scores)
        processed_h5ad = SINGLE_CELL_ROOT / str(row.processed_h5ad)
        if not (network_edges.exists() and attention_scores.exists() and processed_h5ad.exists()):
            continue
        runs.append(
            RunRecord(
                key=RunKey(str(row.model_tier), int(row.max_cells), int(row.seed)),
                experiment_id=str(row.experiment_id),
                network_edges=network_edges,
                attention_scores=attention_scores,
                processed_h5ad=processed_h5ad,
            )
        )

    runs.sort(key=lambda r: (r.key.tier, r.key.cells, r.key.seed))
    return runs


def load_truth(alias_map: dict[str, str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[str], set[str]]:
    trrust = load_dorothea(SINGLE_CELL_ROOT / "external" / "networks" / "trrust_human.tsv", ["A", "B", "C", "D"])
    trrust = normalize_edges(trrust, alias_map)
    trrust_set = set(map(tuple, trrust[["source", "target"]].drop_duplicates().itertuples(index=False, name=None)))

    dorothea = load_dorothea(SINGLE_CELL_ROOT / "external" / "networks" / "dorothea_human.tsv", ["A", "B", "C", "D"])
    dorothea = normalize_edges(dorothea, alias_map)
    dorothea_set = set(map(tuple, dorothea[["source", "target"]].drop_duplicates().itertuples(index=False, name=None)))
    dorothea_sources = set(dorothea["source"].astype(str))
    dorothea_targets = set(dorothea["target"].astype(str))
    return trrust_set, dorothea_set, dorothea_sources, dorothea_targets


def normalize_symbol(symbol: str, alias_map: dict[str, str]) -> str:
    value = str(symbol).strip().upper()
    if not value:
        return value
    return alias_map.get(value, value)


def to_ranked(df: pd.DataFrame) -> RankedEdges:
    table = df.copy()
    table["edge"] = table["source"] + "->" + table["target"]
    table = table.sort_values("score", ascending=False)
    table = table.drop_duplicates(subset=["edge"], keep="first").reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    edges = table["edge"].to_numpy(dtype=object)
    rank_map = dict(zip(edges, table["rank"].astype(int)))
    return RankedEdges(table=table, edges=edges, rank_map=rank_map)


def load_unconstrained_ranked(path: Path, alias_map: dict[str, str]) -> RankedEdges:
    df = pd.read_csv(path, sep="\t", usecols=["source", "target", "score"])
    df["source"] = df["source"].map(lambda x: normalize_symbol(x, alias_map))
    df["target"] = df["target"].map(lambda x: normalize_symbol(x, alias_map))
    df = df[(df["source"] != "") & (df["target"] != "")]
    return to_ranked(df)


def infer_constrained_edges(
    scores: np.ndarray,
    gene_names: list[str],
    source_allowed: set[str] | None,
    target_allowed: set[str] | None,
    top_k_per_source: int = 100,
) -> RankedEdges:
    n_genes = len(gene_names)
    source_idx = np.arange(n_genes)
    target_idx_all = np.arange(n_genes)

    if source_allowed is not None:
        source_mask = np.array([g in source_allowed for g in gene_names], dtype=bool)
        source_idx = source_idx[source_mask]

    if target_allowed is not None:
        target_mask = np.array([g in target_allowed for g in gene_names], dtype=bool)
        target_idx_all = target_idx_all[target_mask]

    rows: list[tuple[str, str, float]] = []
    for i in source_idx:
        candidates = target_idx_all
        if candidates.size == 0:
            continue
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            continue

        row_scores = scores[i, candidates]
        k = min(top_k_per_source, candidates.size)
        if k <= 0:
            continue
        top_local = np.argpartition(-row_scores, k - 1)[:k]
        chosen = candidates[top_local]
        for j in chosen:
            value = float(scores[i, j])
            if not np.isfinite(value):
                continue
            rows.append((gene_names[i], gene_names[j], value))

    df = pd.DataFrame(rows, columns=["source", "target", "score"])
    return to_ranked(df)


def filter_ranked_by_genes(ranked: RankedEdges, genes: set[str]) -> RankedEdges:
    table = ranked.table
    filtered = table[(table["source"].isin(genes)) & (table["target"].isin(genes)) & (table["source"] != table["target"])]
    return to_ranked(filtered[["source", "target", "score"]])


def top_set(ranked: RankedEdges, top_k: int) -> set[str]:
    k = min(top_k, len(ranked.edges))
    return set(ranked.edges[:k])


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    if union == 0:
        return float("nan")
    return len(a & b) / union


def overlap_coeff(a: set[str], b: set[str]) -> float:
    denom = min(len(a), len(b))
    if denom == 0:
        return float("nan")
    return len(a & b) / denom


def union_rank_spearman(a: RankedEdges, b: RankedEdges, limit: int = 20000) -> float:
    union_edges = set(a.edges[: min(limit, len(a.edges))]) | set(b.edges[: min(limit, len(b.edges))])
    if len(union_edges) < 2:
        return float("nan")

    max_rank = limit + 1
    edges = list(union_edges)
    r1 = np.array([a.rank_map.get(e, max_rank) for e in edges], dtype=float)
    r2 = np.array([b.rank_map.get(e, max_rank) for e in edges], dtype=float)
    corr, _ = spearmanr(r1, r2)
    return float(corr)


def expected_jaccard_random(n_a: int, n_b: int, universe_size: int) -> float:
    if universe_size <= 0 or n_a <= 0 or n_b <= 0:
        return float("nan")
    exp_intersection = (n_a * n_b) / universe_size
    denom = n_a + n_b - exp_intersection
    if denom <= 0:
        return float("nan")
    return float(exp_intersection / denom)


def pair_overlap_pvalue(observed_intersection: int, n_a: int, n_b: int, universe_size: int) -> float:
    if universe_size <= 0 or n_a <= 0 or n_b <= 0:
        return float("nan")
    return float(hypergeom.sf(observed_intersection - 1, universe_size, n_b, n_a))


def quantile_or_nan(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def monotone(values: list[float]) -> list[float]:
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

    points.sort(key=lambda x: x[0])
    c = [p[0] for p in points]
    v = [p[1] for p in points]

    if v[0] >= threshold:
        return float(c[0])
    if max(v) < threshold:
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
            return float(10**x)

    return float("inf")


def build_policy_caches(
    runs: list[RunRecord],
    alias_map: dict[str, str],
    dorothea_sources: set[str],
    dorothea_targets: set[str],
) -> tuple[dict[str, dict[RunKey, RankedEdges]], dict[RunKey, set[str]], pd.DataFrame]:
    policy_cache: dict[str, dict[RunKey, RankedEdges]] = {policy: {} for policy in POLICIES}
    gene_sets: dict[RunKey, set[str]] = {}

    inventory_rows: list[dict[str, object]] = []

    for run in runs:
        with h5py.File(run.processed_h5ad, "r") as handle:
            raw_index = handle["var"]["_index"][:]
        gene_names = [
            normalize_symbol(g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else str(g), alias_map)
            for g in raw_index
        ]
        gene_set = set(g for g in gene_names if g)
        gene_sets[run.key] = gene_set

        unconstrained = load_unconstrained_ranked(run.network_edges, alias_map)
        policy_cache[POLICY_UNCONSTRAINED][run.key] = unconstrained

        score_mat = np.load(run.attention_scores, mmap_mode="r")

        for policy in [POLICY_TF_SOURCE, POLICY_TF_SOURCE_TARGET]:
            policy_dir = EDGE_CACHE_DIR / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            cached_path = policy_dir / f"{run.experiment_id}.tsv"

            if cached_path.exists():
                cached_df = pd.read_csv(cached_path, sep="\t")
                ranked = to_ranked(cached_df[["source", "target", "score"]])
            else:
                source_allowed = dorothea_sources
                target_allowed = None if policy == POLICY_TF_SOURCE else dorothea_targets
                ranked = infer_constrained_edges(
                    scores=score_mat,
                    gene_names=gene_names,
                    source_allowed=source_allowed,
                    target_allowed=target_allowed,
                    top_k_per_source=100,
                )
                ranked.table[["source", "target", "score"]].to_csv(cached_path, sep="\t", index=False)

            policy_cache[policy][run.key] = ranked

        inventory_rows.append(
            {
                "tier": run.key.tier,
                "cells": run.key.cells,
                "seed": run.key.seed,
                "experiment_id": run.experiment_id,
                "n_genes": len(gene_set),
                "n_edges_unconstrained": len(unconstrained.table),
                "n_edges_tf_source": len(policy_cache[POLICY_TF_SOURCE][run.key].table),
                "n_edges_tf_source_target": len(policy_cache[POLICY_TF_SOURCE_TARGET][run.key].table),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["tier", "cells", "seed"])
    return policy_cache, gene_sets, inventory_df


def compute_reference_stability(
    policy_cache: dict[str, dict[RunKey, RankedEdges]],
    gene_sets: dict[RunKey, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topk_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []

    for policy, runs in policy_cache.items():
        for tier in TIER_ORDER:
            refs = sorted([k for k in runs if k.tier == tier and k.cells == REFERENCE_CELLS], key=lambda x: x.seed)
            cands = sorted([k for k in runs if k.tier == tier and k.cells in CELL_ORDER], key=lambda x: (x.cells, x.seed))
            if not refs:
                continue

            for cand_key in cands:
                cand = runs[cand_key]
                for ref_key in refs:
                    ref = runs[ref_key]
                    shared_genes = gene_sets[cand_key] & gene_sets[ref_key]
                    if len(shared_genes) < 2:
                        continue

                    cand_f = filter_ranked_by_genes(cand, shared_genes)
                    ref_f = filter_ranked_by_genes(ref, shared_genes)

                    rank_rows.append(
                        {
                            "policy": policy,
                            "tier": tier,
                            "cells": cand_key.cells,
                            "candidate_seed": cand_key.seed,
                            "reference_seed": ref_key.seed,
                            "n_shared_genes": len(shared_genes),
                            "rank_scope": "top20000",
                            "spearman": union_rank_spearman(cand_f, ref_f, limit=20000),
                        }
                    )

                    universe_size = len(shared_genes) * (len(shared_genes) - 1)
                    for top_k in TOP_K_WINDOWS:
                        set_a = top_set(cand_f, top_k)
                        set_b = top_set(ref_f, top_k)
                        inter = len(set_a & set_b)
                        j = jaccard(set_a, set_b)
                        expected_j = expected_jaccard_random(len(set_a), len(set_b), universe_size)
                        topk_rows.append(
                            {
                                "policy": policy,
                                "tier": tier,
                                "cells": cand_key.cells,
                                "candidate_seed": cand_key.seed,
                                "reference_seed": ref_key.seed,
                                "top_k": top_k,
                                "n_shared_genes": len(shared_genes),
                                "n_a": len(set_a),
                                "n_b": len(set_b),
                                "intersection": inter,
                                "jaccard": j,
                                "overlap_coeff": overlap_coeff(set_a, set_b),
                                "expected_jaccard_random": expected_j,
                                "jaccard_enrichment": (j / expected_j) if (not math.isnan(j) and expected_j and expected_j > 0) else float("nan"),
                                "p_overlap_geq": pair_overlap_pvalue(inter, len(set_a), len(set_b), universe_size),
                            }
                        )

    topk_df = pd.DataFrame(topk_rows)
    rank_df = pd.DataFrame(rank_rows)

    summary = (
        topk_df.groupby(["policy", "tier", "cells", "top_k"], as_index=False)
        .agg(
            jaccard_mean=("jaccard", "mean"),
            jaccard_p05=("jaccard", lambda s: quantile_or_nan(s, 0.05)),
            jaccard_p95=("jaccard", lambda s: quantile_or_nan(s, 0.95)),
            overlap_mean=("overlap_coeff", "mean"),
            enrichment_mean=("jaccard_enrichment", "mean"),
            p_overlap_median=("p_overlap_geq", "median"),
            n_pairs=("jaccard", "count"),
        )
        .sort_values(["policy", "tier", "cells", "top_k"])
    )

    return topk_df, rank_df, summary


def compute_seed_robustness(
    policy_cache: dict[str, dict[RunKey, RankedEdges]],
    gene_sets: dict[RunKey, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topk_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []

    for policy, runs in policy_cache.items():
        for tier in TIER_ORDER:
            for cells in CELL_ORDER:
                keys = sorted([k for k in runs if k.tier == tier and k.cells == cells], key=lambda x: x.seed)
                if len(keys) < 2:
                    continue

                for key_a, key_b in combinations(keys, 2):
                    ranked_a = runs[key_a]
                    ranked_b = runs[key_b]
                    shared_genes = gene_sets[key_a] & gene_sets[key_b]
                    if len(shared_genes) < 2:
                        continue

                    a_f = filter_ranked_by_genes(ranked_a, shared_genes)
                    b_f = filter_ranked_by_genes(ranked_b, shared_genes)
                    rank_rows.append(
                        {
                            "policy": policy,
                            "tier": tier,
                            "cells": cells,
                            "seed_a": key_a.seed,
                            "seed_b": key_b.seed,
                            "n_shared_genes": len(shared_genes),
                            "rank_scope": "top20000",
                            "spearman": union_rank_spearman(a_f, b_f, limit=20000),
                        }
                    )

                    universe_size = len(shared_genes) * (len(shared_genes) - 1)
                    for top_k in TOP_K_WINDOWS:
                        set_a = top_set(a_f, top_k)
                        set_b = top_set(b_f, top_k)
                        inter = len(set_a & set_b)
                        j = jaccard(set_a, set_b)
                        expected_j = expected_jaccard_random(len(set_a), len(set_b), universe_size)
                        topk_rows.append(
                            {
                                "policy": policy,
                                "tier": tier,
                                "cells": cells,
                                "seed_a": key_a.seed,
                                "seed_b": key_b.seed,
                                "top_k": top_k,
                                "n_shared_genes": len(shared_genes),
                                "n_a": len(set_a),
                                "n_b": len(set_b),
                                "intersection": inter,
                                "jaccard": j,
                                "overlap_coeff": overlap_coeff(set_a, set_b),
                                "expected_jaccard_random": expected_j,
                                "jaccard_enrichment": (j / expected_j) if (not math.isnan(j) and expected_j and expected_j > 0) else float("nan"),
                                "p_overlap_geq": pair_overlap_pvalue(inter, len(set_a), len(set_b), universe_size),
                            }
                        )

    topk_df = pd.DataFrame(topk_rows)
    rank_df = pd.DataFrame(rank_rows)

    summary = (
        topk_df.groupby(["policy", "tier", "cells", "top_k"], as_index=False)
        .agg(
            jaccard_mean=("jaccard", "mean"),
            jaccard_p05=("jaccard", lambda s: quantile_or_nan(s, 0.05)),
            jaccard_p95=("jaccard", lambda s: quantile_or_nan(s, 0.95)),
            overlap_mean=("overlap_coeff", "mean"),
            enrichment_mean=("jaccard_enrichment", "mean"),
            p_overlap_median=("p_overlap_geq", "median"),
            n_pairs=("jaccard", "count"),
        )
        .sort_values(["policy", "tier", "cells", "top_k"])
    )

    return topk_df, rank_df, summary


def estimate_mvcc(reference_summary: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for policy in POLICIES:
        for tier in TIER_ORDER:
            ref_curve = reference_summary[
                (reference_summary["policy"] == policy)
                & (reference_summary["tier"] == tier)
                & (reference_summary["top_k"] == PRIMARY_TOP_K)
            ].sort_values("cells")
            seed_curve = seed_summary[
                (seed_summary["policy"] == policy)
                & (seed_summary["tier"] == tier)
                & (seed_summary["top_k"] == PRIMARY_TOP_K)
            ].sort_values("cells")

            if ref_curve.empty or seed_curve.empty:
                continue

            ref_cells = ref_curve["cells"].astype(int).tolist()
            ref_mean = monotone(ref_curve["jaccard_mean"].astype(float).tolist())
            ref_low = monotone(ref_curve["jaccard_p05"].astype(float).tolist())

            seed_cells = seed_curve["cells"].astype(int).tolist()
            seed_mean = monotone(seed_curve["jaccard_mean"].astype(float).tolist())
            seed_low = monotone(seed_curve["jaccard_p05"].astype(float).tolist())

            mvcc_mean = max(
                interpolate_threshold(ref_cells, ref_mean, REF_THRESHOLD),
                interpolate_threshold(seed_cells, seed_mean, SEED_THRESHOLD),
            )
            mvcc_conservative = max(
                interpolate_threshold(ref_cells, ref_low, REF_THRESHOLD),
                interpolate_threshold(seed_cells, seed_low, SEED_THRESHOLD),
            )
            mvcc_strict = max(
                interpolate_threshold(ref_cells, ref_mean, REF_THRESHOLD_STRICT),
                interpolate_threshold(seed_cells, seed_mean, SEED_THRESHOLD_STRICT),
            )

            rows.append(
                {
                    "policy": policy,
                    "tier": tier,
                    "primary_top_k": PRIMARY_TOP_K,
                    "mvcc_mean_cells": mvcc_mean,
                    "mvcc_conservative_cells": mvcc_conservative,
                    "mvcc_strict_cells": mvcc_strict,
                    "mvcc_mean_label": ">3000" if math.isinf(mvcc_mean) else f"{mvcc_mean:.1f}",
                    "mvcc_conservative_label": ">3000" if math.isinf(mvcc_conservative) else f"{mvcc_conservative:.1f}",
                    "mvcc_strict_label": ">3000" if math.isinf(mvcc_strict) else f"{mvcc_strict:.1f}",
                }
            )

    return pd.DataFrame(rows).sort_values(["policy", "tier"]).reset_index(drop=True)


def split_edge(edge: str) -> tuple[str, str]:
    source, target = edge.split("->", 1)
    return source, target


def compute_emergence(
    policy_cache: dict[str, dict[RunKey, RankedEdges]],
    gene_sets: dict[RunKey, set[str]],
    trrust_set: set[tuple[str, str]],
    dorothea_set: set[tuple[str, str]],
    dorothea_sources: set[str],
    dorothea_targets: set[str],
    top_k: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for policy, runs in policy_cache.items():
        for tier in TIER_ORDER:
            k200 = RunKey(tier, 200, 1)
            k1000 = RunKey(tier, 1000, 1)
            k3000 = RunKey(tier, 3000, 1)
            if k200 not in runs or k1000 not in runs or k3000 not in runs:
                continue

            shared_genes = gene_sets[k200] & gene_sets[k1000] & gene_sets[k3000]
            if len(shared_genes) < 2:
                continue

            e200 = top_set(filter_ranked_by_genes(runs[k200], shared_genes), top_k)
            e1000 = top_set(filter_ranked_by_genes(runs[k1000], shared_genes), top_k)
            r3000 = filter_ranked_by_genes(runs[k3000], shared_genes)
            e3000 = top_set(r3000, top_k)

            classes = {
                "persistent_all_counts": e200 & e1000 & e3000,
                "high_cell_only": e3000 - (e200 | e1000),
                "gained_at_1000_vs_200": e1000 - e200,
            }

            for category, edge_set in classes.items():
                n_edges = len(edge_set)
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

                summary_rows.append(
                    {
                        "policy": policy,
                        "tier": tier,
                        "top_k": top_k,
                        "n_shared_genes": len(shared_genes),
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

                if category == "high_cell_only" and n_edges:
                    lookup = r3000.table.set_index("edge")
                    for edge in sorted(edge_set, key=lambda e: lookup.at[e, "score"], reverse=True)[:100]:
                        source, target = split_edge(edge)
                        pair = (source, target)
                        detail_rows.append(
                            {
                                "policy": policy,
                                "tier": tier,
                                "category": category,
                                "source": source,
                                "target": target,
                                "edge": edge,
                                "score_3000": float(lookup.at[edge, "score"]),
                                "rank_3000": int(lookup.at[edge, "rank"]),
                                "supported_trrust": pair in trrust_set,
                                "supported_dorothea": pair in dorothea_set,
                            }
                        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["policy", "tier", "category"])
    detail_df = pd.DataFrame(detail_rows).sort_values(["policy", "tier", "rank_3000"])
    return summary_df, detail_df


def plot_policy_reference(reference_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for i, policy in enumerate(POLICIES):
        ax = axes[i]
        for tier in TIER_ORDER:
            sub = reference_summary[
                (reference_summary["policy"] == policy)
                & (reference_summary["tier"] == tier)
                & (reference_summary["top_k"] == PRIMARY_TOP_K)
            ].sort_values("cells")
            if sub.empty:
                continue
            ax.plot(sub["cells"], sub["jaccard_mean"], marker="o", linewidth=2, label=tier)
            ax.fill_between(sub["cells"], sub["jaccard_p05"], sub["jaccard_p95"], alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(policy)
        ax.set_xlabel("Cells")
        ax.grid(True, which="both", alpha=0.3)
        if i == 0:
            ax.set_ylabel(f"Top-{PRIMARY_TOP_K} Jaccard vs 3000-cell anchors")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mvcc(mvcc: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for r in mvcc.itertuples(index=False):
        rows.append({"policy": r.policy, "tier": r.tier, "type": "mean", "value": r.mvcc_mean_cells})
        rows.append({"policy": r.policy, "tier": r.tier, "type": "conservative", "value": r.mvcc_conservative_cells})

    plot_df = pd.DataFrame(rows)
    plot_df["value_plot"] = plot_df["value"].replace([float("inf")], 3500)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x_labels = []
    x = []
    y = []
    colors = []
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        x.append(idx)
        y.append(row.value_plot)
        x_labels.append(f"{row.policy}\n{row.tier}\n{row.type}")
        colors.append("#1f77b4" if row.type == "mean" else "#ff7f0e")

    ax.bar(x, y, color=colors)
    ax.axhline(3000, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("MVCC (cells); 3500 = >3000")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_title("MVCC estimates after adversarial repairs")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_emergence_support(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["category"] == "high_cell_only"].copy()
    if data.empty:
        return

    labels = []
    t_rates = []
    d_rates = []
    for policy in POLICIES:
        for tier in TIER_ORDER:
            row = data[(data["policy"] == policy) & (data["tier"] == tier)]
            if row.empty:
                continue
            labels.append(f"{policy}\n{tier}")
            t_rates.append(float(row.iloc[0]["trrust_rate"]))
            d_rates.append(float(row.iloc[0]["dorothea_rate"]))

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.bar(x - w / 2, t_rates, width=w, label="TRRUST")
    ax.bar(x + w / 2, d_rates, width=w, label="DoRothEA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Support rate")
    ax.set_title("High-cell-only edge support by policy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def write_summary(
    inventory: pd.DataFrame,
    reference_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mvcc: pd.DataFrame,
    emergence: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Adversarial Repair Summary")
    lines.append("")
    lines.append(f"- Runs analyzed: {len(inventory)}")
    lines.append("- Policies analyzed: unconstrained, tf_source, tf_source_target")
    lines.append("- Fixes: shared-gene-universe filtering, multi-anchor references, null-calibrated overlap, constrained-policy experiments")
    lines.append("")

    lines.append("## Top-1000 reference stability (mean Jaccard)")
    ref = reference_summary[reference_summary["top_k"] == PRIMARY_TOP_K].sort_values(["policy", "tier", "cells"])
    for row in ref.itertuples(index=False):
        lines.append(
            f"- {row.policy} / {row.tier} / {int(row.cells)} cells: jaccard={row.jaccard_mean:.3f} (p05={row.jaccard_p05:.3f}, p95={row.jaccard_p95:.3f}), enrichment={row.enrichment_mean:.1f}x"
        )
    lines.append("")

    lines.append("## Top-1000 seed robustness (mean Jaccard)")
    seed = seed_summary[seed_summary["top_k"] == PRIMARY_TOP_K].sort_values(["policy", "tier", "cells"])
    for row in seed.itertuples(index=False):
        lines.append(
            f"- {row.policy} / {row.tier} / {int(row.cells)}: jaccard={row.jaccard_mean:.3f} (p05={row.jaccard_p05:.3f}, p95={row.jaccard_p95:.3f}), enrichment={row.enrichment_mean:.1f}x"
        )
    lines.append("")

    lines.append("## MVCC")
    for row in mvcc.itertuples(index=False):
        lines.append(
            f"- {row.policy} / {row.tier}: mean={row.mvcc_mean_label}, conservative={row.mvcc_conservative_label}, strict={row.mvcc_strict_label}"
        )
    lines.append("")

    lines.append("## High-cell-only support")
    high = emergence[emergence["category"] == "high_cell_only"].sort_values(["policy", "tier"])
    for row in high.itertuples(index=False):
        lines.append(
            f"- {row.policy} / {row.tier}: n={int(row.n_edges)}, TRRUST={0.0 if pd.isna(row.trrust_rate) else row.trrust_rate:.4f}, DoRothEA={0.0 if pd.isna(row.dorothea_rate) else row.dorothea_rate:.4f}, TF-source={0.0 if pd.isna(row.source_tf_rate) else row.source_tf_rate:.4f}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()

    runs = load_inventory()
    alias_map = load_hgnc_alias_map(SINGLE_CELL_ROOT / "external" / "hgnc_complete_set.txt")
    trrust_set, dorothea_set, dorothea_sources, dorothea_targets = load_truth(alias_map)

    policy_cache, gene_sets, inventory_df = build_policy_caches(
        runs=runs,
        alias_map=alias_map,
        dorothea_sources=dorothea_sources,
        dorothea_targets=dorothea_targets,
    )

    ref_topk, ref_rank, ref_summary = compute_reference_stability(policy_cache, gene_sets)
    seed_topk, seed_rank, seed_summary = compute_seed_robustness(policy_cache, gene_sets)
    mvcc = estimate_mvcc(ref_summary, seed_summary)
    emergence_summary, emergence_detail = compute_emergence(
        policy_cache=policy_cache,
        gene_sets=gene_sets,
        trrust_set=trrust_set,
        dorothea_set=dorothea_set,
        dorothea_sources=dorothea_sources,
        dorothea_targets=dorothea_targets,
        top_k=5000,
    )

    inventory_df.to_csv(OUTPUT_DIR / "run_inventory.csv", index=False)
    ref_topk.to_csv(OUTPUT_DIR / "reference_topk_pairs.csv", index=False)
    ref_rank.to_csv(OUTPUT_DIR / "reference_rank_pairs.csv", index=False)
    ref_summary.to_csv(OUTPUT_DIR / "reference_topk_summary.csv", index=False)

    seed_topk.to_csv(OUTPUT_DIR / "seed_topk_pairs.csv", index=False)
    seed_rank.to_csv(OUTPUT_DIR / "seed_rank_pairs.csv", index=False)
    seed_summary.to_csv(OUTPUT_DIR / "seed_topk_summary.csv", index=False)

    mvcc.to_csv(OUTPUT_DIR / "mvcc_summary.csv", index=False)
    emergence_summary.to_csv(OUTPUT_DIR / "emergence_summary.csv", index=False)
    emergence_detail.to_csv(OUTPUT_DIR / "high_cell_only_top100_detail.csv", index=False)

    write_summary(
        inventory=inventory_df,
        reference_summary=ref_summary,
        seed_summary=seed_summary,
        mvcc=mvcc,
        emergence=emergence_summary,
        out_path=OUTPUT_DIR / "adversarial_repair_summary.md",
    )

    plot_policy_reference(ref_summary, FIG_DIR / "fig5_adversarial_policy_reference_stability.png")
    plot_mvcc(mvcc, FIG_DIR / "fig6_adversarial_mvcc.png")
    plot_emergence_support(emergence_summary, FIG_DIR / "fig7_adversarial_high_cell_support.png")

    print("Adversarial repair analysis complete")
    print(f"Runs analyzed: {len(inventory_df)}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
