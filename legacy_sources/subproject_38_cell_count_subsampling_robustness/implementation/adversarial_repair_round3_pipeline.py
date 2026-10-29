#!/usr/bin/env python3
"""Adversarial repair round-3 pipeline for subproject 38.

Fixes applied versus the initial analysis:
1. Shared-gene-universe filtering for all cross-cell/seed comparisons.
2. Multi-anchor reference evaluation (all available 3000-cell seeds).
3. Null-calibrated overlap enrichment against random overlap expectation.
4. New constrained-policy experiments (TF-source and TF-source+target).

Round-2 additions:
5. Random size-matched constraint controls to disentangle prior knowledge from search-space size.
6. Benjamini-Hochberg correction for overlap p-values.
7. Multi-trajectory emergence analysis across all available seed combinations.

Round-3 additions:
8. Multi-draw random-control ensembles (single-draw fragility fix).
9. Empirical prior-vs-random null tests with uncertainty intervals.
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
OUTPUT_DIR = SUBPROJECT_ROOT / "implementation" / "outputs_adversarial_round3"
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
POLICY_RANDOM_SOURCE = "random_source"
POLICY_RANDOM_SOURCE_TARGET = "random_source_target"
POLICIES = [
    POLICY_UNCONSTRAINED,
    POLICY_TF_SOURCE,
    POLICY_TF_SOURCE_TARGET,
    POLICY_RANDOM_SOURCE,
    POLICY_RANDOM_SOURCE_TARGET,
]

RANDOM_POLICY_SEED = 20260218
RANDOM_ENSEMBLE_DRAWS = 64


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


def bh_fdr(series: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction with NaN-safe handling."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(values))[0]
    if valid_idx.size == 0:
        return pd.Series(out, index=series.index, dtype=float)

    p = values[valid_idx]
    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    q_ranked = ranked * m / (np.arange(1, m + 1, dtype=float))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_unsorted = np.empty_like(q_ranked)
    q_unsorted[order] = q_ranked
    out[valid_idx] = q_unsorted
    return pd.Series(out, index=series.index, dtype=float)


def add_overlap_qvalues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["q_overlap_global"] = bh_fdr(out["p_overlap_geq"])
    out["q_overlap_policy_topk"] = np.nan

    grouped = out.groupby(["policy", "top_k"]).groups
    for _, idx in grouped.items():
        idx_list = list(idx)
        out.loc[idx_list, "q_overlap_policy_topk"] = bh_fdr(out.loc[idx_list, "p_overlap_geq"]).to_numpy()
    return out


def sample_random_constraints_for_run(
    gene_names: list[str],
    source_count: int,
    target_count: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    unique_genes = np.array(sorted({g for g in gene_names if g}), dtype=object)
    if source_count <= 0 or target_count <= 0:
        return set(), set()
    if source_count > unique_genes.size or target_count > unique_genes.size:
        return set(), set()

    rng = np.random.default_rng(seed)
    random_sources = set(rng.choice(unique_genes, size=source_count, replace=False).tolist())
    random_targets = set(rng.choice(unique_genes, size=target_count, replace=False).tolist())
    return random_sources, random_targets


def build_policy_caches(
    runs: list[RunRecord],
    alias_map: dict[str, str],
    dorothea_sources: set[str],
    dorothea_targets: set[str],
) -> tuple[
    dict[str, dict[RunKey, RankedEdges]],
    dict[RunKey, set[str]],
    pd.DataFrame,
    dict[RunKey, list[str]],
    dict[RunKey, tuple[int, int]],
]:
    policy_cache: dict[str, dict[RunKey, RankedEdges]] = {policy: {} for policy in POLICIES}
    gene_sets: dict[RunKey, set[str]] = {}
    gene_names_by_run: dict[RunKey, list[str]] = {}
    run_random_sizes: dict[RunKey, tuple[int, int]] = {}

    inventory_rows: list[dict[str, object]] = []

    # First pass: collect normalized gene universes and per-run matched random-control sizes.
    for run in runs:
        with h5py.File(run.processed_h5ad, "r") as handle:
            raw_index = handle["var"]["_index"][:]
        gene_names = [
            normalize_symbol(g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else str(g), alias_map)
            for g in raw_index
        ]
        gene_names_by_run[run.key] = gene_names
        gene_set = set(g for g in gene_names if g)
        gene_sets[run.key] = gene_set
        source_count = len(gene_set & dorothea_sources)
        target_count = len(gene_set & dorothea_targets)
        if source_count <= 0 or target_count <= 0:
            raise RuntimeError(f"Run {run.experiment_id} has zero available DoRothEA-matched source/target counts.")
        run_random_sizes[run.key] = (source_count, target_count)

    # Second pass: infer or load policy-specific ranked edge tables.
    for run_idx, run in enumerate(runs):
        unconstrained = load_unconstrained_ranked(run.network_edges, alias_map)
        policy_cache[POLICY_UNCONSTRAINED][run.key] = unconstrained

        gene_names = gene_names_by_run[run.key]
        source_count, target_count = run_random_sizes[run.key]
        random_sources, random_targets = sample_random_constraints_for_run(
            gene_names=gene_names,
            source_count=source_count,
            target_count=target_count,
            seed=RANDOM_POLICY_SEED + run_idx,
        )
        if not random_sources or not random_targets:
            raise RuntimeError(f"Random constraints failed for run {run.experiment_id}.")

        score_mat = np.load(run.attention_scores, mmap_mode="r")

        policy_constraints = {
            POLICY_TF_SOURCE: (dorothea_sources, None),
            POLICY_TF_SOURCE_TARGET: (dorothea_sources, dorothea_targets),
            POLICY_RANDOM_SOURCE: (random_sources, None),
            POLICY_RANDOM_SOURCE_TARGET: (random_sources, random_targets),
        }
        for policy, (source_allowed, target_allowed) in policy_constraints.items():
            policy_dir = EDGE_CACHE_DIR / "run_matched" / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            cached_path = policy_dir / f"{run.experiment_id}.tsv"

            if cached_path.exists():
                cached_df = pd.read_csv(cached_path, sep="\t")
                ranked = to_ranked(cached_df[["source", "target", "score"]])
            else:
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
                "n_genes": len(gene_sets[run.key]),
                "n_random_source_candidates_run": source_count,
                "n_random_target_candidates_run": target_count,
                "n_edges_unconstrained": len(unconstrained.table),
                "n_edges_tf_source": len(policy_cache[POLICY_TF_SOURCE][run.key].table),
                "n_edges_tf_source_target": len(policy_cache[POLICY_TF_SOURCE_TARGET][run.key].table),
                "n_edges_random_source": len(policy_cache[POLICY_RANDOM_SOURCE][run.key].table),
                "n_edges_random_source_target": len(policy_cache[POLICY_RANDOM_SOURCE_TARGET][run.key].table),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["tier", "cells", "seed"])
    return (
        policy_cache,
        gene_sets,
        inventory_df,
        gene_names_by_run,
        run_random_sizes,
    )


def build_random_ensemble_policy_cache(
    draw_idx: int,
    runs: list[RunRecord],
    gene_names_by_run: dict[RunKey, list[str]],
    run_random_sizes: dict[RunKey, tuple[int, int]],
) -> dict[str, dict[RunKey, RankedEdges]]:
    policy_cache: dict[str, dict[RunKey, RankedEdges]] = {
        POLICY_RANDOM_SOURCE: {},
        POLICY_RANDOM_SOURCE_TARGET: {},
    }

    for run_idx, run in enumerate(runs):
        gene_names = gene_names_by_run[run.key]
        source_count, target_count = run_random_sizes[run.key]
        random_sources, random_targets = sample_random_constraints_for_run(
            gene_names=gene_names,
            source_count=source_count,
            target_count=target_count,
            seed=RANDOM_POLICY_SEED + draw_idx * 10000 + run_idx,
        )
        if not random_sources or not random_targets:
            raise RuntimeError(f"Random ensemble constraints failed for draw {draw_idx}, run {run.experiment_id}.")
        score_mat = np.load(run.attention_scores, mmap_mode="r")

        for policy, (source_allowed, target_allowed) in {
            POLICY_RANDOM_SOURCE: (random_sources, None),
            POLICY_RANDOM_SOURCE_TARGET: (random_sources, random_targets),
        }.items():
            cached_dir = EDGE_CACHE_DIR / "random_ensemble_run_matched" / f"draw_{draw_idx:03d}" / policy
            cached_dir.mkdir(parents=True, exist_ok=True)
            cached_path = cached_dir / f"{run.experiment_id}.tsv"

            if cached_path.exists():
                cached_df = pd.read_csv(cached_path, sep="\t")
                ranked = to_ranked(cached_df[["source", "target", "score"]])
            else:
                ranked = infer_constrained_edges(
                    scores=score_mat,
                    gene_names=gene_names,
                    source_allowed=source_allowed,
                    target_allowed=target_allowed,
                    top_k_per_source=100,
                )
                ranked.table[["source", "target", "score"]].to_csv(cached_path, sep="\t", index=False)
            policy_cache[policy][run.key] = ranked

    return policy_cache


def compute_random_ensemble_outputs(
    runs: list[RunRecord],
    gene_names_by_run: dict[RunKey, list[str]],
    run_random_sizes: dict[RunKey, tuple[int, int]],
    gene_sets: dict[RunKey, set[str]],
    trrust_set: set[tuple[str, str]],
    dorothea_set: set[tuple[str, str]],
    dorothea_sources: set[str],
    dorothea_targets: set[str],
    n_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref_rows: list[pd.DataFrame] = []
    seed_rows: list[pd.DataFrame] = []
    emergence_rows: list[pd.DataFrame] = []

    for draw_idx in range(1, n_draws + 1):
        draw_cache = build_random_ensemble_policy_cache(
            draw_idx=draw_idx,
            runs=runs,
            gene_names_by_run=gene_names_by_run,
            run_random_sizes=run_random_sizes,
        )
        _, _, ref_summary = compute_reference_stability(draw_cache, gene_sets)
        _, _, seed_summary = compute_seed_robustness(draw_cache, gene_sets)
        _, emergence_summary, _ = compute_emergence(
            policy_cache=draw_cache,
            gene_sets=gene_sets,
            trrust_set=trrust_set,
            dorothea_set=dorothea_set,
            dorothea_sources=dorothea_sources,
            dorothea_targets=dorothea_targets,
            top_k=5000,
        )

        ref_frame = ref_summary[ref_summary["top_k"] == PRIMARY_TOP_K].copy()
        ref_frame["random_draw"] = draw_idx
        ref_rows.append(ref_frame)

        seed_frame = seed_summary[seed_summary["top_k"] == PRIMARY_TOP_K].copy()
        seed_frame["random_draw"] = draw_idx
        seed_rows.append(seed_frame)

        em_frame = emergence_summary[emergence_summary["category"] == "high_cell_only"].copy()
        em_frame["random_draw"] = draw_idx
        emergence_rows.append(em_frame)

    return (
        pd.concat(ref_rows, ignore_index=True) if ref_rows else pd.DataFrame(),
        pd.concat(seed_rows, ignore_index=True) if seed_rows else pd.DataFrame(),
        pd.concat(emergence_rows, ignore_index=True) if emergence_rows else pd.DataFrame(),
    )


def empirical_pvalue(null_values: np.ndarray, observed: float) -> float:
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(null_values >= observed)) / (null_values.size + 1))


def compute_prior_vs_random_empirical_tests(
    reference_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    emergence_summary: pd.DataFrame,
    random_ref: pd.DataFrame,
    random_seed: pd.DataFrame,
    random_emergence: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [
        (POLICY_TF_SOURCE, POLICY_RANDOM_SOURCE),
        (POLICY_TF_SOURCE_TARGET, POLICY_RANDOM_SOURCE_TARGET),
    ]

    ref_obs = reference_summary[reference_summary["top_k"] == PRIMARY_TOP_K]
    seed_obs = seed_summary[seed_summary["top_k"] == PRIMARY_TOP_K]
    emergence_obs = emergence_summary[emergence_summary["category"] == "high_cell_only"]

    for bio_policy, random_policy in pairs:
        for tier in TIER_ORDER:
            for cells in CELL_ORDER:
                obs_ref = ref_obs[
                    (ref_obs["policy"] == bio_policy) & (ref_obs["tier"] == tier) & (ref_obs["cells"] == cells)
                ]
                null_ref = random_ref[
                    (random_ref["policy"] == random_policy)
                    & (random_ref["tier"] == tier)
                    & (random_ref["cells"] == cells)
                ]["jaccard_mean"].to_numpy(dtype=float)
                if obs_ref.empty:
                    continue
                observed_ref = float(obs_ref.iloc[0]["jaccard_mean"])

                obs_seed = seed_obs[
                    (seed_obs["policy"] == bio_policy) & (seed_obs["tier"] == tier) & (seed_obs["cells"] == cells)
                ]
                null_seed = random_seed[
                    (random_seed["policy"] == random_policy)
                    & (random_seed["tier"] == tier)
                    & (random_seed["cells"] == cells)
                ]["jaccard_mean"].to_numpy(dtype=float)
                observed_seed = float(obs_seed.iloc[0]["jaccard_mean"]) if not obs_seed.empty else float("nan")

                rows.append(
                    {
                        "comparison": f"{bio_policy}_vs_{random_policy}",
                        "tier": tier,
                        "cells": cells,
                        "metric": "reference_top1000_jaccard",
                        "observed": observed_ref,
                        "null_mean": float(np.nanmean(null_ref)) if null_ref.size else float("nan"),
                        "null_p05": float(np.nanquantile(null_ref, 0.05)) if null_ref.size else float("nan"),
                        "null_p95": float(np.nanquantile(null_ref, 0.95)) if null_ref.size else float("nan"),
                        "delta_vs_null_mean": observed_ref - (float(np.nanmean(null_ref)) if null_ref.size else float("nan")),
                        "empirical_p_geq_null": empirical_pvalue(null_ref[np.isfinite(null_ref)], observed_ref),
                        "n_null_draws": int(np.sum(np.isfinite(null_ref))),
                    }
                )

                rows.append(
                    {
                        "comparison": f"{bio_policy}_vs_{random_policy}",
                        "tier": tier,
                        "cells": cells,
                        "metric": "seed_top1000_jaccard",
                        "observed": observed_seed,
                        "null_mean": float(np.nanmean(null_seed)) if null_seed.size else float("nan"),
                        "null_p05": float(np.nanquantile(null_seed, 0.05)) if null_seed.size else float("nan"),
                        "null_p95": float(np.nanquantile(null_seed, 0.95)) if null_seed.size else float("nan"),
                        "delta_vs_null_mean": observed_seed - (float(np.nanmean(null_seed)) if null_seed.size else float("nan")),
                        "empirical_p_geq_null": empirical_pvalue(null_seed[np.isfinite(null_seed)], observed_seed),
                        "n_null_draws": int(np.sum(np.isfinite(null_seed))),
                    }
                )

            obs_em = emergence_obs[
                (emergence_obs["policy"] == bio_policy) & (emergence_obs["tier"] == tier)
            ]
            null_em = random_emergence[
                (random_emergence["policy"] == random_policy) & (random_emergence["tier"] == tier)
            ]["dorothea_rate_mean"].to_numpy(dtype=float)
            if obs_em.empty:
                continue
            observed_em = float(obs_em.iloc[0]["dorothea_rate_mean"])
            rows.append(
                {
                    "comparison": f"{bio_policy}_vs_{random_policy}",
                    "tier": tier,
                    "cells": "high_cell_only",
                    "metric": "high_cell_dorothea_support",
                    "observed": observed_em,
                    "null_mean": float(np.nanmean(null_em)) if null_em.size else float("nan"),
                    "null_p05": float(np.nanquantile(null_em, 0.05)) if null_em.size else float("nan"),
                    "null_p95": float(np.nanquantile(null_em, 0.95)) if null_em.size else float("nan"),
                    "delta_vs_null_mean": observed_em - (float(np.nanmean(null_em)) if null_em.size else float("nan")),
                    "empirical_p_geq_null": empirical_pvalue(null_em[np.isfinite(null_em)], observed_em),
                    "n_null_draws": int(np.sum(np.isfinite(null_em))),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["cells_sort"] = out["cells"].astype(str)
    out = out.sort_values(["comparison", "tier", "metric", "cells_sort"]).drop(columns=["cells_sort"]).reset_index(drop=True)
    out["empirical_q_geq_null"] = bh_fdr(out["empirical_p_geq_null"])
    return out


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

    topk_df = add_overlap_qvalues(pd.DataFrame(topk_rows))
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
            q_overlap_policy_topk_median=("q_overlap_policy_topk", "median"),
            q_overlap_global_median=("q_overlap_global", "median"),
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

    topk_df = add_overlap_qvalues(pd.DataFrame(topk_rows))
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
            q_overlap_policy_topk_median=("q_overlap_policy_topk", "median"),
            q_overlap_global_median=("q_overlap_global", "median"),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for policy, runs in policy_cache.items():
        for tier in TIER_ORDER:
            keys_200 = sorted([k for k in runs if k.tier == tier and k.cells == 200], key=lambda x: x.seed)
            keys_1000 = sorted([k for k in runs if k.tier == tier and k.cells == 1000], key=lambda x: x.seed)
            keys_3000 = sorted([k for k in runs if k.tier == tier and k.cells == 3000], key=lambda x: x.seed)
            if not keys_200 or not keys_1000 or not keys_3000:
                continue

            for key_200 in keys_200:
                for key_1000 in keys_1000:
                    for key_3000 in keys_3000:
                        shared_genes = gene_sets[key_200] & gene_sets[key_1000] & gene_sets[key_3000]
                        if len(shared_genes) < 2:
                            continue

                        ranked_200 = filter_ranked_by_genes(runs[key_200], shared_genes)
                        ranked_1000 = filter_ranked_by_genes(runs[key_1000], shared_genes)
                        ranked_3000 = filter_ranked_by_genes(runs[key_3000], shared_genes)

                        e200 = top_set(ranked_200, top_k)
                        e1000 = top_set(ranked_1000, top_k)
                        e3000 = top_set(ranked_3000, top_k)
                        trajectory_id = f"c200s{key_200.seed}_c1000s{key_1000.seed}_c3000s{key_3000.seed}"

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

                            trajectory_rows.append(
                                {
                                    "policy": policy,
                                    "tier": tier,
                                    "top_k": top_k,
                                    "trajectory_id": trajectory_id,
                                    "seed_200": key_200.seed,
                                    "seed_1000": key_1000.seed,
                                    "seed_3000": key_3000.seed,
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
                                lookup = ranked_3000.table.set_index("edge")
                                for edge in sorted(edge_set, key=lambda e: lookup.at[e, "score"], reverse=True)[:100]:
                                    source, target = split_edge(edge)
                                    pair = (source, target)
                                    detail_rows.append(
                                        {
                                            "policy": policy,
                                            "tier": tier,
                                            "trajectory_id": trajectory_id,
                                            "seed_200": key_200.seed,
                                            "seed_1000": key_1000.seed,
                                            "seed_3000": key_3000.seed,
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

    trajectory_df = pd.DataFrame(trajectory_rows)
    if trajectory_df.empty:
        return trajectory_df, pd.DataFrame(), pd.DataFrame(detail_rows)

    summary_df = (
        trajectory_df.groupby(["policy", "tier", "category"], as_index=False)
        .agg(
            n_trajectories=("trajectory_id", "nunique"),
            n_edges_mean=("n_edges", "mean"),
            n_edges_p05=("n_edges", lambda s: quantile_or_nan(s, 0.05)),
            n_edges_p95=("n_edges", lambda s: quantile_or_nan(s, 0.95)),
            trrust_rate_mean=("trrust_rate", "mean"),
            trrust_rate_p05=("trrust_rate", lambda s: quantile_or_nan(s, 0.05)),
            trrust_rate_p95=("trrust_rate", lambda s: quantile_or_nan(s, 0.95)),
            dorothea_rate_mean=("dorothea_rate", "mean"),
            dorothea_rate_p05=("dorothea_rate", lambda s: quantile_or_nan(s, 0.05)),
            dorothea_rate_p95=("dorothea_rate", lambda s: quantile_or_nan(s, 0.95)),
            source_tf_rate_mean=("source_tf_rate", "mean"),
            target_known_rate_mean=("target_known_rate", "mean"),
        )
        .sort_values(["policy", "tier", "category"])
    )

    detail_df = pd.DataFrame(detail_rows).sort_values(["policy", "tier", "trajectory_id", "rank_3000"])
    return trajectory_df, summary_df, detail_df


def plot_policy_reference(reference_summary: pd.DataFrame, out_path: Path) -> None:
    n_cols = 3
    n_rows = int(math.ceil(len(POLICIES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.0, 3.8 * n_rows), sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, policy in enumerate(POLICIES):
        ax = axes_flat[i]
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
        if i % n_cols == 0:
            ax.set_ylabel(f"Top-{PRIMARY_TOP_K} Jaccard vs 3000-cell anchors")
        ax.legend(fontsize=8)

    for j in range(len(POLICIES), len(axes_flat)):
        axes_flat[j].axis("off")
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
            t_rates.append(float(row.iloc[0]["trrust_rate_mean"]))
            d_rates.append(float(row.iloc[0]["dorothea_rate_mean"]))

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


def compute_prior_control_comparison(
    reference_summary: pd.DataFrame,
    emergence_summary: pd.DataFrame,
    random_ensemble_reference: pd.DataFrame,
    random_ensemble_emergence: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policy_pairs = [
        (POLICY_TF_SOURCE, POLICY_RANDOM_SOURCE),
        (POLICY_TF_SOURCE_TARGET, POLICY_RANDOM_SOURCE_TARGET),
    ]

    ref_top = reference_summary[reference_summary["top_k"] == PRIMARY_TOP_K]
    emergence_high = emergence_summary[emergence_summary["category"] == "high_cell_only"]
    random_ref_mean = (
        random_ensemble_reference.groupby(["policy", "tier", "cells"], as_index=False)
        .agg(jaccard_mean=("jaccard_mean", "mean"))
        .sort_values(["policy", "tier", "cells"])
    )
    random_emergence_mean = (
        random_ensemble_emergence.groupby(["policy", "tier"], as_index=False)
        .agg(dorothea_rate_mean=("dorothea_rate_mean", "mean"))
        .sort_values(["policy", "tier"])
    )

    for bio_policy, random_policy in policy_pairs:
        for tier in TIER_ORDER:
            for cells in CELL_ORDER:
                bio_row = ref_top[
                    (ref_top["policy"] == bio_policy)
                    & (ref_top["tier"] == tier)
                    & (ref_top["cells"] == cells)
                ]
                rnd_row = random_ref_mean[
                    (random_ref_mean["policy"] == random_policy)
                    & (random_ref_mean["tier"] == tier)
                    & (random_ref_mean["cells"] == cells)
                ]
                if bio_row.empty or rnd_row.empty:
                    continue

                bio_j = float(bio_row.iloc[0]["jaccard_mean"])
                rnd_j = float(rnd_row.iloc[0]["jaccard_mean"])
                rows.append(
                    {
                        "comparison": f"{bio_policy}_minus_{random_policy}",
                        "policy_bio": bio_policy,
                        "policy_random": random_policy,
                        "tier": tier,
                        "cells": cells,
                        "reference_jaccard_bio": bio_j,
                        "reference_jaccard_random": rnd_j,
                        "reference_jaccard_delta": bio_j - rnd_j,
                    }
                )

            bio_emergence = emergence_high[
                (emergence_high["policy"] == bio_policy) & (emergence_high["tier"] == tier)
            ]
            rnd_emergence = random_emergence_mean[
                (random_emergence_mean["policy"] == random_policy) & (random_emergence_mean["tier"] == tier)
            ]
            if bio_emergence.empty or rnd_emergence.empty:
                continue

            rows.append(
                {
                    "comparison": f"{bio_policy}_minus_{random_policy}",
                    "policy_bio": bio_policy,
                    "policy_random": random_policy,
                    "tier": tier,
                    "cells": "high_cell_only",
                    "reference_jaccard_bio": float("nan"),
                    "reference_jaccard_random": float("nan"),
                    "reference_jaccard_delta": float("nan"),
                    "dorothea_support_bio": float(bio_emergence.iloc[0]["dorothea_rate_mean"]),
                    "dorothea_support_random": float(rnd_emergence.iloc[0]["dorothea_rate_mean"]),
                    "dorothea_support_delta": float(bio_emergence.iloc[0]["dorothea_rate_mean"])
                    - float(rnd_emergence.iloc[0]["dorothea_rate_mean"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["cells_sort"] = out["cells"].astype(str)
    out = out.sort_values(["comparison", "tier", "cells_sort"]).drop(columns=["cells_sort"]).reset_index(drop=True)
    return out


def write_summary(
    inventory: pd.DataFrame,
    reference_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mvcc: pd.DataFrame,
    emergence_trajectory: pd.DataFrame,
    emergence_summary: pd.DataFrame,
    control_comparison: pd.DataFrame,
    random_ensemble_reference: pd.DataFrame,
    empirical_tests: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Adversarial Repair Round-4 Summary")
    lines.append("")
    lines.append(f"- Runs analyzed: {len(inventory)}")
    lines.append(
        "- Policies analyzed: unconstrained, tf_source, tf_source_target, random_source, random_source_target"
    )
    lines.append(
        "- Round-4 additions: per-run matched random controls, expanded random-control ensembles, and empirical prior-vs-random significance tests"
    )
    lines.append("")

    lines.append("## Top-1000 reference stability (mean Jaccard)")
    ref = reference_summary[reference_summary["top_k"] == PRIMARY_TOP_K].sort_values(["policy", "tier", "cells"])
    for row in ref.itertuples(index=False):
        lines.append(
            f"- {row.policy} / {row.tier} / {int(row.cells)} cells: jaccard={row.jaccard_mean:.3f} (p05={row.jaccard_p05:.3f}, p95={row.jaccard_p95:.3f}), enrichment={row.enrichment_mean:.1f}x, median q={row.q_overlap_policy_topk_median:.2e}"
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

    lines.append("## High-cell-only support (multi-trajectory)")
    high = emergence_summary[emergence_summary["category"] == "high_cell_only"].sort_values(["policy", "tier"])
    for row in high.itertuples(index=False):
        lines.append(
            f"- {row.policy} / {row.tier}: trajectories={int(row.n_trajectories)}, n_edges_mean={row.n_edges_mean:.1f}, TRRUST={0.0 if pd.isna(row.trrust_rate_mean) else row.trrust_rate_mean:.4f}, DoRothEA={0.0 if pd.isna(row.dorothea_rate_mean) else row.dorothea_rate_mean:.4f}, TF-source={0.0 if pd.isna(row.source_tf_rate_mean) else row.source_tf_rate_mean:.4f}"
        )
    lines.append("")

    lines.append("## Random-control ensemble (top-1000 reference Jaccard)")
    if random_ensemble_reference.empty:
        lines.append("- No random ensemble outputs were available.")
    else:
        ensemble_summary = (
            random_ensemble_reference.groupby(["policy", "tier", "cells"], as_index=False)
            .agg(
                draw_mean=("jaccard_mean", "mean"),
                draw_p05=("jaccard_mean", lambda s: quantile_or_nan(s, 0.05)),
                draw_p95=("jaccard_mean", lambda s: quantile_or_nan(s, 0.95)),
                n_draws=("jaccard_mean", "count"),
            )
            .sort_values(["policy", "tier", "cells"])
        )
        for row in ensemble_summary.itertuples(index=False):
            lines.append(
                f"- {row.policy} / {row.tier} / {int(row.cells)}: ensemble_mean={row.draw_mean:.3f} (p05={row.draw_p05:.3f}, p95={row.draw_p95:.3f}), draws={int(row.n_draws)}"
            )
    lines.append("")

    lines.append("## Prior-vs-random control deltas")
    if control_comparison.empty:
        lines.append("- No comparable policy-control rows were available.")
    else:
        ref_delta = control_comparison[control_comparison["cells"].isin(CELL_ORDER)]
        for row in ref_delta.itertuples(index=False):
            lines.append(
                f"- {row.comparison} / {row.tier} / {int(row.cells)} cells: delta_jaccard={row.reference_jaccard_delta:+.3f}"
            )
        support_delta = control_comparison[control_comparison["cells"] == "high_cell_only"]
        for row in support_delta.itertuples(index=False):
            lines.append(
                f"- {row.comparison} / {row.tier} / high_cell_only: delta_dorothea_support={row.dorothea_support_delta:+.4f}"
            )
    lines.append("")

    lines.append("## Empirical Prior-vs-Random Tests")
    if empirical_tests.empty:
        lines.append("- No empirical tests were available.")
    else:
        for row in empirical_tests.itertuples(index=False):
            lines.append(
                f"- {row.comparison} / {row.tier} / {row.cells} / {row.metric}: observed={row.observed:.4f}, null_mean={row.null_mean:.4f}, delta={row.delta_vs_null_mean:+.4f}, p={row.empirical_p_geq_null:.4f}, q={row.empirical_q_geq_null:.4f}, n={int(row.n_null_draws)}"
            )
    lines.append("")

    lines.append("## Emergence trajectory coverage")
    trajectory_counts = (
        emergence_trajectory.groupby(["policy", "tier"], as_index=False)["trajectory_id"].nunique().sort_values(["policy", "tier"])
    )
    for row in trajectory_counts.itertuples(index=False):
        lines.append(f"- {row.policy} / {row.tier}: trajectories={int(row.trajectory_id)}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()

    runs = load_inventory()
    alias_map = load_hgnc_alias_map(SINGLE_CELL_ROOT / "external" / "hgnc_complete_set.txt")
    trrust_set, dorothea_set, dorothea_sources, dorothea_targets = load_truth(alias_map)

    (
        policy_cache,
        gene_sets,
        inventory_df,
        gene_names_by_run,
        run_random_sizes,
    ) = build_policy_caches(
        runs=runs,
        alias_map=alias_map,
        dorothea_sources=dorothea_sources,
        dorothea_targets=dorothea_targets,
    )

    ref_topk, ref_rank, ref_summary = compute_reference_stability(policy_cache, gene_sets)
    seed_topk, seed_rank, seed_summary = compute_seed_robustness(policy_cache, gene_sets)
    mvcc = estimate_mvcc(ref_summary, seed_summary)
    emergence_trajectory, emergence_summary, emergence_detail = compute_emergence(
        policy_cache=policy_cache,
        gene_sets=gene_sets,
        trrust_set=trrust_set,
        dorothea_set=dorothea_set,
        dorothea_sources=dorothea_sources,
        dorothea_targets=dorothea_targets,
        top_k=5000,
    )
    random_ref_ensemble, random_seed_ensemble, random_emergence_ensemble = compute_random_ensemble_outputs(
        runs=runs,
        gene_names_by_run=gene_names_by_run,
        run_random_sizes=run_random_sizes,
        gene_sets=gene_sets,
        trrust_set=trrust_set,
        dorothea_set=dorothea_set,
        dorothea_sources=dorothea_sources,
        dorothea_targets=dorothea_targets,
        n_draws=RANDOM_ENSEMBLE_DRAWS,
    )
    empirical_tests = compute_prior_vs_random_empirical_tests(
        reference_summary=ref_summary,
        seed_summary=seed_summary,
        emergence_summary=emergence_summary,
        random_ref=random_ref_ensemble,
        random_seed=random_seed_ensemble,
        random_emergence=random_emergence_ensemble,
    )
    control_comparison = compute_prior_control_comparison(
        reference_summary=ref_summary,
        emergence_summary=emergence_summary,
        random_ensemble_reference=random_ref_ensemble,
        random_ensemble_emergence=random_emergence_ensemble,
    )

    inventory_df.to_csv(OUTPUT_DIR / "run_inventory.csv", index=False)
    ref_topk.to_csv(OUTPUT_DIR / "reference_topk_pairs.csv", index=False)
    ref_rank.to_csv(OUTPUT_DIR / "reference_rank_pairs.csv", index=False)
    ref_summary.to_csv(OUTPUT_DIR / "reference_topk_summary.csv", index=False)

    seed_topk.to_csv(OUTPUT_DIR / "seed_topk_pairs.csv", index=False)
    seed_rank.to_csv(OUTPUT_DIR / "seed_rank_pairs.csv", index=False)
    seed_summary.to_csv(OUTPUT_DIR / "seed_topk_summary.csv", index=False)

    mvcc.to_csv(OUTPUT_DIR / "mvcc_summary.csv", index=False)
    emergence_trajectory.to_csv(OUTPUT_DIR / "emergence_trajectory.csv", index=False)
    emergence_summary.to_csv(OUTPUT_DIR / "emergence_summary.csv", index=False)
    emergence_detail.to_csv(OUTPUT_DIR / "high_cell_only_top100_detail.csv", index=False)
    control_comparison.to_csv(OUTPUT_DIR / "prior_vs_random_control_comparison.csv", index=False)
    random_ref_ensemble.to_csv(OUTPUT_DIR / "random_ensemble_reference_top1000.csv", index=False)
    random_seed_ensemble.to_csv(OUTPUT_DIR / "random_ensemble_seed_top1000.csv", index=False)
    random_emergence_ensemble.to_csv(OUTPUT_DIR / "random_ensemble_emergence_high_cell_only.csv", index=False)
    empirical_tests.to_csv(OUTPUT_DIR / "prior_vs_random_empirical_tests.csv", index=False)

    write_summary(
        inventory=inventory_df,
        reference_summary=ref_summary,
        seed_summary=seed_summary,
        mvcc=mvcc,
        emergence_trajectory=emergence_trajectory,
        emergence_summary=emergence_summary,
        control_comparison=control_comparison,
        random_ensemble_reference=random_ref_ensemble,
        empirical_tests=empirical_tests,
        out_path=OUTPUT_DIR / "adversarial_round4_summary.md",
    )

    plot_policy_reference(ref_summary, FIG_DIR / "fig14_round4_policy_reference_stability.png")
    plot_mvcc(mvcc, FIG_DIR / "fig15_round4_mvcc.png")
    plot_emergence_support(emergence_summary, FIG_DIR / "fig16_round4_high_cell_support.png")

    print("Adversarial round-4 repair analysis complete")
    print(f"Runs analyzed: {len(inventory_df)}")
    print(f"Random-control draws: {RANDOM_ENSEMBLE_DRAWS}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
