#!/usr/bin/env python3
"""Run cluster-resolution sensitivity analysis for mechanistic edge rankings.

This script compares inferred edge rankings when grouping the same cells at
coarse granularity (`broad_cell_class`) versus fine granularity
(`scvi_leiden_res05_*`).

Outputs:
- CSV scorecards and detailed drift tables under `implementation/results/`
- Publication figures under `reports/figures/`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "implementation" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


DATASET_CONFIGS = [
    {
        "dataset": "kidney",
        "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/kidney/processed.h5ad",
        "coarse_key": "broad_cell_class",
        "fine_key": "scvi_leiden_res05_tissue",
    },
    {
        "dataset": "lung",
        "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad",
        "coarse_key": "broad_cell_class",
        "fine_key": "scvi_leiden_res05_tissue",
    },
    {
        "dataset": "immune",
        "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad",
        "coarse_key": "broad_cell_class",
        "fine_key": "scvi_leiden_res05_compartment",
    },
    {
        "dataset": "immune_invariant",
        "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/immune/processed.h5ad",
        "coarse_key": "broad_cell_class",
        "fine_key": "scvi_leiden_res05_compartment",
    },
    {
        "dataset": "external_lung",
        "path": "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/external_lung/processed.h5ad",
        "coarse_key": "compartment",
        "fine_key": "cell_type",
    },
]


@dataclass
class LoadedDataset:
    dataset: str
    x: sparse.csr_matrix
    genes: np.ndarray
    coarse_labels: np.ndarray
    fine_labels: np.ndarray


@dataclass
class RankingResult:
    score_matrix: np.ndarray
    edge_order: np.ndarray
    rank_vector: np.ndarray
    top_group_summary: List[Tuple[str, int]]


def make_hierarchical_fine_labels(
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
) -> np.ndarray:
    if coarse_labels.shape[0] != fine_labels.shape[0]:
        raise ValueError("Coarse and fine label vectors must have the same length.")
    out = np.empty(coarse_labels.shape[0], dtype=object)
    for idx, (coarse, fine) in enumerate(zip(coarse_labels, fine_labels)):
        out[idx] = f"{coarse}:::{fine}"
    return out


def permute_fine_labels_for_null(
    fine_labels: np.ndarray,
    coarse_labels: np.ndarray,
    null_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if null_mode == "global":
        return fine_labels[rng.permutation(fine_labels.shape[0])]

    if null_mode == "within_coarse":
        perm_labels = fine_labels.copy()
        for coarse_label in np.unique(coarse_labels):
            idx = np.where(coarse_labels == coarse_label)[0]
            if idx.size <= 1:
                continue
            perm_labels[idx] = fine_labels[idx][rng.permutation(idx.size)]
        return perm_labels

    raise ValueError(f"Unsupported null_mode={null_mode}")


def _decode_bytes(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        decoded = []
        for item in values:
            if isinstance(item, bytes):
                decoded.append(item.decode("utf-8", errors="ignore"))
            else:
                decoded.append(str(item))
        return np.asarray(decoded, dtype=object)
    return values.astype(object)


def _read_obs_or_var_column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]

    # h5ad stores categorical columns as a group with integer codes + categories.
    # We rebuild the original string vector by indexing categories with codes.
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"][()], dtype=np.int64)
        categories = _decode_bytes(np.asarray(node["categories"][()]))
        out = np.empty(codes.shape[0], dtype=object)
        out[:] = "__missing__"
        valid = codes >= 0
        out[valid] = categories[codes[valid]]
        return out

    values = np.asarray(node[()])
    return _decode_bytes(values)


def _read_gene_names(var_group: h5py.Group) -> np.ndarray:
    if "feature_name" in var_group:
        names = _read_obs_or_var_column(var_group, "feature_name")
    elif "_index" in var_group:
        names = _read_obs_or_var_column(var_group, "_index")
    else:
        raise KeyError("Could not find feature names in var group.")

    # Keep names unique to avoid ambiguous edge identifiers in downstream tables.
    seen: Dict[str, int] = {}
    unique_names = []
    for name in names:
        key = str(name)
        if key not in seen:
            seen[key] = 0
            unique_names.append(key)
        else:
            seen[key] += 1
            unique_names.append(f"{key}__dup{seen[key]}")
    return np.asarray(unique_names, dtype=object)


def load_h5ad_dataset(
    dataset_name: str,
    path: str,
    coarse_key: str,
    fine_key: str,
) -> LoadedDataset:
    with h5py.File(path, "r") as f:
        x_group = f["X"]
        if not isinstance(x_group, h5py.Group) or "data" not in x_group:
            raise ValueError(f"{path} does not contain a CSR-backed X matrix.")

        shape = tuple(int(v) for v in x_group.attrs["shape"])
        x = sparse.csr_matrix(
            (
                np.asarray(x_group["data"][()], dtype=np.float32),
                np.asarray(x_group["indices"][()], dtype=np.int64),
                np.asarray(x_group["indptr"][()], dtype=np.int64),
            ),
            shape=shape,
        )

        obs = f["obs"]
        coarse_labels = _read_obs_or_var_column(obs, coarse_key)
        fine_labels = _read_obs_or_var_column(obs, fine_key)
        genes = _read_gene_names(f["var"])

    return LoadedDataset(
        dataset=dataset_name,
        x=x,
        genes=genes,
        coarse_labels=coarse_labels,
        fine_labels=fine_labels,
    )


def rank_high_variance_genes(
    x: sparse.csr_matrix,
    min_detect_frac: float,
) -> np.ndarray:
    n_cells = x.shape[0]
    detect_counts = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float64)
    detect_frac = detect_counts / max(float(n_cells), 1.0)

    mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float64)
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel().astype(np.float64)
    var = np.maximum(mean_sq - np.square(mean), 0.0)

    keep = np.isfinite(var) & (detect_frac >= min_detect_frac)
    candidate_idx = np.where(keep)[0]
    if candidate_idx.size == 0:
        raise ValueError("No genes passed detection/variance filters.")

    ranked = candidate_idx[np.argsort(var[candidate_idx])[::-1]]
    return ranked


def rank_vector_from_score_matrix(score: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(score.shape[0], 1)
    edge_scores = score[iu]
    edge_order = np.argsort(edge_scores)[::-1]
    rank_vector = np.empty(edge_order.shape[0], dtype=np.int64)
    rank_vector[edge_order] = np.arange(1, edge_order.shape[0] + 1, dtype=np.int64)
    return rank_vector


def _standardize_columns(x_dense: np.ndarray) -> np.ndarray:
    centered = x_dense - x_dense.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1, keepdims=True)
    valid = std[0] > 1e-8

    standardized = np.zeros_like(centered, dtype=np.float32)
    standardized[:, valid] = centered[:, valid] / std[:, valid]
    return standardized


def abs_correlation(x_dense: np.ndarray) -> np.ndarray:
    if x_dense.shape[0] < 3:
        out = np.zeros((x_dense.shape[1], x_dense.shape[1]), dtype=np.float32)
        np.fill_diagonal(out, 0.0)
        return out

    z = _standardize_columns(x_dense.astype(np.float32, copy=False))
    corr = (z.T @ z) / float(max(z.shape[0] - 1, 1))
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def grouped_max_abs_correlation(
    x_dense: np.ndarray,
    labels: np.ndarray,
    min_cells: int,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    summary = sorted(
        [(str(label), int(count)) for label, count in zip(unique_labels, counts)],
        key=lambda t: t[1],
        reverse=True,
    )

    agg = np.zeros((x_dense.shape[1], x_dense.shape[1]), dtype=np.float32)
    used_groups = []
    for label, count in summary:
        if count < min_cells:
            continue
        mask = labels == label
        corr = abs_correlation(x_dense[mask])
        agg = np.maximum(agg, corr)
        used_groups.append((label, count))
    np.fill_diagonal(agg, 0.0)
    return agg, used_groups


def infer_rankings(
    x_dense: np.ndarray,
    labels: np.ndarray,
    min_cells_group: int,
    global_weight: float,
) -> RankingResult:
    global_corr = abs_correlation(x_dense)
    group_corr, used_groups = grouped_max_abs_correlation(
        x_dense, labels, min_cells=min_cells_group
    )
    score = (global_weight * global_corr) + ((1.0 - global_weight) * group_corr)
    np.fill_diagonal(score, 0.0)

    iu = np.triu_indices(score.shape[0], 1)
    edge_scores = score[iu]
    edge_order = np.argsort(edge_scores)[::-1]
    rank_vector = rank_vector_from_score_matrix(score)

    return RankingResult(
        score_matrix=score,
        edge_order=edge_order,
        rank_vector=rank_vector,
        top_group_summary=used_groups,
    )


def compare_rankings(
    coarse_ranks: np.ndarray,
    fine_ranks: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    n_edges = coarse_ranks.shape[0]
    k = int(min(top_k, n_edges))
    coarse_top = set(np.argsort(coarse_ranks)[:k])
    fine_top = set(np.argsort(fine_ranks)[:k])

    shared = coarse_top & fine_top
    union = coarse_top | fine_top

    if shared:
        deltas = np.abs(
            coarse_ranks[np.fromiter(shared, dtype=np.int64)]
            - fine_ranks[np.fromiter(shared, dtype=np.int64)]
        )
        median_drift = float(np.median(deltas) / max(k, 1))
    else:
        median_drift = 1.0

    if len(shared) >= 3:
        shared_idx = np.fromiter(shared, dtype=np.int64)
        rho = float(
            spearmanr(coarse_ranks[shared_idx], fine_ranks[shared_idx]).correlation
        )
        if math.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0

    overlap = float(len(shared) / max(k, 1))
    jaccard = float(len(shared) / max(len(union), 1))

    sensitivity = float(
        (0.40 * (1.0 - overlap))
        + (0.30 * (1.0 - jaccard))
        + (0.30 * median_drift)
    )

    return {
        "topk_overlap": overlap,
        "jaccard": jaccard,
        "median_rank_drift": median_drift,
        "spearman_shared_topk": rho,
        "resolution_sensitivity": sensitivity,
        "k": float(k),
        "shared_edges": float(len(shared)),
    }


def build_edge_drift_rows(
    genes: np.ndarray,
    coarse_rank_vector: np.ndarray,
    fine_rank_vector: np.ndarray,
    iu: Tuple[np.ndarray, np.ndarray],
    top_n_union: int,
) -> List[Dict[str, object]]:
    coarse_top = set(np.argsort(coarse_rank_vector)[:top_n_union])
    fine_top = set(np.argsort(fine_rank_vector)[:top_n_union])
    union_idx = sorted(coarse_top | fine_top)

    rows: List[Dict[str, object]] = []
    for edge_idx in union_idx:
        gi = int(iu[0][edge_idx])
        gj = int(iu[1][edge_idx])
        c_rank = int(coarse_rank_vector[edge_idx])
        f_rank = int(fine_rank_vector[edge_idx])
        rows.append(
            {
                "edge_idx": edge_idx,
                "gene_a": str(genes[gi]),
                "gene_b": str(genes[gj]),
                "coarse_rank": c_rank,
                "fine_rank": f_rank,
                "rank_delta_abs": abs(c_rank - f_rank),
            }
        )

    rows.sort(key=lambda r: int(r["rank_delta_abs"]), reverse=True)
    return rows


def summarize_sensitive_gene_families(
    drift_rows: Sequence[Dict[str, object]],
    delta_threshold: int,
    min_incident_edges: int,
    top_n: int,
) -> List[Dict[str, object]]:
    stats: Dict[str, Dict[str, float]] = {}

    for row in drift_rows:
        delta = int(row["rank_delta_abs"])
        if delta < delta_threshold:
            continue

        for gene_key in ("gene_a", "gene_b"):
            gene = str(row[gene_key])
            rec = stats.setdefault(gene, {"n_edges": 0.0, "delta_sum": 0.0})
            rec["n_edges"] += 1.0
            rec["delta_sum"] += float(delta)

    rows: List[Dict[str, object]] = []
    for gene, rec in stats.items():
        n_edges = int(rec["n_edges"])
        if n_edges < min_incident_edges:
            continue
        mean_delta = float(rec["delta_sum"] / max(rec["n_edges"], 1.0))
        rows.append(
            {
                "gene_family_hub": gene,
                "sensitive_edge_count": n_edges,
                "mean_rank_delta": mean_delta,
                "family_sensitivity_score": mean_delta * math.sqrt(float(n_edges)),
            }
        )

    rows.sort(key=lambda r: float(r["family_sensitivity_score"]), reverse=True)
    return rows[:top_n]


def build_edge_example_panels(
    drift_rows: Sequence[Dict[str, object]],
    stable_delta_threshold: int,
    examples_per_panel: int,
) -> List[Dict[str, object]]:
    by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for row in drift_rows:
        dataset = str(row["dataset"])
        by_dataset.setdefault(dataset, []).append(row)

    panel_rows: List[Dict[str, object]] = []
    for dataset, rows in sorted(by_dataset.items()):
        unstable = sorted(rows, key=lambda r: int(r["rank_delta_abs"]), reverse=True)[
            :examples_per_panel
        ]
        stable_candidates = [
            r for r in rows if int(r["rank_delta_abs"]) <= stable_delta_threshold
        ]
        stable = sorted(stable_candidates, key=lambda r: int(r["rank_delta_abs"]))[
            :examples_per_panel
        ]

        for panel_name, selected in (("unstable", unstable), ("stable", stable)):
            for record in selected:
                panel_rows.append(
                    {
                        "dataset": dataset,
                        "panel": panel_name,
                        "gene_a": record["gene_a"],
                        "gene_b": record["gene_b"],
                        "coarse_rank": int(record["coarse_rank"]),
                        "fine_rank": int(record["fine_rank"]),
                        "rank_delta_abs": int(record["rank_delta_abs"]),
                    }
                )

    return panel_rows


def _metric_to_recommendation(sensitivity: float) -> Tuple[str, str]:
    if sensitivity <= 0.30:
        return (
            "fine",
            "Fine clustering is robust enough for primary ranking claims in this tissue.",
        )
    if sensitivity <= 0.45:
        return (
            "hybrid",
            "Use coarse clustering for headline claims and fine clustering for subtype hypothesis generation.",
        )
    return (
        "coarse",
        "Primary reporting should stay at coarse granularity; fine clusters should be treated as exploratory.",
    )


def _downgrade_recommendation_once(recommendation: str) -> str:
    if recommendation == "fine":
        return "hybrid"
    if recommendation == "hybrid":
        return "coarse"
    return "coarse"


def calibrate_recommendation_with_dual_null(
    base_recommendation: str,
    global_null_pass: int,
    constrained_null_pass: int,
) -> Tuple[str, str]:
    global_ok = int(global_null_pass) == 1
    constrained_ok = int(constrained_null_pass) == 1

    n_fail = int(not global_ok) + int(not constrained_ok)
    calibrated = base_recommendation
    for _ in range(n_fail):
        calibrated = _downgrade_recommendation_once(calibrated)

    if n_fail == 0:
        return (
            base_recommendation,
            "Dual-null calibrated: observed coarse-vs-fine structure is separated from both global-shuffle and within-coarse permutation baselines.",
        )

    if n_fail == 1:
        failed = (
            "global-shuffle"
            if not global_ok
            else "within-coarse constrained-shuffle"
        )
        return (
            calibrated,
            f"Downgraded by dual-null calibration: failed {failed} separation, so recommendation is made more conservative.",
        )

    return (
        calibrated,
        "Downgraded by dual-null calibration: failed both global-shuffle and within-coarse constrained-shuffle separation.",
    )


def within_coarse_group_analysis(
    x_dense: np.ndarray,
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
    global_weight: float,
    min_cells_coarse: int,
    min_cells_fine: int,
    top_k_local: int,
) -> List[Dict[str, object]]:
    iu = np.triu_indices(x_dense.shape[1], 1)
    results: List[Dict[str, object]] = []

    coarse_unique, coarse_counts = np.unique(coarse_labels, return_counts=True)
    coarse_sorted = sorted(
        [(str(c), int(n)) for c, n in zip(coarse_unique, coarse_counts)],
        key=lambda t: t[1],
        reverse=True,
    )

    for coarse_label, n_cells in coarse_sorted:
        if n_cells < min_cells_coarse:
            continue

        mask = coarse_labels == coarse_label
        x_group = x_dense[mask]
        fine_group = fine_labels[mask]

        coarse_corr = abs_correlation(x_group)
        fine_corr_agg, used_fine = grouped_max_abs_correlation(
            x_group,
            fine_group,
            min_cells=min_cells_fine,
        )
        if len(used_fine) < 2:
            continue

        # Within a single coarse class, the coarse score reduces to global correlation.
        # For fine partitioning we keep the same scoring formula as pooled analysis.
        fine_score_matrix = (global_weight * coarse_corr) + (
            (1.0 - global_weight) * fine_corr_agg
        )
        np.fill_diagonal(fine_score_matrix, 0.0)

        coarse_scores = coarse_corr[iu]
        fine_scores = fine_score_matrix[iu]

        coarse_order = np.argsort(coarse_scores)[::-1]
        fine_order = np.argsort(fine_scores)[::-1]

        coarse_rank_vec = np.empty(coarse_order.shape[0], dtype=np.int64)
        coarse_rank_vec[coarse_order] = np.arange(1, coarse_order.shape[0] + 1)

        fine_rank_vec = np.empty(fine_order.shape[0], dtype=np.int64)
        fine_rank_vec[fine_order] = np.arange(1, fine_order.shape[0] + 1)

        metrics = compare_rankings(
            coarse_ranks=coarse_rank_vec,
            fine_ranks=fine_rank_vec,
            top_k=top_k_local,
        )

        results.append(
            {
                "coarse_label": coarse_label,
                "n_cells": n_cells,
                "n_fine_subclusters_used": len(used_fine),
                **metrics,
            }
        )

    return results


def permutation_null_calibration(
    x_dense: np.ndarray,
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
    coarse_rank_vector: np.ndarray,
    observed_metrics: Dict[str, float],
    min_cells_group: int,
    global_weight: float,
    top_k: int,
    n_permutations: int,
    rng_seed: int,
    null_mode: str,
) -> Dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    perm_overlaps: List[float] = []
    perm_jaccards: List[float] = []
    perm_sensitivity: List[float] = []

    for _ in range(n_permutations):
        perm_labels = permute_fine_labels_for_null(
            fine_labels=fine_labels,
            coarse_labels=coarse_labels,
            null_mode=null_mode,
            rng=rng,
        )
        perm_result = infer_rankings(
            x_dense=x_dense,
            labels=perm_labels,
            min_cells_group=min_cells_group,
            global_weight=global_weight,
        )
        perm_metrics = compare_rankings(
            coarse_ranks=coarse_rank_vector,
            fine_ranks=perm_result.rank_vector,
            top_k=top_k,
        )
        perm_overlaps.append(float(perm_metrics["topk_overlap"]))
        perm_jaccards.append(float(perm_metrics["jaccard"]))
        perm_sensitivity.append(float(perm_metrics["resolution_sensitivity"]))

    overlap_arr = np.asarray(perm_overlaps, dtype=np.float64)
    jaccard_arr = np.asarray(perm_jaccards, dtype=np.float64)
    sensitivity_arr = np.asarray(perm_sensitivity, dtype=np.float64)

    obs_overlap = float(observed_metrics["topk_overlap"])
    obs_jaccard = float(observed_metrics["jaccard"])
    obs_sensitivity = float(observed_metrics["resolution_sensitivity"])

    overlap_mean = float(np.mean(overlap_arr))
    overlap_std = float(np.std(overlap_arr, ddof=1)) if overlap_arr.size > 1 else 0.0
    jaccard_mean = float(np.mean(jaccard_arr))
    jaccard_std = float(np.std(jaccard_arr, ddof=1)) if jaccard_arr.size > 1 else 0.0
    sensitivity_mean = float(np.mean(sensitivity_arr))
    sensitivity_std = (
        float(np.std(sensitivity_arr, ddof=1)) if sensitivity_arr.size > 1 else 0.0
    )

    overlap_z = (
        (obs_overlap - overlap_mean) / overlap_std if overlap_std > 1e-12 else 0.0
    )
    sensitivity_improvement_z = (
        (sensitivity_mean - obs_sensitivity) / sensitivity_std
        if sensitivity_std > 1e-12
        else 0.0
    )

    overlap_empirical_p_ge = float(
        (np.sum(overlap_arr >= obs_overlap) + 1.0) / (overlap_arr.size + 1.0)
    )
    sensitivity_empirical_p_le = float(
        (np.sum(sensitivity_arr <= obs_sensitivity) + 1.0)
        / (sensitivity_arr.size + 1.0)
    )
    jaccard_empirical_p_ge = float(
        (np.sum(jaccard_arr >= obs_jaccard) + 1.0) / (jaccard_arr.size + 1.0)
    )

    return {
        "null_mode": null_mode,
        "n_permutations": int(n_permutations),
        "observed_topk_overlap": obs_overlap,
        "null_mean_topk_overlap": overlap_mean,
        "null_std_topk_overlap": overlap_std,
        "overlap_zscore": float(overlap_z),
        "overlap_empirical_p_ge": overlap_empirical_p_ge,
        "observed_jaccard": obs_jaccard,
        "null_mean_jaccard": jaccard_mean,
        "null_std_jaccard": jaccard_std,
        "jaccard_empirical_p_ge": jaccard_empirical_p_ge,
        "observed_resolution_sensitivity": obs_sensitivity,
        "null_mean_resolution_sensitivity": sensitivity_mean,
        "null_std_resolution_sensitivity": sensitivity_std,
        "sensitivity_improvement_zscore": float(sensitivity_improvement_z),
        "sensitivity_empirical_p_le": sensitivity_empirical_p_le,
        "null_separation_pass": int(
            overlap_empirical_p_ge <= 0.05 and sensitivity_empirical_p_le <= 0.05
        ),
    }


def summarize_robustness_grid(
    robustness_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in robustness_rows:
        grouped[str(row["dataset"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for dataset, rows in sorted(grouped.items()):
        sensitivity_values = np.asarray(
            [float(r["resolution_sensitivity"]) for r in rows], dtype=np.float64
        )
        overlap_values = np.asarray(
            [float(r["topk_overlap"]) for r in rows], dtype=np.float64
        )
        rec_counter = Counter(str(r["recommended_granularity"]) for r in rows)
        mode_rec, mode_count = rec_counter.most_common(1)[0]
        consistency = float(mode_count / max(len(rows), 1))

        sensitivity_mean = float(np.mean(sensitivity_values))
        sensitivity_std = (
            float(np.std(sensitivity_values, ddof=1)) if sensitivity_values.size > 1 else 0.0
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "n_grid_runs": len(rows),
                "sensitivity_mean": sensitivity_mean,
                "sensitivity_std": sensitivity_std,
                "sensitivity_min": float(np.min(sensitivity_values)),
                "sensitivity_max": float(np.max(sensitivity_values)),
                "sensitivity_cv": float(
                    sensitivity_std / sensitivity_mean if sensitivity_mean > 1e-12 else 0.0
                ),
                "topk_overlap_mean": float(np.mean(overlap_values)),
                "topk_overlap_min": float(np.min(overlap_values)),
                "topk_overlap_max": float(np.max(overlap_values)),
                "recommendation_mode": mode_rec,
                "recommendation_consistency": consistency,
                "recommendation_stable": int(consistency >= 0.70),
            }
        )
        if "null_separation_pass" in rows[0]:
            dual_pass = np.asarray(
                [int(r["null_separation_pass"]) for r in rows], dtype=np.float64
            )
            global_pass = np.asarray(
                [int(r.get("null_global_pass", 0)) for r in rows], dtype=np.float64
            )
            constrained_pass = np.asarray(
                [int(r.get("null_constrained_pass", 0)) for r in rows],
                dtype=np.float64,
            )
            summary_rows[-1]["dual_null_pass_rate"] = float(np.mean(dual_pass))
            summary_rows[-1]["global_null_pass_rate"] = float(np.mean(global_pass))
            summary_rows[-1]["constrained_null_pass_rate"] = float(
                np.mean(constrained_pass)
            )

    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_permutation_count_for_alpha(
    n_permutations: int,
    alpha: float,
    label: str,
    allow_zero: bool = False,
) -> None:
    if allow_zero and int(n_permutations) == 0:
        return
    n = int(n_permutations)
    if n <= 0:
        raise ValueError(f"{label} must be positive (got {n}).")
    min_empirical_p = 1.0 / float(n + 1)
    if min_empirical_p > alpha:
        min_required = int(math.ceil((1.0 / alpha) - 1.0))
        raise ValueError(
            f"{label}={n} cannot reach empirical p<={alpha:.2f}; "
            f"minimum required is {min_required} permutations."
        )


def _figure_pooled_sensitivity(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    datasets = [str(r["dataset"]) for r in rows]
    values = [float(r["resolution_sensitivity"]) for r in rows]
    colors = plt.cm.tab20(np.linspace(0.05, 0.95, len(datasets)))

    plt.figure(figsize=(7.2, 4.2))
    bars = plt.bar(datasets, values, color=colors)
    plt.ylim(0, max(values) * 1.3 if values else 1.0)
    plt.ylabel("Resolution Sensitivity Score")
    plt.title("Pooled Coarse-vs-Fine Sensitivity by Tissue")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _figure_overlap_vs_drift(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    plt.figure(figsize=(6.8, 4.5))
    for row in rows:
        x = float(row["topk_overlap"])
        y = float(row["median_rank_drift"])
        label = str(row["dataset"])
        plt.scatter([x], [y], s=90)
        plt.text(x + 0.005, y + 0.005, label, fontsize=9)

    plt.xlabel("Top-k Overlap")
    plt.ylabel("Median Normalized Rank Drift")
    plt.title("Coarse-vs-Fine Rank Agreement")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _figure_within_group_heatmap(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    datasets = sorted({str(r["dataset"]) for r in rows})
    coarse_labels = sorted({str(r["coarse_label"]) for r in rows})

    matrix = np.full((len(coarse_labels), len(datasets)), np.nan, dtype=np.float32)
    for row in rows:
        i = coarse_labels.index(str(row["coarse_label"]))
        j = datasets.index(str(row["dataset"]))
        matrix[i, j] = float(row["resolution_sensitivity"])

    plt.figure(figsize=(8.8, max(5.2, 0.24 * len(coarse_labels) + 2.0)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    im = plt.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Resolution Sensitivity")
    plt.xticks(ticks=np.arange(len(datasets)), labels=datasets)
    plt.yticks(ticks=np.arange(len(coarse_labels)), labels=coarse_labels)
    plt.title("Within-Coarse-Class Sensitivity (Coarse vs Fine Subclusters)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _figure_sensitive_gene_families(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    agg: Dict[str, float] = {}
    for row in rows:
        gene = str(row["gene_family_hub"])
        agg[gene] = agg.get(gene, 0.0) + float(row["family_sensitivity_score"])

    top = sorted(agg.items(), key=lambda t: t[1], reverse=True)[:15]
    genes = [t[0] for t in top][::-1]
    scores = [t[1] for t in top][::-1]

    plt.figure(figsize=(8.4, max(4.5, 0.32 * len(genes) + 1.5)))
    plt.barh(genes, scores, color="#4c78a8")
    plt.xlabel("Aggregated Family Sensitivity Score")
    plt.title("Most Resolution-Sensitive Hub Gene Families")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _figure_null_calibration(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    datasets = [str(r["dataset"]) for r in rows]
    observed_overlap = [float(r["global_observed_topk_overlap"]) for r in rows]
    global_null_overlap = [float(r["global_null_mean_topk_overlap"]) for r in rows]
    global_null_overlap_std = [float(r["global_null_std_topk_overlap"]) for r in rows]
    constrained_null_overlap = [
        float(r["constrained_null_mean_topk_overlap"]) for r in rows
    ]
    constrained_null_overlap_std = [
        float(r["constrained_null_std_topk_overlap"]) for r in rows
    ]

    observed_sens = [float(r["global_observed_resolution_sensitivity"]) for r in rows]
    global_null_sens = [
        float(r["global_null_mean_resolution_sensitivity"]) for r in rows
    ]
    global_null_sens_std = [
        float(r["global_null_std_resolution_sensitivity"]) for r in rows
    ]
    constrained_null_sens = [
        float(r["constrained_null_mean_resolution_sensitivity"]) for r in rows
    ]
    constrained_null_sens_std = [
        float(r["constrained_null_std_resolution_sensitivity"]) for r in rows
    ]

    x = np.arange(len(datasets))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    axes[0].bar(x - width, observed_overlap, width=width, label="Observed")
    axes[0].bar(
        x,
        global_null_overlap,
        width=width,
        yerr=global_null_overlap_std,
        capsize=3,
        label="Global-shuffle null",
    )
    axes[0].bar(
        x + width,
        constrained_null_overlap,
        width=width,
        yerr=constrained_null_overlap_std,
        capsize=3,
        label="Within-coarse null",
    )
    axes[0].set_xticks(x, datasets)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Top-k Overlap vs Null")
    axes[0].set_ylabel("Overlap")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width, observed_sens, width=width, label="Observed")
    axes[1].bar(
        x,
        global_null_sens,
        width=width,
        yerr=global_null_sens_std,
        capsize=3,
        label="Global-shuffle null",
    )
    axes[1].bar(
        x + width,
        constrained_null_sens,
        width=width,
        yerr=constrained_null_sens_std,
        capsize=3,
        label="Within-coarse null",
    )
    axes[1].set_xticks(x, datasets)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Sensitivity Score vs Null")
    axes[1].set_ylabel("Resolution Sensitivity")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _figure_robustness_heatmap(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
) -> None:
    if not rows:
        return

    datasets = sorted({str(r["dataset"]) for r in rows})
    gene_values = sorted({int(r["top_genes"]) for r in rows})
    weight_values = sorted({float(r["global_weight"]) for r in rows})

    # Plot one panel per dataset with sensitivity over (genes x weight) grid.
    n_ds = len(datasets)
    fig, axes = plt.subplots(
        1,
        n_ds,
        figsize=(4.1 * n_ds, 4.4),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, dataset in enumerate(datasets):
        panel = np.full((len(gene_values), len(weight_values)), np.nan, dtype=np.float32)
        for row in rows:
            if str(row["dataset"]) != dataset:
                continue
            gi = gene_values.index(int(row["top_genes"]))
            wi = weight_values.index(float(row["global_weight"]))
            panel[gi, wi] = float(row["resolution_sensitivity"])

        ax = axes[0, idx]
        im = ax.imshow(panel, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(weight_values)))
        ax.set_xticklabels([f"{w:.1f}" for w in weight_values], rotation=45)
        ax.set_yticks(np.arange(len(gene_values)))
        ax.set_yticklabels([str(g) for g in gene_values])
        ax.set_xlabel("Global weight")
        if idx == 0:
            ax.set_ylabel("Top genes")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="RSS")
    fig.suptitle("Hyperparameter Robustness of Resolution Sensitivity", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_pipeline(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    validate_permutation_count_for_alpha(
        n_permutations=args.null_permutations,
        alpha=0.05,
        label="--null-permutations",
    )
    validate_permutation_count_for_alpha(
        n_permutations=args.null_permutations_constrained,
        alpha=0.05,
        label="--null-permutations-constrained",
    )
    validate_permutation_count_for_alpha(
        n_permutations=args.robust_null_permutations,
        alpha=0.05,
        label="--robust-null-permutations",
        allow_zero=True,
    )
    validate_permutation_count_for_alpha(
        n_permutations=args.robust_null_permutations_constrained,
        alpha=0.05,
        label="--robust-null-permutations-constrained",
        allow_zero=True,
    )

    top_gene_grid = sorted(
        {
            int(v)
            for v in [args.top_genes, *[int(x) for x in args.robust_top_genes.split(",") if x.strip()]]
            if int(v) >= 50
        }
    )
    global_weight_grid = sorted(
        {
            float(v)
            for v in [args.global_weight, *[float(x) for x in args.robust_global_weights.split(",") if x.strip()]]
            if 0.0 <= float(v) <= 1.0
        }
    )

    pooled_rows: List[Dict[str, object]] = []
    within_rows: List[Dict[str, object]] = []
    drift_rows_all: List[Dict[str, object]] = []
    family_rows_all: List[Dict[str, object]] = []
    null_rows_global: List[Dict[str, object]] = []
    null_rows_constrained: List[Dict[str, object]] = []
    null_rows_combined: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []

    for config in DATASET_CONFIGS:
        loaded = load_h5ad_dataset(
            dataset_name=config["dataset"],
            path=config["path"],
            coarse_key=config["coarse_key"],
            fine_key=config["fine_key"],
        )

        ranked_gene_idx = rank_high_variance_genes(
            x=loaded.x,
            min_detect_frac=args.min_detect_frac,
        )
        if ranked_gene_idx.size == 0:
            continue

        max_grid_genes = min(max(top_gene_grid), int(ranked_gene_idx.size))
        active_top_gene_grid = sorted(
            {g for g in top_gene_grid if g <= max_grid_genes and g >= 50}
        )
        if not active_top_gene_grid:
            continue

        x_ranked_dense = loaded.x[:, ranked_gene_idx[:max_grid_genes]].toarray().astype(
            np.float32
        )
        coarse_labels = loaded.coarse_labels
        fine_labels = make_hierarchical_fine_labels(
            coarse_labels=coarse_labels,
            fine_labels=loaded.fine_labels,
        )

        selected_top_genes = min(args.top_genes, max_grid_genes)
        x_selected = x_ranked_dense[:, :selected_top_genes]
        selected_genes = loaded.genes[ranked_gene_idx[:selected_top_genes]]

        coarse_result = infer_rankings(
            x_dense=x_selected,
            labels=coarse_labels,
            min_cells_group=args.min_cells_group,
            global_weight=args.global_weight,
        )
        fine_result = infer_rankings(
            x_dense=x_selected,
            labels=fine_labels,
            min_cells_group=args.min_cells_group,
            global_weight=args.global_weight,
        )

        pooled_metrics = compare_rankings(
            coarse_ranks=coarse_result.rank_vector,
            fine_ranks=fine_result.rank_vector,
            top_k=args.top_k,
        )
        base_recommendation, base_guidance = _metric_to_recommendation(
            float(pooled_metrics["resolution_sensitivity"])
        )
        dataset_seed = args.null_seed + sum(ord(ch) for ch in loaded.dataset)

        null_global = permutation_null_calibration(
            x_dense=x_selected,
            coarse_labels=coarse_labels,
            fine_labels=fine_labels,
            coarse_rank_vector=coarse_result.rank_vector,
            observed_metrics=pooled_metrics,
            min_cells_group=args.min_cells_group,
            global_weight=args.global_weight,
            top_k=args.top_k,
            n_permutations=args.null_permutations,
            rng_seed=dataset_seed,
            null_mode="global",
        )
        null_constrained = permutation_null_calibration(
            x_dense=x_selected,
            coarse_labels=coarse_labels,
            fine_labels=fine_labels,
            coarse_rank_vector=coarse_result.rank_vector,
            observed_metrics=pooled_metrics,
            min_cells_group=args.min_cells_group,
            global_weight=args.global_weight,
            top_k=args.top_k,
            n_permutations=args.null_permutations_constrained,
            rng_seed=dataset_seed + 100003,
            null_mode="within_coarse",
        )
        combined_null_pass = int(
            int(null_global["null_separation_pass"]) == 1
            and int(null_constrained["null_separation_pass"]) == 1
        )
        calibrated_recommendation, calibration_note = calibrate_recommendation_with_dual_null(
            base_recommendation=base_recommendation,
            global_null_pass=int(null_global["null_separation_pass"]),
            constrained_null_pass=int(null_constrained["null_separation_pass"]),
        )

        pooled_rows.append(
            {
                "dataset": loaded.dataset,
                "n_cells": int(x_selected.shape[0]),
                "n_genes_selected": int(x_selected.shape[1]),
                "n_coarse_groups_used": len(coarse_result.top_group_summary),
                "n_fine_groups_used": len(fine_result.top_group_summary),
                **pooled_metrics,
                "base_recommendation": base_recommendation,
                "base_guidance": base_guidance,
                "null_global_pass": int(null_global["null_separation_pass"]),
                "null_constrained_pass": int(null_constrained["null_separation_pass"]),
                "null_separation_pass": combined_null_pass,
                "recommended_granularity": calibrated_recommendation,
                "guidance": calibration_note,
            }
        )
        null_rows_global.append(
            {
                "dataset": loaded.dataset,
                "base_recommendation": base_recommendation,
                "calibrated_recommendation": calibrated_recommendation,
                **null_global,
            }
        )
        null_rows_constrained.append(
            {
                "dataset": loaded.dataset,
                "base_recommendation": base_recommendation,
                "calibrated_recommendation": calibrated_recommendation,
                **null_constrained,
            }
        )
        null_rows_combined.append(
            {
                "dataset": loaded.dataset,
                "base_recommendation": base_recommendation,
                "calibrated_recommendation": calibrated_recommendation,
                "global_null_separation_pass": int(null_global["null_separation_pass"]),
                "constrained_null_separation_pass": int(
                    null_constrained["null_separation_pass"]
                ),
                "null_separation_pass": combined_null_pass,
                "n_null_families_passed": int(null_global["null_separation_pass"])
                + int(null_constrained["null_separation_pass"]),
                **{f"global_{k}": v for k, v in null_global.items()},
                **{f"constrained_{k}": v for k, v in null_constrained.items()},
            }
        )

        iu = np.triu_indices(x_selected.shape[1], 1)
        drift_rows = build_edge_drift_rows(
            genes=selected_genes,
            coarse_rank_vector=coarse_result.rank_vector,
            fine_rank_vector=fine_result.rank_vector,
            iu=iu,
            top_n_union=args.top_union,
        )
        for row in drift_rows:
            row["dataset"] = loaded.dataset
        drift_rows_all.extend(drift_rows)

        family_rows = summarize_sensitive_gene_families(
            drift_rows=drift_rows,
            delta_threshold=args.family_delta_threshold,
            min_incident_edges=args.family_min_edges,
            top_n=args.family_top_n,
        )
        for row in family_rows:
            row["dataset"] = loaded.dataset
        family_rows_all.extend(family_rows)

        within_rows_dataset = within_coarse_group_analysis(
            x_dense=x_selected,
            coarse_labels=coarse_labels,
            fine_labels=fine_labels,
            global_weight=args.global_weight,
            min_cells_coarse=args.min_cells_within_coarse,
            min_cells_fine=args.min_cells_within_fine,
            top_k_local=args.top_k_within,
        )
        for row in within_rows_dataset:
            row["dataset"] = loaded.dataset
        within_rows.extend(within_rows_dataset)

        # Hyperparameter robustness sweep (adversarial check).
        for top_idx, sweep_top_genes in enumerate(active_top_gene_grid):
            x_sweep = x_ranked_dense[:, :sweep_top_genes]
            for weight_idx, sweep_weight in enumerate(global_weight_grid):
                coarse_sweep = infer_rankings(
                    x_dense=x_sweep,
                    labels=coarse_labels,
                    min_cells_group=args.min_cells_group,
                    global_weight=sweep_weight,
                )
                fine_sweep = infer_rankings(
                    x_dense=x_sweep,
                    labels=fine_labels,
                    min_cells_group=args.min_cells_group,
                    global_weight=sweep_weight,
                )
                sweep_metrics = compare_rankings(
                    coarse_ranks=coarse_sweep.rank_vector,
                    fine_ranks=fine_sweep.rank_vector,
                    top_k=args.top_k,
                )
                sweep_base_rec, _ = _metric_to_recommendation(
                    float(sweep_metrics["resolution_sensitivity"])
                )
                sweep_global_null_pass = -1
                sweep_constrained_null_pass = -1
                sweep_combined_null_pass = -1
                sweep_rec = sweep_base_rec
                if (
                    args.robust_null_permutations > 0
                    and args.robust_null_permutations_constrained > 0
                ):
                    # Keep sweep seeds deterministic and independent from default-run nulls.
                    sweep_seed = dataset_seed + 500000 + (top_idx * 1000) + weight_idx
                    sweep_null_global = permutation_null_calibration(
                        x_dense=x_sweep,
                        coarse_labels=coarse_labels,
                        fine_labels=fine_labels,
                        coarse_rank_vector=coarse_sweep.rank_vector,
                        observed_metrics=sweep_metrics,
                        min_cells_group=args.min_cells_group,
                        global_weight=sweep_weight,
                        top_k=args.top_k,
                        n_permutations=args.robust_null_permutations,
                        rng_seed=sweep_seed,
                        null_mode="global",
                    )
                    sweep_null_constrained = permutation_null_calibration(
                        x_dense=x_sweep,
                        coarse_labels=coarse_labels,
                        fine_labels=fine_labels,
                        coarse_rank_vector=coarse_sweep.rank_vector,
                        observed_metrics=sweep_metrics,
                        min_cells_group=args.min_cells_group,
                        global_weight=sweep_weight,
                        top_k=args.top_k,
                        n_permutations=args.robust_null_permutations_constrained,
                        rng_seed=sweep_seed + 200003,
                        null_mode="within_coarse",
                    )
                    sweep_global_null_pass = int(
                        sweep_null_global["null_separation_pass"]
                    )
                    sweep_constrained_null_pass = int(
                        sweep_null_constrained["null_separation_pass"]
                    )
                    sweep_combined_null_pass = int(
                        sweep_global_null_pass == 1
                        and sweep_constrained_null_pass == 1
                    )
                    sweep_rec, _ = calibrate_recommendation_with_dual_null(
                        base_recommendation=sweep_base_rec,
                        global_null_pass=sweep_global_null_pass,
                        constrained_null_pass=sweep_constrained_null_pass,
                    )
                robustness_rows.append(
                    {
                        "dataset": loaded.dataset,
                        "top_genes": int(sweep_top_genes),
                        "global_weight": float(sweep_weight),
                        **sweep_metrics,
                        "base_recommended_granularity": sweep_base_rec,
                        "null_global_pass": sweep_global_null_pass,
                        "null_constrained_pass": sweep_constrained_null_pass,
                        "null_separation_pass": sweep_combined_null_pass,
                        "recommended_granularity": sweep_rec,
                    }
                )

    write_csv(RESULTS_DIR / "pooled_resolution_sensitivity.csv", pooled_rows)
    write_csv(RESULTS_DIR / "within_coarse_group_sensitivity.csv", within_rows)
    write_csv(RESULTS_DIR / "edge_rank_drift_top_union.csv", drift_rows_all)
    write_csv(RESULTS_DIR / "sensitive_edge_families.csv", family_rows_all)
    write_csv(RESULTS_DIR / "null_calibration_global.csv", null_rows_global)
    write_csv(
        RESULTS_DIR / "null_calibration_constrained.csv", null_rows_constrained
    )
    write_csv(RESULTS_DIR / "null_calibration.csv", null_rows_combined)
    write_csv(RESULTS_DIR / "robustness_grid.csv", robustness_rows)
    write_csv(
        RESULTS_DIR / "robustness_summary.csv",
        summarize_robustness_grid(robustness_rows),
    )
    example_panels = build_edge_example_panels(
        drift_rows=drift_rows_all,
        stable_delta_threshold=args.example_stable_delta_threshold,
        examples_per_panel=args.example_edges_per_panel,
    )
    write_csv(RESULTS_DIR / "edge_example_panels.csv", example_panels)

    # Scorecard consolidates primary recommendation and agreement metrics.
    scorecard_rows = []
    for row in pooled_rows:
        scorecard_rows.append(
            {
                "dataset": row["dataset"],
                "resolution_sensitivity": row["resolution_sensitivity"],
                "topk_overlap": row["topk_overlap"],
                "median_rank_drift": row["median_rank_drift"],
                "base_recommendation": row["base_recommendation"],
                "null_separation_pass": row["null_separation_pass"],
                "recommended_granularity": row["recommended_granularity"],
                "guidance": row["guidance"],
            }
        )
    write_csv(RESULTS_DIR / "resolution_sensitivity_scorecard.csv", scorecard_rows)

    # Also store a machine-readable run summary for quick downstream parsing.
    write_json(
        RESULTS_DIR / "run_summary.json",
        {
            "datasets": [r["dataset"] for r in pooled_rows],
            "top_genes": args.top_genes,
            "top_k": args.top_k,
            "global_weight": args.global_weight,
            "null_permutations": args.null_permutations,
            "null_permutations_constrained": args.null_permutations_constrained,
            "robust_null_permutations": args.robust_null_permutations,
            "robust_null_permutations_constrained": args.robust_null_permutations_constrained,
            "robust_top_gene_grid": top_gene_grid,
            "robust_global_weight_grid": global_weight_grid,
            "pooled_rows": pooled_rows,
            "n_within_rows": len(within_rows),
            "n_drift_rows": len(drift_rows_all),
            "n_family_rows": len(family_rows_all),
            "n_example_panel_rows": len(example_panels),
            "n_null_rows_global": len(null_rows_global),
            "n_null_rows_constrained": len(null_rows_constrained),
            "n_null_rows_combined": len(null_rows_combined),
            "n_robustness_rows": len(robustness_rows),
        },
    )

    _figure_pooled_sensitivity(
        pooled_rows,
        FIGURES_DIR / "fig1_pooled_resolution_sensitivity.png",
    )
    _figure_overlap_vs_drift(
        pooled_rows,
        FIGURES_DIR / "fig2_overlap_vs_rank_drift.png",
    )
    _figure_within_group_heatmap(
        within_rows,
        FIGURES_DIR / "fig3_within_group_sensitivity_heatmap.png",
    )
    _figure_sensitive_gene_families(
        family_rows_all,
        FIGURES_DIR / "fig4_sensitive_gene_families.png",
    )
    _figure_null_calibration(
        null_rows_combined,
        FIGURES_DIR / "fig5_null_calibration.png",
    )
    _figure_robustness_heatmap(
        robustness_rows,
        FIGURES_DIR / "fig6_robustness_heatmap.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-genes", type=int, default=600)
    parser.add_argument("--min-detect-frac", type=float, default=0.03)
    parser.add_argument("--min-cells-group", type=int, default=120)
    parser.add_argument("--global-weight", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-union", type=int, default=2000)
    parser.add_argument("--min-cells-within-coarse", type=int, default=250)
    parser.add_argument("--min-cells-within-fine", type=int, default=80)
    parser.add_argument("--top-k-within", type=int, default=300)
    parser.add_argument("--family-delta-threshold", type=int, default=600)
    parser.add_argument("--family-min-edges", type=int, default=5)
    parser.add_argument("--family-top-n", type=int, default=20)
    parser.add_argument("--example-stable-delta-threshold", type=int, default=100)
    parser.add_argument("--example-edges-per-panel", type=int, default=15)
    parser.add_argument("--null-permutations", type=int, default=49)
    parser.add_argument("--null-permutations-constrained", type=int, default=49)
    parser.add_argument("--robust-null-permutations", type=int, default=25)
    parser.add_argument("--robust-null-permutations-constrained", type=int, default=25)
    parser.add_argument("--null-seed", type=int, default=42)
    parser.add_argument("--robust-top-genes", type=str, default="400,600,800")
    parser.add_argument("--robust-global-weights", type=str, default="0.4,0.6,0.8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
