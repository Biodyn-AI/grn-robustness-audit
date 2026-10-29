from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import precision_recall_f1
from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config


@dataclass
class RunRecord:
    experiment_id: str
    model_tier: str
    tissue: str
    seed: int
    max_cells: str
    preprocess_hvg: int
    max_genes: int
    gene_subset: str
    candidate_sources: bool
    candidate_targets: bool
    edge_method: str
    top_k: int | None
    threshold_percentile: float | None
    sample_strategy: str
    sample_seed: int | None
    processed_h5ad: str
    network_edges: str


def _read_manifest(path: Path) -> List[RunRecord]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            records.append(
                RunRecord(
                    experiment_id=row["experiment_id"],
                    model_tier=row["model_tier"],
                    tissue=row["tissue"],
                    seed=int(row["seed"]),
                    max_cells=row.get("max_cells", ""),
                    preprocess_hvg=int(row["preprocess_hvg"]),
                    max_genes=int(row["max_genes"]),
                    gene_subset=row["gene_subset"],
                    candidate_sources=row.get("candidate_sources_from_dorothea", "false").lower() == "true",
                    candidate_targets=row.get("candidate_targets_from_dorothea", "false").lower() == "true",
                    edge_method=row["edge_method"],
                    top_k=_parse_optional_int(row.get("top_k")),
                    threshold_percentile=_parse_optional_float(row.get("threshold_percentile")),
                    sample_strategy=row.get("sample_strategy", ""),
                    sample_seed=_parse_optional_int(row.get("sample_seed")),
                    processed_h5ad=row["processed_h5ad"],
                    network_edges=row["network_edges"],
                )
            )
    return records


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def _load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "score" not in df.columns:
        df["score"] = np.nan
    return df[["source", "target", "score"]]


def _candidate_masks(
    adata: sc.AnnData,
    true_edges: pd.DataFrame,
    alias_map: Dict[str, str],
    use_sources: bool,
    use_targets: bool,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if not (use_sources or use_targets):
        return None, None
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    true_edges_norm = normalize_edges(true_edges, alias_map)
    source_mask = None
    target_mask = None
    if use_sources:
        sources = set(true_edges_norm["source"].unique())
        source_mask = np.array([name in sources for name in gene_names_norm], dtype=bool)
    if use_targets:
        targets = set(true_edges_norm["target"].unique())
        target_mask = np.array([name in targets for name in gene_names_norm], dtype=bool)
    return source_mask, target_mask


def _sample_adata(
    adata: sc.AnnData,
    max_cells: int | None,
    sample_strategy: str,
    sample_seed: int | None,
    fallback_seed: int,
) -> sc.AnnData:
    if not max_cells or adata.n_obs <= max_cells:
        return adata
    if sample_strategy == "random":
        seed = fallback_seed if sample_seed is None else int(sample_seed)
        rng = np.random.default_rng(seed)
        indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
        return adata[np.sort(indices)].copy()
    return adata[:max_cells].copy()


def _evaluate_edges(
    pred_edges: pd.DataFrame,
    adata: sc.AnnData,
    true_edges: pd.DataFrame,
    alias_map: Dict[str, str],
) -> Dict[str, float]:
    pred_edges = pred_edges[["source", "target"]].drop_duplicates()
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_set = set(gene_names_norm)

    pred_edges = normalize_edges(pred_edges, alias_map)
    pred_edges = pred_edges[
        pred_edges["source"].isin(gene_set) & pred_edges["target"].isin(gene_set)
    ].drop_duplicates()

    true_edges = normalize_edges(true_edges, alias_map)
    true_edges = true_edges[
        true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
    ].drop_duplicates()
    return precision_recall_f1(pred_edges, true_edges)


def _rankdata(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return rankdata(values)
    except Exception:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _edge_scores(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    return {(row["source"], row["target"]): float(row["score"]) for _, row in df.iterrows()}


def _jaccard(a: Iterable[Tuple[str, str]], b: Iterable[Tuple[str, str]]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return float("nan")
    return len(set_a & set_b) / len(set_a | set_b)


def _aggregate_scaling(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "model_tier",
        "tissue",
        "max_cells",
        "preprocess_hvg",
        "max_genes",
        "gene_subset",
        "edge_method",
        "top_k",
        "threshold_percentile",
    ]
    records = []
    for metric_name in ("precision", "recall", "f1", "n_edges"):
        for keys, group in metrics_df.groupby(grouping, dropna=False):
            mean = group[metric_name].mean()
            std = group[metric_name].std(ddof=0)
            record = dict(zip(grouping, keys))
            record.update(
                {
                    "metric": metric_name,
                    "mean": mean,
                    "std": std,
                    "n": len(group),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iters: int) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(iters, values.size), replace=True)
    means = samples.mean(axis=1)
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    return float(lower), float(upper)


def _aggregate_confidence(
    metrics_df: pd.DataFrame,
    iters: int,
    seed: int,
) -> pd.DataFrame:
    grouping = [
        "model_tier",
        "tissue",
        "max_cells",
        "preprocess_hvg",
        "max_genes",
        "gene_subset",
        "edge_method",
        "top_k",
        "threshold_percentile",
    ]
    records = []
    rng = np.random.default_rng(seed)
    for metric_name in ("precision", "recall", "f1"):
        for keys, group in metrics_df.groupby(grouping, dropna=False):
            values = group[metric_name].to_numpy(dtype=float)
            mean = float(np.mean(values)) if values.size else float("nan")
            lower, upper = _bootstrap_ci(values, rng, iters)
            record = dict(zip(grouping, keys))
            record.update(
                {
                    "metric": metric_name,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n": len(values),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def _aggregate_robustness(run_data: Dict[str, Dict], metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "model_tier",
        "tissue",
        "max_cells",
        "preprocess_hvg",
        "max_genes",
        "gene_subset",
        "edge_method",
        "top_k",
        "threshold_percentile",
    ]
    records = []
    for keys, group in metrics_df.groupby(grouping, dropna=False):
        run_ids = group["experiment_id"].tolist()
        if len(run_ids) < 2:
            continue
        jaccards = []
        spearmans = []
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                run_a = run_data[run_ids[i]]
                run_b = run_data[run_ids[j]]
                edges_a = run_a["edges"]
                edges_b = run_b["edges"]
                jaccards.append(_jaccard(edges_a.keys(), edges_b.keys()))
                union = set(edges_a.keys()) | set(edges_b.keys())
                if not union:
                    spearmans.append(float("nan"))
                else:
                    scores_a = np.array([edges_a.get(edge, 0.0) for edge in union], dtype=float)
                    scores_b = np.array([edges_b.get(edge, 0.0) for edge in union], dtype=float)
                    spearmans.append(_spearman_correlation(scores_a, scores_b))
        record = dict(zip(grouping, keys))
        record.update(
            {
                "jaccard_mean": float(np.nanmean(jaccards)),
                "jaccard_std": float(np.nanstd(jaccards)),
                "spearman_mean": float(np.nanmean(spearmans)),
                "spearman_std": float(np.nanstd(spearmans)),
                "n_pairs": len(jaccards),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _compute_coexpression_edges(
    adata: sc.AnnData,
    config: NetworkConfig,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
) -> pd.DataFrame:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return infer_edges(corr, adata.var_names, config, source_mask, target_mask)


def _compute_random_edges(
    gene_names: List[str],
    edge_count: int,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    source_indices = np.arange(n_genes)
    target_indices = np.arange(n_genes)
    if source_mask is not None:
        source_indices = source_indices[source_mask]
    if target_mask is not None:
        target_indices = target_indices[target_mask]
    pairs = []
    if source_indices.size == 0 or target_indices.size == 0:
        return pd.DataFrame(pairs, columns=["source", "target", "score"])
    while len(pairs) < edge_count:
        sources = rng.choice(source_indices, size=edge_count, replace=True)
        targets = rng.choice(target_indices, size=edge_count, replace=True)
        for s, t in zip(sources, targets):
            if s == t:
                continue
            pairs.append((gene_names[s], gene_names[t], rng.random()))
            if len(pairs) >= edge_count:
                break
    return pd.DataFrame(pairs, columns=["source", "target", "score"])


def _load_baseline_edges(
    baseline_dir: Path,
    experiment_id: str,
    method: str,
) -> pd.DataFrame | None:
    path = baseline_dir / f"{experiment_id}_{method}_edges.tsv"
    if not path.exists():
        return None
    return _load_edges(path)


def _load_pretraining_scales(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    inventory = data.get("inventory", [])
    scales = {}
    for item in inventory:
        tier = item.get("tier")
        cells = item.get("pretraining_cells_millions")
        if tier and cells:
            scales[tier] = float(cells)
    return scales


def _fit_loglog(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), eps)
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    y_pred = slope * lx + intercept
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _compute_scaling_fits(
    scaling_df: pd.DataFrame,
    pretraining_scales: Dict[str, float],
) -> pd.DataFrame:
    records = []
    f1_df = scaling_df[
        (scaling_df["metric"] == "f1")
        & (scaling_df["gene_subset"] == "full_hvg")
        & (scaling_df["edge_method"] == "top_k")
        & (scaling_df["top_k"] == 100)
    ].copy()
    for (tissue, model_tier), group in f1_df.groupby(["tissue", "model_tier"]):
        if group["max_cells"].nunique() < 2:
            continue
        x = group["max_cells"].astype(float).to_numpy()
        y = group["mean"].astype(float).to_numpy()
        fit = _fit_loglog(x, y)
        records.append(
            {
                "fit_type": "cells",
                "tissue": tissue,
                "model_tier": model_tier,
                "metric": "f1",
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "n": len(group),
            }
        )
    if pretraining_scales:
        for tissue in f1_df["tissue"].unique():
            subset = f1_df[(f1_df["tissue"] == tissue) & (f1_df["max_cells"] == 1000)]
            if subset["model_tier"].nunique() < 2:
                continue
            x = np.array([pretraining_scales.get(tier, np.nan) for tier in subset["model_tier"]])
            y = subset["mean"].astype(float).to_numpy()
            mask = np.isfinite(x)
            if mask.sum() < 2:
                continue
            fit = _fit_loglog(x[mask], y[mask])
            records.append(
                {
                    "fit_type": "pretraining_cells",
                    "tissue": tissue,
                    "model_tier": "all",
                    "metric": "f1",
                    "slope": fit["slope"],
                    "intercept": fit["intercept"],
                    "r2": fit["r2"],
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(records)


def _permute_gene_labels(adata: sc.AnnData, edges_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = np.array(adata.var_names)
    permuted = genes.copy()
    rng.shuffle(permuted)
    mapping = dict(zip(genes, permuted))
    perm_df = edges_df.copy()
    perm_df["source"] = perm_df["source"].map(mapping).fillna(perm_df["source"])
    perm_df["target"] = perm_df["target"].map(mapping).fillna(perm_df["target"])
    return perm_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scaling runs and compute robustness metrics.")
    parser.add_argument("--manifest", default="configs/scaling_workshop_manifest.csv")
    parser.add_argument("--base-config", default="configs/scaling_base.yaml")
    parser.add_argument("--metrics-out", default="outputs/scaling/run_metrics.csv")
    parser.add_argument("--scaling-out", default="outputs/scaling/scaling_curves.csv")
    parser.add_argument("--robustness-out", default="outputs/scaling/robustness_variance.tsv")
    parser.add_argument("--confidence-out", default="outputs/scaling/scaling_confidence.csv")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--fits-out", default="outputs/scaling/scaling_fits.csv")
    parser.add_argument("--baseline-out", default="outputs/scaling/baseline_metrics.csv")
    parser.add_argument("--baseline-experiment-id", action="append")
    parser.add_argument("--baseline-methods", default="coexpression,random")
    parser.add_argument("--baseline-edges-dir", default="outputs/scaling/baselines")
    parser.add_argument("--control-out", default="outputs/scaling/control_metrics.csv")
    parser.add_argument("--control-experiment-id", action="append")
    parser.add_argument("--control-methods", default="permute_labels")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    dorothea_path = Path(base_cfg["paths"]["dorothea_tsv"])
    alias_path = Path(base_cfg["paths"]["hgnc_alias_tsv"])
    alias_map = load_hgnc_alias_map(alias_path)
    true_edges = load_dorothea(dorothea_path, confidence_levels=base_cfg.get("evaluation", {}).get("dorothea_confidence"))

    records = _read_manifest(Path(args.manifest))
    run_metrics = []
    run_data = {}
    adata_cache: Dict[str, sc.AnnData] = {}

    for record in records:
        edges_path = Path(record.network_edges)
        processed_path = Path(record.processed_h5ad)
        if not edges_path.exists() or not processed_path.exists():
            continue
        if processed_path.as_posix() not in adata_cache:
            adata_cache[processed_path.as_posix()] = sc.read_h5ad(processed_path)
        adata = adata_cache[processed_path.as_posix()]
        edges_df = _load_edges(edges_path)
        metrics = _evaluate_edges(edges_df, adata, true_edges, alias_map)
        run_metrics.append(
            {
                "experiment_id": record.experiment_id,
                "model_tier": record.model_tier,
                "tissue": record.tissue,
                "seed": record.seed,
                "max_cells": record.max_cells,
                "preprocess_hvg": record.preprocess_hvg,
                "max_genes": record.max_genes,
                "gene_subset": record.gene_subset,
                "edge_method": record.edge_method,
                "top_k": record.top_k,
                "threshold_percentile": record.threshold_percentile,
                "n_edges": len(edges_df),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "method": "attention",
            }
        )
        run_data[record.experiment_id] = {
            "edges": _edge_scores(edges_df),
        }

    metrics_df = pd.DataFrame(run_metrics)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)

    scaling_df = _aggregate_scaling(metrics_df)
    scaling_out = Path(args.scaling_out)
    scaling_out.parent.mkdir(parents=True, exist_ok=True)
    scaling_df.to_csv(scaling_out, index=False)

    robustness_df = _aggregate_robustness(run_data, metrics_df)
    robustness_out = Path(args.robustness_out)
    robustness_out.parent.mkdir(parents=True, exist_ok=True)
    robustness_df.to_csv(robustness_out, sep="\t", index=False)

    confidence_df = _aggregate_confidence(metrics_df, args.bootstrap_iters, args.bootstrap_seed)
    confidence_out = Path(args.confidence_out)
    confidence_out.parent.mkdir(parents=True, exist_ok=True)
    confidence_df.to_csv(confidence_out, index=False)

    pretraining_scales = _load_pretraining_scales(ROOT_DIR / "configs" / "scaling_checkpoints.yaml")
    fits_df = _compute_scaling_fits(scaling_df, pretraining_scales)
    fits_out = Path(args.fits_out)
    fits_out.parent.mkdir(parents=True, exist_ok=True)
    fits_df.to_csv(fits_out, index=False)

    baseline_ids = set(args.baseline_experiment_id or [])
    if baseline_ids:
        baseline_rows = []
        methods = [method.strip() for method in args.baseline_methods.split(",") if method.strip()]
        baseline_dir = Path(args.baseline_edges_dir)
        for record in records:
            if record.experiment_id not in baseline_ids:
                continue
            processed_path = Path(record.processed_h5ad)
            if processed_path.as_posix() not in adata_cache:
                adata_cache[processed_path.as_posix()] = sc.read_h5ad(processed_path)
            adata = adata_cache[processed_path.as_posix()]
            max_cells = _parse_optional_int(record.max_cells) if isinstance(record.max_cells, str) else record.max_cells
            sample_strategy = record.sample_strategy or "random"
            sample_seed = record.sample_seed if record.sample_seed is not None else record.seed
            adata_sampled = _sample_adata(adata, max_cells, sample_strategy, sample_seed, record.seed)
            source_mask, target_mask = _candidate_masks(
                adata_sampled,
                true_edges,
                alias_map,
                record.candidate_sources,
                record.candidate_targets,
            )
            net_cfg = NetworkConfig(
                top_k=record.top_k,
                threshold_percentile=record.threshold_percentile or 99.0,
            )
            for method in methods:
                if method == "coexpression":
                    baseline_edges = _compute_coexpression_edges(adata_sampled, net_cfg, source_mask, target_mask)
                elif method == "random":
                    baseline_edges = _compute_random_edges(
                        list(adata_sampled.var_names),
                        edge_count=record.top_k or 0,
                        source_mask=source_mask,
                        target_mask=target_mask,
                        seed=record.seed,
                    )
                elif method in {"grnboost2", "genie3"}:
                    baseline_edges = _load_baseline_edges(
                        baseline_dir,
                        record.experiment_id,
                        method,
                    )
                    if baseline_edges is None:
                        print(
                            f"Warning: missing baseline edges for {record.experiment_id} ({method}); skipping."
                        )
                        continue
                else:
                    raise ValueError(f"Unknown baseline method: {method}")
                metrics = _evaluate_edges(baseline_edges, adata_sampled, true_edges, alias_map)
                baseline_rows.append(
                    {
                        "experiment_id": record.experiment_id,
                        "method": method,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "n_edges": len(baseline_edges),
                    }
                )
        baseline_df = pd.DataFrame(baseline_rows)
        baseline_out = Path(args.baseline_out)
        baseline_out.parent.mkdir(parents=True, exist_ok=True)
        baseline_df.to_csv(baseline_out, index=False)

    control_ids = set(args.control_experiment_id or [])
    if control_ids:
        control_rows = []
        methods = [method.strip() for method in args.control_methods.split(",") if method.strip()]
        for record in records:
            if record.experiment_id not in control_ids:
                continue
            edges_path = Path(record.network_edges)
            if not edges_path.exists():
                continue
            processed_path = Path(record.processed_h5ad)
            if processed_path.as_posix() not in adata_cache:
                adata_cache[processed_path.as_posix()] = sc.read_h5ad(processed_path)
            adata = adata_cache[processed_path.as_posix()]
            edges_df = _load_edges(edges_path)
            for method in methods:
                if method == "permute_labels":
                    control_edges = _permute_gene_labels(adata, edges_df, record.seed)
                else:
                    raise ValueError(f"Unknown control method: {method}")
                metrics = _evaluate_edges(control_edges, adata, true_edges, alias_map)
                control_rows.append(
                    {
                        "experiment_id": record.experiment_id,
                        "method": method,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "n_edges": len(control_edges),
                    }
                )
        control_df = pd.DataFrame(control_rows)
        control_out = Path(args.control_out)
        control_out.parent.mkdir(parents=True, exist_ok=True)
        control_df.to_csv(control_out, index=False)


if __name__ == "__main__":
    main()
