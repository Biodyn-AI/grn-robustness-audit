#!/usr/bin/env python3
"""Integration-method sensitivity analysis (Harmony vs. none).

This script executes an end-to-end comparison of TF-target edge rankings derived
from unintegrated expression versus Harmony-corrected latent embeddings, then
reconstructed back to expression space.

Core outputs:
- Per-dataset edge rankings for baseline and Harmony variants
- Rank/overlap/sign-stability metrics
- Regulator-level sensitivity summaries
- External reference precision@k summaries (DoRothEA/TRRUST)
- Bootstrap uncertainty for top-k overlap
- Publication-ready figures and machine-readable summary tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import harmonypy as hm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.decomposition import PCA


# Reuse curated regulator/target panel from the existing leakage audit so that
# the new Harmony analysis is directly comparable to prior artifacts.
TF_LIST = [
    "STAT1",
    "STAT3",
    "STAT4",
    "STAT5A",
    "STAT5B",
    "STAT6",
    "IRF1",
    "IRF4",
    "IRF7",
    "IRF8",
    "NFKB1",
    "NFKB2",
    "RELA",
    "REL",
    "RELB",
    "TBX21",
    "GATA3",
    "RORC",
    "FOXP3",
    "BCL6",
    "SPI1",
    "RUNX1",
    "RUNX3",
    "ETS1",
    "ETS2",
    "MYC",
    "MAX",
    "JUN",
    "FOS",
    "JUNB",
    "JUND",
    "FOSL1",
    "FOSL2",
    "HIF1A",
    "EPAS1",
    "ARNT",
    "TP53",
    "RB1",
    "E2F1",
    "E2F4",
    "PAX5",
    "EBF1",
    "TCF7",
    "LEF1",
    "CEBPA",
    "CEBPB",
    "CEBPD",
    "SOX2",
    "SOX4",
    "SOX9",
    "KLF2",
    "KLF4",
    "KLF6",
    "NR4A1",
    "NR4A2",
    "NR4A3",
    "BATF",
    "BATF3",
    "MAF",
    "MAFB",
    "EOMES",
    "PRDM1",
    "ID2",
    "ID3",
    "NFE2L2",
    "BACH1",
    "BACH2",
    "ZEB1",
    "ZEB2",
    "SNAI1",
    "TWIST1",
    "FOXO1",
    "FOXO3",
    "AHR",
    "IKZF1",
    "IKZF3",
]

TARGET_POOL = [
    "CD3D",
    "CD3E",
    "CD4",
    "CD8A",
    "CD8B",
    "CD19",
    "MS4A1",
    "CD79A",
    "CD79B",
    "CD14",
    "CD68",
    "ITGAM",
    "ITGAX",
    "FCGR3A",
    "CSF1R",
    "IL2",
    "IL2RA",
    "IL7R",
    "IL6",
    "IL10",
    "IL17A",
    "IL21",
    "IL4",
    "IL13",
    "IFNG",
    "TNF",
    "TNFRSF9",
    "TNFRSF18",
    "GZMB",
    "GZMA",
    "GZMK",
    "PRF1",
    "GNLY",
    "NKG7",
    "CCL5",
    "CCR7",
    "CXCR4",
    "CXCL13",
    "CCL2",
    "CXCL10",
    "HLA-DRA",
    "HLA-DRB1",
    "HLA-A",
    "HLA-B",
    "HLA-C",
    "B2M",
    "LYZ",
    "S100A8",
    "S100A9",
    "S100A12",
    "SELL",
    "ITGA4",
    "ITGB1",
    "ICAM1",
    "VCAM1",
    "MKI67",
    "TOP2A",
    "CDK1",
    "PCNA",
    "BCL2",
    "BCL2L1",
    "MCL1",
    "BAX",
    "BAK1",
    "CTLA4",
    "PDCD1",
    "LAG3",
    "HAVCR2",
    "TIGIT",
    "TOX",
    "FN1",
    "COL1A1",
    "COL3A1",
    "VIM",
    "ACTA2",
    "CDH1",
    "EPCAM",
    "SFTPC",
    "SFTPB",
    "SFTPA1",
    "SCGB1A1",
    "MUC5AC",
    "FOXJ1",
    "PECAM1",
    "CDH5",
    "VWF",
    "KDR",
    "NOTCH1",
    "DLL4",
    "JAG1",
    "WNT5A",
    "CTNNB1",
    "TGFB1",
    "TGFBR1",
    "SMAD3",
    "SMAD7",
    "SOD2",
    "HMOX1",
    "NQO1",
    "GSTP1",
    "GAPDH",
    "ACTB",
    "RPL13A",
    "RPS18",
    "PTPRC",
    "THY1",
    "ENG",
    "NT5E",
]

INTEGRATION_KEYS: List[Tuple[str, str]] = [
    ("_scvi_batch", "harmony_batch"),
    ("donor_id", "harmony_donor"),
    ("method", "harmony_method"),
]

TOP_KS = [100, 250, 500, 1000]


def ensure_upper(values: Iterable[str]) -> List[str]:
    """Normalize gene symbols to uppercase string values."""
    out: List[str] = []
    for v in values:
        if pd.isna(v):
            continue
        out.append(str(v).strip().upper())
    return out


def build_symbol_index(adata: ad.AnnData) -> Tuple[np.ndarray, Dict[str, int], int]:
    """Return uppercase symbol vector plus first-occurrence lookup.

    If the dataset stores Ensembl ids in `var_names` and symbols in
    `var['feature_name']`, use `feature_name` as the symbol source.
    """
    if "feature_name" in adata.var.columns and str(adata.var_names[0]).startswith("ENSG"):
        symbols = ensure_upper(adata.var["feature_name"].astype(str).values)
    else:
        symbols = ensure_upper(adata.var_names.astype(str).values)

    first_index: Dict[str, int] = {}
    duplicate_count = 0
    for idx, sym in enumerate(symbols):
        if sym not in first_index:
            first_index[sym] = idx
        else:
            duplicate_count += 1
    return np.array(symbols), first_index, duplicate_count


def subset_cells(adata: ad.AnnData, max_cells: int, seed: int) -> ad.AnnData:
    """Randomly subsample cells to keep compute bounded and reproducible."""
    if adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[idx].copy()


def normalize_log1p_sparse(matrix: sparse.spmatrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    """Library-size normalize + log1p while preserving sparse format."""
    x = matrix.tocsr().astype(np.float32)
    cell_sums = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(target_sum, np.maximum(cell_sums, 1e-12), dtype=np.float32)
    x = x.multiply(scale[:, None])
    if not sparse.isspmatrix_csr(x):
        x = x.tocsr()
    x.data = np.log1p(x.data)
    return x


def select_panel_indices(
    symbol_to_index: Dict[str, int],
    tf_list: Sequence[str],
    target_pool: Sequence[str],
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Resolve TF/target genes present in the dataset and return ordered indices."""
    tf_genes = [g for g in ensure_upper(tf_list) if g in symbol_to_index]
    target_genes = [g for g in ensure_upper(target_pool) if g in symbol_to_index]

    edge_gene_order: List[str] = []
    seen = set()
    for g in tf_genes + target_genes:
        if g not in seen:
            seen.add(g)
            edge_gene_order.append(g)

    edge_gene_indices = [symbol_to_index[g] for g in edge_gene_order]
    return tf_genes, target_genes, edge_gene_order, edge_gene_indices


def select_hvg_indices(x_norm: sparse.csr_matrix, n_hvg: int) -> np.ndarray:
    """Select high-variance genes from normalized sparse expression."""
    mean = np.asarray(x_norm.mean(axis=0)).ravel()
    mean_sq = np.asarray(x_norm.power(2).mean(axis=0)).ravel()
    var = mean_sq - np.square(mean)
    var = np.where(np.isfinite(var), var, 0.0)
    if n_hvg >= len(var):
        return np.arange(len(var), dtype=int)
    return np.argpartition(var, -n_hvg)[-n_hvg:]


def compute_edge_scores_from_expr(
    edge_expr: np.ndarray,
    edge_gene_order: Sequence[str],
    tf_genes: Sequence[str],
    target_genes: Sequence[str],
) -> pd.DataFrame:
    """Compute TF-target correlation scores using vectorized matrix operations."""
    gene_to_col = {g: i for i, g in enumerate(edge_gene_order)}

    tf_names = [g for g in tf_genes if g in gene_to_col]
    tg_names = [g for g in target_genes if g in gene_to_col]

    tf_cols = np.array([gene_to_col[g] for g in tf_names], dtype=int)
    tg_cols = np.array([gene_to_col[g] for g in tg_names], dtype=int)

    x_tf = edge_expr[:, tf_cols].astype(np.float64, copy=False)
    x_tg = edge_expr[:, tg_cols].astype(np.float64, copy=False)

    tf_mean = x_tf.mean(axis=0)
    tg_mean = x_tg.mean(axis=0)
    tf_std = x_tf.std(axis=0, ddof=1)
    tg_std = x_tg.std(axis=0, ddof=1)

    tf_valid = tf_std > 1e-8
    tg_valid = tg_std > 1e-8

    tf_names = [name for name, keep in zip(tf_names, tf_valid) if keep]
    tg_names = [name for name, keep in zip(tg_names, tg_valid) if keep]

    x_tf = x_tf[:, tf_valid]
    x_tg = x_tg[:, tg_valid]
    tf_mean = tf_mean[tf_valid]
    tg_mean = tg_mean[tg_valid]
    tf_std = tf_std[tf_valid]
    tg_std = tg_std[tg_valid]

    z_tf = (x_tf - tf_mean) / tf_std
    z_tg = (x_tg - tg_mean) / tg_std

    corr = (z_tf.T @ z_tg) / max(edge_expr.shape[0] - 1, 1)

    records: List[Dict[str, object]] = []
    for i, tf in enumerate(tf_names):
        for j, tg in enumerate(tg_names):
            if tf == tg:
                continue
            score = float(np.clip(corr[i, j], -1.0, 1.0))
            if not np.isfinite(score):
                continue
            records.append(
                {
                    "edge": f"{tf}->{tg}",
                    "tf": tf,
                    "target": tg,
                    "score": score,
                    "abs_score": abs(score),
                }
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("abs_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def fit_pca_latent(x_pca_input: np.ndarray, n_components: int, seed: int):
    """Fit PCA on standardized expression and return components/latent scores."""
    means = x_pca_input.mean(axis=0)
    stds = x_pca_input.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    x_scaled = (x_pca_input - means) / stds
    n_comp = min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1] - 1)
    n_comp = max(n_comp, 2)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    pcs = pca.fit_transform(x_scaled)
    return {
        "means": means,
        "stds": stds,
        "pca": pca,
        "pcs": pcs,
    }


def harmony_correct_edge_expression(
    latent: Dict[str, object],
    labels: Sequence[str],
    edge_positions_in_pca: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Run Harmony on PCA space and reconstruct corrected edge-gene expression."""
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2:
        pcs = latent["pcs"]
    else:
        meta = pd.DataFrame({"integration_key": labels})
        harmony_output = hm.run_harmony(
            latent["pcs"],
            meta,
            vars_use=["integration_key"],
            max_iter_harmony=20,
            random_state=random_state,
            verbose=False,
        )
        # Harmonypy can expose Z_corr either as (n_pcs, n_cells) or
        # (n_cells, n_pcs) depending on version/path; normalize to (n_cells, n_pcs).
        z_corr = np.asarray(harmony_output.Z_corr)
        expected_cells = latent["pcs"].shape[0]
        if z_corr.shape[0] == expected_cells:
            pcs = z_corr
        elif z_corr.shape[1] == expected_cells:
            pcs = z_corr.T
        else:
            raise ValueError(
                "Unexpected Harmony Z_corr shape "
                f"{z_corr.shape} for expected n_cells={expected_cells}."
            )

    pca: PCA = latent["pca"]
    means: np.ndarray = latent["means"]
    stds: np.ndarray = latent["stds"]

    edge_components = pca.components_[:, edge_positions_in_pca]
    edge_scaled_corr = pcs @ edge_components
    edge_expr_corr = edge_scaled_corr * stds[edge_positions_in_pca] + means[edge_positions_in_pca]

    # Harmony reconstruction can produce tiny numeric anomalies; stabilize.
    return np.nan_to_num(edge_expr_corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def overlap_metrics(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    top_ks: Sequence[int],
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute global and top-k edge-ranking agreement metrics."""
    merged = baseline_df.merge(method_df, on="edge", suffixes=("_base", "_method"))
    merged["rank_shift_abs"] = (merged["rank_base"] - merged["rank_method"]).abs()
    merged["score_delta_abs"] = (merged["score_base"] - merged["score_method"]).abs()
    merged["sign_flip"] = (merged["score_base"] * merged["score_method"] < 0).astype(int)

    score_pearson = pearsonr(merged["score_base"], merged["score_method"])[0]
    rank_spearman = spearmanr(merged["rank_base"], merged["rank_method"]).correlation

    summary = {
        "score_pearson": float(score_pearson),
        "rank_spearman": float(rank_spearman),
        "sign_flip_rate": float(merged["sign_flip"].mean()),
        "median_abs_rank_shift": float(merged["rank_shift_abs"].median()),
        "mean_abs_rank_shift": float(merged["rank_shift_abs"].mean()),
        "mean_abs_score_delta": float(merged["score_delta_abs"].mean()),
    }

    baseline_sorted = baseline_df.sort_values("rank")["edge"].tolist()
    method_sorted = method_df.sort_values("rank")["edge"].tolist()

    topk_rows: List[Dict[str, float]] = []
    for k in top_ks:
        b_set = set(baseline_sorted[:k])
        m_set = set(method_sorted[:k])
        overlap = len(b_set & m_set)
        union = len(b_set | m_set)

        topk_rows.append(
            {
                "k": int(k),
                "overlap_count": int(overlap),
                "overlap_fraction": float(overlap / max(k, 1)),
                "jaccard": float(overlap / max(union, 1)),
            }
        )

    tf_rows = (
        merged.groupby("tf_base", as_index=False)
        .agg(
            mean_abs_rank_shift=("rank_shift_abs", "mean"),
            median_abs_rank_shift=("rank_shift_abs", "median"),
            sign_flip_rate=("sign_flip", "mean"),
            mean_abs_score_delta=("score_delta_abs", "mean"),
            n_edges=("edge", "size"),
        )
        .rename(columns={"tf_base": "tf"})
        .sort_values("mean_abs_rank_shift", ascending=False)
    )

    return summary, pd.DataFrame(topk_rows), tf_rows


def load_reference_sets(repo_root: Path) -> Dict[str, set]:
    """Load directed reference edges for external support checks."""
    ref_dir = repo_root / "single_cell_mechinterp" / "external" / "networks"

    dorothea = pd.read_csv(ref_dir / "dorothea_human.tsv", sep="\t")
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()

    confidence_str = dorothea["confidence"].astype(str)
    ab_mask = confidence_str.str.contains(r"(?:^|;)(?:A|B)(?:$|;)", regex=True)
    dorothea_ab = set(
        (dorothea.loc[ab_mask, "source"])
        + "->"
        + dorothea.loc[ab_mask, "target"]
    )
    dorothea_abcd = set(dorothea["source"] + "->" + dorothea["target"])

    dorothea_chip = pd.read_csv(ref_dir / "dorothea_chipseq_human.tsv", sep="\t")
    dorothea_chip["source"] = dorothea_chip["source"].astype(str).str.upper()
    dorothea_chip["target"] = dorothea_chip["target"].astype(str).str.upper()
    chip_conf = dorothea_chip["confidence"].astype(str)
    chip_ab_mask = chip_conf.str.contains(r"(?:^|;)(?:A|B)(?:$|;)", regex=True)
    dorothea_chip_ab = set(
        (dorothea_chip.loc[chip_ab_mask, "source"])
        + "->"
        + dorothea_chip.loc[chip_ab_mask, "target"]
    )
    dorothea_chip_all = set(dorothea_chip["source"] + "->" + dorothea_chip["target"])

    trrust = pd.read_csv(
        ref_dir / "trrust_human.tsv",
        sep="\t",
        header=None,
        names=["source", "target", "mode", "pmid"],
    )
    trrust["source"] = trrust["source"].astype(str).str.upper()
    trrust["target"] = trrust["target"].astype(str).str.upper()
    trrust_edges = set(trrust["source"] + "->" + trrust["target"])

    return {
        "dorothea_ab": dorothea_ab,
        "dorothea_abcd": dorothea_abcd,
        "dorothea_chipseq_ab": dorothea_chip_ab,
        "dorothea_chipseq_all": dorothea_chip_all,
        "trrust": trrust_edges,
    }


def precision_at_k(edge_df: pd.DataFrame, reference_edges: set, ks: Sequence[int]) -> pd.DataFrame:
    """Compute directed-edge precision@k against a reference set."""
    ranked_edges = edge_df.sort_values("rank")["edge"].tolist()
    rows: List[Dict[str, float]] = []
    for k in ks:
        top = ranked_edges[:k]
        hits = sum(1 for e in top if e in reference_edges)
        rows.append(
            {
                "k": int(k),
                "hits": int(hits),
                "precision": float(hits / max(k, 1)),
            }
        )
    return pd.DataFrame(rows)


def bh_fdr_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction with NaN-safe behavior."""
    pvals_arr = np.array(pvals, dtype=float)
    out = np.full_like(pvals_arr, np.nan, dtype=float)
    valid = np.where(np.isfinite(pvals_arr))[0]
    if valid.size == 0:
        return out

    pv = pvals_arr[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = float(len(ranked))
    q = ranked * n / (np.arange(1, len(ranked) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    restored = np.empty_like(q)
    restored[order] = q
    out[valid] = restored
    return out


def swap_enrichment_test(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    support_edges: set,
    k: int,
) -> Dict[str, float]:
    """Paired sign-style test on gained vs lost supported edges in top-k swaps."""
    b_set = set(baseline_df.sort_values("rank").head(k)["edge"])
    m_set = set(method_df.sort_values("rank").head(k)["edge"])

    baseline_hits = len(b_set & support_edges)
    method_hits = len(m_set & support_edges)
    gained_hits = len((m_set - b_set) & support_edges)
    lost_hits = len((b_set - m_set) & support_edges)
    total_changed = gained_hits + lost_hits

    if total_changed > 0:
        p_two = float(binomtest(gained_hits, total_changed, p=0.5, alternative="two-sided").pvalue)
        p_greater = float(binomtest(gained_hits, total_changed, p=0.5, alternative="greater").pvalue)
    else:
        p_two = np.nan
        p_greater = np.nan

    return {
        "k": int(k),
        "baseline_hits": int(baseline_hits),
        "method_hits": int(method_hits),
        "delta_hits": int(method_hits - baseline_hits),
        "delta_precision": float((method_hits - baseline_hits) / max(k, 1)),
        "gained_hits": int(gained_hits),
        "lost_hits": int(lost_hits),
        "swap_test_n": int(total_changed),
        "p_two_sided": p_two,
        "p_greater": p_greater,
    }


def load_perturbation_validation_tables(repo_root: Path) -> Dict[str, pd.DataFrame]:
    """Load perturbation-backed validation tables from prior leakage subproject."""
    pert_dir = repo_root / "subproject_33_batch_donor_leakage_audit" / "implementation"
    mapping = {
        "Immune": pert_dir / "phase2_perturbation_combined_Immune.csv",
        "Lung": pert_dir / "phase2_perturbation_combined_Lung.csv",
    }
    out: Dict[str, pd.DataFrame] = {}
    for dataset, path in mapping.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["tf"] = df["tf"].astype(str).str.upper()
        df["target"] = df["target"].astype(str).str.upper()
        df["edge"] = df["tf"] + "->" + df["target"]
        if "pert_significant" in df.columns:
            df["pert_significant"] = df["pert_significant"].astype(bool)
        if "pert_abs_delta" in df.columns:
            df["pert_abs_delta"] = pd.to_numeric(df["pert_abs_delta"], errors="coerce")
        out[dataset] = df
    return out


def perturbation_topk_metrics(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    perturb_df: pd.DataFrame,
    ks: Sequence[int],
) -> pd.DataFrame:
    """Evaluate top-k agreement with perturbation-significant edges."""
    pert_map = perturb_df.set_index("edge")
    sig_edges = set(perturb_df.loc[perturb_df["pert_significant"], "edge"])
    rows: List[Dict[str, float]] = []

    for k in ks:
        b_top = set(baseline_df.sort_values("rank").head(k)["edge"])
        m_top = set(method_df.sort_values("rank").head(k)["edge"])

        for label, top in [("baseline", b_top), ("method", m_top)]:
            matched = sorted(top & set(pert_map.index))
            matched_df = pert_map.loc[matched] if matched else pd.DataFrame(columns=pert_map.columns)
            sig_hits = int(matched_df["pert_significant"].sum()) if not matched_df.empty else 0
            rows.append(
                {
                    "k": int(k),
                    "set_type": label,
                    "topk_size": int(k),
                    "matched_edges": int(len(matched)),
                    "pert_significant_hits": int(sig_hits),
                    "pert_significant_rate": float(sig_hits / max(len(matched), 1)),
                    "mean_pert_abs_delta": float(matched_df["pert_abs_delta"].mean()) if len(matched) > 0 else np.nan,
                }
            )

        swap_stats = swap_enrichment_test(
            baseline_df=baseline_df,
            method_df=method_df,
            support_edges=sig_edges,
            k=k,
        )
        swap_stats["set_type"] = "swap_test"
        swap_stats["matched_edges"] = np.nan
        swap_stats["pert_significant_hits"] = np.nan
        swap_stats["pert_significant_rate"] = np.nan
        swap_stats["mean_pert_abs_delta"] = np.nan
        rows.append(swap_stats)

    return pd.DataFrame(rows)


def bootstrap_topk_overlap(
    baseline_edge_expr: np.ndarray,
    method_edge_expr: np.ndarray,
    edge_gene_order: Sequence[str],
    tf_genes: Sequence[str],
    target_genes: Sequence[str],
    n_boot: int,
    k: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap uncertainty for top-k overlap under cell resampling."""
    rng = np.random.default_rng(seed)
    n_cells = baseline_edge_expr.shape[0]
    sample_size = int(0.8 * n_cells)

    rows: List[Dict[str, float]] = []
    for b in range(n_boot):
        idx = rng.choice(n_cells, size=sample_size, replace=True)

        base_df = compute_edge_scores_from_expr(
            baseline_edge_expr[idx], edge_gene_order, tf_genes, target_genes
        )
        meth_df = compute_edge_scores_from_expr(
            method_edge_expr[idx], edge_gene_order, tf_genes, target_genes
        )

        b_set = set(base_df.sort_values("rank").head(k)["edge"])
        m_set = set(meth_df.sort_values("rank").head(k)["edge"])
        overlap = len(b_set & m_set)

        rows.append(
            {
                "bootstrap_iter": b,
                "k": int(k),
                "overlap_count": int(overlap),
                "overlap_fraction": float(overlap / k),
            }
        )

    return pd.DataFrame(rows)


def make_figures(
    out_fig_dir: Path,
    topk_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    tf_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    pcs_sweep_df: pd.DataFrame,
    perturb_df: pd.DataFrame,
) -> None:
    """Create paper-ready summary figures."""
    sns.set_theme(style="whitegrid", context="talk")

    # Figure 1: Top-k overlap trajectories.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, dataset in zip(axes, sorted(topk_df["dataset"].unique())):
        sub = topk_df[topk_df["dataset"] == dataset]
        sns.lineplot(
            data=sub,
            x="k",
            y="overlap_fraction",
            hue="method",
            marker="o",
            ax=ax,
        )
        ax.set_title(dataset)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Top-k")
        ax.set_ylabel("Overlap Fraction")
        ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("Top-k Edge Overlap: Harmony vs Unintegrated", y=1.03)
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig1_topk_overlap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Rank stability summary.
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_rank = summary_df.pivot(index="dataset", columns="method", values="rank_spearman")
    sns.heatmap(pivot_rank, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Rank Spearman Correlation vs Baseline")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig2_rank_spearman_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Sign-flip rates.
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_flip = summary_df.pivot(index="dataset", columns="method", values="sign_flip_rate")
    sns.heatmap(pivot_flip, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=0.5, ax=ax)
    ax.set_title("Sign Flip Rate vs Baseline")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig3_sign_flip_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Most integration-sensitive regulators (batch integration only).
    tf_batch = tf_df[tf_df["method"] == "harmony_batch"].copy()
    tf_batch_summary = (
        tf_batch.groupby("tf", as_index=False)["mean_abs_rank_shift"].mean().sort_values(
            "mean_abs_rank_shift", ascending=False
        )
    )
    top_tf = tf_batch_summary.head(15)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=top_tf,
        x="mean_abs_rank_shift",
        y="tf",
        color="#2B7A78",
        ax=ax,
    )
    ax.set_title("Top 15 Regulators by Mean Absolute Rank Shift (Harmony Batch)")
    ax.set_xlabel("Mean Absolute Rank Shift")
    ax.set_ylabel("TF")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig4_tf_sensitivity_harmony_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: External reference precision@k.
    # Keep k=500 for readability.
    ref_k500 = ref_df[ref_df["k"] == 500].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=ref_k500,
        x="dataset",
        y="precision",
        hue="method_ref",
        ax=ax,
    )
    ax.set_title("Precision@500 vs External References")
    ax.set_ylabel("Precision@500")
    ax.set_xlabel("Dataset")
    ax.legend(title="Method / Reference", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig5_reference_precision_k500.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: Bootstrap overlap uncertainty.
    if not bootstrap_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=bootstrap_df,
            x="dataset",
            y="overlap_fraction",
            color="#3A86FF",
            ax=ax,
        )
        sns.stripplot(
            data=bootstrap_df,
            x="dataset",
            y="overlap_fraction",
            color="black",
            alpha=0.25,
            size=3,
            ax=ax,
        )
        ax.set_title("Bootstrap Distribution: Top-500 Overlap (Baseline vs Harmony Batch)")
        ax.set_ylabel("Overlap Fraction")
        ax.set_xlabel("Dataset")
        fig.tight_layout()
        fig.savefig(out_fig_dir / "fig6_bootstrap_overlap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Figure 7: Harmony batch sensitivity to number of PCs.
    if not pcs_sweep_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.lineplot(
            data=pcs_sweep_df,
            x="n_pcs",
            y="top500_overlap_fraction",
            hue="dataset",
            marker="o",
            ax=axes[0],
        )
        axes[0].set_title("Top-500 Overlap vs Number of PCs (Harmony Batch)")
        axes[0].set_xlabel("PCA Components")
        axes[0].set_ylabel("Top-500 Overlap Fraction")
        axes[0].set_ylim(0.0, 1.0)

        sns.lineplot(
            data=pcs_sweep_df,
            x="n_pcs",
            y="precision500_dorothea_abcd",
            hue="dataset",
            marker="o",
            ax=axes[1],
        )
        axes[1].set_title("DoRothEA-ABCD Precision@500 vs Number of PCs")
        axes[1].set_xlabel("PCA Components")
        axes[1].set_ylabel("Precision@500")
        axes[1].set_ylim(bottom=0.0)

        fig.tight_layout()
        fig.savefig(out_fig_dir / "fig7_harmony_batch_pcs_sweep.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Figure 8: Perturbation-significant hit rate at k=500.
    if not perturb_df.empty:
        pert_plot = perturb_df[
            (perturb_df["k"] == 500)
            & (perturb_df["set_type"] == "method")
        ].copy()
        if not pert_plot.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.barplot(
                data=pert_plot,
                x="dataset",
                y="pert_significant_rate",
                hue="method",
                ax=ax,
            )
            ax.set_title("Perturbation-Significant Rate in Top-500 Edges")
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Significant Rate (among matched perturbation edges)")
            ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(out_fig_dir / "fig8_perturbation_significant_rate_k500.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integration-method sensitivity analysis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--pcs-sweep", type=str, default="30,50,80")
    parser.add_argument("--bootstrap-reps", type=int, default=25)
    parser.add_argument("--bootstrap-k", type=int, default=500)
    parser.add_argument("--max-cells-lung", type=int, default=20000)
    parser.add_argument("--max-cells-immune", type=int, default=20000)
    parser.add_argument("--max-cells-kidney", type=int, default=20000)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    repo_root = script_path.parents[2]

    out_data = project_root / "data"
    out_fig = project_root / "reports" / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    pcs_sweep_values = sorted(
        {
            int(v.strip())
            for v in args.pcs_sweep.split(",")
            if v.strip()
        }
    )
    if args.n_pcs not in pcs_sweep_values:
        pcs_sweep_values.append(args.n_pcs)
        pcs_sweep_values = sorted(set(pcs_sweep_values))

    dataset_cfg = [
        {
            "dataset": "Immune",
            "path": repo_root / "single_cell_mechinterp" / "data" / "raw" / "tabula_sapiens_immune_subset_20000.h5ad",
            "max_cells": args.max_cells_immune,
        },
        {
            "dataset": "Lung",
            "path": repo_root / "single_cell_mechinterp" / "data" / "raw" / "tabula_sapiens_lung.h5ad",
            "max_cells": args.max_cells_lung,
        },
        {
            "dataset": "Kidney",
            "path": repo_root / "single_cell_mechinterp" / "data" / "raw" / "tabula_sapiens_kidney.h5ad",
            "max_cells": args.max_cells_kidney,
        },
    ]

    reference_sets = load_reference_sets(repo_root)
    perturbation_tables = load_perturbation_validation_tables(repo_root)

    summary_rows: List[Dict[str, object]] = []
    topk_rows: List[Dict[str, object]] = []
    tf_rows: List[Dict[str, object]] = []
    ref_rows: List[Dict[str, object]] = []
    ref_swap_rows: List[Dict[str, object]] = []
    perturbation_rows: List[Dict[str, object]] = []
    pcs_sweep_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []
    dataset_meta_rows: List[Dict[str, object]] = []

    for cfg in dataset_cfg:
        dataset = cfg["dataset"]
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")

        adata = ad.read_h5ad(cfg["path"])
        adata = subset_cells(adata, cfg["max_cells"], args.seed)
        print(f"Cells x genes: {adata.n_obs} x {adata.n_vars}")

        symbols, sym_to_idx, dup_count = build_symbol_index(adata)
        tf_genes, target_genes, edge_gene_order, edge_gene_indices = select_panel_indices(
            sym_to_idx,
            TF_LIST,
            TARGET_POOL,
        )

        if len(tf_genes) < 5 or len(target_genes) < 10:
            raise RuntimeError(f"Insufficient TF/target coverage for {dataset}.")

        if sparse.issparse(adata.X):
            x_raw = adata.X.tocsr()
        else:
            x_raw = sparse.csr_matrix(np.asarray(adata.X))

        x_norm = normalize_log1p_sparse(x_raw)

        # Baseline edge expression matrix (only panel genes).
        baseline_edge_expr = x_norm[:, edge_gene_indices].toarray().astype(np.float32)
        baseline_edges = compute_edge_scores_from_expr(
            baseline_edge_expr,
            edge_gene_order,
            tf_genes,
            target_genes,
        )
        baseline_edges.to_csv(out_data / f"{dataset.lower()}_edges_baseline.csv", index=False)

        edges_by_method: Dict[str, pd.DataFrame] = {"baseline": baseline_edges}

        # External support for baseline.
        for ref_name, ref_set in reference_sets.items():
            p_df = precision_at_k(baseline_edges, ref_set, TOP_KS)
            p_df["dataset"] = dataset
            p_df["method"] = "baseline"
            p_df["reference"] = ref_name
            p_df["method_ref"] = f"baseline/{ref_name}"
            ref_rows.extend(p_df.to_dict("records"))

        # Build PCA gene set: HVG union edge genes.
        hvg_idx = select_hvg_indices(x_norm, args.n_hvg)
        pca_gene_indices = np.array(sorted(set(hvg_idx.tolist()) | set(edge_gene_indices)), dtype=int)
        edge_pos_map = {gene_idx: i for i, gene_idx in enumerate(pca_gene_indices.tolist())}
        edge_positions = np.array([edge_pos_map[i] for i in edge_gene_indices], dtype=int)

        x_pca_input = x_norm[:, pca_gene_indices].toarray().astype(np.float32)
        latent = fit_pca_latent(x_pca_input, n_components=args.n_pcs, seed=args.seed)

        method_edge_expr_cache: Dict[str, np.ndarray] = {}

        for obs_key, method_name in INTEGRATION_KEYS:
            if obs_key not in adata.obs.columns:
                print(f"Skipping {method_name}: obs key '{obs_key}' not found.")
                continue

            labels = adata.obs[obs_key].astype(str).values
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                print(f"Skipping {method_name}: '{obs_key}' has a single class.")
                continue

            print(f"Running {method_name} (key={obs_key}, n_classes={len(unique_labels)})")
            corrected_edge_expr = harmony_correct_edge_expression(
                latent,
                labels,
                edge_positions,
                random_state=args.seed,
            )
            method_edge_expr_cache[method_name] = corrected_edge_expr

            method_edges = compute_edge_scores_from_expr(
                corrected_edge_expr,
                edge_gene_order,
                tf_genes,
                target_genes,
            )
            edges_by_method[method_name] = method_edges
            method_edges.to_csv(out_data / f"{dataset.lower()}_edges_{method_name}.csv", index=False)

            metric_summary, topk_df, tf_df = overlap_metrics(
                baseline_edges,
                method_edges,
                TOP_KS,
            )
            metric_summary.update(
                {
                    "dataset": dataset,
                    "method": method_name,
                    "integration_key": obs_key,
                    "n_integration_classes": int(len(unique_labels)),
                }
            )
            summary_rows.append(metric_summary)

            topk_df["dataset"] = dataset
            topk_df["method"] = method_name
            topk_rows.extend(topk_df.to_dict("records"))

            tf_df["dataset"] = dataset
            tf_df["method"] = method_name
            tf_rows.extend(tf_df.to_dict("records"))

            for ref_name, ref_set in reference_sets.items():
                p_df = precision_at_k(method_edges, ref_set, TOP_KS)
                p_df["dataset"] = dataset
                p_df["method"] = method_name
                p_df["reference"] = ref_name
                p_df["method_ref"] = f"{method_name}/{ref_name}"
                ref_rows.extend(p_df.to_dict("records"))

        # Paired significance test for reference-support gains vs baseline.
        for method_name, method_edges in edges_by_method.items():
            if method_name == "baseline":
                continue
            for ref_name, ref_set in reference_sets.items():
                for k in TOP_KS:
                    stats = swap_enrichment_test(
                        baseline_df=baseline_edges,
                        method_df=method_edges,
                        support_edges=ref_set,
                        k=k,
                    )
                    stats.update(
                        {
                            "dataset": dataset,
                            "method": method_name,
                            "reference": ref_name,
                        }
                    )
                    ref_swap_rows.append(stats)

        # Perturbation-backed top-k validation (new external data in this loop).
        if dataset in perturbation_tables:
            perturb_df = perturbation_tables[dataset]
            for method_name, method_edges in edges_by_method.items():
                if method_name == "baseline":
                    continue
                p_eval = perturbation_topk_metrics(
                    baseline_df=baseline_edges,
                    method_df=method_edges,
                    perturb_df=perturb_df,
                    ks=TOP_KS,
                )
                p_eval["dataset"] = dataset
                p_eval["method"] = method_name
                perturbation_rows.extend(p_eval.to_dict("records"))

        # Harmony batch hyperparameter sensitivity sweep.
        if "_scvi_batch" in adata.obs.columns and adata.obs["_scvi_batch"].nunique() > 1:
            batch_labels = adata.obs["_scvi_batch"].astype(str).values
            for n_pcs in pcs_sweep_values:
                latent_sweep = fit_pca_latent(x_pca_input, n_components=n_pcs, seed=args.seed)
                corrected_sweep = harmony_correct_edge_expression(
                    latent_sweep,
                    batch_labels,
                    edge_positions,
                    random_state=args.seed,
                )
                edges_sweep = compute_edge_scores_from_expr(
                    corrected_sweep,
                    edge_gene_order,
                    tf_genes,
                    target_genes,
                )
                sweep_summary, sweep_topk, _ = overlap_metrics(
                    baseline_df=baseline_edges,
                    method_df=edges_sweep,
                    top_ks=TOP_KS,
                )
                top500_overlap = float(
                    sweep_topk.loc[sweep_topk["k"] == 500, "overlap_fraction"].iloc[0]
                )
                precision500_dorothea = float(
                    precision_at_k(edges_sweep, reference_sets["dorothea_abcd"], [500])["precision"].iloc[0]
                )
                pcs_sweep_rows.append(
                    {
                        "dataset": dataset,
                        "method": "harmony_batch",
                        "n_pcs": int(n_pcs),
                        "rank_spearman": float(sweep_summary["rank_spearman"]),
                        "score_pearson": float(sweep_summary["score_pearson"]),
                        "sign_flip_rate": float(sweep_summary["sign_flip_rate"]),
                        "top500_overlap_fraction": top500_overlap,
                        "precision500_dorothea_abcd": precision500_dorothea,
                    }
                )

        # Bootstrap only for the primary batch-integration comparison.
        if "harmony_batch" in method_edge_expr_cache and args.bootstrap_reps > 0:
            print(f"Running bootstrap overlap for {dataset} (n={args.bootstrap_reps})")
            boot_df = bootstrap_topk_overlap(
                baseline_edge_expr,
                method_edge_expr_cache["harmony_batch"],
                edge_gene_order,
                tf_genes,
                target_genes,
                n_boot=args.bootstrap_reps,
                k=args.bootstrap_k,
                seed=args.seed,
            )
            boot_df["dataset"] = dataset
            boot_df["method"] = "harmony_batch"
            bootstrap_rows.extend(boot_df.to_dict("records"))

        dataset_meta_rows.append(
            {
                "dataset": dataset,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_tf_present": int(len(tf_genes)),
                "n_target_present": int(len(target_genes)),
                "n_edges": int(len(baseline_edges)),
                "duplicate_symbols": int(dup_count),
                "n_donors": int(adata.obs["donor_id"].nunique()) if "donor_id" in adata.obs else np.nan,
                "n_batches": int(adata.obs["_scvi_batch"].nunique()) if "_scvi_batch" in adata.obs else np.nan,
                "n_methods": int(adata.obs["method"].nunique()) if "method" in adata.obs else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    topk_df = pd.DataFrame(topk_rows)
    tf_df = pd.DataFrame(tf_rows)
    ref_df = pd.DataFrame(ref_rows)
    ref_swap_df = pd.DataFrame(ref_swap_rows)
    perturbation_df = pd.DataFrame(perturbation_rows)
    pcs_sweep_df = pd.DataFrame(pcs_sweep_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    dataset_meta_df = pd.DataFrame(dataset_meta_rows)

    if not ref_swap_df.empty:
        ref_swap_df["p_two_sided_fdr"] = bh_fdr_adjust(ref_swap_df["p_two_sided"].values)
        ref_swap_df["p_greater_fdr"] = bh_fdr_adjust(ref_swap_df["p_greater"].values)

    if not perturbation_df.empty:
        swap_mask = perturbation_df["set_type"] == "swap_test"
        if swap_mask.any():
            perturbation_df.loc[swap_mask, "p_two_sided_fdr"] = bh_fdr_adjust(
                perturbation_df.loc[swap_mask, "p_two_sided"].values
            )
            perturbation_df.loc[swap_mask, "p_greater_fdr"] = bh_fdr_adjust(
                perturbation_df.loc[swap_mask, "p_greater"].values
            )

    summary_df.to_csv(out_data / "integration_summary.csv", index=False)
    topk_df.to_csv(out_data / "topk_overlap.csv", index=False)
    tf_df.to_csv(out_data / "tf_sensitivity.csv", index=False)
    ref_df.to_csv(out_data / "reference_precision.csv", index=False)
    ref_swap_df.to_csv(out_data / "reference_swap_significance.csv", index=False)
    perturbation_df.to_csv(out_data / "perturbation_validation.csv", index=False)
    pcs_sweep_df.to_csv(out_data / "harmony_batch_pcs_sweep.csv", index=False)
    bootstrap_df.to_csv(out_data / "bootstrap_overlap.csv", index=False)
    dataset_meta_df.to_csv(out_data / "dataset_metadata.csv", index=False)

    make_figures(
        out_fig_dir=out_fig,
        topk_df=topk_df,
        summary_df=summary_df,
        tf_df=tf_df,
        ref_df=ref_df,
        bootstrap_df=bootstrap_df,
        pcs_sweep_df=pcs_sweep_df,
        perturb_df=perturbation_df,
    )

    # Build a concise machine-readable findings summary for downstream writing.
    findings = {
        "settings": {
            "seed": args.seed,
            "n_hvg": args.n_hvg,
            "n_pcs": args.n_pcs,
            "pcs_sweep": pcs_sweep_values,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_k": args.bootstrap_k,
        },
        "datasets": dataset_meta_rows,
        "best_rank_stability": (
            summary_df.sort_values("rank_spearman", ascending=False)
            .head(10)
            .to_dict("records")
            if not summary_df.empty
            else []
        ),
        "highest_sign_flip": (
            summary_df.sort_values("sign_flip_rate", ascending=False)
            .head(10)
            .to_dict("records")
            if not summary_df.empty
            else []
        ),
        "reference_swap_significance_top": (
            ref_swap_df.sort_values("delta_precision", ascending=False)
            .head(20)
            .to_dict("records")
            if not ref_swap_df.empty
            else []
        ),
        "perturbation_topk_summary": (
            perturbation_df[
                (perturbation_df["k"] == 500) & (perturbation_df["set_type"] == "method")
            ]
            .to_dict("records")
            if not perturbation_df.empty
            else []
        ),
    }

    with open(out_data / "summary_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print("\nAnalysis complete.")
    print(f"Saved tables to: {out_data}")
    print(f"Saved figures to: {out_fig}")


if __name__ == "__main__":
    main()
