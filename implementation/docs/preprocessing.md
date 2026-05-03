# Preprocessing — shared parameters

This document freezes the preprocessing choices referenced by the paper. A
reviewer can treat this page as ground truth for any claim the Methods
section makes about normalization, HVG selection, or gene filtering.

## Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Normalization | `depth` (scanpy `normalize_total`) | Matches Luecken & Theis 2019 best-practices; reproduces pre-revision numbers bit-for-bit. |
| `target_sum` | 10,000 | Convention in the scGPT / SCENIC / Tabula Sapiens workflows. |
| `log1p` | `True` | Standard for correlation-based scorers. |
| HVG flavor | `seurat_v3` | Matches published Tabula Sapiens analyses; falls back to `seurat` if raw counts are not in `X`. |
| `n_top_genes` | **2,000** | Previously the code used 600 in Axes 2/3 and larger counts elsewhere. The revision standardises at 2,000 HVGs with a `WP-13` sensitivity sweep over {300, 600, 1200, 2000, 3000}. |
| `min_cells_per_gene` | 10 | Applied at load time in `data.loader`. |

The canonical combination `depth + log1p + seurat_v3 HVG` is what the
paper's figures in Sections 3.1–3.5 are computed under. Any ablation
(`pearson_residuals`, `size_factor`, alternative HVG flavors) is
explicitly named in the caption of the relevant supplementary figure.

## Why this matters

Reviewer 3 noted that cell-wise depth normalization "increases the
variance of low-count cells" so that these cells dominate correlation
estimates. Work package WP-12 measures the practical effect of this by
recomputing the Axis-1 kidney curve and the Axis-3 rare-vs-abundant gap
under the three normalization schemes above. The result lives in
`outputs/wp12/normalization_ablation.csv` and is referenced in
Supplementary Figure S6.

## Provenance

Every pipeline run stamps the preprocessing parameters into
`adata.uns["fragility_preprocessing"]` and emits them in `provenance.json`
alongside the primary CSV. If a figure in the manuscript cannot be traced
back to a specific parameter set via these artefacts, that is a bug.
