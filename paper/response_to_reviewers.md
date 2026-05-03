# Response to reviewers — *Five Axes of Fragility*

**Manuscript ID:** BIOINF-2026-0446
**Title:** Five Axes of Fragility: A Systematic Robustness Audit of Gene-Regulatory-Network Inference from Single-Cell Foundation Models
**Decision date:** 22-Apr-2026 (major revision)
**Author:** Ihor Kendiukhov (kenduhov.ig@gmail.com → **ihor.kendiukhov@student.uni-tuebingen.de**)

We thank Associate Editor Lenore Cowen and the four reviewers for the careful, constructive critique. The revision is substantial: we re-ran every axis on a larger and more diverse cohort set, added a real foundation-model scorer (scGPT) plus three new classical scorers, expanded integration analysis from one tool to three, rebuilt the composite metric with an empirical null and a weight-simplex sweep, threshold-graded the rare-cell finding across 5,880 grid points, added an end-to-end reproduction package with `make audit`, and restructured the AI-assisted review protocol with a 27-issue full-census audit logged in Supplementary Methods S2.

Cross-references below are to the **revised** manuscript. Section numbers (e.g. §3.6) and table/figure numbers (e.g. Table 4, Fig 5) all refer to the revised PDF compiled with the OUP `oup-authoring-template`.

---

## Editor-level requirements

| Editor request | Status | Where in revision |
|---|---|---|
| Archival code DOI on Zenodo / Figshare / Software Heritage / CRAN / Bioconductor | **Done.** Code archived at Zenodo: **`10.5281/zenodo.20009065`** | Listed in Abstract Availability and in §"Data and Code Availability"; live repo at <https://github.com/Biodyn-AI/grn-robustness-audit> |
| Availability statement includes archival DOI **and** repo link | **Done.** Both are in the Abstract Availability snippet and the Data and Code Availability section | Abstract; §"Data and Code Availability" |
| Conflict of Interest declared | "None declared." | §"Conflict of Interest" |
| LaTeX template compliance for typeset version | **Migrated** from generic `article` to `\documentclass[unnumsec,webpdf,contemporary,large,namedate]{oup-authoring-template}`; abstract converted to the journal's structured `\abstract{...\textbf{Motivation:}...}` block; `\keywords{}`, `\corresp{}`, `\address{}`, `\authormark{}`, `\ORCID{}`, `\received/\revised/\accepted` all in place | Preamble |
| Page-limit compliance (Original Paper ≤ 7 pages ≈ 5,000 words excluding figures) | We trimmed body+abstract from ~8,074 → 5,771 words (~30% reduction). The OUP-typeset PDF is 12 pages; tables and the seven figures (plus seven supplementary) account for the remainder. We are happy to defer further compression to copy-editing if the journal requests | §1–§5 throughout |
| CRediT taxonomy in Author Contributions | Recast in CRediT terms (Conceptualization / Methodology / Software / Formal analysis / Investigation / Data curation / Writing — original draft / Writing — review & editing / Visualization / Project administration) | §"Author Contributions" |
| ORCID at submission | Provided: 0000-0001-5342-1499 | Title block |
| Marked-up "track-changes" version | Will be uploaded alongside the revised manuscript at submission. The complete pre-revision LaTeX is preserved as `paper/main.pre_revision.tex` so the change set can be auto-generated | Submission package |

---

## Cross-cutting changes requested by multiple reviewers

### 1. "Foundation-model" framing vs Pearson-only evidence (R1·a·1, R2·2, R3·title, R4·1, R4·minor·7)

**The core complaint was correct on the original submission.** The pre-revision paper named scGPT prominently in the title and abstract but only ran Pearson correlation in the body. That is no longer true.

**Action.** We built a pluggable scorer engine (`fragility/scorers/`) and implemented five edge scorers exercised across the paper:

1. **Pearson correlation** on column-standardised expression
2. **Mutual information** via `mutual_info_regression` (Kraskov–Stögbauer–Grassberger k-NN estimator, k=3)
3. **GRNBoost2** (gradient-boosted regression importance)
4. **GENIE3** (random-forest regression importance)
5. **scGPT attention** extracted from the public *whole-human* checkpoint via a torchtext-free wrapper (FlashAttention-trained `Wqkv` keys remapped to standard `in_proj` keys)

scGPT is now a real first-class scorer; we explicitly state in §2.2 that the model is used as a **frozen pretrained feature extractor** (no fine-tuning is performed) and therefore the journal's "ML training-dataset subsection" requirement does not apply.

**Headline new finding** (Table 3, §3.6): scGPT attention rankings on the same 800 cells are **statistically uncorrelated with every classical scorer** — Spearman ρ ranges from −0.05 (vs Pearson) to +0.20 (vs MI), top-100 Jaccard 0.02–0.06, and the bounded-drift RSS composite sits at or above the rank-permutation null mean of ≈0.72. So **within-scorer fragility profiles do not transfer across scorer families**. This new orthogonality finding is reflected in the abstract (point vi) and §3.6.

**Title and framing — explicit decision.** The title still reads "*from Single-Cell Foundation Models*". This is now defensible because the audit *does* include a real foundation-model scorer and reports the divergence between scGPT and classical scorers as a primary contribution. We will, however, accept the editor's preferred wording if Bioinformatics would prefer a more neutral title (suggested alternative: *"Five Axes of Fragility: A Multi-Scorer Robustness Audit of Gene-Regulatory-Network Inference from Single-Cell Transcriptomics"*).

### 2. MVCC circularity and single-tissue claim (R1·a·2, R4·2)

**Action.** Re-ran Axis 1 on **all six cohorts** (four Tabula Sapiens tissues + Travaglini external lung + Delorey COVID lung) with an **anchor ablation** at {3,000, 8,000} cells (and {3,000, 8,000, 15,000} on Delorey, whose 116,313 cells permit it).

- **Result**: MVCC is **anchor-invariant** at every cohort (Fig 1B, Supp Fig S1) — identical values emerge against 3k, 8k, and 15k anchors, ruling out anchor–threshold circularity.
- **MVCC is scorer-specific**: under Pearson on 600 HVGs, MVCC = 200 cells for kidney/lung/immune-invariant/external lung and 500 cells for primary immune and Delorey COVID. Under scGPT attention, MVCC > 3,000 cells. The original ">3,000" guideline therefore applies only to scGPT — under classical scorers, 200–500 cells suffice on the cohorts tested.
- The full **Jaccard-vs-cell-count curves** for every (cohort, anchor, scorer) triple are now in Supp Fig S1 (R4·2 directly addressed).

### 3. Dataset diversity beyond Tabula Sapiens (R2·1, R4·2)

**Action.** Two additions beyond the original four-tissue Tabula Sapiens panel:

- **Travaglini et al. 2020 SmartSeq2 lung atlas** (\citep{Travaglini2020}) — non-Tabula-Sapiens, non-10x technology, runs on Axes 2 and 3
- **Delorey et al. 2021 COVID-19 lung atlas** (116,313 cells, 27 donors, downloaded from CellxGene) — *disease cohort*, runs on Axes 1, 2, and 3

On Delorey we observe a qualitatively new failure mode in Axis 3: rank-stability is high (Spearman 0.85–0.89) and Jaccard reasonable (0.53–0.63), but null-separation AUC is uniformly weak (0.50–0.64) so **no cell type passes the gate in any rarity class**. We interpret this as disease-driven shared-expression programmes (COVID-induced myeloid / fibroblast remodelling) compressing the observed-vs-null gap, and call out the need for disease-specific recalibration in §3.3.

The geneRNBI dataset collection (R2·1) is a useful pointer — we considered it during revision but deemed an immediate switch out of scope; we cite it as a natural extension for follow-up work.

### 4. Single integration tool (R4·8, R2·2)

**Action.** Cross-method comparison on the same three tissues (immune, lung, kidney) on identical inputs (Table 2, §3.5):

| Tissue | Harmony ρ | scVI ρ | Scanorama ρ | Sign-flip range |
|---|---|---|---|---|
| Kidney | 0.78 | 0.57 | **−0.04** | 15% / 24% / **47%** |
| Lung | 0.63 | 0.36 | **−0.02** | 17% / 30% / **44%** |
| Immune | 0.87 | 0.54 | 0.10 | 6% / 18% / 26% |

Three regimes emerge: Harmony preserves rankings (6–17 % sign-flip), scVI moderately perturbs them (18–30 %), and Scanorama on kidney/lung produces near-random rankings (44–47 % sign-flip; RSS at the rank-permutation null mean). Integration-method choice therefore matters far more than the pre-revision Harmony-only analysis suggested. The Limitations section (§4) flags MNN \citep{Haghverdi2018} and BBKNN \citep{Polanski2020} as natural follow-up integrators.

### 5. Opaque composite metric, threshold choices, panel choices (R1·a·3, R1·a·4, R1·a·5, R3·metric-defs, R4·4, R4·5, R4·6)

**RSS rebuilt with bounded drift, empirical null, and weight sweep.**

- **Bounded drift term** (R1·a·5): `drift_norm = min(1, median |Δrank| / k)` ∈ [0,1]. Worst-case rank change inside a top-k set is k, so dividing by k gives a [0,1] statistic; the explicit `min(1, ·)` clip handles edge cases. RSS components (overlap_loss, jaccard_loss, drift_norm) are now reported separately in addition to the composite (R1·a·4).
- **Empirical null distribution** built from 1,000 rank-permutations per cohort (R4·4): null mean ≈ 0.79, null 5th-percentile ≈ 0.73. Every observed RSS sits **8.8 to 13.1 σ below null** at empirical p = 0.001 (Supp Fig S3). The previous "uninterpretable 0.276–0.440" range becomes "every cohort highly significantly more stable than a random coarse-vs-fine pairing".
- **Full 66-point weight simplex sweep** at step 0.1 (R1·a·3, R4·4): the dual-null recommendation is **weight-invariant** — the within-coarse null fails universally regardless of weight choice, so no simplex point rescues a "fine" recommendation (Supp Fig S2).
- **Why one composite vs three separate metrics** (R1·a·4): we now report both. The composite gives a single ordering for cross-cohort comparison; the components let readers see where the ranking change comes from. §2.3 carries the formula and §3.2 reports both.

**Reliability-gate thresholds threshold-graded** (R4·5, R1·a·5). 5,880-point 5-D grid sweep over (stability, Jaccard, AUC, tail gap, cell count): the abundant-vs-rare gap direction is preserved at **100% of grid points**; rare pass rate = 0 at 66.7%; the grid-mean gap (+0.344) is wider than the primary-threshold gap (+0.182), so primary thresholds are *conservative* (Supp Fig S4). Thresholds were pre-registered in code before the grid sweep ran (R4·5 circularity concern addressed).

**Panel choices unified** (R4·6). A shared panel registry (`fragility.panels.registry`, Supp Table S1) holds every named panel as an exact edge list at runtime: `primary` (DoRothEA A/B ∩ TRRUST, 3,384 edges), `dorothea_ab`, `trrust`, `hematopoiesis_76×108` (Axis 4, 8,208 directed edges before gene-universe restriction), and `shared_36` (Axis 3 cross-dataset sensitivity). Selection criteria are documented in code and reproduced in Supp Table S1.

### 6. AI-assisted review protocol (R2·4, R3·AI-agent, R4·7)

**Action.** §2.10 of the revised paper carries a one-paragraph summary; the **full protocol** lives in Supplementary Methods S2:

- **Base model**: Claude Opus 4.7 (1M-token context) via Claude Code
- **Rubric**: confounded null designs, overfitted hyperparameters, leakage, composite-metric uninterpretability, panel-selection arbitrariness, figure/caption mismatches, threshold sensitivity
- **Triggering criterion for corrective experiments**: any rubric category flagged with concrete reproducible evidence
- **Workflow**: agent specifies experiment → human executes → agent verifies result before claim is updated
- **Audit log**: 27 agent-raised issues classified as **23 material, 4 minor, 0 false positive, 0 unclear** under a pre-registered rubric. Round-by-round logs and the issue census are in Supp Methods S2.

The agent does *not* search the metric space adversarially (R3 worried about this); the rubric is fixed in advance and the agent applies it. So it is not "trying to find metrics where stability is poorest" — it is checking standard threats-to-validity that any reviewer would check, more systematically and reproducibly than a single human pass.

### 7. Code release (R1·b·1, R3·code)

**Action.** The full reproduction package is now public at <https://github.com/Biodyn-AI/grn-robustness-audit> (mirrored on Zenodo at <https://doi.org/10.5281/zenodo.20009065>) under the `implementation/` directory. The package contains:

- `fragility/preprocessing.py` — shared scRNA-seq preprocessing
- `fragility/panels/` — TF–target panel registry
- `fragility/scorers/` — pluggable edge-scoring engine (Pearson, MI, GRNBoost2, GENIE3, scGPT attention)
- `fragility/nulls/` — null-model library (global, within-coarse, gene-shuffle, rank-permutation, degree-preserving rewire)
- `fragility/metrics/` — bounded-drift RSS, stability, top-k, null-AUC, tail-gap
- `fragility/axes/` — five axis pipelines + work-package runners
- `fragility/cli/` — `python -m fragility` entrypoint
- `configs/` — YAML configs for every axis and work-package run
- `tests/` — 35 unit tests, all passing
- `Makefile` — `make audit` reproduces every CSV and figure in the manuscript from raw h5ad input

Adding a new scorer or dataset requires implementing one interface and registering it in a YAML config (R3 explicitly asked for extensibility).

### 8. Joint validation of recommendations (R4·3)

**Action.** A WP-7 prospective audit, in which a single cohort is constructed to satisfy every recommendation in Table 5 and then re-audited across all five axes, is acknowledged in §"Discussion → Limitations" as the natural next experiment. Given that the within-coarse null fails universally and scGPT rankings are statistically orthogonal to classical rankings, the revision's honest position is that **the recommendations are necessary** (each prevents a specific failure mode) **but are not jointly shown to be sufficient**. The prescriptive language in Table 5 has been softened accordingly.

---

## Per-reviewer checklist

### Reviewer 1

| # | Comment | Response | Where |
|---|---|---|---|
| a·1 | Foundation-model framing vs Pearson-only evidence | Real scGPT + 3 classical baselines; new orthogonality finding | §2.2, §3.6, Table 3 |
| a·2 | MVCC on single dataset only | Re-ran on six cohorts with anchor ablation; MVCC anchor-invariant | §3.1, Fig 1, Supp Fig S1 |
| a·3 | Unequal RSS weights | 66-point weight simplex sweep at step 0.1; recommendation invariant | §3.2, Supp Fig S2 |
| a·4 | Overlap and Jaccard both in RSS | Components reported separately as well as the composite | §2.3, §3.2 |
| a·5 | Median drift unbounded | New `drift_norm = min(1, median \|Δrank\| / k)` ∈ [0,1] | §2.3 RSS formula |
| a·6 | Recommendations may not transfer to other GRN methods | Now five scorers; Table 3 shows the orthogonality; Table 5 includes a per-scorer "Scorer choice" row | §3.6, Table 5 |
| b·1 | GitHub repo incomplete (only CSVs and figure scripts) | Full `fragility` package + `make audit` now in `implementation/` | Repo `implementation/` |
| b·2 | Top-k as proportion vs absolute | Top-k scan at k ∈ {100, 500, 1k, 5k} plus top-1%/5%/10% (Supp Fig S5) | §3.1, Supp Fig S5 |
| b·3 | 76×108 panel selection | Encoded in `fragility.panels.registry`; full gene list in Supp Table S1 | §2.7, Supp Table S1 |
| b·4 | Fig 1B legend overlaps data; error bars referenced but not shown | Fig 1A redrawn with shaded SD bands across 8 bootstraps; Fig 1B is now an MVCC heatmap (no overlapping legend); caption rewritten to match what is plotted. Additionally we just repositioned the legends in Fig 2A, Fig 5A, and Fig 5C below the panels to eliminate any other in-plot overlaps | Fig 1, Fig 2, Fig 5 |

### Reviewer 2

| # | Comment | Response | Where |
|---|---|---|---|
| 1 | Single atlas | Travaglini external lung + Delorey COVID lung added; six cohorts total | §2.1 (Table 1), §3.1–§3.3 |
| 2a | Foundation-model framing | Real scGPT included; orthogonality with classical scorers reported as a primary finding | §2.2, §3.6 |
| 2b | Use a benchmarking framework (geneRNBI) | Cited as a natural follow-up extension; immediate switch was out of scope for this revision | Limitations |
| 2c | Foundation models perform poorly on GRN inference | Our evidence is consistent with this — scGPT attention is uncorrelated with classical scorers and exhibits a much higher cell-count requirement (>3,000 cells vs 200–500). Reported as a finding rather than a positive recommendation for foundation-model use | §3.1, §3.6 |
| 3a | Writing clarity | Manuscript heavily rewritten and tightened (~30% body trim); structured abstract; OUP template; consistent terminology | Throughout |
| 3b | Position in literature; recent benchmarks | Introduction now references Pratapa 2020 BEELINE, Chen & Mar 2018, Luecken 2022 atlas integration benchmark, Tran 2020 batch-effect benchmark, Heumos 2023 best-practices review, plus newly-added Theodoris 2023 (Geneformer), Hao 2024 (Seurat v5), Squair 2021 (single-cell DE false discoveries) | §1, Discussion |
| 4 | AI-agent methodological detail | New §2.10 in main text; full protocol + 27-issue audit in Supp Methods S2 | §2.10, Supp Methods S2 |

### Reviewer 3

| # | Comment | Response | Where |
|---|---|---|---|
| Title | Foundation-model framing | Real scGPT now included; we are open to an editor-preferred neutral wording (option proposed in cross-cutting #1) | Title, §1 |
| Abstract narrative | Poorly structured | Abstract rewritten end-to-end with the journal's structured Motivation / Results / Availability and Implementation / Contact / Supplementary Information labels and an explicit "Our central findings are that…" framing | Abstract |
| Why these metrics | Motivation per task | Motivation given in §2.3 metric definitions; AI-agent rubric in §2.10 + Supp S2. The agent does not adversarially search the metric space — the rubric is fixed in advance | §2.3, §2.10 |
| Metric definitions formal | Yes, every metric defined | RSS formula + bounded drift in §2.3; "cleared" defined explicitly ("≥19 permutations, empirical p ≤ 0.05"); per-axis metric usage stated in each axis subsection | §2.3 |
| Code repo incomplete | Full package + `make audit` | See cross-cutting #7 | `implementation/` |
| Normalization (depth-wise variance inflation) | Ablation requested | Three normalization schemes (depth-log1p / size-factor-log1p / Pearson residuals \citep{Lause2021}) tested on kidney Axis-2: all give *coarse* recommendation (Supp Fig S6) | §2.1, §3.2, Supp Fig S6 |
| HVG count | Report 600 explicitly | Stated in §2.1; gene-dimension sweep over {300, 600, 900, 1,200} in §3.2 — recommendation invariant for 4/5 cohorts | §2.1, §3.2 |
| Top-k low for 600 genes | Add per-target top-k | Per-target top-5 and top-10 reported in Supp Fig S5; top-k scan at k ∈ {100, 500, 1k, 5k, 1%, 5%, 10%} | §3.1, Supp Fig S5 |
| "Base RSS recommendation" undefined | Define | §2.5 explicitly maps RSS thresholds to {fine ≤ 0.30, hybrid ≤ 0.45, else coarse} and describes the per-failed-null downgrade | §2.5 |
| Section 2.7 "synthesis" | Rename | Renamed to "Cross-axis analysis" throughout; the §3.6 (scorer divergence) and §3.7 (cross-axis) Results subsections are now also called "Cross-cutting analyses" in the merged Methods §2.9 | §2.9, §3.7 |
| "Anchor" wording | Define / rename | §2.4 calls it the "reference ranking" / anchor explicitly and states why a 3,000-cell anchor is used | §2.4 |
| BH correction — multiple of what | Specify family | §2.4 states the corrected family explicitly: "per-(policy, cell-count) p-values testing whether prior-constrained rankings differ from matched random-gene-set controls" | §2.4 |
| Constrained-shuffle over-stringency | Triple-null check | A degree-preserving bipartite rewire was added as a third, deliberately permissive null. Every cohort clears it at the permutation floor (p = 0.02) while still failing within-coarse at p ≥ 0.22, so the within-coarse failure is genuine over-partitioning, not null over-stringency (Supp Fig S7) | §3.2, Supp Fig S7 |
| Table 2 gene-dimension robustness | Sweep | {300, 600, 900, 1,200}: 4/5 cohorts coarse at every HVG count; the fifth (immune-invariant) flips coarse↔hybrid but never to fine | §3.2 |
| Section 3.2 plots show only coarse | Show all five cohorts | Fig 2 redesigned: Panel A shows observed RSS for every cohort vs the rank-permutation null with error bars; Panel B is the dual-null pass/fail heatmap; Panel C the top-15 most fine-sensitive coarse groups | Fig 2 |
| Show low-overlap example for 3.3 | Illustrative plot | Fig 3A's null-AUC vs Jaccard scatter, coloured by rarity, shows exactly this — high stability can coexist with null-level AUC | Fig 3A |
| Lines 227–230 tangential | Trim | Removed; methods rewritten as a clean fixed-holdout description | §2.7 |
| "Every tissue" → "Every dataset" | Fix | Changed to "every cohort" throughout | §3.2, §3.7 |
| Fig 1A caption mismatch | Rewrite caption | Caption fully rewritten; matches the actual rendered panels (Pearson scorer, Jaccard vs cells curves, MVCC heatmap) | Fig 1 caption |
| Fig 2A, Fig 2B captions | Rewrite | Captions match what is plotted (observed RSS + null + dual-null heatmap + top-15 most fine-sensitive groups) | Fig 2 caption |
| Fig 3A caption mismatch | Rewrite | Caption matches the null-AUC vs Jaccard scatter | Fig 3 caption |
| Fig 5B caption ("filled bars FDR < 0.05") and 5A batch-key mapping | Rewrite | Caption clarified; filled vs unfilled distinction visible (some bars are now correctly grey for non-significant) | Fig 5 caption |
| Line 26 comma after (GRNs) | Fixed | Removed | §1 |
| Line 132 axis's | Fixed | "axis'" with apostrophe-only | §3.7 |

### Reviewer 4

| # | Comment | Response | Where |
|---|---|---|---|
| 1 | Foundation-model framing | Real scGPT + 3 classical scorers; orthogonality finding | §2.2, §3.6 |
| 2 | MVCC circularity + tissue specificity | Anchor ablation + 6 cohorts + scorer-specific MVCC | §3.1, Fig 1, Supp Fig S1 |
| 3 | Recommendations never jointly validated | Acknowledged in Limitations as next experiment; Table 5 prescriptive language softened to "guidance" | Discussion, Table 5 |
| 4 | RSS interpretability | Empirical null (8.8–13.1 σ), 66-point weight simplex sweep, components reported separately | §3.2, Supp Fig S2/S3 |
| 5 | Reliability-gate thresholds | 5,880-point grid sweep; gap direction preserved at 100% of grid points; thresholds pre-registered before grid sweep | §3.3, Supp Fig S4 |
| 6 | Panel inconsistency across axes | Shared panel registry; documented selection criteria; cross-axis conclusions carefully phrased | §2.1, Supp Table S1 |
| 7 | AI-agent protocol underdescribed | New §2.10 main text + Supp Methods S2 with prompt rubric, base model, round logs, 27-issue full census | §2.10, Supp Methods S2 |
| 8 | Only Harmony for integration | scVI and Scanorama added; three-method comparison reveals Scanorama produces near-random rankings on kidney/lung | §2.8, §3.5, Table 2 |
| minor 1 | HVG count (600) not stated | Stated in §2.1; sweep in §3.2 | §2.1, §3.2 |
| minor 2 | Fig 1A error bars not visible | Redrawn with shaded SD bands across 8 bootstraps | Fig 1A |
| minor 3 | "high-cell-only edge support" undefined | Defined in §3.1 | §3.1 |
| minor 4 | DoRothEA-constrained policies undefined before use | Defined in §2.4 (Axis 1 methods) | §2.4 |
| minor 5 | "cleared" null family not formally defined | Defined in §2.3 ("≥ 19 permutations, empirical p ≤ 0.05") | §2.3 |
| minor 6 | Fig 5A batch-key mapping unclear | Caption clarified; per-tissue per-batch-key labels visible on x-axis | Fig 5 caption |
| minor 7 | scGPT-only as "foundation models" | scGPT is the only foundation model implemented in this revision; we acknowledge this and have added Geneformer \citep{Theodoris2023} as a citation alongside scGPT in the introduction so readers see the wider class. Extending the audit to Geneformer / scFoundation is acknowledged as a natural follow-up | §1, Discussion |

---

## Numerical headlines that changed between submissions

| Claim (original submission) | Revised claim |
|---|---|
| "MVCC > 3,000 cells" (kidney only) | **Scorer-specific**: 200 cells (Pearson/tree, 4 of 6 cohorts), 500 cells (Pearson, primary immune and Delorey COVID), > 3,000 cells (scGPT attention, anchor-invariant) |
| "RSS 0.276–0.440 (uninterpretable)" | 0.218–0.402 with empirical null mean ≈ 0.79: every cohort 8.8–13.1 σ below null at p = 0.001 floor |
| "All 5 datasets default to coarse" | 4 of 5 Tabula Sapiens cohorts default to **coarse**, immune-invariant to **hybrid** under primary parameters; **all five** to coarse or hybrid under every weight choice on the 66-point simplex and every HVG count in {300, 600, 900, 1,200}. Delorey adds a 6th cohort with the lowest raw RSS (0.173) but still failing within-coarse |
| "0% rare vs 18% abundant reliability passage" | Preserved at primary thresholds; **gap direction preserved at 100% of 5,880 threshold combinations**; rare pass rate = 0 at 66.7% of grid points; primary-threshold gap (+0.182) is *conservative* relative to the grid mean (+0.344) |
| "6–8 donors needed for immune" | **Cells-per-donor dependent**: 6–8 under strict 300 cells/donor (90%/95% of best); 2–4 under relaxed 200 cells/donor. Both regimes reported; cells-per-donor policy must be pre-registered |
| "Harmony displaces 31–45% of top-500 edges" | 25–35% panel-restricted at top-500/top-1k under bounded-drift RSS; Harmony rankings remain ≥ 37 σ below the rank-permutation null. **scVI moderately perturbs (18–30% sign-flip); Scanorama on kidney/lung produces near-random rankings (44–47% sign-flip, RSS at null mean)** |
| (new) | **scGPT attention is uncorrelated with classical-scorer rankings** on the same cells (ρ ≤ 0.20, top-100 Jaccard ≤ 0.06). Within-scorer fragility profiles do not transfer across scorer families |

---

## Updates after the first revision response (April 24 → May 3, 2026)

The following changes were made between the initial revision and the present response, in addition to everything documented above:

- **Zenodo archival DOI** (`10.5281/zenodo.20009065`) added to the abstract Availability snippet and to the Data and Code Availability section, addressing the editor's explicit request
- **Corresponding-author email** updated to ihor.kendiukhov@student.uni-tuebingen.de
- **OUP `oup-authoring-template`** adopted (`[unnumsec, webpdf, contemporary, large, namedate]`); structured abstract, `\keywords{}`, `\address{}`, `\corresp{}`, `\authormark{}`, `\ORCID{}`, `\received/\revised/\accepted` macros all populated
- **Word count** reduced from ~8,074 → ~5,771 (body + abstract, excluding figures), within editorial discretion of the 5,000-word soft target
- **Author Contributions** rewritten in CRediT taxonomy
- **Bibliography expanded from 21 → 35 references**, all verified via PubMed / Crossref / journal of record. Three errors caught and corrected in the *original* bibliography:
  - Han et al. 2018 (TRRUST v2) — pages were `D199–D202`; correct `D380–D386`
  - Moerman et al. 2019 (GRNBoost2) — third author was `González-Blas, C. B.`; correct full surname `Bravo González-Blas, C.`
  - Cell papers (Dixit 2016, Replogle 2022, Stuart 2019) — page ranges now include the canonical `.eXX` electronic suffix
  
  New references added: Travaglini2020 (lung atlas, was cited but missing from bib), Margolin2006 (ARACNE), Stuart2019 (Seurat v3 / HVG selection), Traag2019 (Leiden), Kraskov2004 (KSG MI estimator), BenjaminiHochberg1995, Pedregosa2011 (scikit-learn), Theodoris2023 (Geneformer), Heumos2023 (best practices), Replogle2022 (genome-scale Perturb-seq), Squair2021 (single-cell DE false discoveries), Polanski2020 (BBKNN), Haghverdi2018 (MNN), Hao2024 (Seurat v5)
- **Table 4 layout fixed** — was a 4-column `tabularx{lXlX}` whose long Threshold-column text overflowed the OUP 2-column layout, leaving rows nearly empty; restructured to a 3-column `{l X X}` (Axis | Calibrated criterion | Recommendation)
- **Figure 2A, 5A, 5C legends** repositioned below the panels via `bbox_to_anchor=(0.5, -0.28..-0.32), loc='upper center', ncol=2/3` — bars are now fully visible
- **§2.2 sentence added** clarifying that scGPT is used as a *frozen pretrained feature extractor* (no training is performed in this work) so the journal's machine-learning training-dataset subsection requirement does not apply

---

## Reproducibility

Every numerical claim, table, and figure in the revised manuscript is reproducible from raw h5ad input via:

```
git clone https://github.com/Biodyn-AI/grn-robustness-audit
cd grn-robustness-audit/implementation
pip install -e ".[all]"
make audit OUT=./outputs
```

The Zenodo archive at `10.5281/zenodo.20009065` is a frozen snapshot of the code accompanying this submission.

We are happy to provide any additional analyses or clarifications the editor and reviewers request.

— Ihor Kendiukhov
