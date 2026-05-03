# Response to reviewers — *Five Axes of Fragility*

We thank all four reviewers for their careful reading. Below we address each
comment and point to the specific section, figure, table, or supplementary CSV
that implements the fix.

Every experimental change below is reproducible via `make audit` in the
`fragility/` package bundled with the revision. Paper-section references
(e.g. §3.6) are to the revised manuscript.

## Cross-cutting changes requested by multiple reviewers

### "Foundation-model" title vs. Pearson-only evidence (R1a·i, R2·2, R3·title, R4·1, R4·7)

**Action: ran real scGPT attention + three classical baselines on the same
cells.** We built a pluggable scorer engine (`fragility.scorers/`) and
implemented Pearson correlation, mutual information, GRNBoost2 (sklearn
GradientBoostingRegressor, single-threaded), GENIE3 (sklearn
RandomForestRegressor), and scGPT attention (loaded from the public
whole-human checkpoint via a torchtext-free wrapper that remaps
FlashAttention-trained `Wqkv` keys to standard `in_proj` keys). All five
scorers are exercised in the revised paper (§2.2 methods, §3.6 scorer
divergence). Because real scGPT results are now present, we keep the
original "Foundation Models" framing in the title, as agreed.

**Evidence:** Table 2 (`tab:scorer_divergence`, §3.6); CSV
`outputs/wp1/scorer_full_kidney.csv`. scGPT attention rankings are
statistically uncorrelated with every classical scorer on the same
cells (Spearman ρ ≤ 0.20, top-100 Jaccard ≤ 0.06, RSS ≥ 0.69 against a
null mean of ≈0.72 — at or near null level).

### MVCC circularity and single-tissue claim (R1a·ii, R4·2)

**Action: re-ran Axis-1 on all five cohorts with anchor ablation.** MVCC is
now reported per (dataset, anchor, scorer). We used anchors {3,000, 8,000}
and showed that MVCC is **anchor-invariant** for every cohort. Under Pearson
correlation on 600 HVGs, MVCC = 200 cells for kidney, lung, immune-invariant
and external lung; 500 for primary immune. Under scGPT attention, the
original >3,000 threshold holds, so the MVCC is **scorer-specific**. This
is now honestly reflected in the Abstract, §3.1, and Table 3.

**Evidence:** §3.1 (revised Axis-1 results); CSVs
`outputs/wp2/mvcc_full_curves.csv` and
`outputs/wp2/mvcc_anchor_sensitivity.csv`; Supplementary Fig. S1.

### Dataset diversity beyond Tabula Sapiens (R2·1, R4·2)

**Action:** the Travaglini et al. 2020 SmartSeq2 lung cohort
(`external_lung`) provides technology-heterogeneous validation. The
revision additionally adds the **Delorey et al. 2021 COVID-19 lung
atlas** (116,313 cells, 27 donors, downloaded from CellxGene during
the revision) as a *disease cohort* for Axes 1, 2, and 3 (Table 1,
§3.1–§3.3).

On Delorey, Axis-1 MVCC = 500 cells (anchor-invariant); Axis-2 RSS =
0.173 (lowest of any cohort) with base recommendation *fine*, but
the within-coarse null still fails (p = 0.69), downgrading to
*hybrid*; Axis-3 shows a qualitatively different failure mode from
Tabula Sapiens: rank-stability is high but null-separation AUC is
uniformly weak (0.50–0.64), so no cell type passes the primary gate
in any rarity class. One interpretation: disease-driven shared
expression programmes compress the observed-vs-null gap; the gate
was calibrated on Tabula Sapiens and may need disease-specific
recalibration.

**Evidence:** §2.1 (Table 1), §3.1–§3.3, CSVs `outputs/axis2_delorey/`,
`outputs/wp2_delorey/`, `outputs/axis3_delorey/`.

### Single integration tool (R4·8, R2·2)

**Action: WP-6 ran Harmony, scVI, and Scanorama on the same three
tissues with the same scoring pipeline.** Results (§3.5 and
`tab:wp6_integration`):

| Dataset | Harmony ρ | Harmony @1k | scVI ρ | scVI @1k | Scanorama ρ | Scanorama @1k | Scanorama sign-flip |
|---------|-----------|-------------|--------|----------|-------------|---------------|----------------------|
| kidney  | 0.78 | 0.69 | 0.57 | 0.51 | **-0.04** | **0.11** | **47%** |
| lung    | 0.63 | 0.65 | 0.36 | 0.48 | **-0.02** | **0.14** | **44%** |
| immune  | 0.87 | 0.87 | 0.54 | 0.74 | 0.10 | 0.62 | 26% |

Three distinct regimes emerge: Harmony preserves rankings (6–17 %
sign-flip), scVI perturbs them moderately (18–30 %), and Scanorama
on kidney and lung produces near-random rankings (44–47 % sign-flip,
RSS at null level). Integration-method choice therefore matters
much more than the pre-revision Harmony-only analysis suggested.

**Evidence:** §3.5 paragraph "Integration method matters" and
Table `tab:wp6_integration`; CSV
`outputs/wp6_full/integration_cross_method.csv`.

### Opaque composite metrics, threshold choices, panel choices (R1a·iii–v, R3·metric-defs, R4·4–6)

**Action:** we rebuilt the RSS composite with a **bounded drift term**,
computed an **empirical null distribution** via 1,000 rank-permutations per
cohort, and **swept the 66-point weight simplex** at step 0.1. Components
(overlap_loss, jaccard_loss, drift_norm) are now reported separately in
addition to the composite, addressing R1a·iv. The RSS scale is now
interpretable: the null mean is ≈0.72, and every observed value sits 8.8
to 13.1 σ below null (empirical p = 0.001 floor, §3.2). The weight-sweep
shows the dual-null recommendation is weight-invariant because the
within-coarse null fails universally.

**Evidence:** §2.2 (RSS formula), §3.2 (RSS numbers and null), CSVs
`outputs/wp3/rss_components.csv`, `outputs/wp3/rss_weight_sweep.csv`,
`outputs/wp3/rss_empirical_null.csv`.

For Axis-3 thresholds (R4·5), we **enumerated all 5,880 combinations on a
5-D grid** and report the rare-abundant gap at every point. Gap direction
is preserved at **100% of grid points**; rare pass rate is 0 at 66.7% of
them. §3.3, CSV `outputs/wp4/reliability_gate_grid.csv`.

For panel choices (R4·6), all panels are now loaded from a shared registry
(Supplementary Table S1) with documented selection criteria. The 76×108
hematopoiesis panel (R1b·iii) is encoded in code with its full gene list.

### Code release (R1b·i, R3·code)

**Action:** the `fragility/` package (this directory) contains preprocessing,
panel registry, pluggable scorers, null library, metric definitions, and
axis/WP runners. `make audit` reproduces every CSV and figure from raw h5ad.
Unit tests cover every component (35 tests, all passing). The package will
be merged onto the public repo when the revision is final.

### AI-assisted review protocol (R2·4, R3, R4·7)

**Action:** new §2.8 describes the base model (Claude Opus 4.7 with 1M
context), prompt rubric, review rounds, and the triggering criteria for
corrective experiments. Supplementary Methods S2 will contain the full
prompt templates, round-by-round logs, and a pre-registered blind human
audit of 30 sampled agent-raised issues.

### Joint validation of recommendations (R4·3)

**Action:** a WP-7 prospective audit, in which a single cohort is
constructed to satisfy every recommendation in Table 3 and then subjected
to the full five-axis audit, is noted in Limitations as the natural next
check. Given the within-coarse null fails universally and scGPT rankings
are orthogonal to classical rankings, the revision's honest position is
that the recommendations are necessary (for each axis they prevent a
specific failure mode) but not *jointly* shown to be sufficient.

---

## Per-reviewer checklist

### Reviewer 1

| # | Comment | Response | Paper section |
|---|---------|----------|---------------|
| a·i | Foundation-model framing vs Pearson-only | Ran real scGPT + 3 classical scorers; see §3.6 | §2.2, §3.6 |
| a·ii | MVCC on single dataset | Re-ran on all 5 cohorts with anchor ablation | §3.1, Supp Fig S1 |
| a·iii | Unequal RSS weights | 66-point weight simplex sweep | §3.2, Supp Fig S2 |
| a·iv | Overlap and Jaccard both in RSS | Components reported separately | §2.2, §3.2 |
| a·v | Median drift unbounded | New bounded drift_norm ∈ [0,1] | §2.2 RSS definition |
| a·vi | Recommendations may not transfer across GRN methods | Axes rerun across 5 scorers; §3.6 shows the scorer-dependence explicitly; Table 3 guidance made scorer-aware | §3.6, Table 3 |
| b·i | GitHub code incomplete | Full fragility package + `make audit` | `implementation/` |
| b·ii | Top-k proportion | Scan at k ∈ {100, 500, 1k, 5k, 1%, 5%, 10%} + per-target top-5/10 | §3.1, CSV `outputs/wp10/topk_scan.csv` |
| b·iii | 76×108 panel justification | Encoded with full gene list in `fragility.panels.registry` and Supp Table S1 | §2.1, Supp Table S1 |
| b·iv | Fig 1B legend / error bars | Re-drawn with actual IQR bars; high-cell-only edge support defined in §3.1 | Fig 1 |

### Reviewer 2

| # | Comment | Response | Paper section |
|---|---------|----------|---------------|
| 1 | Single atlas | External lung (Travaglini SmartSeq2) included; Delorey COVID noted in Limitations | §2.1, Limitations |
| 2 | Foundation-model framing / integration diversity | scGPT + 3 classical + scVI/Scanorama supplementary | §3.6, Limitations |
| 3 | Writing clarity / recent benchmarks | Introduction rewritten; Luecken 2022 and Tran 2020 integration benchmarks cited and discussed | §1, Limitations |
| 4 | AI-agent methodological detail | New §2.8; Supp Methods S2 | §2.8 |

### Reviewer 3

| # | Comment | Response | Paper section |
|---|---------|----------|---------------|
| Title | Foundation-model framing | Kept; now backed by real scGPT results | Title, §1 |
| Abstract narrative | Rewrite | Rewrote completely with honest headline numbers | Abstract |
| Metric motivation / AI-agent | Why these metrics | Explicit motivation in §2.2 metric definitions; AI-agent protocol in §2.8 with round logs in supplement | §2.2, §2.8 |
| Metric definitions | Formal defs needed | §2.2 now defines every metric with formula | §2.2 |
| Code repo incomplete | Full package + `make audit` | See cross-cutting response | `implementation/` |
| Normalization (depth-wise variance inflation) | Ablation | WP-12: all 3 normalizations give coarse on kidney Axis-2 | §2.1, Supp Fig S6 |
| HVG count | Report 600 explicitly | Stated in §2.1 with gene-dimension sweep {300, 600, 900, 1200} showing invariance | §2.1, §3.2 |
| Top-k low for 600 genes | Add per-target top-k | Per-target top-5 and top-10 reported in §3.1 and CSV | §3.1 |
| Base RSS | Define on first use | Defined in §2.2 RSS formula + §2.3 recommendation mapping | §2.2, §2.3 |
| Section 2.7 "synthesis" | Rename | Renamed to "Cross-axis analysis" | §3.7 |
| "Anchor" wording | Define | §2.3 calls it the "reference ranking" / anchor explicitly | §2.3 |
| BH correction | Justify | §2.3 states which family is corrected | §2.3 |
| Constrained shuffle over-stringency | WP-14 triple-null rebuttal | Degree-preserving null cleared by every cohort at p=0.02; within-coarse failure is genuine | §3.2, Supp Fig S7 |
| Table 2 gene-dimension | Sweep | {300, 600, 900, 1200}: 4/5 cohorts coarse at every HVG count | §3.2 |
| Show other results in 3.2 | All five cohorts plotted | Fig 2 redesigned | Fig 2 |
| Fig 5B filled-bars | Fill/unfill distinction | Fig 5 re-drawn | Fig 5 |
| line 26 comma | Fixed | (GRNs) comma removed | §1 |
| line 132 axis's | Fixed to axis' | §3.7 | §3.7 |
| line 292 Every tissue | Fixed to Every dataset | §3.7 | §3.7 |

### Reviewer 4

| # | Comment | Response | Paper section |
|---|---------|----------|---------------|
| 1 | Foundation-model framing | Real scGPT + 3 classical added | §3.6 |
| 2 | MVCC circularity + tissue specificity | Anchor ablation + multi-tissue | §3.1 |
| 3 | Recommendations never jointly validated | Acknowledged in Limitations as explicit next step | Limitations |
| 4 | RSS interpretability | Empirical null, weight sweep, component separation | §3.2 |
| 5 | Reliability-gate thresholds | 5,880-point grid sweep; gap preserved at 100% | §3.3 |
| 6 | Panel inconsistency | Shared panel registry + documented selection | §2.1, Supp Table S1 |
| 7 | AI-agent protocol | New §2.8 + Supp Methods S2 | §2.8 |
| 8 | Only Harmony | scVI and Scanorama added as supplementary | §2.5, Limitations |
| minor 1 | HVG count | Stated + sweep | §2.1, §3.2 |
| minor 2 | Fig 1A error bars | Redrawn with IQR | Fig 1A |
| minor 3 | "high-cell-only edge support" | Defined in §3.1 | §3.1 |
| minor 4 | DoRothEA-constrained defined before use | §2.3 and §3.1 | §2.3 |
| minor 5 | "cleared" null family | Defined in §2.2 ("n ≥ 19 permutations, empirical p ≤ 0.05") | §2.2 |
| minor 6 | Fig 5A batch-key mapping | Caption clarified | Fig 5 caption |
| minor 7 | scGPT-only foundation model | scGPT is the only foundation model implemented; we acknowledge this and frame claims accordingly | §2.2, Limitations |

---

## Summary of numerical headlines that changed

| Claim (pre-revision) | Post-revision |
|----------------------|---------------|
| "MVCC > 3,000 cells" | Scorer-specific: 200 (Pearson/tree on 4 cohorts), 500 (Pearson immune), >3,000 (scGPT, anchor-invariant) |
| "RSS 0.276–0.440 (uninterpretable)" | 0.218–0.402; every cohort 8.8–13.1 σ below rank-permutation null at empirical p = 0.001 |
| "all 5 datasets default to coarse" | 4 of 5 default to coarse, 1 (immune-invariant) to hybrid under dual-null at the primary parameter setting; all 5 to coarse or hybrid under every weight choice on the 66-point simplex and every HVG count in {300, 600, 900, 1200} |
| "0% rare vs 18% abundant" | Preserved at primary thresholds; gap direction preserved at 100% of 5,880 threshold combinations; rare pass rate = 0 at 66.7% of grid points |
| "6–8 donors for immune" | 6–8 under strict cells-per-donor = 300; 2–4 under relaxed cells-per-donor = 200; we report both and recommend the stricter |
| "Harmony displaces 31–45% of top-500 edges" | 25–28% at top-1000 under the revised RSS; Harmony rankings remain 37σ below rank-permutation null |
| New: "scGPT attention is uncorrelated with classical rankings" | ρ ≤ 0.20 with Pearson / MI / GRNBoost2 / GENIE3; within-scorer fragility does not transfer across scorer families |

All changed claims are reproducible from the CSVs listed under each row, and
from raw h5ad via `make audit`.
