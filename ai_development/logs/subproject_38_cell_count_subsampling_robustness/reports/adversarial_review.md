# Adversarial Review: Subproject 38 (Cell-Count Subsampling Robustness)

## Review scope
- Code reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/analyze_cell_count_subsampling_robustness.py`
- Manuscript reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/paper.md`
- Goal: identify failure modes that can produce misleading biological conclusions.

## Findings (ordered by severity)

### [P0] Cross-cell comparisons were confounded by gene-universe drift
- Evidence:
  - Edge rankings were loaded per run (`.../analyze_cell_count_subsampling_robustness.py:149`) and compared directly (`...:225-239`) without harmonizing the gene universe between runs.
  - Emergence classes were built from raw sets across counts (`...:465-477`) with no intersection-universe correction.
- Why this is critical:
  - If gene sets differ by cell-count run, an edge can be “unstable” simply because one endpoint gene is absent at another count, not because ranking geometry changed.
- Fix implemented:
  - Rebuilt all pairwise metrics with explicit shared-gene-universe filtering before top-k or rank comparisons.
  - New implementation: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_pipeline.py`.

### [P1] Reference stability relied on a single anchor seed, creating anchor fragility
- Evidence:
  - Hard-coded `REFERENCE_SEED = 1` (`.../analyze_cell_count_subsampling_robustness.py:39`) and only that seed is used for reference curves (`...:221-224`).
- Why this matters:
  - Any idiosyncrasy in one 3000-cell run directly propagates into all MVCC estimates and conclusions.
- Fix implemented:
  - Multi-anchor evaluation now uses all available 3000-cell seeds per tier and reports distribution summaries (`p05`, `mean`, `p95`).
  - Output: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial/reference_topk_summary.csv`.

### [P1] Stability values had no null calibration, so practical meaning was ambiguous
- Evidence:
  - Original pipeline reported raw Jaccard/overlap only (`.../analyze_cell_count_subsampling_robustness.py:237-239`, `290-292`).
- Why this matters:
  - Very small absolute overlap can still be far above random in huge edge universes, and vice versa.
- Fix implemented:
  - Added random-overlap baseline (expected Jaccard under hypergeometric model), enrichment ratio, and overlap p-values.
  - Outputs: `reference_topk_pairs.csv`, `seed_topk_pairs.csv`, `reference_topk_summary.csv`, `seed_topk_summary.csv` in `outputs_adversarial`.

### [P1] Biological conclusions were drawn from unconstrained policy only
- Evidence:
  - Manuscript conclusions about emergent-edge plausibility were based on full-HVG unconstrained results (`.../paper.md:131-159`).
- Why this matters:
  - This can conflate a policy choice with a biological impossibility claim.
- Fix implemented:
  - Ran new constrained-policy experiments from attention matrices (TF-source and TF-source+target) across all 22 runs.
  - Cached generated edge sets: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial/policy_edges/`.
  - Added policy-level support and MVCC comparisons (`emergence_summary.csv`, `mvcc_summary.csv`).

## Repair status
- All high-severity findings above are fixed in code and rerun outputs.
- Updated figures:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig5_adversarial_policy_reference_stability.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig6_adversarial_mvcc.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig7_adversarial_high_cell_support.png`

## Net impact on conclusions
- The original unconstrained-policy instability claim still holds under adversarial controls.
- However, constrained-policy experiments show strong stability and materially improved external-support rates, meaning the instability is strongly protocol-policy dependent rather than purely cell-count dependent.
