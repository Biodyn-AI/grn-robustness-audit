# Adversarial Review Round 4: Subproject 38 (Cell-Count Subsampling Robustness)

## Review scope
- Previous code reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_round3_pipeline.py`
- Previous manuscript reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/paper.md`
- Goal: stress-test whether round-3 prior-vs-random conclusions used a valid random null geometry.

## Findings (ordered by severity)

### [P0] Random-control null still leaked cross-run structure in round-3
- Evidence:
  - Round-3 sampled one random source/target set per draw from a global shared-gene pool and reused it across all runs in that draw (`sample_random_constraint_sets` -> `build_random_ensemble_policy_cache` in the round-3 implementation).
  - This preserved source/target identity across cell counts and seeds, which can inflate random-policy overlap by construction.
- Why this is critical:
  - The null is no longer “arbitrary constraints per run”; it partially inherits cross-run consistency and can bias prior-vs-random inference.
- Fix implemented:
  - Replaced global per-draw random sets with per-run matched random sets: each run now samples random source/target genes from its own gene universe, matched to the run-specific DoRothEA source/target cardinalities.
  - Implemented in `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_round4_pipeline.py`.

### [P1] Round-3 Monte Carlo depth was too small for stable empirical claims
- Evidence:
  - Round-3 used 12 draws, yielding coarse minimum empirical p-value (`1/(12+1)=0.0769`).
- Why this matters:
  - Important effects can remain unresolved purely due low null resolution.
- Fix implemented:
  - Increased random-control ensemble depth to 64 draws (`1/(64+1)=0.0154` minimum p-value).
  - New empirical tests in `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round4/prior_vs_random_empirical_tests.csv`.

### [P1] Round-3 summary still mixed single-draw and ensemble interpretations
- Evidence:
  - Prior-vs-random delta reporting in round-3 primarily referenced deterministic policy baselines.
- Why this matters:
  - This reintroduces single-draw dependence in interpretation.
- Fix implemented:
  - Prior-vs-random deltas are now computed against random-ensemble means in the round-4 summary pipeline.

## Repair status
- All round-4 findings above are fixed in code and rerun outputs.
- Round-4 outputs:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round4/`
- Round-4 figures:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig14_round4_policy_reference_stability.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig15_round4_mvcc.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig16_round4_high_cell_support.png`

## Net impact on conclusions
- Unconstrained instability remains robust.
- Under corrected per-run random nulls, random-policy stability collapses at 200/1000 cells, and prior-vs-random stability separation becomes large and statistically supported at current Monte Carlo depth.
- Biological-support separation also remains strong and now statistically resolved under the stronger null design.
