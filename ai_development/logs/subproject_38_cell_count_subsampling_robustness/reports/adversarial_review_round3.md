# Adversarial Review Round 3: Subproject 38 (Cell-Count Subsampling Robustness)

## Review scope
- Previous code reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_round2_pipeline.py`
- Previous manuscript reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/paper.md`
- Goal: test whether round-2 prior-vs-random conclusions are robust to random-control uncertainty.

## Findings (ordered by severity)

### [P0] Round-2 prior-vs-random conclusion depended on a single random draw and was therefore outlier-fragile
- Evidence:
  - Round-2 random controls were generated from one deterministic sample (`RANDOM_POLICY_SEED = 20260218`) and treated as the random baseline.
  - Single-draw random baseline reported low top-1000 reference Jaccard (e.g., `random_source` 200-cell values around `0.378-0.385`).
  - Round-3 multi-draw ensemble (12 draws) shows much higher random baseline mean: `random_source` 200-cell Jaccard ~`0.668-0.677`; `random_source_target` 200-cell ~`0.711-0.714`.
- Why this is critical:
  - A single null draw cannot identify expected random-control behavior; it can make prior-vs-random effect sizes look artificially large.
- Fix implemented:
  - Added multi-draw random-control ensemble generation and persistence in `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_round3_pipeline.py`.
  - New outputs:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round3/random_ensemble_reference_top1000.csv`
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round3/random_ensemble_seed_top1000.csv`

### [P1] Round-2 lacked uncertainty-calibrated statistical tests for prior-vs-random deltas
- Evidence:
  - Round-2 reported deterministic deltas (`prior_vs_random_control_comparison.csv`) but no empirical null p-values.
- Why this matters:
  - Without a null distribution, it is unclear whether observed deltas exceed random-control variability.
- Fix implemented:
  - Added empirical p-values and BH q-values for prior-vs-random tests in round-3.
  - New output:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round3/prior_vs_random_empirical_tests.csv`

### [P1] Round-2 biological-support separation lacked random-ensemble confidence bounds
- Evidence:
  - Round-2 high-cell support comparison used one random draw.
- Why this matters:
  - Even when random support appears near zero, uncertainty should be quantified across random draws.
- Fix implemented:
  - Added random-ensemble high-cell support distribution output:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round3/random_ensemble_emergence_high_cell_only.csv`

## Repair status
- All round-3 findings above are fixed in code and rerun outputs.
- Round-3 outputs are in:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round3/`
- Round-3 figures:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig11_round3_policy_reference_stability.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig12_round3_mvcc.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig13_round3_high_cell_support.png`

## Net impact on conclusions
- Unconstrained instability remains robust.
- The round-2 claim that prior policies are much more stable than random controls weakens after ensemble nulling: random controls are often fairly stable too.
- The strongest surviving prior-specific signal is biological validity (DoRothEA support in high-cell-only edges), with large effect-size separation vs random ensembles, though with limited Monte Carlo resolution at 12 draws.
