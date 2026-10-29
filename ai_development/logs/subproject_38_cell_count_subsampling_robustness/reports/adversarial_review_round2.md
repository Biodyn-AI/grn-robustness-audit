# Adversarial Review Round 2: Subproject 38 (Cell-Count Subsampling Robustness)

## Review scope
- Previous repaired code reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_pipeline.py`
- Current manuscript reviewed: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/paper.md`
- Goal: test whether round-1 conclusions remain valid under stronger adversarial controls.

## Findings (ordered by severity)

### [P0] Round-1 constrained-policy gains were not separated from search-space-size effects
- Evidence:
  - Round-1 policy set only included `unconstrained`, `tf_source`, and `tf_source_target` (`.../adversarial_repair_pipeline.py:51-54`).
  - No size-matched random-source/target control policies were run, so improved stability could still be caused by candidate-space shrinkage alone.
- Why this is critical:
  - Without a random control at matched source/target cardinality, “biological prior helps” is not identifiable.
- Fix implemented:
  - Added `random_source` and `random_source_target` policies with deterministic shared-universe sampling in `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/adversarial_repair_round2_pipeline.py`.
  - Generated direct deltas in `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round2/prior_vs_random_control_comparison.csv`.

### [P1] Emergence conclusions were still trajectory-fragile in round-1
- Evidence:
  - Round-1 emergence computation is hard-coded to seed-1 triplets (`RunKey(tier, 200, 1)`, `RunKey(tier, 1000, 1)`, `RunKey(tier, 3000, 1)`; `.../adversarial_repair_pipeline.py:622-624`).
- Why this matters:
  - High-cell-only and persistence classes can vary strongly by seed pairing; a single trajectory is not robust evidence.
- Fix implemented:
  - Replaced single-path emergence with full cross-seed trajectory combinations (200-seed x 1000-seed x 3000-seed).
  - New artifacts:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round2/emergence_trajectory.csv`
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round2/emergence_summary.csv`

### [P1] Overlap significance lacked multiplicity control in round-1
- Evidence:
  - Round-1 computed `p_overlap_geq` for many pairwise tests but did not apply FDR correction (`.../adversarial_repair_pipeline.py:439`, `...:522`; no q-value stage).
- Why this matters:
  - Large test volume can exaggerate significance interpretation if unadjusted p-values are used.
- Fix implemented:
  - Added Benjamini-Hochberg q-values globally and per policy-top_k stratum.
  - New columns included in pairwise and summary outputs under `outputs_adversarial_round2`.

## Repair status
- All round-2 findings above are fixed in code and rerun outputs.
- New round-2 outputs are in:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/implementation/outputs_adversarial_round2/`
- New round-2 figures:
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig8_round2_policy_reference_stability.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig9_round2_mvcc.png`
  - `/Users/ihorkendiukhov/biodyn-work/subproject_38_cell_count_subsampling_robustness/reports/figures/fig10_round2_high_cell_support.png`

## Net impact on conclusions
- Unconstrained instability remains severe.
- Constrained policies remain much more stable than unconstrained.
- New control experiments show that random size-matched constraints also improve stability, but they do **not** recover biological support (high-cell-only DoRothEA support remains ~0), while DoRothEA-based constraints retain non-zero support (~0.055-0.066).
- Updated interpretation: stability gains are partly geometric (search-space restriction), while biological priors add a distinct biological-validity gain.
