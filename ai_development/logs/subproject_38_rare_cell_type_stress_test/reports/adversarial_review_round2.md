# Adversarial Review (Round 2) and Fix Log

Scope reviewed:
- `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py`
- `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/reports/paper.md`
- generated outputs under `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/results/`

## Findings (ordered by severity)

### [P1] Single-null dependence could overstate or understate rare-cell failure
- Evidence of risk:
  - Original loop used one null family only (cell permutation), which does not probe mapping-level artifacts.
  - This made conclusions vulnerable to null-choice criticism.
- Fix implemented:
  - Added dual-null evaluation in `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py:691` with:
    - cell-permutation null,
    - gene-shuffle null,
    - conservative null summaries (`min` AUC and max-tail baseline).
  - Added aggregate outputs:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/results/null_model_summary.csv`
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/results/global_rare_vs_abundant_conservative_comparison.csv`
- Validation after fix:
  - Primary gap remains strong (cell-permutation null AUC gap 0.0999, p=5.29e-8).
  - Conservative gap is smaller but still significant (AUC gap 0.0175, p=6.78e-4).

### [P1] Cross-dataset comparability was vulnerable to panel-construction confounding
- Evidence of risk:
  - Adaptive panels are dataset-specific by design; global pooled claims can be criticized as panel artifacts.
- Fix implemented:
  - Added shared-panel sensitivity analysis:
    - shared gene intersection and fixed edge panel export at `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py:1213`
    - per-dataset shared-panel reruns at `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py:1222`
  - Added sensitivity outputs:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/results/shared_panel_cell_type_metrics.csv`
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/results/adaptive_vs_shared_panel_sensitivity.csv`
- Validation after fix:
  - Reliability agreement adaptive vs shared remains high in most tissues (0.786-1.0).
  - Null-AUC rank agreement remains moderate-to-strong in 3/4 tissues (0.74, 0.93, 0.76; external lung lower at 0.09).

### [P1] Original data universe lacked out-of-sample stress testing
- Evidence of risk:
  - Prior loop used only kidney/lung/immune processed datasets.
- Fix implemented:
  - Added external lung dataset support in `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py:45`.
  - External cohort now included in default run unless explicitly excluded.
- Validation after fix:
  - External lung contributes 28 cell types.
  - Reliability remains 0/28 in external lung, strengthening the rarity-risk conclusion beyond the original cohort trio.

### [P2] Minimum-size guideline fitting could produce non-interpretable extrapolations
- Evidence of risk:
  - Unbounded log-linear extrapolation previously produced implausible values in some scenarios.
- Fix implemented:
  - Added bounded estimation status and empirical fallback in:
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py:481`
    - `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py:1117`
  - Added `guideline_status` and explicit unresolved states.
- Validation after fix:
  - Guidelines now include provenance labels (`model_based`, `empirical_reliable_min`) and avoid presenting unresolved extrapolations as hard recommendations.

## Residual risks (not fully resolved in this round)
1. Shared-panel intersection is small (36 edges), which may under-represent pathway diversity.
2. Environment emits non-fatal NumPy/numexpr compatibility traceback noise during imports.
3. External validation currently adds one external tissue cohort (external lung), not a broader multi-organ external panel.

## Commands executed in this round
```bash
python /Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/implementation/run_rare_cell_type_stress_test.py
python /Users/ihorkendiukhov/biodyn-work/reports/scripts/validate_scientific_writing.py --file /Users/ihorkendiukhov/biodyn-work/subproject_38_rare_cell_type_stress_test/reports/paper.md
```
