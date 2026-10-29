# Adversarial Review (Round 2) and Fix Log

Project: `/Users/ihorkendiukhov/biodyn-work/subproject_38_leave_one_donor_out_generalization`

Scope: adversarially stress the initial LODO generalization manuscript and close high-risk methodological weaknesses with new experiments.

## Findings (ranked by severity)

### [F1 - High] Train-donor scaling was confounded with holdout shrinkage
- Problem: the original train-count curves used complementary splits, so as train donors increased, holdout donors decreased. This can inflate or distort apparent “optimal train donor count”.
- Evidence in original draft: immune performance peaked at 6 and then dropped at 10-11 train donors, but those settings differ not only in train size but also in holdout size.
- Risk: donor-count recommendation can be optimistic or non-identifiable.

### [F2 - High] Donor selection bias toward high-count donors
- Problem: immune analysis only used top-12 donors by cell count.
- Risk: conclusions may not transfer to long-tail donors with smaller cohorts.

### [F3 - Medium] Composition confounding not controlled in split protocol
- Problem: original train/holdout evaluation did not enforce donor-wise cell-type composition matching.
- Risk: measured donor generalization can reflect composition transfer rather than stable within-cell-type regulatory structure.

## Fixes implemented

### Fix for F1: Fixed-holdout protocol (new experiment)
- Implementation: `implementation/run_adversarial_followup.py`
- Artifacts:
  - `reports/artifacts/adversarial_immune_top12_fixed_holdout_split_metrics.csv`
  - `reports/artifacts/adversarial_immune_top12_fixed_holdout_summary.csv`
  - `reports/figures/adversarial_immune_top12_fixed_holdout_curves.png`
- Protocol: holdout donor count fixed at 2; train donors varied (2,4,6,8,10).
- Result:
  - Spearman median increases monotonically from `0.568` (train=2) to `0.683` (train=10).
  - Top-100 overlap increases from `0.445` to `0.540`.
- Interpretation: after removing holdout-size confounding, no interior optimum is observed in this range; earlier “peak at 6” was partially protocol-driven.

### Fix for F2: Expanded donor panel (new data regime)
- Implementation: same script.
- Artifacts:
  - `reports/artifacts/adversarial_immune_top20_fixed_holdout_split_metrics.csv`
  - `reports/artifacts/adversarial_immune_top20_fixed_holdout_summary.csv`
  - `reports/figures/adversarial_immune_top20_fixed_holdout_curves.png`
  - `reports/artifacts/adversarial_followup_metadata.json`
- Protocol: expanded to top-20 donors (down to donors with 430 cells in source), 150 cells/donor, fixed holdout=2.
- Result:
  - Train=16 median Spearman `0.627`, overlap@100 `0.460`.
  - Performance is lower than top-12 panel at comparable train fractions.
- Interpretation: long-tail donor inclusion weakens transfer metrics, so top-12-only results were optimistic for broader deployment.

### Fix for F3: Composition-controlled evaluations (new experiment)
- Implementation: same script.
- Artifacts:
  - `reports/artifacts/adversarial_composition_control_summary.csv`
  - `reports/artifacts/adversarial_composition_control_top8_summary.csv`
  - `reports/figures/adversarial_composition_control_spearman.png`
  - `reports/figures/adversarial_composition_control_overlap100.png`
  - `reports/figures/adversarial_composition_control_top8_spearman.png`
  - `reports/figures/adversarial_composition_control_top8_overlap100.png`
- Results:
  - Top-12 all-donor match feasibility is poor (only 2 common broad classes; 24 cells/donor), and performance drops sharply:
    - Spearman: `0.648 -> 0.356` (-45.1%)
    - Overlap@100: `0.495 -> 0.305` (-38.4%)
  - Top-8 matched control (3 classes; 62 cells/donor) gives a fairer estimate:
    - Spearman: `0.707 -> 0.591` (-16.3%)
    - Overlap@100: `0.610 -> 0.545` (-10.7%)
- Interpretation: a substantial part of donor transfer signal is composition-mediated.

## Recommendation update

Original recommendation in paper: immune minimum train donors = 4.

Updated from fixed-holdout top-12 analysis:
- 90%-of-best criterion: **6 train donors**.
- 95%-of-best criterion: **8 train donors**.
- Absolute threshold (Spearman>=0.70 and overlap@100>=0.55): not reached in this stricter protocol.

Artifacts:
- `reports/artifacts/adversarial_recommendations.csv`

## Residual risks
1. Lung was not rerun under fixed-holdout>1 because only 4 donors exist; additional tissues with >=6 donors would strengthen cross-tissue calibration.
2. Composition matching used `broad_cell_class`; finer matched controls (`cell_type`) remain data-limited at high donor counts.
3. No perturbation-backed validation of the newly identified conservative recommendation regime yet.

## Status
- F1: resolved with fixed-holdout reruns.
- F2: resolved with top-20 donor expansion.
- F3: partially resolved with broad-class matching and top-8 matched controls; fine-grained matching remains future work.
