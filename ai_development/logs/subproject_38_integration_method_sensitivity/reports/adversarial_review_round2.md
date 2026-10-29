# Adversarial Review - Round 2 (Subproject 38 Integration Sensitivity)

## Scope reviewed
- Analysis code: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/implementation/run_integration_sensitivity.py`
- Manuscript: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/reports/paper.md`
- Prior outputs in `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/data/`

## Findings (pre-fix)

1. **[P1] Precision-gain claims lacked statistical significance testing**
- Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/reports/paper.md` (Results 3.3, precision deltas reported as point estimates).
- Risk: observed deltas (especially in Lung) may be within rank-swap noise; conclusions about "improvement" can be overstated.
- Fix implemented in this loop: added paired swap-based binomial significance tests (gained vs lost supported edges in top-k) with BH-FDR correction across tests.

2. **[P1] No perturbation-grounded validation despite causal claims context**
- Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/implementation/run_integration_sensitivity.py` (original pipeline only evaluated DoRothEA/TRRUST support).
- Risk: reference-overlap gains can reflect prior conformity rather than causal/functional validity.
- Fix implemented in this loop: added perturbation-backed validation using new external data from `/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation/phase2_perturbation_combined_{Immune,Lung}.csv`.

3. **[P2] Reference suite omitted direct ChIP-backed DoRothEA atlas**
- Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/implementation/run_integration_sensitivity.py` (`load_reference_sets`, original set lacked `dorothea_chipseq_human.tsv`).
- Risk: conclusions can be sensitive to one curation lineage and miss binding-backed agreement behavior.
- Fix implemented in this loop: added `dorothea_chipseq_ab` and `dorothea_chipseq_all` reference sets.

4. **[P2] Harmony conclusion was under-tested across hyperparameters**
- Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/reports/paper.md` (limitations acknowledged fixed parameters), and implementation fixed `n_pcs=50` in practice.
- Risk: headline conclusions may be artifacts of one latent dimensionality.
- Fix implemented in this loop: added `harmony_batch` PCA-component sweep (`n_pcs` in {30, 50, 80} by default), with dedicated output table and figure.

## Post-fix verification checklist
- [x] Re-run full pipeline and regenerate outputs.
- [x] Confirm new tables exist: `reference_swap_significance.csv`, `perturbation_validation.csv`, `harmony_batch_pcs_sweep.csv`.
- [x] Confirm new figures exist: `fig7_harmony_batch_pcs_sweep.png`, `fig8_perturbation_significant_rate_k500.png`.
- [x] Update manuscript with new statistically grounded conclusions and failure modes.

## Post-fix evidence summary
1. **Statistical significance added and executed**
   Output: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/data/reference_swap_significance.csv`
   Key result: after BH-FDR, statistically significant gains at `k=500` are concentrated in TRRUST for Immune/Kidney (`delta_precision` about `+0.026` to `+0.028`; `p_greater_fdr < 0.05`). No FDR-significant losses at `k=500`.

2. **Perturbation validation added and executed (new data source)**
   Inputs: `/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation/phase2_perturbation_combined_Immune.csv`, `/Users/ihorkendiukhov/biodyn-work/subproject_33_batch_donor_leakage_audit/implementation/phase2_perturbation_combined_Lung.csv`
   Output: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/data/perturbation_validation.csv`
   Key result: at `k=500`, Harmony variants show small negative perturbation delta vs baseline (`-0.002` to `-0.004` precision units), with no statistically significant improvement (`p_two_sided_fdr=1.0`).

3. **ChIP-seq-backed reference suite added**
   New references: `dorothea_chipseq_ab`, `dorothea_chipseq_all`
   Output: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/data/reference_precision.csv`
   Key result: ChIP-seq support is low overall and dataset-dependent; Harmony improves it in Immune/Kidney but not in Lung.

4. **Harmony hyperparameter sensitivity added**
   Output: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/data/harmony_batch_pcs_sweep.csv`
   Figure: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/reports/figures/fig7_harmony_batch_pcs_sweep.png`
   Key result: ranking stability (top-500 overlap) increases with more PCs, but DoRothEA-ABCD precision peaks near `n_pcs=50` (then declines), highlighting a robustness-performance tradeoff.
