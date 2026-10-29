# Adversarial Review: Subproject 38 (Third Loop)

## Findings (Severity-Ordered)

### [P0] Robustness sweep used RSS-only recommendations, inconsistent with primary dual-null policy (fixed)
- **Issue:** The previous robustness grid (`top_genes` x `global_weight`) assigned recommendations from RSS thresholds only, while primary conclusions used dual-null calibration.
- **Why critical:** This creates policy inconsistency: sweep outputs suggested mixed `fine/hybrid/coarse` even when the project’s decision rule had moved to null-calibrated conservatism.
- **Fix implemented:** Added dual-null calibration at every sweep grid point and persisted both base and calibrated recommendation fields.
- **Code location:** `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/run_resolution_sensitivity.py:1235`.
- **New evidence:** `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/results/robustness_grid.csv`, `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/results/robustness_summary.csv`.

### [P0] Initial robustness null-permutation count made significance passing mathematically impossible (fixed)
- **Issue:** The first implementation used `robust_null_permutations=8`, but empirical p-value with +1 correction has floor `1/(n+1)=1/9=0.111`, so a `<=0.05` gate could never pass.
- **Why critical:** This would force systematic false-fail in robustness null checks.
- **Fix implemented:**
  - raised robustness null defaults to 25 per family;
  - added explicit permutation-count validation to ensure p-value floor can reach alpha (requires >=19 for alpha=0.05).
- **Code location:** `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/run_resolution_sensitivity.py:786`, `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/run_resolution_sensitivity.py:1291`, `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/run_resolution_sensitivity.py:1448`.
- **Validation test:** `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/tests/test_resolution_sensitivity.py:165`.

### [P1] Baseline null calibration p-value resolution was coarse (partially fixed)
- **Issue:** Prior baseline null used 25 permutations (minimum p-step 0.0385).
- **Risk:** Borderline effects could flip under finer resolution.
- **Fix implemented:** Raised baseline defaults to 49 permutations for both null families.
- **Code location:** `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/run_resolution_sensitivity.py:1446`.
- **Observed impact:** Kidney/lung remain global-null pass at p=0.02; constrained-null still fails across datasets.

## Post-Fix Outcome Snapshot
### Default configuration (top_genes=600, global_weight=0.6)
- Final calibrated recommendations remain: all datasets -> `coarse`.
- Source: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/results/resolution_sensitivity_scorecard.csv`.

### Robustness grid after dual-null calibration (45 configs total)
- Recommendation mode by dataset:
  - kidney: `coarse` (consistency 0.67)
  - lung: `coarse` (consistency 0.78)
  - immune: `coarse` (consistency 1.00)
  - immune_invariant: `coarse` (consistency 1.00)
  - external_lung: `coarse` (consistency 1.00)
- Dual-null pass rate: 0.0 for all datasets.
- Global-null pass rates: kidney 1.0, lung 0.89, external_lung 0.11, immune 0.0, immune_invariant 0.0.
- Constrained-null pass rates: 0.0 for all datasets.
- Source: `/Users/ihorkendiukhov/biodyn-work/subproject_38_cluster_resolution_sensitivity/implementation/results/robustness_summary.csv`.

## Interpretation
The third loop removes a methodological inconsistency and corrects an inferential bug in robustness null testing. After fixing both, the central conclusion remains conservative: constrained null separation is the dominant failure mode, and no dataset/configuration demonstrates robust dual-null separation under the current scoring family.

## Residual Risks
1. Dual-null conclusions are still method-family-specific (correlation blend only).
2. Constrained null may be extremely stringent for max-aggregation scoring; orthogonal score formulations should be tested.
3. Robustness nulls use 25 permutations per grid point for tractability; higher counts may refine rates but are unlikely to reverse zero constrained-pass pattern observed here.
