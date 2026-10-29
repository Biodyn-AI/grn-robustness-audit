# WP-9 blinded second-rater worksheet

Pre-registered rubric (frozen before rating begins):

Each issue is classified as exactly **one** of:

- **material bug** — if left unaddressed, would change a numerical
  headline by ≥10 % or invert the direction of a claim.
- **minor issue** — real and worth fixing, but would not change
  headline sign or magnitude.
- **false positive** — on inspection of the cited code or data, the
  issue does not exist (concern was misread, or the code already
  handled it).
- **unclear** — issue description is too vague to rate against
  evidence.

Rater instructions:

1. Do **not** look at the adversarial review markdown files directly
   or any subsequent corrective experiments.
2. For each issue below, examine the cited code / CSV / figure **in
   its pre-fix state** (git history or the issue excerpt only).
3. Record your classification in the rightmost column; add a one-line
   rationale if the call was borderline.
4. When done, save the table and compare counts against the
   author-pass classification in
   `implementation/outputs/wp9/classified.json`.

The issues below are randomised (seed = 20260424, regenerated with
this worksheet) and stripped of axis/round labels to reduce bias.

---

| # | Priority | Title | Excerpt | Classification |
|---|----------|-------|---------|----------------|
| 1 | P1 | Round-3 summary still mixed single-draw and ensemble interpretations | - Evidence:   - Prior-vs-random delta reporting in round-3 primarily referenced deterministic policy baselines. - Why this matters:   - This reintroduces single-draw dependence in interpretation. - Fi | __________ |
| 2 | P1 | Baseline null calibration p-value resolution was coarse | - **Issue:** Prior baseline null used 25 permutations (minimum p-step 0.0385). - **Risk:** Borderline effects could flip under finer resolution. - **Fix implemented:** Raised baseline defaults to 49 p | __________ |
| 3 | P1 | Precision-gain claims lacked statistical significance testing | - Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/reports/paper.md` (Results 3.3, precision deltas reported as point estimates). - Risk: observed deltas (espe | __________ |
| 4 | P0 | Random-control null still leaked cross-run structure in round-3 | - Evidence:   - Round-3 sampled one random source/target set per draw from a global shared-gene pool and reused it across all runs in that draw (`sample_random_constraint_sets` -> `build_random_ensemb | __________ |
| 5 | P1 | Single-null dependence could overstate or understate rare-cell failure | - Evidence of risk:   - Original loop used one null family only (cell permutation), which does not probe mapping-level artifacts.   - This made conclusions vulnerable to null-choice criticism. - Fix i | __________ |
| 6 | F1 - High | Train-donor scaling was confounded with holdout shrinkage | - Problem: the original train-count curves used complementary splits, so as train donors increased, holdout donors decreased. This can inflate or distort apparent “optimal train donor count”. - Eviden | __________ |
| 7 | P1 | Biological conclusions were drawn from unconstrained policy only | - Evidence:   - Manuscript conclusions about emergent-edge plausibility were based on full-HVG unconstrained results (`.../paper.md:131-159`). - Why this matters:   - This can conflate a policy choice | __________ |
| 8 | P1 | Overlap significance lacked multiplicity control in round-1 | - Evidence:   - Round-1 computed `p_overlap_geq` for many pairwise tests but did not apply FDR correction (`.../adversarial_repair_pipeline.py:439`, `...:522`; no q-value stage). - Why this matters:   | __________ |
| 9 | P2 | Harmony conclusion was under-tested across hyperparameters | - Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/reports/paper.md` (limitations acknowledged fixed parameters), and implementation fixed `n_pcs=50` in practi | __________ |
| 10 | P0 | Round-1 constrained-policy gains were not separated from search-space-size effects | - Evidence:   - Round-1 policy set only included `unconstrained`, `tf_source`, and `tf_source_target` (`.../adversarial_repair_pipeline.py:51-54`).   - No size-matched random-source/target control pol | __________ |
| 11 | P0 | Initial robustness null-permutation count made significance passing mathematically impossible | - **Issue:** The first implementation used `robust_null_permutations=8`, but empirical p-value with +1 correction has floor `1/(n+1)=1/9=0.111`, so a `<=0.05` gate could never pass. - **Why critical:* | __________ |
| 12 | P1 | Round-3 Monte Carlo depth was too small for stable empirical claims | - Evidence:   - Round-3 used 12 draws, yielding coarse minimum empirical p-value (`1/(12+1)=0.0769`). - Why this matters:   - Important effects can remain unresolved purely due low null resolution. -  | __________ |
| 13 | P0 | Cross-cell comparisons were confounded by gene-universe drift | - Evidence:   - Edge rankings were loaded per run (`.../analyze_cell_count_subsampling_robustness.py:149`) and compared directly (`...:225-239`) without harmonizing the gene universe between runs.   - | __________ |
| 14 | P0 | Robustness sweep used RSS-only recommendations, inconsistent with primary dual-null policy | - **Issue:** The previous robustness grid (`top_genes` x `global_weight`) assigned recommendations from RSS thresholds only, while primary conclusions used dual-null calibration. - **Why critical:** T | __________ |
| 15 | P0 | Round-2 prior-vs-random conclusion depended on a single random draw and was therefore outlier-fragile | - Evidence:   - Round-2 random controls were generated from one deterministic sample (`RANDOM_POLICY_SEED = 20260218`) and treated as the random baseline.   - Single-draw random baseline reported low  | __________ |
| 16 | P1 | Reference stability relied on a single anchor seed, creating anchor fragility | - Evidence:   - Hard-coded `REFERENCE_SEED = 1` (`.../analyze_cell_count_subsampling_robustness.py:39`) and only that seed is used for reference curves (`...:221-224`). - Why this matters:   - Any idi | __________ |
| 17 | P1 | No perturbation-grounded validation despite causal claims context | - Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/implementation/run_integration_sensitivity.py` (original pipeline only evaluated DoRothEA/TRRUST support). - | __________ |
| 18 | P1 | Round-2 lacked uncertainty-calibrated statistical tests for prior-vs-random deltas | - Evidence:   - Round-2 reported deterministic deltas (`prior_vs_random_control_comparison.csv`) but no empirical null p-values. - Why this matters:   - Without a null distribution, it is unclear whet | __________ |
| 19 | F3 - Medium | Composition confounding not controlled in split protocol | - Problem: original train/holdout evaluation did not enforce donor-wise cell-type composition matching. - Risk: measured donor generalization can reflect composition transfer rather than stable within | __________ |
| 20 | F2 - High | Donor selection bias toward high-count donors | - Problem: immune analysis only used top-12 donors by cell count. - Risk: conclusions may not transfer to long-tail donors with smaller cohorts.  ### [F3 - Medium] Composition confounding not controll | __________ |
| 21 | P1 | Emergence conclusions were still trajectory-fragile in round-1 | - Evidence:   - Round-1 emergence computation is hard-coded to seed-1 triplets (`RunKey(tier, 200, 1)`, `RunKey(tier, 1000, 1)`, `RunKey(tier, 3000, 1)`; `.../adversarial_repair_pipeline.py:622-624`). | __________ |
| 22 | P1 | Original data universe lacked out-of-sample stress testing | - Evidence of risk:   - Prior loop used only kidney/lung/immune processed datasets. - Fix implemented:   - Added external lung dataset support in `/Users/ihorkendiukhov/biodyn-work/subproject_38_rare_ | __________ |
| 23 | P2 | Reference suite omitted direct ChIP-backed DoRothEA atlas | - Location: `/Users/ihorkendiukhov/biodyn-work/subproject_38_integration_method_sensitivity/implementation/run_integration_sensitivity.py` (`load_reference_sets`, original set lacked `dorothea_chipseq | __________ |
| 24 | P2 | Minimum-size guideline fitting could produce non-interpretable extrapolations | - Evidence of risk:   - Unbounded log-linear extrapolation previously produced implausible values in some scenarios. - Fix implemented:   - Added bounded estimation status and empirical fallback in:   | __________ |
| 25 | P1 | Stability values had no null calibration, so practical meaning was ambiguous | - Evidence:   - Original pipeline reported raw Jaccard/overlap only (`.../analyze_cell_count_subsampling_robustness.py:237-239`, `290-292`). - Why this matters:   - Very small absolute overlap can sti | __________ |
| 26 | P1 | Cross-dataset comparability was vulnerable to panel-construction confounding | - Evidence of risk:   - Adaptive panels are dataset-specific by design; global pooled claims can be criticized as panel artifacts. - Fix implemented:   - Added shared-panel sensitivity analysis:     - | __________ |
| 27 | P1 | Round-2 biological-support separation lacked random-ensemble confidence bounds | - Evidence:   - Round-2 high-cell support comparison used one random draw. - Why this matters:   - Even when random support appears near zero, uncertainty should be quantified across random draws. - F | __________ |
