# Cohort-stratified biological context

This bounded descriptive analysis re-summarizes the stored **Pearson**, shared
`primary`-panel Axis-3 results at their existing coarse-compartment resolution.
It does not retrain scGPT, infer new edges, create expression measurements, or
add a species. It keeps all six named cohort subsets separate.

## Reproduce

From the workspace root:

```bash
python3 subproject_merged_A_robustness_audit/submission_plos_one/revision/biology/summarize_tissue_context.py
python3 -m unittest discover -s subproject_merged_A_robustness_audit/submission_plos_one/revision/biology/tests -v
```

The script requires Python 3.9+ and only its standard library. After copying this
entire directory anywhere, the portable commands are:

```bash
python3 summarize_tissue_context.py --out-dir outputs
python3 -m unittest discover -s tests -v
```

## Input identity

- `inputs/axis3_primary_metrics.csv` is a byte-for-byte copy of
  `implementation/outputs/axis3/cell_type_metrics.csv` in the merged project:
  39 eligible coarse compartments from kidney, lung, immune,
  immune_invariant, and external_lung.
- `inputs/axis3_delorey_metrics.csv` is a byte-for-byte copy of
  `implementation/outputs/axis3_delorey/cell_type_metrics.csv`:
  nine COVID-19-lung coarse compartments.
- `outputs/input_checksums.csv` records SHA-256 hashes of both input copies.
  The source acquisition and transformation history belongs in the separate
  provenance documentation; this script starts from the stored compartment metrics.

All underlying cohorts are **Homo sapiens**. `immune_invariant` is an alternative
immune subset, not a fourth Tabula Sapiens tissue or an independent species.
The immune compartment contains cells from multiple organs. The source outputs
retain their original cohort names to keep the mapping unambiguous.

## Definitions and limits

A compartment passes only when stability >= 0.75, top-100 Jaccard >= 0.35,
null-separation AUC >= 0.65, tail gap > 0, and cell count >= 80. These are the
existing primary thresholds; the analysis does not optimize them. Rarity is the
original within-cohort count-quartile label, assigned among eligible compartments
(minimum 40 cells); it is never reassigned globally. Thus a rare compartment in
one cohort can contain more cells than an abundant compartment in another.

Outputs give compartment-level denominators and counts, the median rank
stability and null AUC, the AUC range, and rarity-specific denominators/pass
counts. They summarize 48 compartments in six named subsets, with no pooled
hypothesis test. Exact input rows, not rounded summary numbers, determine passage.

These results are distinct from the 88 fine-cell-type adaptive-panel analysis.
Compartment annotations, sample sizes, source laboratories, and disease states
differ. Cohort differences cannot be attributed solely to tissue biology.
The alternative immune subset is not an independent biological replication.
No two tissue-specific inferred networks are compared edge by edge here.
A null-separation AUC measures separation from the permutation null, not
accuracy against known causal interactions. This table supports the need for
context-specific calibration within humans; it does not establish conserved
regulatory edges, species-general numerical thresholds, or universal failure.

## Validation

Three focused tests were written before implementation: inclusive threshold
boundaries with strict positive tail gap; separate cohort denominators retaining
original rarity labels; and rejection of duplicate compartments. All pass.
A fresh-directory execution produces byte-identical output tables and hashes.
