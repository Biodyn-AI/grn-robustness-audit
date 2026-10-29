# Matched-cell-budget kidney method control

`design.json` was written before computing this control's score vectors. The
question was motivated by the existing analysis: the historical WP1 wrapper
selected an 800-cell parent but supplied 400 cells to mutual information and
200 to the active attention extractor. The control selects the exact historical
attention subset once and gives the identical ordered 200-cell matrix to all
five implemented scoring methods. It changes neither old results nor model weights.

## Run from the full analysis workspace

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Users/ihorkendiukhov/anaconda3/envs/bio_mech_interp/bin/python -u subproject_merged_A_robustness_audit/submission_plos_one/revision/biology/matched_method/run_matched_comparison.py --workspace /Users/ihorkendiukhov/biodyn-work
```

The runner refuses an existing output directory. Use `--out-dir /path/to/new/run`
for a separate replay; it never overwrites the stored analysis. The required full
workspace contains the `fragility` implementation, the processed kidney h5ad,
the original scGPT checkout, and the whole-human checkpoint. This command is a
local compute replay, not a claim that these large external assets are bundled
with the code. Restore them using the acquisition/provenance documentation.

## Selection and meaningful checks

```bash
OMP_NUM_THREADS=2 /Users/ihorkendiukhov/anaconda3/envs/bio_mech_interp/bin/python -m unittest discover -s subproject_merged_A_robustness_audit/submission_plos_one/revision/biology/matched_method/tests -v
```

The two tests were written before implementation. They independently replay the
historical nested RNG sequence and verify that the intervening MI subsample draw
does not affect the attention subset; they also check that method handoff preserves
the same matrix object and row order. The numerical runner marks the matrix read-only,
checks its SHA-256 after every scorer, and verifies every score vector's shape and
finite values. Every scorer receives 200 rows, so the attention extractor's
`n_cells=200` condition performs no further subsampling.

## Portable replay from the stored vectors

With Python, NumPy and SciPy installed, this command requires only this directory:

```bash
python3 replay_from_vectors.py --check --output /path/to/recomputed_method_agreement.csv
python3 -m unittest discover -s tests -v
```

The standalone replay verifies the full TF-target identity and order before
recomputing every metric for all ten method pairs from all five complete vectors.
It uses the original argpartition top-k selection and stable full-rank convention,
including the original boundary-tie behavior. `--check` requires agreement with
the stored table at absolute tolerance 1e-14; it is verified in a fresh directory.
Two additional tests cover a toy ranking with analytically known overlap, Jaccard,
drift and RSS, and every stored method-pair result. All four matched-control tests
pass. This checks metrics from recorded scores; it does not rerun neural inference.

## Output and interpretation

- `outputs/matched_input.npz`: exact selected expression matrix, ordered gene/cell
  identifiers, TF and target indices, source row indices, parent and inner indices.
- `outputs/selected_cells.csv`: human-readable cell selection and source-row mapping.
- `outputs/scores_*.csv`: all TF-target score values, deterministic ranks, and signs
  where supported, for each of the five methods.
- `outputs/method_agreement.csv`: ten pairwise signed Spearman correlations,
  top-100/top-500 Jaccard, and top-500 bounded-drift RSS/components.
- `outputs/provenance.json`: source-data/checkpoint/implementation SHA-256 hashes,
  versions, input identity, hardware/device, completion status and elapsed time.
- `outputs/checkpoint_load.json`: original checkpoint arguments and loader diagnostics.
- `outputs/timings.json`, `run.log`: method timings and original console output.

This is a budget control on one kidney sample, not an estimate of uncertainty,
causal-edge accuracy, tissue transfer, or species transfer. It uses the frozen
historical continuous-input attention extractor; the checkpoint's training arguments
specify binned inputs. Matching cell counts does not repair or test that encoding
difference. MPS is used as an available supported inference device.

The restricted panel contains 75 TFs and 99 targets (7,425 Cartesian pairs); the
legacy registry name `hematopoiesis_76x108` describes the pre-restriction panel.
Self-pairs are retained for comparison with the historical universe. They are not
interpreted as self-regulatory evidence. The names `grnboost2` and `genie3` in
machine-readable keys identify legacy registry entries; their actual implementations
are sklearn gradient-boosting and random-forest proxies, not canonical arboreto
GRNBoost2/GENIE3 software. Labels in the frozen design and timing file make this
explicit. No foundation model is trained or fine-tuned; the statistical regressors are fitted
on the fixed selected input exactly as their original implementations require.

## Observed control results

All five scorers completed in 57.165 seconds including initialization. Their
ordered 200-by-8,143 input matrices had an identical SHA-256. For attention versus
Pearson, mutual information, gradient boosting and random forest, respectively,
Spearman correlation was -0.022832, 0.085294, 0.081051 and 0.089603; top-100
Jaccard was 0.025641, 0, 0.036269 and 0.036269. These are weak agreements
under this extraction/input specification. The two tree proxies agreed much
more strongly (Spearman 0.941057). No significance claim is made from these
point estimates or from treating edges as independent experimental replicates.

All 174 distinct panel genes were in vocabulary, so this control has no
out-of-vocabulary panel padding. The recorded loader diagnostics show missing
weights only in the unused classification decoder and an unexpected flag encoder;
all attention/value-encoding weights used in this extractor were loaded.
