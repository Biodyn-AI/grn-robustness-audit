# Supporting numerical results

`manifest.csv` maps every deposited gzip file to its original workspace-relative
source, exact restored path, byte counts, and SHA-256 hashes of both compressed
and original content. The files are existing computed results; compression does
not change their values. This directory contains no new biological measurements.

The 220 files contain all CSV/JSON/YAML files retained in the unified implementation
output directory and the selected original per-axis result directories. This is a
transparent inventory of retained outputs, not a claim that every historical random
number, intermediate matrix, or raw sequencing read was saved. Smoke-test tables
remain identifiable by their original directory names and are not independent cohorts.

| Restored directory | Content |
|---|---|
| `implementation_outputs/` | Axis/WP scores, per-draw cell-count curves, per-cell-type metrics, donor split metrics, null-calibration tables, configs and recorded provenance |
| `original_axis1/` | Original scGPT-derived sample-size/prior comparison tables and pair/edge observations |
| `original_axis2/` | Original cluster-resolution edge/rank and null-calibration tables |
| `original_axis3/` | Fine cell-type and shared-panel metrics, edges and dataset metadata |
| `original_axis4/` | Donor-level and split-level comparisons, composition controls and panel metadata |
| `original_axis5/` | Per-edge integration rankings, bootstrap overlap, reference/perturbation comparisons and metadata |

From the repository root:

```bash
python3 scripts/reproduce_supporting_results.py --verify-only
python3 scripts/reproduce_supporting_results.py --out reproduced
python3 -m pip install -r requirements.txt
python3 scripts/reproduce_supporting_results.py --out reproduced --figures
python3 biological_context/summarize_tissue_context.py
python3 biological_context/matched_method/replay_from_vectors.py --check --output matched_recomputed.csv
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s biological_context/tests -v
```

The first command checks both compressed and uncompressed hashes. Restoration
recreates exact original tables and independently recomputes the Pearson MVCC
summary from per-draw curves. The figure command redraws seven main and seven
supplementary plots from restored tables. It does not re-run scGPT attention,
integration algorithms, cell subsampling, or null permutations. The tissue-context
script recomputes the descriptive, cohort-specific reliability counts.

Figures may differ in PDF metadata or font rendering across matplotlib versions;
checksums guarantee the deposited numerical inputs, not byte-identical new PDFs.

The complete release inventory additionally checks source, configuration and new
matched-control input/vector bytes when `supporting_archive_contents.csv` is present.
Matched-vector replay needs NumPy/SciPy, not the original source H5AD or checkpoint.
