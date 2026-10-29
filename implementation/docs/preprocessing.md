# Preprocessing provenance

Preprocessing is cohort- and analysis-specific. The shared helper defaults are
not evidence that a historical runner called that helper. Axis 2 and the Pearson
cell-count runner load processed H5AD `.X` directly; their 600-gene selection ranks
expression variance among genes detected in at least 5% of cells. Axis 3 and Axis 4
also consume processed matrices directly and score specified edge panels.

The preprocessing modules and original YAML files that constructed processed
inputs are retained under `../../legacy_sources/single_cell_mechinterp/`; saved
resolved configurations and provenance are in `../../data/supporting_results/`.
Provider matrices and processed input matrices must be distinguished when
interpreting normalization, filtering and gene selection. The normalization
ablation table describes only its tested Axis-2 kidney analysis, not all axes.

The portable stored-result workflow in the repository README does not reapply
normalization or claim to reconstruct an unrecorded historical matrix.
