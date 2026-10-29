# Original analysis and preprocessing sources

These copies preserve original author-generated scripts, shared preprocessing
modules, and configurations in their workspace-relative hierarchy. `manifest.csv`
records their original source paths and SHA-256 values. They supplement the
`implementation/fragility` package and make original dataset construction and
per-axis calculations inspectable.

Historical source defaults contain paths to the author's workspace. They are
retained as source provenance, not presented as a portable raw-data workflow.
Source H5AD matrices, external pretrained checkpoints, and third-party dependencies
must be obtained separately before adapting these scripts to a new environment.
The portable deposited-results entry point is
`python3 scripts/reproduce_supporting_results.py` from the repository root.

No generative model created the measured single-cell data. These scripts transform
published measurements and generate derived scores, sampled comparisons and null
controls. Frozen scGPT inference is implemented in the shared and fragility source;
the language-model-assisted critique protocol is described separately in the paper.
