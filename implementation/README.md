# fragility — reproduction package for "Five Axes of Fragility"

This directory is the full reproduction package for the paper. It bundles the
five per-axis pipelines together with the cross-cohort, cross-scorer, and
integration-method machinery added during revision (Axis 1 MVCC, Axis 2 dual
null, Axis 3 reliability gate, Axis 4 donor generalization, Axis 5 integration
sensitivity, plus the Delorey COVID disease cohort and the three-method
Harmony/scVI/Scanorama comparison).

## Install

```bash
python -m pip install -e .
# optional: method-diversity stack (GRNBoost2/GENIE3, Harmony, scVI, Scanorama)
python -m pip install -e ".[all]"
```

## Reproduce a single axis

```bash
make axis1    # cell-count subsampling (WP-2 drops into the same code path)
make axis2    # cluster-resolution sensitivity (WP-3, WP-14 live here)
make axis3    # rare-cell reliability (WP-4)
make axis4    # donor generalization
make axis5    # integration-method sensitivity (WP-6 extends scorers list)
```

## Reproduce the full revision

```bash
make audit OUT=./outputs
```

This regenerates every CSV and figure referenced in the revised manuscript.

## Layout

```
fragility/
  preprocessing.py   # shared scRNA-seq preprocessing (P2)
  panels/            # TF–target panel registry (P3)
  scorers/           # pluggable edge-scoring engine (P4)
  nulls/             # shared null-model library (P5)
  metrics/           # stability, top-k Jaccard, RSS, null-AUC, tail gap...
  axes/              # five axis pipelines + work-package runners
  cli/               # `python -m fragility` entrypoint
configs/             # YAML configs (one per axis/WP)
tests/               # smoke + unit tests
outputs/             # all generated CSVs, JSONs, figures (gitignored; regenerable)
docs/                # preprocessing.md, adversarial_review/, etc.
```

The `outputs/` and `data_external/` directories are not tracked in git;
running `make audit` (or the per-axis targets above) regenerates them from
the configs and the public source datasets. See `../paper/response_to_reviewers.md`
for the mapping from each Axis/WP to the reviewer comment it addresses.
