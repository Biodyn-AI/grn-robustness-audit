# fragility analysis source

This directory contains the shared scoring, null, metric and axis implementations.
Original per-axis scripts and preprocessing modules are also retained under
`../legacy_sources/`. Deposited numerical outputs and their source/hash manifest
are under `../data/supporting_results/`.

## Verified stored-results reproduction

Run from the repository root:

```bash
python3 scripts/reproduce_supporting_results.py --verify-only
python3 scripts/reproduce_supporting_results.py --out reproduced
python3 -m pip install -r requirements.txt
python3 scripts/reproduce_supporting_results.py --out reproduced --figures
python3 biological_context/summarize_tissue_context.py
```

This verifies and restores saved tables, recomputes Pearson MVCC estimates from
per-draw values, and redraws seven main plus seven supplementary figures. It does
not recreate raw measurements or re-run model extraction/integration/null draws.

## Analysis environment

```bash
python3 -m pip install -e .
python3 -m pip install -e '.[all,test]'
python3 -m pytest tests -q
```

The legacy Makefile and YAML defaults record the original working environment.
They are not a complete turnkey raw-input workflow: some targets refer to absent
configurations/CLI modules, and dataset, network and checkpoint defaults include
author-specific paths. In particular, `make audit` must not be interpreted as a
verified way to regenerate every reported result from a fresh clone. For an
available axis runner, supply an explicit YAML configuration with local input
paths and inspect the saved provenance in the supporting results.

The shared `preprocessing.py` helper is not called by every historical axis runner.
Inspect each runner's matrix loading and the deposited resolved configuration;
do not infer preprocessing from helper defaults alone. Processed `.X` inputs,
variance-based 600-gene selection, reference-restricted scoring, and normalization
ablations are distinct operations.
