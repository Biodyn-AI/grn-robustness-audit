# Source and transformation provenance

The metadata tables were inspected directly from the retained H5ADs. They distinguish
source/processed dimensions, donor/tissue categories, and alternative immune subsets.
All 15,000 invariant-immune observation identifiers occur in the 20,000-cell primary
immune input; those subsets are not independent replication.

`h5ad_source_metadata.json` retains original provider citation strings and versioned
CELLxGENE file URLs. `script_output_map.csv` maps acquisition, preprocessing, each
axis and figure generation to the source modules/configurations. Source paths in
that map are relative to the original author workspace; archived copies of author
code/configs are under `legacy_sources/` and `implementation/`.

```bash
# List exact source URLs without downloading:
python3 scripts/fetch_source_data.py
# Retrieve one provider distribution (large; source measurements, not processed inputs):
python3 scripts/fetch_source_data.py --cohort kidney --out-dir source_h5ad
```

The historical first-download commands and some intermediate sampling steps were
not retained. This new utility retrieves the version URLs recovered from file
metadata; it does not claim to recover those commands. Processed inputs additionally
reflect recorded filtering, first-N caps, depth/log transformations, vocabulary and
reference-gene retention. The provider `.X` must not be assumed to be raw counts.
The deposited-results workflow needs none of these downloads.

For a local layout matching the source paths in `cohort_metadata.csv`, the
metadata export can be repeated without loading expression matrices:

```bash
python3 provenance/inspect_h5ad_metadata.py --workspace /path/to/biodyn-work --out-dir metadata_export
```

This needs h5py and the corresponding local H5ADs. `--help` only lists options;
it does not inspect files or write output. `--workspace` defaults to the current
directory and `--out-dir` defaults to this provenance directory. Pass explicit
locations to export elsewhere. Metadata counting and cohort definitions are
unchanged; missing files are reported, not replaced by another distribution.

```bash
python3 -m unittest discover -s provenance/tests -v
# Optional integration test against a complete original local layout:
FRAGILITY_METADATA_TEST_WORKSPACE=/path/to/biodyn-work python3 -m unittest discover -s provenance/tests -v
```

The help test runs without source H5ADs. The optional test exports to a temporary
directory and compares all four files byte-for-byte against the retained snapshots.
