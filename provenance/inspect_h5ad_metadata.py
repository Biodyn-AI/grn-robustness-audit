"""Read H5AD metadata without loading expression; retain input identity evidence.

Run from the biodyn-work root using the bio_mech_interp Python environment.
No expression matrix is modified and no acquisition history is inferred.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


WORKSPACE = Path.cwd()
DESTINATION = Path(__file__).resolve().parent
SOURCES = {
    "kidney_source": "single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
    "lung_source": "single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
    "immune_20k_source": "single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
    "external_lung_source": "single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad",
    "kidney": "single_cell_mechinterp/outputs/invariant_causal_edges/kidney/processed.h5ad",
    "lung": "single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad",
    "immune_invariant": "single_cell_mechinterp/outputs/invariant_causal_edges/immune/processed.h5ad",
    "immune": "single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad",
    "external_lung": "single_cell_mechinterp/outputs/invariant_causal_edges/external_lung/processed.h5ad",
    "delorey_covid_lung": "subproject_merged_A_robustness_audit/implementation/data_external/delorey_covid_lung.h5ad",
}


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_column(group, key):
    import h5py

    node = group[key]
    if isinstance(node, h5py.Group) and "codes" in node:
        categories = [decode(item) for item in node["categories"][:]]
        return [categories[int(code)] if code >= 0 else "<missing>" for code in node["codes"][:]]
    return [decode(item) for item in node[:]]


def inspect(workspace=WORKSPACE, destination=DESTINATION):
    """Read metadata at explicit locations without loading expression arrays."""
    # Keep --help usable without the optional metadata-inspection dependency.
    import h5py

    workspace = Path(workspace).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cohort_rows, category_rows, details = [], [], {}
    observation_ids = {}
    for cohort, relative_path in SOURCES.items():
        path = workspace / relative_path
        cohort_record = {"cohort": cohort, "path": relative_path, "exists": path.exists()}
        if not path.exists():
            cohort_rows.append(cohort_record)
            continue
        with h5py.File(path, "r") as handle:
            expression = handle["X"]
            shape = expression.attrs["shape"] if isinstance(expression, h5py.Group) else expression.shape
            uns = handle.get("uns", {})
            citation = decode(uns["citation"][()]) if "citation" in uns else ""
            versions = re.findall(r"https://datasets\.cellxgene\.cziscience\.com/([^ ]+)\.h5ad", citation)
            cohort_record.update({
                "n_cells": int(shape[0]), "n_genes": int(shape[1]),
                "provider_version": versions[0] if versions else "",
                "title": decode(uns["title"][()]) if "title" in uns else "",
                "has_raw": "raw" in handle,
                "layers": ";".join(handle.get("layers", {}).keys()),
                "size_bytes": path.stat().st_size,
            })
            obs = handle["obs"]
            for column in ["donor_id", "tissue", "tissue_in_publication", "method", "assay", "_scvi_batch", "broad_cell_class", "compartment", "cell_type_main", "disease"]:
                if column not in obs:
                    continue
                values = read_column(obs, column)
                counts = Counter(values)
                cohort_record["n_" + column] = len(counts)
                for value, count in sorted(counts.items()):
                    category_rows.append({"cohort": cohort, "column": column, "value": value, "n_cells": count})
            index_key = decode(obs.attrs.get("_index", "_index"))
            observation_ids[cohort] = set(read_column(obs, index_key))
            details[cohort] = {"citation": citation, "uns_keys": sorted(uns.keys())}
        cohort_rows.append(cohort_record)
    overlap_rows = []
    names = ["immune", "immune_invariant"]
    if all(name in observation_ids for name in names):
        left, right = [observation_ids[name] for name in names]
        overlap_rows.append({"cohort_a": names[0], "cohort_b": names[1], "n_a": len(left), "n_b": len(right), "shared_observation_ids": len(left & right)})
    for filename, rows in [("cohort_metadata.csv", cohort_rows), ("cohort_category_counts.csv", category_rows), ("cohort_overlap.csv", overlap_rows)]:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (destination / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (destination / "h5ad_source_metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"cohorts_inspected": len(details), "missing_source_files": [row["cohort"] for row in cohort_rows if not row["exists"]], "immune_overlap": overlap_rows}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE,
                        help="Root containing the documented source/processed H5AD layout")
    parser.add_argument("--out-dir", type=Path, default=DESTINATION,
                        help="Directory for the four exported metadata snapshots")
    args = parser.parse_args(argv)
    inspect(args.workspace, args.out_dir)


if __name__ == "__main__":
    main()
