#!/usr/bin/env python3
"""Summarize stored Axis-3 compartments without pooling cohorts or re-fitting models.

The original within-cohort rarity labels and primary reliability gate are retained.
Inputs contain Pearson / shared-primary-panel results, not scGPT-attention results
or the separate adaptive-panel fine-cell-type analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from statistics import median

COHORTS = {
    "kidney": ("Kidney", "Tabula Sapiens", "kidney"),
    "lung": ("Lung", "Tabula Sapiens", "lung"),
    "immune": ("Immune", "Tabula Sapiens", "multi-organ immune compartment"),
    "immune_invariant": ("Immune alternative subset", "Tabula Sapiens", "multi-organ immune compartment"),
    "external_lung": ("External lung", "Travaglini et al.", "lung"),
    "delorey_covid_lung": ("COVID-19 lung", "Delorey et al.", "lung"),
}
NUMERIC_FIELDS = ("cell_count", "stability", "topk_jaccard", "null_auc", "tail_gap")


def passes_gate(compartment):
    """Use the original primary gate; tail-gap positivity is strictly required."""
    return (compartment["stability"] >= 0.75
            and compartment["topk_jaccard"] >= 0.35
            and compartment["null_auc"] >= 0.65
            and compartment["tail_gap"] > 0
            and compartment["cell_count"] >= 80)


def read_inputs(input_paths):
    """Reject incomplete/nonfinite values rather than silently changing denominators."""
    compartments = []
    for input_path in input_paths:
        with input_path.open(newline="", encoding="utf-8") as stream:
            for raw_row in csv.DictReader(stream):
                compartment = dict(raw_row)
                for field in NUMERIC_FIELDS:
                    compartment[field] = float(raw_row[field])
                    if not math.isfinite(compartment[field]):
                        raise ValueError(f"Nonfinite {field}: {input_path}")
                compartment["cell_count"] = int(compartment["cell_count"])
                compartments.append(compartment)
    return compartments


def summarize(compartments):
    """Count each compartment once, preserving cohort and rarity denominators."""
    identities = set()
    for compartment in compartments:
        identity = (compartment["dataset"], compartment["cell_type"])
        if identity in identities:
            raise ValueError(f"Duplicate compartment: {identity}")
        identities.add(identity)
        if compartment["dataset"] not in COHORTS:
            raise ValueError(f"Unknown cohort: {compartment['dataset']}")
        if compartment["rarity"] not in ("rare", "intermediate", "abundant"):
            raise ValueError(f"Unknown rarity: {compartment['rarity']}")
    summaries = []
    for cohort, (label, study, tissue) in COHORTS.items():
        members = [row for row in compartments if row["dataset"] == cohort]
        if not members:
            continue
        summary = dict(dataset=cohort, cohort_label=label, study=study,
                       species="Homo sapiens", tissue_context=tissue,
                       scorer="pearson", panel="primary", n_compartments=len(members),
                       n_gate_pass=sum(passes_gate(row) for row in members),
                       median_stability=median(row["stability"] for row in members),
                       median_null_auc=median(row["null_auc"] for row in members),
                       min_null_auc=min(row["null_auc"] for row in members),
                       max_null_auc=max(row["null_auc"] for row in members))
        for rarity in ("rare", "intermediate", "abundant"):
            group = [row for row in members if row["rarity"] == rarity]
            summary[f"{rarity}_n"] = len(group)
            summary[f"{rarity}_pass"] = sum(passes_gate(row) for row in group)
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    """Write stable portable tables, retaining six-decimal summary precision."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: f"{value:.6f}" if isinstance(value, float) else value
                         for key, value in row.items()} for row in rows)


def write_latex(path, summaries):
    """Keep the comparison descriptive; percentages obscure these small denominators."""
    lines = [r"\begin{tabular}{|l|r|c|c|c|}", r"\hline",
             r"\textbf{Cohort} & \textbf{Compartments} & \textbf{Gate pass} & \textbf{Median stability} & \textbf{Median null AUC} \\ \hline"]
    for row in summaries:
        lines.append(f"{row['cohort_label']} & {row['n_compartments']} & "
                     f"{row['n_gate_pass']}/{row['n_compartments']} & "
                     f"{row['median_stability']:.3f} & {row['median_null_auc']:.3f} "
                     + r"\\ \hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    input_paths = [args.input_dir / "axis3_primary_metrics.csv", args.input_dir / "axis3_delorey_metrics.csv"]
    summaries = summarize(read_inputs(input_paths))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tissue_context_summary.csv", summaries)
    write_latex(args.out_dir / "tissue_context_table.tex", summaries)
    manifest = [dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for path in input_paths]
    write_csv(args.out_dir / "input_checksums.csv", manifest)
    print(f"Summarized {sum(row['n_compartments'] for row in summaries)} compartments in {len(summaries)} cohort subsets.")


if __name__ == "__main__":
    main()
