#!/usr/bin/env python3
"""Recompute all method-agreement metrics from portable stored edge-score vectors.

This restates the original fragility.metrics formulas using NumPy/SciPy only.
It does not need the source h5ad, checkpoint, torch, scGPT, or a fragility install.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

METHODS = ("pearson", "mutual_info", "grnboost2", "genie3", "scgpt_attention")


def top_set(scores, k):
    """Preserve the historical NumPy argpartition boundary-tie behavior exactly."""
    if k >= len(scores):
        return np.argsort(-scores, kind="mergesort")
    chosen = np.argpartition(-scores, k)[:k]
    return chosen[np.argsort(-scores[chosen], kind="mergesort")]


def agreement(first, second, k):
    """Compute signed Spearman plus the declared bounded-drift RSS components."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) < k:
        raise ValueError("Aligned score vectors must be one-dimensional and at least k long.")
    first_top, second_top = top_set(first, k), top_set(second, k)
    common = np.intersect1d(first_top, second_top, assume_unique=True)
    overlap = len(common) / k
    jaccard = len(common) / len(np.union1d(first_top, second_top))
    if len(common):
        first_ranks = np.argsort(np.argsort(-first, kind="mergesort"), kind="mergesort")
        second_ranks = np.argsort(np.argsort(-second, kind="mergesort"), kind="mergesort")
        drift = float(min(1, np.median(np.abs(first_ranks[common] - second_ranks[common])) / k))
    else:
        drift = 1.0
    return dict(spearman=float(spearmanr(first, second).statistic), overlap=overlap,
                jaccard=jaccard, drift_norm=drift,
                rss=0.4 * (1 - overlap) + 0.3 * (1 - jaccard) + 0.3 * drift)


def read_vectors(input_dir):
    """Validate full edge identity/order before comparing the numerical vectors."""
    vectors, identities = {}, None
    for method in METHODS:
        with (input_dir / f"scores_{method}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        edges = [(row["tf"], row["target"]) for row in rows]
        if identities is None:
            identities = edges
        if edges != identities or len(set(edges)) != len(edges):
            raise ValueError(f"Misaligned or duplicate edge identities: {method}")
        values = np.asarray([float(row["score"]) for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite score: {method}")
        vectors[method] = values
    return vectors


def recompute(input_dir):
    vectors = read_vectors(input_dir)
    rows = []
    for first, second in itertools.combinations(METHODS, 2):
        small = agreement(vectors[first], vectors[second], k=100)
        large = agreement(vectors[first], vectors[second], k=500)
        rows.append(dict(scorer_a=first, scorer_b=second, spearman=large["spearman"],
                         top100_jaccard=small["jaccard"], top500_jaccard=large["jaccard"],
                         top500_rss=large["rss"], top500_overlap=large["overlap"],
                         top500_drift_norm=large["drift_norm"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--output", type=Path, help="Optional path for the recomputed comparison CSV.")
    parser.add_argument("--check", action="store_true", help="Check all metrics against the stored table at absolute tolerance 1e-14.")
    args = parser.parse_args()
    rows = recompute(args.input_dir)
    if args.check:
        with (args.input_dir / "method_agreement.csv").open(newline="") as stream:
            stored_rows = list(csv.DictReader(stream))
        if len(stored_rows) != len(rows):
            raise ValueError("Stored table has a different number of method pairs.")
        for stored, fresh in zip(stored_rows, rows):
            for key, value in fresh.items():
                if key.startswith("scorer_"):
                    if stored[key] != value:
                        raise ValueError("Stored method-pair order differs.")
                elif not np.isclose(float(stored[key]), value, atol=1e-14, rtol=0):
                    raise ValueError(f"Stored metric differs: {fresh['scorer_a']}/{fresh['scorer_b']}/{key}")
        print("All 10 method pairs match the stored table at absolute tolerance 1e-14.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")
    if not args.check and not args.output:
        parser.error("Specify --check, --output, or both.")


if __name__ == "__main__":
    main()
