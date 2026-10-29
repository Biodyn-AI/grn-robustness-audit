#!/usr/bin/env python3
"""Verify and restore deposited result tables; optionally redraw their figures.

This is a stored-results path. It does not acquire or generate single-cell
measurements and does not re-run scGPT, integrations, or the original null draws.
Only the Python standard library is needed unless --figures is requested.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
BUNDLE = REPOSITORY / 'data' / 'supporting_results'


def safe_path(root: Path, relative: str) -> Path:
    """Resolve manifest paths without allowing extraction outside its directory."""
    candidate = Path(relative)
    root = root.resolve()
    resolved = (root / candidate).resolve()
    if candidate.is_absolute() or root not in resolved.parents:
        raise ValueError(f'unsafe manifest path: {relative}')
    return resolved


def stream_hash(stream) -> str:
    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        checksum.update(block)
    return checksum.hexdigest()


def verify_file(bundle: Path, row: dict) -> None:
    """Check both the deposited bytes and the original uncompressed content."""
    compressed_path = safe_path(bundle, row['path'])
    with compressed_path.open('rb') as stream:
        actual = stream_hash(stream)
    if actual != row['sha256']:
        raise ValueError(f'compressed checksum mismatch: {row["path"]}')
    opener = gzip.open if compressed_path.suffix == '.gz' else open
    with opener(compressed_path, 'rb') as stream:
        actual = stream_hash(stream)
    if actual != row['uncompressed_sha256']:
        raise ValueError(f'original checksum mismatch: {row["path"]}')


def restore_file(bundle: Path, destination: Path, row: dict) -> None:
    """Materialize one checked table under its documented analysis-relative path."""
    source = safe_path(bundle, row['path'])
    target = safe_path(destination, row['restore_path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as incoming, target.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def estimate_mvcc(curve_path: Path) -> list[dict]:
    """Recompute Pearson MVCC at k=1000 from stored per-draw Jaccards.

    The original runner uses a mean across draws at each cell count, then
    selects the smallest count whose mean is at least 0.5. Recomputing this
    table is a numerical check; it does not regenerate the underlying draws.
    """
    grouped = defaultdict(list)
    with curve_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if int(row['k']) == 1000:
                key = (row['dataset'], row['scorer'], int(row['anchor_cells']), int(row['cells']))
                grouped[key].append(float(row['jaccard_mean']))
    strata = defaultdict(list)
    for (dataset, scorer, anchor, cells), values in grouped.items():
        if sum(values) / len(values) >= 0.5:
            strata[dataset, scorer, anchor].append(cells)
        else:
            strata[dataset, scorer, anchor]
    return [dict(dataset=dataset, scorer=scorer, anchor_cells=anchor,
                 mvcc_cells=min(cells) if cells else '')
            for (dataset, scorer, anchor), cells in sorted(strata.items())]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reproduced'))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--figures', action='store_true', help='requires numpy, pandas, matplotlib')
    args = parser.parse_args(argv)
    release_inventory = REPOSITORY / 'supporting_archive_contents.csv'
    if release_inventory.exists():
        with release_inventory.open(newline='') as stream:
            for row in csv.DictReader(stream):
                with safe_path(REPOSITORY, row['path']).open('rb') as artifact:
                    if stream_hash(artifact) != row['sha256']:
                        raise ValueError(f'release checksum mismatch: {row["path"]}')
    with (BUNDLE / 'manifest.csv').open(newline='') as stream:
        entries = list(csv.DictReader(stream))
    for row in entries:
        verify_file(BUNDLE, row)
    print(json.dumps({'verified_files': len(entries), 'mode': 'stored-results'}))
    if args.verify_only:
        return 0
    destination = args.out.resolve()
    for row in entries:
        restore_file(BUNDLE, destination, row)
    summaries = destination / 'recomputed'
    summaries.mkdir(parents=True, exist_ok=True)
    for cohort in ['wp2', 'wp2_delorey']:
        curves = destination / 'implementation_outputs' / cohort / 'mvcc_full_curves.csv'
        estimates = estimate_mvcc(curves)
        with (summaries / f'{cohort}_mvcc.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['dataset', 'scorer', 'anchor_cells', 'mvcc_cells'])
            writer.writeheader()
            writer.writerows(estimates)
    if args.figures:
        environment = dict(os.environ, FRAGILITY_RESULTS_ROOT=str(destination),
                           FRAGILITY_FIGURE_DIR=str(destination / 'figures'))
        subprocess.run([sys.executable, str(REPOSITORY / 'scripts' / 'generate_audit_figures.py')],
                       env=environment, check=True)
    print(f'Exact tables and recomputed MVCC summaries: {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
