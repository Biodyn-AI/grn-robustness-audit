#!/usr/bin/env python3
"""Retrieve a versioned provider H5AD referenced in the retained input metadata.

This retrieval utility makes the recorded file URLs explicit. It is not the
unrecovered historical download command and does not preprocess the data.
Downloads may be several gigabytes. No data are fetched without --cohort.
"""
import argparse
from pathlib import Path
import shutil
import urllib.request

VERSIONS = {
    'kidney': '65ca6e36-73b0-4c88-b0f3-7b23b48844ad',
    'lung': '40f8b1a3-9f76-4ac4-8761-32078555ed4e',
    'immune': 'e62b0182-368f-4875-8ac1-f49e3d5eda60',
    'external_lung': 'c88e0403-da93-40f4-99b5-f5fdeb81a82c',
    'delorey_covid_lung': '75c059c8-8fb7-4e6e-a618-a3e01ac42060',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=sorted(VERSIONS))
    parser.add_argument('--out-dir', type=Path, default=Path('source_h5ad'))
    args = parser.parse_args()
    if not args.cohort:
        for cohort, version in VERSIONS.items():
            print(f'{cohort}\thttps://datasets.cellxgene.cziscience.com/{version}.h5ad')
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / f'{args.cohort}.h5ad'
    if target.exists():
        raise FileExistsError(f'refusing to overwrite {target}')
    partial = target.with_suffix('.h5ad.partial')
    url = f'https://datasets.cellxgene.cziscience.com/{VERSIONS[args.cohort]}.h5ad'
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.rename(target)
    print(f'{url}\nSaved {target}; {target.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
