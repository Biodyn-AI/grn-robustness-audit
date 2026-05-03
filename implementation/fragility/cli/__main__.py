"""``python -m fragility <command> [...]`` dispatcher.

Commands are discovered via :mod:`fragility.cli.commands`. Each command
exposes a ``run(args)`` function and a ``help`` string; adding a new
work-package runner is a matter of dropping a module into that package.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Dict


# Mapping from CLI command name to the module that implements ``run(args)``.
COMMANDS: Dict[str, str] = {
    "axis1": "fragility.axes.axis1_cell_count.cli",
    "axis2": "fragility.axes.axis2_resolution.cli",
    "axis3": "fragility.axes.axis3_rare.cli",
    "axis4": "fragility.axes.axis4_donor.cli",
    "axis5": "fragility.axes.axis5_integration.cli",
    "wp1": "fragility.axes.wp1_method_diversity.cli",
    "wp2": "fragility.axes.wp2_mvcc_multi.cli",
    "wp3": "fragility.axes.wp3_rss_redesign.cli",
    "wp4": "fragility.axes.wp4_reliability_grid.cli",
    "wp6": "fragility.axes.wp6_integration.cli",
    "wp7": "fragility.axes.wp7_compliant_cohort.cli",
    "wp8": "fragility.axes.wp8_external_datasets.cli",
    "wp10": "fragility.axes.wp10_topk_scan.cli",
    "wp12": "fragility.axes.wp12_normalization.cli",
    "wp13": "fragility.axes.wp13_gene_dim.cli",
    "wp14": "fragility.axes.wp14_triple_null.cli",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fragility",
        description="Five Axes of Fragility reproduction runner",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    module_name = COMMANDS[args.command]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"error: {args.command} runner not yet implemented ({e})", file=sys.stderr)
        return 2
    return int(module.run(args.rest) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
