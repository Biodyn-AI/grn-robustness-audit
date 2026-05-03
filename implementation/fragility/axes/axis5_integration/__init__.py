"""Axis 5: Integration-method sensitivity.

This module wraps the existing subproject_38_integration_method_sensitivity
outputs into the canonical ``scored_pairs.csv`` format so WP-3 (RSS)
and WP-10 (top-k scan) can consume them without re-running the full
integration pipeline.

For WP-6 (scVI + Scanorama additions) the runner is separately extended
in ``wp6_integration.py``; this module handles the baseline Harmony data
that already exists on disk.
"""

from .export import export_scored_pairs

__all__ = ["export_scored_pairs"]
