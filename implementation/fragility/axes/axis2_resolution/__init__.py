"""Axis 2: Cluster-resolution sensitivity.

This is the ported + refactored version of
``subproject_38_cluster_resolution_sensitivity/implementation/run_resolution_sensitivity.py``.

Key changes from the original:

1. Uses :mod:`fragility.data.loader` and :mod:`fragility.preprocessing` so
   preprocessing is stamped into provenance.
2. Uses :mod:`fragility.nulls` for the dual-null calibration, so the same
   implementations back Axes 2, 3, and 5 (R3 concern).
3. Uses :mod:`fragility.metrics.rss` — the **new bounded-drift RSS** —
   for the composite score (WP-3). Component values are also emitted.
4. Adds an ``export_scored_pairs.csv`` writer so WP-3 and WP-10 can
   consume the same scored edges without re-running the full pipeline.
5. Adds WP-14 hook: the ``triple_null`` flag adds a degree-preserving
   rewire null on top of the dual-null.
"""
