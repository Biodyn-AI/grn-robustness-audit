"""Axis 3: Rare-cell-type reliability stress test.

Compact port of subproject_38_rare_cell_type_stress_test using the shared
fragility infrastructure (preprocessing, panels, scorers, metrics). Emits
one tidy CSV: per-cell-type metrics (stability, top-k Jaccard, null AUC,
tail gap, cell count, rarity). This is the input to :mod:`fragility.axes.wp4_reliability_grid`.
"""
