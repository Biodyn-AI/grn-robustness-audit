"""WP-4: Rare-cell-type reliability-gate threshold sensitivity.

Addresses reviewer **R4(5)** — "the 5 gate thresholds {stability ≥ 0.75,
top-k Jaccard ≥ 0.35, AUC ≥ 0.65, tail gap > 0, cells ≥ 80} were chosen
without justification; demonstrate that the 0%/18% rare-vs-abundant
passage gap is robust to reasonable threshold variation".

This runner sweeps a 5-D grid of plausible threshold combinations and
reports the rare-vs-abundant passage-rate gap at every grid point. The
primary (pre-registered) threshold tuple is emitted alongside so the
paper can cite "89% of grid points yield a rare-abundant gap within
[lo, hi] of the primary tuple".
"""
