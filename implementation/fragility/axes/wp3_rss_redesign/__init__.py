"""WP-3: RSS redesign, weight sweep, and empirical null.

Addresses reviewers **R1a(iii)**, **R1a(iv)**, **R1a(v)**, **R4(4)** and
**R3** (metric definitions). The runner computes for every (dataset,
scorer, resolution-pair) combination:

1. The three RSS components separately (overlap_loss, jaccard_loss,
   drift_norm) using the new bounded drift.
2. The composite RSS under a grid of weight triples covering the
   simplex at step 0.1.
3. An empirical null-RSS distribution built from rank-permutation
   re-rankings, so the observed value has a calibrated z-score.
"""
