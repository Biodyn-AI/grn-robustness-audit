"""fragility: unified implementation for the Five Axes of Fragility audit.

This package is the code underlying the paper
"Five Axes of Fragility: A Systematic Robustness Audit of Gene-Regulatory-
Network Inference" (subproject_merged_A_robustness_audit).

It is organised as:

- ``fragility.preprocessing`` -- shared scRNA-seq preprocessing (P2).
- ``fragility.panels``        -- TF/target panel registry (P3).
- ``fragility.scorers``       -- pluggable edge-scoring engine (P4).
- ``fragility.nulls``         -- shared null-model library (P5).
- ``fragility.metrics``       -- metric definitions (Spearman stability,
  top-k Jaccard, RSS, null-separation AUC, tail gap, sign-flip rate, ...).
- ``fragility.axes``          -- the five axis pipelines, each one a thin
  orchestrator over the shared components.
- ``fragility.cli``           -- ``python -m fragility`` entrypoint.

Design rules
------------

1. Every numeric parameter is passed in from a config object, never
   hard-coded deep inside a function.
2. Every run emits (a) a JSON provenance block, (b) a single tidy CSV with
   one row per observation, and (c) no pickles. Plotting code consumes the
   CSVs, never re-runs the pipeline.
3. Every scorer implements the same ``score(X, tf_idx, target_idx) ->
   np.ndarray`` contract so axes can iterate over scorers without knowing
   which one they have.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
