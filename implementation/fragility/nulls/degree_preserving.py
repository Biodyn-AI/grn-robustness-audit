"""Degree-preserving edge rewiring null.

WP-14 (triple-null calibration): preserves each TF's out-degree and each
target's in-degree but rewires which specific targets they connect to.
This is less stringent than the global shuffle but more stringent than
the within-coarse shuffle. Addresses R3's "constrained shuffles may be
overly strict" concern by bracketing the plausible-null space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import NullFamily, NullResult, register_null


@register_null
class DegreePreservingNull(NullFamily):
    """Configuration-model rewire on the TF-target bipartite graph.

    Inputs are interpreted differently from the data-level nulls:

    * ``X`` is the ``(n_tfs, n_targets)`` score matrix.
    * ``labels`` is unused.

    The null picks a random adjacency with identical row/column sums
    (binarised at the given top-k threshold stored in ``options``).
    """

    name = "degree_preserving"
    label = "Degree-preserving rewire"
    clear_threshold = 0.05

    def permute(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> NullResult:
        k = int(self.options.get("top_k", 1000))
        scores = np.asarray(X)
        if scores.ndim != 2:
            raise ValueError("DegreePreservingNull expects a 2D score matrix")

        # Binarise at top-k across the full matrix
        flat = scores.ravel()
        if k >= flat.size:
            threshold = -np.inf
        else:
            threshold = np.partition(flat, flat.size - k)[flat.size - k]
        adj = (scores >= threshold).astype(int)
        row_sums = adj.sum(axis=1)
        col_sums = adj.sum(axis=0)

        # Configuration-model rewire: iteratively pick two edges and swap
        # their targets if the swap does not create a duplicate.
        edges = np.argwhere(adj == 1)
        n_edges = len(edges)
        if n_edges == 0:
            return NullResult(X=scores, labels=labels, meta={"null": self.name})

        swaps = n_edges * 4  # rewire aggressively
        perm = edges.copy()
        perm_set = {tuple(e) for e in perm}
        for _ in range(swaps):
            i, j = rng.integers(0, n_edges, size=2)
            if i == j:
                continue
            a, b = perm[i], perm[j]
            new_a = (a[0], b[1])
            new_b = (b[0], a[1])
            if new_a in perm_set or new_b in perm_set:
                continue
            perm_set.discard(tuple(a))
            perm_set.discard(tuple(b))
            perm_set.add(new_a)
            perm_set.add(new_b)
            perm[i] = new_a
            perm[j] = new_b

        # Reassemble a null score matrix with the same row/column totals.
        null_scores = np.zeros_like(scores)
        # Distribute each TF's original scores uniformly across its
        # *rewired* target set so the per-TF score distribution is unchanged.
        for i in range(scores.shape[0]):
            old_targets = np.where(adj[i] == 1)[0]
            new_targets = perm[perm[:, 0] == i][:, 1]
            if len(old_targets) == 0 or len(new_targets) == 0:
                continue
            # Preserve the sorted score order within the TF.
            sorted_scores = np.sort(scores[i, old_targets])[::-1]
            new_targets_sorted = new_targets[: len(sorted_scores)]
            null_scores[i, new_targets_sorted] = sorted_scores[: len(new_targets_sorted)]
        return NullResult(
            X=null_scores,
            labels=labels,
            meta={"null": self.name, "top_k": int(k)},
        )
