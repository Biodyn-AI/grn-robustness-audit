"""Real scGPT attention scorer built on the minimal loader.

Replaces the stub in ``scgpt_attention.py`` with a working implementation.
Accepts TF/target indices over a shared gene universe, returns the
attention-derived (n_tfs, n_targets) score matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import EdgeScorer, register_scorer


@register_scorer
class ScGPTAttentionRealScorer(EdgeScorer):
    """scGPT attention score between each TF and target.

    Configuration (kwargs):

    * ``checkpoint`` — directory name under ``scGPT_checkpoints/``.
      Default ``"whole-human"``.
    * ``n_cells`` — number of cells to average attention over.
      Default 200.
    * ``max_seq_len`` — max tokens per forward pass. Default 1200.
    * ``layer_aggregation`` / ``head_aggregation`` — ``"mean"`` / ``"max"`` / ``"last"``.
    * ``device`` — ``"cpu"`` / ``"mps"`` / ``"cuda"``.
    """

    name = "scgpt_attention"
    label = "scGPT attention"
    supports_sign = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._vocab = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from .scgpt_minimal import load_scgpt_model

        ckpt = Path(
            "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/"
            "external/scGPT_checkpoints"
        ) / str(self.options.get("checkpoint", "whole-human"))
        device = str(self.options.get("device", "cpu"))
        model, vocab, args, _ = load_scgpt_model(ckpt, device=device)
        self._model = model
        self._vocab = vocab
        self._device = device

    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        from .scgpt_minimal import attention_matrix_for_cells

        self._ensure_loaded()
        gene_names = np.asarray(self.options["gene_names"])
        # Subsample cells if requested (attention is averaged uniformly).
        n_cells_total = X.shape[0]
        n_cells_use = int(self.options.get("n_cells", min(200, n_cells_total)))
        if n_cells_use < n_cells_total:
            rng = np.random.default_rng(int(self.options.get("random_state", 20260218)))
            idx = rng.choice(n_cells_total, size=n_cells_use, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        all_gene_idx = np.unique(np.concatenate([tf_idx, target_idx]))
        gene_subset_names = list(gene_names[all_gene_idx])
        X_slice = X_sub[:, all_gene_idx]

        attn, in_vocab = attention_matrix_for_cells(
            self._model, self._vocab,
            gene_symbols=gene_subset_names,
            expression=X_slice,
            device=self._device,
            layer_aggregation=str(self.options.get("layer_aggregation", "mean")),
            head_aggregation=str(self.options.get("head_aggregation", "mean")),
            max_seq_len=int(self.options.get("max_seq_len", 1200)),
        )
        # Build TF -> target matrix in the caller's row/col order.
        local_idx = {g: i for i, g in enumerate(all_gene_idx.tolist())}
        scores = np.zeros((len(tf_idx), len(target_idx)), dtype=float)
        for r, t in enumerate(tf_idx.tolist()):
            for c, g in enumerate(target_idx.tolist()):
                scores[r, c] = attn[local_idx[t], local_idx[g]]
        return scores, None
