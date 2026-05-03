"""scGPT attention-based edge scorer.

This is the scorer that directly addresses **R1a(i), R2(2), R3(title),
R4(1), R4(7)** — the reviewers' convergent complaint that the paper's
title claims foundation-model coverage while the empirical work used only
Pearson correlation.

For each TF–target pair the scorer returns the layer/head-averaged
cross-attention weight extracted from the scGPT checkpoint, following the
recipe in scGPT's own GRN tutorial (Cui et al. 2024, Fig. 3).

Checkpoint layout on this machine:
```
single_cell_mechinterp/external/scGPT_checkpoints/{whole-human,kidney,brain}/
    args.json
    vocab.json
    best_model.pt
```

Because this is a heavy dependency chain (torch + scGPT), the scorer is
imported lazily from :mod:`fragility.scorers.__init__`. If either the
checkpoint or the torch stack is missing, :class:`ScGPTAttentionScorer`
raises a ``RuntimeError`` on instantiation so the axes can skip it
gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


_CHECKPOINT_ROOT = Path(
    "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/"
    "external/scGPT_checkpoints"
)


try:  # pragma: no cover - torch may be missing
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False


from .base import EdgeScorer, register_scorer


@register_scorer
class ScGPTAttentionScorer(EdgeScorer):
    """Layer/head-averaged cross-attention score from scGPT.

    Hyperparameters (passed as kwargs at construction):

    * ``checkpoint``: directory name under ``scGPT_checkpoints/``. Default
      ``"whole-human"``. Known-valid alternates: ``"kidney"``.
    * ``n_cells_attn``: number of cells to aggregate attention over
      (default 512). Attention is averaged across this random subsample.
    * ``layers``: list of transformer layer indices to average over.
      Default ``None`` (all layers).
    * ``heads``: list of attention-head indices to average over. Default
      ``None`` (all heads).
    * ``batch_size``: cells per forward pass.
    * ``device``: ``"cpu"`` / ``"cuda"`` / ``None`` (auto-detect).
    """

    name = "scgpt_attention"
    label = "scGPT attention"
    supports_sign = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError(
                "torch is not importable in this environment; "
                "scGPT attention scorer is unavailable."
            )
        ckpt_name = str(self.options.get("checkpoint", "whole-human"))
        ckpt_dir = _CHECKPOINT_ROOT / ckpt_name
        if not ckpt_dir.exists():
            raise RuntimeError(f"scGPT checkpoint not found: {ckpt_dir}")
        self._ckpt_dir = ckpt_dir
        self._loaded = False

    # -- lazy model load so construction is cheap even on low-RAM nodes --

    def _load_model(self):  # pragma: no cover - requires real checkpoint
        import sys

        scgpt_root = Path(
            "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/external/scGPT"
        )
        if str(scgpt_root) not in sys.path:
            sys.path.insert(0, str(scgpt_root))
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab

        args_path = self._ckpt_dir / "args.json"
        vocab_path = self._ckpt_dir / "vocab.json"
        model_path = self._ckpt_dir / "best_model.pt"
        with args_path.open() as f:
            args = json.load(f)
        vocab = GeneVocab.from_file(vocab_path)

        model = TransformerModel(
            ntoken=len(vocab),
            d_model=args["embsize"],
            nhead=args["nheads"],
            d_hid=args["d_hid"],
            nlayers=args["nlayers"],
            nlayers_cls=args.get("n_layers_cls", 3),
            n_cls=1,
            vocab=vocab,
            dropout=args.get("dropout", 0.0),
            pad_token=args.get("pad_token", "<pad>"),
            pad_value=args.get("pad_value", -2),
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            n_input_bins=args.get("n_bins", 51),
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )
        device = self.options.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        state = torch.load(model_path, map_location=device)
        # scGPT checkpoints ship both raw state_dict and nested dicts.
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        self._model = model
        self._vocab = vocab
        self._device = device
        self._loaded = True

    def _score(
        self,
        X: np.ndarray,
        tf_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:  # pragma: no cover
        """Return cross-attention weight matrix for each TF → target pair."""

        if not self._loaded:
            self._load_model()

        # Heavy extraction path is implemented in a separate module that
        # understands scGPT's tokenisation. This method delegates to it.
        from .scgpt_attention_extract import extract_attention_matrix

        gene_names = np.asarray(self.options["gene_names"])
        matrix = extract_attention_matrix(
            X=X,
            gene_names=gene_names,
            tf_idx=tf_idx,
            target_idx=target_idx,
            model=self._model,
            vocab=self._vocab,
            device=self._device,
            n_cells_attn=int(self.options.get("n_cells_attn", 512)),
            batch_size=int(self.options.get("batch_size", 32)),
            layers=self.options.get("layers"),
            heads=self.options.get("heads"),
        )
        return matrix, None
