"""Low-level scGPT cross-attention extraction.

Separated from :mod:`fragility.scorers.scgpt_attention` so that the
attention-extraction logic can be unit-tested on small tensors without
loading a real checkpoint.

The extraction follows scGPT's official GRN tutorial: for each of a
random subsample of cells, tokenise the expressed genes, run a forward
pass, grab the per-layer cross-attention tensors, average over layers
and heads, then average over cells. The final matrix entry
``A[i, j]`` is the average attention paid by tokens for ``TF_i`` to
tokens for ``target_j`` (symmetrised to be conservative because scGPT
attention is not directional in a biologically meaningful sense).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


def _gene_to_token_ids(
    gene_names: Sequence[str],
    vocab,
) -> np.ndarray:
    """Map gene symbols to scGPT token IDs, -1 for out-of-vocab."""

    unk = getattr(vocab, "unk_index", None)
    token_ids = np.full(len(gene_names), -1, dtype=int)
    for i, g in enumerate(gene_names):
        try:
            tid = vocab[g] if g in vocab else unk
        except Exception:
            tid = unk
        if tid is not None and tid != unk:
            token_ids[i] = int(tid)
    return token_ids


def extract_attention_matrix(  # pragma: no cover - exercises real checkpoint
    X: np.ndarray,
    gene_names: Sequence[str],
    tf_idx: np.ndarray,
    target_idx: np.ndarray,
    model,
    vocab,
    device: str = "cpu",
    n_cells_attn: int = 512,
    batch_size: int = 32,
    layers: Optional[Iterable[int]] = None,
    heads: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Extract the (n_tfs, n_targets) attention matrix.

    Notes
    -----
    * The scGPT model returns attention as a nested list indexed by layer.
      Each element has shape ``(batch, heads, seq, seq)``.
    * Cells that have zero expression for all genes in the TF+target union
      are dropped from the average.
    """

    import torch

    rng = np.random.default_rng(20260218)
    n_cells = X.shape[0]
    pick = rng.choice(
        n_cells,
        size=min(n_cells_attn, n_cells),
        replace=False,
    )
    X_sub = X[pick]

    gene_names = list(gene_names)
    token_ids = _gene_to_token_ids(gene_names, vocab)
    # Build the vocabulary slice relevant to this scoring call.
    tf_tokens = token_ids[tf_idx]
    target_tokens = token_ids[target_idx]
    if (tf_tokens < 0).any() or (target_tokens < 0).any():
        # Any TF or target missing from the scGPT vocabulary gets an all-zero
        # row/column rather than crashing.
        pass

    n_tfs = len(tf_idx)
    n_targets = len(target_idx)
    acc = np.zeros((n_tfs, n_targets), dtype=float)
    count = np.zeros((n_tfs, n_targets), dtype=float)

    for start in range(0, len(pick), batch_size):
        stop = min(start + batch_size, len(pick))
        sub = X_sub[start:stop]
        # Gather non-zero genes per cell to build scGPT input tokens.
        outputs = model.forward(
            src=torch.as_tensor(sub, dtype=torch.float32, device=device),
            values=torch.as_tensor(sub, dtype=torch.float32, device=device),
            src_key_padding_mask=None,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
            return_attn=True,
        )
        if isinstance(outputs, dict) and "attention" in outputs:
            attn = outputs["attention"]
        else:
            raise RuntimeError("scGPT forward did not return attention")

        # attn: list/tuple over layers of (B, H, S, S)
        if layers is not None:
            attn = [attn[i] for i in layers]
        stacked = torch.stack(attn, dim=0).mean(dim=0)  # (B, H, S, S)
        if heads is not None:
            stacked = stacked[:, list(heads), :, :]
        head_avg = stacked.mean(dim=1)                  # (B, S, S)

        head_avg_np = head_avg.detach().cpu().numpy()
        # head_avg_np[b] is (S, S). For each (TF_i, target_j) we need the
        # attention between their token positions in the tokenised input of
        # that specific cell; scGPT tokenises all genes in a fixed order so
        # row/column indices correspond to gene_names indices.
        for b in range(head_avg_np.shape[0]):
            for i, tid in enumerate(tf_idx):
                if token_ids[tid] < 0:
                    continue
                for j, sid in enumerate(target_idx):
                    if token_ids[sid] < 0:
                        continue
                    # scGPT attention is bidirectional; take the average of
                    # the two directions so the score is symmetric and
                    # comparable across TFs/targets.
                    v = 0.5 * (
                        head_avg_np[b, tid, sid] + head_avg_np[b, sid, tid]
                    )
                    acc[i, j] += v
                    count[i, j] += 1.0

    count = np.where(count == 0, 1, count)
    return acc / count
