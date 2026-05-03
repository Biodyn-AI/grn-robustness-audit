"""Minimal scGPT loading helper that bypasses the broken torchtext C ext.

scGPT's ``TransformerModel`` needs only ``vocab[pad_token]`` at init time,
so a plain Python ``dict`` works as a stand-in vocab once ``torchtext``
is stubbed out in :data:`sys.modules`. This module performs that stub
once, exposes :func:`load_scgpt_model`, and leaves the rest of the scGPT
machinery untouched.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# torchtext stub (installed once at import time)
# ---------------------------------------------------------------------------


def _install_torchtext_stub() -> None:
    if "torchtext" in sys.modules and hasattr(sys.modules["torchtext"], "vocab"):
        return

    class _StubVocab:
        def __init__(self, tokens=None, default_index=None):
            self._tokens: List[str] = list(tokens or [])
            self._stoi: Dict[str, int] = {t: i for i, t in enumerate(self._tokens)}
            self._default = default_index

        def __contains__(self, token):
            return token in self._stoi

        def __getitem__(self, token):
            if token in self._stoi:
                return self._stoi[token]
            if self._default is not None:
                return self._default
            raise KeyError(token)

        def __len__(self):
            return len(self._tokens)

        def get_stoi(self):
            return dict(self._stoi)

        def get_itos(self):
            return list(self._tokens)

        def set_default_index(self, idx):
            self._default = idx

        def get_default_index(self):
            return self._default

        def append_token(self, token):
            if token not in self._stoi:
                self._stoi[token] = len(self._tokens)
                self._tokens.append(token)

        def insert_token(self, token, index):
            self._tokens.insert(index, token)
            self._stoi = {t: i for i, t in enumerate(self._tokens)}

        def __call__(self, tokens):
            return [self[t] for t in tokens]

        def lookup_indices(self, tokens):
            return [self[t] for t in tokens]

        def lookup_tokens(self, indices):
            return [self._tokens[i] for i in indices]

    def _vocab(ordered_dict, specials=None, special_first=True, **kwargs):
        tokens = list(ordered_dict.keys())
        if specials:
            if special_first:
                tokens = list(specials) + tokens
            else:
                tokens = tokens + list(specials)
        return _StubVocab(tokens=tokens)

    tt = types.ModuleType("torchtext")
    tt_vocab = types.ModuleType("torchtext.vocab")
    tt_vocab.Vocab = _StubVocab
    tt_vocab.vocab = _vocab
    tt.vocab = tt_vocab
    sys.modules["torchtext"] = tt
    sys.modules["torchtext.vocab"] = tt_vocab


# Always install the stub before anything scGPT-related is imported.
_install_torchtext_stub()


# ---------------------------------------------------------------------------
# Dict-like vocab for scGPT TransformerModel initialization
# ---------------------------------------------------------------------------


class DictVocab:
    """Minimal vocab that supports ``vocab[token] -> int`` and ``token in vocab``.

    scGPT's TransformerModel only calls ``vocab[pad_token]`` once, so this
    suffices even though it lacks ``append_token`` / ``set_default_index``.
    """

    def __init__(self, token_to_idx: Dict[str, int]):
        self._t2i = dict(token_to_idx)
        self._i2t = {i: t for t, i in self._t2i.items()}
        self._pad_token: Optional[str] = None
        if "<pad>" in self._t2i:
            self._pad_token = "<pad>"

    @classmethod
    def from_json(cls, path: Path | str) -> "DictVocab":
        path = Path(path)
        with path.open("r") as f:
            t2i = json.load(f)
        return cls(t2i)

    def __contains__(self, token) -> bool:
        return token in self._t2i

    def __getitem__(self, token):
        if token in self._t2i:
            return self._t2i[token]
        raise KeyError(token)

    def __len__(self):
        return max(self._t2i.values()) + 1 if self._t2i else 0

    def get(self, token, default=None):
        return self._t2i.get(token, default)

    def get_stoi(self):
        return dict(self._t2i)

    def set_default_index(self, idx):
        self._default = idx

    @property
    def pad_token(self):
        return self._pad_token


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _add_scgpt_to_path() -> None:
    p = Path(
        "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/external/scGPT"
    )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_scgpt_model(
    checkpoint_dir: Path | str,
    device: str = "cpu",
):
    """Load scGPT model weights from a checkpoint directory.

    Returns ``(model, vocab, args)`` where ``vocab`` is a :class:`DictVocab`.
    """

    _install_torchtext_stub()
    _add_scgpt_to_path()

    import torch

    checkpoint_dir = Path(checkpoint_dir)
    args_path = checkpoint_dir / "args.json"
    vocab_path = checkpoint_dir / "vocab.json"
    model_path = checkpoint_dir / "best_model.pt"

    with args_path.open() as f:
        args = json.load(f)
    vocab = DictVocab.from_json(vocab_path)

    # Lazy import so stub is in place before scGPT touches torchtext
    from scgpt.model import TransformerModel

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        nlayers_cls=args.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=0.0,   # inference: no dropout
        pad_token=args.get("pad_token", "<pad>"),
        pad_value=args.get("pad_value", -2),
        do_mvc=True,    # mvc_decoder present in state_dict
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        n_input_bins=args.get("n_bins", 51),
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,    # we skip flash-attn for attention extraction
        pre_norm=False,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # scGPT whole-human checkpoint was trained with FlashAttention which
    # stores combined QKV weights under ``self_attn.Wqkv.weight``. The
    # shape ``(3 * d_model, d_model)`` is identical to ``in_proj_weight``
    # in ``nn.MultiheadAttention``, so we rename the keys.
    remapped: Dict[str, "torch.Tensor"] = {}
    for k, v in state.items():
        new_k = (
            k.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
             .replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
        )
        remapped[new_k] = v
    state = remapped

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, vocab, args, {"missing": missing, "unexpected": unexpected}


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------


def attention_matrix_for_cells(
    model,
    vocab: DictVocab,
    gene_symbols: List[str],
    expression: np.ndarray,
    device: str = "cpu",
    layer_aggregation: str = "mean",
    head_aggregation: str = "mean",
    max_seq_len: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (n_genes, n_genes) attention for one cell or a batch.

    Parameters
    ----------
    model
        Loaded scGPT TransformerModel.
    vocab
        DictVocab mapping gene symbol -> token id.
    gene_symbols
        Ordered list of gene symbols (upper-case).
    expression
        ``(n_cells, n_genes)`` expression matrix (already normalised + log1p).
    layer_aggregation
        ``"mean"`` / ``"last"`` / ``"max"`` over layers.
    head_aggregation
        ``"mean"`` / ``"max"`` over attention heads.

    Returns
    -------
    attention : np.ndarray
        ``(n_genes, n_genes)`` averaged over cells, heads, layers.
    in_vocab : np.ndarray[bool]
        Per-gene mask indicating which genes have a token in the scGPT
        vocabulary. Attention rows/cols for OOV genes are zero.
    """

    import torch

    n_cells, n_genes = expression.shape
    token_ids = np.array([vocab.get(g, -1) for g in gene_symbols], dtype=int)
    in_vocab = token_ids >= 0

    # Truncate to max_seq_len by selecting the top-|max_seq_len| genes per cell
    # by expression, but keep the same global gene order in the output matrix.
    # For this extractor we keep the top ``max_seq_len`` highest-variance
    # in-vocab genes globally.
    if in_vocab.sum() > max_seq_len:
        # Prefer highest-variance in-vocab genes
        var = expression[:, in_vocab].var(axis=0)
        top_local = np.argsort(var)[::-1][:max_seq_len]
        in_vocab_idx = np.where(in_vocab)[0][top_local]
        keep_mask = np.zeros(n_genes, dtype=bool)
        keep_mask[in_vocab_idx] = True
    else:
        keep_mask = in_vocab

    kept_idx = np.where(keep_mask)[0]
    kept_tokens = torch.as_tensor(token_ids[kept_idx], dtype=torch.long, device=device)
    kept_expr = torch.as_tensor(
        expression[:, kept_idx], dtype=torch.float32, device=device,
    )

    attention_accum = np.zeros((n_genes, n_genes), dtype=np.float32)
    count = np.zeros((n_genes, n_genes), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for cell_i in range(n_cells):
            tokens = kept_tokens.unsqueeze(0)                 # (1, L)
            values = kept_expr[cell_i].unsqueeze(0)            # (1, L)
            pad_mask = torch.zeros_like(tokens, dtype=torch.bool)

            # Use model.encoder + value_encoder + transformer_encoder directly
            # with attention hooks.
            gene_emb = model.encoder(tokens)                   # (1, L, d)
            val_emb = model.value_encoder(values.unsqueeze(-1))  # (1, L, d) for continuous encoder that accepts (L,1)
            # ContinuousValueEncoder expects values shape (B, L); check
            try:
                val_emb = model.value_encoder(values)
            except Exception:
                val_emb = model.value_encoder(values.unsqueeze(-1))
            x = gene_emb + val_emb                              # (1, L, d)

            # Run each transformer layer manually with need_weights=True
            cell_attn: List[np.ndarray] = []
            for layer in model.transformer_encoder.layers:
                # torch.nn.TransformerEncoderLayer: self_attn is nn.MultiheadAttention
                attn_out, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=pad_mask,
                    need_weights=True,
                    average_attn_weights=(head_aggregation == "mean"),
                )
                # Standard post-norm path
                x = layer.norm1(x + layer.dropout1(attn_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff))
                cell_attn.append(attn_weights[0].detach().cpu().numpy())

            arr = np.stack(cell_attn, axis=0)   # (L_layers, Lseq, Lseq)
            if layer_aggregation == "mean":
                attn2d = arr.mean(axis=0)
            elif layer_aggregation == "last":
                attn2d = arr[-1]
            elif layer_aggregation == "max":
                attn2d = arr.max(axis=0)
            else:
                raise ValueError(f"unknown layer_aggregation {layer_aggregation}")

            # Symmetrise: scGPT attention has direction; for TF->target scoring
            # we average both directions to be conservative.
            attn2d = 0.5 * (attn2d + attn2d.T)
            # Accumulate into the global gene-gene matrix
            row_idx = kept_idx[:, None]
            col_idx = kept_idx[None, :]
            attention_accum[row_idx, col_idx] += attn2d
            count[row_idx, col_idx] += 1.0

    count = np.where(count == 0, 1, count)
    return attention_accum / count, in_vocab


__all__ = [
    "DictVocab",
    "load_scgpt_model",
    "attention_matrix_for_cells",
]
