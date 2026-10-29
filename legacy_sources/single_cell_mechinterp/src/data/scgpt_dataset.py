from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import scipy.sparse as sp
import torch
import zlib
from torch.utils.data import Dataset


@dataclass
class ScGPTDatasetConfig:
    max_genes: int = 2048
    include_zero: bool = False
    sort_by_expression: bool = True
    pad_token_id: int | None = None
    cls_token_id: int | None = None
    force_gene_names: list[str] | None = None
    reserve_nonzero_gene_names: list[str] | None = None
    pre_binning_mask_gene_names: list[str] | None = None
    n_bins: int | None = None
    binning_seed: int = 0
    binning_key: str = "index"
    cache_items: bool = False


def bin_expression_values(
    values: np.ndarray,
    n_bins: int,
    seed: int,
) -> np.ndarray:
    """Apply scGPT-style per-cell quantile binning deterministically."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("Expression values must be non-negative before binning")

    binned = np.zeros(values.shape, dtype=np.float32)
    nonzero_indices = np.flatnonzero(values > 0)
    if nonzero_indices.size == 0:
        return binned

    nonzero_values = values[nonzero_indices]
    edges = np.quantile(nonzero_values, np.linspace(0, 1, n_bins - 1))
    left_digits = np.digitize(nonzero_values, edges)
    right_digits = np.digitize(nonzero_values, edges, right=True)
    # Match scGPT's legacy NumPy tie-breaking distribution without mutating
    # global RNG state, so repeated access to the same cell is reproducible.
    rng = np.random.RandomState(seed)
    digits = np.ceil(
        rng.rand(nonzero_values.size) * (right_digits - left_digits) + left_digits
    ).astype(np.int64)
    binned[nonzero_indices] = digits.astype(np.float32)
    return binned


class ScGPTDataset(Dataset):
    def __init__(
        self,
        adata,
        gene_to_id: Dict[str, int],
        config: ScGPTDatasetConfig,
    ):
        self.adata = adata
        self.config = config
        self.gene_to_id = gene_to_id
        if self.config.pad_token_id is None:
            raise ValueError("pad_token_id must be set before constructing the dataset")
        self.gene_names = list(adata.var_names)
        missing = [name for name in self.gene_names if name not in gene_to_id]
        if missing:
            raise ValueError(f"Genes missing from vocab: {len(missing)}")
        self.gene_ids = np.array([gene_to_id[name] for name in self.gene_names], dtype=np.int64)
        self.gene_indices = np.arange(len(self.gene_names), dtype=np.int64)
        self.force_gene_indices = None
        self.reserve_nonzero_gene_indices = None
        self.pre_binning_mask_gene_indices = None
        self._item_cache: dict[int, Dict[str, torch.Tensor]] = {}
        if self.config.force_gene_names:
            name_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
            indices = [
                name_to_idx[name]
                for name in self.config.force_gene_names
                if name in name_to_idx
            ]
            if indices:
                self.force_gene_indices = np.array(sorted(set(indices)), dtype=np.int64)
        if self.config.reserve_nonzero_gene_names:
            name_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
            reserve_indices = [
                name_to_idx[name]
                for name in self.config.reserve_nonzero_gene_names
                if name in name_to_idx
            ]
            if reserve_indices:
                self.reserve_nonzero_gene_indices = np.array(
                    sorted(set(reserve_indices)), dtype=np.int64
                )
        if self.config.pre_binning_mask_gene_names:
            name_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
            mask_indices = [
                name_to_idx[name]
                for name in self.config.pre_binning_mask_gene_names
                if name in name_to_idx
            ]
            if mask_indices:
                self.pre_binning_mask_gene_indices = np.array(
                    sorted(set(mask_indices)), dtype=np.int64
                )

    def __len__(self) -> int:
        return self.adata.n_obs

    def set_active_reserved_gene_indices(self, indices: Iterable[int]) -> None:
        """Reserve expressed genes for the next pass and invalidate token cache."""
        values = sorted({int(index) for index in indices})
        self.reserve_nonzero_gene_indices = (
            np.asarray(values, dtype=np.int64) if values else None
        )
        self._item_cache.clear()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.config.cache_items and idx in self._item_cache:
            return self._item_cache[idx]

        row = self.adata.X[idx]
        if sp.issparse(row):
            values = row.toarray().ravel()
        else:
            values = np.asarray(row).ravel()

        if self.pre_binning_mask_gene_indices is not None:
            values = values.copy()
            values[self.pre_binning_mask_gene_indices] = 0.0

        if self.config.n_bins is not None:
            if self.config.binning_key == "obs_name":
                cell_key = zlib.crc32(
                    str(self.adata.obs_names[idx]).encode("utf-8")
                )
            elif self.config.binning_key == "index":
                cell_key = int(idx)
            else:
                raise ValueError(f"Unsupported binning_key: {self.config.binning_key}")
            values = bin_expression_values(
                values,
                n_bins=int(self.config.n_bins),
                seed=(int(self.config.binning_seed) + cell_key) % (2**32),
            )

        if self.config.include_zero:
            mask = np.ones_like(values, dtype=bool)
        else:
            mask = values > 0
        if self.force_gene_indices is not None:
            mask[self.force_gene_indices] = True

        gene_ids = self.gene_ids[mask]
        gene_indices = self.gene_indices[mask]
        expr_values = values[mask]

        if gene_ids.size == 0:
            gene_ids = np.array([], dtype=np.int64)
            gene_indices = np.array([], dtype=np.int64)
            expr_values = np.array([], dtype=np.float32)

        if self.config.sort_by_expression and gene_ids.size > 1:
            order = np.argsort(-expr_values)
            gene_ids = gene_ids[order]
            gene_indices = gene_indices[order]
            expr_values = expr_values[order]

        max_genes = self.config.max_genes
        if self.config.cls_token_id is not None and max_genes > 0:
            max_genes = max_genes - 1
        if max_genes > 0 and (
            self.force_gene_indices is not None
            or self.reserve_nonzero_gene_indices is not None
        ) and gene_ids.size > 0:
            priority_mask = np.zeros(gene_indices.shape, dtype=bool)
            if self.force_gene_indices is not None:
                priority_mask |= np.isin(gene_indices, self.force_gene_indices)
            if self.reserve_nonzero_gene_indices is not None:
                # Reserved genes reach this point only if expressed; unlike
                # force_gene_indices, reservation never injects an absent token.
                priority_mask |= np.isin(
                    gene_indices, self.reserve_nonzero_gene_indices
                )
            forced_ids = gene_ids[priority_mask]
            forced_indices = gene_indices[priority_mask]
            forced_values = expr_values[priority_mask]
            if forced_ids.size > max_genes:
                forced_ids = forced_ids[:max_genes]
                forced_indices = forced_indices[:max_genes]
                forced_values = forced_values[:max_genes]
                gene_ids = forced_ids
                gene_indices = forced_indices
                expr_values = forced_values
            else:
                remaining_mask = ~priority_mask
                remaining_ids = gene_ids[remaining_mask]
                remaining_indices = gene_indices[remaining_mask]
                remaining_values = expr_values[remaining_mask]
                remaining_count = max_genes - forced_ids.size
                if remaining_count < 0:
                    remaining_count = 0
                gene_ids = np.concatenate([forced_ids, remaining_ids[:remaining_count]])
                gene_indices = np.concatenate([forced_indices, remaining_indices[:remaining_count]])
                expr_values = np.concatenate([forced_values, remaining_values[:remaining_count]])
        elif gene_ids.size > max_genes > 0:
            gene_ids = gene_ids[:max_genes]
            gene_indices = gene_indices[:max_genes]
            expr_values = expr_values[:max_genes]

        if self.config.cls_token_id is not None:
            gene_ids = np.concatenate(([self.config.cls_token_id], gene_ids))
            gene_indices = np.concatenate(([-1], gene_indices))
            expr_values = np.concatenate(([0.0], expr_values))

        pad_len = self.config.max_genes - gene_ids.size
        if pad_len < 0:
            pad_len = 0

        if pad_len:
            gene_ids = np.pad(
                gene_ids,
                (0, pad_len),
                mode="constant",
                constant_values=self.config.pad_token_id,
            )
            gene_indices = np.pad(
                gene_indices,
                (0, pad_len),
                mode="constant",
                constant_values=-1,
            )
            expr_values = np.pad(expr_values, (0, pad_len), mode="constant", constant_values=0.0)

        src_key_padding_mask = gene_ids == self.config.pad_token_id

        item = {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "gene_values": torch.tensor(expr_values, dtype=torch.float32),
            "gene_indices": torch.tensor(gene_indices, dtype=torch.long),
            "src_key_padding_mask": torch.tensor(src_key_padding_mask, dtype=torch.bool),
        }
        if self.config.cache_items:
            self._item_cache[idx] = item
        return item


def collate_scgpt(batch: Iterable[Dict[str, torch.Tensor]]):
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
