"""Dataset registry and loading helpers."""

from .registry import DATASETS, DatasetSpec, resolve, list_datasets
from .loader import load_anndata

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "resolve",
    "list_datasets",
    "load_anndata",
]
