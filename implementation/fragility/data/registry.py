"""Central dataset registry.

Each dataset is declared once, with its on-disk h5ad path, cell-type
annotation column, batch key candidates, and any dataset-specific notes.
All axis/WP runners look up datasets through this registry rather than
hard-coding paths in individual scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSpec:
    """On-disk dataset + canonical metadata columns."""

    name: str
    path: Path
    cell_type_key: str
    donor_key: Optional[str] = None
    batch_key: Optional[str] = None
    method_key: Optional[str] = None
    tissue: Optional[str] = None
    source: Optional[str] = None
    note: str = ""

    def exists(self) -> bool:
        return self.path.exists()


_RAW_ROOT = Path(
    "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"
)
_PROCESSED_ROOT = Path(
    "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs"
)
_INVARIANT_ROOT = _PROCESSED_ROOT / "invariant_causal_edges"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Cell-type / donor / batch keys chosen to match the columns used by the
# per-axis subproject scripts. Missing or unknown keys are None and must be
# resolved at load time with a runtime error.

_DATASETS: Dict[str, DatasetSpec] = {
    "kidney": DatasetSpec(
        name="kidney",
        path=_RAW_ROOT / "tabula_sapiens_kidney.h5ad",
        cell_type_key="cell_ontology_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="kidney",
        source="Tabula Sapiens (2022)",
    ),
    "lung": DatasetSpec(
        name="lung",
        path=_RAW_ROOT / "tabula_sapiens_lung.h5ad",
        cell_type_key="cell_ontology_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="lung",
        source="Tabula Sapiens (2022)",
    ),
    "immune": DatasetSpec(
        name="immune",
        path=_RAW_ROOT / "tabula_sapiens_immune.h5ad",
        cell_type_key="cell_ontology_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="immune",
        source="Tabula Sapiens (2022)",
    ),
    "immune_subset_20k": DatasetSpec(
        name="immune_subset_20k",
        path=_RAW_ROOT / "tabula_sapiens_immune_subset_20000.h5ad",
        cell_type_key="cell_ontology_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="immune",
        source="Tabula Sapiens (2022) -- 20k cell subset used in Axes 3+4",
    ),
    "external_lung": DatasetSpec(
        name="external_lung",
        path=_RAW_ROOT / "krasnow_lung_smartsq2.h5ad",
        cell_type_key="free_annotation",
        donor_key=None,
        batch_key=None,
        method_key=None,
        tissue="lung",
        source="Travaglini et al. 2020 (Krasnow SmartSeq2)",
        note=(
            "Used as external cohort in Axes 2/3; no per-donor column in the "
            "h5ad as shipped."
        ),
    ),
    # Processed (pre-clustered / pre-integrated) variants used by Axis 2 and
    # parts of Axis 5. These come from the single_cell_mechinterp outputs and
    # carry pre-computed coarse / fine cluster labels.
    "kidney_processed": DatasetSpec(
        name="kidney_processed",
        path=_INVARIANT_ROOT / "kidney" / "processed.h5ad",
        cell_type_key="broad_cell_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="kidney",
        source="Tabula Sapiens, Axis-2 processed h5ad",
        note="carries scvi_leiden_res05_tissue fine labels",
    ),
    "lung_processed": DatasetSpec(
        name="lung_processed",
        path=_INVARIANT_ROOT / "lung" / "processed.h5ad",
        cell_type_key="broad_cell_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="lung",
        source="Tabula Sapiens, Axis-2 processed h5ad",
        note="carries scvi_leiden_res05_tissue fine labels",
    ),
    "immune_invariant": DatasetSpec(
        name="immune_invariant",
        path=_INVARIANT_ROOT / "immune" / "processed.h5ad",
        cell_type_key="broad_cell_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="immune",
        source="Tabula Sapiens immune invariant-causal-edges processed h5ad",
    ),
    "immune_hpn": DatasetSpec(
        name="immune_hpn",
        path=_PROCESSED_ROOT / "tabula_sapiens_immune_subset_hpn_processed.h5ad",
        cell_type_key="broad_cell_class",
        donor_key="donor",
        batch_key="_batch",
        method_key="method",
        tissue="immune",
        source="Tabula Sapiens immune HPN processed h5ad (Axis 2/3 primary)",
        note="carries scvi_leiden_res05_compartment fine labels",
    ),
    "external_lung_processed": DatasetSpec(
        name="external_lung_processed",
        path=_INVARIANT_ROOT / "external_lung" / "processed.h5ad",
        cell_type_key="compartment",
        donor_key=None,
        batch_key=None,
        method_key=None,
        tissue="lung",
        source="Krasnow SmartSeq2 (external) processed h5ad",
        note="fine_key = cell_type in this dataset",
    ),
    "delorey_covid_lung": DatasetSpec(
        name="delorey_covid_lung",
        path=Path(
            "/Users/ihorkendiukhov/biodyn-work/"
            "subproject_merged_A_robustness_audit/implementation/"
            "data_external/delorey_covid_lung.h5ad"
        ),
        cell_type_key="cell_type_main",
        donor_key="donor_id",
        batch_key=None,
        method_key="assay",
        tissue="lung",
        source=(
            "Delorey et al. 2021, \"A molecular single-cell lung atlas of "
            "lethal COVID-19\". CellxGene dataset 75c059c8. 116,313 cells, "
            "27 donors (COVID + normal). Downloaded 2026-04-24."
        ),
        note="Axis-2 coarse=cell_type_main, fine=cell_type (ontology)",
    ),
}


# Public read-only handle for enumeration.
DATASETS: Dict[str, DatasetSpec] = dict(_DATASETS)


def resolve(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name`` or raise ``KeyError``."""

    if name not in _DATASETS:
        raise KeyError(
            f"unknown dataset '{name}'. Known: {sorted(_DATASETS)}"
        )
    spec = _DATASETS[name]
    if not spec.exists():
        raise FileNotFoundError(
            f"dataset '{name}' expected at {spec.path} but file is missing"
        )
    return spec


def list_datasets() -> List[str]:
    return sorted(_DATASETS)
