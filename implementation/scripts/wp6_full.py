import sys, site
sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, ".")
from pathlib import Path
from fragility.axes.axis2_resolution.runner import Axis2DatasetSpec
from fragility.axes.wp6_integration.runner import run, WP6Config

specs = [
    Axis2DatasetSpec(name="kidney",
        path=Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/kidney/processed.h5ad"),
        coarse_key="broad_cell_class", fine_key="scvi_leiden_res05_tissue"),
    Axis2DatasetSpec(name="lung",
        path=Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad"),
        coarse_key="broad_cell_class", fine_key="scvi_leiden_res05_tissue"),
    Axis2DatasetSpec(name="immune",
        path=Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad"),
        coarse_key="broad_cell_class", fine_key="scvi_leiden_res05_compartment"),
]
paths = run(specs, Path("outputs/wp6_full"), config=WP6Config(n_cells=1200, n_pcs=30))
import pandas as pd
df = pd.read_csv(paths["integration_cross_method"])
print(df[["dataset","method","status","n_batches","spearman","overlap_500","overlap_1000","rss_500","sign_flip_rate"]].to_string(index=False))
