"""GRNBoost2 + GENIE3 on kidney with a threaded dask scheduler.

arboreto defaults to a process-pool distributed cluster that needs a
real script path; using a thread-based dask client avoids that and also
avoids the torch/numpy multiprocessing fork issues.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fragility.axes.axis2_resolution.runner import Axis2DatasetSpec, load_h5ad
from fragility.panels import load_panel


def _make_client():
    """Threaded dask client — single process, avoids the arboreto fork issue."""

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        processes=False,
        threads_per_worker=4,
        n_workers=1,
    )
    return Client(cluster)


def main() -> None:
    from arboreto.algo import grnboost2, genie3

    spec = Axis2DatasetSpec(
        name="kidney",
        path=Path(
            "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/outputs/"
            "invariant_causal_edges/kidney/processed.h5ad"
        ),
        coarse_key="broad_cell_class",
        fine_key="scvi_leiden_res05_tissue",
    )
    ds = load_h5ad(spec)
    gene_names = [str(g).upper() for g in ds.gene_ids]

    panel = load_panel("hematopoiesis_76x108", gene_universe=gene_names)
    tfs = sorted(panel["source"].unique())
    targets = sorted(panel["target"].unique())

    rng = np.random.default_rng(20260218)
    n_cells = 800
    cell_idx = rng.choice(ds.x.shape[0], size=n_cells, replace=False)
    X = ds.x[cell_idx].toarray().astype(np.float32)

    required = sorted(set(tfs) | set(targets))
    col_idx = np.array(
        [gene_names.index(g) for g in required if g in gene_names], dtype=int
    )
    used_names = [gene_names[i] for i in col_idx]
    expr_df = pd.DataFrame(X[:, col_idx], columns=used_names)
    print(f"expr_df shape: {expr_df.shape}", flush=True)

    client = _make_client()
    try:
        for fn_name, fn in (("grnboost2", grnboost2), ("genie3", genie3)):
            t0 = time.time()
            try:
                edges = fn(
                    expression_data=expr_df,
                    tf_names=tfs,
                    gene_names=used_names,
                    client_or_address=client,
                    verbose=False,
                    seed=20260218,
                )
                print(
                    f"{fn_name}: {time.time()-t0:.1f}s, {len(edges)} edges",
                    flush=True,
                )
                out_path = ROOT / "outputs" / "wp1" / f"kidney_{fn_name}_edges.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                edges.to_csv(out_path, index=False)
                print(f"  wrote {out_path}", flush=True)
            except Exception as e:
                print(f"{fn_name} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
