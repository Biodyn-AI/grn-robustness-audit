"""WP-1 full scorer comparison on kidney.

Includes Pearson + MI + GRNBoost2 + GENIE3 + scGPT attention on a shared
panel (76 x 108 hematopoiesis edges) and shared cell subsample.
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
from fragility.metrics import rss_with_weights, spearman_rank_stability, topk_jaccard
from fragility.panels import load_panel
from fragility.scorers import get_scorer


def main() -> None:
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
    tf_idx = np.array([gene_names.index(t) for t in tfs])
    target_idx = np.array([gene_names.index(t) for t in targets])
    print(f"{len(tfs)}x{len(targets)}={len(tfs)*len(targets)} edges", flush=True)

    rng = np.random.default_rng(20260218)
    n_cells = 800
    cell_idx = rng.choice(ds.x.shape[0], size=n_cells, replace=False)
    X = ds.x[cell_idx].toarray().astype(np.float32)

    results = {}
    for name in ["pearson", "mutual_info", "grnboost2", "genie3", "scgpt_attention"]:
        t0 = time.time()
        kwargs = {}
        if name in ("grnboost2", "genie3", "scgpt_attention"):
            kwargs["gene_names"] = gene_names
        if name == "scgpt_attention":
            kwargs["n_cells"] = 200
        if name == "mutual_info":
            X_use = X[rng.choice(X.shape[0], size=400, replace=False)]
        else:
            X_use = X
        try:
            scorer = get_scorer(name, **kwargs)
            out = scorer.score(X_use, tf_idx, target_idx, gene_names)
            results[name] = out
            print(
                f"{name}: {time.time() - t0:.1f}s, "
                f"range [{out.scores.min():.4f}, {out.scores.max():.4f}]",
                flush=True,
            )
        except Exception as e:
            print(f"{name} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

    names = list(results.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rows = []
    for a, b in pairs:
        rho = spearman_rank_stability(results[a].scores, results[b].scores)
        j100 = topk_jaccard(results[a].scores, results[b].scores, k=100)
        j500 = topk_jaccard(results[a].scores, results[b].scores, k=500)
        rss = rss_with_weights(results[a].scores, results[b].scores, k=500)
        rows.append(
            {
                "scorer_a": a,
                "scorer_b": b,
                "spearman": round(rho, 3),
                "top100_jaccard": round(j100, 3),
                "top500_jaccard": round(j500, 3),
                "top500_rss": round(rss.composite, 3),
                "top500_overlap": round(1 - rss.overlap_loss, 3),
                "top500_drift_norm": round(rss.drift_norm, 3),
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== scorer-pair agreement (kidney, n=800 cells, 76x108 panel) ===")
    print(df.to_string(index=False))
    out_path = ROOT / "outputs" / "wp1" / "scorer_full_kidney.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
