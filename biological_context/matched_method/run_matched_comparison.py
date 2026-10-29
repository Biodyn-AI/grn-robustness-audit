#!/usr/bin/env python3
"""A fixed 200-cell budget control using the historical method implementations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np

SEED = 20260218
METHODS = ("pearson", "mutual_info", "grnboost2", "genie3", "scgpt_attention")


def select_matched_cells(n_source):
    """Reconstruct WP1's parent sample and the active attention scorer's inner sample."""
    parent = np.random.default_rng(SEED).choice(n_source, size=800, replace=False)
    inner = np.random.default_rng(SEED).choice(800, size=200, replace=False)
    return dict(parent_indices=parent, inner_positions=inner, selected_indices=parent[inner])


def score_without_resampling(scorer, matched_expression, tf_indices, target_indices, gene_names):
    """Pass the exact same ordered, read-only matrix to each scorer."""
    if matched_expression.shape[0] != 200:
        raise ValueError("The matched budget must be exactly 200 cells.")
    return scorer.score(matched_expression, tf_indices, target_indices, gene_names)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    implementation = workspace / "subproject_merged_A_robustness_audit/implementation"
    sys.path.insert(0, str(implementation))
    import h5py
    import torch
    from fragility.axes.axis2_resolution.runner import Axis2DatasetSpec, load_h5ad, _read_obs_column
    from fragility.metrics import rss_with_weights, spearman_rank_stability, topk_jaccard
    from fragility.panels import load_panel
    from fragility.scorers import get_scorer
    from fragility.scorers.scgpt_minimal import load_scgpt_model

    # Refuse stale output mixing; the design file was written before this numerical run.
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    design_path = Path(__file__).resolve().parent / "design.json"
    design = json.loads(design_path.read_text())
    write_json(out_dir / "design.json", design)
    frozen_design_hash = sha256(design_path)
    source_path = workspace / design["dataset"]
    checkpoint = workspace / "single_cell_mechinterp/external/scGPT_checkpoints/whole-human"
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("The frozen run design requests MPS, which is unavailable.")
    started = time.perf_counter()
    # The process-level timeout is a further guard around model inference.
    def timed_out(signum, frame):
        raise TimeoutError("Frozen 600-second compute budget exceeded.")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(int(design["compute_limit_seconds"]))

    provenance_paths = [source_path, Path(__file__).resolve(), design_path,
                        implementation / "scripts/wp1_kidney_full.py",
                        implementation / "fragility/axes/axis2_resolution/runner.py",
                        implementation / "fragility/scorers/base.py",
                        implementation / "fragility/scorers/__init__.py",
                        implementation / "fragility/scorers/pearson.py",
                        implementation / "fragility/scorers/mutual_information.py",
                        implementation / "fragility/scorers/tree.py",
                        implementation / "fragility/scorers/scgpt_scorer.py",
                        implementation / "fragility/scorers/scgpt_minimal.py",
                        checkpoint / "args.json", checkpoint / "vocab.json", checkpoint / "best_model.pt"]
    provenance = {str(path.relative_to(workspace)): sha256(path) for path in provenance_paths}
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "h5py")}
    run_record = dict(design_sha256=frozen_design_hash, source_files_sha256=provenance,
                      python=sys.version, platform=platform.platform(), package_versions=versions,
                      device="mps", threads=2, preprocessing="unchanged stored /X; no new transformation",
                      torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      environment={name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    write_json(out_dir / "provenance.json", run_record)
    spec = Axis2DatasetSpec("kidney", source_path, "broad_cell_class", "scvi_leiden_res05_tissue")
    dataset = load_h5ad(spec)
    gene_names = [str(gene).upper() for gene in dataset.gene_ids]
    panel = load_panel("hematopoiesis_76x108", gene_universe=gene_names)
    tf_names = sorted(panel["source"].unique())
    target_names = sorted(panel["target"].unique())
    tf_indices = np.array([gene_names.index(gene) for gene in tf_names])
    target_indices = np.array([gene_names.index(gene) for gene in target_names])
    selection = select_matched_cells(dataset.x.shape[0])
    selected = selection["selected_indices"]
    matched_expression = dataset.x[selected].toarray().astype(np.float32)
    matched_expression.setflags(write=False)
    expected_expression_hash = array_hash(matched_expression)
    with h5py.File(source_path, "r") as handle:
        index_key = handle["obs"].attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        cell_ids = _read_obs_column(handle, index_key)
    with (out_dir / "selected_cells.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("matched_position", "source_row", "parent_position", "cell_id"))
        writer.writerows((i, int(source_row), int(selection["inner_positions"][i]), str(cell_ids[source_row]))
                         for i, source_row in enumerate(selected))
    np.savez_compressed(out_dir / "matched_input.npz", expression=matched_expression,
                        source_row=selected, gene_names=np.asarray(gene_names, dtype=str),
                        cell_ids=np.asarray(cell_ids[selected], dtype=str),
                        tf_indices=tf_indices, target_indices=target_indices,
                        parent_indices=selection["parent_indices"], inner_positions=selection["inner_positions"])
    run_record.update(source_shape=list(dataset.x.shape), matched_shape=list(matched_expression.shape),
                      input_expression_sha256=expected_expression_hash,
                      source_indices_sha256=array_hash(selected), n_tfs=len(tf_names),
                      n_targets=len(target_names), n_edges=len(tf_names) * len(target_names),
                      matched_input_sha256=sha256(out_dir / "matched_input.npz"),
                      selected_cells_sha256=sha256(out_dir / "selected_cells.csv"))
    write_json(out_dir / "provenance.json", run_record)
    print(f"Fixed input: {matched_expression.shape}; {len(tf_names)} TFs x {len(target_names)} targets", flush=True)
    outputs, timings = {}, []
    try:
        for method in METHODS:
            method_started = time.perf_counter()
            options = dict(design["methods"][method])
            label = options.pop("label")
            # The old tree code fixes these three options internally. Record them
            # in the design but avoid implying a different estimator implementation.
            if method == "grnboost2":
                for key in ("learning_rate", "max_depth", "subsample"):
                    options.pop(key)
            if method in ("grnboost2", "genie3", "scgpt_attention"):
                options["gene_names"] = gene_names
            scorer = get_scorer(method, **options)
            if method == "scgpt_attention":
                model, vocab, checkpoint_args, diagnostics = load_scgpt_model(checkpoint, device="mps")
                # Use the exact historical scorer with the loader's full diagnostics
                # retained. Loading it here avoids a second model load in _score.
                scorer._model, scorer._vocab, scorer._device = model, vocab, "mps"
                write_json(out_dir / "checkpoint_load.json", dict(args=checkpoint_args, diagnostics=diagnostics))
                run_record["scgpt_in_vocab_panel_genes"] = sum(gene in vocab for gene in set(tf_names + target_names))
                run_record["scgpt_panel_genes"] = len(set(tf_names + target_names))
            print(f"Starting {method}: {type(scorer).__name__}", flush=True)
            outputs[method] = score_without_resampling(scorer, matched_expression, tf_indices, target_indices, gene_names)
            if method == "scgpt_attention":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - method_started
            if array_hash(matched_expression) != expected_expression_hash:
                raise RuntimeError(f"Input matrix mutated by {method}.")
            scores = outputs[method]
            if not np.isfinite(scores.scores).all() or len(scores.scores) != len(tf_names) * len(target_names):
                raise ValueError(f"Invalid score vector: {method}")
            with (out_dir / f"scores_{method}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("tf", "target", "score", "rank", "sign"))
                writer.writerows((tf, target, float(score), int(rank),
                                  "" if scores.signs is None else int(scores.signs[i]))
                                 for i, (tf, target, score, rank) in enumerate(zip(scores.tf_names, scores.target_names, scores.scores, scores.ranks)))
            timings.append(dict(method=method, label=label, implementation=type(scorer).__name__,
                                seconds=elapsed, n_cells=200, input_sha256=expected_expression_hash))
            write_json(out_dir / "timings.json", timings)
            print(f"Completed {method}: {elapsed:.3f} seconds", flush=True)
        metrics = []
        for first, second in itertools.combinations(METHODS, 2):
            first_scores, second_scores = outputs[first].scores, outputs[second].scores
            rss = rss_with_weights(first_scores, second_scores, k=500)
            metrics.append(dict(scorer_a=first, scorer_b=second,
                                spearman=spearman_rank_stability(first_scores, second_scores),
                                top100_jaccard=topk_jaccard(first_scores, second_scores, k=100),
                                top500_jaccard=topk_jaccard(first_scores, second_scores, k=500),
                                top500_rss=rss.composite, top500_overlap=1 - rss.overlap_loss,
                                top500_drift_norm=rss.drift_norm))
        with (out_dir / "method_agreement.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        run_record["status"] = "complete"
        run_record["total_seconds"] = time.perf_counter() - started
        write_json(out_dir / "provenance.json", run_record)
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException as error:
        run_record.update(status="failed", error=f"{type(error).__name__}: {error}",
                          total_seconds=time.perf_counter() - started)
        write_json(out_dir / "provenance.json", run_record)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
