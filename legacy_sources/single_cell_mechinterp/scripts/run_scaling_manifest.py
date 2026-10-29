from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _write_config(config: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _append_log(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _tail_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _parse_metrics(stdout_text: str) -> str:
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                metrics = ast.literal_eval(line)
            except Exception:
                continue
            if isinstance(metrics, dict):
                return json.dumps(metrics, sort_keys=True)
    return ""


def _build_config(base_cfg: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))

    project_cfg = cfg.setdefault("project", {})
    project_cfg["seed"] = int(row["seed"])

    run_cfg = cfg.setdefault("run", {})
    run_cfg["experiment_id"] = row["experiment_id"]
    if not run_cfg.get("description"):
        run_cfg["description"] = "scaling run"

    paths = cfg["paths"]
    paths["scgpt_checkpoint"] = row["scgpt_checkpoint"]
    paths["scgpt_vocab"] = row["scgpt_vocab"]
    paths["tabula_sapiens_h5ad"] = row["tabula_sapiens_h5ad"]
    paths["processed_h5ad"] = row["processed_h5ad"]
    paths["attention_scores"] = row["attention_scores"]
    paths["attention_counts"] = row["attention_counts"]
    paths["network_edges"] = row["network_edges"]

    cfg["preprocess"]["hvg"] = int(row["preprocess_hvg"])
    cfg["scgpt_dataset"]["max_genes"] = int(row["max_genes"])

    attention_cfg = cfg.setdefault("attention", {})
    attention_cfg["max_cells"] = _parse_optional_int(row.get("max_cells"))
    attention_cfg["sample_strategy"] = row.get("sample_strategy", attention_cfg.get("sample_strategy", "head"))
    attention_cfg["sample_seed"] = _parse_optional_int(row.get("sample_seed"))

    network_cfg = cfg.setdefault("network", {})
    network_cfg["candidate_sources_from_dorothea"] = _parse_bool(
        row.get("candidate_sources_from_dorothea")
    )
    network_cfg["candidate_targets_from_dorothea"] = _parse_bool(
        row.get("candidate_targets_from_dorothea")
    )
    network_cfg["top_k"] = _parse_optional_int(row.get("top_k"))
    network_cfg["threshold_percentile"] = _parse_optional_float(row.get("threshold_percentile"))

    return cfg


def _run_command(command: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
    )


def _build_step_commands(config_path: Path, batch_size: int, device: str | None, steps: Iterable[str]) -> List[List[str]]:
    commands = []
    for step in steps:
        if step == "prepare-data":
            commands.append(["python", "-m", "src.cli", "--config", str(config_path), "prepare-data"])
        elif step == "extract-attention":
            cmd = [
                "python",
                "-m",
                "src.cli",
                "--config",
                str(config_path),
                "extract-attention",
                "--batch-size",
                str(batch_size),
            ]
            if device:
                cmd.extend(["--device", device])
            commands.append(cmd)
        elif step == "infer-network":
            commands.append(["python", "-m", "src.cli", "--config", str(config_path), "infer-network"])
        elif step == "evaluate":
            commands.append(["python", "-m", "src.cli", "--config", str(config_path), "evaluate"])
        else:
            raise ValueError(f"Unknown step: {step}")
    return commands


def _step_from_command(command: List[str]) -> str:
    for step in ("prepare-data", "extract-attention", "infer-network", "evaluate"):
        if step in command:
            return step
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaling grid from a manifest CSV.")
    parser.add_argument("--manifest", default="configs/scaling_experiment_manifest.csv")
    parser.add_argument("--base-config", default="configs/scaling_base.yaml")
    parser.add_argument("--config-out-dir", default="configs/scaling_runs")
    parser.add_argument("--log-path", default="outputs/scaling/run_log.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["prepare-data", "extract-attention", "infer-network", "evaluate"],
    )
    parser.add_argument("--tissue", action="append")
    parser.add_argument("--model-tier", action="append")
    parser.add_argument("--experiment-id", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--mkl-threads", type=int, default=1)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_cfg_path = Path(args.base_config)
    config_out_dir = Path(args.config_out_dir)
    log_path = Path(args.log_path)

    rows = _load_manifest(manifest_path)
    if args.tissue:
        rows = [row for row in rows if row["tissue"] in set(args.tissue)]
    if args.model_tier:
        rows = [row for row in rows if row["model_tier"] in set(args.model_tier)]
    if args.experiment_id:
        rows = [row for row in rows if row["experiment_id"] in set(args.experiment_id)]
    if args.seed:
        rows = [row for row in rows if int(row["seed"]) in set(args.seed)]
    if args.limit is not None:
        rows = rows[: args.limit]

    base_cfg = _load_base_config(base_cfg_path)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    env["MKL_NUM_THREADS"] = str(args.mkl_threads)

    for row in rows:
        experiment_id = row["experiment_id"]
        config_path = config_out_dir / f"{experiment_id}.yaml"
        config = _build_config(base_cfg, row)
        _write_config(config, config_path)

        if args.dry_run:
            _append_log(
                log_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment_id": experiment_id,
                    "step": "dry_run",
                    "status": "skipped",
                    "return_code": "",
                    "elapsed_sec": "",
                    "metrics": "",
                    "stdout_tail": "",
                    "stderr_tail": "",
                },
            )
            continue

        if args.skip_existing:
            network_edges = Path(row["network_edges"])
            if network_edges.exists():
                _append_log(
                    log_path,
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_id": experiment_id,
                        "step": "skip",
                        "status": "skipped",
                        "return_code": "",
                        "elapsed_sec": "",
                        "metrics": "",
                        "stdout_tail": "",
                        "stderr_tail": "",
                    },
                )
                continue

        commands = _build_step_commands(config_path, args.batch_size, args.device, args.steps)
        for cmd in commands:
            step = _step_from_command(cmd)
            start_time = time.time()
            result = _run_command(cmd, env)
            elapsed = time.time() - start_time
            metrics = _parse_metrics(result.stdout) if step == "evaluate" else ""
            status = "success" if result.returncode == 0 else "failed"

            _append_log(
                log_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment_id": experiment_id,
                    "step": step,
                    "status": status,
                    "return_code": result.returncode,
                    "elapsed_sec": f"{elapsed:.2f}",
                    "metrics": metrics,
                    "stdout_tail": _tail_text(result.stdout),
                    "stderr_tail": _tail_text(result.stderr),
                },
            )
            if status != "success" and not args.continue_on_failure:
                return


if __name__ == "__main__":
    main()
