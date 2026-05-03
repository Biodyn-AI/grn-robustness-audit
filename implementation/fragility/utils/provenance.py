"""Provenance block written alongside every CSV output.

Each pipeline run emits one ``provenance.json`` alongside its primary CSV.
The block captures enough information to rerun the experiment without
reading any other file: git commit, package versions, seed, config hash,
and a pointer to the input datasets by absolute path + file hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]


@dataclass
class ProvenanceBlock:
    """All fields that identify a run."""

    pipeline: str
    config: Dict[str, Any]
    config_hash: str
    base_seed: int
    derived_seeds: Dict[str, int]
    input_files: Sequence[Dict[str, str]] = field(default_factory=list)
    git_commit: Optional[str] = None
    package_versions: Dict[str, str] = field(default_factory=dict)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    created_utc: Optional[str] = None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _package_versions(packages: Sequence[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in packages:
        try:
            mod = __import__(name)
            v = getattr(mod, "__version__", None)
            if v is None:
                try:
                    from importlib.metadata import version as _v

                    v = _v(name)
                except Exception:
                    v = "unknown"
            versions[name] = str(v)
        except Exception:
            versions[name] = "not-installed"
    return versions


def write_provenance(
    out_dir: PathLike,
    pipeline: str,
    config: Mapping,
    base_seed: int,
    derived_seeds: Mapping[str, int],
    input_files: Sequence[PathLike] = (),
    packages: Sequence[str] = (
        "fragility",
        "numpy",
        "pandas",
        "scipy",
        "scanpy",
        "anndata",
        "sklearn",
    ),
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Serialize a ``ProvenanceBlock`` to ``<out_dir>/provenance.json``."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_bytes = json.dumps(dict(config), sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()[:16]

    file_blocks = []
    for p in input_files:
        p = Path(p)
        if p.exists():
            file_blocks.append(
                {
                    "path": str(p),
                    "size_bytes": str(p.stat().st_size),
                    "sha256_16": _file_hash(p),
                }
            )
        else:
            file_blocks.append({"path": str(p), "missing": "true"})

    from datetime import datetime, timezone

    block = ProvenanceBlock(
        pipeline=pipeline,
        config=dict(config),
        config_hash=config_hash,
        base_seed=int(base_seed),
        derived_seeds={k: int(v) for k, v in derived_seeds.items()},
        input_files=file_blocks,
        git_commit=_git_commit(),
        package_versions=_package_versions(packages),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )

    payload = asdict(block)
    if extra:
        payload["extra"] = dict(extra)

    out_path = out_dir / "provenance.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path
