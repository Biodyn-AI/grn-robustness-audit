"""YAML config loading with deep-merge over a default.

Configs are plain dicts. Any axis/WP can define its own default via a
module-level ``DEFAULT_CONFIG`` dict; user-supplied YAML is deep-merged
over that default so partial overrides remain concise.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

Config = Dict[str, Any]
PathLike = Union[str, Path]


def _deep_merge(base: Mapping, override: Mapping) -> Config:
    out = dict(deepcopy(base))
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: PathLike, default: Mapping | None = None) -> Config:
    """Load a YAML config and merge it over ``default``.

    Raises if ``path`` does not exist.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r") as f:
        user = yaml.safe_load(f) or {}
    if default is None:
        return dict(user)
    return _deep_merge(default, user)


def dump_config(config: Mapping, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(dict(config), f, sort_keys=True)
