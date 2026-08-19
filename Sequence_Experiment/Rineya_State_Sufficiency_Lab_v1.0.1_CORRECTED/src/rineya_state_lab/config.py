from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import deep_merge, deep_set, parse_scalar

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(f"Environment variable {name!r} is required by the configuration")

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)
    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(_expand_string(str(include)))
        if not include_path.is_absolute():
            include_path = config_path.parent / include_path
        merged = deep_merge(merged, load_config(include_path))
    merged = deep_merge(merged, raw)
    merged = _expand(merged)
    merged.setdefault("_meta", {})
    merged["_meta"].update({"config_path": str(config_path), "config_dir": str(config_path.parent)})
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must have the form key=value, got {override!r}")
        key, value = override.split("=", 1)
        deep_set(merged, key.strip(), parse_scalar(value.strip()))
    return merged


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing required configuration field: {dotted_key}")
        current = current[part]
    return current


def resolved_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()
