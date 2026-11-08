"""Strict configuration loading for PCSM experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .hashing import canonical_json_hash


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    if value.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported schema_version in {path}")
    return value


def config_hash(config: dict[str, Any]) -> str:
    return canonical_json_hash(config)


def required_env_path(name: str, *, must_exist: bool = True) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"Required environment variable is unset: {name}")
    path = Path(raw).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_phase_c_config(config: dict[str, Any]) -> None:
    fractions = config.get("budget_fractions")
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("budget_fractions must be a non-empty list")
    parsed = [float(value) for value in fractions]
    if any(value <= 0.0 or value > 1.0 for value in parsed):
        raise ValueError("Every budget fraction must be in (0, 1]")
    if len(parsed) != len(set(parsed)):
        raise ValueError("budget_fractions must be unique")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 5:
        raise ValueError("Phase C requires at least five seeds")
    split = config.get("split")
    if not isinstance(split, dict) or not split.get(
        "forbid_sealed_read_during_selection"
    ):
        raise ValueError(
            "Phase C must fail closed against sealed-data selection access"
        )
