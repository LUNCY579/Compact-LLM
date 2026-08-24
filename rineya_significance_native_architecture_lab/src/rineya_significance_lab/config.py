"""Configuration loading, validation, and reproducible hashing.

Configuration paths are resolved relative to the file that declares them.  The
loader deliberately supports only a small, auditable YAML/JSON surface: one
optional ``extends`` parent, environment substitutions, and explicit dotted-key
overrides.  Resolved configurations are plain dictionaries so every run can
serialize exactly what it used.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """Raised when a configuration is malformed or scientifically ambiguous."""


def _load_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised without the YAML extra
            raise ConfigError("PyYAML is required to read YAML configuration files") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def _merge(base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in child.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if fallback is not None:
                return fallback
            raise ConfigError(f"required environment variable is unset: {name}")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _parse_override(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any] | None) -> None:
    """Apply dotted-key overrides in place.

    Unknown intermediate mappings are rejected; a typo must not create a new
    experimental condition silently.  The leaf may be new to allow explicit
    additions to an otherwise open metadata mapping.
    """

    for dotted, raw_value in (overrides or {}).items():
        keys = dotted.split(".")
        if not keys or any(not key for key in keys):
            raise ConfigError(f"invalid override key: {dotted!r}")
        cursor: dict[str, Any] = config
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                raise ConfigError(f"override path does not name a mapping: {dotted}")
            cursor = child
        value = _parse_override(raw_value) if isinstance(raw_value, str) else raw_value
        cursor[keys[-1]] = value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate invariants that otherwise create plausible but invalid runs."""

    protocol = config.get("protocol", {})
    if protocol.get("execution_mode") not in {
        "open_loop_trace_simulation",
        "closed_loop_model_execution",
    }:
        raise ConfigError("protocol.execution_mode must name open-loop or closed-loop execution")

    significance = config.get("significance", {})
    exponent = float(significance.get("exponent", 1.0))
    if not 0.0 <= exponent <= 1.0:
        raise ConfigError("significance.exponent must be in [0, 1]")
    pooling = significance.get("score_pooling", {})
    width = int(pooling.get("width", 1))
    if width < 1 or width % 2 == 0:
        raise ConfigError("score_pooling.width must be a positive odd integer")

    cache = config.get("cache", {})
    budget = int(cache.get("total_budget_slots", 0))
    sinks = int(cache.get("sink_tokens", 0))
    recent = int(cache.get("recent_tokens", 0))
    if budget < 0 or sinks < 0 or recent < 0:
        raise ConfigError("cache budgets and protection counts must be non-negative")
    if significance.get("enabled", False) and budget <= sinks + recent:
        raise ConfigError(
            "significance selection has zero heavy slots: total_budget_slots must exceed "
            "sink_tokens + recent_tokens"
        )

    query_mode = config.get("query", {}).get("mode", "agnostic")
    if query_mode not in {"agnostic", "conditioned"}:
        raise ConfigError("query.mode must be 'agnostic' or 'conditioned'")
    if query_mode == "conditioned" and not significance.get("enabled", False):
        raise ConfigError("query-conditioned scoring requires significance.enabled=true")

    attention_prior = config.get("attention_prior", {})
    if (
        attention_prior.get("enabled", False)
        and protocol.get("execution_mode") != "closed_loop_model_execution"
    ):
        raise ConfigError("attention_prior changes future hidden state and requires closed-loop execution")
    prior_mode = attention_prior.get("mode", "normalized")
    if prior_mode not in {"normalized", "raw", "shuffled", "age"}:
        raise ConfigError("attention_prior.mode must be normalized, raw, shuffled, or age")
    clip = float(attention_prior.get("clip", 2.0))
    epsilon = float(attention_prior.get("epsilon", 1.0e-6))
    max_abs_lambda = float(attention_prior.get("max_abs_lambda", 0.25))
    if not all(math.isfinite(value) and value > 0 for value in (clip, epsilon, max_abs_lambda)):
        raise ConfigError("attention_prior clip, epsilon, and max_abs_lambda must be finite and positive")
    global_lambda = float(attention_prior.get("lambda", 0.0))
    if not math.isfinite(global_lambda) or abs(global_lambda) > max_abs_lambda:
        raise ConfigError("attention_prior.lambda exceeds its finite max_abs_lambda bound")
    per_layer = attention_prior.get("per_layer_lambdas")
    if per_layer is not None:
        if isinstance(per_layer, list):
            layer_values = per_layer
        elif isinstance(per_layer, Mapping):
            try:
                if any(int(index) < 0 for index in per_layer):
                    raise ConfigError("attention_prior per-layer indices must be non-negative")
            except (TypeError, ValueError) as exc:
                raise ConfigError("attention_prior per-layer indices must be integers") from exc
            layer_values = list(per_layer.values())
        else:
            raise ConfigError("attention_prior.per_layer_lambdas must be a list or mapping")
        try:
            invalid_layer_lambda = any(
                not math.isfinite(float(value)) or abs(float(value)) > max_abs_lambda
                for value in layer_values
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError("attention_prior per-layer lambdas must be numeric") from exc
        if invalid_layer_lambda:
            raise ConfigError("attention_prior per-layer lambda exceeds max_abs_lambda")
    if attention_prior.get("enabled", False):
        access = attention_prior.get("access")
        if not isinstance(access, Mapping):
            raise ConfigError("enabled attention_prior requires an explicit access declaration")
        if access.get("query") != query_mode:
            raise ConfigError("attention_prior.access.query must match query.mode")
        prohibited = {
            "answer": access.get("answer"),
            "oracle_evidence": access.get("oracle_evidence"),
            "future_tokens": access.get("future_tokens"),
            "future_hidden_states": access.get("future_hidden_states"),
        }
        if any(value != "none" for value in prohibited.values()):
            raise ConfigError(
                "attention_prior access must explicitly set answer, oracle_evidence, future_tokens, "
                "and future_hidden_states to 'none'"
            )

    for feature in ("attention_prior", "memory_slots", "controller"):
        section = config.get(feature, {})
        if section.get("enabled", False) and protocol.get("execution_mode") != "closed_loop_model_execution":
            raise ConfigError(f"{feature} changes future hidden state and requires closed-loop execution")


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and fully resolve a YAML or JSON experiment configuration."""

    source = Path(path).expanduser().resolve()
    document = _load_document(source)
    parent_value = document.pop("extends", None)
    if parent_value is not None:
        parent = Path(str(parent_value))
        if not parent.is_absolute():
            parent = source.parent / parent
        parent_config = load_config(parent)
        document = _merge(parent_config, document)
    resolved_value = _expand_environment(document)
    if not isinstance(resolved_value, dict):  # defensive: document roots are mappings
        raise ConfigError("resolved configuration root must remain a mapping")
    resolved: dict[str, Any] = resolved_value
    apply_overrides(resolved, overrides)
    resolved.setdefault("_meta", {})
    resolved["_meta"]["source_config"] = str(source)
    validate_config(resolved)
    resolved["_meta"]["config_sha256"] = config_hash(resolved, omit_self_hash=True)
    return resolved


def config_hash(config: Mapping[str, Any], *, omit_self_hash: bool = False) -> str:
    value = copy.deepcopy(dict(config))
    if omit_self_hash and isinstance(value.get("_meta"), dict):
        value["_meta"].pop("config_sha256", None)
        # The declaring file is provenance, not experimental semantics.  An
        # absolute Windows path must not make the same resolved experiment hash
        # differently on another machine.
        value["_meta"].pop("source_config", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
