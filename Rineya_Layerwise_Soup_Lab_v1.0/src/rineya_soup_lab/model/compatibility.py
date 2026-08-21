from __future__ import annotations

import json
import math

import torch
from pathlib import Path
from typing import Any, Mapping

from ..errors import CompatibilityError
from ..utils import sha256_file, write_json
from .checkpoint import CheckpointReader, checkpoint_fingerprint
from .layer_map import ParameterClassifier


CRITICAL_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "tie_word_embeddings",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "sliding_window",
    "use_sliding_window",
    "attention_bias",
    "mlp_bias",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "attention_dropout",
    "partial_rotary_factor",
    "rope_parameters",
    "qkv_bias",
    "use_qk_norm",
    "layer_types",
    "sliding_window_pattern",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "chat_template.jinja",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)

TIED_LM_HEAD_KEY = "lm_head.weight"
TIED_EMBED_KEY = "model.embed_tokens.weight"
DEFAULT_ROPE_TYPE = "default"
DEFAULT_LAYER_TYPE = "full_attention"


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _effective_rope_theta(config: Mapping[str, Any]) -> Any:
    top = config.get("rope_theta")
    if top is not None:
        as_float = _as_float(top)
        return as_float if as_float is not None else top
    params = config.get("rope_parameters")
    if isinstance(params, Mapping) and params.get("rope_theta") is not None:
        value = params.get("rope_theta")
        as_float = _as_float(value)
        return as_float if as_float is not None else value
    return None


def _normalized_rope_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    theta = _effective_rope_theta(config)
    params = config.get("rope_parameters")
    if not isinstance(params, Mapping) or not params:
        return {"rope_theta": theta, "rope_type": DEFAULT_ROPE_TYPE}
    normalized = {str(key): value for key, value in params.items()}
    normalized.setdefault("rope_theta", theta)
    normalized.setdefault("rope_type", DEFAULT_ROPE_TYPE)
    as_float = _as_float(normalized.get("rope_theta"))
    if as_float is not None:
        normalized["rope_theta"] = as_float
    return normalized


def _effective_layer_types(config: Mapping[str, Any]) -> list[str]:
    types = config.get("layer_types")
    layer_count = int(config.get("num_hidden_layers") or 0)
    if types is None:
        return [DEFAULT_LAYER_TYPE] * layer_count
    return [str(item) for item in types]


def config_field_equivalent(field: str, anchor: Mapping[str, Any], greedy: Mapping[str, Any]) -> bool:
    """Treat Hugging Face save_pretrained rewrites as the same architecture."""
    if field == "rope_theta":
        return _effective_rope_theta(anchor) == _effective_rope_theta(greedy)
    if field == "rope_parameters":
        return _normalized_rope_parameters(anchor) == _normalized_rope_parameters(greedy)
    if field == "layer_types":
        return _effective_layer_types(anchor) == _effective_layer_types(greedy)
    return False


def can_alias_tied_lm_head(
    anchor_keys: set[str],
    greedy_keys: set[str],
    anchor_config: Mapping[str, Any],
    greedy_config: Mapping[str, Any],
) -> bool:
    """True when the only key-set gap is a tied lm_head omitted by one serializer."""
    if not bool(anchor_config.get("tie_word_embeddings")):
        return False
    if not bool(greedy_config.get("tie_word_embeddings")):
        return False
    if TIED_EMBED_KEY not in anchor_keys or TIED_EMBED_KEY not in greedy_keys:
        return False
    extra = (anchor_keys - greedy_keys) | (greedy_keys - anchor_keys)
    return extra == {TIED_LM_HEAD_KEY}


def tokenizer_fingerprint(path: Path) -> dict[str, Any]:
    entries = []
    for name in TOKENIZER_FILES:
        file_path = path / name
        if file_path.exists() and file_path.is_file():
            entries.append({"name": name, "bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)})
    return {"path": str(path), "files": entries}


def compare_checkpoints(
    anchor_path: Path,
    greedy_path: Path,
    *,
    layer_key_regex: str,
    expected_layers: int | None,
    allow_quantized_sources: bool,
    require_non_floating_equal: bool,
    lineage_acknowledged: bool,
    tokenizer_path: Path | None = None,
    allow_hf_serialization_drift: bool = False,
) -> dict[str, Any]:
    classifier = ParameterClassifier(layer_key_regex)
    errors: list[str] = []
    warnings: list[str] = []
    with CheckpointReader(anchor_path) as anchor, CheckpointReader(greedy_path) as greedy:
        if not allow_quantized_sources:
            if anchor.has_quantization_markers():
                errors.append("Anchor checkpoint contains quantization markers; merge a dequantized checkpoint.")
            if greedy.has_quantization_markers():
                errors.append("Greedy checkpoint contains quantization markers; merge a dequantized checkpoint.")
        anchor_config = anchor.config()
        greedy_config = greedy.config()
        config_differences: dict[str, Any] = {}
        equivalent_config_fields: list[str] = []
        for field in CRITICAL_CONFIG_FIELDS:
            if anchor_config.get(field) == greedy_config.get(field):
                continue
            entry: dict[str, Any] = {
                "anchor": anchor_config.get(field),
                "greedy": greedy_config.get(field),
            }
            if allow_hf_serialization_drift and config_field_equivalent(
                field, anchor_config, greedy_config
            ):
                entry["equivalent"] = True
                equivalent_config_fields.append(field)
            config_differences[field] = entry
        blocking_config = {
            field: value
            for field, value in config_differences.items()
            if not value.get("equivalent")
        }
        if blocking_config:
            errors.append(f"Critical config fields differ: {sorted(blocking_config)}")
        elif equivalent_config_fields:
            warnings.append(
                "Hugging Face serialization-equivalent config fields differ: "
                f"{sorted(equivalent_config_fields)}"
            )

        anchor_keys = set(anchor.keys)
        greedy_keys = set(greedy.keys)
        missing_in_anchor = sorted(greedy_keys - anchor_keys)
        missing_in_greedy = sorted(anchor_keys - greedy_keys)
        aliased_tied_lm_head = bool(allow_hf_serialization_drift) and can_alias_tied_lm_head(
            anchor_keys, greedy_keys, anchor_config, greedy_config
        )
        if aliased_tied_lm_head:
            missing_in_anchor = [key for key in missing_in_anchor if key != TIED_LM_HEAD_KEY]
            missing_in_greedy = [key for key in missing_in_greedy if key != TIED_LM_HEAD_KEY]
            warnings.append(
                "Tied lm_head.weight is present on only one checkpoint; "
                f"aliasing {TIED_EMBED_KEY} for merge."
            )
        if missing_in_anchor or missing_in_greedy:
            errors.append(
                f"Tensor key sets differ: missing_in_anchor={len(missing_in_anchor)}, "
                f"missing_in_greedy={len(missing_in_greedy)}"
            )

        shape_mismatches: list[dict[str, Any]] = []
        dtype_mismatches: list[dict[str, Any]] = []
        non_floating_mismatches: list[str] = []
        floating_kind_mismatches: list[dict[str, Any]] = []
        non_finite_tensors: list[dict[str, Any]] = []
        parameter_count = 0
        weight_bytes = 0
        delta_accumulators: dict[str, dict[str, float]] = {}
        comparable_greedy_keys = set(greedy_keys)
        if aliased_tied_lm_head and TIED_LM_HEAD_KEY in anchor_keys:
            comparable_greedy_keys.add(TIED_LM_HEAD_KEY)
        common_keys = sorted(anchor_keys & comparable_greedy_keys)
        for key in common_keys:
            a = anchor.get_tensor(key)
            if key in greedy_keys:
                g = greedy.get_tensor(key)
            else:
                g = greedy.get_tensor(TIED_EMBED_KEY)
            if tuple(a.shape) != tuple(g.shape):
                shape_mismatches.append({"key": key, "anchor": list(a.shape), "greedy": list(g.shape)})
                continue
            parameter_count += int(a.numel())
            weight_bytes += int(a.numel() * a.element_size())
            if a.dtype != g.dtype:
                dtype_mismatches.append({"key": key, "anchor": str(a.dtype), "greedy": str(g.dtype)})
            if a.is_floating_point() != g.is_floating_point():
                floating_kind_mismatches.append(
                    {"key": key, "anchor": str(a.dtype), "greedy": str(g.dtype)}
                )
                continue
            if a.is_floating_point() and g.is_floating_point():
                location = classifier.classify(key)
                group_key = (
                    f"layer_{location.layer_index:03d}"
                    if location.layer_index is not None
                    else location.group
                )
                flat_a = a.reshape(-1)
                flat_g = g.reshape(-1)
                chunk = 1_048_576
                local = {
                    "anchor_sq": 0.0,
                    "greedy_sq": 0.0,
                    "delta_sq": 0.0,
                    "anchor_greedy_dot": 0.0,
                }
                anchor_non_finite = 0
                greedy_non_finite = 0
                for start in range(0, flat_a.numel(), chunk):
                    av = flat_a[start : start + chunk].float()
                    gv = flat_g[start : start + chunk].float()
                    finite_a = torch.isfinite(av)
                    finite_g = torch.isfinite(gv)
                    anchor_non_finite += int((~finite_a).sum().item())
                    greedy_non_finite += int((~finite_g).sum().item())
                    if not bool(finite_a.all()) or not bool(finite_g.all()):
                        continue
                    dv = gv - av
                    local["anchor_sq"] += float(torch.sum(av * av).item())
                    local["greedy_sq"] += float(torch.sum(gv * gv).item())
                    local["delta_sq"] += float(torch.sum(dv * dv).item())
                    local["anchor_greedy_dot"] += float(torch.sum(av * gv).item())
                if anchor_non_finite or greedy_non_finite:
                    non_finite_tensors.append(
                        {
                            "key": key,
                            "anchor_non_finite": anchor_non_finite,
                            "greedy_non_finite": greedy_non_finite,
                        }
                    )
                else:
                    accumulator = delta_accumulators.setdefault(
                        group_key,
                        {
                            "parameter_count": 0.0,
                            "anchor_sq": 0.0,
                            "greedy_sq": 0.0,
                            "delta_sq": 0.0,
                            "anchor_greedy_dot": 0.0,
                        },
                    )
                    for statistic, value in local.items():
                        accumulator[statistic] += value
                    accumulator["parameter_count"] += float(a.numel())
            if not a.is_floating_point() and require_non_floating_equal and not a.equal(g):
                non_floating_mismatches.append(key)
        if shape_mismatches:
            errors.append(f"Found {len(shape_mismatches)} tensor shape mismatches.")
        if floating_kind_mismatches:
            errors.append(
                f"Found {len(floating_kind_mismatches)} tensors where one endpoint is "
                "floating-point and the other is not."
            )
        if non_floating_mismatches:
            errors.append(f"Found {len(non_floating_mismatches)} unequal non-floating tensors.")
        if non_finite_tensors:
            errors.append(f"Found {len(non_finite_tensors)} tensors containing NaN or infinity.")
        if dtype_mismatches:
            warnings.append(f"Found {len(dtype_mismatches)} dtype mismatches; FP32 accumulation will be used.")

        layers = classifier.layer_indices(common_keys)
        if expected_layers is not None:
            expected = list(range(int(expected_layers)))
            if layers != expected:
                errors.append(f"Layer indices are {layers}, expected exactly {expected}.")
        if not lineage_acknowledged:
            warnings.append(
                "Shared initialization/lineage cannot be proven from tensor shapes alone. "
                "Set models.lineage_acknowledged: true only after manual confirmation."
            )

        anchor_tokenizer = tokenizer_fingerprint(anchor_path)
        greedy_tokenizer = tokenizer_fingerprint(greedy_path)
        configured_tokenizer = tokenizer_fingerprint(tokenizer_path) if tokenizer_path else None
        endpoint_tokenizers = [
            ("anchor", anchor_tokenizer),
            ("greedy", greedy_tokenizer),
        ]
        use_configured_tokenizer = bool(
            allow_hf_serialization_drift and configured_tokenizer and configured_tokenizer["files"]
        )
        if use_configured_tokenizer:
            if (
                anchor_tokenizer["files"]
                and greedy_tokenizer["files"]
                and anchor_tokenizer["files"] != greedy_tokenizer["files"]
            ):
                warnings.append(
                    "Tokenizer file hashes differ between endpoints; generation uses "
                    "models.tokenizer_path."
                )
            for endpoint_name, endpoint_tokenizer in endpoint_tokenizers:
                if (
                    endpoint_tokenizer["files"]
                    and endpoint_tokenizer["files"] != configured_tokenizer["files"]
                ):
                    warnings.append(
                        f"Configured tokenizer files differ from the {endpoint_name} "
                        "checkpoint tokenizer; generation uses models.tokenizer_path."
                    )
        else:
            if anchor_tokenizer["files"] and greedy_tokenizer["files"]:
                if anchor_tokenizer["files"] != greedy_tokenizer["files"]:
                    errors.append(
                        "Tokenizer file hashes differ between anchor and greedy checkpoints."
                    )
            if configured_tokenizer and configured_tokenizer["files"]:
                for endpoint_name, endpoint_tokenizer in endpoint_tokenizers:
                    if (
                        endpoint_tokenizer["files"]
                        and endpoint_tokenizer["files"] != configured_tokenizer["files"]
                    ):
                        errors.append(
                            f"Configured tokenizer files differ from the {endpoint_name} "
                            "checkpoint tokenizer."
                        )
            elif not (anchor_tokenizer["files"] and greedy_tokenizer["files"]):
                warnings.append(
                    "No complete tokenizer fingerprint was found in the endpoints or at "
                    "models.tokenizer_path; tokenizer compatibility could not be proven."
                )

        delta_statistics = []
        for group_key, values in sorted(delta_accumulators.items()):
            anchor_norm = math.sqrt(max(0.0, values["anchor_sq"]))
            greedy_norm = math.sqrt(max(0.0, values["greedy_sq"]))
            delta_norm = math.sqrt(max(0.0, values["delta_sq"]))
            denominator = anchor_norm * greedy_norm
            cosine = values["anchor_greedy_dot"] / denominator if denominator > 0 else None
            layer_index = int(group_key.split("_", 1)[1]) if group_key.startswith("layer_") else None
            delta_statistics.append(
                {
                    "group": group_key,
                    "layer_index": layer_index,
                    "parameter_count": int(values["parameter_count"]),
                    "anchor_l2_norm": anchor_norm,
                    "greedy_l2_norm": greedy_norm,
                    "delta_l2_norm": delta_norm,
                    "relative_delta_norm": delta_norm / max(1e-30, anchor_norm),
                    "anchor_greedy_cosine": cosine,
                }
            )

        report = {
            "compatible": not errors,
            "errors": errors,
            "warnings": warnings,
            "anchor": checkpoint_fingerprint(anchor_path),
            "greedy": checkpoint_fingerprint(greedy_path),
            "anchor_tokenizer": anchor_tokenizer,
            "greedy_tokenizer": greedy_tokenizer,
            "configured_tokenizer": configured_tokenizer,
            "critical_config_differences": config_differences,
            "equivalent_config_fields": equivalent_config_fields,
            "aliased_tied_lm_head": aliased_tied_lm_head,
            "tensor_key_count": len(common_keys),
            "parameter_count": parameter_count,
            "anchor_weight_bytes_unsharded": weight_bytes,
            "layer_indices": layers,
            "shape_mismatches": shape_mismatches[:100],
            "dtype_mismatches": dtype_mismatches[:100],
            "non_floating_mismatches": non_floating_mismatches[:100],
            "floating_kind_mismatches": floating_kind_mismatches[:100],
            "non_finite_tensors": non_finite_tensors[:100],
            "parameter_delta_statistics": delta_statistics,
            "missing_in_anchor": missing_in_anchor[:100],
            "missing_in_greedy": missing_in_greedy[:100],
        }
    return report


def require_compatible(report: Mapping[str, Any]) -> None:
    if not report.get("compatible"):
        detail = "\n  - ".join(str(item) for item in report.get("errors") or [])
        raise CompatibilityError(f"Checkpoints are not safe to merge:\n  - {detail}")


def write_compatibility_report(path: Path, report: Mapping[str, Any]) -> None:
    write_json(path, dict(report))
    lines = [
        "# CHECKPOINT COMPATIBILITY REPORT",
        "",
        f"Compatible: **{bool(report.get('compatible'))}**",
        f"Parameters: **{int(report.get('parameter_count', 0)):,}**",
        f"Layers: `{report.get('layer_indices')}`",
        "",
        "## Errors",
        "",
    ]
    errors = report.get("errors") or []
    lines.extend([f"- {item}" for item in errors] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings") or []
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend(
        [
            "",
            "## Provenance fingerprints",
            "",
            "```json",
            json.dumps({"anchor": report.get("anchor"), "greedy": report.get("greedy")}, indent=2),
            "```",
            "",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
