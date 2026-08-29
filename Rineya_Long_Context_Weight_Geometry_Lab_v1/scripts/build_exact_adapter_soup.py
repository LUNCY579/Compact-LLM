#!/usr/bin/env python3
"""Build an exact weighted LoRA-delta soup as one concatenated-rank adapter.

For vanilla LoRA adapter ``i``, the effective update is::

    delta_i = scale_i * B_i @ A_i

This utility stores the literal weighted sum without loading the base model by
concatenating ``coefficient_i * scale_i * A_i`` along the rank dimension and
the corresponding ``B_i`` tensors along their rank dimension.  The emitted
adapter has alpha equal to rank, so its own PEFT scaling is exactly one.

Coefficients are intentionally literal: they may be negative and they are not
normalized.  Thus the tool supports both conventional checkpoint averaging
and base-anchored task-vector/delta-dose recipes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"
VARIABLE_CONFIG_FIELDS = {
    "alpha_pattern",
    "base_model_name_or_path",
    "inference_mode",
    "lora_alpha",
    "r",
    "rank_pattern",
    "revision",
}
UNSUPPORTED_TRUTHY_CONFIG_FIELDS = (
    "alora_invocation_tokens",
    "arrow_config",
    "layer_replication",
    "lora_bias",
    "modules_to_save",
    "target_parameters",
    "trainable_token_indices",
    "use_bdlora",
    "use_dora",
    "use_qalora",
)


class SoupValidationError(ValueError):
    """Raised when adapters cannot be composed exactly by this utility."""


class ModuleSchema(NamedTuple):
    prefix: str
    a_key: str
    b_key: str
    rank: int
    in_features: int
    out_features: int
    dtype: str


class SourceAdapter(NamedTuple):
    path: Path
    coefficient: float
    config: dict[str, Any]
    weights_path: Path
    rank: int
    alpha: float
    scale: float
    modules: tuple[ModuleSchema, ...]
    config_sha256: str
    weights_sha256: str


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SoupValidationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SoupValidationError(f"Expected a JSON object in {path}")
    return value


def _finite_float(value: Any, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SoupValidationError(f"{source}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise SoupValidationError(f"{source}: {field} must be finite")
    return result


def _positive_int(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool):
        raise SoupValidationError(f"{source}: {field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SoupValidationError(
            f"{source}: {field} must be a positive integer"
        ) from exc
    if result <= 0 or result != value:
        raise SoupValidationError(f"{source}: {field} must be a positive integer")
    return result


def _base_identity(config: dict[str, Any], source: Path) -> dict[str, Any]:
    raw = config.get("base_model_name_or_path")
    if not isinstance(raw, str) or not raw.strip():
        raise SoupValidationError(
            f"{source}: base_model_name_or_path must be a non-empty string"
        )
    raw = raw.strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.exists():
        kind = "local_path"
        value = os.path.normcase(str(candidate.resolve()))
    else:
        relative = source / candidate
        if relative.exists():
            kind = "local_path"
            value = os.path.normcase(str(relative.resolve()))
        else:
            kind = "configured_identifier"
            value = os.path.normcase(os.path.normpath(raw))
    return {
        "kind": kind,
        "value": value,
        "revision": config.get("revision"),
    }


def _canonical_config_topology(config: dict[str, Any]) -> dict[str, Any]:
    topology = {
        key: value for key, value in config.items() if key not in VARIABLE_CONFIG_FIELDS
    }
    target_modules = topology.get("target_modules")
    if isinstance(target_modules, list):
        topology["target_modules"] = sorted(str(value) for value in target_modules)
    exclude_modules = topology.get("exclude_modules")
    if isinstance(exclude_modules, list):
        topology["exclude_modules"] = sorted(str(value) for value in exclude_modules)
    return topology


def _validate_supported_config(config: dict[str, Any], source: Path) -> tuple[int, float, float]:
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise SoupValidationError(f"{source}: only peft_type=LORA is supported")
    if str(config.get("bias", "none")).lower() != "none":
        raise SoupValidationError(f"{source}: LoRA bias tensors are not supported")
    for field in UNSUPPORTED_TRUTHY_CONFIG_FIELDS:
        if config.get(field) not in (None, False, [], {}):
            raise SoupValidationError(
                f"{source}: {field} is not supported by exact vanilla-LoRA concatenation"
            )
    if config.get("rank_pattern") not in (None, {}):
        raise SoupValidationError(f"{source}: non-empty rank_pattern is not supported")
    if config.get("alpha_pattern") not in (None, {}):
        raise SoupValidationError(f"{source}: non-empty alpha_pattern is not supported")

    rank = _positive_int(config.get("r"), "r", source)
    alpha = _finite_float(config.get("lora_alpha"), "lora_alpha", source)
    use_rslora = bool(config.get("use_rslora", False))
    scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    if not math.isfinite(scale):
        raise SoupValidationError(f"{source}: computed LoRA scale is not finite")
    return rank, alpha, scale


def _inspect_tensor_schema(
    weights_path: Path, expected_rank: int
) -> tuple[ModuleSchema, ...]:
    modules: list[ModuleSchema] = []
    with safe_open(str(weights_path), framework="numpy") as handle:
        keys = set(handle.keys())
        unexpected = sorted(
            key for key in keys if not key.endswith((A_SUFFIX, B_SUFFIX))
        )
        if unexpected:
            raise SoupValidationError(
                f"{weights_path}: unsupported non-A/B tensor {unexpected[0]!r}"
            )
        a_keys = sorted(key for key in keys if key.endswith(A_SUFFIX))
        b_keys = sorted(key for key in keys if key.endswith(B_SUFFIX))
        if not a_keys:
            raise SoupValidationError(f"{weights_path}: no LoRA A tensors found")

        expected_b = {key[: -len(A_SUFFIX)] + B_SUFFIX for key in a_keys}
        missing_b = sorted(expected_b - set(b_keys))
        expected_a = {key[: -len(B_SUFFIX)] + A_SUFFIX for key in b_keys}
        missing_a = sorted(expected_a - set(a_keys))
        if missing_b or missing_a:
            detail = missing_b[0] if missing_b else missing_a[0]
            raise SoupValidationError(
                f"{weights_path}: unpaired LoRA A/B tensor; missing {detail!r}"
            )

        observed_dtypes: set[str] = set()
        for a_key in a_keys:
            prefix = a_key[: -len(A_SUFFIX)]
            b_key = prefix + B_SUFFIX
            a = handle.get_tensor(a_key)
            b = handle.get_tensor(b_key)
            if a.ndim != 2 or b.ndim != 2:
                raise SoupValidationError(
                    f"{weights_path}: {prefix} must contain two-dimensional A/B tensors"
                )
            if not np.issubdtype(a.dtype, np.floating) or not np.issubdtype(
                b.dtype, np.floating
            ):
                raise SoupValidationError(
                    f"{weights_path}: {prefix} must contain floating-point tensors"
                )
            if a.dtype != b.dtype:
                raise SoupValidationError(
                    f"{weights_path}: {prefix} A/B dtypes differ ({a.dtype} vs {b.dtype})"
                )
            if a.shape[0] != expected_rank or b.shape[1] != expected_rank:
                raise SoupValidationError(
                    f"{weights_path}: {prefix} tensor rank does not match config r={expected_rank}"
                )
            if a.shape[0] != b.shape[1]:
                raise SoupValidationError(
                    f"{weights_path}: {prefix} A/B rank dimensions do not match"
                )
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise SoupValidationError(
                    f"{weights_path}: {prefix} contains NaN or infinite values"
                )
            dtype = str(a.dtype)
            observed_dtypes.add(dtype)
            modules.append(
                ModuleSchema(
                    prefix=prefix,
                    a_key=a_key,
                    b_key=b_key,
                    rank=int(a.shape[0]),
                    in_features=int(a.shape[1]),
                    out_features=int(b.shape[0]),
                    dtype=dtype,
                )
            )
        if len(observed_dtypes) != 1:
            raise SoupValidationError(
                f"{weights_path}: mixed adapter tensor dtypes are not supported"
            )
    return tuple(modules)


def inspect_source(path: Path, coefficient: float) -> SourceAdapter:
    source = path.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise SoupValidationError(f"Adapter source is not a directory: {source}")
    coefficient = _finite_float(coefficient, "coefficient", source)
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file():
        raise SoupValidationError(f"Missing adapter_config.json under {source}")
    if not weights_path.is_file():
        raise SoupValidationError(f"Missing adapter_model.safetensors under {source}")

    config = read_json_object(config_path)
    rank, alpha, scale = _validate_supported_config(config, source)
    modules = _inspect_tensor_schema(weights_path, rank)
    return SourceAdapter(
        path=source,
        coefficient=coefficient,
        config=config,
        weights_path=weights_path,
        rank=rank,
        alpha=alpha,
        scale=scale,
        modules=modules,
        config_sha256=sha256_file(config_path),
        weights_sha256=sha256_file(weights_path),
    )


def validate_sources(sources: Sequence[SourceAdapter]) -> None:
    if not sources:
        raise SoupValidationError("At least one adapter source is required")
    paths = [source.path for source in sources]
    if len(paths) != len(set(paths)):
        raise SoupValidationError("Adapter source paths must be unique")

    reference = sources[0]
    reference_base = _base_identity(reference.config, reference.path)
    reference_topology = _canonical_config_topology(reference.config)
    reference_modules = {
        module.prefix: (module.in_features, module.out_features, module.dtype)
        for module in reference.modules
    }
    for source in sources[1:]:
        if _base_identity(source.config, source.path) != reference_base:
            raise SoupValidationError(
                f"Base-model identity mismatch: {reference.path} vs {source.path}"
            )
        if _canonical_config_topology(source.config) != reference_topology:
            raise SoupValidationError(
                f"PEFT configuration/topology mismatch: {reference.path} vs {source.path}"
            )
        modules = {
            module.prefix: (module.in_features, module.out_features, module.dtype)
            for module in source.modules
        }
        if modules != reference_modules:
            missing = sorted(set(reference_modules) - set(modules))
            extra = sorted(set(modules) - set(reference_modules))
            mismatched = sorted(
                prefix
                for prefix in set(reference_modules) & set(modules)
                if reference_modules[prefix] != modules[prefix]
            )
            detail = (
                f"missing={missing[:1]}, extra={extra[:1]}, "
                f"shape_or_dtype={mismatched[:1]}"
            )
            raise SoupValidationError(
                f"Adapter tensor topology mismatch for {source.path}: {detail}"
            )


def _output_config(reference: SourceAdapter, total_rank: int) -> dict[str, Any]:
    config = dict(reference.config)
    config.update(
        {
            "alpha_pattern": {},
            "inference_mode": True,
            "lora_alpha": total_rank,
            "r": total_rank,
            "rank_pattern": {},
            "use_rslora": False,
        }
    )
    return config


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_concatenated_tensors(
    sources: Sequence[SourceAdapter], total_rank: int
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    with contextlib.ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(str(source.weights_path), framework="numpy"))
            for source in sources
        ]
        for module in sources[0].modules:
            a_blocks: list[np.ndarray] = []
            b_blocks: list[np.ndarray] = []
            for source, handle in zip(sources, handles):
                a = handle.get_tensor(module.a_key)
                b = handle.get_tensor(module.b_key)
                factor = source.coefficient * source.scale
                scaled_a = np.multiply(
                    a,
                    np.asarray(factor, dtype=a.dtype),
                    dtype=a.dtype,
                )
                if not np.isfinite(scaled_a).all():
                    raise SoupValidationError(
                        f"Weighted tensor overflowed for {source.path}: {module.a_key}"
                    )
                a_blocks.append(np.ascontiguousarray(scaled_a))
                b_blocks.append(np.ascontiguousarray(b))
            a_out = np.ascontiguousarray(np.concatenate(a_blocks, axis=0))
            b_out = np.ascontiguousarray(np.concatenate(b_blocks, axis=1))
            if a_out.shape[0] != total_rank or b_out.shape[1] != total_rank:
                raise AssertionError(f"Internal rank mismatch while building {module.prefix}")
            output[module.a_key] = a_out
            output[module.b_key] = b_out
    return output


def _even_sample(items: Sequence[ModuleSchema], count: int) -> list[ModuleSchema]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indexes = {
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    }
    return [items[index] for index in sorted(indexes)]


def select_verification_modules(
    modules: Sequence[ModuleSchema],
    mode: str,
    sample_count: int,
    small_max_elements: int,
) -> list[ModuleSchema]:
    if mode not in {"sample", "all"}:
        raise SoupValidationError("verification mode must be 'sample' or 'all'")
    if sample_count < 0 or small_max_elements < 0:
        raise SoupValidationError("verification limits must be non-negative")
    if mode == "all":
        return list(modules)
    small = [
        module
        for module in modules
        if module.in_features * module.out_features <= small_max_elements
    ]
    small_prefixes = {module.prefix for module in small}
    large = [module for module in modules if module.prefix not in small_prefixes]
    sampled = _even_sample(large, sample_count)
    selected = {module.prefix: module for module in (*small, *sampled)}
    if not selected and modules:
        fallback = _even_sample(list(modules), max(1, sample_count))
        selected = {module.prefix: module for module in fallback}
    return [selected[prefix] for prefix in sorted(selected)]


def verify_dense_identities(
    sources: Sequence[SourceAdapter],
    output_weights: Path,
    *,
    mode: str = "sample",
    sample_count: int = 8,
    small_max_elements: int = 1_100_000,
    rtol: float = 2e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    selected = select_verification_modules(
        sources[0].modules, mode, sample_count, small_max_elements
    )
    worst_relative = 0.0
    worst_absolute = 0.0
    checked: list[str] = []
    with contextlib.ExitStack() as stack:
        source_handles = [
            stack.enter_context(safe_open(str(source.weights_path), framework="numpy"))
            for source in sources
        ]
        output_handle = stack.enter_context(
            safe_open(str(output_weights), framework="numpy")
        )
        for module in selected:
            desired = np.zeros(
                (module.out_features, module.in_features), dtype=np.float64
            )
            for source, handle in zip(sources, source_handles):
                a = handle.get_tensor(module.a_key).astype(np.float64, copy=False)
                b = handle.get_tensor(module.b_key).astype(np.float64, copy=False)
                desired += (source.coefficient * source.scale) * (b @ a)
            a_out = output_handle.get_tensor(module.a_key).astype(
                np.float64, copy=False
            )
            b_out = output_handle.get_tensor(module.b_key).astype(
                np.float64, copy=False
            )
            actual = b_out @ a_out
            difference = actual - desired
            max_absolute = float(np.max(np.abs(difference), initial=0.0))
            desired_norm = float(np.linalg.norm(desired))
            relative = float(np.linalg.norm(difference)) / max(
                desired_norm, np.finfo(np.float64).tiny
            )
            worst_relative = max(worst_relative, relative)
            worst_absolute = max(worst_absolute, max_absolute)
            if not np.allclose(actual, desired, rtol=rtol, atol=atol):
                raise SoupValidationError(
                    "Dense identity verification failed for "
                    f"{module.prefix}: relative={relative:.3e}, max_abs={max_absolute:.3e}"
                )
            checked.append(module.prefix)
    return {
        "mode": mode,
        "checked_module_count": len(checked),
        "total_module_count": len(sources[0].modules),
        "checked_modules": checked,
        "sample_count": sample_count,
        "small_max_elements": small_max_elements,
        "rtol": rtol,
        "atol": atol,
        "worst_relative_frobenius_error": worst_relative,
        "worst_max_absolute_error": worst_absolute,
    }


def _manifest(
    sources: Sequence[SourceAdapter],
    output_dir: Path,
    output_config: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    config_path = output_dir / "adapter_config.json"
    weights_path = output_dir / "adapter_model.safetensors"
    return {
        "schema_version": "1.0",
        "kind": "exact_concatenated_lora_delta_soup",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "Delta = sum(coefficient_i * scale_i * B_i @ A_i)",
        "coefficient_sum": float(sum(source.coefficient for source in sources)),
        "base_identity": _base_identity(sources[0].config, sources[0].path),
        "sources": [
            {
                "path": str(source.path),
                "coefficient": source.coefficient,
                "rank": source.rank,
                "lora_alpha": source.alpha,
                "effective_scale": source.scale,
                "adapter_config_sha256": source.config_sha256,
                "adapter_model_sha256": source.weights_sha256,
                "adapter_model_bytes": source.weights_path.stat().st_size,
            }
            for source in sources
        ],
        "output": {
            "rank": int(output_config["r"]),
            "lora_alpha": output_config["lora_alpha"],
            "effective_scale": 1.0,
            "module_count": len(sources[0].modules),
            "tensor_count": len(sources[0].modules) * 2,
            "dtype": sources[0].modules[0].dtype,
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_model_sha256": sha256_file(weights_path),
            "adapter_model_bytes": weights_path.stat().st_size,
        },
        "verification": verification,
    }


def build_exact_adapter_soup(
    source_specs: Sequence[tuple[Path, float]],
    output_dir: Path,
    *,
    verification_mode: str = "sample",
    verification_samples: int = 8,
    verification_small_max_elements: int = 1_100_000,
) -> dict[str, Any]:
    sources = [inspect_source(path, coefficient) for path, coefficient in source_specs]
    validate_sources(sources)

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    for source in sources:
        try:
            output.relative_to(source.path)
        except ValueError:
            continue
        raise SoupValidationError(
            f"Output directory must not be inside a source checkpoint: {source.path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    total_rank = sum(source.rank for source in sources)
    output_config = _output_config(sources[0], total_rank)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    committed = False
    try:
        tensors = _build_concatenated_tensors(sources, total_rank)
        weights_path = staging / "adapter_model.safetensors"
        save_file(
            tensors,
            str(weights_path),
            metadata={
                "format": "pt",
                "rineya_merge": "exact_concatenated_lora_delta_soup",
            },
        )
        del tensors
        _write_json(staging / "adapter_config.json", output_config)
        verification = verify_dense_identities(
            sources,
            weights_path,
            mode=verification_mode,
            sample_count=verification_samples,
            small_max_elements=verification_small_max_elements,
        )
        manifest = _manifest(sources, staging, output_config, verification)
        _write_json(staging / "soup_manifest.json", manifest)
        staging.replace(output)
        committed = True
        return manifest
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        required=True,
        metavar=("ADAPTER_PATH", "COEFFICIENT"),
        help="Repeat once per source adapter. Coefficients are literal and not normalized.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", choices=("sample", "all"), default="sample")
    parser.add_argument("--verify-samples", type=int, default=8)
    parser.add_argument("--verify-small-max-elements", type=int, default=1_100_000)
    args = parser.parse_args()

    specs: list[tuple[Path, float]] = []
    for raw_path, raw_coefficient in args.source:
        try:
            coefficient = float(raw_coefficient)
        except ValueError as exc:
            parser.error(f"Invalid coefficient {raw_coefficient!r}: {exc}")
        specs.append((Path(raw_path), coefficient))

    manifest = build_exact_adapter_soup(
        specs,
        args.output_dir,
        verification_mode=args.verify,
        verification_samples=args.verify_samples,
        verification_small_max_elements=args.verify_small_max_elements,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.expanduser().resolve()),
                "rank": manifest["output"]["rank"],
                "coefficient_sum": manifest["coefficient_sum"],
                "verification": manifest["verification"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
