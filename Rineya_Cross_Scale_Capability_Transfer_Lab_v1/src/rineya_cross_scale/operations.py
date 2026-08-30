from __future__ import annotations

import gc
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from .checkpoint import (
    CheckpointError,
    CheckpointReader,
    StreamingSafetensorsWriter,
    checkpoint_fingerprint,
    copy_sidecars,
    validate_safe_destination,
)
from .diagnostics import tensor_delta_stats
from .hashing import atomic_write_json, canonical_sha256


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _staging(
    destination: Path, sources: list[Path], *, overwrite: bool
) -> tuple[Path, Path]:
    destination = validate_safe_destination(destination, sources)
    staging = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    validate_safe_destination(staging, sources)
    for path, permitted in ((destination, overwrite), (staging, True)):
        if not path.exists():
            continue
        if not permitted:
            raise FileExistsError(f"destination already exists: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    staging.mkdir(parents=True)
    return destination, staging


def _neutralize_supa_gate(config_path: Path) -> dict[str, Any] | None:
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "abstain_logit_bias" not in config:
        return None
    original = config["abstain_logit_bias"]
    config["abstain_logit_bias"] = 0.0
    atomic_write_json(config_path, config)
    return {
        "field": "abstain_logit_bias",
        "original": original,
        "candidate_value": 0.0,
        "reason": "primary weight-space evaluation is ungated",
    }


def _require_source_tensor(
    reader: CheckpointReader, key: str, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    if key not in reader.weight_map:
        raise CheckpointError(f"source checkpoint lacks tensor: {key}")
    value = reader.get_tensor(key)
    if tuple(value.shape) != expected_shape:
        raise CheckpointError(
            f"shape mismatch for {key}: {tuple(value.shape)} vs {expected_shape}"
        )
    return value


def build_delta(
    *,
    projected_path: Path,
    reference_path: Path,
    destination: Path,
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    projected_path = projected_path.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve()
    destination, staging = _staging(
        destination, [projected_path, reference_path], overwrite=overwrite
    )
    records: list[dict[str, Any]] = []
    try:
        writer = StreamingSafetensorsWriter(staging, max_shard_bytes)
        with CheckpointReader(projected_path) as projected, CheckpointReader(
            reference_path
        ) as reference:
            for ordinal, key in enumerate(projected.keys, 1):
                p = projected.get_tensor(key)
                r = _require_source_tensor(reference, key, tuple(p.shape))
                if p.is_floating_point() and r.is_floating_point():
                    delta = p.float().sub(r.float())
                    if not torch.isfinite(delta).all().item():
                        raise CheckpointError(f"non-finite delta: {key}")
                elif torch.equal(p, r):
                    delta = torch.zeros_like(p)
                else:
                    raise CheckpointError(f"non-floating tensor differs: {key}")
                writer.add(key, delta)
                stats = tensor_delta_stats(key, delta, base=r)
                records.append(stats)
                del p, r, delta
                if ordinal % 40 == 0:
                    gc.collect()
        output_index = writer.finalize()
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "cross_scale_delta",
            "equation": "projected_donor - official_qwen3_0.6b",
            "projected_path": str(projected_path),
            "reference_path": str(reference_path),
            "source_fingerprints": source_fingerprints or {},
            "arithmetic_dtype": "float32",
            "output_dtype": "float32",
            "output_index": output_index,
            "tensor_count": len(records),
            "diagnostics_sha256": canonical_sha256(records),
        }
        atomic_write_json(staging / "delta_stats.json", records)
        atomic_write_json(staging / "delta_manifest.json", manifest)
        os.replace(staging, destination)
        manifest["output_path"] = str(destination)
        manifest["output_fingerprint"] = checkpoint_fingerprint(destination)
        atomic_write_json(destination / "delta_manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_candidate(
    *,
    base_path: Path,
    source_paths: list[Path],
    destination: Path,
    operation: str,
    coefficient: float,
    tensor_operation: Callable[[str, torch.Tensor, list[torch.Tensor]], torch.Tensor],
    output_dtype: str,
    max_shard_bytes: int,
    overwrite: bool,
    operation_parameters: dict[str, Any] | None = None,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not math_is_finite(coefficient):
        raise CheckpointError("coefficient must be finite")
    if output_dtype not in DTYPES:
        raise CheckpointError(f"unsupported output dtype: {output_dtype}")
    base_path = base_path.expanduser().resolve()
    source_paths = [path.expanduser().resolve() for path in source_paths]
    destination, staging = _staging(
        destination, [base_path, *source_paths], overwrite=overwrite
    )
    records: list[dict[str, Any]] = []
    try:
        writer = StreamingSafetensorsWriter(staging, max_shard_bytes)
        with CheckpointReader(base_path) as base:
            source_contexts = [CheckpointReader(path) for path in source_paths]
            with context_stack(source_contexts) as sources:
                for ordinal, key in enumerate(base.keys, 1):
                    b = base.get_tensor(key)
                    values = [
                        _require_source_tensor(source, key, tuple(b.shape))
                        for source in sources
                    ]
                    if b.is_floating_point() and all(
                        value.is_floating_point() for value in values
                    ):
                        candidate_fp32 = tensor_operation(key, b.float(), values)
                        if candidate_fp32.dtype != torch.float32:
                            candidate_fp32 = candidate_fp32.float()
                        if not torch.isfinite(candidate_fp32).all().item():
                            raise CheckpointError(f"non-finite candidate tensor: {key}")
                        delta = candidate_fp32 - b.float()
                        output = candidate_fp32.to(DTYPES[output_dtype])
                        records.append(tensor_delta_stats(key, delta, base=b))
                    elif all(torch.equal(b, value) for value in values):
                        output = b
                    else:
                        raise CheckpointError(f"non-floating source mismatch: {key}")
                    writer.add(key, output)
                    del b, values, output
                    if ordinal % 40 == 0:
                        gc.collect()
        output_index = writer.finalize()
        sidecars = copy_sidecars(base_path, staging)
        gate_control = _neutralize_supa_gate(staging / "config.json")
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "coefficient": coefficient,
            "operation_parameters": operation_parameters or {},
            "base_path": str(base_path),
            "source_paths": [str(path) for path in source_paths],
            "source_fingerprints": source_fingerprints or {},
            "arithmetic_dtype": "float32",
            "output_dtype": output_dtype,
            "gate_control": gate_control,
            "output_index": output_index,
            "sidecars": sidecars,
            "tensor_diagnostics_sha256": canonical_sha256(records),
        }
        atomic_write_json(staging / "candidate_tensor_stats.json", records)
        atomic_write_json(staging / "candidate_manifest.json", manifest)
        os.replace(staging, destination)
        manifest["output_path"] = str(destination)
        manifest["output_fingerprint"] = checkpoint_fingerprint(destination)
        atomic_write_json(destination / "candidate_manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def context_stack(readers: list[CheckpointReader]):
    import contextlib

    stack = contextlib.ExitStack()
    return _ReaderStack(stack, readers)


class _ReaderStack:
    def __init__(self, stack: Any, readers: list[CheckpointReader]) -> None:
        self.stack = stack
        self.readers = readers

    def __enter__(self) -> list[CheckpointReader]:
        return [self.stack.enter_context(reader) for reader in self.readers]

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stack.close()


def math_is_finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


def merge_absolute(
    *, base_path: Path, projected_path: Path, destination: Path, alpha: float, **kwargs: Any
) -> dict[str, Any]:
    return _build_candidate(
        base_path=base_path,
        source_paths=[projected_path],
        destination=destination,
        operation="absolute_interpolation",
        coefficient=alpha,
        tensor_operation=lambda key, base, values: base.mul(1.0 - alpha).add(
            values[0].float(), alpha=alpha
        ),
        **kwargs,
    )


def merge_task_vector(
    *, base_path: Path, delta_path: Path, destination: Path, alpha: float, **kwargs: Any
) -> dict[str, Any]:
    return _build_candidate(
        base_path=base_path,
        source_paths=[delta_path],
        destination=destination,
        operation="cross_scale_task_vector",
        coefficient=alpha,
        tensor_operation=lambda key, base, values: base.add(
            values[0].float(), alpha=alpha
        ),
        **kwargs,
    )


def merge_consensus(
    *,
    base_path: Path,
    delta4_path: Path,
    delta8_path: Path,
    destination: Path,
    alpha: float,
    method: str = "mean",
    cosine_threshold: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    if method not in {"mean", "sign_agree", "tensor_cosine"}:
        raise CheckpointError(f"unsupported consensus method: {method}")

    def operation(key: str, base: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        d4 = values[0].float()
        d8 = values[1].float()
        combined = (d4 + d8) * 0.5
        if method == "sign_agree":
            combined = combined * (torch.sign(d4) == torch.sign(d8))
        elif method == "tensor_cosine":
            denominator = torch.linalg.vector_norm(d4) * torch.linalg.vector_norm(d8)
            cosine = (
                float(torch.dot(d4.reshape(-1), d8.reshape(-1)).item() / denominator.item())
                if denominator.item() > 0
                else 0.0
            )
            if cosine <= cosine_threshold:
                combined = torch.zeros_like(combined)
        return base.add(combined, alpha=alpha)

    return _build_candidate(
        base_path=base_path,
        source_paths=[delta4_path, delta8_path],
        destination=destination,
        operation="cross_scale_consensus",
        coefficient=alpha,
        tensor_operation=operation,
        operation_parameters={
            "method": method,
            "cosine_threshold": cosine_threshold,
        },
        **kwargs,
    )
