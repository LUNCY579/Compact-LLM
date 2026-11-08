from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .diagnostics import classify_tensor
from .operations import _build_candidate


def tensor_selected(
    key: str,
    *,
    layers: set[int] | None = None,
    modules: set[str] | None = None,
) -> bool:
    metadata = classify_tensor(key)
    if layers is not None and metadata["layer"] not in layers:
        return False
    if modules is None:
        return True
    module = str(metadata["module"])
    return any(module == requested or module.startswith(requested + ".") for requested in modules)


def merge_masked_delta(
    *,
    base_path: Path,
    delta_path: Path,
    destination: Path,
    alpha: float,
    layers: set[int] | None = None,
    modules: set[str] | None = None,
    output_dtype: str = "float16",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_keys: list[str] = []

    def operation(key: str, base: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        if tensor_selected(key, layers=layers, modules=modules):
            selected_keys.append(key)
            return base.add(values[0].float(), alpha=alpha)
        return base

    manifest = _build_candidate(
        base_path=base_path,
        source_paths=[delta_path],
        destination=destination,
        operation="masked_task_vector_injection",
        coefficient=alpha,
        tensor_operation=operation,
        output_dtype=output_dtype,
        max_shard_bytes=max_shard_bytes,
        overwrite=overwrite,
        operation_parameters={
            "layers": sorted(layers) if layers is not None else None,
            "modules": sorted(modules) if modules is not None else None,
        },
        source_fingerprints=source_fingerprints,
    )
    manifest["selected_tensor_count"] = len(selected_keys)
    manifest["selected_keys"] = selected_keys
    return manifest
