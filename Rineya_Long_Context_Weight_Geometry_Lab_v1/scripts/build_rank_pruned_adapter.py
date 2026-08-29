#!/usr/bin/env python3
"""Prune each LoRA module to its strongest structured rank components.

For component ``j`` of a LoRA pair, the ranking score is the Frobenius norm of
its rank-one outer product::

    importance_j = ||B[:, j]||_2 * ||A[j, :]||_2

The top-k components are retained independently for every module.  The output
adapter keeps the source's effective alpha/r scaling.  Optional module-wise
rescaling preserves the exact Frobenius norm of each dense LoRA delta, but is
disabled by default so the pruning experiment does not silently amplify the
surviving components.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from build_exact_adapter_soup import (
    SoupValidationError,
    SourceAdapter,
    inspect_source,
    sha256_file,
)
from safetensors import safe_open
from safetensors.numpy import save_file


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _serialized_number(value: float) -> int | float:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12):
        return int(rounded)
    return float(value)


def _output_config(source: SourceAdapter, target_rank: int) -> dict[str, Any]:
    if bool(source.config.get("use_rslora", False)):
        raise SoupValidationError(
            "Rank pruning currently supports standard alpha/r LoRA scaling only"
        )
    config = dict(source.config)
    config.update(
        {
            "alpha_pattern": {},
            "inference_mode": True,
            "lora_alpha": _serialized_number(source.scale * target_rank),
            "r": target_rank,
            "rank_pattern": {},
            "use_rslora": False,
        }
    )
    return config


def component_importance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Frobenius norm of every B-column/A-row outer product."""

    a_norm = np.linalg.norm(a.astype(np.float64, copy=False), axis=1)
    b_norm = np.linalg.norm(b.astype(np.float64, copy=False), axis=0)
    importance = a_norm * b_norm
    if not np.isfinite(importance).all():
        raise SoupValidationError("Component importance contains NaN or infinity")
    return importance


def select_top_components(importance: np.ndarray, target_rank: int) -> np.ndarray:
    """Select top components deterministically, returning original-rank order."""

    if importance.ndim != 1:
        raise SoupValidationError("Component importance must be one-dimensional")
    if target_rank <= 0 or target_rank > importance.size:
        raise SoupValidationError(
            f"target rank must be in [1, {importance.size}], got {target_rank}"
        )
    indices = np.arange(importance.size)
    ranked = np.lexsort((indices, -importance))
    return np.sort(ranked[:target_rank])


def _dense_delta_norm(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    dense = b.astype(np.float64, copy=False) @ a.astype(np.float64, copy=False)
    return abs(scale) * float(np.linalg.norm(dense))


def _norm_rescale(
    original_a: np.ndarray,
    original_b: np.ndarray,
    pruned_a: np.ndarray,
    pruned_b: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    before = _dense_delta_norm(original_a, original_b, scale)
    after_unscaled = _dense_delta_norm(pruned_a, pruned_b, scale)
    if after_unscaled == 0.0:
        if before != 0.0:
            raise SoupValidationError(
                "Cannot norm-rescale a zero pruned delta to a non-zero source delta"
            )
        factor = 1.0
    else:
        factor = before / after_unscaled
    root_factor = math.sqrt(factor)
    a_out = np.multiply(
        pruned_a,
        np.asarray(root_factor, dtype=pruned_a.dtype),
        dtype=pruned_a.dtype,
    )
    b_out = np.multiply(
        pruned_b,
        np.asarray(root_factor, dtype=pruned_b.dtype),
        dtype=pruned_b.dtype,
    )
    if not np.isfinite(a_out).all() or not np.isfinite(b_out).all():
        raise SoupValidationError("Norm-preserving rescale overflowed adapter tensors")
    after = _dense_delta_norm(a_out, b_out, scale)
    return a_out, b_out, factor, before, after


def _build_pruned_tensors(
    source: SourceAdapter,
    target_rank: int,
    norm_preserving_rescale: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    tensors: dict[str, np.ndarray] = {}
    module_rows: list[dict[str, Any]] = []
    with safe_open(str(source.weights_path), framework="numpy") as handle:
        for module in source.modules:
            a = handle.get_tensor(module.a_key)
            b = handle.get_tensor(module.b_key)
            importance = component_importance(a, b)
            kept = select_top_components(importance, target_rank)
            pruned_a = np.ascontiguousarray(a[kept, :])
            pruned_b = np.ascontiguousarray(b[:, kept])

            rescale_factor = 1.0
            norm_before: float | None = None
            norm_after: float | None = None
            if norm_preserving_rescale:
                (
                    pruned_a,
                    pruned_b,
                    rescale_factor,
                    norm_before,
                    norm_after,
                ) = _norm_rescale(a, b, pruned_a, pruned_b, source.scale)

            component_energy = np.square(importance, dtype=np.float64)
            total_importance = float(importance.sum())
            kept_importance_sum = float(importance[kept].sum())
            total_energy = float(component_energy.sum())
            kept_energy = float(component_energy[kept].sum())
            kept_importance_fraction = (
                kept_importance_sum / total_importance
                if total_importance > 0.0
                else 1.0
            )
            kept_fraction = kept_energy / total_energy if total_energy > 0.0 else 1.0
            tensors[module.a_key] = pruned_a
            tensors[module.b_key] = pruned_b
            module_rows.append(
                {
                    "module": module.prefix,
                    "source_rank": source.rank,
                    "target_rank": target_rank,
                    "kept_indices": [int(index) for index in kept],
                    "kept_importance": [float(importance[index]) for index in kept],
                    "total_importance": total_importance,
                    "kept_importance_sum": kept_importance_sum,
                    "kept_importance_fraction": kept_importance_fraction,
                    "total_component_energy": total_energy,
                    "kept_component_energy": kept_energy,
                    "kept_component_energy_fraction": kept_fraction,
                    "kept_energy_fraction": kept_fraction,
                    "norm_rescale_factor": rescale_factor,
                    "dense_delta_frobenius_before": norm_before,
                    "dense_delta_frobenius_after": norm_after,
                }
            )
    return tensors, module_rows


def _verify_output(
    source: SourceAdapter,
    output_weights: Path,
    target_rank: int,
    module_rows: list[dict[str, Any]],
    norm_preserving_rescale: bool,
) -> dict[str, Any]:
    row_by_module = {row["module"]: row for row in module_rows}
    worst_norm_relative_error = 0.0
    with contextlib.ExitStack() as stack:
        source_handle = stack.enter_context(
            safe_open(str(source.weights_path), framework="numpy")
        )
        output_handle = stack.enter_context(
            safe_open(str(output_weights), framework="numpy")
        )
        output_keys = set(output_handle.keys())
        expected_keys = {
            key
            for module in source.modules
            for key in (module.a_key, module.b_key)
        }
        if output_keys != expected_keys:
            raise SoupValidationError("Pruned output tensor keys do not match the source")
        for module in source.modules:
            a_out = output_handle.get_tensor(module.a_key)
            b_out = output_handle.get_tensor(module.b_key)
            if a_out.shape != (target_rank, module.in_features):
                raise SoupValidationError(f"Unexpected output A shape for {module.prefix}")
            if b_out.shape != (module.out_features, target_rank):
                raise SoupValidationError(f"Unexpected output B shape for {module.prefix}")
            if not np.isfinite(a_out).all() or not np.isfinite(b_out).all():
                raise SoupValidationError(f"Non-finite pruned output for {module.prefix}")
            if norm_preserving_rescale:
                a_source = source_handle.get_tensor(module.a_key)
                b_source = source_handle.get_tensor(module.b_key)
                before = _dense_delta_norm(a_source, b_source, source.scale)
                after = _dense_delta_norm(a_out, b_out, source.scale)
                relative = abs(after - before) / max(
                    before, np.finfo(np.float64).tiny
                )
                worst_norm_relative_error = max(worst_norm_relative_error, relative)
                if not math.isclose(after, before, rel_tol=2e-5, abs_tol=1e-7):
                    raise SoupValidationError(
                        f"Dense-delta norm verification failed for {module.prefix}"
                    )
                row = row_by_module[module.prefix]
                row["dense_delta_frobenius_after"] = after
    return {
        "module_count": len(source.modules),
        "all_shapes_and_finiteness_verified": True,
        "norm_preservation_verified": norm_preserving_rescale,
        "worst_norm_relative_error": worst_norm_relative_error,
    }


def _manifest(
    source: SourceAdapter,
    staging: Path,
    target_rank: int,
    output_config: dict[str, Any],
    module_rows: list[dict[str, Any]],
    norm_preserving_rescale: bool,
    verification: dict[str, Any],
) -> dict[str, Any]:
    total_energy = float(sum(row["total_component_energy"] for row in module_rows))
    kept_energy = float(sum(row["kept_component_energy"] for row in module_rows))
    kept_fraction = kept_energy / total_energy if total_energy > 0.0 else 1.0
    config_path = staging / "adapter_config.json"
    weights_path = staging / "adapter_model.safetensors"
    return {
        "schema_version": "1.0",
        "kind": "structured_lora_rank_component_pruning",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "importance_formula": "l2_norm(B[:, j]) * l2_norm(A[j, :])",
        "component_energy_definition": "sum(importance_j ** 2); proxy ignores cross-component terms",
        "source": {
            "path": str(source.path),
            "rank": source.rank,
            "lora_alpha": source.alpha,
            "effective_scale": source.scale,
            "base_model_name_or_path": source.config.get("base_model_name_or_path"),
            "revision": source.config.get("revision"),
            "adapter_config_sha256": source.config_sha256,
            "adapter_model_sha256": source.weights_sha256,
            "adapter_model_bytes": source.weights_path.stat().st_size,
        },
        "pruning": {
            "target_rank": target_rank,
            "norm_preserving_rescale": norm_preserving_rescale,
            "total_component_energy": total_energy,
            "kept_component_energy": kept_energy,
            "kept_component_energy_fraction": kept_fraction,
            "kept_energy_fraction": kept_fraction,
            "modules": module_rows,
        },
        "output": {
            "rank": target_rank,
            "lora_alpha": output_config["lora_alpha"],
            "effective_scale": source.scale,
            "module_count": len(source.modules),
            "tensor_count": len(source.modules) * 2,
            "dtype": source.modules[0].dtype,
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_model_sha256": sha256_file(weights_path),
            "adapter_model_bytes": weights_path.stat().st_size,
        },
        "verification": verification,
    }


def build_rank_pruned_adapter(
    source_path: Path,
    target_rank: int,
    output_dir: Path,
    *,
    norm_preserving_rescale: bool = False,
) -> dict[str, Any]:
    source = inspect_source(source_path, 1.0)
    if bool(source.config.get("use_rslora", False)):
        raise SoupValidationError(
            "Rank pruning currently supports standard alpha/r LoRA scaling only"
        )
    if target_rank <= 0 or target_rank > source.rank:
        raise SoupValidationError(
            f"target rank must be in [1, {source.rank}], got {target_rank}"
        )

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    try:
        output.relative_to(source.path)
    except ValueError:
        pass
    else:
        raise SoupValidationError(
            f"Output directory must not be inside the source checkpoint: {source.path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    output_config = _output_config(source, target_rank)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    committed = False
    try:
        tensors, module_rows = _build_pruned_tensors(
            source, target_rank, norm_preserving_rescale
        )
        weights_path = staging / "adapter_model.safetensors"
        save_file(
            tensors,
            str(weights_path),
            metadata={
                "format": "pt",
                "rineya_transform": "structured_lora_rank_component_pruning",
            },
        )
        del tensors
        _write_json(staging / "adapter_config.json", output_config)
        verification = _verify_output(
            source,
            weights_path,
            target_rank,
            module_rows,
            norm_preserving_rescale,
        )
        manifest = _manifest(
            source,
            staging,
            target_rank,
            output_config,
            module_rows,
            norm_preserving_rescale,
            verification,
        )
        _write_json(staging / "rank_pruning_manifest.json", manifest)
        staging.replace(output)
        committed = True
        return manifest
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True, dest="target_rank")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--norm-preserving-rescale",
        action="store_true",
        help="Preserve each module's dense-delta Frobenius norm after pruning.",
    )
    args = parser.parse_args()
    manifest = build_rank_pruned_adapter(
        args.source,
        args.target_rank,
        args.output_dir,
        norm_preserving_rescale=args.norm_preserving_rescale,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.expanduser().resolve()),
                "source_rank": manifest["source"]["rank"],
                "target_rank": manifest["output"]["rank"],
                "lora_alpha": manifest["output"]["lora_alpha"],
                "kept_component_energy_fraction": manifest["pruning"][
                    "kept_component_energy_fraction"
                ],
                "norm_preserving_rescale": manifest["pruning"][
                    "norm_preserving_rescale"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
