from __future__ import annotations

import gc
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .architecture import center_preserving_depth_map, uniform_depth_map
from .checkpoint import (
    CheckpointError,
    CheckpointReader,
    StreamingSafetensorsWriter,
    checkpoint_fingerprint,
    copy_sidecars,
    validate_safe_destination,
)
from .hashing import atomic_write_json, canonical_sha256, sha256_file


@dataclass(frozen=True)
class Alignment:
    target_from_donor: torch.Tensor  # [target_hidden, donor_hidden]
    donor_from_target: torch.Tensor  # [donor_hidden, target_hidden]
    sample_ids: tuple[int, ...]
    heldout_ids: tuple[int, ...]
    metrics: dict[str, float | None]
    method: str
    ridge_absolute_donor: float | None
    ridge_absolute_target: float | None
    scale: float


def deterministic_row_ids(vocab_size: int, count: int, phase: float) -> tuple[int, ...]:
    if count <= 0 or count > vocab_size:
        raise ValueError("invalid alignment sample count")
    values = {
        min(vocab_size - 1, math.floor((index + phase) * vocab_size / count))
        for index in range(count)
    }
    if len(values) != count:
        # This only occurs for very small synthetic vocabularies.
        values = set(range(count))
    return tuple(sorted(values))


def _ridge_dual_solution(
    x: torch.Tensor, y: torch.Tensor, ridge_relative: float
) -> tuple[torch.Tensor, float]:
    # Solve argmin_P ||xP-y||^2 + lambda||P||^2 using the sample-space dual.
    gram = x @ x.T
    scale = float(torch.trace(gram).item() / max(1, gram.shape[0]))
    ridge_absolute = max(float(ridge_relative) * scale, 1e-8)
    regularized = gram + ridge_absolute * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    coefficients = torch.linalg.solve(regularized, y)
    return x.T @ coefficients, ridge_absolute


def fit_embedding_alignment(
    target_embedding: torch.Tensor,
    donor_embedding: torch.Tensor,
    *,
    sample_count: int = 512,
    heldout_count: int = 128,
    ridge_relative: float = 1e-3,
    device: str = "cpu",
) -> Alignment:
    if target_embedding.ndim != 2 or donor_embedding.ndim != 2:
        raise CheckpointError("embedding alignment requires matrices")
    if target_embedding.shape[0] != donor_embedding.shape[0]:
        raise CheckpointError("embedding vocabularies differ")
    vocab_size = int(target_embedding.shape[0])
    sample_ids = deterministic_row_ids(vocab_size, min(sample_count, vocab_size), 0.5)
    heldout_candidates = deterministic_row_ids(
        vocab_size, min(max(heldout_count * 4, heldout_count), vocab_size), 0.0
    )
    heldout_ids = tuple(
        value for value in heldout_candidates if value not in set(sample_ids)
    )[: min(heldout_count, max(0, vocab_size - len(sample_ids)))]
    compute = torch.device(device)
    sample_index = torch.tensor(sample_ids, dtype=torch.long)
    target_sample = target_embedding.index_select(0, sample_index).float().to(compute)
    donor_sample = donor_embedding.index_select(0, sample_index).float().to(compute)
    donor_to_target, donor_ridge = _ridge_dual_solution(
        donor_sample, target_sample, ridge_relative
    )
    target_to_donor, target_ridge = _ridge_dual_solution(
        target_sample, donor_sample, ridge_relative
    )
    target_from_donor = donor_to_target.T.contiguous()
    donor_from_target = target_to_donor.T.contiguous()

    with torch.no_grad():
        fitted = donor_sample @ donor_to_target
        sample_relative_rmse = float(
            torch.linalg.vector_norm(fitted - target_sample).item()
            / max(torch.linalg.vector_norm(target_sample).item(), 1e-30)
        )
        if heldout_ids:
            heldout_index = torch.tensor(heldout_ids, dtype=torch.long)
            target_heldout = target_embedding.index_select(0, heldout_index).float().to(compute)
            donor_heldout = donor_embedding.index_select(0, heldout_index).float().to(compute)
            heldout_projected = donor_heldout @ donor_to_target
            heldout_relative_rmse = float(
                torch.linalg.vector_norm(heldout_projected - target_heldout).item()
                / max(torch.linalg.vector_norm(target_heldout).item(), 1e-30)
            )
            cosine = torch.nn.functional.cosine_similarity(
                heldout_projected, target_heldout, dim=1
            )
            heldout_mean_cosine = float(cosine.mean().item())
        else:
            heldout_relative_rmse = None
            heldout_mean_cosine = None
        cycle = target_from_donor @ donor_from_target
        identity = torch.eye(cycle.shape[0], dtype=cycle.dtype, device=cycle.device)
        target_cycle_relative_error = float(
            torch.linalg.vector_norm(cycle - identity).item()
            / torch.linalg.vector_norm(identity).item()
        )
    return Alignment(
        target_from_donor=target_from_donor.cpu(),
        donor_from_target=donor_from_target.cpu(),
        sample_ids=sample_ids,
        heldout_ids=heldout_ids,
        metrics={
            "sample_relative_rmse": sample_relative_rmse,
            "heldout_relative_rmse": heldout_relative_rmse,
            "heldout_mean_cosine": heldout_mean_cosine,
            "target_cycle_relative_error": target_cycle_relative_error,
        },
        ridge_absolute_donor=donor_ridge,
        ridge_absolute_target=target_ridge,
        method="ridge_dual",
        scale=1.0,
    )


def fit_embedding_procrustes(
    target_embedding: torch.Tensor,
    donor_embedding: torch.Tensor,
    *,
    sample_count: int = 4096,
    heldout_count: int = 512,
    device: str = "cpu",
) -> Alignment:
    if target_embedding.ndim != 2 or donor_embedding.ndim != 2:
        raise CheckpointError("embedding alignment requires matrices")
    if target_embedding.shape[0] != donor_embedding.shape[0]:
        raise CheckpointError("embedding vocabularies differ")
    vocab_size = int(target_embedding.shape[0])
    sample_ids = deterministic_row_ids(vocab_size, min(sample_count, vocab_size), 0.5)
    heldout_candidates = deterministic_row_ids(
        vocab_size, min(max(heldout_count * 4, heldout_count), vocab_size), 0.0
    )
    sample_set = set(sample_ids)
    heldout_ids = tuple(value for value in heldout_candidates if value not in sample_set)[
        : min(heldout_count, max(0, vocab_size - len(sample_ids)))
    ]
    compute = torch.device(device)
    sample_index = torch.tensor(sample_ids, dtype=torch.long)
    target_sample = target_embedding.index_select(0, sample_index).float().to(compute)
    donor_sample = donor_embedding.index_select(0, sample_index).float().to(compute)
    cross_covariance = donor_sample.T @ target_sample
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    donor_to_target_orthogonal = u @ vh
    unscaled = donor_sample @ donor_to_target_orthogonal
    denominator = float(torch.dot(unscaled.reshape(-1), unscaled.reshape(-1)).item())
    scale = float(
        torch.dot(unscaled.reshape(-1), target_sample.reshape(-1)).item()
        / max(denominator, 1e-30)
    )
    donor_to_target = donor_to_target_orthogonal * scale
    target_from_donor = donor_to_target.T.contiguous()
    donor_from_target = (donor_to_target_orthogonal / max(abs(scale), 1e-30)).contiguous()
    fitted = donor_sample @ donor_to_target
    sample_relative_rmse = float(
        torch.linalg.vector_norm(fitted - target_sample).item()
        / max(torch.linalg.vector_norm(target_sample).item(), 1e-30)
    )
    if heldout_ids:
        heldout_index = torch.tensor(heldout_ids, dtype=torch.long)
        target_heldout = target_embedding.index_select(0, heldout_index).float().to(compute)
        donor_heldout = donor_embedding.index_select(0, heldout_index).float().to(compute)
        heldout_projected = donor_heldout @ donor_to_target
        heldout_relative_rmse = float(
            torch.linalg.vector_norm(heldout_projected - target_heldout).item()
            / max(torch.linalg.vector_norm(target_heldout).item(), 1e-30)
        )
        heldout_mean_cosine = float(
            torch.nn.functional.cosine_similarity(
                heldout_projected, target_heldout, dim=1
            ).mean().item()
        )
    else:
        heldout_relative_rmse = None
        heldout_mean_cosine = None
    cycle = target_from_donor @ donor_from_target
    identity = torch.eye(cycle.shape[0], dtype=cycle.dtype, device=cycle.device)
    target_cycle_relative_error = float(
        torch.linalg.vector_norm(cycle - identity).item()
        / torch.linalg.vector_norm(identity).item()
    )
    return Alignment(
        target_from_donor=target_from_donor.cpu(),
        donor_from_target=donor_from_target.cpu(),
        sample_ids=sample_ids,
        heldout_ids=heldout_ids,
        metrics={
            "sample_relative_rmse": sample_relative_rmse,
            "heldout_relative_rmse": heldout_relative_rmse,
            "heldout_mean_cosine": heldout_mean_cosine,
            "target_cycle_relative_error": target_cycle_relative_error,
        },
        method="scaled_rectangular_orthogonal_procrustes",
        ridge_absolute_donor=None,
        ridge_absolute_target=None,
        scale=scale,
    )


def _depth_mapping(strategy: str, target_layers: int, donor_layers: int) -> list[int]:
    if strategy == "D1":
        return list(range(target_layers))
    if strategy == "D2":
        return uniform_depth_map(target_layers, donor_layers)
    if strategy == "D3":
        return center_preserving_depth_map(target_layers, donor_layers)
    raise CheckpointError(f"unsupported depth strategy: {strategy}")


def _mm(left: torch.Tensor, right: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (left.float().to(device) @ right.float().to(device)).cpu()


def _project_embedding(
    donor_embedding: torch.Tensor,
    donor_to_target: torch.Tensor,
    device: torch.device,
    chunk_rows: int = 2048,
) -> torch.Tensor:
    output = torch.empty(
        donor_embedding.shape[0], donor_to_target.shape[1], dtype=torch.float32
    )
    right = donor_to_target.to(device)
    for start in range(0, donor_embedding.shape[0], chunk_rows):
        end = min(donor_embedding.shape[0], start + chunk_rows)
        output[start:end] = donor_embedding[start:end].float().to(device) @ right
    return output


def _prepare_output(
    destination: Path, sources: list[Path], overwrite: bool
) -> tuple[Path, Path]:
    destination = validate_safe_destination(destination, sources)
    staging = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    validate_safe_destination(staging, sources)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {destination}")
        shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
    if staging.exists():
        shutil.rmtree(staging) if staging.is_dir() else staging.unlink()
    staging.mkdir(parents=True)
    return destination, staging


def project_aligned_checkpoint(
    *,
    template_path: Path,
    reference_path: Path,
    donor_path: Path,
    destination: Path,
    depth_strategy: str = "D2",
    output_dtype: str = "bfloat16",
    sample_count: int = 512,
    heldout_count: int = 128,
    ridge_relative: float = 1e-3,
    alignment_method: str = "ridge",
    compute_device: str = "cpu",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if output_dtype not in dtype_map:
        raise CheckpointError(f"unsupported output dtype: {output_dtype}")
    if compute_device.startswith("cuda") and not torch.cuda.is_available():
        raise CheckpointError("CUDA alignment requested but unavailable")
    if compute_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(compute_device)
    template_path = template_path.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve()
    donor_path = donor_path.expanduser().resolve()
    destination, staging = _prepare_output(
        destination, [template_path, reference_path, donor_path], overwrite
    )
    try:
        with CheckpointReader(template_path) as template, CheckpointReader(
            reference_path
        ) as reference, CheckpointReader(donor_path) as donor:
            target_config = reference.config()
            donor_config = donor.config()
            if target_config["vocab_size"] != donor_config["vocab_size"]:
                raise CheckpointError("vocabulary mismatch")
            if target_config["head_dim"] != donor_config["head_dim"]:
                raise CheckpointError("head dimension mismatch")
            mapping = _depth_mapping(
                depth_strategy,
                int(target_config["num_hidden_layers"]),
                int(donor_config["num_hidden_layers"]),
            )
            target_embedding = reference.get_tensor("model.embed_tokens.weight")
            donor_embedding = donor.get_tensor("model.embed_tokens.weight")
            if alignment_method == "ridge":
                alignment = fit_embedding_alignment(
                    target_embedding,
                    donor_embedding,
                    sample_count=sample_count,
                    heldout_count=heldout_count,
                    ridge_relative=ridge_relative,
                    device=compute_device,
                )
                strategy = "P2"
                width_strategy = "W5"
            elif alignment_method == "procrustes":
                alignment = fit_embedding_procrustes(
                    target_embedding,
                    donor_embedding,
                    sample_count=sample_count,
                    heldout_count=heldout_count,
                    device=compute_device,
                )
                strategy = "P3"
                width_strategy = "W4"
            else:
                raise CheckpointError(
                    f"unsupported embedding alignment method: {alignment_method}"
                )
            a = alignment.target_from_donor.to(device)
            b = alignment.donor_from_target.to(device)
            head_dim = int(target_config["head_dim"])
            target_q_heads = int(target_config["num_attention_heads"])
            target_kv_heads = int(target_config["num_key_value_heads"])
            target_mlp = int(target_config["intermediate_size"])
            q_features = target_q_heads * head_dim
            kv_features = target_kv_heads * head_dim
            writer = StreamingSafetensorsWriter(staging, max_shard_bytes)
            records = []
            for ordinal, key in enumerate(template.keys, 1):
                descriptor = template.descriptor(key)
                if key == "model.embed_tokens.weight":
                    projected = _project_embedding(
                        donor_embedding, alignment.target_from_donor.T, device
                    )
                    source_key = "model.embed_tokens.weight"
                    rule = "donor_embedding @ fitted_donor_to_target"
                elif key == "model.norm.weight":
                    projected = reference.get_tensor(key).float()
                    source_key = key
                    rule = "target_reference_norm"
                elif key.startswith("model.layers."):
                    parts = key.split(".")
                    target_layer = int(parts[2])
                    suffix = ".".join(parts[3:])
                    donor_key = f"model.layers.{mapping[target_layer]}.{suffix}"
                    if suffix in {
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                    }:
                        projected = reference.get_tensor(key).float()
                        source_key = key
                        rule = "target_reference_norm"
                    else:
                        value = donor.get_tensor(donor_key)
                        source_key = donor_key
                        if suffix == "self_attn.q_proj.weight":
                            projected = _mm(value[:q_features], b, device)
                            rule = "selected_query_heads @ donor_from_target"
                        elif suffix in {
                            "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight",
                        }:
                            projected = _mm(value[:kv_features], b, device)
                            rule = "selected_kv_heads @ donor_from_target"
                        elif suffix == "self_attn.o_proj.weight":
                            projected = _mm(a, value[:, :q_features], device)
                            rule = "target_from_donor @ selected_query_columns"
                        elif suffix in {
                            "mlp.gate_proj.weight",
                            "mlp.up_proj.weight",
                        }:
                            projected = _mm(value[:target_mlp], b, device)
                            rule = "selected_mlp_rows @ donor_from_target"
                        elif suffix == "mlp.down_proj.weight":
                            projected = _mm(a, value[:, :target_mlp], device)
                            rule = "target_from_donor @ selected_mlp_columns"
                        elif tuple(value.shape) == descriptor.shape:
                            projected = value.float()
                            rule = "semantic_shape_copy"
                        else:
                            raise CheckpointError(
                                f"no aligned projection rule for {key}: {tuple(value.shape)}"
                            )
                else:
                    value = donor.get_tensor(key)
                    if tuple(value.shape) != descriptor.shape:
                        raise CheckpointError(f"no aligned non-layer rule for {key}")
                    projected = value.float()
                    source_key = key
                    rule = "semantic_shape_copy"
                if tuple(projected.shape) != descriptor.shape:
                    raise CheckpointError(
                        f"aligned shape mismatch {key}: {tuple(projected.shape)} vs {descriptor.shape}"
                    )
                if not torch.isfinite(projected).all().item():
                    raise CheckpointError(f"non-finite aligned projection: {key}")
                output = projected.to(dtype_map[output_dtype])
                writer.add(key, output)
                records.append(
                    {
                        "target_key": key,
                        "source_key": source_key,
                        "rule": rule,
                        "shape": list(descriptor.shape),
                        "output_dtype": output_dtype,
                    }
                )
                del projected, output
                if ordinal % 20 == 0:
                    gc.collect()
            output_index = writer.finalize()
            sidecars = copy_sidecars(reference_path, staging)

        matrices_path = staging / "alignment_matrices.safetensors"
        save_file(
            {
                "target_from_donor": alignment.target_from_donor.contiguous(),
                "donor_from_target": alignment.donor_from_target.contiguous(),
            },
            str(matrices_path),
            metadata={"format": "pt"},
        )
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "embedding_anchored_linear_alignment_projection",
            "strategy": strategy,
            "width_strategy": width_strategy,
            "depth_strategy": depth_strategy,
            "layer_mapping": mapping,
            "template_path": str(template_path),
            "reference_path": str(reference_path),
            "donor_path": str(donor_path),
            "source_fingerprints": source_fingerprints or {},
            "calibration": {
                "source": "shared-vocabulary embedding rows only",
                "benchmark_examples_used": 0,
                "sample_count": len(alignment.sample_ids),
                "sample_ids_sha256": canonical_sha256(alignment.sample_ids),
                "heldout_count": len(alignment.heldout_ids),
                "heldout_ids_sha256": canonical_sha256(alignment.heldout_ids),
                "ridge_relative": ridge_relative,
                "ridge_absolute_donor": alignment.ridge_absolute_donor,
                "ridge_absolute_target": alignment.ridge_absolute_target,
                "alignment_method": alignment.method,
                "scale": alignment.scale,
                "metrics": alignment.metrics,
            },
            "residual_transform": "target_from_donor @ W_donor @ donor_from_target",
            "attention_head_selection": "prefix complete heads",
            "mlp_selection": "prefix neurons before residual-basis transform",
            "normalization_policy": "official_qwen3_0.6b target-basis norms",
            "arithmetic_dtype": "float32",
            "compute_device": compute_device,
            "tf32": False,
            "output_dtype": output_dtype,
            "output_index": output_index,
            "sidecars": sidecars,
            "tensor_count": len(records),
            "tensor_map_sha256": canonical_sha256(records),
            "alignment_matrices": {
                "bytes": matrices_path.stat().st_size,
                "sha256": sha256_file(matrices_path),
            },
        }
        atomic_write_json(staging / "tensor_projection_map.json", records)
        atomic_write_json(staging / "projection_manifest.json", manifest)
        os.replace(staging, destination)
        manifest["output_path"] = str(destination)
        manifest["output_fingerprint"] = checkpoint_fingerprint(destination)
        atomic_write_json(destination / "projection_manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
