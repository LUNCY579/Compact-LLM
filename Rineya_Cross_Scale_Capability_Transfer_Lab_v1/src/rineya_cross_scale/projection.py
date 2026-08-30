from __future__ import annotations

import gc
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .architecture import center_preserving_depth_map, uniform_depth_map
from .checkpoint import (
    CheckpointError,
    CheckpointReader,
    StreamingSafetensorsWriter,
    checkpoint_fingerprint,
    copy_sidecars,
    validate_safe_destination,
)
from .hashing import atomic_write_json, canonical_sha256


LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ProjectionPlan:
    strategy: str
    depth_strategy: str
    width_strategy: str
    target_layers: int
    donor_layers: int
    layer_mapping: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    query_head_indices: tuple[int, ...]
    kv_head_indices: tuple[int, ...]
    mlp_indices: tuple[int, ...]
    head_dim: int
    target_tied_embeddings: bool
    donor_tied_embeddings: bool
    tied_projection: str

    @property
    def id(self) -> str:
        return canonical_sha256(asdict(self))[:16]


def _depth_map(strategy: str, target_layers: int, donor_layers: int) -> list[int]:
    if donor_layers < target_layers:
        raise CheckpointError(
            f"donor depth {donor_layers} is smaller than target depth {target_layers}"
        )
    if strategy == "D1":
        return list(range(target_layers))
    if strategy == "D2":
        return uniform_depth_map(target_layers, donor_layers)
    if strategy == "D3":
        return center_preserving_depth_map(target_layers, donor_layers)
    raise CheckpointError(f"unsupported depth strategy: {strategy}")


def build_projection_plan(
    target_config: dict[str, Any],
    donor_config: dict[str, Any],
    *,
    depth_strategy: str,
    width_strategy: str,
    tied_projection: str = "embedding",
) -> ProjectionPlan:
    if target_config.get("model_type") != donor_config.get("model_type"):
        raise CheckpointError("model_type mismatch")
    if target_config.get("vocab_size") != donor_config.get("vocab_size"):
        raise CheckpointError("vocabulary size mismatch")
    target_head_dim = int(target_config["head_dim"])
    donor_head_dim = int(donor_config["head_dim"])
    if target_head_dim != donor_head_dim:
        raise CheckpointError(
            f"head dimension mismatch: {target_head_dim} vs {donor_head_dim}"
        )
    target_hidden = int(target_config["hidden_size"])
    donor_hidden = int(donor_config["hidden_size"])
    target_mlp = int(target_config["intermediate_size"])
    donor_mlp = int(donor_config["intermediate_size"])
    target_q_heads = int(target_config["num_attention_heads"])
    donor_q_heads = int(donor_config["num_attention_heads"])
    target_kv_heads = int(target_config["num_key_value_heads"])
    donor_kv_heads = int(donor_config["num_key_value_heads"])
    if donor_hidden < target_hidden or donor_mlp < target_mlp:
        raise CheckpointError("donor width is smaller than target width")
    if donor_q_heads < target_q_heads or donor_kv_heads < target_kv_heads:
        raise CheckpointError("donor has fewer complete attention heads than target")
    if width_strategy not in {"W1", "W2"}:
        raise CheckpointError(
            f"width strategy {width_strategy} is not implemented by deterministic P1"
        )
    allowed_tied_projection = {"embedding", "output_head", "mean"}
    if tied_projection not in allowed_tied_projection:
        raise CheckpointError(
            f"unsupported tied projection policy: {tied_projection}"
        )
    target_tied = bool(target_config.get("tie_word_embeddings"))
    donor_tied = bool(donor_config.get("tie_word_embeddings"))
    if not target_tied and tied_projection != "embedding":
        raise CheckpointError(
            "output_head/mean collapse policies require a tied target"
        )
    if donor_tied and tied_projection != "embedding":
        raise CheckpointError(
            "output_head/mean policies are redundant for a tied donor; use embedding"
        )
    target_layers = int(target_config["num_hidden_layers"])
    donor_layers = int(donor_config["num_hidden_layers"])
    return ProjectionPlan(
        strategy="P1",
        depth_strategy=depth_strategy,
        width_strategy=width_strategy,
        target_layers=target_layers,
        donor_layers=donor_layers,
        layer_mapping=tuple(_depth_map(depth_strategy, target_layers, donor_layers)),
        hidden_indices=tuple(range(target_hidden)),
        query_head_indices=tuple(range(target_q_heads)),
        kv_head_indices=tuple(range(target_kv_heads)),
        mlp_indices=tuple(range(target_mlp)),
        head_dim=target_head_dim,
        target_tied_embeddings=target_tied,
        donor_tied_embeddings=donor_tied,
        tied_projection=tied_projection,
    )


def donor_key_for_target(target_key: str, plan: ProjectionPlan) -> str:
    if (
        target_key in {"model.embed_tokens.weight", "lm_head.weight"}
        and plan.target_tied_embeddings
    ):
        if not plan.donor_tied_embeddings and plan.tied_projection == "output_head":
            return "lm_head.weight"
        return "model.embed_tokens.weight"
    match = LAYER_KEY.match(target_key)
    if not match:
        return target_key
    target_layer = int(match.group(1))
    if target_layer >= len(plan.layer_mapping):
        raise CheckpointError(f"target layer outside projection plan: {target_key}")
    donor_layer = plan.layer_mapping[target_layer]
    return f"model.layers.{donor_layer}.{match.group(2)}"


def _flat_head_indices(heads: tuple[int, ...], head_dim: int, device: torch.device) -> torch.Tensor:
    values = [head * head_dim + offset for head in heads for offset in range(head_dim)]
    return torch.tensor(values, dtype=torch.long, device=device)


def _indices(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def project_tensor(
    target_key: str,
    donor_tensor: torch.Tensor,
    target_shape: tuple[int, ...],
    plan: ProjectionPlan,
) -> torch.Tensor:
    device = donor_tensor.device
    hidden = _indices(plan.hidden_indices, device)
    mlp = _indices(plan.mlp_indices, device)
    q_features = _flat_head_indices(plan.query_head_indices, plan.head_dim, device)
    kv_features = _flat_head_indices(plan.kv_head_indices, plan.head_dim, device)
    match = LAYER_KEY.match(target_key)
    suffix = match.group(2) if match else target_key

    if suffix in {"model.embed_tokens.weight", "lm_head.weight"}:
        result = donor_tensor.index_select(1, hidden)
    elif suffix == "model.norm.weight" or suffix.endswith("layernorm.weight"):
        result = donor_tensor.index_select(0, hidden)
    elif suffix.endswith("self_attn.q_proj.weight"):
        result = donor_tensor.index_select(0, q_features).index_select(1, hidden)
    elif suffix.endswith("self_attn.k_proj.weight") or suffix.endswith(
        "self_attn.v_proj.weight"
    ):
        result = donor_tensor.index_select(0, kv_features).index_select(1, hidden)
    elif suffix.endswith("self_attn.o_proj.weight"):
        result = donor_tensor.index_select(0, hidden).index_select(1, q_features)
    elif suffix.endswith("mlp.gate_proj.weight") or suffix.endswith(
        "mlp.up_proj.weight"
    ):
        result = donor_tensor.index_select(0, mlp).index_select(1, hidden)
    elif suffix.endswith("mlp.down_proj.weight"):
        result = donor_tensor.index_select(0, hidden).index_select(1, mlp)
    elif suffix.endswith("self_attn.q_proj.bias"):
        result = donor_tensor.index_select(0, q_features)
    elif suffix.endswith("self_attn.k_proj.bias") or suffix.endswith(
        "self_attn.v_proj.bias"
    ):
        result = donor_tensor.index_select(0, kv_features)
    elif suffix.endswith("self_attn.o_proj.bias") or suffix.endswith(
        "mlp.down_proj.bias"
    ):
        result = donor_tensor.index_select(0, hidden)
    elif suffix.endswith("mlp.gate_proj.bias") or suffix.endswith("mlp.up_proj.bias"):
        result = donor_tensor.index_select(0, mlp)
    elif tuple(donor_tensor.shape) == target_shape:
        result = donor_tensor
    else:
        raise CheckpointError(
            f"no semantic projection rule for {target_key}: "
            f"donor={tuple(donor_tensor.shape)}, target={target_shape}"
        )
    if tuple(result.shape) != target_shape:
        raise CheckpointError(
            f"projected shape mismatch for {target_key}: "
            f"got={tuple(result.shape)}, expected={target_shape}"
        )
    return result


def _prepare_staging(
    destination: Path, sources: list[Path], *, overwrite: bool
) -> tuple[Path, Path]:
    destination = validate_safe_destination(destination, sources)
    staging = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    validate_safe_destination(staging, sources)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"destination already exists: {destination}")
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if staging.exists():
        if staging.is_dir():
            shutil.rmtree(staging)
        else:
            staging.unlink()
    staging.mkdir(parents=True)
    return destination, staging


def project_checkpoint(
    *,
    template_path: Path,
    donor_path: Path,
    destination: Path,
    metadata_template_path: Path | None = None,
    depth_strategy: str = "D2",
    width_strategy: str = "W2",
    tied_projection: str = "embedding",
    output_dtype: str = "bfloat16",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dtype not in DTYPES:
        raise CheckpointError(f"unsupported output dtype: {output_dtype}")
    template_path = template_path.expanduser().resolve()
    donor_path = donor_path.expanduser().resolve()
    metadata_template_path = (
        metadata_template_path.expanduser().resolve()
        if metadata_template_path is not None
        else template_path
    )
    destination, staging = _prepare_staging(
        destination,
        [template_path, donor_path, metadata_template_path],
        overwrite=overwrite,
    )
    tensor_records: list[dict[str, Any]] = []
    try:
        with CheckpointReader(template_path) as template, CheckpointReader(
            donor_path
        ) as donor:
            if template.is_quantized() or donor.is_quantized():
                raise CheckpointError("projection requires floating-point checkpoints")
            plan = build_projection_plan(
                template.config(),
                donor.config(),
                depth_strategy=depth_strategy,
                width_strategy=width_strategy,
                tied_projection=tied_projection,
            )
            writer = StreamingSafetensorsWriter(staging, max_shard_bytes)
            for ordinal, target_key in enumerate(template.keys, 1):
                target_descriptor = template.descriptor(target_key)
                donor_key = donor_key_for_target(target_key, plan)
                if donor_key not in donor.weight_map:
                    raise CheckpointError(
                        f"donor lacks tensor for {target_key}: expected {donor_key}"
                    )
                donor_tensor = donor.get_tensor(donor_key)
                if not donor_tensor.is_floating_point():
                    if tuple(donor_tensor.shape) != target_descriptor.shape:
                        raise CheckpointError(
                            f"non-floating shape mismatch for {target_key}"
                        )
                    projected = donor_tensor
                else:
                    projected = project_tensor(
                        target_key,
                        donor_tensor.to(torch.float32),
                        target_descriptor.shape,
                        plan,
                    )
                    if (
                        target_key in {"model.embed_tokens.weight", "lm_head.weight"}
                        and plan.target_tied_embeddings
                        and not plan.donor_tied_embeddings
                        and plan.tied_projection == "mean"
                    ):
                        if "lm_head.weight" not in donor.weight_map:
                            raise CheckpointError(
                                "mean tied projection requires donor lm_head.weight"
                            )
                        donor_head = donor.get_tensor("lm_head.weight").float()
                        projected_head = project_tensor(
                            "lm_head.weight",
                            donor_head,
                            target_descriptor.shape,
                            plan,
                        )
                        projected = projected.add(projected_head).mul_(0.5)
                        del donor_head, projected_head
                    if not torch.isfinite(projected).all().item():
                        raise CheckpointError(f"non-finite projected tensor: {target_key}")
                    projected = projected.to(DTYPES[output_dtype])
                writer.add(target_key, projected)
                tensor_records.append(
                    {
                        "target_key": target_key,
                        "donor_key": donor_key,
                        "target_shape": list(target_descriptor.shape),
                        "donor_shape": list(donor_tensor.shape),
                        "donor_dtype": str(donor_tensor.dtype).removeprefix("torch."),
                        "output_dtype": str(projected.dtype).removeprefix("torch."),
                    }
                )
                del donor_tensor, projected
                if ordinal % 40 == 0:
                    gc.collect()
            output_index = writer.finalize()
            sidecars = copy_sidecars(metadata_template_path, staging)

        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "deterministic_cross_scale_projection",
            "plan": {**asdict(plan), "id": plan.id},
            "template_path": str(template_path),
            "metadata_template_path": str(metadata_template_path),
            "donor_path": str(donor_path),
            "source_fingerprints": source_fingerprints or {},
            "arithmetic_dtype": "float32",
            "output_dtype": output_dtype,
            "output_index": output_index,
            "sidecars": sidecars,
            "tensor_count": len(tensor_records),
            "tensor_map_sha256": canonical_sha256(tensor_records),
        }
        atomic_write_json(staging / "projection_manifest.json", manifest)
        atomic_write_json(staging / "tensor_projection_map.json", tensor_records)
        os.replace(staging, destination)
        manifest["output_path"] = str(destination)
        manifest["output_fingerprint"] = checkpoint_fingerprint(destination)
        atomic_write_json(destination / "projection_manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
