from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .architecture import uniform_depth_map
from .checkpoint import (
    CheckpointError,
    CheckpointReader,
    StreamingSafetensorsWriter,
    checkpoint_fingerprint,
    copy_sidecars,
    validate_safe_destination,
)
from .hashing import atomic_write_json, canonical_sha256


def _seed(key: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _low_rank_spectrum(
    matrix: torch.Tensor, rank: int, *, seed: int, iterations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = min(max(rank + 4, rank), min(matrix.shape))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(matrix.float(), q=q, niter=iterations)
    order = torch.argsort(s, descending=True)
    return u[:, order[:rank]], s[order[:rank]], v[:, order[:rank]]


def spectral_profile_transfer(
    target: torch.Tensor,
    donor: torch.Tensor,
    *,
    rank: int = 8,
    strength: float = 1.0,
    seed_key: str = "tensor",
    iterations: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if target.ndim != 2 or donor.ndim != 2:
        raise CheckpointError("spectral transfer requires matrices")
    effective_rank = min(rank, min(target.shape), min(donor.shape))
    if effective_rank <= 0 or not 0.0 <= strength <= 1.0:
        raise CheckpointError("invalid spectral transfer parameters")
    target_value = target.float()
    donor_value = donor.float()
    target_u, target_s, target_v = _low_rank_spectrum(
        target_value,
        effective_rank,
        seed=_seed(seed_key, "target"),
        iterations=iterations,
    )
    _, donor_s, _ = _low_rank_spectrum(
        donor_value,
        effective_rank,
        seed=_seed(seed_key, "donor"),
        iterations=iterations,
    )
    target_fro = torch.linalg.vector_norm(target_value)
    donor_fro = torch.linalg.vector_norm(donor_value)
    donor_normalized = donor_s / torch.clamp(donor_fro, min=1e-30)
    desired = donor_normalized * target_fro
    desired = target_s.mul(1.0 - strength).add(desired, alpha=strength)
    desired_energy = torch.dot(desired, desired)
    total_energy = target_fro.square()
    if desired_energy >= total_energy:
        desired = desired * torch.sqrt(total_energy * 0.95 / desired_energy)
        desired_energy = torch.dot(desired, desired)
    target_top_energy = torch.dot(target_s, target_s)
    residual_denominator = torch.clamp(total_energy - target_top_energy, min=1e-30)
    residual_scale = torch.sqrt(
        torch.clamp(total_energy - desired_energy, min=0.0) / residual_denominator
    )
    correction = desired - residual_scale * target_s
    projected = target_value.mul(residual_scale).add(
        (target_u * correction.unsqueeze(0)) @ target_v.T
    )
    projected_fro = torch.linalg.vector_norm(projected)
    projected = projected * (target_fro / torch.clamp(projected_fro, min=1e-30))
    stats = {
        "rank": effective_rank,
        "strength": strength,
        "target_fro": float(target_fro.item()),
        "donor_fro": float(donor_fro.item()),
        "projected_fro": float(torch.linalg.vector_norm(projected).item()),
        "residual_scale": float(residual_scale.item()),
        "target_top_singular_values": target_s.tolist(),
        "donor_top_singular_values": donor_s.tolist(),
        "desired_target_basis_singular_values": desired.tolist(),
        "relative_change": float(
            torch.linalg.vector_norm(projected - target_value).item()
            / max(target_fro.item(), 1e-30)
        ),
    }
    return projected, stats


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


def project_spectral_checkpoint(
    *,
    template_path: Path,
    reference_path: Path,
    donor_path: Path,
    destination: Path,
    rank: int = 8,
    strength: float = 1.0,
    iterations: int = 1,
    output_dtype: str = "bfloat16",
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
    template_path = template_path.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve()
    donor_path = donor_path.expanduser().resolve()
    destination, staging = _prepare_output(
        destination, [template_path, reference_path, donor_path], overwrite
    )
    records = []
    try:
        with CheckpointReader(template_path) as template, CheckpointReader(
            reference_path
        ) as reference, CheckpointReader(donor_path) as donor:
            target_config = reference.config()
            donor_config = donor.config()
            mapping = uniform_depth_map(
                int(target_config["num_hidden_layers"]),
                int(donor_config["num_hidden_layers"]),
            )
            writer = StreamingSafetensorsWriter(staging, max_shard_bytes)
            for ordinal, key in enumerate(template.keys, 1):
                target_value = reference.get_tensor(key)
                descriptor = template.descriptor(key)
                if key.startswith("model.layers.") and target_value.ndim == 2:
                    parts = key.split(".")
                    donor_key = (
                        f"model.layers.{mapping[int(parts[2])]}."
                        + ".".join(parts[3:])
                    )
                    donor_value = donor.get_tensor(donor_key)
                    projected, stats = spectral_profile_transfer(
                        target_value,
                        donor_value,
                        rank=rank,
                        strength=strength,
                        seed_key=key,
                        iterations=iterations,
                    )
                    rule = "target_basis_donor_spectral_profile"
                else:
                    donor_key = None
                    projected = target_value.float()
                    stats = None
                    rule = "target_reference_preserved"
                if tuple(projected.shape) != descriptor.shape:
                    raise CheckpointError(f"spectral shape mismatch: {key}")
                if not torch.isfinite(projected).all().item():
                    raise CheckpointError(f"non-finite spectral tensor: {key}")
                output = projected.to(dtype_map[output_dtype])
                writer.add(key, output)
                records.append(
                    {
                        "target_key": key,
                        "donor_key": donor_key,
                        "rule": rule,
                        "shape": list(descriptor.shape),
                        "stats": stats,
                    }
                )
                del target_value, projected, output
                if ordinal % 10 == 0:
                    gc.collect()
            output_index = writer.finalize()
            sidecars = copy_sidecars(reference_path, staging)
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "target_basis_spectral_profile_projection",
            "strategy": "P4",
            "width_strategy": "W4",
            "depth_strategy": "D2",
            "layer_mapping": mapping,
            "rank": rank,
            "strength": strength,
            "iterations": iterations,
            "benchmark_examples_used": 0,
            "template_path": str(template_path),
            "reference_path": str(reference_path),
            "donor_path": str(donor_path),
            "source_fingerprints": source_fingerprints or {},
            "preserved": [
                "official Qwen3-0.6B embeddings and tied output head",
                "all 1D tensors and RMS norms",
                "target singular-vector bases",
                "target tensor Frobenius norms",
            ],
            "arithmetic_dtype": "float32",
            "output_dtype": output_dtype,
            "output_index": output_index,
            "sidecars": sidecars,
            "tensor_count": len(records),
            "tensor_map_sha256": canonical_sha256(records),
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
