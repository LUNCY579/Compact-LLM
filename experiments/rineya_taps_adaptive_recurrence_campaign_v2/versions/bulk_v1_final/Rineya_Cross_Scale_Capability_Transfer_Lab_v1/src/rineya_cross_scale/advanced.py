from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .checkpoint import CheckpointError
from .hashing import atomic_write_json, canonical_sha256
from .operations import _build_candidate


def trim_magnitude(delta: torch.Tensor, keep_fraction: float) -> tuple[torch.Tensor, int]:
    """Keep an exact deterministic top-magnitude fraction of one tensor."""
    if not 0.0 <= keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in [0, 1]")
    flat = delta.detach().float().reshape(-1)
    count = flat.numel()
    keep = int(math.ceil(count * keep_fraction))
    if keep == 0:
        return torch.zeros_like(delta, dtype=torch.float32), 0
    if keep == count:
        return delta.detach().float().clone(), count
    magnitude = flat.abs()
    threshold = torch.kthvalue(magnitude, count - keep + 1).values
    mask = magnitude > threshold
    selected = int(mask.sum().item())
    needed = keep - selected
    if needed > 0:
        ties = torch.nonzero(magnitude == threshold, as_tuple=False).reshape(-1)
        mask[ties[:needed]] = True
    trimmed = torch.where(mask, flat, torch.zeros_like(flat)).reshape(delta.shape)
    return trimmed, int(mask.sum().item())


def ties_consensus(
    deltas: list[torch.Tensor], keep_fraction: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not deltas:
        raise ValueError("TIES requires at least one delta")
    shape = deltas[0].shape
    if any(delta.shape != shape for delta in deltas):
        raise ValueError("TIES delta shapes differ")
    trimmed = []
    retained = []
    for delta in deltas:
        value, count = trim_magnitude(delta, keep_fraction)
        trimmed.append(value)
        retained.append(count)
    stack = torch.stack(trimmed)
    elected = torch.sign(stack.sum(dim=0))
    agreeing = (torch.sign(stack) == elected.unsqueeze(0)) & (
        elected.unsqueeze(0) != 0
    )
    numerator = torch.where(agreeing, stack, torch.zeros_like(stack)).sum(dim=0)
    denominator = agreeing.sum(dim=0)
    result = torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=1),
        torch.zeros_like(numerator),
    )
    nonzero_elected = elected != 0
    conflict = nonzero_elected & (agreeing.sum(dim=0) < len(deltas))
    stats = {
        "keep_fraction_requested": keep_fraction,
        "retained_coordinates_per_delta": retained,
        "coordinate_count": result.numel(),
        "elected_positive": int((elected > 0).sum().item()),
        "elected_negative": int((elected < 0).sum().item()),
        "elected_zero": int((elected == 0).sum().item()),
        "conflict_coordinates": int(conflict.sum().item()),
        "result_nonzero": int((result != 0).sum().item()),
    }
    return result, stats


def _tensor_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def dare_delta(
    delta: torch.Tensor,
    *,
    drop_probability: float,
    seed: int,
    key: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must be in [0, 1)")
    value = delta.detach().float()
    generator = torch.Generator(device="cpu")
    resolved_seed = _tensor_seed(seed, key)
    generator.manual_seed(resolved_seed)
    keep = torch.rand(value.shape, generator=generator) >= drop_probability
    result = value * keep / (1.0 - drop_probability)
    return result, {
        "seed": seed,
        "resolved_tensor_seed": resolved_seed,
        "drop_probability": drop_probability,
        "coordinate_count": value.numel(),
        "kept": int(keep.sum().item()),
        "actual_keep_fraction": float(keep.sum().item() / max(1, value.numel())),
        "rescale": 1.0 / (1.0 - drop_probability),
    }


def dare_ties_consensus(
    deltas: list[torch.Tensor],
    *,
    drop_probability: float,
    seed: int,
    key: str,
    keep_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dared = []
    dare_stats = []
    for index, delta in enumerate(deltas):
        value, stats = dare_delta(
            delta,
            drop_probability=drop_probability,
            seed=seed + index * 1_000_003,
            key=key,
        )
        dared.append(value)
        dare_stats.append(stats)
    result, ties_stats = ties_consensus(dared, keep_fraction)
    return result, {"dare": dare_stats, "ties": ties_stats}


def fisher_weighted_parameters(
    base: torch.Tensor,
    donor: torch.Tensor,
    fisher_base: torch.Tensor,
    fisher_donor: torch.Tensor,
    *,
    donor_lambda: float,
    epsilon: float = 1e-12,
    fisher_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if donor_lambda < 0 or epsilon <= 0:
        raise ValueError("invalid Fisher merge parameters")
    values = [base, donor, fisher_base, fisher_donor]
    if any(value.shape != base.shape for value in values):
        raise ValueError("Fisher tensor shapes differ")
    fb = torch.clamp(fisher_base.float(), min=0.0)
    fd = torch.clamp(fisher_donor.float(), min=0.0)
    if fisher_clip is not None:
        if fisher_clip <= 0:
            raise ValueError("fisher_clip must be positive")
        fb = torch.clamp(fb, max=fisher_clip)
        fd = torch.clamp(fd, max=fisher_clip)
    weighted_donor = fd * donor_lambda
    denominator = fb + weighted_donor
    numerator = fb * base.float() + weighted_donor * donor.float()
    result = torch.where(
        denominator > epsilon,
        numerator / denominator.clamp(min=epsilon),
        base.float(),
    )
    return result, {
        "donor_lambda": donor_lambda,
        "epsilon": epsilon,
        "fisher_clip": fisher_clip,
        "zero_or_small_denominator": int((denominator <= epsilon).sum().item()),
        "base_fisher_min": float(fb.min().item()),
        "base_fisher_max": float(fb.max().item()),
        "donor_fisher_min": float(fd.min().item()),
        "donor_fisher_max": float(fd.max().item()),
    }


def _attach_advanced_stats(
    manifest: dict[str, Any], tensor_stats: dict[str, Any]
) -> dict[str, Any]:
    output = Path(manifest["output_path"])
    stats_path = output / "advanced_tensor_stats.json"
    atomic_write_json(stats_path, tensor_stats)
    manifest_path = output / "candidate_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["advanced_tensor_stats"] = {
        "file": stats_path.name,
        "sha256": canonical_sha256(tensor_stats),
        "tensor_count": len(tensor_stats),
    }
    atomic_write_json(manifest_path, payload)
    manifest.update(payload)
    return manifest


def merge_ties_candidate(
    *,
    base_path: Path,
    delta_paths: list[Path],
    destination: Path,
    alpha: float,
    keep_fraction: float,
    output_dtype: str = "float16",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensor_stats: dict[str, Any] = {}

    def operation(key: str, base: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        consensus, stats = ties_consensus(values, keep_fraction)
        tensor_stats[key] = stats
        return base.add(consensus, alpha=alpha)

    manifest = _build_candidate(
        base_path=base_path,
        source_paths=delta_paths,
        destination=destination,
        operation="ties_task_vector",
        coefficient=alpha,
        tensor_operation=operation,
        output_dtype=output_dtype,
        max_shard_bytes=max_shard_bytes,
        overwrite=overwrite,
        operation_parameters={"keep_fraction": keep_fraction},
        source_fingerprints=source_fingerprints,
    )
    return _attach_advanced_stats(manifest, tensor_stats)


def merge_dare_candidate(
    *,
    base_path: Path,
    delta_path: Path,
    destination: Path,
    alpha: float,
    drop_probability: float,
    seed: int,
    output_dtype: str = "float16",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensor_stats: dict[str, Any] = {}

    def operation(key: str, base: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        delta, stats = dare_delta(
            values[0], drop_probability=drop_probability, seed=seed, key=key
        )
        tensor_stats[key] = stats
        return base.add(delta, alpha=alpha)

    manifest = _build_candidate(
        base_path=base_path,
        source_paths=[delta_path],
        destination=destination,
        operation="dare_task_vector",
        coefficient=alpha,
        tensor_operation=operation,
        output_dtype=output_dtype,
        max_shard_bytes=max_shard_bytes,
        overwrite=overwrite,
        operation_parameters={"drop_probability": drop_probability, "seed": seed},
        source_fingerprints=source_fingerprints,
    )
    return _attach_advanced_stats(manifest, tensor_stats)


def merge_dare_ties_candidate(
    *,
    base_path: Path,
    delta_paths: list[Path],
    destination: Path,
    alpha: float,
    drop_probability: float,
    seed: int,
    keep_fraction: float,
    output_dtype: str = "float16",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensor_stats: dict[str, Any] = {}

    def operation(key: str, base: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        delta, stats = dare_ties_consensus(
            values,
            drop_probability=drop_probability,
            seed=seed,
            key=key,
            keep_fraction=keep_fraction,
        )
        tensor_stats[key] = stats
        return base.add(delta, alpha=alpha)

    manifest = _build_candidate(
        base_path=base_path,
        source_paths=delta_paths,
        destination=destination,
        operation="dare_ties_task_vector",
        coefficient=alpha,
        tensor_operation=operation,
        output_dtype=output_dtype,
        max_shard_bytes=max_shard_bytes,
        overwrite=overwrite,
        operation_parameters={
            "drop_probability": drop_probability,
            "seed": seed,
            "keep_fraction": keep_fraction,
        },
        source_fingerprints=source_fingerprints,
    )
    return _attach_advanced_stats(manifest, tensor_stats)


def merge_fisher_candidate(
    *,
    base_path: Path,
    donor_path: Path,
    fisher_base_path: Path,
    fisher_donor_path: Path,
    destination: Path,
    donor_lambda: float,
    epsilon: float = 1e-12,
    fisher_clip: float | None = None,
    output_dtype: str = "float16",
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
    source_fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensor_stats: dict[str, Any] = {}

    def operation(key: str, base: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        merged, stats = fisher_weighted_parameters(
            base,
            values[0],
            values[1],
            values[2],
            donor_lambda=donor_lambda,
            epsilon=epsilon,
            fisher_clip=fisher_clip,
        )
        tensor_stats[key] = stats
        return merged

    manifest = _build_candidate(
        base_path=base_path,
        source_paths=[donor_path, fisher_base_path, fisher_donor_path],
        destination=destination,
        operation="fisher_weighted_merge",
        coefficient=donor_lambda,
        tensor_operation=operation,
        output_dtype=output_dtype,
        max_shard_bytes=max_shard_bytes,
        overwrite=overwrite,
        operation_parameters={
            "donor_lambda": donor_lambda,
            "epsilon": epsilon,
            "fisher_clip": fisher_clip,
        },
        source_fingerprints=source_fingerprints,
    )
    return _attach_advanced_stats(manifest, tensor_stats)
