"""Paired functional and support metrics for the four registered arms."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def delta_cosine(delta_a: torch.Tensor, delta_b: torch.Tensor) -> float | None:
    """Cosine of two finite deltas; ``None`` is explicit for zero vectors."""
    a = delta_a.detach().float().reshape(-1)
    b = delta_b.detach().float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("delta tensors must have equal flattened size")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("delta cosine received nonfinite values")
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return _safe_ratio(float(torch.dot(a, b)), denominator)


def support_jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float | None:
    """J(A,B)=|intersection|/|union| for boolean supports."""
    a = mask_a.detach().bool().reshape(-1)
    b = mask_b.detach().bool().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("support masks must have equal flattened size")
    union = int((a | b).sum().item())
    return _safe_ratio(float((a & b).sum().item()), float(union))


def support_counts(mask: torch.Tensor) -> dict[str, int | float]:
    total = int(mask.numel())
    active = int(mask.detach().bool().sum().item())
    return {"active": active, "total": total,
            "density": float(active / total) if total else 0.0,
            "sparsity": float(1.0 - active / total) if total else 0.0}


def support_pair_metrics(mask_a: torch.Tensor, mask_b: torch.Tensor) -> dict[str, Any]:
    a = mask_a.detach().bool().reshape(-1)
    b = mask_b.detach().bool().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("support masks must have equal flattened size")
    intersection = int((a & b).sum().item())
    union = int((a | b).sum().item())
    return {
        "a": support_counts(a),
        "b": support_counts(b),
        "intersection": intersection,
        "union": union,
        "jaccard": _safe_ratio(float(intersection), float(union)),
    }


def layerwise_support_metrics(named_masks_a: Mapping[str, torch.Tensor],
                              named_masks_b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys_a = set(named_masks_a)
    keys_b = set(named_masks_b)
    if keys_a != keys_b:
        raise ValueError(f"layer/module support keys differ: {sorted(keys_a ^ keys_b)}")
    return {name: support_pair_metrics(named_masks_a[name], named_masks_b[name])
            for name in sorted(keys_a)}


def validate_paired_ids(arm_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    """Require exactly the same ordered evaluation IDs across every arm."""
    if not arm_records:
        raise ValueError("at least one arm is required")
    normalized: dict[str, list[str]] = {}
    for arm, rows in arm_records.items():
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate evaluation ID in arm {arm}")
        normalized[str(arm)] = ids
    first = next(iter(normalized.values()))
    for arm, ids in normalized.items():
        if ids != first:
            raise ValueError(f"paired evaluation IDs differ for arm {arm}")
    return first


def repair_damage_counts(base_correct: Sequence[bool],
                         adapted_correct: Sequence[bool]) -> dict[str, int]:
    if len(base_correct) != len(adapted_correct):
        raise ValueError("paired correctness vectors must have equal length")
    base = torch.tensor(list(base_correct), dtype=torch.bool)
    adapted = torch.tensor(list(adapted_correct), dtype=torch.bool)
    return {
        "n": int(base.numel()),
        "base_correct": int(base.sum().item()),
        "adapted_correct": int(adapted.sum().item()),
        "repair": int((~base & adapted).sum().item()),
        "damage": int((base & ~adapted).sum().item()),
        "preserved_correct": int((base & adapted).sum().item()),
        "preserved_incorrect": int((~base & ~adapted).sum().item()),
    }


def functional_interference(base_correct: Sequence[bool],
                            individual_a: Sequence[bool],
                            individual_b: Sequence[bool],
                            composed_ab: Sequence[bool],
                            composed_ba: Sequence[bool]) -> dict[str, Any]:
    """Summarize damage/repair for both acquisition orders on paired items."""
    validate = [base_correct, individual_a, individual_b, composed_ab, composed_ba]
    if len({len(values) for values in validate}) != 1:
        raise ValueError("all functional correctness vectors must be paired")
    return {
        "a": repair_damage_counts(base_correct, individual_a),
        "b": repair_damage_counts(base_correct, individual_b),
        "a_then_b": repair_damage_counts(individual_a, composed_ab),
        "b_then_a": repair_damage_counts(individual_b, composed_ba),
        "composed_ab_vs_base": repair_damage_counts(base_correct, composed_ab),
        "composed_ba_vs_base": repair_damage_counts(base_correct, composed_ba),
    }
