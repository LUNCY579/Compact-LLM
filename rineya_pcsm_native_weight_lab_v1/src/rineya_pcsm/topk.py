"""Exact, stable TopK support selection with an exact-forward STE."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor


def budget_to_k(numel: int, fraction: float) -> int:
    if numel <= 0:
        raise ValueError("numel must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("budget fraction must be in (0, 1]")
    resolved = math.floor(numel * fraction + 0.5)
    return min(numel, max(1, resolved))


def stable_topk_mask(scores: Tensor, k: int) -> Tensor:
    """Return exactly ``k`` ones, resolving ties by lower flat index."""

    if scores.numel() == 0:
        raise ValueError("cannot select support from an empty tensor")
    if not 0 <= k <= scores.numel():
        raise ValueError("k must be between zero and scores.numel()")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("support scores must be finite")
    flat = scores.detach().reshape(-1)
    order = torch.argsort(flat, descending=True, stable=True)
    mask = torch.zeros_like(flat, dtype=torch.float32)
    if k:
        mask[order[:k]] = 1.0
    return mask.reshape(scores.shape)


class _ExactTopKSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: Tensor, k: int, temperature: float) -> Tensor:
        hard = stable_topk_mask(scores, int(k)).to(device=scores.device)
        flat = scores.detach().reshape(-1)
        if k == 0:
            threshold = flat.max() + 1.0
        elif k == scores.numel():
            threshold = flat.min() - 1.0
        else:
            order = torch.argsort(flat, descending=True, stable=True)
            threshold = flat[order[int(k) - 1]]
        soft = torch.sigmoid((scores - threshold) / temperature)
        ctx.save_for_backward(soft)
        ctx.temperature = float(temperature)
        return hard

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None, None]:
        (soft,) = ctx.saved_tensors
        surrogate = soft * (1.0 - soft) / ctx.temperature
        return gradient * surrogate, None, None


def topk_ste(scores: Tensor, k: int, *, temperature: float = 1.0) -> Tensor:
    """Exact binary forward mask with a smooth threshold-surrogate backward."""

    if not 0 <= k <= scores.numel():
        raise ValueError("k must be between zero and scores.numel()")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    result = _ExactTopKSTE.apply(  # type: ignore[no-untyped-call]
        scores, k, float(temperature)
    )
    return cast(Tensor, result)


def flat_support_indices(mask: Tensor) -> Tensor:
    """Return sorted int64 flat indices after validating a binary mask."""

    flat = mask.detach().reshape(-1)
    if flat.dtype == torch.bool:
        binary = flat
    else:
        if not bool(torch.logical_or(flat == 0, flat == 1).all()):
            raise ValueError("support mask must be exactly binary")
        binary = flat.bool()
    return torch.nonzero(binary, as_tuple=False).flatten().to(torch.int64).cpu()


@dataclass(frozen=True, slots=True)
class FrozenSupport:
    mask: Tensor
    flat_indices: Tensor
    k: int
    numel: int


def freeze_support(scores: Tensor, k: int) -> FrozenSupport:
    mask = stable_topk_mask(scores, k).bool().contiguous()
    indices = flat_support_indices(mask)
    if indices.numel() != k:
        raise AssertionError("exact TopK support cardinality was not preserved")
    return FrozenSupport(mask=mask, flat_indices=indices, k=k, numel=mask.numel())


def support_union_stats(masks: Mapping[str, Tensor]) -> dict[str, object]:
    """Report per-target K and exact union/intersection support statistics."""

    if not masks:
        raise ValueError("at least one support mask is required")
    names = tuple(sorted(masks))
    numel = masks[names[0]].numel()
    if any(masks[name].numel() != numel for name in names):
        raise ValueError("union statistics require masks with equal numel")
    binary = [
        torch.zeros(numel, dtype=torch.bool).scatter_(
            0, flat_support_indices(masks[name]), True
        )
        for name in names
    ]
    union = torch.stack(binary).any(dim=0)
    intersection = torch.stack(binary).all(dim=0)
    per_target_k = {
        name: int(item.sum().item()) for name, item in zip(names, binary, strict=True)
    }
    return {
        "numel": numel,
        "per_target_k": per_target_k,
        "sum_k": sum(per_target_k.values()),
        "union_k": int(union.sum().item()),
        "union_fraction": float(union.float().mean().item()),
        "intersection_k": int(intersection.sum().item()),
    }


__all__ = [
    "FrozenSupport",
    "budget_to_k",
    "flat_support_indices",
    "freeze_support",
    "stable_topk_mask",
    "support_union_stats",
    "topk_ste",
]
