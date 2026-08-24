"""Cornerstone-1 ranking and matched-budget selection policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SignificancePolicy:
    """Per-layer/per-KV-head cache selection.

    ``count_exponent=0`` preserves the audited implementation's raw-accumulation
    default. Experiments studying observation normalization should always set it
    explicitly to ``1.0``. Pooling is off by default so Cornerstone 1 remains
    isolated from the span-oriented Cornerstone-2 mechanism.
    """

    budget: int
    sinks: int = 4
    recent: int = 32
    count_exponent: float = 0.0
    pool_width: int = 1
    protect_sinks: bool = True
    require_heavy: bool = False
    ranking_mode: str = "significance"

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError(f"budget must be positive, got {self.budget}")
        if self.sinks < 0 or self.recent < 0:
            raise ValueError("sinks and recent must be non-negative")
        if not math.isfinite(float(self.count_exponent)) or self.count_exponent < 0:
            raise ValueError("count_exponent must be finite and non-negative")
        if self.pool_width <= 0 or self.pool_width % 2 == 0:
            raise ValueError("pool_width must be a positive odd integer")
        if self.ranking_mode not in {"significance", "position"}:
            raise ValueError("ranking_mode must be 'significance' or 'position'")

    def allocation(self, length: int) -> tuple[int, int, int]:
        """Return ``(sink, heavy, recent)`` slots for a sequence length."""

        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        retained = min(self.budget, length)
        sink = min(self.sinks if self.protect_sinks else 0, retained)
        recent = min(self.recent, max(0, retained - sink), max(0, length - sink))
        heavy = max(0, retained - sink - recent)
        if self.require_heavy and length > retained and heavy == 0:
            raise RuntimeError(
                f"budget {self.budget} is exhausted by {sink} sinks and {recent} recent entries; "
                "content-significance selection would be disabled"
            )
        return sink, heavy, recent


def _stable_heavy_indices(
    scores: torch.Tensor,
    count: int,
    *,
    positions: torch.Tensor | None,
    entry_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Lexicographic top-k: score desc, position asc, entry ID asc, slot asc."""

    batch, heads, length = scores.shape
    order = (
        torch.arange(length, device=scores.device, dtype=torch.long)
        .view(1, 1, -1)
        .expand(batch, heads, -1)
        .clone()
    )
    default_metadata = order
    for key, descending in (
        (entry_ids if entry_ids is not None else default_metadata, False),
        (positions if positions is not None else default_metadata, False),
        (scores, True),
    ):
        if key.shape != scores.shape:
            raise ValueError("tie-break metadata must match score shape")
        values = key.to(scores.device).gather(2, order)
        permutation = torch.argsort(values, dim=-1, descending=descending, stable=True)
        order = order.gather(2, permutation)
    return order[..., :count]


def select_keep_indices(
    scores: torch.Tensor,
    policy: SignificancePolicy,
    *,
    positions: torch.Tensor | None = None,
    entry_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select sorted per-KV-head indices with shape ``[batch, heads, keep]``."""

    if scores.ndim != 3:
        raise ValueError(f"scores must have shape [batch, kv_heads, positions], got {tuple(scores.shape)}")
    if not scores.is_floating_point():
        raise TypeError("scores must be floating point")
    if not torch.isfinite(scores).all():
        raise RuntimeError("ranking scores contain non-finite values")
    batch, heads, length = scores.shape
    if length <= policy.budget:
        return (
            torch.arange(length, device=scores.device, dtype=torch.long)
            .view(1, 1, -1)
            .expand(batch, heads, -1)
            .clone()
        )

    n_sink, n_heavy, n_recent = policy.allocation(length)
    eligible = scores.clone()
    if n_sink:
        eligible[..., :n_sink] = float("-inf")
    if n_recent:
        eligible[..., length - n_recent :] = float("-inf")

    if policy.pool_width > 1 and length > policy.pool_width:
        eligible = F.max_pool1d(
            eligible,
            kernel_size=policy.pool_width,
            stride=1,
            padding=policy.pool_width // 2,
        )[..., :length]
        # Pooling propagates finite neighbours back into masked regions.
        if n_sink:
            eligible[..., :n_sink] = float("-inf")
        if n_recent:
            eligible[..., length - n_recent :] = float("-inf")

    pieces: list[torch.Tensor] = []
    if n_sink:
        pieces.append(
            torch.arange(n_sink, device=scores.device, dtype=torch.long)
            .view(1, 1, -1)
            .expand(batch, heads, -1)
        )
    if n_heavy:
        pieces.append(
            _stable_heavy_indices(
                eligible,
                n_heavy,
                positions=positions,
                entry_ids=entry_ids,
            )
        )
    if n_recent:
        pieces.append(
            torch.arange(length - n_recent, length, device=scores.device, dtype=torch.long)
            .view(1, 1, -1)
            .expand(batch, heads, -1)
        )
    if not pieces:
        return torch.empty((batch, heads, 0), dtype=torch.long, device=scores.device)
    selected = torch.cat(pieces, dim=-1).sort(dim=-1).values
    expected = min(policy.budget, length)
    if selected.shape[-1] != expected:
        raise RuntimeError(f"selection produced {selected.shape[-1]} entries, expected {expected}")
    if selected.shape[-1] > 1 and (selected[..., 1:] == selected[..., :-1]).any():
        raise RuntimeError("selection produced duplicate indices")
    return selected
