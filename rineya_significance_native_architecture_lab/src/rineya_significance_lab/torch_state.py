"""Torch-native persistent significance state.

The Cornerstone-1 invariant is stronger than keeping a score tensor beside a
cache: every surviving ``(K, V, S, C, position, entry_id)`` tuple must describe
the same semantic cache entry after growth, selection, gather, and rollback.
This module owns the four non-KV tensors and centralises their validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _require_rank3(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, kv_heads, positions]; got {tuple(tensor.shape)}")


def _normalise_metadata(
    name: str,
    tensor: torch.Tensor,
    *,
    batch_size: int,
    kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Return integer metadata as ``[batch, kv_heads, positions]``."""

    if tensor.ndim == 1:
        tensor = tensor.view(1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape [positions], [batch, positions], or "
            f"[batch, kv_heads, positions]; got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] not in (1, batch_size):
        raise ValueError(f"{name} batch {tensor.shape[0]} is incompatible with cache batch {batch_size}")
    if tensor.shape[1] not in (1, kv_heads):
        raise ValueError(f"{name} head count {tensor.shape[1]} is incompatible with {kv_heads} KV heads")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must use an integer dtype; got {tensor.dtype}")
    return tensor.to(device=device, dtype=torch.long).expand(batch_size, kv_heads, -1).clone()


def gather_counts(
    counts: torch.Tensor,
    indices: torch.Tensor,
    *,
    cache_length: int,
) -> torch.Tensor:
    """Gather exposure counts with the audited self-compress lag semantics.

    The original self-compressing implementation silently discarded ``C`` when
    a newly appended, not-yet-scored cache tail made ``C`` shorter than KV. That
    made non-zero ranking exponents behave exactly like raw accumulation. A
    missing tail is instead padded with zero observations; counts longer than KV
    indicate corruption and fail loudly.
    """

    _require_rank3("counts", counts)
    _require_rank3("indices", indices)
    if cache_length < 0:
        raise ValueError(f"cache_length must be non-negative, got {cache_length}")
    if counts.shape[:2] != indices.shape[:2]:
        raise ValueError(
            f"count/index batch-head mismatch: {tuple(counts.shape[:2])} vs {tuple(indices.shape[:2])}"
        )
    have = counts.shape[-1]
    if have < cache_length:
        counts = torch.cat(
            [
                counts,
                torch.zeros(
                    (*counts.shape[:2], cache_length - have),
                    dtype=counts.dtype,
                    device=counts.device,
                ),
            ],
            dim=-1,
        )
    elif have > cache_length:
        raise RuntimeError(
            f"observation counts cover {have} entries but KV covers {cache_length}; "
            "dropping counts would silently disable exposure normalization"
        )
    if indices.numel():
        lo = int(indices.min().item())
        hi = int(indices.max().item())
        if lo < 0 or hi >= cache_length:
            raise IndexError(f"gather indices [{lo}, {hi}] are outside cache length {cache_length}")
    return counts.gather(2, indices.to(device=counts.device, dtype=torch.long)).contiguous()


@dataclass
class TorchSignificanceState:
    """Persistent per-layer, per-KV-head Cornerstone-1 tensors."""

    scores: torch.Tensor
    counts: torch.Tensor
    positions: torch.Tensor
    entry_ids: torch.Tensor
    scoring_events: int = 0

    @classmethod
    def empty(
        cls,
        batch_size: int,
        kv_heads: int,
        *,
        device: torch.device | str,
        score_dtype: torch.dtype = torch.float32,
    ) -> TorchSignificanceState:
        if batch_size <= 0 or kv_heads <= 0:
            raise ValueError("batch_size and kv_heads must be positive")
        device = torch.device(device)
        shape = (batch_size, kv_heads, 0)
        return cls(
            scores=torch.empty(shape, dtype=score_dtype, device=device),
            counts=torch.empty(shape, dtype=score_dtype, device=device),
            positions=torch.empty(shape, dtype=torch.long, device=device),
            entry_ids=torch.empty(shape, dtype=torch.long, device=device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.scores.shape[0])

    @property
    def kv_heads(self) -> int:
        return int(self.scores.shape[1])

    @property
    def length(self) -> int:
        return int(self.scores.shape[-1])

    def validate(self, *, expected_length: int | None = None) -> None:
        _require_rank3("scores", self.scores)
        reference = self.scores.shape
        for name, tensor in (
            ("counts", self.counts),
            ("positions", self.positions),
            ("entry_ids", self.entry_ids),
        ):
            _require_rank3(name, tensor)
            if tensor.shape != reference:
                raise RuntimeError(f"unaligned significance state: scores {reference}, {name} {tensor.shape}")
            if tensor.device != self.scores.device:
                raise RuntimeError(
                    f"unaligned significance state devices: scores on {self.scores.device}, "
                    f"{name} on {tensor.device}"
                )
        if self.positions.dtype != torch.long or self.entry_ids.dtype != torch.long:
            raise RuntimeError("positions and entry_ids must be torch.long")
        if not self.scores.is_floating_point() or not self.counts.is_floating_point():
            raise RuntimeError("scores and counts must be floating point")
        if expected_length is not None and self.length != expected_length:
            raise RuntimeError(
                f"significance state covers {self.length} entries but cache covers {expected_length}"
            )
        if not torch.isfinite(self.scores).all():
            raise RuntimeError("significance scores contain non-finite values")
        if not torch.isfinite(self.counts).all() or (self.counts < 0).any():
            raise RuntimeError("observation counts must be finite and non-negative")
        if self.scoring_events < 0:
            raise RuntimeError("scoring_events cannot be negative")

    def append(self, positions: torch.Tensor, entry_ids: torch.Tensor) -> None:
        """Append zero-initialized significance state for newly cached entries."""

        self.validate()
        pos = _normalise_metadata(
            "positions",
            positions,
            batch_size=self.batch_size,
            kv_heads=self.kv_heads,
            device=self.scores.device,
        )
        ids = _normalise_metadata(
            "entry_ids",
            entry_ids,
            batch_size=self.batch_size,
            kv_heads=self.kv_heads,
            device=self.scores.device,
        )
        if pos.shape != ids.shape:
            raise ValueError(f"position/entry-id append mismatch: {pos.shape} vs {ids.shape}")
        if ids.shape[-1] > 1:
            sorted_ids = ids.sort(dim=-1).values
            if (sorted_ids[..., 1:] == sorted_ids[..., :-1]).any():
                raise RuntimeError("new entry_ids must be unique within every batch/KV-head sequence")
        if pos.shape[-1] and self.length:
            duplicates = ids.unsqueeze(-1) == self.entry_ids.unsqueeze(-2)
            if duplicates.any():
                duplicate_id = int(ids[duplicates.any(dim=-1)].flatten()[0].item())
                raise RuntimeError(f"entry_id {duplicate_id} is already resident in at least one KV head")
        zeros = torch.zeros((*self.scores.shape[:2], pos.shape[-1]), dtype=self.scores.dtype, device=self.scores.device)
        self.scores = torch.cat([self.scores, zeros], dim=-1)
        self.counts = torch.cat([self.counts, torch.zeros_like(zeros)], dim=-1)
        self.positions = torch.cat([self.positions, pos], dim=-1)
        self.entry_ids = torch.cat([self.entry_ids, ids], dim=-1)
        self.validate()

    def update(self, contribution: torch.Tensor, *, exposure: float | torch.Tensor = 1.0) -> None:
        """Accumulate one scoring event over every currently resident entry."""

        self.validate()
        if contribution.shape != self.scores.shape:
            raise RuntimeError(
                f"score contribution {tuple(contribution.shape)} does not align with state {tuple(self.scores.shape)}"
            )
        contribution = contribution.to(device=self.scores.device, dtype=self.scores.dtype)
        if not torch.isfinite(contribution).all():
            raise RuntimeError("score contribution contains non-finite values")
        if isinstance(exposure, torch.Tensor):
            exposure_tensor = exposure.to(device=self.counts.device, dtype=self.counts.dtype)
            try:
                exposure_tensor = torch.broadcast_to(exposure_tensor, self.counts.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"exposure {tuple(exposure.shape)} does not broadcast to {tuple(self.counts.shape)}"
                ) from exc
        else:
            if not math.isfinite(float(exposure)) or float(exposure) < 0:
                raise ValueError(f"exposure must be finite and non-negative, got {exposure}")
            exposure_tensor = torch.full_like(self.counts, float(exposure))
        if not torch.isfinite(exposure_tensor).all() or (exposure_tensor < 0).any():
            raise ValueError("exposure must be finite and non-negative")
        self.scores = (self.scores + contribution).contiguous()
        self.counts = (self.counts + exposure_tensor).contiguous()
        self.scoring_events += 1
        self.validate()

    def ranked(self, exponent: float) -> torch.Tensor:
        """Return ``S / max(C, 1)**exponent`` without mutating raw state."""

        self.validate()
        exponent = float(exponent)
        if not math.isfinite(exponent) or exponent < 0:
            raise ValueError(f"count exponent must be finite and non-negative, got {exponent}")
        if exponent == 0.0:
            return self.scores
        return self.scores / self.counts.clamp_min(1.0).pow(exponent)

    def gather(self, indices: torch.Tensor, *, cache_length: int | None = None) -> None:
        """Gather every semantic tensor with one shared per-head index tensor."""

        self.validate()
        _require_rank3("indices", indices)
        if indices.shape[:2] != self.scores.shape[:2]:
            raise ValueError(
                f"index batch-head shape {indices.shape[:2]} does not match state {self.scores.shape[:2]}"
            )
        cache_length = self.length if cache_length is None else int(cache_length)
        if cache_length != self.length:
            raise RuntimeError(
                f"scores/metadata cover {self.length} entries but requested cache length is {cache_length}"
            )
        idx = indices.to(device=self.scores.device, dtype=torch.long)
        if idx.numel():
            lo, hi = int(idx.min().item()), int(idx.max().item())
            if lo < 0 or hi >= cache_length:
                raise IndexError(f"gather indices [{lo}, {hi}] are outside cache length {cache_length}")
        self.scores = self.scores.gather(2, idx).contiguous()
        self.counts = gather_counts(self.counts, idx, cache_length=cache_length)
        self.positions = self.positions.gather(2, idx).contiguous()
        self.entry_ids = self.entry_ids.gather(2, idx).contiguous()
        self.validate(expected_length=idx.shape[-1])

    def truncate(self, length: int) -> None:
        """Remove an append-only temporary suffix from every state tensor."""

        self.validate()
        if length < 0 or length > self.length:
            raise ValueError(f"cannot truncate state length {self.length} to {length}")
        self.scores = self.scores[..., :length].contiguous()
        self.counts = self.counts[..., :length].contiguous()
        self.positions = self.positions[..., :length].contiguous()
        self.entry_ids = self.entry_ids[..., :length].contiguous()
        self.validate(expected_length=length)

    def matrix_bytes(self) -> int:
        """Bytes in the full Significance Matrix: accumulated scores plus counts."""

        return self.scores.numel() * self.scores.element_size() + self.counts.numel() * self.counts.element_size()

    def metadata_bytes(self) -> int:
        """Bytes in explicit position and semantic-entry alignment metadata."""

        return (
            self.positions.numel() * self.positions.element_size()
            + self.entry_ids.numel() * self.entry_ids.element_size()
        )

    def clone(self) -> TorchSignificanceState:
        return TorchSignificanceState(
            scores=self.scores.clone(),
            counts=self.counts.clone(),
            positions=self.positions.clone(),
            entry_ids=self.entry_ids.clone(),
            scoring_events=self.scoring_events,
        )
