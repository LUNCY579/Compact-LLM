"""Exact padded storage for heterogeneous per-layer/per-KV-head budgets.

Transformers attention caches are rectangular tensors with shape
``[batch, KV heads, slots, head_dim]``.  A genuinely different retained count
for each KV head therefore cannot be represented by a plain ``DynamicCache``.
This module uses the strongest exact representation available under that
contract: selected entries are packed at the left of each head, shorter heads
are zero padded to the longest head in their layer, and a persistent boolean
validity tensor makes those padded slots semantically nonexistent.

The global budget is a count of *logical* KV entries per sequence.  Physical
bytes are reported separately and include rectangular padding and the validity
mask.  This distinction is intentional; padded storage must never be presented
as the memory saving of a truly ragged kernel.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
from transformers.cache_utils import Cache, DynamicLayer

from ..torch_state import TorchSignificanceState
from .policy import SignificancePolicy, select_keep_indices


def _batch_metadata(
    values: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    query_length: int,
    device: torch.device,
) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [q] or [batch, q], got {tuple(values.shape)}")
    if values.shape[0] not in (1, batch_size) or values.shape[1] != query_length:
        raise ValueError(
            f"{name} {tuple(values.shape)} is incompatible with batch/query "
            f"{(batch_size, query_length)}"
        )
    if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
        raise TypeError(f"{name} must use an integer dtype, got {values.dtype}")
    return values.to(device=device, dtype=torch.long).expand(batch_size, -1).clone()


@dataclass
class HeterogeneousLayerSnapshot:
    physical_length: int
    true_length: int
    next_entry_id: int
    evictions: int
    scoring_events: int
    scores: torch.Tensor
    counts: torch.Tensor
    positions: torch.Tensor
    entry_ids: torch.Tensor
    validity: torch.Tensor
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None


@dataclass
class HeterogeneousCacheSnapshot:
    layers: list[HeterogeneousLayerSnapshot]
    exact_tensors: bool


@dataclass(frozen=True)
class HeterogeneousCompressionReport:
    """One fixed-global-budget compression boundary."""

    allocations: tuple[tuple[int, ...], ...]
    global_budget: int
    before_logical_entries: int
    after_logical_entries: int
    before_physical_slots: int
    after_physical_slots: int
    evicted_logical_entries: int


@dataclass(frozen=True)
class CacheByteAccounting:
    """Exact persistent-byte ledger for the padded representation.

    ``logical_*`` fields price only valid semantic entries. ``physical_*``
    fields are tensor ``numel * element_size`` values and therefore include
    padding. ``representation_overhead_bytes`` includes both padding and the
    mandatory validity mask.
    """

    batch_size: int
    logical_entries_per_sequence: int
    logical_kv_bytes: int
    physical_kv_bytes: int
    logical_matrix_bytes: int
    physical_matrix_bytes: int
    logical_metadata_bytes: int
    physical_metadata_bytes: int
    validity_mask_bytes: int
    physical_total_bytes: int
    logical_payload_bytes: int
    representation_overhead_bytes: int


class HeterogeneousSignificanceLayer(DynamicLayer):
    """Rectangular KV layer with exact per-head logical validity."""

    def __init__(self, layer_idx: int, *, require_explicit_positions: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.layer_idx = int(layer_idx)
        self.require_explicit_positions = bool(require_explicit_positions)
        self.state: TorchSignificanceState | None = None
        self.validity: torch.Tensor | None = None
        self.true_length = 0
        self.next_entry_id = 0
        self.evictions = 0

    def physical_length(self) -> int:
        if not self.is_initialized or self.keys is None or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])

    def stored_length(self) -> int:
        """Compatibility alias for the physical rectangular length."""

        return self.physical_length()

    def get_seq_length(self) -> int:
        """Return consumed absolute positions, not resident/padded slots."""

        return int(self.true_length)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.physical_length() + int(query_length), 0

    def _validate_new_positions(self, positions: torch.Tensor) -> None:
        expected = torch.arange(
            self.true_length,
            self.true_length + positions.shape[-1],
            device=positions.device,
            dtype=torch.long,
        ).view(1, -1)
        if not torch.equal(positions, expected.expand_as(positions)):
            raise RuntimeError(
                f"layer {self.layer_idx}: positions must be the contiguous absolute range "
                f"[{self.true_length}, {self.true_length + positions.shape[-1]}); got "
                f"{positions.tolist()}"
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        positions: torch.Tensor | None = None,
        entry_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.shape != key_states.shape:
            raise ValueError(
                f"layer {self.layer_idx}: K/V must share [batch, kv_heads, q, dim], got "
                f"{tuple(key_states.shape)} and {tuple(value_states.shape)}"
            )
        batch, kv_heads, query_length, _ = key_states.shape
        if positions is None:
            if self.require_explicit_positions:
                raise RuntimeError(
                    f"layer {self.layer_idx}: explicit absolute positions were not prepared"
                )
            positions = torch.arange(
                self.true_length,
                self.true_length + query_length,
                device=key_states.device,
            ).view(1, -1)
        positions = _batch_metadata(
            positions,
            name="positions",
            batch_size=batch,
            query_length=query_length,
            device=key_states.device,
        )
        self._validate_new_positions(positions)
        if entry_ids is None:
            entry_ids = torch.arange(
                self.next_entry_id,
                self.next_entry_id + query_length,
                device=key_states.device,
            ).view(1, -1)
        entry_ids = _batch_metadata(
            entry_ids,
            name="entry_ids",
            batch_size=batch,
            query_length=query_length,
            device=key_states.device,
        )

        keys, values = super().update(key_states, value_states, *args, **kwargs)
        if self.state is None:
            self.state = TorchSignificanceState.empty(
                batch,
                kv_heads,
                device=key_states.device,
                score_dtype=torch.float32,
            )
            self.validity = torch.empty(
                (batch, kv_heads, 0),
                dtype=torch.bool,
                device=key_states.device,
            )
        elif self.state.scores.shape[:2] != (batch, kv_heads):
            raise RuntimeError(
                f"layer {self.layer_idx}: cache batch/heads changed from "
                f"{tuple(self.state.scores.shape[:2])} to {(batch, kv_heads)}"
            )
        assert self.validity is not None
        self.state.append(positions, entry_ids)
        self.validity = torch.cat(
            [
                self.validity,
                torch.ones(
                    (batch, kv_heads, query_length),
                    dtype=torch.bool,
                    device=key_states.device,
                ),
            ],
            dim=-1,
        ).contiguous()
        self.true_length += query_length
        if entry_ids.numel():
            self.next_entry_id = max(self.next_entry_id, int(entry_ids.max().item()) + 1)
        self.assert_aligned()
        return keys, values

    def logical_counts(self) -> torch.Tensor:
        if self.validity is None:
            return torch.empty((0, 0), dtype=torch.long)
        return self.validity.sum(dim=-1, dtype=torch.long)

    def logical_entries_per_sequence(self) -> int:
        counts = self.logical_counts()
        if counts.numel() == 0:
            return 0
        reference = counts[0]
        if not torch.equal(counts, reference.unsqueeze(0).expand_as(counts)):
            raise RuntimeError(
                f"layer {self.layer_idx}: batches have different logical per-head lengths"
            )
        return int(reference.sum().item())

    def assert_aligned(self) -> None:
        physical = self.physical_length()
        if self.is_initialized:
            if self.keys is None or self.values is None or self.keys.shape != self.values.shape:
                raise RuntimeError(f"layer {self.layer_idx}: K/V are missing or misaligned")
        if self.state is None or self.validity is None:
            if physical:
                raise RuntimeError(f"layer {self.layer_idx}: KV exists without significance validity")
            return
        self.state.validate(expected_length=physical)
        if self.validity.shape != self.state.scores.shape or self.validity.dtype != torch.bool:
            raise RuntimeError(
                f"layer {self.layer_idx}: validity {tuple(self.validity.shape)} does not match "
                f"state {tuple(self.state.scores.shape)}"
            )
        if self.validity.device != self.state.scores.device:
            raise RuntimeError(f"layer {self.layer_idx}: validity and significance state differ in device")
        if self.keys is not None:
            if self.state.scores.shape[:2] != self.keys.shape[:2]:
                raise RuntimeError(
                    f"layer {self.layer_idx}: state batch/heads {self.state.scores.shape[:2]} "
                    f"do not match KV {self.keys.shape[:2]}"
                )
            if self.state.scores.device != self.keys.device:
                raise RuntimeError(f"layer {self.layer_idx}: state and KV differ in device")
        invalid = ~self.validity
        if invalid.any():
            if (self.state.scores[invalid] != 0).any() or (self.state.counts[invalid] != 0).any():
                raise RuntimeError(f"layer {self.layer_idx}: padded S/C slots must be zero")
            if (self.state.positions[invalid] != -1).any() or (self.state.entry_ids[invalid] != -1).any():
                raise RuntimeError(f"layer {self.layer_idx}: padded metadata slots must use -1")
            if self.keys is not None:
                expanded = invalid.unsqueeze(-1).expand_as(self.keys)
                if (self.keys[expanded] != 0).any() or (self.values[expanded] != 0).any():
                    raise RuntimeError(f"layer {self.layer_idx}: padded K/V slots must be zero")
        if physical > self.true_length:
            # Padding can make the physical width exceed the logical count of an
            # individual head, but never the total number of absolute positions
            # consumed by the layer.
            raise RuntimeError(
                f"layer {self.layer_idx}: physical length {physical} exceeds true length {self.true_length}"
            )

    def record(self, contribution: torch.Tensor, *, exposure: float = 1.0) -> None:
        self.assert_aligned()
        if self.state is None or self.validity is None:
            raise RuntimeError(f"layer {self.layer_idx}: cannot score an uninitialized cache")
        if contribution.shape != self.state.scores.shape:
            raise RuntimeError(
                f"layer {self.layer_idx}: contribution {tuple(contribution.shape)} does not match "
                f"state {tuple(self.state.scores.shape)}"
            )
        if not torch.isfinite(contribution).all():
            raise RuntimeError("score contribution contains non-finite values")
        if not math.isfinite(float(exposure)) or exposure < 0:
            raise ValueError("exposure must be finite and non-negative")
        valid = self.validity.to(dtype=self.state.scores.dtype)
        self.state.scores = (
            self.state.scores
            + contribution.to(
                device=self.state.scores.device,
                dtype=self.state.scores.dtype,
            )
            * valid
        ).contiguous()
        self.state.counts = (self.state.counts + valid * float(exposure)).contiguous()
        self.state.scoring_events += 1
        self.assert_aligned()

    def _ranking(self, policy: SignificancePolicy) -> torch.Tensor:
        assert self.state is not None
        if policy.ranking_mode == "position":
            return self.state.positions.to(dtype=torch.float32)
        return self.state.ranked(policy.count_exponent)

    def compress(self, policy: SignificancePolicy, allocations: torch.Tensor) -> None:
        """Select and left-pack an exact logical budget for every KV head."""

        self.assert_aligned()
        if self.state is None or self.validity is None or self.keys is None or self.values is None:
            raise RuntimeError(f"layer {self.layer_idx}: cannot compress an uninitialized cache")
        batch, heads, _, dim = self.keys.shape
        allocation = torch.as_tensor(allocations, dtype=torch.long, device=self.keys.device)
        if allocation.shape != (heads,):
            raise ValueError(
                f"layer {self.layer_idx}: allocations must have shape {(heads,)}, got "
                f"{tuple(allocation.shape)}"
            )
        if (allocation < 0).any():
            raise ValueError("heterogeneous allocations must be non-negative")
        counts = self.logical_counts()
        if counts.shape != (batch, heads):
            raise RuntimeError("logical-count shape is corrupt")
        if (counts < allocation.view(1, -1)).any():
            raise ValueError(
                f"layer {self.layer_idx}: allocation {allocation.tolist()} exceeds logical "
                f"capacity {counts.tolist()}"
            )
        if ((self.state.counts <= 0) & self.validity).any():
            raise RuntimeError(
                f"layer {self.layer_idx}: compression requested with one or more unscored valid entries"
            )

        ranking = self._ranking(policy)
        max_keep = int(allocation.max().item()) if allocation.numel() else 0
        new_keys = self.keys.new_zeros((batch, heads, max_keep, dim))
        new_values = self.values.new_zeros((batch, heads, max_keep, dim))
        new_scores = self.state.scores.new_zeros((batch, heads, max_keep))
        new_counts = self.state.counts.new_zeros((batch, heads, max_keep))
        new_positions = self.state.positions.new_full((batch, heads, max_keep), -1)
        new_entry_ids = self.state.entry_ids.new_full((batch, heads, max_keep), -1)
        new_validity = torch.zeros(
            (batch, heads, max_keep), dtype=torch.bool, device=self.keys.device
        )

        for batch_idx in range(batch):
            for head_idx in range(heads):
                keep = int(allocation[head_idx].item())
                if keep == 0:
                    continue
                valid_slots = torch.nonzero(
                    self.validity[batch_idx, head_idx], as_tuple=False
                ).flatten()
                compact_scores = ranking[batch_idx, head_idx, valid_slots].view(1, 1, -1)
                compact_positions = self.state.positions[
                    batch_idx, head_idx, valid_slots
                ].view(1, 1, -1)
                compact_ids = self.state.entry_ids[batch_idx, head_idx, valid_slots].view(
                    1, 1, -1
                )
                local_policy = replace(policy, budget=keep)
                local = select_keep_indices(
                    compact_scores,
                    local_policy,
                    positions=compact_positions,
                    entry_ids=compact_ids,
                )[0, 0]
                selected = valid_slots[local]
                new_keys[batch_idx, head_idx, :keep] = self.keys[
                    batch_idx, head_idx, selected
                ]
                new_values[batch_idx, head_idx, :keep] = self.values[
                    batch_idx, head_idx, selected
                ]
                new_scores[batch_idx, head_idx, :keep] = self.state.scores[
                    batch_idx, head_idx, selected
                ]
                new_counts[batch_idx, head_idx, :keep] = self.state.counts[
                    batch_idx, head_idx, selected
                ]
                new_positions[batch_idx, head_idx, :keep] = self.state.positions[
                    batch_idx, head_idx, selected
                ]
                new_entry_ids[batch_idx, head_idx, :keep] = self.state.entry_ids[
                    batch_idx, head_idx, selected
                ]
                new_validity[batch_idx, head_idx, :keep] = True

        old_counts = counts
        self.keys = new_keys.contiguous()
        self.values = new_values.contiguous()
        self.state = TorchSignificanceState(
            scores=new_scores.contiguous(),
            counts=new_counts.contiguous(),
            positions=new_positions.contiguous(),
            entry_ids=new_entry_ids.contiguous(),
            scoring_events=self.state.scoring_events,
        )
        self.validity = new_validity.contiguous()
        if (old_counts != allocation.view(1, -1)).any():
            self.evictions += 1
        self.assert_aligned()

    def snapshot(self, *, exact_tensors: bool) -> HeterogeneousLayerSnapshot:
        self.assert_aligned()
        if self.state is None or self.validity is None:
            raise RuntimeError(f"layer {self.layer_idx}: cannot snapshot an uninitialized layer")
        return HeterogeneousLayerSnapshot(
            physical_length=self.physical_length(),
            true_length=self.true_length,
            next_entry_id=self.next_entry_id,
            evictions=self.evictions,
            scoring_events=self.state.scoring_events,
            scores=self.state.scores.clone(),
            counts=self.state.counts.clone(),
            positions=self.state.positions.clone(),
            entry_ids=self.state.entry_ids.clone(),
            validity=self.validity.clone(),
            keys=self.keys.clone() if exact_tensors else None,
            values=self.values.clone() if exact_tensors else None,
        )

    def rollback(
        self,
        snapshot: HeterogeneousLayerSnapshot,
        *,
        keep_score_updates: bool,
        verify_exact: bool,
    ) -> None:
        self.assert_aligned()
        if self.state is None or self.validity is None or self.keys is None or self.values is None:
            raise RuntimeError(f"layer {self.layer_idx}: cannot rollback an uninitialized layer")
        if self.physical_length() < snapshot.physical_length:
            raise RuntimeError(f"layer {self.layer_idx}: cache shrank during temporary query probe")
        length = snapshot.physical_length
        prefix_keys = self.keys[..., :length, :].contiguous()
        prefix_values = self.values[..., :length, :].contiguous()
        current_scores = self.state.scores[..., :length].contiguous()
        current_counts = self.state.counts[..., :length].contiguous()
        current_positions = self.state.positions[..., :length].contiguous()
        current_ids = self.state.entry_ids[..., :length].contiguous()
        current_validity = self.validity[..., :length].contiguous()
        current_events = self.state.scoring_events

        if verify_exact:
            if snapshot.keys is None or snapshot.values is None:
                raise RuntimeError("exact rollback verification requested without cloned KV")
            pairs = (
                ("keys", prefix_keys, snapshot.keys),
                ("values", prefix_values, snapshot.values),
                ("positions", current_positions, snapshot.positions),
                ("entry_ids", current_ids, snapshot.entry_ids),
                ("validity", current_validity, snapshot.validity),
            )
            mismatches = [name for name, got, expected in pairs if not torch.equal(got, expected)]
            if mismatches:
                self.keys = snapshot.keys.clone()
                self.values = snapshot.values.clone()
                self.state = TorchSignificanceState(
                    snapshot.scores.clone(),
                    snapshot.counts.clone(),
                    snapshot.positions.clone(),
                    snapshot.entry_ids.clone(),
                    snapshot.scoring_events,
                )
                self.validity = snapshot.validity.clone()
                self.true_length = snapshot.true_length
                self.next_entry_id = snapshot.next_entry_id
                self.evictions = snapshot.evictions
                raise RuntimeError(
                    f"layer {self.layer_idx}: temporary query changed structural tensors: {mismatches}"
                )

        self.keys = prefix_keys
        self.values = prefix_values
        self.state = TorchSignificanceState(
            scores=current_scores if keep_score_updates else snapshot.scores.clone(),
            counts=current_counts if keep_score_updates else snapshot.counts.clone(),
            positions=snapshot.positions.clone(),
            entry_ids=snapshot.entry_ids.clone(),
            scoring_events=current_events if keep_score_updates else snapshot.scoring_events,
        )
        self.validity = snapshot.validity.clone()
        self.true_length = snapshot.true_length
        self.next_entry_id = snapshot.next_entry_id
        self.evictions = snapshot.evictions
        self.assert_aligned()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.is_initialized:
            return
        assert self.keys is not None and self.values is not None
        index = beam_idx.to(device=self.keys.device, dtype=torch.long)
        self.keys = self.keys.index_select(0, index)
        self.values = self.values.index_select(0, index)
        if self.state is None or self.validity is None:
            raise RuntimeError(f"layer {self.layer_idx}: KV exists without semantic state")
        self.state.scores = self.state.scores.index_select(0, index).contiguous()
        self.state.counts = self.state.counts.index_select(0, index).contiguous()
        self.state.positions = self.state.positions.index_select(0, index).contiguous()
        self.state.entry_ids = self.state.entry_ids.index_select(0, index).contiguous()
        self.validity = self.validity.index_select(0, index).contiguous()
        self.assert_aligned()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        if not self.is_initialized:
            return
        assert self.keys is not None and self.values is not None
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        if self.state is None or self.validity is None:
            raise RuntimeError(f"layer {self.layer_idx}: KV exists without semantic state")
        self.state.scores = self.state.scores.repeat_interleave(repeats, dim=0)
        self.state.counts = self.state.counts.repeat_interleave(repeats, dim=0)
        self.state.positions = self.state.positions.repeat_interleave(repeats, dim=0)
        self.state.entry_ids = self.state.entry_ids.repeat_interleave(repeats, dim=0)
        self.validity = self.validity.repeat_interleave(repeats, dim=0)
        self.assert_aligned()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices.to(dtype=torch.long))

    def crop(self, tokens_to_remove: int) -> None:
        raise RuntimeError(
            "generic cache.crop() is ambiguous after heterogeneous compression; use exact rollback"
        )

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.state = None
        self.validity = None
        self.true_length = 0
        self.next_entry_id = 0
        self.evictions = 0

    def offload(self) -> None:
        super().offload()
        if self.state is not None and self.validity is not None:
            self.state.scores = self.state.scores.to("cpu", non_blocking=True)
            self.state.counts = self.state.counts.to("cpu", non_blocking=True)
            self.state.positions = self.state.positions.to("cpu", non_blocking=True)
            self.state.entry_ids = self.state.entry_ids.to("cpu", non_blocking=True)
            self.validity = self.validity.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        super().prefetch()
        if self.state is not None and self.validity is not None and self.state.scores.device != self.device:
            self.state.scores = self.state.scores.to(self.device, non_blocking=True)
            self.state.counts = self.state.counts.to(self.device, non_blocking=True)
            self.state.positions = self.state.positions.to(self.device, non_blocking=True)
            self.state.entry_ids = self.state.entry_ids.to(self.device, non_blocking=True)
            self.validity = self.validity.to(self.device, non_blocking=True)


class HeterogeneousSignificanceCache(Cache):
    """Transformers cache supporting unequal logical layer/head budgets."""

    def __init__(
        self,
        config=None,
        *,
        num_hidden_layers: int | None = None,
        require_explicit_positions: bool = True,
    ):
        if config is not None:
            text_config = (
                config.get_text_config(decoder=True)
                if hasattr(config, "get_text_config")
                else config
            )
            num_hidden_layers = int(text_config.num_hidden_layers)
        if num_hidden_layers is None or num_hidden_layers <= 0:
            raise ValueError("config or a positive num_hidden_layers is required")
        layers = [
            HeterogeneousSignificanceLayer(
                i, require_explicit_positions=require_explicit_positions
            )
            for i in range(num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.require_explicit_positions = bool(require_explicit_positions)
        self._pending_positions: torch.Tensor | None = None
        self._pending_entry_ids: torch.Tensor | None = None
        self._pending_layers: set[int] = set()
        self._recording_window: int | None = None
        self._attention_adapter_active = False

    @property
    def recording_enabled(self) -> bool:
        return self._recording_window is not None

    @property
    def recording_window(self) -> int:
        if self._recording_window is None:
            raise RuntimeError("significance recording is not enabled")
        return self._recording_window

    @contextmanager
    def recording(self, *, window: int) -> Iterator[None]:
        if window <= 0:
            raise ValueError("recording window must be positive")
        if self.recording_enabled:
            raise RuntimeError("nested significance recording contexts are not supported")
        self._recording_window = int(window)
        try:
            yield
        finally:
            self._recording_window = None

    @property
    def attention_adapter_active(self) -> bool:
        return self._attention_adapter_active

    @contextmanager
    def attention_adapter(self) -> Iterator[None]:
        if self._attention_adapter_active:
            raise RuntimeError("nested heterogeneous attention adapters are not supported")
        self._attention_adapter_active = True
        try:
            yield
        finally:
            self._attention_adapter_active = False

    def attention_adapter_required(self) -> bool:
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if not initialized:
            return False
        widths = {layer.physical_length() for layer in initialized}
        has_padding = any(
            layer.validity is not None and not bool(layer.validity.all().item())
            for layer in initialized
        )
        return len(widths) > 1 or has_padding

    def prepare_append(self, positions: torch.Tensor, entry_ids: torch.Tensor | None = None) -> None:
        if self._pending_positions is not None:
            raise RuntimeError("a previous prepared append was not completed by every layer")
        if positions.ndim not in (1, 2):
            raise ValueError("prepared positions must have shape [q] or [batch, q]")
        self._pending_positions = positions
        self._pending_entry_ids = entry_ids
        self._pending_layers.clear()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(f"layer index {layer_idx} is outside cache with {len(self.layers)} layers")
        if layer_idx in self._pending_layers:
            raise RuntimeError(f"layer {layer_idx} updated twice during one prepared append")
        if self._pending_positions is None and self.require_explicit_positions:
            raise RuntimeError("call cache.prepare_append(position_ids) before each model forward")
        if self.attention_adapter_required() and not self.attention_adapter_active:
            raise RuntimeError(
                "heterogeneous cache padding/layer widths require "
                "heterogeneous_transformers_sdpa for every model forward"
            )
        keys, values = self.layers[layer_idx].update(
            key_states,
            value_states,
            *args,
            positions=self._pending_positions,
            entry_ids=self._pending_entry_ids,
            **kwargs,
        )
        self._pending_layers.add(layer_idx)
        if len(self._pending_layers) == len(self.layers):
            self._pending_positions = None
            self._pending_entry_ids = None
            self._pending_layers.clear()
        return keys, values

    def assert_forward_complete(self) -> None:
        if self._pending_positions is not None:
            raise RuntimeError(
                f"cache append reached only layers {sorted(self._pending_layers)} of {len(self.layers)}"
            )
        self.assert_aligned()

    def assert_aligned(self) -> None:
        initialized = [layer for layer in self.layers if layer.is_initialized]
        for layer in initialized:
            layer.assert_aligned()
        if initialized:
            true_lengths = {layer.true_length for layer in initialized}
            batches = {int(layer.keys.shape[0]) for layer in initialized if layer.keys is not None}
            heads = {int(layer.keys.shape[1]) for layer in initialized if layer.keys is not None}
            if len(true_lengths) != 1:
                raise RuntimeError(f"cache layers disagree on true length: {sorted(true_lengths)}")
            if len(batches) != 1 or len(heads) != 1:
                raise RuntimeError("cache layers disagree on batch size or KV-head count")

    def max_stored_length(self) -> int:
        return max((layer.physical_length() for layer in self.layers), default=0)

    def stored_length(self) -> int:
        """Compatibility alias returning the largest physical layer width."""

        return self.max_stored_length()

    def layer_physical_lengths(self) -> tuple[int, ...]:
        return tuple(layer.physical_length() for layer in self.layers)

    @property
    def true_length(self) -> int:
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if not initialized:
            return 0
        lengths = {layer.true_length for layer in initialized}
        if len(lengths) != 1:
            raise RuntimeError(f"cache layers disagree on true length: {sorted(lengths)}")
        return lengths.pop()

    def logical_entries_per_sequence(self) -> int:
        return sum(layer.logical_entries_per_sequence() for layer in self.layers)

    def physical_slots_per_sequence(self) -> int:
        total = 0
        for layer in self.layers:
            if layer.keys is not None:
                total += int(layer.keys.shape[1]) * layer.physical_length()
        return total

    def record_attention(self, layer_idx: int, contribution: torch.Tensor) -> None:
        if not self.recording_enabled:
            raise RuntimeError("attention recorder called outside a recording context")
        self.layers[layer_idx].record(contribution)

    def scoring_events(self) -> list[int]:
        return [layer.state.scoring_events if layer.state is not None else 0 for layer in self.layers]

    def require_one_score_event(self, before: list[int]) -> None:
        after = self.scoring_events()
        if len(before) != len(after):
            raise ValueError("score-event baseline has the wrong layer count")
        bad = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if b != a + 1]
        if bad:
            raise RuntimeError(
                f"expected exactly one significance event in every layer; failed layers {bad}, "
                f"before={before}, after={after}"
            )

    def compress(
        self,
        policy: SignificancePolicy,
        allocations: torch.Tensor,
        *,
        global_budget: int,
    ) -> HeterogeneousCompressionReport:
        """Compress to an explicit fixed global logical-entry budget."""

        self.assert_forward_complete()
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if len(initialized) != len(self.layers):
            raise RuntimeError("compression requires every model layer to be initialized")
        first = initialized[0]
        assert first.keys is not None
        heads = int(first.keys.shape[1])
        allocation = torch.as_tensor(allocations, dtype=torch.long, device=first.keys.device)
        if allocation.shape != (len(self.layers), heads):
            raise ValueError(
                f"allocations must have shape {(len(self.layers), heads)}, got "
                f"{tuple(allocation.shape)}"
            )
        if global_budget < 0 or int(allocation.sum().item()) != int(global_budget):
            raise ValueError(
                f"allocation sum {int(allocation.sum().item())} does not equal fixed global "
                f"budget {global_budget}"
            )
        before_logical = self.logical_entries_per_sequence()
        before_physical = self.physical_slots_per_sequence()
        for layer_idx, layer in enumerate(self.layers):
            layer.compress(policy, allocation[layer_idx])
        self.assert_aligned()
        after_logical = self.logical_entries_per_sequence()
        after_physical = self.physical_slots_per_sequence()
        if after_logical != global_budget:
            raise RuntimeError(
                f"closed-loop compression retained {after_logical} entries, expected {global_budget}"
            )
        alloc_tuple = tuple(
            tuple(int(value) for value in row)
            for row in allocation.detach().cpu().tolist()
        )
        return HeterogeneousCompressionReport(
            allocations=alloc_tuple,
            global_budget=int(global_budget),
            before_logical_entries=before_logical,
            after_logical_entries=after_logical,
            before_physical_slots=before_physical,
            after_physical_slots=after_physical,
            evicted_logical_entries=before_logical - after_logical,
        )

    def byte_accounting(self) -> CacheByteAccounting:
        self.assert_aligned()
        initialized = [layer for layer in self.layers if layer.is_initialized]
        batch = int(initialized[0].keys.shape[0]) if initialized and initialized[0].keys is not None else 0
        logical_entries_total = 0
        physical_kv = 0
        logical_kv = 0
        physical_matrix = 0
        logical_matrix = 0
        physical_metadata = 0
        logical_metadata = 0
        validity_bytes = 0
        for layer in initialized:
            assert layer.keys is not None and layer.values is not None
            assert layer.state is not None and layer.validity is not None
            valid = int(layer.validity.sum().item())
            logical_entries_total += valid
            physical_kv += (
                layer.keys.numel() * layer.keys.element_size()
                + layer.values.numel() * layer.values.element_size()
            )
            logical_kv += valid * (
                layer.keys.shape[-1] * layer.keys.element_size()
                + layer.values.shape[-1] * layer.values.element_size()
            )
            physical_matrix += layer.state.matrix_bytes()
            logical_matrix += valid * (
                layer.state.scores.element_size() + layer.state.counts.element_size()
            )
            physical_metadata += layer.state.metadata_bytes()
            logical_metadata += valid * (
                layer.state.positions.element_size() + layer.state.entry_ids.element_size()
            )
            validity_bytes += layer.validity.numel() * layer.validity.element_size()
        physical_total = physical_kv + physical_matrix + physical_metadata + validity_bytes
        logical_payload = logical_kv + logical_matrix + logical_metadata
        per_sequence = logical_entries_total // batch if batch else 0
        if batch and per_sequence * batch != logical_entries_total:
            raise RuntimeError("logical entries do not divide evenly across cache batches")
        return CacheByteAccounting(
            batch_size=batch,
            logical_entries_per_sequence=per_sequence,
            logical_kv_bytes=logical_kv,
            physical_kv_bytes=physical_kv,
            logical_matrix_bytes=logical_matrix,
            physical_matrix_bytes=physical_matrix,
            logical_metadata_bytes=logical_metadata,
            physical_metadata_bytes=physical_metadata,
            validity_mask_bytes=validity_bytes,
            physical_total_bytes=physical_total,
            logical_payload_bytes=logical_payload,
            representation_overhead_bytes=physical_total - logical_payload,
        )

    def snapshot_structure(self, *, exact_tensors: bool = False) -> HeterogeneousCacheSnapshot:
        self.assert_forward_complete()
        if any(not layer.is_initialized for layer in self.layers):
            raise RuntimeError("structural snapshots require every model layer to be initialized")
        return HeterogeneousCacheSnapshot(
            [layer.snapshot(exact_tensors=exact_tensors) for layer in self.layers],
            exact_tensors=exact_tensors,
        )

    def rollback_structure(
        self,
        snapshot: HeterogeneousCacheSnapshot,
        *,
        keep_score_updates: bool = True,
        verify_exact: bool | None = None,
    ) -> None:
        if len(snapshot.layers) != len(self.layers):
            raise ValueError("snapshot layer count does not match cache")
        verify = snapshot.exact_tensors if verify_exact is None else bool(verify_exact)
        for layer, layer_snapshot in zip(self.layers, snapshot.layers, strict=True):
            layer.rollback(
                layer_snapshot,
                keep_score_updates=keep_score_updates,
                verify_exact=verify,
            )
        self._pending_positions = None
        self._pending_entry_ids = None
        self._pending_layers.clear()
        self.assert_aligned()
