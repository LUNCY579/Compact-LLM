"""Transformers-compatible cache with aligned persistent significance state."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, DynamicLayer

from ..torch_state import TorchSignificanceState
from .policy import SignificancePolicy, select_keep_indices


@dataclass
class LayerStructureSnapshot:
    stored_length: int
    true_length: int
    next_entry_id: int
    evictions: int
    scoring_events: int
    scores: torch.Tensor
    counts: torch.Tensor
    positions: torch.Tensor
    entry_ids: torch.Tensor
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None


@dataclass
class CacheStructureSnapshot:
    layers: list[LayerStructureSnapshot]
    exact_tensors: bool


@dataclass(frozen=True)
class CompressionReport:
    layers: int
    before_entries: int
    after_entries: int
    evictions: int

    @property
    def reduction(self) -> float:
        return self.before_entries / self.after_entries if self.after_entries else 1.0


def _batch_positions(
    positions: torch.Tensor,
    *,
    batch_size: int,
    query_length: int,
    device: torch.device,
) -> torch.Tensor:
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape [q] or [batch, q], got {tuple(positions.shape)}")
    if positions.shape[0] not in (1, batch_size) or positions.shape[1] != query_length:
        raise ValueError(
            f"positions {tuple(positions.shape)} are incompatible with batch/query {(batch_size, query_length)}"
        )
    if positions.dtype == torch.bool or positions.is_floating_point() or positions.is_complex():
        raise TypeError(f"positions must use an integer dtype, got {positions.dtype}")
    return positions.to(device=device, dtype=torch.long).expand(batch_size, -1).clone()


class SignificanceDynamicLayer(DynamicLayer):
    """Append-only KV layer whose semantic metadata survives per-head gather."""

    def __init__(self, layer_idx: int, *, require_explicit_positions: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.layer_idx = int(layer_idx)
        self.require_explicit_positions = bool(require_explicit_positions)
        self.state: TorchSignificanceState | None = None
        self.true_length = 0
        self.next_entry_id = 0
        self.evictions = 0

    def stored_length(self) -> int:
        if not self.is_initialized or self.keys is None or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])

    def get_seq_length(self) -> int:
        """Return consumed absolute positions, never compressed resident length."""

        return int(self.true_length)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # This is the physical mask shape. Runtime helpers still pass an explicit
        # bottom-right mask because absolute position IDs and resident slot IDs
        # are not interchangeable after gather.
        return self.stored_length() + int(query_length), 0

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
                    f"layer {self.layer_idx}: explicit absolute positions were not prepared for cache update"
                )
            positions = torch.arange(
                self.true_length,
                self.true_length + query_length,
                device=key_states.device,
            ).view(1, -1)
        positions = _batch_positions(
            positions,
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
        entry_ids = _batch_positions(
            entry_ids,
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
        elif self.state.scores.shape[:2] != (batch, kv_heads):
            raise RuntimeError(
                f"layer {self.layer_idx}: cache batch/heads changed from "
                f"{tuple(self.state.scores.shape[:2])} to {(batch, kv_heads)}"
            )
        self.state.append(positions, entry_ids)
        self.true_length += query_length
        self.next_entry_id = max(self.next_entry_id, int(entry_ids.max().item()) + 1 if entry_ids.numel() else 0)
        self.assert_aligned()
        return keys, values

    def assert_aligned(self) -> None:
        stored = self.stored_length()
        if self.is_initialized:
            if self.keys is None or self.values is None or self.keys.shape != self.values.shape:
                raise RuntimeError(f"layer {self.layer_idx}: K/V are missing or misaligned")
        if self.state is None:
            if stored:
                raise RuntimeError(f"layer {self.layer_idx}: KV exists without significance state")
            return
        self.state.validate(expected_length=stored)
        if self.keys is not None:
            if self.state.scores.shape[:2] != self.keys.shape[:2]:
                raise RuntimeError(
                    f"layer {self.layer_idx}: state batch/heads {self.state.scores.shape[:2]} "
                    f"do not match KV {self.keys.shape[:2]}"
                )
            if self.state.scores.device != self.keys.device:
                raise RuntimeError(f"layer {self.layer_idx}: state and KV are on different devices")
        if stored > self.true_length:
            raise RuntimeError(
                f"layer {self.layer_idx}: resident length {stored} exceeds true length {self.true_length}"
            )

    def record(self, contribution: torch.Tensor, *, exposure: float = 1.0) -> None:
        self.assert_aligned()
        if self.state is None:
            raise RuntimeError(f"layer {self.layer_idx}: cannot record scores before cache initialization")
        self.state.update(contribution, exposure=exposure)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.is_initialized:
            return
        index = beam_idx.to(device=self.keys.device, dtype=torch.long)
        self.keys = self.keys.index_select(0, index)
        self.values = self.values.index_select(0, index)
        if self.state is None:
            raise RuntimeError(f"layer {self.layer_idx}: KV exists without significance state")
        self.state.scores = self.state.scores.index_select(0, index).contiguous()
        self.state.counts = self.state.counts.index_select(0, index).contiguous()
        self.state.positions = self.state.positions.index_select(0, index).contiguous()
        self.state.entry_ids = self.state.entry_ids.index_select(0, index).contiguous()
        self.assert_aligned()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        if not self.is_initialized:
            return
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        if self.state is None:
            raise RuntimeError(f"layer {self.layer_idx}: KV exists without significance state")
        self.state.scores = self.state.scores.repeat_interleave(repeats, dim=0)
        self.state.counts = self.state.counts.repeat_interleave(repeats, dim=0)
        self.state.positions = self.state.positions.repeat_interleave(repeats, dim=0)
        self.state.entry_ids = self.state.entry_ids.repeat_interleave(repeats, dim=0)
        self.assert_aligned()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices.to(dtype=torch.long))

    def crop(self, tokens_to_remove: int) -> None:
        raise RuntimeError(
            "generic cache.crop() is ambiguous after per-head compression; use the exact "
            "structural snapshot/rollback API"
        )

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.state = None
        self.true_length = 0
        self.next_entry_id = 0
        self.evictions = 0

    def offload(self) -> None:
        super().offload()
        if self.state is not None:
            self.state.scores = self.state.scores.to("cpu", non_blocking=True)
            self.state.counts = self.state.counts.to("cpu", non_blocking=True)
            self.state.positions = self.state.positions.to("cpu", non_blocking=True)
            self.state.entry_ids = self.state.entry_ids.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        super().prefetch()
        if self.state is not None and self.state.scores.device != self.device:
            self.state.scores = self.state.scores.to(self.device, non_blocking=True)
            self.state.counts = self.state.counts.to(self.device, non_blocking=True)
            self.state.positions = self.state.positions.to(self.device, non_blocking=True)
            self.state.entry_ids = self.state.entry_ids.to(self.device, non_blocking=True)

    def compress(
        self,
        policy: SignificancePolicy,
        *,
        ranking_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress explicitly after a scoring pass and return retained indices.

        ``ranking_counts`` is an experimental denominator used only to choose
        the retained entries.  The persistent, entry-aligned realized counts in
        ``self.state.counts`` are still gathered with K/V and survive unchanged.
        This separation is required for causal count controls: mutating ``C``
        itself would make a one-boundary perturbation contaminate every later
        boundary.
        """

        self.assert_aligned()
        stored = self.stored_length()
        if self.state is None:
            raise RuntimeError(f"layer {self.layer_idx}: no significance state")
        if stored <= policy.budget:
            return (
                torch.arange(stored, device=self.keys.device, dtype=torch.long)
                .view(1, 1, -1)
                .expand(self.keys.shape[0], self.keys.shape[1], -1)
                .clone()
            )
        if (self.state.counts <= 0).any():
            raise RuntimeError(
                f"layer {self.layer_idx}: compression requested with one or more unscored entries; "
                "this would silently turn missing significance into a positional policy"
            )
        if ranking_counts is not None:
            if ranking_counts.shape != self.state.counts.shape:
                raise RuntimeError(
                    f"layer {self.layer_idx}: ranking counts {tuple(ranking_counts.shape)} "
                    f"do not match realized counts {tuple(self.state.counts.shape)}"
                )
            ranking_counts = ranking_counts.to(
                device=self.state.counts.device,
                dtype=self.state.counts.dtype,
            )
            if not torch.isfinite(ranking_counts).all() or (ranking_counts < 0).any():
                raise RuntimeError(
                    f"layer {self.layer_idx}: ranking counts must be finite and non-negative"
                )
        if policy.ranking_mode == "position":
            ranking = self.state.positions.to(dtype=torch.float32)
        elif ranking_counts is None:
            ranking = self.state.ranked(policy.count_exponent)
        elif policy.count_exponent == 0:
            ranking = self.state.scores
        else:
            ranking = self.state.scores / ranking_counts.clamp_min(1.0).pow(
                policy.count_exponent
            )
        indices = select_keep_indices(
            ranking,
            policy,
            positions=self.state.positions,
            entry_ids=self.state.entry_ids,
        )
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, self.keys.shape[-1])
        self.keys = self.keys.gather(2, gather_idx).contiguous()
        self.values = self.values.gather(2, gather_idx).contiguous()
        self.state.gather(indices, cache_length=stored)
        self.evictions += 1
        self.assert_aligned()
        return indices

    def snapshot(self, *, exact_tensors: bool) -> LayerStructureSnapshot:
        self.assert_aligned()
        if self.state is None:
            raise RuntimeError(f"layer {self.layer_idx}: cannot snapshot an uninitialized layer")
        return LayerStructureSnapshot(
            stored_length=self.stored_length(),
            true_length=self.true_length,
            next_entry_id=self.next_entry_id,
            evictions=self.evictions,
            scoring_events=self.state.scoring_events,
            scores=self.state.scores.clone(),
            counts=self.state.counts.clone(),
            positions=self.state.positions.clone(),
            entry_ids=self.state.entry_ids.clone(),
            keys=self.keys.clone() if exact_tensors else None,
            values=self.values.clone() if exact_tensors else None,
        )

    def rollback(
        self,
        snapshot: LayerStructureSnapshot,
        *,
        keep_score_updates: bool,
        verify_exact: bool,
    ) -> None:
        self.assert_aligned()
        if self.state is None or self.stored_length() < snapshot.stored_length:
            raise RuntimeError(f"layer {self.layer_idx}: cache shrank during temporary query probe")
        prefix_keys = self.keys[..., : snapshot.stored_length, :].contiguous()
        prefix_values = self.values[..., : snapshot.stored_length, :].contiguous()
        current_scores = self.state.scores[..., : snapshot.stored_length].contiguous()
        current_counts = self.state.counts[..., : snapshot.stored_length].contiguous()
        current_events = self.state.scoring_events
        current_positions = self.state.positions[..., : snapshot.stored_length].contiguous()
        current_ids = self.state.entry_ids[..., : snapshot.stored_length].contiguous()

        if verify_exact:
            if snapshot.keys is None or snapshot.values is None:
                raise RuntimeError("exact rollback verification requested without cloned KV snapshot")
            structural_pairs = (
                ("keys", prefix_keys, snapshot.keys),
                ("values", prefix_values, snapshot.values),
                ("positions", current_positions, snapshot.positions),
                ("entry_ids", current_ids, snapshot.entry_ids),
            )
            mismatches = [name for name, got, expected in structural_pairs if not torch.equal(got, expected)]
            if mismatches:
                # Restore the known-good structure and complete state before raising.
                self.keys = snapshot.keys.clone()
                self.values = snapshot.values.clone()
                self.state = TorchSignificanceState(
                    snapshot.scores.clone(),
                    snapshot.counts.clone(),
                    snapshot.positions.clone(),
                    snapshot.entry_ids.clone(),
                    snapshot.scoring_events,
                )
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
        self.true_length = snapshot.true_length
        self.next_entry_id = snapshot.next_entry_id
        self.evictions = snapshot.evictions
        self.assert_aligned()


class SignificanceCache(Cache):
    """A Transformers 5.14-style cache with explicit append metadata."""

    def __init__(
        self,
        config=None,
        *,
        num_hidden_layers: int | None = None,
        require_explicit_positions: bool = True,
    ):
        if config is not None:
            text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
            num_hidden_layers = int(text_config.num_hidden_layers)
        if num_hidden_layers is None or num_hidden_layers <= 0:
            raise ValueError("config or a positive num_hidden_layers is required")
        layers = [
            SignificanceDynamicLayer(i, require_explicit_positions=require_explicit_positions)
            for i in range(num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.require_explicit_positions = bool(require_explicit_positions)
        self._pending_positions: torch.Tensor | None = None
        self._pending_entry_ids: torch.Tensor | None = None
        self._pending_layers: set[int] = set()
        self._recording_window: int | None = None

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

    def prepare_append(self, positions: torch.Tensor, entry_ids: torch.Tensor | None = None) -> None:
        """Declare absolute positions/semantic IDs for exactly one model forward."""

        if self._pending_positions is not None:
            raise RuntimeError("a previous prepared cache append was not completed by every layer")
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
        positions = self._pending_positions
        entry_ids = self._pending_entry_ids
        keys, values = self.layers[layer_idx].update(
            key_states,
            value_states,
            *args,
            positions=positions,
            entry_ids=entry_ids,
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
            stored_lengths = {layer.stored_length() for layer in initialized}
            if len(true_lengths) != 1 or len(stored_lengths) != 1:
                raise RuntimeError(
                    f"cache layers disagree: true lengths {sorted(true_lengths)}, "
                    f"stored lengths {sorted(stored_lengths)}"
                )

    def stored_length(self) -> int:
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if not initialized:
            return 0
        lengths = {layer.stored_length() for layer in initialized}
        if len(lengths) != 1:
            raise RuntimeError(f"cache layers have different resident lengths: {sorted(lengths)}")
        return lengths.pop()

    @property
    def true_length(self) -> int:
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if not initialized:
            return 0
        lengths = {layer.true_length for layer in initialized}
        if len(lengths) != 1:
            raise RuntimeError(f"cache layers have different true lengths: {sorted(lengths)}")
        return lengths.pop()

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
                f"before={before}, after={after}. Ensure the model uses Transformers SDPA."
            )

    def compress(self, policy: SignificancePolicy) -> CompressionReport:
        self.assert_forward_complete()
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if len(initialized) != len(self.layers):
            raise RuntimeError(
                f"compression requires every model layer to be initialized; got "
                f"{len(initialized)}/{len(self.layers)}"
            )
        before = sum(layer.stored_length() for layer in self.layers if layer.is_initialized)
        evicted = 0
        compressed_layers = 0
        for layer in self.layers:
            if not layer.is_initialized:
                continue
            old = layer.stored_length()
            layer.compress(policy)
            if layer.stored_length() < old:
                compressed_layers += 1
                evicted += old - layer.stored_length()
        self.assert_aligned()
        after = sum(layer.stored_length() for layer in self.layers if layer.is_initialized)
        return CompressionReport(compressed_layers, before, after, evicted)

    def matrix_bytes(self) -> int:
        return sum(layer.state.matrix_bytes() for layer in self.layers if layer.state is not None)

    def metadata_bytes(self) -> int:
        return sum(layer.state.metadata_bytes() for layer in self.layers if layer.state is not None)

    def snapshot_structure(self, *, exact_tensors: bool = False) -> CacheStructureSnapshot:
        self.assert_forward_complete()
        initialized = [layer for layer in self.layers if layer.is_initialized]
        if len(initialized) != len(self.layers):
            raise RuntimeError("structural snapshots require every model layer to be initialized")
        return CacheStructureSnapshot(
            [layer.snapshot(exact_tensors=exact_tensors) for layer in self.layers],
            exact_tensors=exact_tensors,
        )

    def rollback_structure(
        self,
        snapshot: CacheStructureSnapshot,
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
