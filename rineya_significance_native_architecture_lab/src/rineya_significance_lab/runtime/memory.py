"""Bounded, fixed-slot compression of evicted post-RoPE K/V state.

The memory channel is deliberately independent from cache selection.  It only
accepts tensors that already belong to evicted cache entries and never accepts
answers, evidence annotations, or other benchmark-side metadata.  Each slot is
assigned by stable semantic entry ID, so membership does not depend on tensor
storage order after per-head gathers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, get_args

import torch

MemoryVariant = Literal[
    "mean",
    "recency",
    "raw_significance",
    "normalized_significance",
    "random",
]
MEMORY_VARIANTS: tuple[MemoryVariant, ...] = get_args(MemoryVariant)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class MemoryByteLedger:
    """Exact bytes occupied by every persistent memory tensor."""

    slot_capacity: int
    valid_slots: int
    key_bytes: int
    value_bytes: int
    validity_bytes: int
    source_position_bytes: int
    created_at_bytes: int
    last_updated_at_bytes: int
    source_count_bytes: int
    weight_mass_bytes: int

    @property
    def kv_bytes(self) -> int:
        return self.key_bytes + self.value_bytes

    @property
    def metadata_bytes(self) -> int:
        return (
            self.validity_bytes
            + self.source_position_bytes
            + self.created_at_bytes
            + self.last_updated_at_bytes
            + self.source_count_bytes
            + self.weight_mass_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.kv_bytes + self.metadata_bytes


@dataclass(frozen=True)
class MemoryBudgetMatch:
    """Integer ordinary-cache budget matched against a zero-slot baseline."""

    baseline_ordinary_budget: int
    candidate_ordinary_budget: int
    ordinary_bytes_per_budget_unit: int
    memory_bytes: int
    baseline_total_bytes: int
    candidate_total_bytes: int
    discrete_slack_bytes: int

    @property
    def ordinary_budget_reduction(self) -> int:
        return self.baseline_ordinary_budget - self.candidate_ordinary_budget


def rectangular_position_bytes(
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
    score_dtype: torch.dtype = torch.float32,
    position_dtype: torch.dtype = torch.long,
) -> int:
    """Bytes added by one ordinary resident position in every layer/KV head.

    The ledger includes K, V, S, C, absolute position, and semantic entry ID.
    It deliberately excludes active Python objects and temporary attention
    tensors because the experiment matches persistent architecture state.
    """

    dimensions = (num_layers, batch_size, num_kv_heads, head_dim)
    if any(int(value) <= 0 for value in dimensions):
        raise ValueError(f"rectangular geometry must be positive, got {dimensions}")
    if not kv_dtype.is_floating_point or not score_dtype.is_floating_point:
        raise TypeError("K/V and S/C dtypes must be floating point")
    if position_dtype == torch.bool or position_dtype.is_floating_point:
        raise TypeError("position/entry-ID dtype must be integral")
    kv_element = torch.empty((), dtype=kv_dtype).element_size()
    score_element = torch.empty((), dtype=score_dtype).element_size()
    position_element = torch.empty((), dtype=position_dtype).element_size()
    bytes_per_head_position = (
        2 * int(head_dim) * kv_element + 2 * score_element + 2 * position_element
    )
    return int(num_layers) * int(batch_size) * int(num_kv_heads) * bytes_per_head_position


def match_memory_total_budget(
    *,
    baseline_ordinary_budget: int,
    ordinary_bytes_per_budget_unit: int,
    memory_bytes: int,
    minimum_ordinary_budget: int = 1,
) -> MemoryBudgetMatch:
    """Solve the greatest integer ordinary budget that fits under M=0 total bytes."""

    values = {
        "baseline_ordinary_budget": baseline_ordinary_budget,
        "ordinary_bytes_per_budget_unit": ordinary_bytes_per_budget_unit,
        "memory_bytes": memory_bytes,
        "minimum_ordinary_budget": minimum_ordinary_budget,
    }
    if baseline_ordinary_budget <= 0 or ordinary_bytes_per_budget_unit <= 0:
        raise ValueError(f"baseline budget and unit bytes must be positive: {values}")
    if memory_bytes < 0 or minimum_ordinary_budget <= 0:
        raise ValueError(f"memory bytes must be non-negative and minimum positive: {values}")
    baseline_total = int(baseline_ordinary_budget) * int(ordinary_bytes_per_budget_unit)
    available = baseline_total - int(memory_bytes)
    candidate = available // int(ordinary_bytes_per_budget_unit)
    if candidate < int(minimum_ordinary_budget):
        raise ValueError(
            f"fixed memory state leaves no scientifically usable ordinary-cache budget: {values}"
        )
    candidate_total = candidate * int(ordinary_bytes_per_budget_unit) + int(memory_bytes)
    slack = baseline_total - candidate_total
    if candidate_total > baseline_total or not 0 <= slack < int(ordinary_bytes_per_budget_unit):
        raise RuntimeError("integer total-byte budget solver violated its maximality invariant")
    return MemoryBudgetMatch(
        baseline_ordinary_budget=int(baseline_ordinary_budget),
        candidate_ordinary_budget=int(candidate),
        ordinary_bytes_per_budget_unit=int(ordinary_bytes_per_budget_unit),
        memory_bytes=int(memory_bytes),
        baseline_total_bytes=baseline_total,
        candidate_total_bytes=candidate_total,
        discrete_slack_bytes=slack,
    )


@dataclass(frozen=True)
class MemoryWriteReport:
    """Diagnostics for one layer write."""

    layer_index: int
    variant: MemoryVariant
    evicted_entries: int
    touched_slots: int
    newly_valid_slots: int
    valid_slots_after: int
    zero_weight_fallbacks: int
    boundary_position: int


@dataclass
class MemorySlotState:
    """Persistent tensors for one layer.

    ``keys`` and ``values`` are weighted aggregates of post-RoPE cache states.
    ``source_positions`` is the greatest absolute position represented by a
    slot; it is diagnostic metadata and is never re-used as a RoPE position.
    ``created_at`` is the first boundary at which a slot became visible and
    ``last_updated_at`` is the boundary after which its current contents may be
    attended.  Invalid slots use ``-1`` for all position-like metadata.
    """

    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    source_positions: torch.Tensor
    created_at: torch.Tensor
    last_updated_at: torch.Tensor
    source_counts: torch.Tensor
    weight_mass: torch.Tensor

    def clone(self) -> MemorySlotState:
        return MemorySlotState(
            keys=self.keys.clone(),
            values=self.values.clone(),
            valid=self.valid.clone(),
            source_positions=self.source_positions.clone(),
            created_at=self.created_at.clone(),
            last_updated_at=self.last_updated_at.clone(),
            source_counts=self.source_counts.clone(),
            weight_mass=self.weight_mass.clone(),
        )


@dataclass(frozen=True)
class MemorySnapshot:
    """Exact memory snapshot used by invariant tests and guarded runtimes."""

    layers: tuple[MemorySlotState, ...]


class SignificanceMemory:
    """Tiny fixed-size memory channel for evicted K/V states.

    All layers have exactly ``num_slots`` physical slots for every batch item
    and KV head, including when no source has yet been written.  ``M=0`` is a
    first-class empty state with zero persistent bytes.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        num_slots: int,
        head_dim: int,
        variant: MemoryVariant,
        device: torch.device | str,
        kv_dtype: torch.dtype,
        recency_tau: float = 32.0,
        random_seed: int = 0,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "batch_size": batch_size,
            "num_kv_heads": num_kv_heads,
            "num_slots": num_slots,
            "head_dim": head_dim,
        }
        if any(int(value) < 0 for value in dimensions.values()):
            raise ValueError(f"memory dimensions must be non-negative: {dimensions}")
        if num_layers <= 0 or batch_size <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"all memory dimensions except num_slots must be positive: {dimensions}"
            )
        if variant not in MEMORY_VARIANTS:
            raise ValueError(
                f"unknown memory variant {variant!r}; expected one of {MEMORY_VARIANTS}"
            )
        if not kv_dtype.is_floating_point:
            raise TypeError(f"memory K/V dtype must be floating point, got {kv_dtype}")
        if not math.isfinite(recency_tau) or recency_tau <= 0:
            raise ValueError("recency_tau must be finite and positive")

        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.num_slots = int(num_slots)
        self.head_dim = int(head_dim)
        self.variant: MemoryVariant = variant
        self.device = torch.device(device)
        self.kv_dtype = kv_dtype
        self.recency_tau = float(recency_tau)
        self.random_seed = int(random_seed)

        shape = (self.batch_size, self.num_kv_heads, self.num_slots)
        kv_shape = (*shape, self.head_dim)
        self._layers = [
            MemorySlotState(
                keys=torch.zeros(kv_shape, device=self.device, dtype=self.kv_dtype),
                values=torch.zeros(kv_shape, device=self.device, dtype=self.kv_dtype),
                valid=torch.zeros(shape, device=self.device, dtype=torch.bool),
                source_positions=torch.full(shape, -1, device=self.device, dtype=torch.long),
                created_at=torch.full(shape, -1, device=self.device, dtype=torch.long),
                last_updated_at=torch.full(shape, -1, device=self.device, dtype=torch.long),
                source_counts=torch.zeros(shape, device=self.device, dtype=torch.long),
                weight_mass=torch.zeros(shape, device=self.device, dtype=torch.float32),
            )
            for _ in range(self.num_layers)
        ]
        self.assert_valid()

    def _validate_layer_index(self, layer_index: int) -> int:
        layer = int(layer_index)
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} is outside memory with {self.num_layers} layers")
        return layer

    def retrieve_slots(self, layer_index: int, *, clone: bool = False) -> MemorySlotState:
        """Return the fixed physical slots for one layer.

        The non-cloned result is a read-only-by-contract view used by the
        attention adapter.  External code that needs to retain or mutate a
        value must request ``clone=True``.
        """

        state = self._layers[self._validate_layer_index(layer_index)]
        return state.clone() if clone else state

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(tuple(layer.clone() for layer in self._layers))

    def restore(self, snapshot: MemorySnapshot) -> None:
        if len(snapshot.layers) != self.num_layers:
            raise ValueError("memory snapshot has the wrong layer count")
        self._layers = [layer.clone() for layer in snapshot.layers]
        self.assert_valid()

    def assert_valid(self) -> None:
        expected = (self.batch_size, self.num_kv_heads, self.num_slots)
        expected_kv = (*expected, self.head_dim)
        for layer_index, state in enumerate(self._layers):
            shapes = {
                "keys": tuple(state.keys.shape),
                "values": tuple(state.values.shape),
                "valid": tuple(state.valid.shape),
                "source_positions": tuple(state.source_positions.shape),
                "created_at": tuple(state.created_at.shape),
                "last_updated_at": tuple(state.last_updated_at.shape),
                "source_counts": tuple(state.source_counts.shape),
                "weight_mass": tuple(state.weight_mass.shape),
            }
            if shapes["keys"] != expected_kv or shapes["values"] != expected_kv:
                raise RuntimeError(f"layer {layer_index}: invalid memory K/V shapes {shapes}")
            if any(shapes[name] != expected for name in shapes if name not in {"keys", "values"}):
                raise RuntimeError(f"layer {layer_index}: invalid memory metadata shapes {shapes}")
            tensors = (
                state.keys,
                state.values,
                state.valid,
                state.source_positions,
                state.created_at,
                state.last_updated_at,
                state.source_counts,
                state.weight_mass,
            )
            if any(tensor.device != self.device for tensor in tensors):
                raise RuntimeError(f"layer {layer_index}: memory tensors are on different devices")
            if state.keys.dtype != self.kv_dtype or state.values.dtype != self.kv_dtype:
                raise RuntimeError(f"layer {layer_index}: memory K/V dtype changed")
            if state.valid.dtype != torch.bool:
                raise RuntimeError(f"layer {layer_index}: validity must be boolean")
            if any(
                tensor.dtype != torch.long
                for tensor in (
                    state.source_positions,
                    state.created_at,
                    state.last_updated_at,
                    state.source_counts,
                )
            ):
                raise RuntimeError(f"layer {layer_index}: memory integer metadata dtype changed")
            if state.weight_mass.dtype != torch.float32:
                raise RuntimeError(f"layer {layer_index}: weight mass must be float32")
            if (
                not torch.isfinite(state.keys.float()).all()
                or not torch.isfinite(state.values.float()).all()
            ):
                raise RuntimeError(f"layer {layer_index}: memory K/V contain non-finite values")
            if not torch.isfinite(state.weight_mass).all() or (state.weight_mass < 0).any():
                raise RuntimeError(f"layer {layer_index}: invalid memory weight mass")

            invalid = ~state.valid
            if invalid.any():
                if (state.source_positions[invalid] != -1).any():
                    raise RuntimeError(f"layer {layer_index}: invalid slots carry source positions")
                if (state.created_at[invalid] != -1).any() or (
                    state.last_updated_at[invalid] != -1
                ).any():
                    raise RuntimeError(
                        f"layer {layer_index}: invalid slots carry boundary positions"
                    )
                if (state.source_counts[invalid] != 0).any() or (
                    state.weight_mass[invalid] != 0
                ).any():
                    raise RuntimeError(f"layer {layer_index}: invalid slots carry aggregate mass")
            if state.valid.any():
                valid = state.valid
                if (state.source_positions[valid] < 0).any():
                    raise RuntimeError(f"layer {layer_index}: valid slots need source positions")
                if (state.created_at[valid] < 0).any():
                    raise RuntimeError(f"layer {layer_index}: valid slots need creation boundaries")
                if (state.last_updated_at[valid] <= state.source_positions[valid]).any():
                    raise RuntimeError(
                        f"layer {layer_index}: slot update must follow every represented source"
                    )
                if (state.last_updated_at[valid] < state.created_at[valid]).any():
                    raise RuntimeError(f"layer {layer_index}: slot update predates creation")
                if (state.source_counts[valid] <= 0).any() or (state.weight_mass[valid] <= 0).any():
                    raise RuntimeError(
                        f"layer {layer_index}: valid slots need positive source mass"
                    )

    def assert_visible(self, layer_index: int, *, query_start_position: int) -> None:
        """Reject retroactive reads before a slot's latest write boundary."""

        state = self.retrieve_slots(layer_index)
        if (
            state.valid.any()
            and (state.last_updated_at[state.valid] > int(query_start_position)).any()
        ):
            raise RuntimeError(
                f"layer {layer_index}: memory written after query start {query_start_position}"
            )

    def byte_ledger(self) -> MemoryByteLedger:
        values = {
            "key_bytes": 0,
            "value_bytes": 0,
            "validity_bytes": 0,
            "source_position_bytes": 0,
            "created_at_bytes": 0,
            "last_updated_at_bytes": 0,
            "source_count_bytes": 0,
            "weight_mass_bytes": 0,
        }
        valid_slots = 0
        for state in self._layers:
            values["key_bytes"] += _tensor_bytes(state.keys)
            values["value_bytes"] += _tensor_bytes(state.values)
            values["validity_bytes"] += _tensor_bytes(state.valid)
            values["source_position_bytes"] += _tensor_bytes(state.source_positions)
            values["created_at_bytes"] += _tensor_bytes(state.created_at)
            values["last_updated_at_bytes"] += _tensor_bytes(state.last_updated_at)
            values["source_count_bytes"] += _tensor_bytes(state.source_counts)
            values["weight_mass_bytes"] += _tensor_bytes(state.weight_mass)
            valid_slots += int(state.valid.sum().item())
        return MemoryByteLedger(
            slot_capacity=self.num_layers * self.batch_size * self.num_kv_heads * self.num_slots,
            valid_slots=valid_slots,
            **values,
        )

    def state_sha256(self) -> str:
        """Hash exact persistent tensor bytes, shapes, and dtypes."""

        digest = hashlib.sha256()
        for layer_index, state in enumerate(self._layers):
            digest.update(layer_index.to_bytes(4, "little", signed=False))
            for name in (
                "keys",
                "values",
                "valid",
                "source_positions",
                "created_at",
                "last_updated_at",
                "source_counts",
                "weight_mass",
            ):
                tensor = getattr(state, name).detach().contiguous().cpu()
                digest.update(name.encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(repr(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _random_weights(
        self,
        entry_ids: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        """Return positive deterministic weights keyed only by model-state IDs."""

        result = torch.empty(entry_ids.shape, dtype=torch.float32, device=self.device)
        ids = entry_ids.detach().to(device="cpu", dtype=torch.long)
        for batch_index in range(self.batch_size):
            for head_index in range(self.num_kv_heads):
                weights: list[float] = []
                for entry_id in ids[batch_index, head_index].tolist():
                    payload = (
                        f"{self.random_seed}:{layer_index}:{batch_index}:"
                        f"{head_index}:{int(entry_id)}"
                    ).encode("ascii")
                    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
                    weights.append((integer + 1) / (2**64 + 1))
                result[batch_index, head_index] = torch.tensor(
                    weights,
                    dtype=torch.float32,
                    device=self.device,
                )
        return result

    def _weights(
        self,
        *,
        scores: torch.Tensor,
        counts: torch.Tensor,
        positions: torch.Tensor,
        entry_ids: torch.Tensor,
        boundary_position: int,
        layer_index: int,
    ) -> torch.Tensor:
        if self.variant == "mean":
            return torch.ones_like(scores, dtype=torch.float32)
        if self.variant == "recency":
            age = int(boundary_position) - positions.float()
            # Clamp only the common log-weight scale to keep very old but still
            # valid sources representable in float32.  Their influence remains
            # negligible without turning a valid slot into zero persistent mass.
            return torch.exp((-age / self.recency_tau).clamp_min(-80.0))
        if self.variant == "raw_significance":
            return scores.float()
        if self.variant == "normalized_significance":
            return scores.float() / counts.float()
        return self._random_weights(entry_ids, layer_index=layer_index)

    def compress_evicted(
        self,
        *,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
        counts: torch.Tensor,
        positions: torch.Tensor,
        entry_ids: torch.Tensor,
        boundary_position: int,
    ) -> MemoryWriteReport:
        """Merge evicted model state into fixed slots for one layer.

        The explicit signature is a leakage boundary: only cache K/V,
        significance S/C, semantic positions/IDs, and the causal compression
        boundary can influence a write.
        """

        layer = self._validate_layer_index(layer_index)
        expected_prefix = (self.batch_size, self.num_kv_heads)
        if (
            keys.ndim != 4
            or tuple(keys.shape[:2]) != expected_prefix
            or keys.shape[-1] != self.head_dim
        ):
            raise ValueError(
                f"evicted keys must have [batch, kv_heads, entries, head_dim] with "
                f"prefix {expected_prefix} and dim {self.head_dim}; got {tuple(keys.shape)}"
            )
        if values.shape != keys.shape:
            raise ValueError("evicted K/V shapes must match")
        entry_count = int(keys.shape[-2])
        expected_metadata = (*expected_prefix, entry_count)
        metadata = {
            "scores": scores,
            "counts": counts,
            "positions": positions,
            "entry_ids": entry_ids,
        }
        if any(tuple(tensor.shape) != expected_metadata for tensor in metadata.values()):
            raise ValueError(
                f"evicted metadata must all have shape {expected_metadata}; got "
                f"{ {name: tuple(tensor.shape) for name, tensor in metadata.items()} }"
            )
        if any(tensor.device != self.device for tensor in (keys, values, *metadata.values())):
            raise ValueError("evicted inputs and memory must share a device")
        if keys.dtype != self.kv_dtype or values.dtype != self.kv_dtype:
            raise TypeError(
                f"evicted K/V dtype must match memory {self.kv_dtype}; got {keys.dtype}/{values.dtype}"
            )
        if positions.dtype == torch.bool or positions.is_floating_point() or positions.is_complex():
            raise TypeError("evicted positions must use an integer dtype")
        if entry_ids.dtype == torch.bool or entry_ids.is_floating_point() or entry_ids.is_complex():
            raise TypeError("evicted entry IDs must use an integer dtype")
        if not scores.is_floating_point() or not counts.is_floating_point():
            raise TypeError("evicted S/C must be floating point")
        if not torch.isfinite(keys.float()).all() or not torch.isfinite(values.float()).all():
            raise ValueError("evicted K/V contain non-finite values")
        if not torch.isfinite(scores).all() or not torch.isfinite(counts).all():
            raise ValueError("evicted S/C contain non-finite values")
        if (scores < 0).any() or (counts <= 0).any():
            raise ValueError("evicted scores must be non-negative and counts strictly positive")
        if (positions < 0).any() or (positions >= int(boundary_position)).any():
            raise ValueError("every evicted source position must precede the write boundary")
        if (entry_ids < 0).any():
            raise ValueError("evicted entry IDs must be non-negative")

        state = self._layers[layer]
        valid_before = int(state.valid.sum().item())
        if entry_count == 0 or self.num_slots == 0:
            return MemoryWriteReport(
                layer_index=layer,
                variant=self.variant,
                evicted_entries=self.batch_size * self.num_kv_heads * entry_count,
                touched_slots=0,
                newly_valid_slots=0,
                valid_slots_after=valid_before,
                zero_weight_fallbacks=0,
                boundary_position=int(boundary_position),
            )

        weights = self._weights(
            scores=scores,
            counts=counts,
            positions=positions,
            entry_ids=entry_ids,
            boundary_position=int(boundary_position),
            layer_index=layer,
        )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise RuntimeError("memory aggregation produced invalid weights")

        # Recency weights are defined relative to the current boundary.  Bring
        # previously stored mass to the same reference before merging.
        if self.variant == "recency" and state.valid.any():
            age_delta = int(boundary_position) - state.last_updated_at[state.valid].float()
            decayed = state.weight_mass[state.valid] * torch.exp(
                (-age_delta / self.recency_tau).clamp_min(-80.0)
            )
            state.weight_mass[state.valid] = decayed.clamp_min(torch.finfo(torch.float32).tiny)
            state.last_updated_at[state.valid] = int(boundary_position)

        assignments = entry_ids.to(dtype=torch.long).remainder(self.num_slots)
        touched_slots = 0
        zero_fallbacks = 0
        for batch_index in range(self.batch_size):
            for head_index in range(self.num_kv_heads):
                for slot_index in range(self.num_slots):
                    members = assignments[batch_index, head_index] == slot_index
                    member_count = int(members.sum().item())
                    if member_count == 0:
                        continue
                    touched_slots += 1
                    member_weights = weights[batch_index, head_index, members].float()
                    new_mass = member_weights.sum()
                    if float(new_mass.item()) <= 0:
                        member_weights = torch.ones_like(member_weights)
                        new_mass = member_weights.sum()
                        zero_fallbacks += 1
                    new_keys = keys[batch_index, head_index, members].float()
                    new_values = values[batch_index, head_index, members].float()
                    weighted_key_sum = torch.einsum("n,nd->d", member_weights, new_keys)
                    weighted_value_sum = torch.einsum("n,nd->d", member_weights, new_values)

                    was_valid = bool(state.valid[batch_index, head_index, slot_index].item())
                    old_mass = (
                        state.weight_mass[batch_index, head_index, slot_index]
                        if was_valid
                        else torch.zeros((), device=self.device, dtype=torch.float32)
                    )
                    total_mass = old_mass + new_mass
                    if was_valid:
                        weighted_key_sum += (
                            state.keys[batch_index, head_index, slot_index].float() * old_mass
                        )
                        weighted_value_sum += (
                            state.values[batch_index, head_index, slot_index].float() * old_mass
                        )
                    state.keys[batch_index, head_index, slot_index] = (
                        weighted_key_sum / total_mass
                    ).to(dtype=self.kv_dtype)
                    state.values[batch_index, head_index, slot_index] = (
                        weighted_value_sum / total_mass
                    ).to(dtype=self.kv_dtype)
                    maximum_source = int(positions[batch_index, head_index, members].max().item())
                    if was_valid:
                        maximum_source = max(
                            maximum_source,
                            int(state.source_positions[batch_index, head_index, slot_index].item()),
                        )
                    else:
                        state.created_at[batch_index, head_index, slot_index] = int(
                            boundary_position
                        )
                    state.valid[batch_index, head_index, slot_index] = True
                    state.source_positions[batch_index, head_index, slot_index] = maximum_source
                    state.last_updated_at[batch_index, head_index, slot_index] = int(
                        boundary_position
                    )
                    state.source_counts[batch_index, head_index, slot_index] += member_count
                    state.weight_mass[batch_index, head_index, slot_index] = total_mass

        self.assert_valid()
        valid_after = int(state.valid.sum().item())
        return MemoryWriteReport(
            layer_index=layer,
            variant=self.variant,
            evicted_entries=self.batch_size * self.num_kv_heads * entry_count,
            touched_slots=touched_slots,
            newly_valid_slots=valid_after - valid_before,
            valid_slots_after=valid_after,
            zero_weight_fallbacks=zero_fallbacks,
            boundary_position=int(boundary_position),
        )
