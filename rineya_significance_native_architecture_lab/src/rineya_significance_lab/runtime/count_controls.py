"""Closed-loop causal controls for the realized exposure denominator.

The live cache always keeps the true, entry-aligned observation count ``C``.
This module supplies a counterfactual denominator only at a compression
boundary, records exactly what was ranked, and then gathers the unmodified true
state.  Consequently a perturbation tests the causal use of ``C`` without
silently changing future exposure accumulation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from .cache import CompressionReport, SignificanceCache
from .policy import SignificancePolicy

CountControlName = Literal[
    "true",
    "constant",
    "age_derived",
    "shuffled",
    "wrong_token",
    "delayed",
    "noisy",
    "head_averaged",
    "eligible_shuffled_v2",
    "eligible_circular_v2",
]

COUNT_CONTROL_NAMES: tuple[CountControlName, ...] = (
    "true",
    "constant",
    "age_derived",
    "shuffled",
    "wrong_token",
    "delayed",
    "noisy",
    "head_averaged",
)

# Version 1 is intentionally retained above because its global shuffle and
# circular controls are part of an immutable, already-audited run.  These v2
# controls instead intervene only on the entries that can compete for heavy
# slots; protected sink/recent denominators stay attached to their entries.
CLEAN_COUNT_CONTROL_NAMES_V2: tuple[CountControlName, ...] = (
    "true",
    "eligible_shuffled_v2",
    "eligible_circular_v2",
)

_ALL_COUNT_CONTROL_NAMES = frozenset((*COUNT_CONTROL_NAMES, *CLEAN_COUNT_CONTROL_NAMES_V2))
_ELIGIBLE_INTERVENTIONS_V2 = frozenset(CLEAN_COUNT_CONTROL_NAMES_V2[1:])


@dataclass(frozen=True, slots=True)
class CountControlSpec:
    """One frozen denominator intervention."""

    condition_id: str
    transform: CountControlName
    seed: int
    circular_shift: int = 1
    noise_std_fraction_of_mean: float = 0.10
    noisy_clamp_min: float = 1.0

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("count-control condition_id must be non-empty")
        if self.transform not in _ALL_COUNT_CONTROL_NAMES:
            raise ValueError(f"unknown count control: {self.transform!r}")
        if self.circular_shift == 0:
            raise ValueError("wrong-token circular shift must be non-zero")
        if (
            not math.isfinite(self.noise_std_fraction_of_mean)
            or self.noise_std_fraction_of_mean < 0
        ):
            raise ValueError("noise_std_fraction_of_mean must be finite and non-negative")
        if not math.isfinite(self.noisy_clamp_min) or self.noisy_clamp_min < 0:
            raise ValueError("noisy_clamp_min must be finite and non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, seed: int) -> CountControlSpec:
        transform = str(value.get("count_transform", ""))
        if transform not in _ALL_COUNT_CONTROL_NAMES:
            raise ValueError(f"unknown count_transform: {transform!r}")
        return cls(
            condition_id=str(value.get("id", "")),
            transform=transform,  # type: ignore[arg-type]
            seed=int(value.get("seed", seed)),
            circular_shift=int(value.get("shift", 1)),
            noise_std_fraction_of_mean=float(
                value.get("noise_std_fraction_of_mean", 0.10)
            ),
            noisy_clamp_min=float(value.get("clamp_min", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class BoundarySummary:
    """Small JSON-safe index for one persisted boundary trace."""

    boundary_index: int
    true_length: int
    stored_before: int
    stored_after: int
    layer_count: int
    kv_heads: int
    selected_per_head: int
    transform: CountControlName

    def to_dict(self) -> dict[str, int | str]:
        return {
            "boundary_index": self.boundary_index,
            "true_length": self.true_length,
            "stored_before": self.stored_before,
            "stored_after": self.stored_after,
            "layer_count": self.layer_count,
            "kv_heads": self.kv_heads,
            "selected_per_head": self.selected_per_head,
            "transform": self.transform,
        }


def _stable_seed(*parts: object) -> int:
    payload = "\x00".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _entry_count_map(
    entry_ids: torch.Tensor,
    counts: torch.Tensor,
) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], float] = {}
    for batch in range(entry_ids.shape[0]):
        for head in range(entry_ids.shape[1]):
            for slot in range(entry_ids.shape[2]):
                key = (batch, head, int(entry_ids[batch, head, slot].item()))
                if key in result:
                    raise RuntimeError(f"duplicate semantic entry in one KV head: {key}")
                result[key] = float(counts[batch, head, slot].item())
    return result


class RuntimeCountController:
    """Stateful, deterministic count intervention and boundary trace recorder."""

    def __init__(
        self,
        spec: CountControlSpec,
        *,
        case_id: str,
        chunk_size: int,
        protected_sinks: int | None = None,
        protected_recent: int | None = None,
    ):
        if not case_id:
            raise ValueError("case_id must be non-empty")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if protected_sinks is not None and protected_sinks < 0:
            raise ValueError("protected_sinks must be non-negative")
        if protected_recent is not None and protected_recent < 0:
            raise ValueError("protected_recent must be non-negative")
        if spec.transform in _ELIGIBLE_INTERVENTIONS_V2 and (
            protected_sinks is None or protected_recent is None
        ):
            raise ValueError(
                "eligible-pool v2 controls require explicit protected_sinks and protected_recent"
            )
        self.spec = spec
        self.case_id = case_id
        self.chunk_size = int(chunk_size)
        self.protected_sinks = int(protected_sinks or 0)
        self.protected_recent = int(protected_recent or 0)
        self._has_explicit_protection = (
            protected_sinks is not None and protected_recent is not None
        )
        self._previous_by_layer: list[dict[tuple[int, int, int], float]] | None = None
        self._arrays: dict[str, np.ndarray[Any, Any]] = {}
        self._summaries: list[BoundarySummary] = []
        self._effective_source_entry_ids: list[torch.Tensor | None] = []

    @property
    def summaries(self) -> tuple[BoundarySummary, ...]:
        return tuple(self._summaries)

    def diagnostic_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return copies of the exact arrays needed for boundary evidence scoring."""

        return {name: value.copy() for name, value in self._arrays.items()}

    def _permuted(self, values: torch.Tensor, *, boundary: int, layer: int) -> torch.Tensor:
        result = torch.empty_like(values)
        for batch in range(values.shape[0]):
            for head in range(values.shape[1]):
                generator = torch.Generator(device="cpu").manual_seed(
                    _stable_seed(
                        self.spec.seed,
                        self.case_id,
                        self.spec.condition_id,
                        boundary,
                        layer,
                        batch,
                        head,
                        "count-shuffle",
                    )
                )
                order = torch.randperm(values.shape[-1], generator=generator)
                if values.shape[-1] > 1 and torch.equal(
                    order, torch.arange(values.shape[-1], dtype=torch.long)
                ):
                    order = torch.roll(order, 1)
                result[batch, head] = values[batch, head, order.to(values.device)]
        return result

    def _eligible_slots(self, length: int, *, device: torch.device) -> torch.Tensor:
        stop = length - self.protected_recent
        if self.protected_sinks >= stop:
            raise RuntimeError(
                "eligible-pool v2 control has fewer than two heavy-ranking candidates after "
                "sink/recent exclusion"
            )
        slots = torch.arange(self.protected_sinks, stop, dtype=torch.long, device=device)
        if slots.numel() < 2:
            raise RuntimeError("eligible-pool v2 controls require at least two eligible entries")
        return slots

    def _eligible_shuffled_v2(
        self,
        values: torch.Tensor,
        entry_ids: torch.Tensor,
        *,
        boundary: int,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Seeded cycle permutation over eligible entries, with no fixed semantic points."""

        result = values.clone()
        sources = entry_ids.clone()
        slots = self._eligible_slots(values.shape[-1], device=values.device)
        local_identity = torch.arange(slots.numel(), dtype=torch.long)
        for batch in range(values.shape[0]):
            for head in range(values.shape[1]):
                generator = torch.Generator(device="cpu").manual_seed(
                    _stable_seed(
                        self.spec.seed,
                        self.case_id,
                        self.spec.condition_id,
                        boundary,
                        layer,
                        batch,
                        head,
                        "eligible-count-shuffle-v2",
                    )
                )
                # A random cyclic ordering is a deterministic derangement: every
                # eligible destination receives C from another semantic entry.
                cycle = torch.randperm(slots.numel(), generator=generator)
                source_for_destination = local_identity.clone()
                source_for_destination[cycle] = torch.roll(cycle, shifts=1)
                source_slots = slots[source_for_destination.to(slots.device)]
                result[batch, head, slots] = values[batch, head, source_slots]
                sources[batch, head, slots] = entry_ids[batch, head, source_slots]
        self._assert_eligible_v2_invariants(values, result, entry_ids, sources, slots)
        return result, sources

    def _eligible_circular_v2(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        entry_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Circularly rotate C in semantic position order within the eligible pool only."""

        result = values.clone()
        sources = entry_ids.clone()
        slots = self._eligible_slots(values.shape[-1], device=values.device)
        for batch in range(values.shape[0]):
            for head in range(values.shape[1]):
                ordered_slots = sorted(
                    slots.tolist(),
                    key=lambda slot: (
                        int(positions[batch, head, slot].item()),
                        int(entry_ids[batch, head, slot].item()),
                    ),
                )
                order = torch.tensor(ordered_slots, dtype=torch.long, device=values.device)
                shift = self.spec.circular_shift % int(order.numel())
                if shift == 0:
                    raise RuntimeError(
                        "eligible circular shift is an identity for the realized pool size"
                    )
                source_slots = torch.roll(order, shifts=shift, dims=0)
                result[batch, head, order] = values[batch, head, source_slots]
                sources[batch, head, order] = entry_ids[batch, head, source_slots]
        self._assert_eligible_v2_invariants(values, result, entry_ids, sources, slots)
        return result, sources

    @staticmethod
    def _assert_eligible_v2_invariants(
        true_counts: torch.Tensor,
        effective_counts: torch.Tensor,
        entry_ids: torch.Tensor,
        source_entry_ids: torch.Tensor,
        eligible_slots: torch.Tensor,
    ) -> None:
        """Fail closed if a clean intervention crosses protection or loses identity."""

        length = true_counts.shape[-1]
        protected = torch.ones(length, dtype=torch.bool, device=true_counts.device)
        protected[eligible_slots] = False
        if not torch.equal(effective_counts[..., protected], true_counts[..., protected]):
            raise RuntimeError("eligible-pool v2 transform changed a protected denominator")
        if not torch.equal(source_entry_ids[..., protected], entry_ids[..., protected]):
            raise RuntimeError("eligible-pool v2 transform changed protected semantic identity")
        if not torch.equal(
            torch.sort(effective_counts[..., eligible_slots], dim=-1).values,
            torch.sort(true_counts[..., eligible_slots], dim=-1).values,
        ):
            raise RuntimeError("eligible-pool v2 transform changed the eligible C multiset")
        if not torch.equal(
            torch.sort(source_entry_ids[..., eligible_slots], dim=-1).values,
            torch.sort(entry_ids[..., eligible_slots], dim=-1).values,
        ):
            raise RuntimeError("eligible-pool v2 transform changed the eligible identity multiset")
        if torch.any(source_entry_ids[..., eligible_slots] == entry_ids[..., eligible_slots]):
            raise RuntimeError("eligible-pool v2 transform left a semantic fixed point")

    def _wrong_token(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        entry_ids: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty_like(values)
        for batch in range(values.shape[0]):
            for head in range(values.shape[1]):
                # Stable semantic order, independent of the current physical slots.
                order = sorted(
                    range(values.shape[-1]),
                    key=lambda slot: (
                        int(positions[batch, head, slot].item()),
                        int(entry_ids[batch, head, slot].item()),
                    ),
                )
                order_tensor = torch.tensor(order, dtype=torch.long, device=values.device)
                shifted = torch.roll(
                    values[batch, head, order_tensor],
                    shifts=self.spec.circular_shift,
                    dims=0,
                )
                result[batch, head, order_tensor] = shifted
        return result

    @staticmethod
    def _head_average(
        values: torch.Tensor,
        positions: torch.Tensor,
        entry_ids: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty_like(values)
        for batch in range(values.shape[0]):
            grouped: dict[int, list[tuple[int, int, float, int]]] = {}
            for head in range(values.shape[1]):
                for slot in range(values.shape[2]):
                    entry_id = int(entry_ids[batch, head, slot].item())
                    grouped.setdefault(entry_id, []).append(
                        (
                            head,
                            slot,
                            float(values[batch, head, slot].item()),
                            int(positions[batch, head, slot].item()),
                        )
                    )
            for entry_id, rows in grouped.items():
                if len({row[3] for row in rows}) != 1:
                    raise RuntimeError(
                        f"semantic entry {entry_id} maps to inconsistent absolute positions"
                    )
                mean = sum(row[2] for row in rows) / len(rows)
                for head, slot, _, _ in rows:
                    result[batch, head, slot] = mean
        return result

    def _delayed(
        self,
        entry_ids: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        result = torch.zeros(entry_ids.shape, dtype=torch.float32, device=entry_ids.device)
        if self._previous_by_layer is None:
            return result
        previous = self._previous_by_layer[layer_index]
        for batch in range(entry_ids.shape[0]):
            for head in range(entry_ids.shape[1]):
                for slot in range(entry_ids.shape[2]):
                    key = (batch, head, int(entry_ids[batch, head, slot].item()))
                    result[batch, head, slot] = previous.get(key, 0.0)
        return result

    def _noisy(self, values: torch.Tensor, *, boundary: int, layer: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed(
                self.spec.seed,
                self.case_id,
                self.spec.condition_id,
                boundary,
                layer,
                "count-noise",
            )
        )
        mean = float(values.float().mean().item())
        scale = mean * self.spec.noise_std_fraction_of_mean
        noise = torch.randn(values.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(device=values.device, dtype=values.dtype) * scale
        return (values + noise).clamp_min(self.spec.noisy_clamp_min)

    def effective_counts(
        self,
        cache: SignificanceCache,
        *,
        boundary_index: int,
    ) -> list[torch.Tensor]:
        """Compute this boundary's denominator and advance delayed-count history.

        Delayed history is keyed by ``(batch, KV head, semantic entry ID)`` rather
        than physical slot, so per-head gather and eviction cannot misalign it.
        """

        if boundary_index < 0:
            raise ValueError("boundary_index must be non-negative")
        cache.assert_forward_complete()
        if not all(layer.is_initialized and layer.state is not None for layer in cache.layers):
            raise RuntimeError("count controls require every model layer to be initialized")
        current_maps: list[dict[tuple[int, int, int], float]] = []
        transformed: list[torch.Tensor] = []
        source_entry_ids_by_layer: list[torch.Tensor | None] = []
        for layer_index, layer in enumerate(cache.layers):
            state = layer.state
            assert state is not None
            values = state.counts
            positions = state.positions
            entry_ids = state.entry_ids
            current_maps.append(_entry_count_map(entry_ids, values))
            if self.spec.transform == "true":
                result = values.clone()
                source_entry_ids: torch.Tensor | None = entry_ids.clone()
            elif self.spec.transform == "constant":
                result = torch.ones_like(values)
                source_entry_ids = None
            elif self.spec.transform == "age_derived":
                age = (cache.true_length - 1 - positions).clamp_min(0)
                result = torch.div(age, self.chunk_size, rounding_mode="floor").to(values.dtype) + 1
                source_entry_ids = None
            elif self.spec.transform == "shuffled":
                result = self._permuted(values, boundary=boundary_index, layer=layer_index)
                source_entry_ids = None
            elif self.spec.transform == "wrong_token":
                result = self._wrong_token(values, positions, entry_ids)
                source_entry_ids = None
            elif self.spec.transform == "delayed":
                result = self._delayed(entry_ids, layer_index=layer_index).to(values.dtype)
                source_entry_ids = None
            elif self.spec.transform == "noisy":
                result = self._noisy(values, boundary=boundary_index, layer=layer_index)
                source_entry_ids = None
            elif self.spec.transform == "head_averaged":
                result = self._head_average(values, positions, entry_ids)
                source_entry_ids = None
            elif self.spec.transform == "eligible_shuffled_v2":
                result, source_entry_ids = self._eligible_shuffled_v2(
                    values,
                    entry_ids,
                    boundary=boundary_index,
                    layer=layer_index,
                )
            elif self.spec.transform == "eligible_circular_v2":
                result, source_entry_ids = self._eligible_circular_v2(
                    values,
                    positions,
                    entry_ids,
                )
            else:  # pragma: no cover - CountControlSpec rejects this
                raise AssertionError(self.spec.transform)
            if result.shape != values.shape or result.device != values.device:
                raise RuntimeError("count control changed denominator shape or device")
            if not torch.isfinite(result).all() or (result < 0).any():
                raise RuntimeError("count control produced non-finite or negative values")
            transformed.append(result.contiguous())
            source_entry_ids_by_layer.append(
                None if source_entry_ids is None else source_entry_ids.contiguous()
            )
        self._previous_by_layer = current_maps
        self._effective_source_entry_ids = source_entry_ids_by_layer
        return transformed

    def compress(
        self,
        cache: SignificanceCache,
        policy: SignificancePolicy,
        boundary_index: int,
    ) -> CompressionReport:
        """Compression callback compatible with ``chunked_prefill``."""

        if policy.ranking_mode != "significance" or policy.count_exponent != 1.0:
            raise ValueError("count causal controls require significance ranking with exponent 1.0")
        if self.spec.transform in _ELIGIBLE_INTERVENTIONS_V2 and (
            policy.sinks != self.protected_sinks or policy.recent != self.protected_recent
        ):
            raise ValueError(
                "eligible-pool v2 protection must exactly match the compression policy"
            )
        counts = self.effective_counts(cache, boundary_index=boundary_index)
        before = cache.stored_length()
        true_length = cache.true_length
        evicted = 0
        compressed_layers = 0
        kv_heads: int | None = None
        selected_per_head: int | None = None
        for layer_index, (layer, effective) in enumerate(zip(cache.layers, counts, strict=True)):
            state = layer.state
            assert state is not None
            prefix = f"b{boundary_index:03d}_l{layer_index:03d}"
            positions = state.positions.detach().cpu().clone()
            entry_ids = state.entry_ids.detach().cpu().clone()
            scores = state.scores.detach().cpu().clone()
            true_counts = state.counts.detach().cpu().clone()
            effective_cpu = effective.detach().cpu().clone()
            ranking = scores / effective_cpu.clamp_min(1.0)
            old = layer.stored_length()
            indices = layer.compress(policy, ranking_counts=effective)
            selected = torch.zeros_like(true_counts, dtype=torch.uint8)
            selected.scatter_(2, indices.detach().cpu(), 1)
            self._arrays[f"{prefix}_positions"] = positions.numpy()
            self._arrays[f"{prefix}_entry_ids"] = entry_ids.numpy()
            self._arrays[f"{prefix}_scores"] = scores.numpy()
            self._arrays[f"{prefix}_true_counts"] = true_counts.numpy()
            self._arrays[f"{prefix}_effective_counts"] = effective_cpu.numpy()
            source_entry_ids = self._effective_source_entry_ids[layer_index]
            if source_entry_ids is not None and (
                self.spec.transform in _ELIGIBLE_INTERVENTIONS_V2
                or self._has_explicit_protection
            ):
                self._arrays[f"{prefix}_effective_count_source_entry_ids"] = (
                    source_entry_ids.detach().cpu().numpy().copy()
                )
                eligible_mask = torch.zeros_like(true_counts, dtype=torch.uint8)
                eligible_slots = self._eligible_slots(
                    true_counts.shape[-1], device=true_counts.device
                )
                eligible_mask[..., eligible_slots] = 1
                self._arrays[f"{prefix}_eligible_heavy_mask"] = eligible_mask.numpy()
            self._arrays[f"{prefix}_ranking"] = ranking.numpy()
            self._arrays[f"{prefix}_selected"] = selected.numpy()
            self._arrays[f"{prefix}_retained_positions"] = (
                layer.state.positions.detach().cpu().numpy().copy()
            )
            if layer.stored_length() < old:
                compressed_layers += 1
                evicted += old - layer.stored_length()
            kv_heads = int(true_counts.shape[1])
            selected_per_head = int(indices.shape[-1])
        cache.assert_aligned()
        after = cache.stored_length()
        if kv_heads is None or selected_per_head is None:
            raise RuntimeError("count control compressed no layers")
        self._summaries.append(
            BoundarySummary(
                boundary_index=boundary_index,
                true_length=true_length,
                stored_before=before,
                stored_after=after,
                layer_count=len(cache.layers),
                kv_heads=kv_heads,
                selected_per_head=selected_per_head,
                transform=self.spec.transform,
            )
        )
        # CompressionReport counts entries per layer, matching SignificanceCache.compress.
        return CompressionReport(
            layers=compressed_layers,
            before_entries=before * len(cache.layers),
            after_entries=after * len(cache.layers),
            evictions=evicted,
        )


def boundary_resident_positions(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    *,
    boundary_index: int,
    layer_count: int,
    batch_index: int = 0,
) -> list[list[list[int]]]:
    """Decode one boundary into the existing exact-evidence scorer input.

    The result is ``[layers][KV heads][resident absolute positions]``.  Keeping
    this conversion beside the trace writer makes the persisted NPZ contract
    executable rather than relying on an undocumented array naming convention.
    """

    if boundary_index < 0 or layer_count <= 0 or batch_index < 0:
        raise ValueError("boundary/batch indices must be non-negative and layer_count positive")
    result: list[list[list[int]]] = []
    for layer_index in range(layer_count):
        name = f"b{boundary_index:03d}_l{layer_index:03d}_retained_positions"
        if name not in arrays:
            raise KeyError(f"boundary diagnostic is missing {name}")
        value = np.asarray(arrays[name])
        if value.ndim != 3 or batch_index >= value.shape[0]:
            raise ValueError(f"{name} must have shape [batch, KV heads, positions]")
        result.append(value[batch_index].astype(np.int64, copy=False).tolist())
    return result
