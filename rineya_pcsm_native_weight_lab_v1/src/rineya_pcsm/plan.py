"""Immutable qualification plans connecting peer geometry to native updates."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from torch import Tensor

from .descriptors import MAX_DESCRIPTOR_DIMENSION, MIN_DESCRIPTOR_DIMENSION
from .topk import budget_to_k, flat_support_indices, freeze_support

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class TensorPlan:
    """One target's exact support, stabilizer, and peer-attention evidence."""

    name: str
    role: str
    shape: tuple[int, ...]
    support_mask: Tensor
    stabilizer: Tensor
    peer_names: tuple[str, ...]
    attention_alphas: Tensor
    stabilizer_lower: float = 0.25
    stabilizer_upper: float = 1.0
    module_type: str = ""
    layer_depth: int | None = None
    support_fraction: float | None = None
    feature_names: tuple[str, ...] = ()
    feature_weights: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )
    generator_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.role:
            raise ValueError("tensor plan name and role must be nonempty")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor plan shape must contain positive dimensions")
        expected_numel = 1
        for dimension in self.shape:
            expected_numel *= dimension
        if self.support_mask.numel() != expected_numel:
            raise ValueError("support mask numel does not match declared shape")
        if tuple(self.support_mask.shape) != self.shape:
            raise ValueError("support mask shape does not match declared shape")
        if tuple(self.stabilizer.shape) != self.shape:
            raise ValueError("stabilizer shape does not match declared shape")
        if not (
            math.isfinite(self.stabilizer_lower)
            and math.isfinite(self.stabilizer_upper)
            and 0.0 < self.stabilizer_lower < self.stabilizer_upper
        ):
            raise ValueError("stabilizer bounds must be finite and ordered")
        mask = self.support_mask.detach().bool().cpu().contiguous()
        # Validate exact binary values before bool conversion.
        original_mask = self.support_mask.detach()
        if original_mask.dtype != torch.bool and not bool(
            torch.logical_or(original_mask == 0, original_mask == 1).all()
        ):
            raise ValueError("support mask must be exactly binary")
        stabilizer = self.stabilizer.detach().float().cpu().contiguous()
        if not bool(torch.isfinite(stabilizer).all()):
            raise ValueError("stabilizer must be finite")
        if bool((stabilizer < self.stabilizer_lower).any()) or bool(
            (stabilizer > self.stabilizer_upper).any()
        ):
            raise ValueError("stabilizer lies outside its declared bounds")
        if len(set(self.peer_names)) != len(self.peer_names):
            raise ValueError("peer names must be unique")
        if self.name in self.peer_names:
            raise ValueError("self advice is forbidden in persisted plans")
        alphas = self.attention_alphas.detach().float().cpu().reshape(-1).contiguous()
        if alphas.numel() != len(self.peer_names):
            raise ValueError("one attention alpha is required for every peer")
        if not bool(torch.isfinite(alphas).all()) or bool((alphas < 0).any()):
            raise ValueError("attention alphas must be finite and nonnegative")
        if alphas.numel() and not torch.isclose(
            alphas.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("attention alphas must sum to one")
        k = int(mask.sum().item())
        if k <= 0:
            raise ValueError("tensor plan support must select at least one coordinate")
        support_fraction = (
            k / expected_numel
            if self.support_fraction is None
            else float(self.support_fraction)
        )
        if (
            not math.isfinite(support_fraction)
            or not 0.0 < support_fraction <= 1.0
            or budget_to_k(expected_numel, support_fraction) != k
        ):
            raise ValueError("support fraction does not resolve to exact K")
        module_type = self.module_type.strip() or self.role
        if self.layer_depth is not None and (
            isinstance(self.layer_depth, bool) or self.layer_depth < 0
        ):
            raise ValueError("layer depth must be a nonnegative integer or None")
        feature_names = tuple(self.feature_names)
        if len(feature_names) != len(set(feature_names)) or any(
            not name for name in feature_names
        ):
            raise ValueError("feature names must be unique and nonempty")
        feature_weights = (
            self.feature_weights.detach().float().cpu().reshape(-1).contiguous()
        )
        if feature_weights.numel() != len(feature_names):
            raise ValueError("one persisted weight is required for every feature")
        if not bool(torch.isfinite(feature_weights).all()) or bool(
            (feature_weights < 0).any()
        ):
            raise ValueError("feature weights must be finite and nonnegative")
        if feature_weights.numel() and not torch.isclose(
            feature_weights.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("feature weights must sum to one")
        generator_config = dict(self.generator_config)
        try:
            json.dumps(generator_config, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("generator config must be JSON-compatible") from error
        object.__setattr__(self, "support_mask", mask)
        object.__setattr__(self, "stabilizer", stabilizer)
        object.__setattr__(self, "attention_alphas", alphas)
        object.__setattr__(self, "peer_names", tuple(self.peer_names))
        object.__setattr__(self, "module_type", module_type)
        object.__setattr__(self, "support_fraction", support_fraction)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_weights", feature_weights)
        object.__setattr__(self, "generator_config", MappingProxyType(generator_config))

    @property
    def numel(self) -> int:
        return self.support_mask.numel()

    @property
    def k(self) -> int:
        return int(self.support_mask.sum().item())

    @property
    def flat_indices(self) -> Tensor:
        """Canonical sorted indices for conversion to the native update plan."""

        return flat_support_indices(self.support_mask)

    def audit_record(self) -> dict[str, object]:
        """Return required per-target JSON manifest evidence."""

        return {
            "parameter_name": self.name,
            "semantic_role": self.role,
            "module_type": self.module_type,
            "layer_depth": self.layer_depth,
            "shape": list(self.shape),
            "exact_k": self.k,
            "requested_fraction": self.support_fraction,
            "selected_fraction": self.k / self.numel,
            "feature_names": list(self.feature_names),
            "feature_weights": self.feature_weights.tolist(),
            "generator_config": dict(self.generator_config),
        }


def _hydrate_entry_from_record(
    entry: TensorPlan, record: Mapping[str, object]
) -> TensorPlan:
    """Restore audit-only fields serialized through plan metadata."""

    raw_names = record.get("feature_names", entry.feature_names)
    raw_weights = record.get("feature_weights", entry.feature_weights.tolist())
    raw_config = record.get("generator_config", entry.generator_config)
    if not isinstance(raw_names, (list, tuple)):
        raise TypeError(f"invalid feature names for {entry.name}")
    if not isinstance(raw_weights, (list, tuple)):
        raise TypeError(f"invalid feature weights for {entry.name}")
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"invalid generator config for {entry.name}")
    raw_depth = record.get("layer_depth", entry.layer_depth)
    if raw_depth is None:
        layer_depth = None
    elif isinstance(raw_depth, int) and not isinstance(raw_depth, bool):
        layer_depth = raw_depth
    else:
        raise TypeError(f"invalid layer depth for {entry.name}")
    raw_fraction = record.get("requested_fraction", entry.support_fraction)
    if not isinstance(raw_fraction, (int, float)) or isinstance(raw_fraction, bool):
        raise TypeError(f"invalid support fraction for {entry.name}")
    try:
        return replace(
            entry,
            module_type=str(record.get("module_type", entry.module_type)),
            layer_depth=layer_depth,
            support_fraction=float(raw_fraction),
            feature_names=tuple(str(item) for item in raw_names),
            feature_weights=torch.tensor(raw_weights, dtype=torch.float32),
            generator_config=dict(raw_config),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid audit record for {entry.name}") from error


@dataclass(frozen=True, slots=True)
class PCSMPlan:
    """Fail-closed plan tied to one exact base-state fingerprint."""

    base_fingerprint: str
    descriptor_dimension: int
    entries: tuple[TensorPlan, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported PCSM plan schema: {self.schema_version}")
        if not _SHA256_PATTERN.fullmatch(self.base_fingerprint):
            raise ValueError("base fingerprint must be a lowercase SHA-256 digest")
        if not (
            MIN_DESCRIPTOR_DIMENSION
            <= self.descriptor_dimension
            <= MAX_DESCRIPTOR_DIMENSION
        ):
            raise ValueError("descriptor dimension is outside the qualified range")
        if not self.entries:
            raise ValueError("a PCSM plan must contain at least one tensor entry")
        metadata = dict(self.metadata)
        entries = tuple(self.entries)
        raw_records = metadata.get("target_plan_records")
        if raw_records is not None:
            if not isinstance(raw_records, Mapping):
                raise ValueError("target_plan_records metadata must be an object")
            hydrated: list[TensorPlan] = []
            for entry in entries:
                raw_record = raw_records.get(entry.name)
                if raw_record is None:
                    hydrated.append(entry)
                elif isinstance(raw_record, Mapping):
                    hydrated.append(_hydrate_entry_from_record(entry, raw_record))
                else:
                    raise ValueError("target plan audit record must be an object")
            entries = tuple(hydrated)
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("PCSM plan target names must be unique")
        metadata["target_plan_records"] = {
            entry.name: entry.audit_record() for entry in entries
        }
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("plan metadata must be JSON-compatible") from error
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def total_numel(self) -> int:
        return sum(entry.numel for entry in self.entries)

    @property
    def union_k(self) -> int:
        # Names define disjoint parameter domains, so global union is additive.
        return sum(entry.k for entry in self.entries)

    def support_stats(self) -> dict[str, object]:
        return {
            "per_target_k": {entry.name: entry.k for entry in self.entries},
            "total_numel": self.total_numel,
            "union_k": self.union_k,
            "union_fraction": self.union_k / self.total_numel,
        }


def make_tensor_plan(
    *,
    name: str,
    role: str,
    scores: Tensor,
    stabilizer: Tensor,
    k: int,
    peer_names: tuple[str, ...],
    attention_alphas: Tensor,
    stabilizer_lower: float = 0.25,
    stabilizer_upper: float = 1.0,
    module_type: str = "",
    layer_depth: int | None = None,
    support_fraction: float | None = None,
    feature_names: tuple[str, ...] = (),
    feature_weights: Tensor | None = None,
    generator_config: Mapping[str, object] | None = None,
) -> TensorPlan:
    """Freeze TopK once and construct a validated tensor plan."""

    support = freeze_support(scores, k)
    return TensorPlan(
        name=name,
        role=role,
        shape=tuple(scores.shape),
        support_mask=support.mask,
        stabilizer=stabilizer,
        peer_names=peer_names,
        attention_alphas=attention_alphas,
        stabilizer_lower=stabilizer_lower,
        stabilizer_upper=stabilizer_upper,
        module_type=module_type,
        layer_depth=layer_depth,
        support_fraction=support_fraction,
        feature_names=feature_names,
        feature_weights=(
            torch.empty(0, dtype=torch.float32)
            if feature_weights is None
            else feature_weights
        ),
        generator_config={} if generator_config is None else generator_config,
    )


__all__ = ["PCSMPlan", "TensorPlan", "make_tensor_plan"]
