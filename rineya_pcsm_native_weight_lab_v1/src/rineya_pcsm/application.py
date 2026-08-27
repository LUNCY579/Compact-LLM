"""Apply an immutable exact-coordinate update plan to native parameters.

The plan types in this module deliberately contain only names, flat indices, and
per-coordinate stabilizers.  That small protocol lets a geometry/planning layer
persist richer artifacts without coupling it to the optimizer implementation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def _cpu_indices(value: torch.Tensor | Sequence[int]) -> torch.Tensor:
    indices = torch.as_tensor(value, dtype=torch.int64, device="cpu").detach().clone()
    if indices.ndim != 1:
        raise ValueError("flat_indices must be one-dimensional")
    return indices.contiguous()


def _cpu_stabilizer(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    stabilizer = (
        torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()
    )
    if stabilizer.ndim != 1:
        raise ValueError("stabilizer must be one-dimensional")
    return stabilizer.contiguous()


@dataclass(frozen=True, slots=True)
class TensorUpdatePlan:
    """Exact support and post-Adam stabilizer for one named parameter tensor."""

    name: str
    numel: int
    flat_indices: torch.Tensor | Sequence[int]
    stabilizer: torch.Tensor | Sequence[float]
    shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plan tensor name must be non-empty")
        if self.numel <= 0:
            raise ValueError("plan tensor numel must be positive")

        indices = _cpu_indices(self.flat_indices)
        scale = _cpu_stabilizer(self.stabilizer)
        if indices.numel() == 0:
            raise ValueError(f"{self.name}: exact support must not be empty")
        if indices.numel() != scale.numel():
            raise ValueError(f"{self.name}: indices and stabilizer lengths differ")
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= self.numel)):
            raise ValueError(f"{self.name}: support index is out of range")
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"{self.name}: stabilizer contains a non-finite value")
        if bool(torch.any(scale <= 0)):
            raise ValueError(f"{self.name}: stabilizer must be strictly positive")

        order = torch.argsort(indices, stable=True)
        indices = indices.index_select(0, order)
        scale = scale.index_select(0, order)
        if indices.numel() > 1 and bool(torch.any(indices[1:] == indices[:-1])):
            raise ValueError(f"{self.name}: support indices must be unique")

        normalized_shape: tuple[int, ...] | None = None
        if self.shape is not None:
            normalized_shape = tuple(int(size) for size in self.shape)
            if any(size < 0 for size in normalized_shape):
                raise ValueError(f"{self.name}: shape dimensions must be non-negative")
            if math.prod(normalized_shape) != self.numel:
                raise ValueError(f"{self.name}: shape does not match numel")

        object.__setattr__(self, "flat_indices", indices)
        object.__setattr__(self, "stabilizer", scale)
        object.__setattr__(self, "shape", normalized_shape)

    @property
    def k(self) -> int:
        """Number of selected coordinates."""

        return int(cast(torch.Tensor, self.flat_indices).numel())

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        result: dict[str, object] = {
            "name": self.name,
            "numel": self.numel,
            "flat_indices": cast(torch.Tensor, self.flat_indices).tolist(),
            "stabilizer": cast(torch.Tensor, self.stabilizer).tolist(),
        }
        if self.shape is not None:
            result["shape"] = list(self.shape)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorUpdatePlan:
        """Parse the minimal plan representation from a persisted artifact."""

        name = value.get("name")
        numel = value.get("numel")
        indices = value.get("flat_indices")
        stabilizer = value.get("stabilizer")
        shape_value = value.get("shape")
        if not isinstance(name, str) or not isinstance(numel, int):
            raise TypeError("plan entry requires string name and integer numel")
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise TypeError(f"{name}: flat_indices must be a sequence")
        if not isinstance(stabilizer, Sequence) or isinstance(stabilizer, (str, bytes)):
            raise TypeError(f"{name}: stabilizer must be a sequence")
        shape: tuple[int, ...] | None = None
        if shape_value is not None:
            if not isinstance(shape_value, Sequence) or isinstance(
                shape_value, (str, bytes)
            ):
                raise TypeError(f"{name}: shape must be a sequence")
            shape = tuple(int(item) for item in shape_value)
        return cls(
            name=name,
            numel=numel,
            flat_indices=[int(item) for item in indices],
            stabilizer=[float(item) for item in stabilizer],
            shape=shape,
        )


def _update_int(digest: Any, value: int) -> None:
    digest.update(struct.pack("<q", value))


def _canonical_hash(
    entries: Sequence[TensorUpdatePlan], *, include_stabilizer: bool
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"rineya-pcsm-plan-v1\0" if include_stabilizer else b"rineya-pcsm-support-v1\0"
    )
    for entry in sorted(entries, key=lambda item: item.name):
        encoded_name = entry.name.encode("utf-8")
        _update_int(digest, len(encoded_name))
        digest.update(encoded_name)
        _update_int(digest, entry.numel)
        shape = entry.shape or ()
        _update_int(digest, len(shape))
        for size in shape:
            _update_int(digest, size)
        _update_int(digest, entry.k)
        indices = cast(torch.Tensor, entry.flat_indices)
        for index in indices.tolist():
            _update_int(digest, int(index))
        if include_stabilizer:
            stabilizer = cast(torch.Tensor, entry.stabilizer)
            for scale in stabilizer.tolist():
                digest.update(struct.pack("<f", float(scale)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeUpdatePlan:
    """A complete, content-addressed exact-support native update plan."""

    entries: tuple[TensorUpdatePlan, ...] | Sequence[TensorUpdatePlan]
    schema_version: str = "1.0"
    support_hash: str = field(init=False)
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("native update plan must contain at least one tensor")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("native update plan contains duplicate tensor names")
        entries = tuple(sorted(entries, key=lambda entry: entry.name))
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self, "support_hash", _canonical_hash(entries, include_stabilizer=False)
        )
        object.__setattr__(
            self, "plan_hash", _canonical_hash(entries, include_stabilizer=True)
        )

    @property
    def k(self) -> int:
        """Total exact support size in the disjoint named-coordinate namespace."""

        return sum(entry.k for entry in self.entries)

    def to_dict(self) -> dict[str, object]:
        """Return a self-verifying, JSON-compatible plan artifact."""

        return {
            "schema_version": self.schema_version,
            "support_hash": self.support_hash,
            "plan_hash": self.plan_hash,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> NativeUpdatePlan:
        """Load and verify a plan artifact, including both content hashes."""

        raw_entries = value.get("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes)
        ):
            raise TypeError("plan artifact requires an entries sequence")
        parsed: list[TensorUpdatePlan] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise TypeError("each plan entry must be a mapping")
            parsed.append(TensorUpdatePlan.from_dict(raw_entry))
        schema = value.get("schema_version", "1.0")
        if not isinstance(schema, str):
            raise TypeError("schema_version must be a string")
        result = cls(tuple(parsed), schema_version=schema)
        for key, actual in (
            ("support_hash", result.support_hash),
            ("plan_hash", result.plan_hash),
        ):
            declared = value.get(key)
            if declared is not None and declared != actual:
                raise ValueError(f"declared {key} does not match plan contents")
        return result


@dataclass(frozen=True, slots=True)
class AppliedTensorPlan:
    """A validated plan entry bound to its live native parameter."""

    name: str
    parameter: nn.Parameter
    flat_indices: torch.Tensor
    stabilizer: torch.Tensor
    numel: int
    shape: tuple[int, ...]

    @property
    def k(self) -> int:
        return int(self.flat_indices.numel())


@dataclass(slots=True)
class AppliedUpdatePlan:
    """Live exact-support plan and removable coordinate-gradient guards."""

    entries: tuple[AppliedTensorPlan, ...]
    support_hash: str
    plan_hash: str
    _hook_handles: list[RemovableHandle] = field(default_factory=list, repr=False)
    _original_requires_grad: dict[str, tuple[nn.Parameter, bool]] = field(
        default_factory=dict, repr=False
    )

    @property
    def k(self) -> int:
        return sum(entry.k for entry in self.entries)

    def close(self) -> None:
        """Remove gradient guards while keeping parameter freeze flags in place."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def restore(self) -> None:
        """Remove guards and restore pre-application ``requires_grad`` flags."""

        self.close()
        for parameter, original in self._original_requires_grad.values():
            parameter.requires_grad_(original)

    def __enter__(self) -> AppliedUpdatePlan:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.restore()


def _coordinate_gradient_guard(
    flat_indices: torch.Tensor,
) -> Any:
    indices_cpu = flat_indices.detach().to(dtype=torch.int64, device="cpu").clone()

    def guard(gradient: torch.Tensor) -> torch.Tensor:
        if gradient.is_sparse:
            raise RuntimeError("PCSM exact-support plans require dense gradients")
        indices = indices_cpu.to(device=gradient.device)
        flat_gradient = gradient.reshape(-1)
        selected = flat_gradient.index_select(0, indices)
        guarded = torch.zeros_like(flat_gradient)
        guarded.index_copy_(0, indices, selected)
        return guarded.reshape_as(gradient)

    return guard


def _validate_parameter_binding(
    entry: TensorUpdatePlan, parameter: nn.Parameter
) -> tuple[int, ...]:
    if parameter.numel() != entry.numel:
        raise ValueError(
            f"{entry.name}: plan numel {entry.numel} != parameter numel "
            f"{parameter.numel()}"
        )
    shape = tuple(parameter.shape)
    if entry.shape is not None and shape != entry.shape:
        raise ValueError(
            f"{entry.name}: plan shape {entry.shape} != parameter shape {shape}"
        )
    if not parameter.is_contiguous():
        raise ValueError(f"{entry.name}: native parameter must be contiguous")
    return shape


def apply_update_plan(
    module: nn.Module,
    plan: NativeUpdatePlan,
    *,
    install_gradient_guards: bool = True,
) -> AppliedUpdatePlan:
    """Freeze a module to a plan and bind entries for ``ExactSupportAdamW``.

    Whole unplanned tensors are disabled with ``requires_grad=False``.  For a
    planned tensor, an optional autograd hook zeros every unselected coordinate.
    Native parameter writes still belong to ``ExactSupportAdamW``, whose indexed
    write is what prevents momentum and weight decay from touching frozen values.
    """

    named_parameters = dict(module.named_parameters())
    planned_names = {entry.name for entry in plan.entries}
    missing = planned_names.difference(named_parameters)
    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"plan references missing parameters: {joined}")

    validated: list[tuple[TensorUpdatePlan, nn.Parameter, tuple[int, ...]]] = []
    for entry in plan.entries:
        parameter = named_parameters[entry.name]
        validated.append(
            (entry, parameter, _validate_parameter_binding(entry, parameter))
        )

    original: dict[str, tuple[nn.Parameter, bool]] = {}
    for name, parameter in named_parameters.items():
        original[name] = (parameter, parameter.requires_grad)
        if name not in planned_names:
            parameter.grad = None
            parameter.requires_grad_(False)

    applied_entries: list[AppliedTensorPlan] = []
    handles: list[RemovableHandle] = []
    try:
        for entry, parameter, shape in validated:
            parameter.requires_grad_(True)
            if install_gradient_guards and entry.k < entry.numel:
                indices = cast(torch.Tensor, entry.flat_indices)
                guard = _coordinate_gradient_guard(indices)
                if parameter.grad is not None:
                    parameter.grad = guard(parameter.grad)
                handles.append(
                    parameter.register_hook(  # type: ignore[no-untyped-call]
                        guard
                    )
                )
            entry_indices = cast(torch.Tensor, entry.flat_indices)
            entry_stabilizer = cast(torch.Tensor, entry.stabilizer)
            applied_entries.append(
                AppliedTensorPlan(
                    name=entry.name,
                    parameter=parameter,
                    flat_indices=entry_indices.detach().clone(),
                    stabilizer=entry_stabilizer.detach().clone(),
                    numel=entry.numel,
                    shape=shape,
                )
            )
    except Exception:
        for handle in handles:
            handle.remove()
        for parameter, flag in original.values():
            parameter.requires_grad_(flag)
        raise

    return AppliedUpdatePlan(
        entries=tuple(applied_entries),
        support_hash=plan.support_hash,
        plan_hash=plan.plan_hash,
        _hook_handles=handles,
        _original_requires_grad=original,
    )
