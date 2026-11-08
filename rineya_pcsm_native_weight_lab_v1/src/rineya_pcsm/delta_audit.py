"""Bitwise native-weight delta auditing for exact-support PCSM updates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .application import NativeUpdatePlan


@dataclass(frozen=True, slots=True)
class ParameterSnapshot:
    """Detached CPU copies used for an independent before/after delta audit."""

    parameters: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TensorDeltaAudit:
    """Exact-support delta statistics for one named native tensor."""

    name: str
    numel: int
    selected_k: int
    changed_union_size: int
    selected_changed_count: int
    outside_support_changed_count: int
    selected_delta_l2: float
    outside_support_delta_l2: float
    total_delta_l2: float
    selected_delta_max_abs: float
    outside_support_delta_max_abs: float


@dataclass(frozen=True, slots=True)
class DeltaAuditReport:
    """Aggregate support-leakage verdict and update norm report."""

    valid: bool
    planned_k: int
    support_union_size: int
    total_native_coordinates: int
    support_union_fraction: float
    changed_union_size: int
    selected_changed_count: int
    outside_support_changed_count: int
    selected_delta_l2: float
    outside_support_delta_l2: float
    total_delta_l2: float
    selected_delta_max_abs: float
    outside_support_delta_max_abs: float
    plan_hash: str
    support_hash: str
    leakage_examples: tuple[tuple[str, int], ...]
    tensors: tuple[TensorDeltaAudit, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit record."""

        result = asdict(self)
        result["leakage_examples"] = [
            {"name": name, "flat_index": index} for name, index in self.leakage_examples
        ]
        return result


def capture_parameter_snapshot(module: nn.Module) -> ParameterSnapshot:
    """Capture every named parameter without dtype conversion."""

    return ParameterSnapshot(
        {
            name: parameter.detach().to(device="cpu").contiguous().clone()
            for name, parameter in module.named_parameters()
        }
    )


def _snapshot_from_after(
    after: nn.Module | ParameterSnapshot | Mapping[str, torch.Tensor],
) -> ParameterSnapshot:
    if isinstance(after, nn.Module):
        return capture_parameter_snapshot(after)
    if isinstance(after, ParameterSnapshot):
        return after
    return ParameterSnapshot(after)


def _bitwise_changed(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if before.dtype != after.dtype:
        raise ValueError("delta audit requires identical before/after dtypes")
    if before.shape != after.shape:
        raise ValueError("delta audit requires identical before/after shapes")
    if before.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    before_bytes = (
        before.contiguous()
        .view(torch.uint8)
        .reshape(before.numel(), before.element_size())
    )
    after_bytes = (
        after.contiguous()
        .view(torch.uint8)
        .reshape(after.numel(), after.element_size())
    )
    return torch.any(before_bytes != after_bytes, dim=1)


def _numeric_delta(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if before.is_complex():
        return after.to(torch.complex128) - before.to(torch.complex128)
    return after.to(torch.float64) - before.to(torch.float64)


def _masked_norms(delta: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    if not bool(torch.any(mask)):
        return 0.0, 0.0
    values = torch.abs(delta.reshape(-1)[mask])
    norm = float(torch.linalg.vector_norm(values).item())
    maximum = float(torch.max(values).item())
    if not math.isfinite(norm) or not math.isfinite(maximum):
        return math.inf, math.inf
    return norm, maximum


def audit_parameter_delta(
    before: ParameterSnapshot | Mapping[str, torch.Tensor],
    after: nn.Module | ParameterSnapshot | Mapping[str, torch.Tensor],
    plan: NativeUpdatePlan,
    *,
    max_leakage_examples: int = 32,
    native_coordinate_count_override: int | None = None,
) -> DeltaAuditReport:
    """Bitwise-audit all native parameters against a fixed exact support.

    Any changed bit outside a selected coordinate invalidates the report.  This
    intentionally catches changes that have zero numerical norm, such as
    ``+0.0`` becoming ``-0.0``, and changes in NaN payloads.
    """

    if max_leakage_examples < 0:
        raise ValueError("max_leakage_examples must be non-negative")
    before_snapshot = (
        before if isinstance(before, ParameterSnapshot) else ParameterSnapshot(before)
    )
    after_snapshot = _snapshot_from_after(after)
    before_names = set(before_snapshot.parameters)
    after_names = set(after_snapshot.parameters)
    if before_names != after_names:
        missing = sorted(before_names.difference(after_names))
        extra = sorted(after_names.difference(before_names))
        raise ValueError(
            "parameter name set changed during update; "
            f"missing={missing}, extra={extra}"
        )
    total_native_coordinates = sum(
        tensor.numel() for tensor in before_snapshot.parameters.values()
    )
    if native_coordinate_count_override is not None:
        if native_coordinate_count_override < total_native_coordinates:
            raise ValueError(
                "native coordinate override cannot be smaller than audited tensors"
            )
        total_native_coordinates = native_coordinate_count_override

    plan_by_name = {entry.name: entry for entry in plan.entries}
    absent = set(plan_by_name).difference(before_names)
    if absent:
        raise KeyError(f"plan parameters absent from snapshots: {sorted(absent)}")

    tensor_reports: list[TensorDeltaAudit] = []
    leakage_examples: list[tuple[str, int]] = []
    aggregate_selected_squared = 0.0
    aggregate_outside_squared = 0.0
    aggregate_total_squared = 0.0
    aggregate_selected_max = 0.0
    aggregate_outside_max = 0.0
    changed_union = 0
    selected_changed = 0
    outside_changed = 0

    for name in sorted(before_names):
        before_tensor = before_snapshot.parameters[name].detach().to(device="cpu")
        after_tensor = after_snapshot.parameters[name].detach().to(device="cpu")
        changed = _bitwise_changed(before_tensor, after_tensor)
        delta = _numeric_delta(before_tensor, after_tensor).reshape(-1)
        # Identical NaNs have zero delta for auditing purposes.  A changed NaN
        # payload remains non-finite and therefore reports an infinite norm.
        delta = torch.where(changed, delta, torch.zeros_like(delta))
        entry = plan_by_name.get(name)
        selected_mask = torch.zeros(before_tensor.numel(), dtype=torch.bool)
        selected_k = 0
        if entry is not None:
            if entry.numel != before_tensor.numel():
                raise ValueError(f"{name}: plan numel does not match snapshot")
            if entry.shape is not None and entry.shape != tuple(before_tensor.shape):
                raise ValueError(f"{name}: plan shape does not match snapshot")
            selected_mask[entry.flat_indices] = True
            selected_k = entry.k

        selected_changed_mask = changed & selected_mask
        outside_changed_mask = changed & ~selected_mask
        selected_count = int(torch.count_nonzero(selected_changed_mask).item())
        outside_count = int(torch.count_nonzero(outside_changed_mask).item())
        changed_count = int(torch.count_nonzero(changed).item())
        selected_norm, selected_max = _masked_norms(delta, selected_mask)
        outside_norm, outside_max = _masked_norms(delta, ~selected_mask)
        total_norm, _ = _masked_norms(
            delta, torch.ones(before_tensor.numel(), dtype=torch.bool)
        )

        changed_union += changed_count
        selected_changed += selected_count
        outside_changed += outside_count
        aggregate_selected_squared += selected_norm**2
        aggregate_outside_squared += outside_norm**2
        aggregate_total_squared += total_norm**2
        aggregate_selected_max = max(aggregate_selected_max, selected_max)
        aggregate_outside_max = max(aggregate_outside_max, outside_max)

        if outside_count and len(leakage_examples) < max_leakage_examples:
            remaining = max_leakage_examples - len(leakage_examples)
            leak_indices = torch.nonzero(outside_changed_mask, as_tuple=False).reshape(
                -1
            )
            leakage_examples.extend(
                (name, int(index)) for index in leak_indices[:remaining].tolist()
            )

        tensor_reports.append(
            TensorDeltaAudit(
                name=name,
                numel=before_tensor.numel(),
                selected_k=selected_k,
                changed_union_size=changed_count,
                selected_changed_count=selected_count,
                outside_support_changed_count=outside_count,
                selected_delta_l2=selected_norm,
                outside_support_delta_l2=outside_norm,
                total_delta_l2=total_norm,
                selected_delta_max_abs=selected_max,
                outside_support_delta_max_abs=outside_max,
            )
        )

    return DeltaAuditReport(
        valid=outside_changed == 0,
        planned_k=plan.k,
        # Coordinates are keyed by parameter name, so supports of separate
        # tensors are disjoint and their exact union is their summed K.
        support_union_size=plan.k,
        total_native_coordinates=total_native_coordinates,
        support_union_fraction=plan.k / total_native_coordinates,
        changed_union_size=changed_union,
        selected_changed_count=selected_changed,
        outside_support_changed_count=outside_changed,
        selected_delta_l2=math.sqrt(aggregate_selected_squared),
        outside_support_delta_l2=math.sqrt(aggregate_outside_squared),
        total_delta_l2=math.sqrt(aggregate_total_squared),
        selected_delta_max_abs=aggregate_selected_max,
        outside_support_delta_max_abs=aggregate_outside_max,
        plan_hash=plan.plan_hash,
        support_hash=plan.support_hash,
        leakage_examples=tuple(leakage_examples),
        tensors=tuple(tensor_reports),
    )
