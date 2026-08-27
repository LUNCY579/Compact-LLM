"""Structural, nonnegative parameter property maps.

These maps encode element/row/column geometry only.  They make no claim that
rows or columns are neuron identities across independently trained models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _finite_nonnegative(values: Tensor) -> Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return values.clamp_min(0.0).contiguous()


def _mean_normalize_nonnegative(values: Tensor) -> Tensor:
    values = _finite_nonnegative(values)
    if values.numel() == 0:
        return values
    mean = values.mean()
    denominator = torch.where(mean > 0, mean, torch.ones_like(mean))
    return (values / denominator).contiguous()


def _projection_indices(source: int, target: int, device: torch.device) -> Tensor:
    if source <= 0 or target <= 0:
        raise ValueError("source and target dimensions must be positive")
    if target == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    # Explicit nearest-coordinate projection is deterministic and transparent.
    coordinates = torch.linspace(0, source - 1, target, device=device)
    return coordinates.round().to(torch.long)


@dataclass(frozen=True, slots=True)
class PropertyMaps:
    """Element, row, column, and separable joint nonnegative maps."""

    element: Tensor
    row: Tensor
    column: Tensor
    joint: Tensor

    def __post_init__(self) -> None:
        if self.element.ndim != 2 or self.joint.ndim != 2:
            raise ValueError("element and joint property maps must be matrices")
        if self.row.ndim != 1 or self.column.ndim != 1:
            raise ValueError("row and column property maps must be vectors")
        shape = tuple(self.element.shape)
        if tuple(self.joint.shape) != shape:
            raise ValueError("element and joint map shapes differ")
        if self.row.numel() != shape[0] or self.column.numel() != shape[1]:
            raise ValueError("row/column map lengths do not match matrix shape")
        for values in (self.element, self.row, self.column, self.joint):
            if values.dtype != torch.float32:
                raise TypeError("property maps must use float32")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("property maps must be finite")
            if bool((values < 0).any()):
                raise ValueError("property maps must be nonnegative")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.element.shape[0], self.element.shape[1])

    def project(self, shape: tuple[int, int]) -> PropertyMaps:
        """Project maps to ``shape`` by deterministic nearest coordinates."""

        rows, columns = shape
        row_index = _projection_indices(self.shape[0], rows, self.element.device)
        column_index = _projection_indices(self.shape[1], columns, self.element.device)
        element = self.element.index_select(0, row_index).index_select(1, column_index)
        joint = self.joint.index_select(0, row_index).index_select(1, column_index)
        row = self.row.index_select(0, row_index)
        column = self.column.index_select(0, column_index)
        return PropertyMaps(
            element=_finite_nonnegative(element),
            row=_finite_nonnegative(row),
            column=_finite_nonnegative(column),
            joint=_finite_nonnegative(joint),
        )

    def channels(self) -> Tensor:
        """Return four broadcast matrix channels in documented order."""

        rows, columns = self.shape
        return torch.stack(
            (
                self.element,
                self.row[:, None].expand(rows, columns),
                self.column[None, :].expand(rows, columns),
                self.joint,
            ),
            dim=-1,
        )


def build_property_maps(weight: Tensor) -> PropertyMaps:
    """Build centered-magnitude and mean-normalized RMS-energy maps.

    Coordinates are structural statistics only. They do not assert that row or
    column identities correspond between transformer layers.
    """

    if weight.ndim != 2:
        raise ValueError("property maps require a rank-two weight tensor")
    values = torch.nan_to_num(weight.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    centered = values - values.mean()
    centered_rms = centered.square().mean().sqrt()
    centered_scale = torch.where(
        centered_rms > 0, centered_rms, torch.ones_like(centered_rms)
    )
    element = _finite_nonnegative((centered / centered_scale).abs())

    epsilon = torch.finfo(torch.float32).eps
    row_energy = (values.square().mean(dim=1) + epsilon).sqrt()
    column_energy = (values.square().mean(dim=0) + epsilon).sqrt()
    row = _mean_normalize_nonnegative(row_energy)
    column = _mean_normalize_nonnegative(column_energy)
    joint = _finite_nonnegative(row[:, None] * column[None, :])
    return PropertyMaps(element=element, row=row, column=column, joint=joint)


def aggregate_property_maps(
    maps: tuple[PropertyMaps, ...],
    alphas: Tensor,
    shape: tuple[int, int],
    *,
    differentiable_attention: bool = False,
) -> PropertyMaps:
    """Project and alpha-average peer maps into a target shape.

    Static plan generation detaches attention by default, allowing the cached
    aggregate to be reused across calibration steps. Joint attention training
    must opt in and recompute this aggregate for every backward pass.
    """

    if not maps:
        raise ValueError("at least one peer property map is required")
    if alphas.ndim != 1 or alphas.numel() != len(maps):
        raise ValueError("one attention alpha is required for every peer map")
    alphas = alphas.float()
    if not differentiable_attention:
        alphas = alphas.detach()
    if not bool(torch.isfinite(alphas).all()) or bool((alphas < 0).any()):
        raise ValueError("attention alphas must be finite and nonnegative")
    if not torch.isclose(alphas.sum(), alphas.new_tensor(1.0), atol=1e-5):
        raise ValueError("attention alphas must sum to one")
    projected = tuple(item.project(shape) for item in maps)

    def combine(attribute: str) -> Tensor:
        tensors = torch.stack([getattr(item, attribute) for item in projected])
        weights = alphas.to(tensors.device)
        view = (weights.numel(),) + (1,) * (tensors.ndim - 1)
        return _finite_nonnegative((tensors * weights.view(view)).sum(dim=0))

    return PropertyMaps(
        element=combine("element"),
        row=combine("row"),
        column=combine("column"),
        joint=combine("joint"),
    )


__all__ = ["PropertyMaps", "aggregate_property_maps", "build_property_maps"]
