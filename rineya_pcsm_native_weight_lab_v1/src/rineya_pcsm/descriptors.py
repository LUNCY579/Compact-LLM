"""Cheap, scale-normalized float32 descriptors for PCSM peer geometry.

The descriptor deliberately summarizes a matrix instead of embedding native
coordinates. Its first 16 values are always available; optional activation and
calibration-gradient summaries occupy values 16--21 in the qualified
24-dimensional default schema.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import torch
from torch import Tensor

MIN_DESCRIPTOR_DIMENSION = 16
MAX_DESCRIPTOR_DIMENSION = 32


def _safe_stat(value: Tensor) -> Tensor:
    return torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _mean_std(values: Tensor) -> tuple[Tensor, Tensor]:
    if values.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=values.device)
        return zero, zero
    values = values.float()
    return values.mean(), values.std(unbiased=False)


def _rms(values: Tensor) -> Tensor:
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return values.float().square().mean().sqrt()


def _nonzero_scale(value: Tensor) -> Tensor:
    """Use one only for an exactly degenerate scale, preserving invariance."""

    return torch.where(value > 0, value, torch.ones_like(value))


def _normalized_moments(
    values: Tensor | None, *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Return signed mean, std and mean absolute value, normalized by RMS."""

    if values is None:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero, zero
    clean = torch.nan_to_num(
        values.detach().to(device=device, dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).reshape(-1)
    if clean.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero, zero
    scale = _nonzero_scale(_rms(clean))
    return (
        clean.mean() / scale,
        clean.std(unbiased=False) / scale,
        clean.abs().mean() / scale,
    )


def _role_code(role: str | None, *, device: torch.device) -> Tensor:
    """Return a stable role identifier in [-1, 1], independent of Python hash."""

    if not role:
        return torch.zeros((), dtype=torch.float32, device=device)
    digest = hashlib.sha256(role.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    unit = integer / float((1 << 64) - 1)
    return torch.tensor(2.0 * unit - 1.0, dtype=torch.float32, device=device)


def tensor_descriptor(
    tensor: Tensor,
    *,
    dimension: int = 24,
    activations: Tensor | None = None,
    calibration_gradient: Tensor | None = None,
    normalized_depth: float | None = None,
    role: str | None = None,
) -> Tensor:
    """Return a 16--32 dimensional, finite, float32 tensor descriptor.

    Weight moments and row/column RMS summaries are normalized by the matrix
    RMS, so multiplying a matrix by a positive scalar does not change its
    geometry descriptor. ``normalized_depth`` is an architecture-derived
    value in ``[0, 1]``. Optional activation and gradient tensors are reduced
    to scale-normalized moments and never retained.
    """

    if not MIN_DESCRIPTOR_DIMENSION <= dimension <= MAX_DESCRIPTOR_DIMENSION:
        raise ValueError(
            f"descriptor dimension must be in "
            f"[{MIN_DESCRIPTOR_DIMENSION}, {MAX_DESCRIPTOR_DIMENSION}]"
        )
    if normalized_depth is not None and (
        not math.isfinite(normalized_depth) or not 0.0 <= normalized_depth <= 1.0
    ):
        raise ValueError("normalized_depth must be finite and in [0, 1]")

    with torch.no_grad():
        original = tensor.detach()
        finite_mask = torch.isfinite(original)
        values = torch.where(finite_mask, original, torch.zeros_like(original)).float()
        flat = values.reshape(-1)
        abs_flat = flat.abs()
        mean, std = _mean_std(flat)
        rms = _rms(flat)
        scale = _nonzero_scale(rms)

        if values.ndim >= 2:
            matrix = values.reshape(values.shape[0], -1)
        else:
            matrix = values.reshape(-1, 1)
        row_rms = matrix.square().mean(dim=1).sqrt()
        column_rms = matrix.square().mean(dim=0).sqrt()
        row_mean, row_std = _mean_std(row_rms)
        column_mean, column_std = _mean_std(column_rms)
        centered_rms = _rms(flat - mean)

        device = values.device

        def scalar(item: float) -> Tensor:
            return torch.as_tensor(item, dtype=torch.float32, device=device)

        count = max(flat.numel(), 1)
        has_scale = rms > 0
        normalized = flat / scale
        activation_moments = _normalized_moments(activations, device=device)
        gradient_moments = _normalized_moments(calibration_gradient, device=device)
        rows = int(matrix.shape[0])
        columns = int(matrix.shape[1])

        features = torch.stack(
            [
                mean / scale,
                std / scale,
                abs_flat.mean() / scale if flat.numel() else mean,
                abs_flat.max() / scale if flat.numel() else mean,
                row_mean / scale,
                row_std / scale,
                column_mean / scale,
                column_std / scale,
                (flat > 0).sum() / count,
                ((abs_flat >= rms) & has_scale).sum() / count,
                ((abs_flat >= 2.0 * rms) & has_scale).sum() / count,
                finite_mask.sum() / count,
                scalar(float(rows)).log1p(),
                scalar(float(columns)).log1p(),
                scalar(0.0 if normalized_depth is None else normalized_depth),
                _role_code(role, device=device),
                *activation_moments,
                *gradient_moments,
                row_rms.max() / scale if row_rms.numel() else mean,
                column_rms.max() / scale if column_rms.numel() else mean,
                (flat < 0).sum() / count,
                (flat == 0).sum() / count,
                row_rms.min() / scale if row_rms.numel() else mean,
                column_rms.min() / scale if column_rms.numel() else mean,
                centered_rms / scale,
                scalar(math.log((rows + 1.0) / (columns + 1.0))),
                normalized.pow(3).mean() if flat.numel() else mean,
                normalized.pow(4).mean() if flat.numel() else mean,
            ]
        )
        return _safe_stat(features[:dimension]).contiguous()


def describe_tensors(
    tensors: Mapping[str, Tensor],
    *,
    dimension: int = 24,
    activations: Mapping[str, Tensor] | None = None,
    calibration_gradients: Mapping[str, Tensor] | None = None,
    normalized_depths: Mapping[str, float] | None = None,
    roles: Mapping[str, str] | None = None,
) -> dict[str, Tensor]:
    """Describe a name-to-tensor mapping in canonical name order."""

    return {
        name: tensor_descriptor(
            tensors[name],
            dimension=dimension,
            activations=None if activations is None else activations.get(name),
            calibration_gradient=(
                None
                if calibration_gradients is None
                else calibration_gradients.get(name)
            ),
            normalized_depth=(
                None if normalized_depths is None else normalized_depths.get(name)
            ),
            role=None if roles is None else roles.get(name),
        )
        for name in sorted(tensors)
    }


__all__ = [
    "MAX_DESCRIPTOR_DIMENSION",
    "MIN_DESCRIPTOR_DIMENSION",
    "describe_tensors",
    "tensor_descriptor",
]
