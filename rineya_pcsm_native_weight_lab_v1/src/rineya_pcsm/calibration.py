"""First-order, held-out-gradient calibration objectives."""

from __future__ import annotations

import torch


def first_order_loss_change(
    meta_gradient: torch.Tensor,
    calibration_gradient: torch.Tensor,
    support: torch.Tensor,
    stabilizer: torch.Tensor,
    *,
    learning_rate: float,
) -> torch.Tensor:
    """Predict held-out loss change <g_meta, -lr M A g_cal>."""

    if not isinstance(calibration_gradient, torch.Tensor):
        raise TypeError("calibration_gradient must be an available tensor")
    if not isinstance(meta_gradient, torch.Tensor):
        raise TypeError("meta_gradient must be an available tensor")
    if not (
        meta_gradient.shape
        == calibration_gradient.shape
        == support.shape
        == stabilizer.shape
    ):
        raise ValueError("Calibration tensors must have identical shapes")
    if support.dtype is torch.bool:
        support_values = support.to(dtype=torch.float32)
    elif support.is_floating_point():
        if not bool(torch.logical_or(support == 0, support == 1).all()):
            raise ValueError("floating support must have an exact binary forward value")
        support_values = support.float()
    else:
        raise TypeError("support must be boolean or a floating exact-binary STE")
    if stabilizer.dtype is not torch.float32:
        raise TypeError("stabilizer must be float32")
    if not torch.isfinite(stabilizer).all() or (stabilizer < 0).any():
        raise ValueError("stabilizer must be finite and nonnegative")
    update = -float(learning_rate) * support_values * stabilizer * calibration_gradient
    return torch.sum(meta_gradient.float() * update.float())


def normalized_alignment(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(left.float()) * torch.linalg.vector_norm(
        right.float()
    )
    if denominator == 0:
        raise ValueError("Cannot normalize a zero gradient")
    result: torch.Tensor = torch.sum(left.float() * right.float()) / denominator
    return result
