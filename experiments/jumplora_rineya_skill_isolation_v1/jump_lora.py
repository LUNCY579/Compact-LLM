"""Dense, frozen-mask, and JumpLoRA linear adapters.

The implementation follows JumpLoRA v3, arXiv:2604.16171, equations (1)--(5):
the forward gate is a symmetric hard threshold over the *unscaled* materialized
LoRA delta ``B @ A``.  The LoRA scale ``alpha/r`` is applied only after the
dense/sparse interpolation, matching Algorithm 1 line 14.  Threshold/support
selection is deterministic and tie-aware.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class JumpSchedule:
    """Paper-compatible interpolation schedule in optimizer-step units."""

    s_start: int
    s_final: int
    epsilon: float = 1.0e-3

    def __post_init__(self) -> None:
        _require(isinstance(self.s_start, int) and self.s_start >= 0,
                 "s_start must be a non-negative integer")
        _require(isinstance(self.s_final, int) and self.s_final > self.s_start,
                 "s_final must be greater than s_start")
        _require(math.isfinite(float(self.epsilon)) and self.epsilon > 0.0,
                 "epsilon must be finite and positive")

    def gamma(self, step: int) -> float:
        """Return clip((step-s_start)/(s_final-s_start), 0, 1)."""
        _require(isinstance(step, int) and step >= 0, "step must be non-negative")
        return float(max(0.0, min(1.0,
            (step - self.s_start) / float(self.s_final - self.s_start))))


class _SymmetricJumpReLU(Function):
    """Hard symmetric JumpReLU with a band-limited STE for the threshold."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tau: torch.Tensor, epsilon: float) -> torch.Tensor:
        _require(tau.ndim == 0, "JumpLoRA threshold must be scalar")
        _require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(tau).all()),
                 "JumpLoRA received nonfinite values")
        ctx.save_for_backward(x, tau)
        ctx.epsilon = float(epsilon)
        return x * (x.abs() > tau).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, tau = ctx.saved_tensors
        epsilon = ctx.epsilon
        active = (x.abs() > tau).to(dtype=x.dtype)
        # Eq. (2) applied to J_tau(x)-J_tau(-x). The signs differ for the
        # negative term because d[-J_tau(-x)]/d tau has the opposite sign.
        positive_band = (((x - tau) / epsilon) >= -0.5) & (((x - tau) / epsilon) < 0.5)
        negative_band = (((-x - tau) / epsilon) >= -0.5) & (((-x - tau) / epsilon) < 0.5)
        d_tau = (-tau / epsilon) * positive_band.to(dtype=x.dtype)
        d_tau = d_tau + (tau / epsilon) * negative_band.to(dtype=x.dtype)
        grad_x = grad_output * active
        grad_tau = (grad_output * d_tau).sum().reshape_as(tau)
        return grad_x, grad_tau, None


def symmetric_jump_relu(x: torch.Tensor, tau: torch.Tensor | float,
                        epsilon: float = 1.0e-3) -> torch.Tensor:
    """Return ``x * H(abs(x)-tau)`` with the paper's threshold STE."""
    _require(torch.is_floating_point(x), "JumpLoRA input must be floating point")
    _require(bool(torch.isfinite(x).all()), "JumpLoRA input must be finite")
    if not isinstance(tau, torch.Tensor):
        tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    if tau.device != x.device or tau.dtype != x.dtype:
        tau = tau.to(device=x.device, dtype=x.dtype)
    _require(tau.ndim == 0 and bool(torch.isfinite(tau)),
             "JumpLoRA threshold must be one finite scalar")
    _require(math.isfinite(float(epsilon)) and epsilon > 0.0,
             "epsilon must be finite and positive")
    return _SymmetricJumpReLU.apply(x, tau, float(epsilon))


def deterministic_topk_mask(values: torch.Tensor, active_count: int) -> torch.Tensor:
    """Select exactly ``active_count`` largest magnitudes with stable ties."""
    _require(values.ndim >= 1, "support selection requires at least one dimension")
    _require(torch.is_floating_point(values), "support selection requires floating values")
    _require(bool(torch.isfinite(values).all()), "support selection received nonfinite values")
    total = values.numel()
    _require(isinstance(active_count, int) and 0 <= active_count <= total,
             "active_count is outside the tensor size")
    mask = torch.zeros(total, dtype=torch.bool, device=values.device)
    if active_count:
        order = torch.argsort(values.detach().abs().reshape(-1), descending=True, stable=True)
        mask[order[:active_count]] = True
    return mask.reshape(values.shape)


def threshold_for_mask(values: torch.Tensor, active_count: int) -> torch.Tensor:
    """Return a scalar threshold whose strict gate selects exactly ``active_count``.

    A scalar threshold cannot express a deterministic tie break. We therefore
    reject a tied boundary instead of silently reporting a mismatched density.
    """
    _require(values.numel() > 0, "threshold selection requires non-empty values")
    _require(torch.is_floating_point(values) and bool(torch.isfinite(values).all()),
             "threshold selection requires finite floating values")
    _require(isinstance(active_count, int) and 0 <= active_count <= values.numel(),
             "active_count is outside tensor size")
    magnitudes = values.detach().abs().reshape(-1).sort(descending=True, stable=True).values
    if active_count == 0:
        return torch.nextafter(magnitudes[0], torch.tensor(float("inf"), dtype=values.dtype,
                                                           device=values.device)).reshape(())
    if active_count == magnitudes.numel():
        return torch.nextafter(magnitudes[-1], torch.tensor(float("-inf"), dtype=values.dtype,
                                                            device=values.device)).reshape(())
    high = magnitudes[active_count - 1]
    low = magnitudes[active_count]
    _require(bool(high > low), "cannot represent a tied support boundary with one threshold")
    tau = ((high + low) / 2.0).reshape(())
    _require(int(values.detach().abs().gt(tau).sum()) == active_count,
             "strict threshold failed to realize requested support")
    return tau


class LoRALinear(nn.Module):
    """A bias-preserving LoRA wrapper around one existing ``nn.Linear``."""

    arm_name = "J0"

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 32.0):
        super().__init__()
        _require(type(base) is nn.Linear, "LoRA target must be an nn.Linear")
        _require(isinstance(rank, int) and rank > 0, "rank must be a positive integer")
        _require(math.isfinite(float(alpha)) and alpha > 0.0, "alpha must be positive")
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features,
                                               device=base.weight.device,
                                               dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank,
                                               device=base.weight.device,
                                               dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._merged = False
        self._unmerged_weight: torch.Tensor | None = None
        self.train(base.training)

    def raw_delta(self) -> torch.Tensor:
        """Return the paper's unscaled ``Delta W = B @ A``."""
        return self.lora_B @ self.lora_A

    def dense_delta(self) -> torch.Tensor:
        """Return the ordinary LoRA update added to the base, including alpha/r.

        This preserves the package's pre-existing public helper semantics.
        JumpLoRA thresholding itself always operates on :meth:`raw_delta`.
        """
        return self.raw_delta() * self.scale

    def support_mask(self) -> torch.Tensor:
        return self.raw_delta().detach().ne(0)

    def effective_delta(self, step: int | None = None) -> torch.Tensor:
        """Return the update actually added to the frozen base weight."""
        del step
        return self.dense_delta()

    def forward(self, x: torch.Tensor, step: int | None = None) -> torch.Tensor:
        if self._merged:
            return self.base(x)
        return self.base(x) + F.linear(x, self.effective_delta(step))

    def merge(self, step: int | None = None) -> None:
        _require(not self._merged, "adapter is already merged")
        delta = self.effective_delta(step).detach()
        _require(bool(torch.isfinite(delta).all()), "cannot merge a nonfinite adapter")
        self._unmerged_weight = self.base.weight.detach().clone()
        with torch.no_grad():
            self.base.weight.add_(delta.to(dtype=self.base.weight.dtype,
                                            device=self.base.weight.device))
        self._merged = True

    def unmerge(self) -> None:
        _require(self._merged and self._unmerged_weight is not None,
                 "adapter is not merged")
        with torch.no_grad():
            self.base.weight.copy_(self._unmerged_weight)
        self._unmerged_weight = None
        self._merged = False

    @contextmanager
    def merged(self, step: int | None = None) -> Iterator[None]:
        self.merge(step)
        try:
            yield
        finally:
            self.unmerge()


class MagnitudePrunedLoRA(LoRALinear):
    """J1 control: dense training followed by deterministic magnitude pruning."""

    arm_name = "J1"

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 32.0):
        super().__init__(base, rank, alpha)
        self.register_buffer("frozen_mask", torch.ones_like(self.base.weight, dtype=torch.bool))
        self.mask_frozen = False

    def set_frozen_mask(self, mask: torch.Tensor) -> None:
        _require(not self.mask_frozen, "J1 support is already frozen")
        _require(mask.shape == self.base.weight.shape and mask.dtype == torch.bool,
                 "support mask shape or dtype mismatch")
        self.frozen_mask.copy_(mask)
        self.mask_frozen = True

    def effective_delta(self, step: int | None = None) -> torch.Tensor:
        del step
        return self.dense_delta() * self.frozen_mask.to(dtype=self.lora_A.dtype)

    def support_mask(self) -> torch.Tensor:
        return self.frozen_mask.detach().clone()


class EarlyFrozenMaskLoRA(MagnitudePrunedLoRA):
    """J2 control: choose a magnitude support at warmup, then never change it."""

    arm_name = "J2"


class JumpLoRALinear(LoRALinear):
    """J3: learnable symmetric JumpReLU with gradual dense-to-sparse annealing."""

    arm_name = "J3"

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 32.0,
                 schedule: JumpSchedule | None = None, threshold: float = 0.0):
        super().__init__(base, rank, alpha)
        self.schedule = schedule or JumpSchedule(2, 8)
        _require(math.isfinite(float(threshold)) and threshold >= 0.0,
                 "initial threshold must be non-negative")
        self.threshold = nn.Parameter(torch.tensor(float(threshold),
                                                   device=base.weight.device,
                                                   dtype=base.weight.dtype))
        self.current_step = 0

    @property
    def gamma(self) -> float:
        return self.schedule.gamma(self.current_step)

    def set_step(self, step: int) -> None:
        _require(isinstance(step, int) and step >= 0, "step must be non-negative")
        self.current_step = step

    def initialize_threshold(self, active_count: int) -> None:
        with torch.no_grad():
            self.threshold.copy_(threshold_for_mask(self.raw_delta(), active_count))

    def effective_delta(self, step: int | None = None) -> torch.Tensor:
        actual_step = self.current_step if step is None else step
        gamma = self.schedule.gamma(actual_step)
        dense_unscaled = self.raw_delta()
        if gamma == 0.0:
            return dense_unscaled * self.scale
        jumped_unscaled = symmetric_jump_relu(dense_unscaled, self.threshold,
                                               self.schedule.epsilon)
        interpolated_unscaled = ((1.0 - gamma) * dense_unscaled
                                 + gamma * jumped_unscaled)
        return interpolated_unscaled * self.scale

    def support_mask(self) -> torch.Tensor:
        return self.raw_delta().detach().abs().gt(self.threshold.detach())


def build_adapter(base: nn.Linear, arm: str, *, rank: int = 8, alpha: float = 32.0,
                  schedule: JumpSchedule | None = None) -> LoRALinear:
    """Construct one of the four registered arms without loading a model."""
    arm = str(arm).upper()
    if arm == "J0":
        return LoRALinear(base, rank, alpha)
    if arm == "J1":
        return MagnitudePrunedLoRA(base, rank, alpha)
    if arm == "J2":
        return EarlyFrozenMaskLoRA(base, rank, alpha)
    if arm == "J3":
        return JumpLoRALinear(base, rank, alpha, schedule=schedule)
    raise ValueError(f"unknown registered arm: {arm}")
