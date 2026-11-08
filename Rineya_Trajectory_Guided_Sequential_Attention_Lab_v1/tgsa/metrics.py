"""Reproducible diagnostics and paired statistical summaries for TGSA studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .telemetry import TGSATelemetry


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    mean_delta: float
    confidence_low: float
    confidence_high: float
    probability_positive: float
    samples: int


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def paired_bootstrap_mean_difference(
    candidate: Sequence[float] | Tensor,
    baseline: Sequence[float] | Tensor,
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired bootstrap CI for candidate-minus-baseline mean difference."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    candidate_tensor = torch.as_tensor(candidate, dtype=torch.float64).flatten()
    baseline_tensor = torch.as_tensor(baseline, dtype=torch.float64).flatten()
    if candidate_tensor.numel() == 0:
        raise ValueError("inputs must be non-empty")
    if candidate_tensor.shape != baseline_tensor.shape:
        raise ValueError("candidate and baseline must have the same shape")
    if not torch.isfinite(candidate_tensor).all() or not torch.isfinite(baseline_tensor).all():
        raise ValueError("inputs must be finite")

    delta = candidate_tensor - baseline_tensor
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=delta.numel(),
        size=(samples, delta.numel()),
        generator=generator,
    )
    sampled_means = delta[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = torch.quantile(
        sampled_means,
        torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64),
    )
    return PairedBootstrapResult(
        mean_delta=float(delta.mean().item()),
        confidence_low=float(low.item()),
        confidence_high=float(high.item()),
        probability_positive=float((sampled_means > 0).double().mean().item()),
        samples=samples,
    )


def telemetry_summary(telemetry: TGSATelemetry) -> dict[str, float]:
    """Reduce telemetry to run-level diagnostics without claiming quality."""

    velocity = telemetry.velocity_rms.float()
    control_signal = telemetry.control_signal_rms.float()
    cosine = telemetry.velocity_cosine.float()
    update = telemetry.carrier_update_rms.float()
    importance = telemetry.importance_mean.float()
    state_gate = telemetry.state_gate_mean.float()
    control_gate = telemetry.control_gate_mean.float()
    predictor_gate = telemetry.predictor_gate_mean.float()
    correction_gate = telemetry.correction_gate_mean.float()

    if velocity.shape[1] > 2:
        noninitial = velocity[:, 1:]
        convergence_fraction = (
            noninitial[:, 1:] <= noninitial[:, :-1]
        ).float().mean()
    else:
        convergence_fraction = torch.tensor(float("nan"))

    if cosine.shape[1] > 2:
        oscillation_fraction = (cosine[:, 2:] < 0).float().mean()
    else:
        oscillation_fraction = torch.tensor(float("nan"))

    def scalar(value: Tensor) -> float:
        return float(value.detach().cpu().item())

    return {
        "velocity_rms_mean": scalar(velocity.mean()),
        "control_signal_rms_mean": scalar(control_signal.mean()),
        "carrier_update_rms_mean": scalar(update.mean()),
        "velocity_convergence_fraction": scalar(convergence_fraction),
        "velocity_oscillation_fraction": scalar(oscillation_fraction),
        "importance_mean": scalar(importance.mean()),
        "state_gate_mean": scalar(state_gate.mean()),
        "control_gate_mean": scalar(control_gate.mean()),
        "predictor_gate_mean": scalar(predictor_gate.mean()),
        "correction_gate_mean": scalar(correction_gate.mean()),
        "attention_entropy_mean": scalar(telemetry.attention_entropy.float().mean()),
    }
