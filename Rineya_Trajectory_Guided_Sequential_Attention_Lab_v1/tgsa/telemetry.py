"""Typed telemetry emitted by trajectory-guided sequential attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class TGSATelemetry:
    """Per-batch, per-head summaries and optional full trajectory tensors.

    Summary tensors have shape ``[batch, heads]``. Full tensors are intentionally
    optional because attention maps scale quadratically with sequence length.
    """

    velocity_rms: Tensor
    control_signal_rms: Tensor
    acceleration_rms: Tensor
    velocity_cosine: Tensor
    carrier_update_rms: Tensor
    state_gate_mean: Tensor
    control_gate_mean: Tensor
    predictor_gate_mean: Tensor
    correction_gate_mean: Tensor
    importance_mean: Tensor
    importance_min: Tensor
    importance_max: Tensor
    attention_entropy: Tensor
    carrier_states: Tensor | None = None
    velocities: Tensor | None = None
    control_signals: Tensor | None = None
    importance_scores: Tensor | None = None
    attention_maps: Tensor | None = None

    def detached(self) -> "TGSATelemetry":
        """Return a recursively detached copy suitable for logging."""

        def maybe_detach(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.detach()

        return TGSATelemetry(
            velocity_rms=self.velocity_rms.detach(),
            control_signal_rms=self.control_signal_rms.detach(),
            acceleration_rms=self.acceleration_rms.detach(),
            velocity_cosine=self.velocity_cosine.detach(),
            carrier_update_rms=self.carrier_update_rms.detach(),
            state_gate_mean=self.state_gate_mean.detach(),
            control_gate_mean=self.control_gate_mean.detach(),
            predictor_gate_mean=self.predictor_gate_mean.detach(),
            correction_gate_mean=self.correction_gate_mean.detach(),
            importance_mean=self.importance_mean.detach(),
            importance_min=self.importance_min.detach(),
            importance_max=self.importance_max.detach(),
            attention_entropy=self.attention_entropy.detach(),
            carrier_states=maybe_detach(self.carrier_states),
            velocities=maybe_detach(self.velocities),
            control_signals=maybe_detach(self.control_signals),
            importance_scores=maybe_detach(self.importance_scores),
            attention_maps=maybe_detach(self.attention_maps),
        )

    def to_cpu_dict(self) -> dict[str, object]:
        """Convert summary telemetry to JSON-serializable Python values."""

        result: dict[str, object] = {}
        for name in (
            "velocity_rms",
            "control_signal_rms",
            "acceleration_rms",
            "velocity_cosine",
            "carrier_update_rms",
            "state_gate_mean",
            "control_gate_mean",
            "predictor_gate_mean",
            "correction_gate_mean",
            "importance_mean",
            "importance_min",
            "importance_max",
            "attention_entropy",
        ):
            tensor = getattr(self, name)
            result[name] = tensor.detach().cpu().tolist()
        return result


def rms_per_batch(value: Tensor) -> Tensor:
    """Root-mean-square over all non-batch dimensions."""

    dims = tuple(range(1, value.ndim))
    return value.float().square().mean(dim=dims).sqrt()


def cosine_per_batch(left: Tensor, right: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Cosine similarity over all non-batch dimensions."""

    left_flat = left.float().flatten(start_dim=1)
    right_flat = right.float().flatten(start_dim=1)
    numerator = (left_flat * right_flat).sum(dim=-1)
    denominator = left_flat.norm(dim=-1) * right_flat.norm(dim=-1)
    return numerator / denominator.clamp_min(eps)
