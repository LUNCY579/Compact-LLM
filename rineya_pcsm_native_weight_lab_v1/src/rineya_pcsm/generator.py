"""Peer-conditioned score and bounded stabilizer generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .property_maps import PropertyMaps, build_property_maps

LOCAL_FEATURE_NAMES = (
    "task_gradient_magnitude",
    "peer_element_magnitude",
    "peer_row_energy",
    "peer_column_energy",
    "peer_joint_leverage",
    "target_element_magnitude",
    "peer_target_structural_difference",
    "gradient_peer_interaction",
)
GLOBAL_FEATURE_BLOCKS = ("target", "task_gradient", "peer")
GENERATOR_VERSION = "pcsm-generator-v1"


def _standardize(values: Tensor) -> Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _rms_normalize_magnitude(values: Tensor) -> Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).abs()
    rms = values.square().mean().sqrt()
    denominator = torch.where(rms > 0, rms, torch.ones_like(rms))
    return values / denominator


@dataclass(frozen=True, slots=True)
class GeneratorOutput:
    scores: Tensor
    stabilizer: Tensor
    feature_weights: Tensor
    local_features: Tensor
    global_features: Tensor

    def record(self) -> dict[str, object]:
        """Return the auditable, JSON-compatible support-head evidence."""

        return {
            "feature_names": list(LOCAL_FEATURE_NAMES),
            "feature_weights": self.feature_weights.detach().cpu().tolist(),
        }


class PCSMGenerator(nn.Module):
    """Generate support scores and post-Adam stabilizers.

    Target-weight, task-gradient, and peer information are explicit in both the
    feature schema and the returned audit tensors.  The stabilizer is bounded by
    construction and defaults to ``[0.25, 1.0]``.
    """

    def __init__(
        self,
        descriptor_dimension: int = 24,
        *,
        hidden_dimension: int = 64,
        lower: float = 0.25,
        upper: float = 1.0,
    ) -> None:
        super().__init__()
        if descriptor_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("generator dimensions must be positive")
        if not (0.0 < lower < upper):
            raise ValueError("stabilizer bounds must satisfy 0 < lower < upper")
        self.descriptor_dimension = descriptor_dimension
        self.hidden_dimension = hidden_dimension
        self.lower = float(lower)
        self.upper = float(upper)
        global_dimension = descriptor_dimension * len(GLOBAL_FEATURE_BLOCKS)
        # Eight softmax feature logits plus (bias, score, peer) scale controls.
        output_dimension = len(LOCAL_FEATURE_NAMES) + 3
        self.context_network = nn.Sequential(
            nn.Linear(global_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def configuration(self) -> dict[str, object]:
        """Return the complete JSON-compatible generator architecture config."""

        return {
            "version": GENERATOR_VERSION,
            "descriptor_dimension": self.descriptor_dimension,
            "hidden_dimension": self.hidden_dimension,
            "stabilizer_lower": self.lower,
            "stabilizer_upper": self.upper,
            "feature_names": list(LOCAL_FEATURE_NAMES),
            "global_feature_blocks": list(GLOBAL_FEATURE_BLOCKS),
        }

    def forward(
        self,
        target_weight: Tensor,
        task_gradient: Tensor,
        target_descriptor: Tensor,
        task_gradient_descriptor: Tensor,
        peer_descriptor: Tensor,
        peer_properties: PropertyMaps,
    ) -> GeneratorOutput:
        if target_weight.ndim != 2 or task_gradient.shape != target_weight.shape:
            raise ValueError(
                "target weight and task gradient must be matching matrices"
            )
        if peer_properties.shape != tuple(target_weight.shape):
            raise ValueError("peer property maps must be projected to the target shape")
        descriptors = (
            target_descriptor.float().reshape(-1),
            task_gradient_descriptor.float().reshape(-1),
            peer_descriptor.float().reshape(-1),
        )
        if any(item.numel() != self.descriptor_dimension for item in descriptors):
            raise ValueError("generator descriptor has the wrong dimension")
        global_features = torch.nan_to_num(
            torch.cat(descriptors), nan=0.0, posinf=0.0, neginf=0.0
        ).float()
        peer_channels = peer_properties.channels().to(target_weight.device)
        target_channels = (
            build_property_maps(target_weight).channels().to(target_weight.device)
        )
        peer_template = peer_channels.mean(dim=-1)
        target_template = target_channels.mean(dim=-1)
        gradient_magnitude = _rms_normalize_magnitude(task_gradient)
        raw_features = (
            gradient_magnitude,
            peer_channels[..., 0],
            peer_channels[..., 1],
            peer_channels[..., 2],
            peer_channels[..., 3],
            target_channels[..., 0],
            (peer_template - target_template).abs(),
            gradient_magnitude * peer_template,
        )
        local_features = torch.stack(
            tuple(_standardize(feature) for feature in raw_features), dim=-1
        ).float()
        coefficients = self.context_network(global_features)
        split = len(LOCAL_FEATURE_NAMES)
        feature_weights = torch.softmax(coefficients[:split].float(), dim=0)
        stabilizer_parameters = coefficients[split:]
        scores = (local_features * feature_weights.view(1, 1, -1)).sum(dim=-1)
        stabilizer_logits = (
            stabilizer_parameters[0]
            + stabilizer_parameters[1] * _standardize(scores)
            + stabilizer_parameters[2] * _standardize(peer_template)
        )
        stabilizer = self.lower + (self.upper - self.lower) * torch.sigmoid(
            stabilizer_logits
        )
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("generator produced non-finite support scores")
        if not bool(torch.isfinite(stabilizer).all()):
            raise FloatingPointError("generator produced a non-finite stabilizer")
        return GeneratorOutput(
            scores=scores.float(),
            stabilizer=stabilizer.float(),
            feature_weights=feature_weights.float(),
            local_features=local_features,
            global_features=global_features,
        )


__all__ = [
    "GENERATOR_VERSION",
    "GLOBAL_FEATURE_BLOCKS",
    "LOCAL_FEATURE_NAMES",
    "GeneratorOutput",
    "PCSMGenerator",
]
