"""Small, explicit PyTorch implementation of the GFlowRL objective.

The implementation is deliberately independent of any training framework.  It accepts
sequence-level log probabilities captured by a rollout worker and returns a differentiable
loss for the current policy.  Reference, rollout-policy, reward, length, mask, and the
in-batch partition estimate are all treated as frozen quantities exactly where the paper
specifies stop-gradient behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class InvalidRolloutBatch(ValueError):
    """Raised when a rollout batch violates the objective contract."""


@dataclass(frozen=True)
class GFlowRLConfig:
    """Qualified GFlowRL settings.

    ``expected_group_size=4`` is the Rineya qualification setting.  The paper's production
    table uses 16 responses per prompt; qualification intentionally uses a smaller fixed
    group and records that difference in the protocol rather than silently changing it.
    """

    beta: float = 1.0
    epsilon_low: float = 0.20
    epsilon_high: float = 0.28
    importance_clip: float = 2.0
    expected_group_size: int = 4
    length_normalize: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.beta <= 0 or not torch.isfinite(torch.tensor(self.beta)):
            raise ValueError("beta must be finite and positive")
        if self.epsilon_low < 0 or self.epsilon_high < 0:
            raise ValueError("asymmetric clip bounds must be nonnegative")
        if self.importance_clip <= 0 or not torch.isfinite(torch.tensor(self.importance_clip)):
            raise ValueError("importance_clip must be finite and positive")
        if self.expected_group_size < 1:
            raise ValueError("expected_group_size must be positive")


@dataclass
class RolloutBatch:
    """One or more prompt groups of sampled responses.

    Every log probability is a sequence-level sum over the generated response.  The response
    length is supplied separately because GFlowRL divides both policy/reference flow terms by
    response length while retaining an unnormalized Eq. 4 partition estimate.
    """

    current_logprob: torch.Tensor
    old_logprob: torch.Tensor
    reference_logprob: torch.Tensor
    rewards: torch.Tensor
    response_lengths: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        tensors = (
            self.current_logprob,
            self.old_logprob,
            self.reference_logprob,
            self.rewards,
            self.response_lengths,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise InvalidRolloutBatch("all rollout fields must be torch tensors")
        shape = self.current_logprob.shape
        if len(shape) != 2:
            raise InvalidRolloutBatch("rollout tensors must have shape [num_prompts, group_size]")
        if any(value.shape != shape for value in tensors[1:]):
            raise InvalidRolloutBatch("rollout tensor shapes must match")
        if self.valid_mask is None:
            self.valid_mask = torch.ones(shape, dtype=torch.bool, device=self.current_logprob.device)
        elif self.valid_mask.shape != shape:
            raise InvalidRolloutBatch("valid_mask shape must match rollout tensors")
        elif self.valid_mask.dtype != torch.bool:
            raise InvalidRolloutBatch("valid_mask must be boolean")
        if shape[0] == 0 or shape[1] == 0:
            raise InvalidRolloutBatch("rollout batch must be nonempty")
        for name, value in (
            ("current_logprob", self.current_logprob),
            ("old_logprob", self.old_logprob),
            ("reference_logprob", self.reference_logprob),
            ("rewards", self.rewards),
            ("response_lengths", self.response_lengths),
        ):
            if not torch.isfinite(value.detach()).all():
                raise FloatingPointError(f"nonfinite {name}")
        if (self.response_lengths.detach() <= 0).any():
            raise InvalidRolloutBatch("response_lengths must be positive")


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    mode: str
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "loss": float(self.loss.detach()), **self.diagnostics}


def _validate_configured_group(batch: RolloutBatch, config: GFlowRLConfig) -> None:
    if batch.current_logprob.shape[1] != config.expected_group_size:
        raise InvalidRolloutBatch(
            f"expected fixed group size {config.expected_group_size}, "
            f"got {batch.current_logprob.shape[1]}"
        )


def _masked_group_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = mask.sum(dim=1)
    safe_counts = counts.clamp_min(1).to(values.dtype)
    means = (values * mask.to(values.dtype)).sum(dim=1) / safe_counts
    return means, counts


def _safe_importance_weight(log_ratio: torch.Tensor, maximum: float) -> torch.Tensor:
    """Compute min(exp(log_ratio), maximum) without overflowing exp."""

    cap_log = torch.log(torch.as_tensor(maximum, dtype=log_ratio.dtype, device=log_ratio.device))
    return torch.exp(torch.clamp(log_ratio, max=cap_log))


def compute_gflowrl_loss(batch: RolloutBatch, config: GFlowRLConfig) -> ObjectiveResult:
    """Compute Eq. 8 from the paper, with explicit qualification diagnostics."""

    _validate_configured_group(batch, config)
    current = batch.current_logprob
    old = batch.old_logprob.detach()
    reference = batch.reference_logprob.detach()
    rewards = batch.rewards.detach()
    lengths = batch.response_lengths.detach().to(current.dtype)
    mask = batch.valid_mask

    # Eq. 4: mean over the valid rollout group, with no response-length normalization.
    partition_targets = config.beta * rewards + reference - old
    log_z, valid_counts = _masked_group_mean(partition_targets, mask)
    log_z = log_z.detach()

    denom = lengths if config.length_normalize else torch.ones_like(lengths)
    flow_gap = log_z[:, None] + (old - reference) / denom - config.beta * rewards
    clipped_flow_gap = torch.clamp(
        flow_gap, min=-config.epsilon_low, max=config.epsilon_high
    ).detach()
    policy_delta = (current - old) / denom
    residual = clipped_flow_gap + policy_delta

    # Eq. 8: sequence-level importance correction.  Clamping the log ratio before exp is
    # algebraically identical to min(exp(log_ratio), clip) and prevents overflow at 1e3-scale
    # synthetic ratios.
    log_ratio = current - old
    importance_weight = _safe_importance_weight(log_ratio, config.importance_clip)
    per_rollout_loss = importance_weight * residual.square()
    mask_float = mask.to(current.dtype)
    group_loss = (per_rollout_loss * mask_float).sum(dim=1) / valid_counts.clamp_min(1).to(current.dtype)
    active = valid_counts > 0
    if active.any():
        loss = group_loss[active].mean()
    else:
        # Preserve the graph for callers that unconditionally call backward(), while making
        # an all-invalid group a true zero-cost skipped update.
        loss = current.sum() * 0.0

    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("GFlowRL loss is nonfinite")
    diagnostics = {
        "partition_log_z": log_z,
        "partition_targets": partition_targets,
        "valid_counts": valid_counts,
        "flow_gap": flow_gap,
        "clipped_flow_gap": clipped_flow_gap,
        "policy_delta": policy_delta.detach(),
        "importance_weight": importance_weight.detach(),
        "active_groups": active,
        "length_normalized": config.length_normalize,
    }
    return ObjectiveResult(loss=loss, mode="GFlowRL", diagnostics=diagnostics)


def compute_r0_loss(batch: RolloutBatch) -> ObjectiveResult:
    """R0 control: reward-maximizing, group-centered sequence policy gradient.

    This is kept as a separate function and is the dispatch target when GFlowRL is disabled.
    It does not use partition estimates, flow-gap clipping, or importance correction.
    """

    current = batch.current_logprob
    rewards = batch.rewards.detach()
    mask = batch.valid_mask
    centered, counts = _masked_group_mean(rewards, mask)
    advantages = rewards - centered[:, None]
    per_rollout = -(advantages.detach() * current)
    mask_float = mask.to(current.dtype)
    group_loss = (per_rollout * mask_float).sum(dim=1) / counts.clamp_min(1).to(current.dtype)
    active = counts > 0
    loss = group_loss[active].mean() if active.any() else current.sum() * 0.0
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("R0 loss is nonfinite")
    return ObjectiveResult(
        loss=loss,
        mode="R0",
        diagnostics={
            "advantages": advantages.detach(),
            "valid_counts": counts,
            "active_groups": active,
        },
    )


def compute_objective(batch: RolloutBatch, config: GFlowRLConfig) -> ObjectiveResult:
    """Dispatch R0 or GFlowRL without mutating the R0 implementation."""

    if not config.enabled:
        return compute_r0_loss(batch)
    return compute_gflowrl_loss(batch, config)
