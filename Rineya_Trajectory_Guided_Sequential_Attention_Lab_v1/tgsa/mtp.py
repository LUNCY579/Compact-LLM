"""Reference multi-token prediction (MTP) extension for the TGSA Phase-1 model.

The module keeps three questions separate:

* ``dormant``: the parameter-matched MTP module exists but contributes no loss;
* ``backbone``: future-token supervision is active from the backbone state only;
* ``trajectory``: the same MTP module additionally receives TGSA's existing
  differentiable per-head control signals.

No new notion of trajectory is introduced.  ``trajectory`` mode consumes the
``control_signals`` already emitted by the auditable TGSA reference kernel.  For
``learned_delta`` these are finite differences; for ``history_control`` they are
information-matched two-state history means.  The resulting comparison therefore
asks whether TGSA's registered control coordinate improves future-token
prediction, rather than granting one arm an unmatched side channel.

This is a training/reference implementation.  It does not claim speculative
throughput: real speed evidence requires a cache-aware target verifier and timed
end-to-end generation on a Rineya checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .module import RMSNorm, TrajectoryGuidedSequentialAttention
from .phase1_model import (
    ModelArm,
    TinyLMConfig,
    TinyTGSAForCausalLM,
    build_model,
    derive_initialization_seed,
)
from .telemetry import TGSATelemetry

MTPMode = Literal["dormant", "backbone", "trajectory"]


@dataclass(frozen=True, slots=True)
class MTPConfig:
    """Configuration for the parameter-matched sequential MTP head."""

    max_horizon: int = 3
    transition_rank: int = 64
    loss_weight: float = 0.20
    mode: MTPMode = "backbone"
    horizon_weights: tuple[float, ...] = ()
    control_block_index: int = -1
    pad_token_id: int = 0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.max_horizon < 2:
            raise ValueError("max_horizon must be at least 2")
        if self.transition_rank <= 0:
            raise ValueError("transition_rank must be positive")
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")
        if self.mode not in {"dormant", "backbone", "trajectory"}:
            raise ValueError(f"unsupported MTP mode: {self.mode}")
        expected = self.max_horizon - 1
        if self.horizon_weights and len(self.horizon_weights) != expected:
            raise ValueError(
                "horizon_weights must contain exactly max_horizon - 1 entries"
            )
        if any(weight <= 0.0 or not math.isfinite(weight) for weight in self.weights):
            raise ValueError("every horizon weight must be finite and positive")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    @property
    def weights(self) -> tuple[float, ...]:
        if self.horizon_weights:
            return self.horizon_weights
        # Nearer futures are more reliable, but every registered horizon remains
        # supervised.  The normalization in ``multi_token_prediction_loss``
        # prevents this choice from changing the total auxiliary-loss scale.
        return tuple(1.0 / float(horizon) for horizon in range(2, self.max_horizon + 1))

    @property
    def effective_loss_weight(self) -> float:
        return 0.0 if self.mode == "dormant" else self.loss_weight


@dataclass(slots=True)
class TGSAWithMTPOutput:
    """Outputs required by the paired TGSA x MTP experiment."""

    logits: Tensor
    mtp_logits: tuple[Tensor, ...]
    block_telemetry: tuple[TGSATelemetry | None, ...]
    final_hidden_states: Tensor
    trajectory_control: Tensor


@dataclass(slots=True)
class MTPObjective:
    total_loss: Tensor
    autoregressive_loss: Tensor
    mtp_loss: Tensor
    per_horizon_loss: dict[int, Tensor]
    per_horizon_tokens: dict[int, int]


class _MTPTransition(nn.Module):
    """One residual future-horizon transition.

    The output projection is initialized at a tiny common scale, so all factorial
    cells begin from the same near-identity future transition while gradients can
    reach the trajectory-control path on the first backward pass.
    """

    def __init__(self, embed_dim: int, rank: int, *, bias: bool) -> None:
        super().__init__()
        fused_dim = 3 * embed_dim
        self.norm = RMSNorm(fused_dim)
        self.down = nn.Linear(fused_dim, rank, bias=bias)
        self.up = nn.Linear(rank, embed_dim, bias=bias)
        self.gate = nn.Linear(fused_dim, embed_dim, bias=True)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight, gain=1.0e-3)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(
        self,
        state: Tensor,
        previous_future_embedding: Tensor,
        trajectory_control: Tensor,
    ) -> Tensor:
        fused = torch.cat((state, previous_future_embedding, trajectory_control), dim=-1)
        proposal = self.up(self.activation(self.down(self.norm(fused))))
        gate = torch.sigmoid(self.gate(fused))
        return state + gate * proposal


class SequentialMTPHead(nn.Module):
    """Predict horizons 2..K from one backbone state.

    Horizon ``h`` is conditioned on the preceding future token.  During training
    that token is teacher-forced from ``labels``; recursive drafting instead uses
    the previous horizon's argmax.  Token embedding and LM head are supplied by
    the backbone at call time, so the head shares them without registering a
    second copy in its state dict.
    """

    def __init__(self, embed_dim: int, config: MTPConfig) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.config = config
        self.control_norm = RMSNorm(embed_dim)
        self.control_projection = nn.Linear(embed_dim, embed_dim, bias=config.bias)
        self.transitions = nn.ModuleList(
            _MTPTransition(embed_dim, config.transition_rank, bias=config.bias)
            for _ in range(config.max_horizon - 1)
        )
        self.output_norms = nn.ModuleList(
            RMSNorm(embed_dim) for _ in range(config.max_horizon - 1)
        )
        self._reset_control_parameters()

    def _reset_control_parameters(self) -> None:
        # Small, non-zero initialization preserves a real differentiable route
        # into TGSA control on the first backward pass while avoiding a large
        # cold-start perturbation.  The same tensor is used by every MTP mode;
        # backbone/dormant cells simply pass zeros at this input boundary.
        nn.init.xavier_uniform_(self.control_projection.weight, gain=0.05)
        if self.control_projection.bias is not None:
            nn.init.zeros_(self.control_projection.bias)

    def _project_control(self, trajectory_control: Tensor) -> Tensor:
        if trajectory_control.shape[-1] != self.embed_dim:
            raise ValueError("trajectory_control has the wrong feature dimension")
        return self.control_projection(self.control_norm(trajectory_control))

    def teacher_forced(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        trajectory_control: Tensor,
        *,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
    ) -> tuple[Tensor, ...]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, embed_dim]")
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError("labels must match hidden-state batch and sequence dimensions")
        if trajectory_control.shape != hidden_states.shape:
            raise ValueError("trajectory_control must match hidden_states")

        projected_control = self._project_control(trajectory_control)
        state = hidden_states
        outputs: list[Tensor] = []
        for transition_index, (transition, output_norm) in enumerate(
            zip(self.transitions, self.output_norms)
        ):
            # For horizon 2 the preceding future token is x[t+1] (labels[t]);
            # for horizon 3 it is x[t+2] (labels[t+1]), and so on.
            previous_ids = shift_future_targets(
                labels,
                offset=transition_index,
                fill_value=self.config.pad_token_id,
            )
            previous_ids = previous_ids.masked_fill(previous_ids.lt(0), self.config.pad_token_id)
            previous_embedding = token_embedding(previous_ids)
            state = transition(state, previous_embedding, projected_control)
            outputs.append(lm_head(output_norm(state)))
        return tuple(outputs)

    def recursive(
        self,
        hidden_states: Tensor,
        first_future_token_ids: Tensor,
        trajectory_control: Tensor,
        *,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
    ) -> tuple[Tensor, ...]:
        """Recursively draft horizons 2..K with no teacher forcing."""

        if first_future_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("first_future_token_ids must have shape [batch, sequence]")
        if trajectory_control.shape != hidden_states.shape:
            raise ValueError("trajectory_control must match hidden_states")
        projected_control = self._project_control(trajectory_control)
        state = hidden_states
        previous_ids = first_future_token_ids
        outputs: list[Tensor] = []
        for transition, output_norm in zip(self.transitions, self.output_norms):
            previous_embedding = token_embedding(previous_ids)
            state = transition(state, previous_embedding, projected_control)
            logits = lm_head(output_norm(state))
            outputs.append(logits)
            previous_ids = logits.argmax(dim=-1)
        return tuple(outputs)


def shift_future_targets(labels: Tensor, *, offset: int, fill_value: int = -100) -> Tensor:
    """Align ``labels[t + offset]`` with source position ``t``.

    ``labels[t]`` is the ordinary next-token target.  Therefore offset 0 is the
    token required to condition the horizon-2 transition, while offset
    ``horizon - 1`` is the supervised target for horizon ``horizon``.
    """

    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    shifted = torch.full_like(labels, int(fill_value))
    if offset >= labels.shape[1]:
        return shifted
    valid = labels.shape[1] - offset
    shifted[:, :valid] = labels[:, offset:]
    return shifted


def flatten_tgsa_control(
    block_telemetry: Sequence[TGSATelemetry | None],
    *,
    block_index: int,
    embed_dim: int,
) -> Tensor:
    """Return TGSA's registered control tensor as ``[B, S, D]``.

    The reference kernel stores full control as ``[B, H, S, D_head]``.  Heads
    are transposed next to their native feature groups and flattened without any
    learned or arm-specific preprocessing.  Head 0 remains its registered zero
    control.
    """

    if not block_telemetry:
        raise ValueError("trajectory-conditioned MTP requires TGSA telemetry")
    index = block_index if block_index >= 0 else len(block_telemetry) + block_index
    if not 0 <= index < len(block_telemetry):
        raise ValueError("control_block_index is outside the decoder stack")
    telemetry = block_telemetry[index]
    if telemetry is None or telemetry.control_signals is None:
        raise ValueError(
            "trajectory-conditioned MTP requires full, non-detached TGSA control signals"
        )
    control = telemetry.control_signals
    if control.ndim != 4:
        raise ValueError("TGSA control_signals must have shape [batch, heads, sequence, head_dim]")
    batch, heads, sequence, head_dim = control.shape
    if heads * head_dim != embed_dim:
        raise ValueError("TGSA control geometry does not reconstruct the model embedding")
    return control.transpose(1, 2).reshape(batch, sequence, embed_dim)


class TinyTGSAWithMTP(nn.Module):
    """Parameter-matched MTP wrapper around the existing Tiny TGSA model."""

    def __init__(
        self,
        backbone: TinyTGSAForCausalLM,
        mtp_config: MTPConfig,
        *,
        mtp_initialization_seed: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mtp_config = mtp_config
        # Construct the same MTP state for every factorial cell regardless of
        # TGSA arm or MTP mode.  ``fork_rng`` prevents the auxiliary head from
        # perturbing any caller-owned RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(mtp_initialization_seed))
            self.mtp_head = SequentialMTPHead(backbone.config.embed_dim, mtp_config)

        if mtp_config.mode == "trajectory":
            found_tgsa = False
            for block in self.backbone.blocks:
                attention = block.attention
                if isinstance(attention, TrajectoryGuidedSequentialAttention):
                    found_tgsa = True
                    attention.config = attention.config.with_updates(detach_telemetry=False)
            if not found_tgsa:
                raise ValueError("trajectory-conditioned MTP requires a TGSA backbone arm")

    def _backbone_forward(
        self,
        input_ids: Tensor,
        *,
        key_padding_mask: Tensor | None,
        telemetry_level: Literal["none", "full"],
    ) -> tuple[Tensor, Tensor, tuple[TGSATelemetry | None, ...]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence = input_ids.shape
        if sequence > self.backbone.config.max_sequence_length:
            raise ValueError("input sequence exceeds the configured maximum")
        if key_padding_mask is not None and (
            key_padding_mask.dtype != torch.bool or key_padding_mask.shape != input_ids.shape
        ):
            raise ValueError("key_padding_mask must be bool and match input_ids")

        positions = torch.arange(sequence, device=input_ids.device)
        hidden_states = self.backbone.token_embedding(input_ids)
        hidden_states = hidden_states + self.backbone.position_embedding(positions)[None, :, :]
        hidden_states = self.backbone.embedding_dropout(hidden_states)
        telemetry: list[TGSATelemetry | None] = []
        for block in self.backbone.blocks:
            hidden_states, block_telemetry = block(
                hidden_states,
                key_padding_mask=key_padding_mask,
                telemetry_level=telemetry_level,
            )
            telemetry.append(block_telemetry)
        final_hidden = self.backbone.final_norm(hidden_states)
        logits = self.backbone.lm_head(final_hidden)
        return logits, final_hidden, tuple(telemetry)

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        recursive: bool = False,
    ) -> TGSAWithMTPOutput:
        telemetry_level: Literal["none", "full"] = (
            "full" if self.mtp_config.mode == "trajectory" else "none"
        )
        logits, final_hidden, telemetry = self._backbone_forward(
            input_ids,
            key_padding_mask=key_padding_mask,
            telemetry_level=telemetry_level,
        )
        if self.mtp_config.mode == "trajectory":
            trajectory_control = flatten_tgsa_control(
                telemetry,
                block_index=self.mtp_config.control_block_index,
                embed_dim=self.backbone.config.embed_dim,
            )
        else:
            trajectory_control = torch.zeros_like(final_hidden)

        if recursive:
            first_ids = logits.argmax(dim=-1)
            mtp_logits = self.mtp_head.recursive(
                final_hidden,
                first_ids,
                trajectory_control,
                token_embedding=self.backbone.token_embedding,
                lm_head=self.backbone.lm_head,
            )
        else:
            if labels is None:
                raise ValueError("teacher-forced MTP requires labels")
            mtp_logits = self.mtp_head.teacher_forced(
                final_hidden,
                labels,
                trajectory_control,
                token_embedding=self.backbone.token_embedding,
                lm_head=self.backbone.lm_head,
            )
        return TGSAWithMTPOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            block_telemetry=telemetry,
            final_hidden_states=final_hidden,
            trajectory_control=trajectory_control,
        )


def build_tgsa_mtp_model(
    lm_config: TinyLMConfig,
    mtp_config: MTPConfig,
    *,
    initialization_seed: int,
    canonical_state: Mapping[str, Tensor] | None = None,
) -> TinyTGSAWithMTP:
    """Build one factorial cell from paired TGSA and MTP initialization."""

    backbone = build_model(
        lm_config,
        initialization_seed=initialization_seed,
        canonical_state=canonical_state,
    )
    mtp_seed = derive_initialization_seed(initialization_seed, "tgsa_mtp_head_v1")
    return TinyTGSAWithMTP(
        backbone,
        mtp_config,
        mtp_initialization_seed=mtp_seed,
    )


def multi_token_prediction_loss(
    mtp_logits: Sequence[Tensor],
    labels: Tensor,
    *,
    horizon_weights: Sequence[float],
) -> tuple[Tensor, dict[int, Tensor], dict[int, int]]:
    """Compute normalized cross entropy for horizons 2..K."""

    if len(mtp_logits) != len(horizon_weights):
        raise ValueError("MTP logits and horizon weights have different lengths")
    if not mtp_logits:
        raise ValueError("at least one MTP horizon is required")

    weighted_terms: list[Tensor] = []
    active_weights: list[float] = []
    per_horizon: dict[int, Tensor] = {}
    per_tokens: dict[int, int] = {}
    for index, (logits, weight) in enumerate(zip(mtp_logits, horizon_weights)):
        horizon = index + 2
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError("each MTP logit tensor must have shape [batch, sequence, vocab]")
        targets = shift_future_targets(labels, offset=horizon - 1, fill_value=-100)
        selected = targets.ne(-100)
        token_count = int(selected.sum().item())
        per_tokens[horizon] = token_count
        if token_count == 0:
            per_horizon[horizon] = logits.sum() * 0.0
            continue
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(targets)
        horizon_loss = losses[selected].mean()
        per_horizon[horizon] = horizon_loss
        weighted_terms.append(float(weight) * horizon_loss)
        active_weights.append(float(weight))

    if not weighted_terms:
        zero = mtp_logits[0].sum() * 0.0
        return zero, per_horizon, per_tokens
    return sum(weighted_terms) / sum(active_weights), per_horizon, per_tokens


def combined_tgsa_mtp_loss(
    output: TGSAWithMTPOutput,
    labels: Tensor,
    autoregressive_loss_mask: Tensor,
    *,
    mtp_config: MTPConfig,
) -> MTPObjective:
    """Combine the registered answer loss with normalized future supervision."""

    from .phase1_model import masked_answer_loss

    autoregressive = masked_answer_loss(output.logits, labels, autoregressive_loss_mask)
    mtp, per_horizon, per_tokens = multi_token_prediction_loss(
        output.mtp_logits,
        labels,
        horizon_weights=mtp_config.weights,
    )
    total = autoregressive + mtp_config.effective_loss_weight * mtp
    return MTPObjective(
        total_loss=total,
        autoregressive_loss=autoregressive,
        mtp_loss=mtp,
        per_horizon_loss=per_horizon,
        per_horizon_tokens=per_tokens,
    )


@torch.no_grad()
def horizon_token_metrics(
    mtp_logits: Sequence[Tensor],
    labels: Tensor,
) -> dict[int, dict[str, float | int]]:
    """Return teacher-forced token accuracy/NLL for each future horizon."""

    result: dict[int, dict[str, float | int]] = {}
    for index, logits in enumerate(mtp_logits):
        horizon = index + 2
        targets = shift_future_targets(labels, offset=horizon - 1, fill_value=-100)
        selected = targets.ne(-100)
        count = int(selected.sum().item())
        if count == 0:
            result[horizon] = {"tokens": 0, "accuracy": 0.0, "nll": 0.0}
            continue
        predictions = logits.argmax(dim=-1)
        accuracy = float(predictions[selected].eq(targets[selected]).float().mean().item())
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(targets)
        result[horizon] = {
            "tokens": count,
            "accuracy": accuracy,
            "nll": float(losses[selected].mean().item()),
        }
    return result


@torch.no_grad()
def answer_horizon_metrics(
    output: TGSAWithMTPOutput,
    *,
    answer_input_indices: Tensor,
    answer_tokens: Tensor,
) -> dict[int, dict[str, float | int]]:
    """Measure how early each horizon can predict the task's answer token."""

    logits_by_horizon: tuple[Tensor, ...] = (output.logits, *output.mtp_logits)
    batch = output.logits.shape[0]
    rows = torch.arange(batch, device=output.logits.device)
    result: dict[int, dict[str, float | int]] = {}
    for horizon, logits in enumerate(logits_by_horizon, start=1):
        source = answer_input_indices - (horizon - 1)
        eligible = source.ge(0)
        count = int(eligible.sum().item())
        if count == 0:
            result[horizon] = {"examples": 0, "accuracy": 0.0, "nll": 0.0}
            continue
        selected_logits = logits[rows[eligible], source[eligible]]
        targets = answer_tokens[eligible]
        result[horizon] = {
            "examples": count,
            "accuracy": float(selected_logits.argmax(dim=-1).eq(targets).float().mean().item()),
            "nll": float(F.cross_entropy(selected_logits.float(), targets).item()),
        }
    return result


@torch.no_grad()
def teacher_forced_prefix_metrics(
    output: TGSAWithMTPOutput,
    labels: Tensor,
) -> dict[str, object]:
    """Measure consecutive correct future predictions under teacher forcing.

    This is a *quality/feasibility surrogate*, not speculative throughput.  It
    compares the model's horizon predictions with held-out tokens and therefore
    does not include target-verifier cost, cache updates, or sampling overhead.
    """

    logits_by_horizon = (output.logits, *output.mtp_logits)
    correct: list[Tensor] = []
    valid: list[Tensor] = []
    for horizon, logits in enumerate(logits_by_horizon, start=1):
        targets = shift_future_targets(labels, offset=horizon - 1, fill_value=-100)
        horizon_valid = targets.ne(-100)
        correct.append(logits.argmax(dim=-1).eq(targets) & horizon_valid)
        valid.append(horizon_valid)

    all_valid = torch.stack(valid, dim=-1).all(dim=-1)
    eligible = int(all_valid.sum().item())
    if eligible == 0:
        return {
            "eligible_positions": 0,
            "mean_consecutive_correct": 0.0,
            "full_horizon_fraction": 0.0,
            "per_horizon_conditional_accuracy": {},
            "claim_scope": "teacher-forced quality surrogate; not throughput",
        }

    accepted = torch.zeros_like(labels, dtype=torch.long)
    still_correct = all_valid.clone()
    conditional: dict[str, float] = {}
    for horizon, horizon_correct in enumerate(correct, start=1):
        denominator = int(still_correct.sum().item())
        if denominator == 0:
            conditional[str(horizon)] = 0.0
            still_correct = torch.zeros_like(still_correct)
            continue
        conditional[str(horizon)] = float(horizon_correct[still_correct].float().mean().item())
        still_correct = still_correct & horizon_correct
        accepted = accepted + still_correct.long()

    return {
        "eligible_positions": eligible,
        "mean_consecutive_correct": float(accepted[all_valid].float().mean().item()),
        "full_horizon_fraction": float(still_correct[all_valid].float().mean().item()),
        "per_horizon_conditional_accuracy": conditional,
        "claim_scope": "teacher-forced quality surrogate; not throughput",
    }


__all__ = [
    "MTPConfig",
    "MTPMode",
    "MTPObjective",
    "SequentialMTPHead",
    "TGSAWithMTPOutput",
    "TinyTGSAWithMTP",
    "answer_horizon_metrics",
    "build_tgsa_mtp_model",
    "combined_tgsa_mtp_loss",
    "flatten_tgsa_control",
    "horizon_token_metrics",
    "multi_token_prediction_loss",
    "shift_future_targets",
    "teacher_forced_prefix_metrics",
]
