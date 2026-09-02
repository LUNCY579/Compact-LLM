"""Differentiable sequential multi-token prediction for the TGSA reference model.

The module preserves two orthogonal scientific axes.

Conditioning source
-------------------
* ``dormant`` keeps an initialized MTP module but excludes its auxiliary graph;
* ``backbone`` trains the same MTP module from the backbone state and zero control;
* ``trajectory`` additionally consumes TGSA's exact per-head control tensor.

Auxiliary-gradient semantics
----------------------------
* ``joint`` allows the MTP loss to co-adapt the backbone, shared embedding,
  shared LM head, and trajectory;
* ``stop_gradient`` trains only MTP-specific parameters from detached backbone
  states and detached TGSA controls while the autoregressive objective may still
  train the backbone;
* ``frozen_probe`` freezes the backbone entirely and trains only the readout.

Trajectory mode requests ``control_only`` telemetry from exactly one registered
decoder block. It never retains quadratic attention maps. This is a training and
mechanism-identification reference, not a speculative throughput implementation;
runtime speed requires a cache-aware target verifier.
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
    TinyLMConfig,
    TinyTGSAForCausalLM,
    build_model,
    derive_initialization_seed,
)
from .telemetry import TGSATelemetry

MTPMode = Literal["dormant", "backbone", "trajectory"]
MTPGradientMode = Literal["joint", "stop_gradient", "frozen_probe"]


@dataclass(frozen=True, slots=True)
class MTPConfig:
    max_horizon: int = 3
    transition_rank: int = 64
    loss_weight: float = 0.20
    mode: MTPMode = "backbone"
    gradient_mode: MTPGradientMode = "joint"
    horizon_weights: tuple[float, ...] = ()
    control_block_index: int = -1
    pad_token_id: int = 0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.max_horizon < 2:
            raise ValueError("max_horizon must be at least 2")
        if self.transition_rank <= 0:
            raise ValueError("transition_rank must be positive")
        if self.loss_weight < 0.0 or not math.isfinite(self.loss_weight):
            raise ValueError("loss_weight must be finite and non-negative")
        if self.mode not in {"dormant", "backbone", "trajectory"}:
            raise ValueError(f"unsupported MTP mode: {self.mode}")
        if self.gradient_mode not in {
            "joint",
            "stop_gradient",
            "frozen_probe",
        }:
            raise ValueError(
                f"unsupported MTP gradient mode: {self.gradient_mode}"
            )
        if self.mode == "dormant" and self.gradient_mode != "joint":
            raise ValueError(
                "dormant MTP has no auxiliary gradient and therefore permits "
                "only gradient_mode='joint'"
            )
        if self.gradient_mode == "frozen_probe" and self.loss_weight <= 0.0:
            raise ValueError(
                "frozen_probe requires a positive MTP loss_weight"
            )
        if (
            self.horizon_weights
            and len(self.horizon_weights) != self.max_horizon - 1
        ):
            raise ValueError(
                "horizon_weights must contain exactly max_horizon - 1 entries"
            )
        if any(
            weight <= 0.0 or not math.isfinite(weight)
            for weight in self.weights
        ):
            raise ValueError(
                "every horizon weight must be finite and positive"
            )
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    @property
    def weights(self) -> tuple[float, ...]:
        if self.horizon_weights:
            return self.horizon_weights
        return tuple(
            1.0 / float(horizon)
            for horizon in range(2, self.max_horizon + 1)
        )

    @property
    def effective_loss_weight(self) -> float:
        return 0.0 if self.mode == "dormant" else self.loss_weight

    @property
    def auxiliary_updates_backbone(self) -> bool:
        return self.mode != "dormant" and self.gradient_mode == "joint"

    @property
    def backbone_is_frozen(self) -> bool:
        return self.gradient_mode == "frozen_probe"


@dataclass(slots=True)
class TGSAWithMTPOutput:
    logits: Tensor
    mtp_logits: tuple[Tensor, ...]
    block_telemetry: tuple[TGSATelemetry | None, ...]
    final_hidden_states: Tensor
    trajectory_control: Tensor
    gradient_mode: MTPGradientMode


@dataclass(slots=True)
class MTPObjective:
    total_loss: Tensor
    autoregressive_loss: Tensor
    mtp_loss: Tensor
    per_horizon_loss: dict[int, Tensor]
    per_horizon_tokens: dict[int, int]
    autoregressive_in_training_objective: bool


class _MTPTransition(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        rank: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        fused_dim = 3 * embed_dim
        self.norm = RMSNorm(fused_dim)
        self.down = nn.Linear(fused_dim, rank, bias=bias)
        self.up = nn.Linear(rank, embed_dim, bias=bias)
        self.gate = nn.Linear(fused_dim, embed_dim, bias=True)
        self.activation = nn.SiLU()
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
        fused = torch.cat(
            (
                state,
                previous_future_embedding,
                trajectory_control,
            ),
            dim=-1,
        )
        proposal = self.up(
            self.activation(self.down(self.norm(fused)))
        )
        return state + torch.sigmoid(self.gate(fused)) * proposal


def _shared_embedding(
    token_ids: Tensor,
    token_embedding: nn.Embedding,
    *,
    detach_shared_weights: bool,
) -> Tensor:
    if not detach_shared_weights:
        return token_embedding(token_ids)
    return F.embedding(
        token_ids,
        token_embedding.weight.detach(),
        padding_idx=token_embedding.padding_idx,
    )


def _shared_lm_projection(
    hidden_states: Tensor,
    lm_head: nn.Linear,
    *,
    detach_shared_weights: bool,
) -> Tensor:
    if not detach_shared_weights:
        return lm_head(hidden_states)
    bias = None if lm_head.bias is None else lm_head.bias.detach()
    return F.linear(
        hidden_states,
        lm_head.weight.detach(),
        bias,
    )


class SequentialMTPHead(nn.Module):
    """Predict horizons 2..K while sharing registered backbone weights.

    In isolated auxiliary modes, the numerical values of the embedding and LM
    projection remain shared, but their weights are detached from the auxiliary
    graph. This blocks a subtle leakage path that detaching only the hidden state
    and trajectory control would leave open.
    """

    def __init__(self, embed_dim: int, config: MTPConfig) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.config = config
        self.control_norm = RMSNorm(embed_dim)
        self.control_projection = nn.Linear(
            embed_dim,
            embed_dim,
            bias=config.bias,
        )
        self.transitions = nn.ModuleList(
            _MTPTransition(
                embed_dim,
                config.transition_rank,
                bias=config.bias,
            )
            for _ in range(config.max_horizon - 1)
        )
        self.output_norms = nn.ModuleList(
            RMSNorm(embed_dim)
            for _ in range(config.max_horizon - 1)
        )
        nn.init.xavier_uniform_(
            self.control_projection.weight,
            gain=0.05,
        )
        if self.control_projection.bias is not None:
            nn.init.zeros_(self.control_projection.bias)

    def _project_control(self, control: Tensor) -> Tensor:
        if control.shape[-1] != self.embed_dim:
            raise ValueError(
                "trajectory_control has the wrong feature dimension"
            )
        return self.control_projection(self.control_norm(control))

    def teacher_forced(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        trajectory_control: Tensor,
        *,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
        detach_shared_weights: bool = False,
    ) -> tuple[Tensor, ...]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, embed_dim]"
            )
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError(
                "labels must match hidden-state batch and sequence"
            )
        if trajectory_control.shape != hidden_states.shape:
            raise ValueError(
                "trajectory_control must match hidden_states"
            )
        control = self._project_control(trajectory_control)
        state = hidden_states
        outputs: list[Tensor] = []
        for transition_index, (transition, output_norm) in enumerate(
            zip(self.transitions, self.output_norms)
        ):
            previous_ids = shift_future_targets(
                labels,
                offset=transition_index,
                fill_value=self.config.pad_token_id,
            )
            previous_ids = previous_ids.masked_fill(
                previous_ids.lt(0),
                self.config.pad_token_id,
            )
            previous_embedding = _shared_embedding(
                previous_ids,
                token_embedding,
                detach_shared_weights=detach_shared_weights,
            )
            state = transition(
                state,
                previous_embedding,
                control,
            )
            outputs.append(
                _shared_lm_projection(
                    output_norm(state),
                    lm_head,
                    detach_shared_weights=detach_shared_weights,
                )
            )
        return tuple(outputs)

    def recursive(
        self,
        hidden_states: Tensor,
        first_future_token_ids: Tensor,
        trajectory_control: Tensor,
        *,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
        detach_shared_weights: bool = False,
    ) -> tuple[Tensor, ...]:
        if first_future_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "first_future_token_ids must have shape [batch, sequence]"
            )
        if trajectory_control.shape != hidden_states.shape:
            raise ValueError(
                "trajectory_control must match hidden_states"
            )
        control = self._project_control(trajectory_control)
        state = hidden_states
        previous_ids = first_future_token_ids
        outputs: list[Tensor] = []
        for transition, output_norm in zip(
            self.transitions,
            self.output_norms,
        ):
            previous_embedding = _shared_embedding(
                previous_ids,
                token_embedding,
                detach_shared_weights=detach_shared_weights,
            )
            state = transition(
                state,
                previous_embedding,
                control,
            )
            logits = _shared_lm_projection(
                output_norm(state),
                lm_head,
                detach_shared_weights=detach_shared_weights,
            )
            outputs.append(logits)
            previous_ids = logits.argmax(dim=-1)
        return tuple(outputs)


def shift_future_targets(
    labels: Tensor,
    *,
    offset: int,
    fill_value: int = -100,
) -> Tensor:
    """Align ``labels[t + offset]`` with source position ``t``."""

    if labels.ndim != 2:
        raise ValueError(
            "labels must have shape [batch, sequence]"
        )
    if offset < 0:
        raise ValueError("offset must be non-negative")
    shifted = torch.full_like(labels, int(fill_value))
    if offset < labels.shape[1]:
        shifted[:, : labels.shape[1] - offset] = labels[:, offset:]
    return shifted


def flatten_tgsa_control(
    block_telemetry: Sequence[TGSATelemetry | None],
    *,
    block_index: int,
    embed_dim: int,
) -> Tensor:
    """Flatten the registered TGSA control from [B,H,T,Dh] to [B,T,D]."""

    if not block_telemetry:
        raise ValueError(
            "trajectory-conditioned MTP requires TGSA telemetry"
        )
    index = (
        block_index
        if block_index >= 0
        else len(block_telemetry) + block_index
    )
    if not 0 <= index < len(block_telemetry):
        raise ValueError(
            "control_block_index is outside the decoder stack"
        )
    telemetry = block_telemetry[index]
    if telemetry is None or telemetry.control_signals is None:
        raise ValueError(
            "trajectory-conditioned MTP requires control_only or full "
            "TGSA control signals"
        )
    control = telemetry.control_signals
    if control.ndim != 4:
        raise ValueError(
            "TGSA control_signals must be "
            "[batch, heads, sequence, head_dim]"
        )
    batch, heads, sequence, head_dim = control.shape
    if heads * head_dim != embed_dim:
        raise ValueError(
            "TGSA control geometry does not reconstruct the embedding"
        )
    return control.transpose(1, 2).reshape(
        batch,
        sequence,
        embed_dim,
    )


def _resolve_block_index(index: int, block_count: int) -> int:
    resolved = index if index >= 0 else block_count + index
    if not 0 <= resolved < block_count:
        raise ValueError(
            "control_block_index is outside the decoder stack"
        )
    return resolved


class TinyTGSAWithMTP(nn.Module):
    """Tiny TGSA backbone with explicit auxiliary-gradient semantics."""

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
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(mtp_initialization_seed))
            self.mtp_head = SequentialMTPHead(
                backbone.config.embed_dim,
                mtp_config,
            )

        self._control_block_index: int | None = None
        if mtp_config.mode == "trajectory":
            index = _resolve_block_index(
                mtp_config.control_block_index,
                len(self.backbone.blocks),
            )
            attention = self.backbone.blocks[index].attention
            if not isinstance(
                attention,
                TrajectoryGuidedSequentialAttention,
            ):
                raise ValueError(
                    "trajectory-conditioned MTP requires a TGSA "
                    "attention block at control_block_index"
                )
            attention.config = attention.config.with_updates(
                detach_telemetry=(
                    mtp_config.gradient_mode != "joint"
                )
            )
            self._control_block_index = index

        if mtp_config.backbone_is_frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "TinyTGSAWithMTP":
        super().train(mode)
        if self.mtp_config.backbone_is_frozen:
            self.backbone.eval()
        return self

    def _backbone_forward(
        self,
        input_ids: Tensor,
        *,
        key_padding_mask: Tensor | None,
    ) -> tuple[
        Tensor,
        Tensor,
        tuple[TGSATelemetry | None, ...],
    ]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )
        _, sequence = input_ids.shape
        if sequence > self.backbone.config.max_sequence_length:
            raise ValueError(
                "input sequence exceeds the configured maximum"
            )
        if key_padding_mask is not None and (
            key_padding_mask.dtype != torch.bool
            or key_padding_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "key_padding_mask must be bool and match input_ids"
            )

        positions = torch.arange(
            sequence,
            device=input_ids.device,
        )
        hidden = self.backbone.token_embedding(input_ids)
        hidden = (
            hidden
            + self.backbone.position_embedding(positions)[None, :, :]
        )
        hidden = self.backbone.embedding_dropout(hidden)

        telemetry: list[TGSATelemetry | None] = []
        for index, block in enumerate(self.backbone.blocks):
            level: Literal[
                "none",
                "control_only",
            ] = (
                "control_only"
                if index == self._control_block_index
                else "none"
            )
            hidden, block_telemetry = block(
                hidden,
                key_padding_mask=key_padding_mask,
                telemetry_level=level,
            )
            telemetry.append(block_telemetry)
        final_hidden = self.backbone.final_norm(hidden)
        return (
            self.backbone.lm_head(final_hidden),
            final_hidden,
            tuple(telemetry),
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        recursive: bool = False,
    ) -> TGSAWithMTPOutput:
        logits, final_hidden, telemetry = self._backbone_forward(
            input_ids,
            key_padding_mask=key_padding_mask,
        )
        if self.mtp_config.mode == "trajectory":
            control = flatten_tgsa_control(
                telemetry,
                block_index=self.mtp_config.control_block_index,
                embed_dim=self.backbone.config.embed_dim,
            )
        else:
            control = torch.zeros_like(final_hidden)

        isolate_auxiliary = (
            self.mtp_config.gradient_mode != "joint"
        )
        mtp_hidden = (
            final_hidden.detach()
            if isolate_auxiliary
            else final_hidden
        )
        mtp_control = (
            control.detach()
            if isolate_auxiliary
            else control
        )
        shared_arguments = {
            "token_embedding": self.backbone.token_embedding,
            "lm_head": self.backbone.lm_head,
            "detach_shared_weights": isolate_auxiliary,
        }
        if recursive:
            mtp_logits = self.mtp_head.recursive(
                mtp_hidden,
                logits.argmax(dim=-1),
                mtp_control,
                **shared_arguments,
            )
        else:
            if labels is None:
                raise ValueError(
                    "teacher-forced MTP requires labels"
                )
            mtp_logits = self.mtp_head.teacher_forced(
                mtp_hidden,
                labels,
                mtp_control,
                **shared_arguments,
            )
        return TGSAWithMTPOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            block_telemetry=telemetry,
            final_hidden_states=final_hidden,
            trajectory_control=mtp_control,
            gradient_mode=self.mtp_config.gradient_mode,
        )


def build_tgsa_mtp_model(
    lm_config: TinyLMConfig,
    mtp_config: MTPConfig,
    *,
    initialization_seed: int,
    canonical_state: Mapping[str, Tensor] | None = None,
) -> TinyTGSAWithMTP:
    backbone = build_model(
        lm_config,
        initialization_seed=initialization_seed,
        canonical_state=canonical_state,
    )
    return TinyTGSAWithMTP(
        backbone,
        mtp_config,
        mtp_initialization_seed=derive_initialization_seed(
            initialization_seed,
            "tgsa_mtp_head_v1",
        ),
    )


def multi_token_prediction_loss(
    mtp_logits: Sequence[Tensor],
    labels: Tensor,
    *,
    horizon_weights: Sequence[float],
) -> tuple[Tensor, dict[int, Tensor], dict[int, int]]:
    if (
        not mtp_logits
        or len(mtp_logits) != len(horizon_weights)
    ):
        raise ValueError(
            "MTP logits and horizon weights must be non-empty and aligned"
        )
    terms: list[Tensor] = []
    weights: list[float] = []
    per_horizon: dict[int, Tensor] = {}
    per_tokens: dict[int, int] = {}
    for index, (logits, weight) in enumerate(
        zip(mtp_logits, horizon_weights)
    ):
        horizon = index + 2
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError(
                "each MTP logit tensor must be "
                "[batch, sequence, vocab]"
            )
        targets = shift_future_targets(
            labels,
            offset=horizon - 1,
            fill_value=-100,
        )
        selected = targets.ne(-100)
        count = int(selected.sum().item())
        per_tokens[horizon] = count
        if count == 0:
            per_horizon[horizon] = logits.sum() * 0.0
            continue
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(targets)
        value = losses[selected].mean()
        per_horizon[horizon] = value
        terms.append(float(weight) * value)
        weights.append(float(weight))
    if not terms:
        return (
            mtp_logits[0].sum() * 0.0,
            per_horizon,
            per_tokens,
        )
    return (
        sum(terms) / sum(weights),
        per_horizon,
        per_tokens,
    )


def combined_tgsa_mtp_loss(
    output: TGSAWithMTPOutput,
    labels: Tensor,
    autoregressive_loss_mask: Tensor,
    *,
    mtp_config: MTPConfig,
) -> MTPObjective:
    from .phase1_model import masked_answer_loss

    autoregressive = masked_answer_loss(
        output.logits,
        labels,
        autoregressive_loss_mask,
    )
    mtp, per_horizon, per_tokens = multi_token_prediction_loss(
        output.mtp_logits,
        labels,
        horizon_weights=mtp_config.weights,
    )

    if mtp_config.effective_loss_weight == 0.0:
        total = autoregressive
        include_autoregressive = True
    elif mtp_config.backbone_is_frozen:
        total = mtp_config.effective_loss_weight * mtp
        include_autoregressive = False
    else:
        total = (
            autoregressive
            + mtp_config.effective_loss_weight * mtp
        )
        include_autoregressive = True

    return MTPObjective(
        total_loss=total,
        autoregressive_loss=autoregressive,
        mtp_loss=mtp,
        per_horizon_loss=per_horizon,
        per_horizon_tokens=per_tokens,
        autoregressive_in_training_objective=(
            include_autoregressive
        ),
    )


@torch.no_grad()
def horizon_token_metrics(
    mtp_logits: Sequence[Tensor],
    labels: Tensor,
) -> dict[int, dict[str, float | int]]:
    result: dict[int, dict[str, float | int]] = {}
    for index, logits in enumerate(mtp_logits):
        horizon = index + 2
        targets = shift_future_targets(
            labels,
            offset=horizon - 1,
            fill_value=-100,
        )
        selected = targets.ne(-100)
        count = int(selected.sum().item())
        if count == 0:
            result[horizon] = {
                "tokens": 0,
                "accuracy": 0.0,
                "nll": 0.0,
            }
            continue
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(targets)
        result[horizon] = {
            "tokens": count,
            "accuracy": float(
                logits.argmax(dim=-1)[selected]
                .eq(targets[selected])
                .float()
                .mean()
                .item()
            ),
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
    rows = torch.arange(
        output.logits.shape[0],
        device=output.logits.device,
    )
    result: dict[int, dict[str, float | int]] = {}
    for horizon, logits in enumerate(
        (output.logits, *output.mtp_logits),
        start=1,
    ):
        source = answer_input_indices - (horizon - 1)
        eligible = source.ge(0)
        count = int(eligible.sum().item())
        if count == 0:
            result[horizon] = {
                "examples": 0,
                "accuracy": 0.0,
                "nll": 0.0,
            }
            continue
        selected_logits = logits[
            rows[eligible],
            source[eligible],
        ]
        targets = answer_tokens[eligible]
        result[horizon] = {
            "examples": count,
            "accuracy": float(
                selected_logits.argmax(dim=-1)
                .eq(targets)
                .float()
                .mean()
                .item()
            ),
            "nll": float(
                F.cross_entropy(
                    selected_logits.float(),
                    targets,
                ).item()
            ),
        }
    return result


@torch.no_grad()
def teacher_forced_prefix_metrics(
    output: TGSAWithMTPOutput,
    labels: Tensor,
) -> dict[str, object]:
    correct: list[Tensor] = []
    valid: list[Tensor] = []
    for horizon, logits in enumerate(
        (output.logits, *output.mtp_logits),
        start=1,
    ):
        targets = shift_future_targets(
            labels,
            offset=horizon - 1,
            fill_value=-100,
        )
        horizon_valid = targets.ne(-100)
        correct.append(
            logits.argmax(dim=-1).eq(targets)
            & horizon_valid
        )
        valid.append(horizon_valid)
    all_valid = torch.stack(valid, dim=-1).all(dim=-1)
    eligible = int(all_valid.sum().item())
    if eligible == 0:
        return {
            "eligible_positions": 0,
            "mean_consecutive_correct": 0.0,
            "full_horizon_fraction": 0.0,
            "per_horizon_conditional_accuracy": {},
            "claim_scope": (
                "teacher-forced quality surrogate; not throughput"
            ),
        }
    accepted = torch.zeros_like(labels, dtype=torch.long)
    still_correct = all_valid.clone()
    conditional: dict[str, float] = {}
    for horizon, horizon_correct in enumerate(
        correct,
        start=1,
    ):
        denominator = int(still_correct.sum().item())
        conditional[str(horizon)] = (
            float(
                horizon_correct[still_correct]
                .float()
                .mean()
                .item()
            )
            if denominator
            else 0.0
        )
        still_correct = still_correct & horizon_correct
        accepted = accepted + still_correct.long()
    return {
        "eligible_positions": eligible,
        "mean_consecutive_correct": float(
            accepted[all_valid].float().mean().item()
        ),
        "full_horizon_fraction": float(
            still_correct[all_valid].float().mean().item()
        ),
        "per_horizon_conditional_accuracy": conditional,
        "claim_scope": (
            "teacher-forced quality surrogate; not throughput"
        ),
    }


__all__ = [
    "MTPConfig",
    "MTPGradientMode",
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
