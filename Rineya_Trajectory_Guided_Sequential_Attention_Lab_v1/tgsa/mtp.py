"""Multi-token prediction primitives for the TGSA complementarity study.

The main causal-LM head remains responsible for offset 1 (the next token).
Auxiliary heads predict offsets 2..K from the *same causal hidden state*.  No
future token is ever fed into a predictor; future tokens are labels only.

The module is deliberately model-agnostic.  It receives final hidden states and
an output-projection weight, so the tiny Phase-2 model and a pretrained
Rineya/Qwen wrapper use exactly the same target construction and loss semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MTP_TARGET_MODES = ("future", "duplicate_next", "shuffled_batch")
MTPTargetMode = Literal["future", "duplicate_next", "shuffled_batch"]


class _RMSNorm(nn.Module):
    """Version-stable RMSNorm used by the auxiliary residual heads."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        variance = value.float().square().mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.eps).to(value.dtype)
        return normalized * self.weight


@dataclass(frozen=True, slots=True)
class MTPConfig:
    """Configuration for auxiliary future-token heads.

    ``max_offset`` includes the furthest predicted token.  For example,
    ``max_offset=4`` creates heads for offsets 2, 3, and 4; the ordinary LM head
    already predicts offset 1.

    ``loss_weight`` is the total auxiliary contribution after the per-offset
    losses have been normalized.  It therefore does not grow merely because a
    larger horizon is selected.
    """

    hidden_size: int
    vocab_size: int
    max_offset: int = 4
    rank: int = 64
    loss_weight: float = 0.30
    offset_decay: float = 0.75
    target_mode: MTPTargetMode = "future"
    detach_backbone: bool = False
    normalize_hidden: bool = True
    tie_output_embeddings: bool = True
    min_proposal_confidence: float = 0.0
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must exceed one")
        if self.max_offset < 2:
            raise ValueError("max_offset must be at least 2")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")
        if not 0.0 < self.offset_decay <= 1.0:
            raise ValueError("offset_decay must be in (0, 1]")
        if self.target_mode not in MTP_TARGET_MODES:
            raise ValueError(f"unsupported target_mode: {self.target_mode}")
        if not 0.0 <= self.min_proposal_confidence <= 1.0:
            raise ValueError("min_proposal_confidence must be in [0, 1]")

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(range(2, self.max_offset + 1))

    @property
    def offset_weights(self) -> dict[int, float]:
        raw = {offset: self.offset_decay ** (offset - 2) for offset in self.offsets}
        denominator = sum(raw.values())
        return {offset: weight / denominator for offset, weight in raw.items()}

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MTPResidualHead(nn.Module):
    """A zero-impact residual transform around the final hidden state.

    The ``up`` projection is zero-initialized.  Therefore every auxiliary head
    begins as the identity and, when the output embedding is tied, begins with
    the parent model's vocabulary geometry instead of an unrelated random
    classifier.  The auxiliary path never changes the main LM logits.
    """

    def __init__(self, hidden_size: int, rank: int, *, normalize: bool = True) -> None:
        super().__init__()
        self.norm: nn.Module = _RMSNorm(hidden_size) if normalize else nn.Identity()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.activation = nn.SiLU()
        self.gate_logit = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate_logit)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.up(self.activation(self.down(self.norm(hidden_states))))
        return hidden_states + torch.sigmoid(self.gate_logit) * residual


class MultiTokenPredictionHeads(nn.Module):
    """Offset-specific residual heads with a shared/tied vocabulary projection."""

    def __init__(self, config: MTPConfig) -> None:
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict(
            {
                str(offset): MTPResidualHead(
                    config.hidden_size,
                    config.rank,
                    normalize=config.normalize_hidden,
                )
                for offset in config.offsets
            }
        )
        self.untied_output = (
            None
            if config.tie_output_embeddings
            else nn.ModuleDict(
                {
                    str(offset): nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                    for offset in config.offsets
                }
            )
        )
        self.reset_registered_parameters()

    def reset_registered_parameters(self) -> None:
        for head in self.heads.values():
            assert isinstance(head, MTPResidualHead)
            head.reset_parameters()
        if self.untied_output is not None:
            for projection in self.untied_output.values():
                nn.init.normal_(projection.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        output_weight: Tensor | None = None,
        output_bias: Tensor | None = None,
    ) -> dict[int, Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden_size]")
        if self.config.detach_backbone:
            hidden_states = hidden_states.detach()

        if self.untied_output is None:
            if output_weight is None:
                raise ValueError("tied MTP heads require output_weight")
            if output_weight.shape != (self.config.vocab_size, self.config.hidden_size):
                raise ValueError(
                    "output_weight must have shape [vocab_size, hidden_size]; "
                    f"got {tuple(output_weight.shape)}"
                )
        logits: dict[int, Tensor] = {}
        for offset in self.config.offsets:
            transformed = self.heads[str(offset)](hidden_states)
            if self.untied_output is None:
                logits[offset] = F.linear(transformed, output_weight, output_bias)
            else:
                logits[offset] = self.untied_output[str(offset)](transformed)
        return logits

    def proposal_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(slots=True)
class MTPLoss:
    """Normalized auxiliary loss plus auditable per-offset components."""

    total: Tensor
    per_offset: dict[int, Tensor]
    supervised_tokens: dict[int, int]
    normalized_weights: dict[int, float]

    def detached_dict(self) -> dict[str, object]:
        return {
            "total": float(self.total.detach().float()),
            "per_offset": {
                str(offset): float(loss.detach().float())
                for offset, loss in sorted(self.per_offset.items())
            },
            "supervised_tokens": {
                str(offset): count for offset, count in sorted(self.supervised_tokens.items())
            },
            "normalized_weights": {
                str(offset): weight for offset, weight in sorted(self.normalized_weights.items())
            },
        }


@dataclass(slots=True)
class JointCausalLoss:
    main: Tensor
    auxiliary: Tensor
    total: Tensor
    mtp: MTPLoss

    def detached_dict(self) -> dict[str, object]:
        return {
            "main": float(self.main.detach().float()),
            "auxiliary": float(self.auxiliary.detach().float()),
            "total": float(self.total.detach().float()),
            "mtp": self.mtp.detached_dict(),
        }


def _validate_token_masks(
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None,
    target_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask is None:
        attention = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        attention = attention_mask.to(dtype=torch.bool)
    if target_mask is None:
        target = attention
    else:
        if target_mask.shape != input_ids.shape:
            raise ValueError("target_mask must match input_ids")
        target = target_mask.to(dtype=torch.bool) & attention
    return attention, target


def build_offset_targets(
    input_ids: Tensor,
    offset: int,
    *,
    target_mode: MTPTargetMode = "future",
    attention_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
    ignore_index: int = -100,
    donor_index: Tensor | None = None,
) -> Tensor:
    """Create labels for one future offset without exposing future inputs.

    ``target_mask`` is a mask over *tokens that may be supervised*.  For an SFT
    batch it should be true only on assistant/target tokens.  A source position
    receives a label only when both its source token and the selected target
    token are non-padding and the selected target is supervised.

    ``duplicate_next`` keeps the exact same number of source positions as the
    future target but points every auxiliary head at offset 1.  It is a
    parameter- and gradient-budget control for the future-horizon inductive bias.

    ``shuffled_batch`` uses a fixed-point-free donor assignment supplied by the
    caller (or a one-row roll for batch > 1).  It is a negative causal control,
    never a primary training arm.
    """

    if offset < 1:
        raise ValueError("offset must be positive")
    if target_mode not in MTP_TARGET_MODES:
        raise ValueError(f"unsupported target_mode: {target_mode}")
    batch, sequence = input_ids.shape
    attention, supervised = _validate_token_masks(
        input_ids,
        attention_mask=attention_mask,
        target_mask=target_mask,
    )
    labels = torch.full_like(input_ids, ignore_index)
    if sequence <= offset:
        return labels

    source_slice = slice(0, sequence - offset)
    source_valid = attention[:, source_slice]
    if target_mode == "duplicate_next":
        target_positions = input_ids[:, 1 : 1 + (sequence - offset)]
        target_valid = attention[:, 1 : 1 + (sequence - offset)]
        target_supervised = supervised[:, 1 : 1 + (sequence - offset)]
    else:
        target_positions = input_ids[:, offset:]
        target_valid = attention[:, offset:]
        target_supervised = supervised[:, offset:]
        if target_mode == "shuffled_batch":
            if batch < 2:
                raise ValueError("shuffled_batch requires batch size >= 2")
            if donor_index is None:
                donor_index = torch.roll(torch.arange(batch, device=input_ids.device), shifts=1)
            if donor_index.ndim != 1 or donor_index.shape[0] != batch:
                raise ValueError("donor_index must have shape [batch]")
            if torch.any(donor_index == torch.arange(batch, device=donor_index.device)):
                raise ValueError("donor_index must be fixed-point-free")
            target_positions = target_positions.index_select(0, donor_index.to(input_ids.device))
            target_valid = target_valid.index_select(0, donor_index.to(input_ids.device))
            # Receiver supervision controls whether a row is part of the loss;
            # donor padding controls whether its token is valid.

    valid = source_valid & target_valid & target_supervised
    view = labels[:, source_slice]
    view[valid] = target_positions[valid]
    return labels


def causal_next_token_loss(
    logits: Tensor,
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Standard offset-1 causal loss under the same token-mask semantics."""

    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    labels = build_offset_targets(
        input_ids,
        1,
        target_mode="future",
        attention_mask=attention_mask,
        target_mask=target_mask,
        ignore_index=ignore_index,
    )
    selected = labels.ne(ignore_index)
    if not selected.any():
        raise ValueError("main causal loss has no supervised targets")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def compute_mtp_loss(
    mtp_logits: Mapping[int, Tensor],
    input_ids: Tensor,
    config: MTPConfig,
    *,
    attention_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
    donor_index: Tensor | None = None,
) -> MTPLoss:
    """Compute a horizon-normalized future-token objective."""

    expected = set(config.offsets)
    if set(mtp_logits) != expected:
        raise ValueError(
            f"mtp_logits offsets {sorted(mtp_logits)} do not match config {sorted(expected)}"
        )
    weights = config.offset_weights
    losses: dict[int, Tensor] = {}
    token_counts: dict[int, int] = {}
    active_weight = 0.0
    weighted: Tensor | None = None
    for offset in config.offsets:
        logits = mtp_logits[offset]
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError(f"offset {offset}: logits shape is incompatible with input_ids")
        labels = build_offset_targets(
            input_ids,
            offset,
            target_mode=config.target_mode,
            attention_mask=attention_mask,
            target_mask=target_mask,
            ignore_index=config.ignore_index,
            donor_index=donor_index,
        )
        count = int(labels.ne(config.ignore_index).sum().item())
        token_counts[offset] = count
        if count == 0:
            continue
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=config.ignore_index,
        )
        losses[offset] = loss
        weight = weights[offset]
        active_weight += weight
        contribution = weight * loss
        weighted = contribution if weighted is None else weighted + contribution
    if weighted is None or active_weight <= 0.0:
        raise ValueError("MTP loss has no supervised future targets")
    total = weighted / active_weight
    active_weights = {
        offset: (weights[offset] / active_weight if offset in losses else 0.0)
        for offset in config.offsets
    }
    return MTPLoss(
        total=total,
        per_offset=losses,
        supervised_tokens=token_counts,
        normalized_weights=active_weights,
    )


def joint_causal_mtp_loss(
    main_logits: Tensor,
    mtp_logits: Mapping[int, Tensor],
    input_ids: Tensor,
    config: MTPConfig,
    *,
    attention_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
    donor_index: Tensor | None = None,
) -> JointCausalLoss:
    main = causal_next_token_loss(
        main_logits,
        input_ids,
        attention_mask=attention_mask,
        target_mask=target_mask,
        ignore_index=config.ignore_index,
    )
    mtp = compute_mtp_loss(
        mtp_logits,
        input_ids,
        config,
        attention_mask=attention_mask,
        target_mask=target_mask,
        donor_index=donor_index,
    )
    auxiliary = config.loss_weight * mtp.total
    return JointCausalLoss(main=main, auxiliary=auxiliary, total=main + auxiliary, mtp=mtp)
