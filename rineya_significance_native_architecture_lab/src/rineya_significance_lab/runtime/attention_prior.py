"""Bounded historical-significance priors for closed-loop attention experiments.

The prior is deliberately narrower than a generic attention hook.  It may read
only already-resident, previously observed cache entries.  Keys appended by the
current forward receive exactly zero prior, and the ordinary causal/padding mask
continues to decide whether a key is attendable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch

from ..torch_state import TorchSignificanceState
from .masks import bottom_right_causal_allowed

PriorMode = Literal["normalized", "raw", "shuffled", "age"]
QueryAccess = Literal["agnostic", "conditioned"]


@dataclass(frozen=True)
class AttentionPriorAccessDeclaration:
    """Explicitly declare every source of information available to the prior.

    ``query_access='conditioned'`` means the persistent S/C state may already
    contain contributions from a known-question probe at an earlier boundary.
    The prior itself never receives question tokens.  Answer, oracle-evidence,
    future-token, and future-hidden-state access are prohibited rather than
    merely documented.
    """

    query_access: QueryAccess
    answer_access: Literal["none"] = "none"
    oracle_evidence_access: Literal["none"] = "none"
    future_token_access: Literal["none"] = "none"
    future_hidden_state_access: Literal["none"] = "none"

    def __post_init__(self) -> None:
        if self.query_access not in {"agnostic", "conditioned"}:
            raise ValueError("query_access must be 'agnostic' or 'conditioned'")
        forbidden = {
            "answer_access": self.answer_access,
            "oracle_evidence_access": self.oracle_evidence_access,
            "future_token_access": self.future_token_access,
            "future_hidden_state_access": self.future_hidden_state_access,
        }
        violations = [name for name, value in forbidden.items() if value != "none"]
        if violations:
            raise ValueError(f"attention prior forbids information access through {violations}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionPriorApplicationDiagnostics:
    layer_idx: int
    mode: PriorMode
    forward_role: str
    lambda_value: float
    query_length: int
    key_length: int
    historical_key_length: int
    eligible_entries: int
    scoring_events_before_forward: int
    zero_lambda_bypass: bool
    nonzero_bias_applied: bool
    bias_min: float
    bias_max: float
    bias_abs_max: float
    current_key_bias_abs_max: float
    invalid_key_bias_abs_max: float
    attention_entropy_nats_mean: float | None
    attention_entropy_normalized_mean: float | None
    clipped_fraction: float
    source_entropy_normalized: float
    prior_entropy_normalized: float
    old_minus_recent_bias: float


@dataclass
class AttentionPriorTrace:
    """Small JSON-serializable diagnostics collector; it stores no model tensors."""

    applications: list[AttentionPriorApplicationDiagnostics] = field(default_factory=list)

    def record(self, diagnostic: AttentionPriorApplicationDiagnostics) -> None:
        self.applications.append(diagnostic)

    def to_dict(self) -> dict[str, object]:
        rows = [asdict(row) for row in self.applications]
        by_layer: dict[str, dict[str, float | int]] = {}
        for layer_idx in sorted({row.layer_idx for row in self.applications}):
            layer_rows = [row for row in self.applications if row.layer_idx == layer_idx]
            biased = [row for row in layer_rows if row.nonzero_bias_applied]
            entropies = [row.source_entropy_normalized for row in biased]
            attention_entropies = [
                row.attention_entropy_normalized_mean
                for row in layer_rows
                if row.attention_entropy_normalized_mean is not None
            ]
            role_counts: dict[str, int] = {}
            for row in layer_rows:
                role_counts[row.forward_role] = role_counts.get(row.forward_role, 0) + 1
            by_layer[str(layer_idx)] = {
                "applications": len(layer_rows),
                "zero_lambda_bypasses": sum(row.zero_lambda_bypass for row in layer_rows),
                "nonzero_bias_applications": len(biased),
                "max_abs_bias": max((row.bias_abs_max for row in layer_rows), default=0.0),
                "max_current_key_abs_bias": max(
                    (row.current_key_bias_abs_max for row in layer_rows), default=0.0
                ),
                "max_invalid_key_abs_bias": max(
                    (row.invalid_key_bias_abs_max for row in layer_rows), default=0.0
                ),
                "mean_clipped_fraction": (
                    sum(row.clipped_fraction for row in biased) / len(biased) if biased else 0.0
                ),
                "mean_source_entropy_normalized": (
                    sum(entropies) / len(entropies) if entropies else 0.0
                ),
                "mean_attention_entropy_normalized": (
                    sum(attention_entropies) / len(attention_entropies)
                    if attention_entropies
                    else 0.0
                ),
                "attention_entropy_last_minus_first": (
                    attention_entropies[-1] - attention_entropies[0]
                    if len(attention_entropies) >= 2
                    else 0.0
                ),
                "source_entropy_last_minus_first": (
                    entropies[-1] - entropies[0] if len(entropies) >= 2 else 0.0
                ),
                "mean_old_minus_recent_bias": (
                    sum(row.old_minus_recent_bias for row in biased) / len(biased)
                    if biased
                    else 0.0
                ),
                "forward_role_counts": role_counts,
            }
        return {"applications": rows, "layers": by_layer}


@dataclass(frozen=True)
class PriorBiasResult:
    bias: torch.Tensor
    diagnostics: AttentionPriorApplicationDiagnostics


def _expand_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    causal: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return an expanded original mask and its boolean allowed relation."""

    if attention_mask is None:
        allowed = torch.ones(
            (batch, query_heads, query_length, key_length),
            dtype=torch.bool,
            device=device,
        )
        if causal:
            allowed &= bottom_right_causal_allowed(
                query_length,
                key_length,
                device=device,
            ).view(1, 1, query_length, key_length)
        return None, allowed

    if attention_mask.ndim == 2:
        mask = attention_mask.view(1, 1, *attention_mask.shape)
    elif attention_mask.ndim == 3:
        mask = attention_mask.unsqueeze(1)
    elif attention_mask.ndim == 4:
        mask = attention_mask
    else:
        raise ValueError(f"attention mask must have rank 2, 3, or 4; got {attention_mask.ndim}")
    if mask.shape[0] not in (1, batch):
        raise ValueError(f"attention mask batch {mask.shape[0]} does not broadcast to {batch}")
    if mask.shape[-2] < query_length or mask.shape[-1] != key_length:
        raise ValueError(
            f"attention mask tail {tuple(mask.shape[-2:])} is incompatible with "
            f"query/key lengths {(query_length, key_length)}"
        )
    mask = mask[..., -query_length:, :].to(device=device).expand(batch, -1, -1, -1)
    mask_heads = int(mask.shape[1])
    if mask_heads == 1:
        mask = mask.expand(batch, query_heads, query_length, key_length)
    elif mask_heads == kv_heads:
        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        mask = mask.repeat_interleave(query_heads // kv_heads, dim=1)
    elif mask_heads != query_heads:
        raise ValueError(
            f"attention mask head dimension {mask_heads} must be 1, {kv_heads}, or {query_heads}"
        )

    if mask.dtype == torch.bool:
        allowed = mask
    elif mask.is_floating_point():
        # Transformers uses -inf or dtype-min for masked cells.  The -1e4 guard
        # additionally covers common finite additive padding sentinels while
        # retaining ordinary relative/logit biases.
        allowed = torch.isfinite(mask) & (mask > -1.0e4)
    else:
        raise TypeError(f"attention mask must be boolean or floating point, got {mask.dtype}")
    if causal:
        allowed = allowed & bottom_right_causal_allowed(
            query_length,
            key_length,
            device=device,
        ).view(1, 1, query_length, key_length)
    return mask, allowed


def merge_attention_prior_mask(
    attention_mask: torch.Tensor | None,
    bias: torch.Tensor,
    *,
    kv_heads: int,
    causal: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Combine a finite prior with causal/padding constraints for SDPA.

    The returned mask is always explicit.  This is important because supplying
    any additive mask makes Transformers disable SDPA's implicit causal flag.
    """

    if bias.ndim != 4:
        raise ValueError(f"bias must have shape [batch, query_heads, q, k], got {tuple(bias.shape)}")
    batch, query_heads, query_length, key_length = bias.shape
    expanded, allowed = _expand_attention_mask(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_length=query_length,
        key_length=key_length,
        device=bias.device,
        causal=causal,
    )
    if expanded is None or expanded.dtype == torch.bool:
        base = torch.zeros_like(bias, dtype=dtype)
        base = base.masked_fill(~allowed, float("-inf"))
    else:
        base = expanded.to(dtype=dtype)
        # Even a malformed padding-only input mask cannot create future access.
        base = base.masked_fill(~allowed, float("-inf"))
    finite_bias = bias.to(dtype=dtype).masked_fill(~allowed, 0.0)
    if not torch.isfinite(finite_bias).all():
        raise RuntimeError("attention prior contains non-finite values")
    return base + finite_bias


@dataclass(frozen=True)
class SignificanceAttentionPrior:
    """Create a clipped, per-head historical prior from persistent state.

    ``layer_lambdas`` overrides ``global_lambda`` for named layers.  Every
    lambda is constrained by ``max_abs_lambda`` and the normalized source is
    clipped before multiplication, so the final finite logit adjustment is at
    most ``max_abs_lambda * clip`` in magnitude.
    """

    mode: PriorMode = "normalized"
    global_lambda: float = 0.0
    layer_lambdas: Mapping[int, float] | Sequence[float] | None = None
    clip: float = 2.0
    epsilon: float = 1.0e-6
    max_abs_lambda: float = 0.25
    shuffle_seed: int = 0
    access: AttentionPriorAccessDeclaration = field(
        default_factory=lambda: AttentionPriorAccessDeclaration(query_access="agnostic")
    )

    def __post_init__(self) -> None:
        if self.mode not in {"normalized", "raw", "shuffled", "age"}:
            raise ValueError("attention-prior mode must be normalized, raw, shuffled, or age")
        for name, value in (
            ("global_lambda", self.global_lambda),
            ("clip", self.clip),
            ("epsilon", self.epsilon),
            ("max_abs_lambda", self.max_abs_lambda),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.clip <= 0 or self.epsilon <= 0 or self.max_abs_lambda <= 0:
            raise ValueError("clip, epsilon, and max_abs_lambda must be positive")
        layer_values: list[tuple[int, float]]
        if self.layer_lambdas is None:
            layer_values = []
        elif isinstance(self.layer_lambdas, Mapping):
            normalized_mapping: dict[int, float] = {}
            for raw_index, raw_value in self.layer_lambdas.items():
                index = int(raw_index)
                if index in normalized_mapping:
                    raise ValueError(f"duplicate normalized layer-lambda index {index}")
                normalized_mapping[index] = float(raw_value)
            object.__setattr__(self, "layer_lambdas", normalized_mapping)
            layer_values = list(normalized_mapping.items())
        else:
            if isinstance(self.layer_lambdas, (str, bytes)):
                raise TypeError("layer_lambdas must be a numeric sequence or integer-keyed mapping")
            normalized_sequence = tuple(float(value) for value in self.layer_lambdas)
            object.__setattr__(self, "layer_lambdas", normalized_sequence)
            layer_values = list(enumerate(normalized_sequence))
        values = [("global_lambda", float(self.global_lambda))] + [
            (f"layer_lambdas[{index}]", float(value)) for index, value in layer_values
        ]
        for name, value in values:
            if not math.isfinite(value) or abs(value) > self.max_abs_lambda:
                raise ValueError(
                    f"{name}={value} must be finite and within +/-{self.max_abs_lambda}"
                )
        if any(index < 0 for index, _ in layer_values):
            raise ValueError("layer-lambda indices must be non-negative")

    def lambda_for_layer(self, layer_idx: int) -> float:
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if self.layer_lambdas is None:
            return float(self.global_lambda)
        if not isinstance(self.layer_lambdas, Mapping):
            return (
                float(self.layer_lambdas[layer_idx])
                if layer_idx < len(self.layer_lambdas)
                else float(self.global_lambda)
            )
        return float(self.layer_lambdas.get(layer_idx, self.global_lambda))

    def is_zero_for_layer(self, layer_idx: int) -> bool:
        return self.lambda_for_layer(layer_idx) == 0.0

    def _shuffle_eligible(
        self,
        source: torch.Tensor,
        eligible: torch.Tensor,
        *,
        layer_idx: int,
    ) -> torch.Tensor:
        shuffled = source.clone()
        for batch_idx in range(source.shape[0]):
            for head_idx in range(source.shape[1]):
                indices = torch.nonzero(eligible[batch_idx, head_idx], as_tuple=False).flatten()
                if indices.numel() <= 1:
                    continue
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(self.shuffle_seed)
                    + 1_000_003 * int(layer_idx)
                    + 10_007 * int(batch_idx)
                    + 101 * int(head_idx)
                )
                permutation = torch.randperm(indices.numel(), generator=generator).to(indices.device)
                shuffled[batch_idx, head_idx, indices] = source[
                    batch_idx,
                    head_idx,
                    indices[permutation],
                ]
        return shuffled

    @staticmethod
    def _normalized_entropy(source: torch.Tensor, eligible: torch.Tensor) -> float:
        values: list[float] = []
        for batch_idx in range(source.shape[0]):
            for head_idx in range(source.shape[1]):
                selected = source[batch_idx, head_idx][eligible[batch_idx, head_idx]].float()
                if selected.numel() <= 1:
                    continue
                selected = selected.clamp_min(0.0)
                total = selected.sum()
                if float(total) <= 0.0:
                    values.append(0.0)
                    continue
                probabilities = selected / total
                entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum()
                values.append(float(entropy / math.log(selected.numel())))
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _old_minus_recent(
        bias: torch.Tensor,
        eligible: torch.Tensor,
        positions: torch.Tensor,
    ) -> float:
        values: list[float] = []
        for batch_idx in range(bias.shape[0]):
            for head_idx in range(bias.shape[1]):
                indices = torch.nonzero(eligible[batch_idx, head_idx], as_tuple=False).flatten()
                if indices.numel() < 2:
                    continue
                order = torch.argsort(positions[batch_idx, head_idx, indices], stable=True)
                ordered = indices[order]
                width = max(1, int(ordered.numel()) // 4)
                old = bias[batch_idx, head_idx, ordered[:width]].mean()
                recent = bias[batch_idx, head_idx, ordered[-width:]].mean()
                values.append(float(old - recent))
        return sum(values) / len(values) if values else 0.0

    def bias(
        self,
        state: TorchSignificanceState,
        *,
        layer_idx: int,
        query_heads: int,
        query_length: int,
        key_length: int,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
        forward_role: str = "unspecified",
    ) -> PriorBiasResult:
        """Return ``[batch, query_heads, query, key]`` finite logit bias."""

        state.validate(expected_length=key_length)
        if not forward_role or forward_role != forward_role.strip():
            raise ValueError("forward_role must be non-empty with no outer whitespace")
        if query_length <= 0 or key_length < query_length:
            raise ValueError("query_length must be positive and no greater than key_length")
        if query_heads <= 0 or query_heads % state.kv_heads:
            raise ValueError(
                f"query heads {query_heads} must be divisible by {state.kv_heads} KV heads"
            )
        lambda_value = self.lambda_for_layer(layer_idx)
        history_length = key_length - query_length
        _, allowed_q = _expand_attention_mask(
            attention_mask,
            batch=state.batch_size,
            query_heads=query_heads,
            kv_heads=state.kv_heads,
            query_length=query_length,
            key_length=key_length,
            device=state.scores.device,
            causal=causal,
        )
        groups = query_heads // state.kv_heads
        key_allowed = allowed_q.view(
            state.batch_size,
            state.kv_heads,
            groups,
            query_length,
            key_length,
        ).any(dim=(2, 3))
        historical = torch.arange(key_length, device=state.scores.device) < history_length
        historical = historical.view(1, 1, key_length).expand_as(state.scores)
        if self.mode == "age":
            eligible = historical & key_allowed
            newest_position = state.positions.amax(dim=-1, keepdim=True) + 1
            source = (newest_position - state.positions).to(torch.float32).clamp_min(0.0)
        else:
            eligible = historical & key_allowed & (state.counts > 0)
            source = state.scores if self.mode == "raw" else state.ranked(1.0)
            source = source.float()
            if self.mode == "shuffled":
                source = self._shuffle_eligible(source, eligible, layer_idx=layer_idx)

        count = eligible.sum(dim=-1, keepdim=True)
        safe_count = count.clamp_min(1).to(torch.float32)
        mean = (source * eligible).sum(dim=-1, keepdim=True) / safe_count
        centered = (source - mean) * eligible
        variance = centered.square().sum(dim=-1, keepdim=True) / safe_count
        std = variance.clamp_min(0.0).sqrt()
        standardized = torch.where(
            eligible & (count > 1) & (std > self.epsilon),
            centered / (std + self.epsilon),
            torch.zeros_like(centered),
        )
        clipped = standardized.clamp(-self.clip, self.clip)
        per_kv_bias = clipped * lambda_value
        if not torch.isfinite(per_kv_bias).all():
            raise RuntimeError("bounded attention-prior transform produced non-finite values")
        per_query_head = per_kv_bias.repeat_interleave(groups, dim=1)
        full_bias = per_query_head.unsqueeze(2).expand(
            state.batch_size,
            query_heads,
            query_length,
            key_length,
        ).clone()
        full_bias.masked_fill_(~allowed_q, 0.0)

        eligible_entries = int(eligible.sum().item())
        clipped_count = int(((standardized.abs() > self.clip) & eligible).sum().item())
        source_entropy = self._normalized_entropy(source, eligible)
        prior_entropy = self._normalized_entropy(clipped.exp(), eligible)
        diagnostic = AttentionPriorApplicationDiagnostics(
            layer_idx=int(layer_idx),
            mode=self.mode,
            forward_role=forward_role,
            lambda_value=lambda_value,
            query_length=query_length,
            key_length=key_length,
            historical_key_length=history_length,
            eligible_entries=eligible_entries,
            scoring_events_before_forward=state.scoring_events,
            zero_lambda_bypass=lambda_value == 0.0,
            nonzero_bias_applied=lambda_value != 0.0 and bool((full_bias != 0).any()),
            bias_min=float(full_bias.min().item()) if full_bias.numel() else 0.0,
            bias_max=float(full_bias.max().item()) if full_bias.numel() else 0.0,
            bias_abs_max=float(full_bias.abs().max().item()) if full_bias.numel() else 0.0,
            current_key_bias_abs_max=(
                float(full_bias[..., history_length:].abs().max().item())
                if query_length and full_bias.numel()
                else 0.0
            ),
            invalid_key_bias_abs_max=(
                float(full_bias.masked_select(~allowed_q).abs().max().item())
                if (~allowed_q).any()
                else 0.0
            ),
            attention_entropy_nats_mean=None,
            attention_entropy_normalized_mean=None,
            clipped_fraction=clipped_count / eligible_entries if eligible_entries else 0.0,
            source_entropy_normalized=source_entropy,
            prior_entropy_normalized=prior_entropy,
            old_minus_recent_bias=self._old_minus_recent(
                per_kv_bias,
                eligible,
                state.positions,
            ),
        )
        return PriorBiasResult(full_bias, diagnostic)
