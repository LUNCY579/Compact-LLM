"""Trajectory-guided sequential attention for a pretrained Rineya/Qwen decoder.

This is the *retrofit* operator: a drop-in replacement for a Qwen-style
``*Attention`` module that keeps every native pretrained tensor and adds a
sequential trajectory-control pathway on top.

## The exactness property

The single most important design constraint is that **conversion must be exactly
function-preserving before the first optimizer step**. Not approximately: the
retrofit at initialization must compute the same values the native module
computes, so a converted checkpoint starts with the pretrained model's full
capability intact.

That is achieved as follows:

* Q, K, V, and O projections are the **native tensors, reused unchanged**. They
  project from the full hidden state exactly as the pretrained model does.
* RoPE and the GQA head->group mapping are reproduced exactly.
* The trajectory carrier machinery runs, but the only place it can influence the
  output is through the conditioning projections, and those are
  **zero-initialized**. Their contribution is therefore identically zero.

So at conversion the module reduces algebraically to native attention. The
remaining difference is floating-point summation order, which is measured rather
than assumed (see `damage.py`).

## Why the carrier is still alive at step 0

Zero-initializing *both* sides of a residual adapter kills its gradient. Here
only the injection is zeroed:

* ``carrier_init`` and the correction path are randomly initialized, so the
  carrier genuinely evolves across sequential heads and produces a non-trivial
  velocity signal;
* the conditioning projections (``q_trajectory``, ``k_trajectory``,
  ``v_trajectory``) are zero, so the *output* is unchanged;
* the gradient with respect to those conditioning weights is
  ``dL/dq_j (x) control_j``, which is non-zero because ``control_j`` is non-zero.

The model therefore starts as an exact copy of the pretrained function and can
begin learning trajectory control from the very first step.

## Sequential structure and the KV cache

The carrier recurses **across attention heads within a token**, not across
sequence positions. That is what makes the operator cache-safe: K and V are the
native per-group projections of the current hidden state, so the cache layout,
contents, and memory footprint are byte-identical to the native GQA cache.
Prefill and single-token decode both work unchanged.

## A structural detail worth knowing

Head 0 has no previous carrier, so its velocity - and therefore its control
signal - is identically zero. Its conditioning weights are inert and receive no
gradient. The trajectory pathway acts on ``num_attention_heads - 1`` heads; see
:attr:`RineyaTGSAAttention.effective_conditioned_heads`. This is inherited from
the Phase-1 operator and is arguably correct, since there is no trajectory
before the first head, but it must be accounted for in any capacity or
parameter-count comparison.

This module is deliberately not a fused kernel. It spends compute to be correct
and inspectable; speed is a later study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import RineyaTGSAConfig

CONDITIONING_TARGETS = ("q", "k", "v")


def rotate_half(value: Tensor) -> Tensor:
    """Rotate the trailing dimension by half, the standard RoPE convention."""

    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to query and key.

    ``query``/``key`` are ``[batch, heads, sequence, head_dim]``; ``cos``/``sin``
    are ``[batch, sequence, head_dim]`` and are broadcast over the head axis.
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class RotaryEmbedding(nn.Module):
    """Standard RoPE table with the native theta and scaling.

    Kept as a module (not a free function) so the inverse-frequency buffer moves
    with the model across devices and dtypes, and so a converted checkpoint
    carries its positional convention explicitly.
    """

    def __init__(self, head_dim: int, *, theta: float = 10000.0, scaling: float = 1.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.head_dim = head_dim
        self.theta = theta
        self.scaling = scaling
        inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def forward(self, position_ids: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = position_ids.to(torch.float32) * self.scaling
        frequencies = positions[..., None] * self.inverse_frequency.to(positions.device)[None, None, :]
        emb = torch.cat((frequencies, frequencies), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class TrajectoryEncoder(nn.Module):
    """Residual-around-identity encoder for the control signal.

    ``encode(x) = x + up(SiLU(down(RMSNorm(x))))`` with ``up`` zero-initialized,
    so the encoder is exactly the identity at step 0 and the raw control tensor
    is always additively present on the path the mechanism consumes. This is the
    information-matching property established in Phase 1.
    """

    def __init__(self, dim: int, rank: int, *, bias: bool = False) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-6) if hasattr(nn, "RMSNorm") else _RMSNorm(dim)
        self.down = nn.Linear(dim, rank, bias=bias)
        self.up = nn.Linear(rank, dim, bias=bias)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.up(self.activation(self.down(self.norm(value))))


class _RMSNorm(nn.Module):
    """Fallback RMSNorm for torch builds without ``nn.RMSNorm``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        variance = value.float().square().mean(dim=-1, keepdim=True)
        return (value * torch.rsqrt(variance + self.eps).to(value.dtype)) * self.weight


@dataclass(slots=True)
class TrajectoryTelemetry:
    """Per-head summaries. Cheap scalars only; tensors are opt-in."""

    carrier_rms: list[float]
    velocity_rms: list[float]
    control_rms: list[float]
    conditioning_rms: list[float]
    attention_entropy: list[float]
    carrier_states: list[Tensor] | None = None
    velocities: list[Tensor] | None = None
    controls: list[Tensor] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "carrier_rms": self.carrier_rms,
            "velocity_rms": self.velocity_rms,
            "control_rms": self.control_rms,
            "conditioning_rms": self.conditioning_rms,
            "attention_entropy": self.attention_entropy,
        }


class RineyaTGSAAttention(nn.Module):
    """Drop-in trajectory-guided replacement for a Qwen-style attention module.

    The public interface deliberately mirrors the Hugging Face decoder-layer
    contract so the surrounding model, generation loop, and cache plumbing need
    no changes.
    """

    def __init__(self, config: RineyaTGSAConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group_size = config.num_attention_heads // config.num_key_value_heads

        # ---- native pretrained projections, reused unchanged ----
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.output_bias)

        self.rotary = RotaryEmbedding(
            self.head_dim, theta=config.rope_theta, scaling=config.rope_scaling
        )

        # ---- new trajectory machinery ----
        # Live at step 0 so gradients flow, but unable to change the output.
        self.carrier_init = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.trajectory_encoders = nn.ModuleList(
            TrajectoryEncoder(self.head_dim, config.trajectory_rank) for _ in range(self.num_heads)
        )
        self.correction_gate = nn.ModuleList(
            nn.Linear(2 * self.head_dim, self.head_dim, bias=True) for _ in range(self.num_heads)
        )
        self.correction_projection = nn.ModuleList(
            nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(self.num_heads)
        )
        self.carrier_norm = nn.ModuleList(
            (nn.RMSNorm(self.head_dim, eps=1e-6) if hasattr(nn, "RMSNorm") else _RMSNorm(self.head_dim))
            for _ in range(self.num_heads)
        )

        # ---- zero-initialized conditioning: the ONLY path to the output ----
        self.conditioning = nn.ModuleDict()
        for target in CONDITIONING_TARGETS:
            if getattr(config, f"condition_{target}"):
                self.conditioning[target] = nn.ModuleList(
                    nn.Linear(self.head_dim, self.head_dim, bias=False)
                    for _ in range(self.num_heads)
                )

        self._intervention: Callable[[int, Tensor], Tensor] | None = None
        self.reset_trajectory_parameters()

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------

    def reset_trajectory_parameters(self) -> None:
        """Initialize only the new modules. Native projections are untouched.

        Callers that run a model-wide ``apply(...)`` initializer must re-invoke
        this afterwards, or the zero-initialized conditioning is silently
        replaced by a generic draw and exactness is lost.
        """

        nn.init.normal_(self.carrier_init.weight, mean=0.0, std=0.02)
        for encoder in self.trajectory_encoders:
            encoder.reset_parameters()
        for layer in self.correction_gate:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)
        for layer in self.correction_projection:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        for target in self.conditioning:
            for layer in self.conditioning[target]:
                # THE exactness guarantee. Do not change without re-running the
                # conversion-damage report.
                nn.init.zeros_(layer.weight)

    def trajectory_parameter_names(self) -> list[str]:
        """Names of every tensor introduced by the retrofit."""

        native = {"q_proj", "k_proj", "v_proj", "o_proj"}
        return sorted(
            name
            for name, _ in self.named_parameters()
            if name.split(".")[0] not in native
        )

    @property
    def effective_conditioned_heads(self) -> int:
        """Heads whose trajectory conditioning can actually reach the output.

        Head 0 has no previous carrier, so its velocity and hence its control
        signal are identically zero. Its conditioning weights are inert and
        receive no gradient. The trajectory pathway therefore acts on
        ``num_attention_heads - 1`` heads, which is what a parameter-count or
        capacity comparison must use.
        """

        return max(self.num_heads - 1, 0)

    def conditioning_is_neutral(self) -> bool:
        """True when the retrofit provably computes native attention."""

        return all(
            bool(torch.count_nonzero(layer.weight) == 0)
            for target in self.conditioning
            for layer in self.conditioning[target]
        )

    # ------------------------------------------------------------------
    # interventions
    # ------------------------------------------------------------------

    def set_intervention(self, hook: Callable[[int, Tensor], Tensor] | None) -> None:
        """Install a per-head control transform, used by causal falsifiers."""

        self._intervention = hook

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        telemetry: Literal["none", "summary", "full"] = "none",
    ) -> tuple[Tensor, Tensor | None, tuple[Tensor, Tensor] | None, TrajectoryTelemetry | None]:
        batch, sequence, _ = hidden_states.shape
        dtype = hidden_states.dtype

        query = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            offset = past_key_value[0].shape[-2] if past_key_value is not None else 0
            position_ids = torch.arange(
                offset, offset + sequence, device=hidden_states.device
            ).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary(position_ids, dtype=query.dtype)
        query, key = apply_rotary(query, key, cos, sin)

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None

        # ---- carrier recursion across heads (never across positions) ----
        carrier = self.carrier_init(hidden_states)
        previous_carrier = carrier

        contexts: list[Tensor] = []
        weights_out: list[Tensor] = []
        carrier_rms: list[float] = []
        velocity_rms: list[float] = []
        control_rms: list[float] = []
        conditioning_rms: list[float] = []
        entropy: list[float] = []
        full_carriers: list[Tensor] = []
        full_velocities: list[Tensor] = []
        full_controls: list[Tensor] = []

        for head in range(self.num_heads):
            velocity = carrier - previous_carrier
            control = self.trajectory_encoders[head](velocity)
            if self._intervention is not None:
                control = self._intervention(head, control)

            group = head // self.group_size
            q_head = query[:, head]
            k_head = key[:, group]
            v_head = value[:, group]

            conditioning_magnitude = 0.0
            if "q" in self.conditioning:
                delta = self.conditioning["q"][head](control)
                conditioning_magnitude += float(delta.detach().float().square().mean().sqrt())
                q_head = q_head + delta
            if "k" in self.conditioning:
                # Conditioning K would alter cached keys, so it is applied to the
                # CURRENT window only; cached keys stay native by construction.
                delta = self.conditioning["k"][head](control)
                conditioning_magnitude += float(delta.detach().float().square().mean().sqrt())
                k_head = torch.cat((k_head[:, : k_head.shape[1] - sequence], k_head[:, -sequence:] + delta), dim=1)
            if "v" in self.conditioning:
                delta = self.conditioning["v"][head](control)
                conditioning_magnitude += float(delta.detach().float().square().mean().sqrt())
                v_head = torch.cat((v_head[:, : v_head.shape[1] - sequence], v_head[:, -sequence:] + delta), dim=1)

            logits = torch.matmul(q_head, k_head.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                logits = logits + attention_mask[:, 0] if attention_mask.dim() == 4 else logits + attention_mask
            probabilities = torch.softmax(logits.float(), dim=-1).to(dtype)
            context = torch.matmul(probabilities, v_head)
            contexts.append(context)
            if output_attentions:
                weights_out.append(probabilities)

            correction_input = torch.cat((carrier, context), dim=-1)
            gate = torch.sigmoid(self.correction_gate[head](correction_input))
            corrected = carrier + gate * self.correction_projection[head](context)
            corrected = self.carrier_norm[head](corrected)

            if telemetry != "none":
                carrier_rms.append(float(carrier.detach().float().square().mean().sqrt()))
                velocity_rms.append(float(velocity.detach().float().square().mean().sqrt()))
                control_rms.append(float(control.detach().float().square().mean().sqrt()))
                conditioning_rms.append(conditioning_magnitude)
                detached = probabilities.detach().float()
                entropy.append(
                    float(-(detached * detached.clamp_min(1e-12).log()).sum(-1).mean())
                )
                if telemetry == "full":
                    full_carriers.append(carrier.detach())
                    full_velocities.append(velocity.detach())
                    full_controls.append(control.detach())

            previous_carrier = carrier
            carrier = corrected

        merged = torch.stack(contexts, dim=1).transpose(1, 2).reshape(batch, sequence, -1)
        output = self.o_proj(merged)

        report = None
        if telemetry != "none":
            report = TrajectoryTelemetry(
                carrier_rms=carrier_rms,
                velocity_rms=velocity_rms,
                control_rms=control_rms,
                conditioning_rms=conditioning_rms,
                attention_entropy=entropy,
                carrier_states=full_carriers or None,
                velocities=full_velocities or None,
                controls=full_controls or None,
            )
        attentions = torch.stack(weights_out, dim=1) if output_attentions else None
        return output, attentions, present, report


class NativeReferenceAttention(nn.Module):
    """Parallel GQA reference used to prove the retrofit is function-preserving.

    Deliberately written independently of :class:`RineyaTGSAAttention` so an
    exactness test compares two implementations rather than one implementation
    against itself.
    """

    def __init__(self, config: RineyaTGSAConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group_size = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.output_bias)
        self.rotary = RotaryEmbedding(self.head_dim, theta=config.rope_theta, scaling=config.rope_scaling)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, sequence, _ = hidden_states.shape
        dtype = hidden_states.dtype
        query = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            offset = past_key_value[0].shape[-2] if past_key_value is not None else 0
            position_ids = torch.arange(
                offset, offset + sequence, device=hidden_states.device
            ).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary(position_ids, dtype=query.dtype)
        query, key = apply_rotary(query, key, cos, sin)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None
        key = key.repeat_interleave(self.group_size, dim=1)
        value = value.repeat_interleave(self.group_size, dim=1)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            logits = logits + (attention_mask[:, 0] if attention_mask.dim() == 4 else attention_mask)
        probabilities = torch.softmax(logits.float(), dim=-1).to(dtype)
        context = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, sequence, -1)
        return self.o_proj(context), present
