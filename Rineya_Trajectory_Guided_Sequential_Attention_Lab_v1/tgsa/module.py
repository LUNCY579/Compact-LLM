"""Auditable reference implementation of trajectory-guided sequential attention.

This module prioritizes semantic clarity, causal correctness, and ablation
fidelity. It is not yet a fused production kernel. The Qwen-compatible phase
must preserve these semantics while adding RoPE, GQA, cache layouts, and fused
execution.
"""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
from torch import Tensor, nn

from .config import TGSAConfig
from .telemetry import TGSATelemetry, cosine_per_batch, rms_per_batch


class RMSNorm(nn.Module):
    """Minimal RMSNorm that works across supported PyTorch versions."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        variance = value.float().square().mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.eps).to(value.dtype)
        return normalized * self.weight


class TrajectoryEncoder(nn.Module):
    """Encode a control signal in the shared carrier coordinate system.

    ``learned`` mode is a **residual around the identity**::

        encode(x) = x + up(SiLU(down(RMSNorm(x))))

    The raw, same-shaped control tensor is therefore always present on the path
    the downstream mechanism consumes. This is required by the preregistered
    information-matching argument. Given the current carrier ``c_j``, the raw
    finite difference ``d_j = c_j - c_{j-1}`` and the raw history mean
    ``m_j = (c_j + c_{j-1}) / 2`` are affine bijections of each other, so
    ``learned_delta`` and ``history_control`` receive algebraically equivalent
    access to ``c_{j-1}``.

    The pre-remediation encoder was ``up(SiLU(down(RMSNorm(x))))`` with no raw
    term. ``RMSNorm`` discards magnitude and ``SiLU`` is not injective, so that
    path was not invertible and the comparison degenerated into two matched
    *normalized nonlinear encodings* rather than an information-matched control.
    See ``PHASE1_LAUNCH_AUDIT.md`` finding P0-6.
    """

    def __init__(self, dim: int, rank: int, mode: str, bias: bool) -> None:
        super().__init__()
        self.mode = mode
        self.norm = RMSNorm(dim)
        if mode == "learned":
            self.down = nn.Linear(dim, rank, bias=bias)
            self.up = nn.Linear(rank, dim, bias=bias)
            self.activation = nn.SiLU()
        else:
            self.down = None
            self.up = None
            self.activation = None
        self.reset_trajectory_parameters()

    def reset_trajectory_parameters(self) -> None:
        """Apply the registered encoder initialization.

        Callers must invoke this *after* any enclosing model-wide
        ``module.apply(...)`` initializer, which would otherwise overwrite these
        tensors with the generic linear-layer initialization.

        ``up.weight`` is zero so the encoder is exactly the identity at step 0.
        That makes the information-matching property provable at initialization
        and keeps the cold-start perturbation at zero. Gradient reaches ``down``
        from the first optimizer step onward because ``up`` becomes non-zero
        after that step, which is the standard zero-init residual-adapter
        convention.
        """

        if self.mode != "learned":
            return
        assert self.down is not None and self.up is not None
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, velocity: Tensor) -> Tensor:
        if self.mode == "none":
            return torch.zeros_like(velocity)
        if self.mode == "raw":
            return velocity
        assert self.down is not None and self.up is not None
        assert self.activation is not None
        residual = self.up(self.activation(self.down(self.norm(velocity))))
        return velocity + residual


def _add_attention_bias(logits: Tensor, attention_bias: Tensor | None) -> Tensor:
    if attention_bias is None:
        return logits
    if not attention_bias.is_floating_point():
        raise TypeError("attention_bias must be floating point and additive")
    batch, query_len, key_len = logits.shape
    if attention_bias.ndim == 2:
        if attention_bias.shape != (query_len, key_len):
            raise ValueError("2D attention_bias must have shape [query, key]")
        return logits + attention_bias.to(device=logits.device, dtype=logits.dtype)
    if attention_bias.ndim == 3:
        if attention_bias.shape not in {(1, query_len, key_len), (batch, query_len, key_len)}:
            raise ValueError("3D attention_bias must have shape [1|batch, query, key]")
        return logits + attention_bias.to(device=logits.device, dtype=logits.dtype)
    raise ValueError("attention_bias must have rank 2 or 3")


def _apply_masks(
    logits: Tensor,
    *,
    is_causal: bool,
    attention_bias: Tensor | None,
    key_padding_mask: Tensor | None,
) -> Tensor:
    batch, query_len, key_len = logits.shape
    masked = _add_attention_bias(logits, attention_bias)
    mask_value = torch.finfo(masked.dtype).min
    if is_causal:
        if query_len != key_len:
            raise ValueError("reference causal path currently requires query_len == key_len")
        future = torch.ones(
            query_len,
            key_len,
            dtype=torch.bool,
            device=masked.device,
        ).triu(diagonal=1)
        masked = masked.masked_fill(future, mask_value)
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool:
            raise TypeError("key_padding_mask must be bool with True meaning masked")
        if key_padding_mask.shape != (batch, key_len):
            raise ValueError("key_padding_mask must have shape [batch, key]")
        masked = masked.masked_fill(key_padding_mask[:, None, :], mask_value)
    return masked


class ParallelCausalSelfAttention(nn.Module):
    """Conventional parallel-MHA reference control with the same public mask API."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.output = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        is_causal: bool = True,
        attention_bias: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.embed_dim:
            raise ValueError("hidden_states must have shape [batch, sequence, embed_dim]")
        batch, sequence, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(batch, sequence, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        flat_logits = logits.reshape(batch * self.num_heads, sequence, sequence)
        flat_padding = None
        if key_padding_mask is not None:
            flat_padding = key_padding_mask[:, None, :].expand(batch, self.num_heads, sequence)
            flat_padding = flat_padding.reshape(batch * self.num_heads, sequence)
        flat_bias = attention_bias
        if attention_bias is not None and attention_bias.ndim == 3 and attention_bias.shape[0] == batch:
            flat_bias = attention_bias[:, None, :, :].expand(batch, self.num_heads, sequence, sequence)
            flat_bias = flat_bias.reshape(batch * self.num_heads, sequence, sequence)
        flat_logits = _apply_masks(
            flat_logits,
            is_causal=is_causal,
            attention_bias=flat_bias,
            key_padding_mask=flat_padding,
        )
        weights = torch.softmax(flat_logits.float(), dim=-1).to(flat_logits.dtype)
        weights = self.dropout(weights).view(batch, self.num_heads, sequence, sequence)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(batch, sequence, self.embed_dim)
        return self.output(context)


class TrajectoryGuidedSequentialAttention(nn.Module):
    """Differentiable sequential heads controlled by inter-head state trajectory.

    Head ``j`` operates on one native feature group and a shared carrier. The
    carrier makes ``c[j-1] - c[j-2]`` a meaningful finite difference because all
    states live in the same coordinate system. The encoded difference controls
    prediction, feature injection, Q/K/V construction, and a soft key-importance
    prior. A head then corrects the predicted carrier.
    """

    def __init__(self, config: TGSAConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.head_dim
        heads = config.num_heads

        self.carrier_init = nn.Linear(config.embed_dim, dim, bias=config.bias)
        self.trajectory_encoders = nn.ModuleList(
            TrajectoryEncoder(dim, config.trajectory_rank, config.trajectory_mode, config.bias)
            for _ in range(heads)
        )
        self.pre_gate = nn.ModuleList(
            nn.Linear(3 * dim, 3 * dim, bias=True) for _ in range(heads)
        )
        self.state_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.control_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.predictor_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.q_trajectory = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.k_trajectory = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.v_trajectory = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.q_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.k_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.v_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.head_output = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.importance_projection = nn.ModuleList(
            nn.Linear(2 * dim, 1, bias=True) for _ in range(heads)
        )
        self.correction_gate = nn.ModuleList(
            nn.Linear(3 * dim, dim, bias=True) for _ in range(heads)
        )
        self.correction_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=config.bias) for _ in range(heads)
        )
        self.carrier_norm = nn.ModuleList(RMSNorm(dim) for _ in range(heads))
        self.importance_strength_logit = nn.Parameter(torch.zeros(heads))
        self.output_projection = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.output_dropout = nn.Dropout(config.dropout)
        self.attention_dropout = nn.Dropout(config.dropout)

        self._initialize_control_paths()

    def _initialize_control_paths(self) -> None:
        # Neutral gates at initialization. The trajectory residual is exactly zero
        # at step 0 (TrajectoryEncoder.up is zero-initialized), so the encoder
        # starts as the identity and there is no cold-start impulse.
        for layer in self.pre_gate:
            nn.init.zeros_(layer.bias)
        for layer in self.correction_gate:
            nn.init.zeros_(layer.bias)

    def reset_registered_parameters(self) -> None:
        """Re-apply this operator's registered initialization.

        An enclosing model that runs a generic ``apply(...)`` initializer over
        every ``nn.Linear`` would otherwise clobber the trajectory-encoder
        initialization, silently replacing the zero-initialized residual with a
        generic normal draw. Enclosing models must call this after their own
        global initializer.
        """

        for encoder in self.trajectory_encoders:
            encoder.reset_trajectory_parameters()
        self._initialize_control_paths()

    @overload
    def forward(
        self,
        hidden_states: Tensor,
        *,
        is_causal: bool = True,
        attention_bias: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        telemetry_level: Literal["none"] = "none",
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        hidden_states: Tensor,
        *,
        is_causal: bool = True,
        attention_bias: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        telemetry_level: Literal["summary", "full"],
    ) -> tuple[Tensor, TGSATelemetry]: ...

    def forward(
        self,
        hidden_states: Tensor,
        *,
        is_causal: bool = True,
        attention_bias: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        telemetry_level: Literal["none", "summary", "full"] | None = None,
    ) -> Tensor | tuple[Tensor, TGSATelemetry]:
        config = self.config
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != config.embed_dim:
            raise ValueError("hidden_states must have shape [batch, sequence, embed_dim]")
        if telemetry_level is None:
            telemetry_level = config.telemetry_level
        if telemetry_level not in {"none", "summary", "full"}:
            raise ValueError("telemetry_level must be none, summary, or full")

        batch, sequence, _ = hidden_states.shape
        groups = hidden_states.view(batch, sequence, config.num_heads, config.head_dim)
        groups = groups.unbind(dim=2)

        carrier = self.carrier_init(hidden_states)
        previous_carrier = carrier
        previous_velocity = torch.zeros_like(carrier)

        head_outputs: list[Tensor] = []
        velocity_rms: list[Tensor] = []
        control_signal_rms: list[Tensor] = []
        acceleration_rms: list[Tensor] = []
        velocity_cosine: list[Tensor] = []
        update_rms: list[Tensor] = []
        state_gate_mean: list[Tensor] = []
        control_gate_mean: list[Tensor] = []
        predictor_gate_mean: list[Tensor] = []
        correction_gate_mean: list[Tensor] = []
        importance_mean: list[Tensor] = []
        importance_min: list[Tensor] = []
        importance_max: list[Tensor] = []
        attention_entropy: list[Tensor] = []
        full_carriers: list[Tensor] = [carrier] if telemetry_level == "full" else []
        full_velocities: list[Tensor] = []
        full_control_signals: list[Tensor] = []
        full_importance: list[Tensor] = []
        full_attention: list[Tensor] = []

        for head_index, feature_group in enumerate(groups):
            velocity = carrier - previous_carrier
            acceleration = velocity - previous_velocity
            if head_index == 0:
                control_signal = torch.zeros_like(velocity)
            elif config.control_source == "velocity":
                control_signal = velocity
            else:
                control_signal = 0.5 * (carrier + previous_carrier)
            encoded_control = self.trajectory_encoders[head_index](control_signal)

            gate_input = torch.cat((feature_group, carrier, encoded_control), dim=-1)
            state_gate, control_gate, predictor_gate = self.pre_gate[head_index](gate_input).chunk(3, dim=-1)
            state_gate = torch.sigmoid(state_gate)
            control_gate = torch.sigmoid(control_gate)
            predictor_gate = config.max_predictor_step * torch.sigmoid(predictor_gate)

            predicted_carrier = carrier
            if config.use_predictor and config.trajectory_mode != "none":
                predicted_carrier = predicted_carrier + predictor_gate * self.predictor_projection[head_index](encoded_control)

            head_input = feature_group
            if config.use_state_cascade:
                head_input = head_input + state_gate * self.state_projection[head_index](predicted_carrier)
            if config.use_control_injection and config.trajectory_mode != "none":
                head_input = head_input + (
                    config.trajectory_scale
                    * control_gate
                    * self.control_projection[head_index](encoded_control)
                )

            q_input = head_input
            k_input = head_input
            v_input = head_input
            if config.trajectory_mode != "none":
                if config.condition_q:
                    q_input = q_input + self.q_trajectory[head_index](encoded_control)
                if config.condition_k:
                    k_input = k_input + self.k_trajectory[head_index](encoded_control)
                if config.condition_v:
                    v_input = v_input + self.v_trajectory[head_index](encoded_control)

            query = self.q_projection[head_index](q_input)
            key = self.k_projection[head_index](k_input)
            value = self.v_projection[head_index](v_input)
            logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config.head_dim)
            logits = _apply_masks(
                logits,
                is_causal=is_causal,
                attention_bias=attention_bias,
                key_padding_mask=key_padding_mask,
            )

            importance = torch.sigmoid(
                self.importance_projection[head_index](
                    torch.cat((predicted_carrier, encoded_control), dim=-1)
                )
            )
            if config.use_importance_bias:
                bounded_importance = config.importance_floor + (
                    1.0 - config.importance_floor
                ) * importance
                learned_scale = 2.0 * torch.sigmoid(self.importance_strength_logit[head_index])
                key_prior = bounded_importance.log().transpose(-2, -1)
                logits = logits + config.importance_logit_scale * learned_scale * key_prior

            weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
            dropped_weights = self.attention_dropout(weights)
            context = torch.matmul(dropped_weights, value)
            head_output = self.head_output[head_index](context)
            head_outputs.append(head_output)

            correction_input = torch.cat((predicted_carrier, head_output, encoded_control), dim=-1)
            correction_gate = config.max_correction_step * torch.sigmoid(
                self.correction_gate[head_index](correction_input)
            )
            corrected_carrier = predicted_carrier + correction_gate * self.correction_projection[head_index](head_output)
            if config.normalize_carrier:
                corrected_carrier = self.carrier_norm[head_index](corrected_carrier)

            if telemetry_level != "none":
                velocity_rms.append(rms_per_batch(velocity))
                control_signal_rms.append(rms_per_batch(control_signal))
                acceleration_rms.append(rms_per_batch(acceleration))
                velocity_cosine.append(cosine_per_batch(velocity, previous_velocity))
                update_rms.append(rms_per_batch(corrected_carrier - carrier))
                state_gate_mean.append(state_gate.float().mean(dim=(1, 2)))
                control_gate_mean.append(control_gate.float().mean(dim=(1, 2)))
                predictor_gate_mean.append(predictor_gate.float().mean(dim=(1, 2)))
                correction_gate_mean.append(correction_gate.float().mean(dim=(1, 2)))
                importance_mean.append(importance.float().mean(dim=(1, 2)))
                importance_min.append(importance.float().amin(dim=(1, 2)))
                importance_max.append(importance.float().amax(dim=(1, 2)))
                entropy = -(weights.float() * weights.float().clamp_min(1.0e-12).log()).sum(dim=-1)
                attention_entropy.append(entropy.mean(dim=-1))
                if telemetry_level == "full":
                    full_velocities.append(velocity)
                    full_control_signals.append(control_signal)
                    full_importance.append(importance)
                    full_attention.append(weights)
                    full_carriers.append(corrected_carrier)

            previous_carrier = carrier
            previous_velocity = velocity
            carrier = corrected_carrier

        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output_dropout(self.output_projection(concatenated))
        if telemetry_level == "none":
            return output

        telemetry = TGSATelemetry(
            velocity_rms=torch.stack(velocity_rms, dim=1),
            control_signal_rms=torch.stack(control_signal_rms, dim=1),
            acceleration_rms=torch.stack(acceleration_rms, dim=1),
            velocity_cosine=torch.stack(velocity_cosine, dim=1),
            carrier_update_rms=torch.stack(update_rms, dim=1),
            state_gate_mean=torch.stack(state_gate_mean, dim=1),
            control_gate_mean=torch.stack(control_gate_mean, dim=1),
            predictor_gate_mean=torch.stack(predictor_gate_mean, dim=1),
            correction_gate_mean=torch.stack(correction_gate_mean, dim=1),
            importance_mean=torch.stack(importance_mean, dim=1),
            importance_min=torch.stack(importance_min, dim=1),
            importance_max=torch.stack(importance_max, dim=1),
            attention_entropy=torch.stack(attention_entropy, dim=1),
            carrier_states=torch.stack(full_carriers, dim=1) if full_carriers else None,
            velocities=torch.stack(full_velocities, dim=1) if full_velocities else None,
            control_signals=torch.stack(full_control_signals, dim=1) if full_control_signals else None,
            importance_scores=torch.stack(full_importance, dim=1) if full_importance else None,
            attention_maps=torch.stack(full_attention, dim=1) if full_attention else None,
        )
        if config.detach_telemetry:
            telemetry = telemetry.detached()
        return output, telemetry
