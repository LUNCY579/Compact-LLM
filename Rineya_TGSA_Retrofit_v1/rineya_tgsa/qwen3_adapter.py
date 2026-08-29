"""TGSA attention that slots into a real transformers Qwen3 decoder.

## Why this exists separately from `attention.py`

`attention.py` proves the *idea* against a hand-written parallel GQA reference.
That reference was written by me, so it shares my assumptions — and it turned out
to share an omission: **real Qwen3 applies per-head RMSNorm to Q and K before
RoPE** (`q_norm`, `k_norm`). A retrofit without them silently drops two
pretrained tensors per layer while an exactness test against my own reference
stays green.

That is the concrete reason architecture-reference exactness does not imply
real-checkpoint exactness, and why the two are reported separately.

This module removes the class of error entirely by **subclassing
`Qwen3Attention`**. Every native tensor — `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `q_norm`, `k_norm` — is created and shaped by transformers itself, so
`load_state_dict` from the native module is total and no tensor can be
forgotten. RoPE is not recomputed: the `(cos, sin)` the model already passes down
is used verbatim, so the positional convention cannot drift.

## What is still ours

The sequential carrier recursion across query heads, and the zero-initialized
conditioning that is its only route to the output.

## The honest caveat about exactness here

Native Qwen3 dispatches to an attention *interface* (`sdpa` by default, `eager`
optionally) that computes all heads in one batched call. This module computes
attention per head, because the carrier must evolve between heads. Per-head and
batched attention are mathematically identical but not necessarily bitwise
identical, since they reduce in a different order.

So:

* against `eager` in float64, the conversion is expected to be exact;
* against `sdpa` in bfloat16, small non-zero drift is expected and is a
  statement about floating-point reduction order, not about lost pretrained
  function.

Both are measured rather than assumed. `damage.py` reports the distribution.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

from .attention import TrajectoryEncoder, TrajectoryTelemetry, _RMSNorm
from .config import RineyaTGSAConfig

try:  # pragma: no cover - exercised whenever transformers is installed
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    Qwen3Attention = object  # type: ignore[assignment,misc]
    apply_rotary_pos_emb = None  # type: ignore[assignment]
    repeat_kv = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


NATIVE_TENSOR_ROOTS = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


class Qwen3TGSAAttention(Qwen3Attention):  # type: ignore[misc]
    """Drop-in replacement for `Qwen3Attention` with a trajectory pathway.

    Inherits every pretrained tensor from the parent, including the `q_norm` and
    `k_norm` layers a hand-rolled reimplementation would have missed.
    """

    def __init__(self, config, layer_idx: int, tgsa_config: RineyaTGSAConfig) -> None:
        if not TRANSFORMERS_AVAILABLE:  # pragma: no cover
            raise RuntimeError("transformers with Qwen3 support is required")
        super().__init__(config, layer_idx)
        self.tgsa_config = tgsa_config
        self.num_heads = tgsa_config.num_attention_heads
        self.num_kv_heads = tgsa_config.num_key_value_heads
        self.group_size = tgsa_config.group_size

        dim = self.head_dim
        self.carrier_init = nn.Linear(config.hidden_size, dim, bias=False)
        self.trajectory_encoders = nn.ModuleList(
            TrajectoryEncoder(dim, tgsa_config.trajectory_rank) for _ in range(self.num_heads)
        )
        self.correction_gate = nn.ModuleList(
            nn.Linear(2 * dim, dim, bias=True) for _ in range(self.num_heads)
        )
        self.correction_projection = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(self.num_heads)
        )
        self.carrier_norm = nn.ModuleList(
            (nn.RMSNorm(dim, eps=1e-6) if hasattr(nn, "RMSNorm") else _RMSNorm(dim))
            for _ in range(self.num_heads)
        )
        self.conditioning = nn.ModuleDict()
        for target in ("q", "k", "v"):
            if getattr(tgsa_config, f"condition_{target}"):
                self.conditioning[target] = nn.ModuleList(
                    nn.Linear(dim, dim, bias=False) for _ in range(self.num_heads)
                )
        self._intervention: Callable[[int, Tensor], Tensor] | None = None
        self._telemetry_level = "none"
        self.last_telemetry: TrajectoryTelemetry | None = None
        self.reset_trajectory_parameters()

    # ------------------------------------------------------------------

    def reset_trajectory_parameters(self) -> None:
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
                nn.init.zeros_(layer.weight)  # the exactness guarantee

    @property
    def effective_conditioned_heads(self) -> int:
        """Head 0 is the bootstrap head and carries no trajectory."""

        return max(self.num_heads - 1, 0)

    def conditioning_is_neutral(self) -> bool:
        return all(
            bool(torch.count_nonzero(layer.weight) == 0)
            for target in self.conditioning
            for layer in self.conditioning[target]
        )

    def trajectory_parameter_names(self) -> list[str]:
        return sorted(
            name
            for name, _ in self.named_parameters()
            if name.split(".")[0] not in NATIVE_TENSOR_ROOTS
        )

    def load_native_(self, native: nn.Module) -> list[str]:
        """Copy every pretrained tensor from a native `Qwen3Attention`.

        Returns the copied keys. Raises if the native module exposes a tensor
        this class does not have, which is how a future transformers version
        adding a new pretrained sublayer gets caught instead of silently dropped.
        """

        native_state = native.state_dict()
        own_state = self.state_dict()
        missing = [key for key in native_state if key not in own_state]
        if missing:
            raise RuntimeError(
                f"native attention has tensors the retrofit lacks: {missing}; "
                f"conversion would silently drop pretrained weights"
            )
        with torch.no_grad():
            for key, tensor in native_state.items():
                if own_state[key].shape != tensor.shape:
                    raise ValueError(
                        f"{key}: native shape {tuple(tensor.shape)} != retrofit "
                        f"{tuple(own_state[key].shape)}"
                    )
                own_state[key].copy_(tensor)
        return sorted(native_state)

    def set_intervention(self, hook: Callable[[int, Tensor], Tensor] | None) -> None:
        self._intervention = hook

    def set_telemetry(self, level: str) -> None:
        if level not in {"none", "summary", "full"}:
            raise ValueError(f"unsupported telemetry level {level!r}")
        self._telemetry_level = level

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Identical to the parent, INCLUDING q_norm / k_norm.
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        batch, _, sequence, _ = query.shape
        carrier = self.carrier_init(hidden_states)
        previous_carrier = carrier

        key_expanded = repeat_kv(key, self.group_size)
        value_expanded = repeat_kv(value, self.group_size)

        contexts: list[Tensor] = []
        carrier_rms: list[float] = []
        velocity_rms: list[float] = []
        control_rms: list[float] = []
        conditioning_rms: list[float] = []
        entropy: list[float] = []

        for head in range(self.num_heads):
            velocity = carrier - previous_carrier
            control = self.trajectory_encoders[head](velocity)
            if self._intervention is not None:
                control = self._intervention(head, control)

            q_head = query[:, head]
            k_head = key_expanded[:, head]
            v_head = value_expanded[:, head]

            magnitude = 0.0
            if "q" in self.conditioning:
                delta = self.conditioning["q"][head](control)
                magnitude += float(delta.detach().float().square().mean().sqrt())
                q_head = q_head + delta
            if "k" in self.conditioning:
                delta = self.conditioning["k"][head](control)
                magnitude += float(delta.detach().float().square().mean().sqrt())
                k_head = torch.cat(
                    (k_head[:, : k_head.shape[1] - sequence], k_head[:, -sequence:] + delta), dim=1
                )
            if "v" in self.conditioning:
                delta = self.conditioning["v"][head](control)
                magnitude += float(delta.detach().float().square().mean().sqrt())
                v_head = torch.cat(
                    (v_head[:, : v_head.shape[1] - sequence], v_head[:, -sequence:] + delta), dim=1
                )

            logits = torch.matmul(q_head, k_head.transpose(-2, -1)) * self.scaling
            if attention_mask is not None:
                causal = attention_mask[:, 0, :, : k_head.shape[-2]]
                logits = logits + causal
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q_head.dtype)
            contexts.append(torch.matmul(probabilities, v_head))

            correction_input = torch.cat((carrier, contexts[-1]), dim=-1)
            gate = torch.sigmoid(self.correction_gate[head](correction_input))
            corrected = self.carrier_norm[head](
                carrier + gate * self.correction_projection[head](contexts[-1])
            )

            if self._telemetry_level != "none":
                carrier_rms.append(float(carrier.detach().float().square().mean().sqrt()))
                velocity_rms.append(float(velocity.detach().float().square().mean().sqrt()))
                control_rms.append(float(control.detach().float().square().mean().sqrt()))
                conditioning_rms.append(magnitude)
                detached = probabilities.detach().float()
                entropy.append(
                    float(-(detached * detached.clamp_min(1e-12).log()).sum(-1).mean())
                )

            previous_carrier = carrier
            carrier = corrected

        attn_output = torch.stack(contexts, dim=1).transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        self.last_telemetry = (
            TrajectoryTelemetry(
                carrier_rms=carrier_rms,
                velocity_rms=velocity_rms,
                control_rms=control_rms,
                conditioning_rms=conditioning_rms,
                attention_entropy=entropy,
            )
            if self._telemetry_level != "none"
            else None
        )
        return attn_output, None


def convert_qwen3_model(
    model,
    tgsa_config: RineyaTGSAConfig,
    *,
    layers: list[int],
    master_seed: int = 20260829,
) -> dict[str, object]:
    """Replace `self_attn` on selected layers, reusing every pretrained tensor.

    Returns a record of exactly which tensors were reused per layer.
    """

    from .convert import derive_layer_seed

    decoder = model.model.layers
    total = len(decoder)
    for index in layers:
        if not 0 <= index < total:
            raise ValueError(f"layer {index} outside the model's {total} layers")

    records = []
    for index in sorted(set(layers)):
        native = decoder[index].self_attn
        torch.manual_seed(derive_layer_seed(master_seed, index))
        replacement = Qwen3TGSAAttention(model.config, index, tgsa_config)
        replacement = replacement.to(dtype=next(native.parameters()).dtype)
        reused = replacement.load_native_(native)
        if not replacement.conditioning_is_neutral():
            raise RuntimeError(f"layer {index}: conditioning is not neutral after conversion")
        decoder[index].self_attn = replacement
        records.append(
            {
                "layer_index": index,
                "reused_tensors": reused,
                "new_tensor_count": len(replacement.trajectory_parameter_names()),
                "conditioning_is_neutral": True,
                "effective_conditioned_heads": replacement.effective_conditioned_heads,
            }
        )
    return {"converted_layers": sorted(set(layers)), "total_layers": total, "layers": records}
