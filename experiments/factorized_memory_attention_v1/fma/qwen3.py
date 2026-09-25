"""Optional native Qwen3 retrofit for exact-host qualification.

Memory Attention values use the raw Qwen3 key projection before QK
normalization and before RoPE, matching arXiv:2609.28399 and the authors'
reference implementation. Input token IDs are supplied by lightweight model
forward hooks because native attention receives hidden states rather than IDs.
"""
from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .memory import FactorizedMemoryBank

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    Qwen3Attention = object
    apply_rotary_pos_emb = None
    repeat_kv = None
    TRANSFORMERS_AVAILABLE = False


class Qwen3FMAAttention(Qwen3Attention):
    """Drop-in Qwen3 attention whose V path can be switched to factorized MA."""

    def __init__(self, config, layer_idx: int, bank: FactorizedMemoryBank) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers with Qwen3 support is required")
        super().__init__(config, layer_idx)
        object.__setattr__(self, "_fma_bank_ref", weakref.ref(bank))
        self._fma_root_ref = None

    @property
    def fma_bank(self) -> FactorizedMemoryBank:
        bank = self._fma_bank_ref()
        if bank is None:
            raise RuntimeError("FMA memory bank was released")
        return bank

    def bind_root(self, root) -> None:
        self._fma_root_ref = weakref.ref(root)

    def _token_ids(self) -> Tensor:
        if self._fma_root_ref is None or self._fma_root_ref() is None:
            raise RuntimeError("FMA attention is not bound to its root model")
        token_ids = getattr(self._fma_root_ref(), "_fma_current_input_ids", None)
        if token_ids is None:
            raise RuntimeError("Memory Attention requires input_ids for token-indexed lookup")
        return token_ids

    def load_native_(self, native) -> list[str]:
        native_state = native.state_dict()
        own = self.state_dict()
        missing = [key for key in native_state if key not in own]
        if missing:
            raise RuntimeError(f"retrofit missing native tensors: {missing}")
        with torch.no_grad():
            for key, tensor in native_state.items():
                if own[key].shape != tensor.shape:
                    raise ValueError(f"shape mismatch for {key}")
                own[key].copy_(tensor)
        return sorted(native_state)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values=None,
        **kwargs: Any,
    ):
        gate = float(self.fma_bank.gates[self.fma_bank.slot(self.layer_idx)].item())
        if gate == 0.0:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_raw_heads = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_raw_heads = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_pre_rope = key_raw_heads.transpose(1, 2).reshape(*input_shape, -1)

        token_ids = self._token_ids()
        value_flat = self.fma_bank.value(
            hidden_states=hidden_states,
            token_ids=token_ids,
            layer_idx=self.layer_idx,
            key_pre_rope=key_pre_rope,
            standard_value_fn=self.v_proj,
        )
        value = value_flat.view(hidden_shape).transpose(1, 2)

        query = self.q_norm(query_raw_heads)
        key_for_scores = self.k_norm(key_raw_heads)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key_for_scores, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        key_expanded = repeat_kv(key, self.num_key_value_groups)
        value_expanded = repeat_kv(value, self.num_key_value_groups)
        attn_weights = torch.matmul(query, key_expanded.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_expanded.shape[-2]]
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if self.training and self.attention_dropout:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        attn_output = torch.matmul(attn_weights, value_expanded)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


def _input_pre_hook(module, args, kwargs):
    token_ids = kwargs.get("input_ids")
    if token_ids is None and args:
        candidate = args[0]
        if torch.is_tensor(candidate) and candidate.dtype in (torch.int32, torch.int64):
            token_ids = candidate
    module._fma_current_input_ids = token_ids


def _input_post_hook(module, args, kwargs, output):
    module._fma_current_input_ids = None
    return output


def convert_qwen3_model(model, bank: FactorizedMemoryBank, *, layers):
    """Replace selected native attention modules without mutating checkpoint files."""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers with Qwen3 support is required")
    decoder = model.model.layers
    selected = sorted(set(int(index) for index in layers))
    if selected:
        native0 = decoder[selected[0]].self_attn
        bank.to(
            device=next(native0.parameters()).device,
            dtype=next(native0.parameters()).dtype,
        )
    model.fma_memory_bank = bank
    records = []
    for index in selected:
        if index < 0 or index >= len(decoder):
            raise ValueError(f"layer {index} outside model")
        native = decoder[index].self_attn
        replacement = Qwen3FMAAttention(model.config, index, bank)
        replacement = replacement.to(
            device=next(native.parameters()).device,
            dtype=next(native.parameters()).dtype,
        )
        reused = replacement.load_native_(native)
        replacement.bind_root(model)
        decoder[index].self_attn = replacement
        records.append({"layer_index": index, "reused_native_tensors": reused})
    handles = [
        model.register_forward_pre_hook(_input_pre_hook, with_kwargs=True),
        model.register_forward_hook(_input_post_hook, with_kwargs=True),
    ]
    return {"converted_layers": selected, "records": records, "hook_handles": handles}
