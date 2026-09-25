"""Optional native Qwen3 retrofit for exact-host qualification.

This module is intentionally import-safe when Transformers is absent.  It follows
native Qwen3 Q/K normalization and RoPE, but changes only value construction.
Input token IDs are supplied by lightweight model forward hooks because native
attention receives hidden states, not IDs.
"""
from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .memory import FactorizedMemoryBank

try:  # exact-host path only
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb, repeat_kv
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - ordinary CPU CI environment
    Qwen3Attention = object  # type: ignore[misc,assignment]
    apply_rotary_pos_emb = None  # type: ignore[assignment]
    repeat_kv = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


class Qwen3FMAAttention(Qwen3Attention):  # type: ignore[misc]
    """Drop-in Qwen3 attention whose V path can be switched to factorized MA."""

    def __init__(self, config, layer_idx: int, bank: FactorizedMemoryBank) -> None:
        if not TRANSFORMERS_AVAILABLE:  # pragma: no cover
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
        # weakref avoids registering the whole model as a child module.
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
        # Bank tensors are extra; every native tensor must still exist exactly.
        missing = [k for k in native_state if k not in own]
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
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_pre_rope_heads = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        # [B, KV, T, d_h] -> [B, T, KV*d_h] for factorized MA.
        key_pre_rope = key_pre_rope_heads.transpose(1, 2).reshape(*input_shape, -1)
        token_ids = self._token_ids()
        value_flat = self.fma_bank.value(
            hidden_states=hidden_states,
            token_ids=token_ids,
            layer_idx=self.layer_idx,
            key_pre_rope=key_pre_rope,
            standard_value_fn=self.v_proj,
        )
        value = value_flat.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key_pre_rope_heads, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # Qualification deliberately uses the eager mathematical path.  It leaves
        # the causal mask unchanged and mirrors native Qwen3 GQA expansion.  Fused
        # kernel support is a later performance study, not a qualification shortcut.
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


def convert_qwen3_model(model, bank: FactorizedMemoryBank, *, layers: tuple[int, ...] | list[int]):
    """Replace selected native attention modules without modifying source checkpoint files."""
    if not TRANSFORMERS_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Transformers with Qwen3 support is required")
    decoder = model.model.layers
    selected = sorted(set(int(i) for i in layers))
    if selected:
        native0 = decoder[selected[0]].self_attn
        bank.to(device=next(native0.parameters()).device, dtype=next(native0.parameters()).dtype)
    model.fma_memory_bank = bank
    records = []
    for idx in selected:
        if idx < 0 or idx >= len(decoder):
            raise ValueError(f"layer {idx} outside model")
        native = decoder[idx].self_attn
        replacement = Qwen3FMAAttention(model.config, idx, bank)
        replacement = replacement.to(device=next(native.parameters()).device, dtype=next(native.parameters()).dtype)
        reused = replacement.load_native_(native)
        replacement.bind_root(model)
        decoder[idx].self_attn = replacement
        records.append({"layer_index": idx, "reused_native_tensors": reused})
    handles = [
        model.register_forward_pre_hook(_input_pre_hook, with_kwargs=True),
        model.register_forward_hook(_input_post_hook, with_kwargs=True),
    ]
    return {"converted_layers": selected, "records": records, "hook_handles": handles}
