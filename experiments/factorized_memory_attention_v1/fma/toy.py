"""Tiny causal attention/LM used for deterministic mechanical qualification."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from .config import FMAConfig, ModelShape
from .memory import FactorizedMemoryBank


class ToyCausalAttention(nn.Module):
    def __init__(self, shape: ModelShape, bank: FactorizedMemoryBank, layer_idx: int = 0) -> None:
        super().__init__()
        self.shape = shape
        self.bank = bank
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(shape.hidden_size, shape.value_width, bias=False)
        self.k_proj = nn.Linear(shape.hidden_size, shape.value_width, bias=False)
        self.v_proj = nn.Linear(shape.hidden_size, shape.value_width, bias=False)
        self.o_proj = nn.Linear(shape.value_width, shape.hidden_size, bias=False)

    def forward(
        self,
        hidden: Tensor,
        token_ids: Tensor,
        *,
        past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        q = self.q_proj(hidden)
        k_new = self.k_proj(hidden)
        v_new = self.bank.value(
            hidden_states=hidden,
            token_ids=token_ids,
            layer_idx=self.layer_idx,
            key_pre_rope=k_new,
            standard_value_fn=self.v_proj,
        )
        if past is None:
            k_all, v_all = k_new, v_new
            past_len = 0
        else:
            k_all = torch.cat((past[0], k_new), dim=1)
            v_all = torch.cat((past[1], v_new), dim=1)
            past_len = past[0].shape[1]
        score = q @ k_all.transpose(-2, -1) / math.sqrt(self.shape.value_width)
        q_pos = torch.arange(past_len, past_len + hidden.shape[1], device=hidden.device)[:, None]
        k_pos = torch.arange(k_all.shape[1], device=hidden.device)[None, :]
        score = score.masked_fill(k_pos > q_pos, torch.finfo(score.dtype).min)
        probs = torch.softmax(score.float(), dim=-1).to(score.dtype)
        out = self.o_proj(probs @ v_all)
        return out, ((k_all, v_all) if use_cache else None)


class ToyLM(nn.Module):
    def __init__(self, shape: ModelShape, cfg: FMAConfig) -> None:
        super().__init__()
        self.shape = shape
        self.embed = nn.Embedding(shape.vocab_size, shape.hidden_size)
        self.bank = FactorizedMemoryBank(shape, cfg)
        self.attn = ToyCausalAttention(shape, self.bank, layer_idx=cfg.resolved_layers(shape)[0])
        self.norm = nn.LayerNorm(shape.hidden_size)
        self.head = nn.Linear(shape.hidden_size, shape.vocab_size, bias=False)

    def forward(self, ids: Tensor, *, past=None, use_cache=False):
        hidden = self.embed(ids)
        attended, cache = self.attn(hidden, ids, past=past, use_cache=use_cache)
        logits = self.head(self.norm(hidden + attended))
        return logits, cache

    @torch.no_grad()
    def generate(self, ids: Tensor, max_new_tokens: int = 4) -> Tensor:
        past = None
        current = ids
        full = ids
        for step in range(max_new_tokens):
            logits, past = self(current, past=past, use_cache=True)
            current = logits[:, -1].argmax(-1, keepdim=True)
            full = torch.cat((full, current), dim=1)
        return full


def tiny_shape() -> ModelShape:
    return ModelShape(
        vocab_size=41,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=3,
        head_dim=8,
        rms_norm_eps=1e-6,
        model_type="toy",
    )
