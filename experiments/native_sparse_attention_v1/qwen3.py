"""Explicit, reversible Qwen3 research wrapper. No model loading or monkeypatches.

Active NSA supports full-prefix calls only, with canonical causal/right-padding
masks. Cached decode, packing, arbitrary attention biases and deployment export
are intentionally unqualified. Baseline mode delegates directly to the original.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .core import NSAConfig, BlockCompressor, check_valid, sparse_attention


def rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def valid_from_mask(mask: Tensor | None, batch: int, length: int,
                    device: torch.device, chunk: int = 16) -> Tensor:
    """Reject noncanonical masks rather than silently discarding their meaning."""
    if mask is None:
        return torch.ones(batch, length, dtype=torch.bool, device=device)
    if mask.device != device:
        raise ValueError("mask is on the wrong device")
    if mask.ndim == 2:
        if mask.shape != (batch, length) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("2D mask must be binary [batch, sequence]")
        valid = mask.bool()
    elif mask.ndim == 4:
        if mask.shape[:3] != (batch, 1, length) or mask.shape[3] not in (length, length + 1):
            raise ValueError("4D mask must be canonical [batch, 1, T, T or T+1]")
        def allowed(part):
            if part.dtype == torch.bool:
                return part
            if not part.is_floating_point():
                raise ValueError("additive mask must be floating point")
            permitted = (part == 0) | (part == torch.finfo(part.dtype).min) | torch.isneginf(part)
            if not torch.all(permitted):
                raise ValueError("attention bias/NaN/noncanonical mask is not supported")
            return part == 0
        valid = allowed(mask[:, 0, -1, :length])
        keys = torch.arange(length, device=device)
        for start in range(0, length, chunk):
            stop = min(length, start + chunk)
            queries = torch.arange(start, stop, device=device)
            expected = (keys[None] <= queries[:, None])[None] & valid[:, None]
            if not torch.equal(allowed(mask[:, 0, start:stop, :length]), expected):
                raise ValueError("only causal attention plus right key-padding is supported")
            if mask.shape[3] > length and allowed(mask[:, 0, start:stop, length:]).any():
                raise ValueError("extra cache mask column must be masked")
    else:
        raise ValueError("only binary 2D or canonical causal 4D masks are supported")
    check_valid(valid, batch, length)
    return valid


def geometry(native: nn.Module) -> tuple[int, int, int, int]:
    """Use projection/config geometry, NEVER hidden_size // query_head_count."""
    if getattr(getattr(native, "config", None), "model_type", None) != "qwen3":
        raise ValueError("this adapter qualifies Qwen3 interfaces only")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not isinstance(getattr(native, name, None), nn.Linear):
            raise ValueError("ordinary, unquantized Qwen3 Linear projections are required")
    if not all(hasattr(native, name) for name in ("q_norm", "k_norm", "head_dim")):
        raise ValueError("native QK normalization/head dimension is missing")
    d = native.head_dim
    if type(d) is not int or d <= 0 or d % 2:
        raise ValueError("positive even head dimension required")
    w = native.q_proj.in_features
    h, g = native.q_proj.out_features // d, native.k_proj.out_features // d
    if d <= 0 or d % 2 or h <= 0 or g <= 0 or h % g:
        raise ValueError("invalid Qwen3 head geometry")
    if (native.q_proj.out_features != h * d or native.k_proj.out_features != g * d
            or native.k_proj.in_features != w or native.v_proj.in_features != w
            or native.v_proj.out_features != g * d or native.o_proj.in_features != h * d
            or native.o_proj.out_features != w):
        raise ValueError("inconsistent Qwen3 projection geometry")
    if getattr(native, "sliding_window", None) is not None:
        raise ValueError("native sliding-window variants need separate qualification")
    if getattr(native, "scaling", d ** -0.5) != d ** -0.5:
        raise ValueError("nonstandard QK scaling needs separate qualification")
    weights = list(native.parameters())
    if any(p.device.type == "meta" or p.device != weights[0].device or p.dtype != weights[0].dtype for p in weights):
        raise ValueError("native attention must be materialized with uniform device/dtype")
    if not weights[0].is_floating_point():
        raise ValueError("floating-point native weights are required")
    return w, h, g, d


class Qwen3NSAAttention(nn.Module):
    def __init__(self, native: nn.Module, config: NSAConfig, *, mode: str = "baseline"):
        super().__init__()
        if isinstance(native, Qwen3NSAAttention):
            raise ValueError("attention is already wrapped")
        w, h, g, d = geometry(native)
        self.native, self.nsa_config = native, config
        self.width, self.heads, self.groups, self.head_dim = w, h, g, d
        self.key_compressor = BlockCompressor(config.block_size, d, config.compression_hidden)
        self.value_compressor = BlockCompressor(config.block_size, d, config.compression_hidden)
        self.gate = nn.Sequential(nn.Linear(w, config.gate_hidden), nn.SiLU(),
                                  nn.Linear(config.gate_hidden, 3 * h))
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([-2.0, 0.0, 0.0]).repeat(h))
        if config.kv_mode == "independent":
            self.k_cmp, self.v_cmp = copy.deepcopy(native.k_proj), copy.deepcopy(native.v_proj)
            self.k_win, self.v_win = copy.deepcopy(native.k_proj), copy.deepcopy(native.v_proj)
            self.norm_cmp, self.norm_win = copy.deepcopy(native.k_norm), copy.deepcopy(native.k_norm)
        self.to(device=native.q_proj.weight.device, dtype=native.q_proj.weight.dtype)
        self.train(native.training)
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        if mode not in {"baseline", "nsa", "local", "compressed_local"}:
            raise ValueError("unknown NSA mode")
        self.mode = mode

    def forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None, **kwargs):
        if self.mode == "baseline":
            return self.native(hidden_states=hidden_states, position_embeddings=position_embeddings,
                               attention_mask=attention_mask, **kwargs)
        # Fail before projection or cache mutation. Never treat full-prefix replay as cached decode.
        allowed_kwargs = {"past_key_value", "past_key_values", "cache_position", "position_ids",
                          "use_cache", "output_attentions"}
        if set(kwargs) - allowed_kwargs:
            raise ValueError(f"unqualified active kwargs: {sorted(set(kwargs) - allowed_kwargs)}")
        if kwargs.get("past_key_value") is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("active cached decode is NOT qualified; use full prefixes with use_cache=False")
        if kwargs.get("use_cache", False) or kwargs.get("output_attentions", False):
            raise ValueError("active cache/attention-map output is not implemented")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError("hidden states must have native [batch, sequence, width] shape")
        b, t = hidden_states.shape[:2]
        if not 0 < t <= self.nsa_config.max_reference_tokens:
            raise ValueError("reference token ceiling exceeded or empty input")
        if (torch.is_grad_enabled() and t > self.nsa_config.max_grad_reference_tokens
                and (hidden_states.requires_grad or any(p.requires_grad for p in self.parameters()))):
            raise ValueError("reference autograd ceiling exceeded; use no_grad for prefill qualification")
        expected = torch.arange(t, device=hidden_states.device)
        cp = kwargs.get("cache_position")
        if cp is not None and not torch.equal(cp, expected):
            raise ValueError("offset/cache/packed positions are not qualified")
        pi = kwargs.get("position_ids")
        if pi is not None and (pi.ndim != 2 or pi.shape not in {(1, t), (b, t)}
                               or not torch.equal(pi, expected.expand_as(pi))):
            raise ValueError("position IDs must be contiguous from zero")
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise ValueError("native Qwen3 cosine/sine position embeddings are required")
        cos, sin = position_embeddings
        for x in (cos, sin):
            if x.shape not in {(1, t, self.head_dim), (b, t, self.head_dim)} or x.device != hidden_states.device:
                raise ValueError("unexpected RoPE embedding shape/device")
        cos, sin = cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        valid = valid_from_mask(attention_mask, b, t, hidden_states.device, self.nsa_config.query_chunk_size)
        def project(layer, heads):
            return layer(hidden_states).view(b, t, heads, self.head_dim).transpose(1, 2)
        q = rotary(self.native.q_norm(project(self.native.q_proj, self.heads)), cos, sin)
        raw_k, selected_v = project(self.native.k_proj, self.groups), project(self.native.v_proj, self.groups)
        selected_k = rotary(self.native.k_norm(raw_k), cos, sin)
        if self.nsa_config.kv_mode == "independent":
            ck, cv = project(self.k_cmp, self.groups), project(self.v_cmp, self.groups)
            wk, wv = project(self.k_win, self.groups), project(self.v_win, self.groups)
            cn, wn = self.norm_cmp, self.norm_win
        else:
            ck, cv, wk, wv = raw_k, selected_v, raw_k, selected_v
            cn = wn = self.native.k_norm
        # Compress BEFORE rotary positioning; anchor each summary at its block END.
        ck, cv = self.key_compressor(ck), self.value_compressor(cv)
        ends = torch.arange(ck.shape[2], device=hidden_states.device) * self.nsa_config.block_size + self.nsa_config.block_size - 1
        ck = rotary(cn(ck), cos[:, ends], sin[:, ends])
        wk = rotary(wn(wk), cos, sin)
        gates = self.gate(hidden_states).view(b, t, self.heads, 3).transpose(1, 2).sigmoid()
        dropout = float(getattr(self.native, "attention_dropout", 0.0)) if self.training else 0.0
        out, _ = sparse_attention(q, selected_k, selected_v, wk, wv, ck, cv,
                                  gates, valid, self.nsa_config, mode=self.mode, dropout=dropout)
        result = self.native.o_proj(out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))
        return result * valid[..., None], None


def convert_qwen3(model: nn.Module, config: NSAConfig, *, layers, mode: str = "baseline") -> dict:
    """Validate and construct ALL replacements before any model-tree mutation."""
    indices = list(layers)
    decoder = model.model.layers
    if not indices or any(type(i) is not int or i < 0 or i >= len(decoder) for i in indices):
        raise ValueError("an explicit nonempty list of valid integer layer indices is required")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate layer indices")
    for i in indices:
        if isinstance(decoder[i].self_attn, Qwen3NSAAttention):
            raise ValueError("selected attention is already converted")
        geometry(decoder[i].self_attn)
    replacements = [Qwen3NSAAttention(decoder[i].self_attn, config, mode=mode) for i in indices]
    added = sum(sum(p.numel() for p in r.parameters()) - sum(p.numel() for p in r.native.parameters())
                for r in replacements)
    for i, replacement in zip(indices, replacements, strict=True):
        decoder[i].self_attn = replacement
    return {"layers": indices, "added_parameters": added, "mode": mode,
            "native_checkpoint_qualified": False, "cached_nsa_qualified": False}


def restore_qwen3(model: nn.Module, *, layers) -> None:
    indices = list(layers)
    decoder = model.model.layers
    if any(type(i) is not int or i < 0 or i >= len(decoder)
           or not isinstance(decoder[i].self_attn, Qwen3NSAAttention) for i in indices):
        raise ValueError("restore requires valid wrapped layer indices")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate layer indices")
    for i in indices:
        decoder[i].self_attn = decoder[i].self_attn.native
