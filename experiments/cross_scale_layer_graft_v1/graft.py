"""Frozen residual expert, calibration-only ridge bridges, native layer capture."""
from __future__ import annotations

import gc
import math
from pathlib import Path

import psutil
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from armr.hooks import Reapplication, SafetyError
from armr.metrics import hidden_metrics
from rineya_cross_scale.streaming_forward import StreamingWeights


def normalize(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def finite(x):
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite graft/calibration tensor")
    return x


def memory_guard(limit):
    if psutil.Process().memory_info().rss > limit * 2**30:
        raise SafetyError("graft process RSS budget exceeded")


def fit_ridge(x, y, relative):
    """Train affine map in FP64; no model parameters or held-out tensors used."""
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2 or relative <= 0:
        raise ValueError("invalid ridge inputs")
    x, y = finite(x.double()), finite(y.double())
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc
    penalty = relative * gram.diag().mean().clamp_min(1e-12)
    gram.diagonal().add_(penalty)
    w = torch.linalg.solve(gram, xc.T @ yc)
    b = ym - xm @ w
    return finite(w.float()), finite(b.float())


def alignment(predicted, target):
    finite(predicted)
    finite(target)
    return {
        "token_cosine": float(torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()),
        "relative_mse": float((predicted-target).square().mean() / target.square().mean().clamp_min(1e-12)),
        "predicted_rms": float(predicted.square().mean().sqrt()),
        "target_rms": float(target.square().mean().sqrt()),
    }


def scenario_split(rows, seed, count):
    scenarios = sorted({r["scenario_id"] for r in rows})
    if not 0 < count < len(scenarios):
        raise ValueError("holdout must leave training scenarios")
    order = torch.randperm(len(scenarios), generator=torch.Generator().manual_seed(seed)).tolist()
    holdout = {scenarios[i] for i in order[:count]}
    return [r["scenario_id"] not in holdout for r in rows], sorted(holdout)


class Bridges(nn.Module):
    def __init__(self, up, up_bias, down, down_bias):
        super().__init__()
        for name, value in zip(("up", "up_bias", "down", "down_bias"), (up, up_bias, down, down_bias)):
            self.register_buffer(name, finite(value.detach().clone()))

    def project(self, h):
        return normalize(h) @ self.up + self.up_bias

    def delta(self, dz):
        # Bias cancels: affine(z_after) - affine(z_before) = (z_after-z_before) W.
        return dz @ self.down

    @classmethod
    def fit(cls, small, big, relative):
        return cls(*fit_ridge(normalize(small), big, relative), *fit_ridge(big, small, relative))

    @classmethod
    def random(cls, small, big, seed):
        gen = torch.Generator().manual_seed(seed)
        up = torch.randn(small.shape[1], big.shape[1], generator=gen) / math.sqrt(small.shape[1])
        down = torch.randn(big.shape[1], small.shape[1], generator=gen) / math.sqrt(big.shape[1])
        up *= big.square().mean().sqrt() / (normalize(small) @ up).square().mean().sqrt()
        down *= small.square().mean().sqrt() / (big @ down).square().mean().sqrt()
        return cls(up, torch.zeros(big.shape[1]), down, torch.zeros(small.shape[1]))


def load_layer(weights, config, index):
    """Materialize just one native decoder; avoid allocating the entire donor."""
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(config, index)
    prefix = f"model.layers.{index}."
    state = {k: weights.tensor(prefix + k) for k in layer.state_dict()}
    layer.load_state_dict(state, strict=True, assign=True)
    return layer.eval().requires_grad_(False)


def positions_mask(h):
    n = h.shape[1]
    positions = torch.arange(n, device=h.device).unsqueeze(0)
    mask = torch.full((1, 1, n, n), torch.finfo(h.dtype).min, device=h.device, dtype=h.dtype).triu(1)
    return positions, mask


def capture_native(path: Path, encoded, layers, limit=6, progress=None):
    """Run installed HF layers in order; captures raw post-layer states only."""
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "eager"
    rope = Qwen3RotaryEmbedding(config)
    captured = {}
    with torch.inference_mode(), StreamingWeights(path) as weights:
        states = [weights.rows("model.embed_tokens.weight", ids[0]).unsqueeze(0) for ids in encoded]
        weights.close()
        for index in range(max(layers)+1):
            layer = load_layer(weights, config, index)
            for p, h in enumerate(states):
                pos, mask = positions_mask(h)
                states[p] = finite(layer(h, attention_mask=mask, position_ids=pos,
                                        position_embeddings=rope(h, pos), use_cache=False))
                memory_guard(limit)
            if index in layers:
                captured[index] = [h[0].clone() for h in states]
            del layer
            weights.close()
            gc.collect()
            if progress:
                progress(index, max(layers))
    # Clone outside inference_mode so ridge/training operations may use them.
    return {i: [h.clone() for h in values] for i, values in captured.items()}


class ResidualExpert(nn.Module):
    def __init__(self, layer, config, bridges, gate, identity=False):
        super().__init__()
        if not math.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError("gate must be finite in [0,1]")
        self.layer = layer
        self.rope = Qwen3RotaryEmbedding(config)
        self.bridges, self.gate, self.identity = bridges, gate, identity
        self.last = {}
        self.eval().requires_grad_(False)

    def forward(self, h, *, attention_mask, position_ids):
        if self.gate == 0:
            self.last = {"gate": 0, "expert_bypassed": True}
            return h
        z = finite(self.bridges.project(h))
        if position_ids is None:
            raise SafetyError("explicit original token positions required")
        if attention_mask is None or attention_mask.ndim != 4:
            raise SafetyError("explicit causal 4D mask required")
        zp = z if self.identity else finite(self.layer(
            z, attention_mask=attention_mask, position_ids=position_ids,
            position_embeddings=self.rope(z, position_ids), use_cache=False,
            past_key_values=None))
        dz = finite(zp-z)
        delta = finite(self.bridges.delta(dz))
        out = finite(h + self.gate * delta)
        self.last = {"gate": self.gate, "expert_bypassed": False,
                     "projected_rms": float(z.square().mean().sqrt()),
                     "donor_delta_rms": float(dz.square().mean().sqrt()),
                     "mapped_delta_rms": float(delta.square().mean().sqrt()),
                     "gated_delta_to_input_rms": float(self.gate * delta.square().mean().sqrt() /
                                                        h.square().mean().sqrt().clamp_min(1e-12))}
        return out


class Graft(Reapplication):
    def __init__(self, model, expert, layer=12, **kwargs):
        if not 3 <= layer < min(25, len(model.model.layers)):
            raise ValueError("graft must preserve the stem and stay in the middle")
        if expert.training or any(p.requires_grad for p in expert.parameters()):
            raise SafetyError("expert must be frozen")
        self.expert = expert
        super().__init__(model, layer=layer, applications=1, **kwargs)

    def _forward(self, *args, **kwargs):
        if len(args) > 1:
            raise SafetyError("only hidden_states may be positional at the graft site")
        if kwargs.get("use_cache", False) or any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise SafetyError("decoder received cache")
        normal = finite(self.original(*args, **kwargs))
        self.calls += 1
        out = self.expert(normal, attention_mask=kwargs.get("attention_mask"), position_ids=kwargs.get("position_ids"))
        rec, _ = hidden_metrics(out, normal, normal, None, 2)
        rec.update(self.expert.last)
        self.min_rms_ratio = min(self.min_rms_ratio, rec["rms_ratio_to_normal"])
        self.max_rms_ratio = max(self.max_rms_ratio, rec["rms_ratio_to_normal"])
        if self.capture:
            self.records.append(rec)
        if not self.bounds[0] <= rec["rms_ratio_to_normal"] <= self.bounds[1]:
            raise SafetyError("hidden norm emergency bound")
        return out


def layer_flops(config):
    h, q, kv, inter = config.hidden_size, config.num_attention_heads*config.head_dim, config.num_key_value_heads*config.head_dim, config.intermediate_size
    return 2*(2*h*q + 2*h*kv + 3*h*inter)


def extra_flops(expert):
    if expert.gate == 0:
        return 0
    bridges = 2*(expert.bridges.up.numel()+expert.bridges.down.numel())
    return bridges + (0 if expert.identity else layer_flops(expert.layer.self_attn.config))
