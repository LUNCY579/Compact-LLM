"""Full-width augmentation, NOT a pure CED replacement or a cache-saving model.

Reuses the pinned RAB-001 reference's attention primitives and validation semantics.
M adds causal shared-memory reads; P is an exactly parameter-matched tokenwise
sigmoid-gated control. Every original self-attention and FFN remains intact.
Only the added output projections start at zero: no second zero gate kills learning.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from experiments.architecture_benchmark_001.budget import Config, budget, presets
from experiments.architecture_benchmark_001.model import (
    Cache, DenseReference, Output, RMSNorm, SharedMemory, dense_attention, rope,
)


class AddedRead(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.input_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x, memory, qp, kp, qv, kv, qs, ks, mode):
        b, t, _ = x.shape
        c = self.c
        q = self.q_proj(self.input_norm(x)).view(b, t, c.num_attention_heads, c.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q), qp, c.rope_theta)
        if mode == "M":
            a = dense_attention(q, *memory, qp, kp, qv, kv, qs, ks)
        elif mode == "P":
            # Same parameter tensors, all useful. A one-key softmax would instead
            # give identically zero Q/K gradients and be a misleading control.
            group = c.num_attention_heads // c.num_key_value_heads
            k, v = (z[:, :, -t:].repeat_interleave(group, 1) for z in memory)
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            gate = ((q.to(dtype) * k.to(dtype)).sum(-1, keepdim=True) * c.head_dim ** -0.5).sigmoid()
            a = gate.to(v.dtype) * v * qv[:, None, :, None]
        else:
            raise ValueError("mode must be M (causal memory) or P (tokenwise control)")
        return self.o_proj(a.transpose(1, 2).reshape(b, t, -1))


class FullWidthTransition(DenseReference):
    def __init__(self, c: Config, split: int, mode: str):
        if c.architecture != "plain" or type(split) is not int or not 0 < split < c.total_layers:
            raise ValueError("plain donor and a strictly interior integer split required")
        if mode not in {"M", "P"}:
            raise ValueError("unsupported arm")
        super().__init__(c)
        self.split, self.mode = split, mode
        self.read_enabled = True
        self.memory = SharedMemory(c)
        self.reads = nn.ModuleList(AddedRead(c) for _ in range(c.total_layers - split))

    def _signature(self):
        return (self.split, self.mode, self.read_enabled) + super()._signature()

    @classmethod
    def from_source(cls, source: DenseReference, *, split: int, mode: str):
        if type(source) is not DenseReference or source.config.architecture != "plain":
            raise ValueError("an unaugmented ordinary DenseReference donor is required")
        with torch.device("meta"):
            result = cls(source.config, split, mode)
        src = source.state_dict()
        state = {}
        for name in result.state_dict():
            if name.startswith("memory."):
                tail = name[len("memory."):]
                tail = "input_norm.weight" if tail == "norm.weight" else tail
                value = src[f"encoder.{split}.{tail}"]
            elif name.startswith("reads."):
                _, index, tail = name.split(".", 2)
                value = src[f"encoder.{split + int(index)}.{tail}"]
                if tail == "o_proj.weight":
                    state[name] = torch.zeros_like(value)
                    continue
            else:
                value = src[name]
            state[name] = value.detach().clone()
        result.load_state_dict(state, strict=True, assign=True)
        result.eval()
        if sum(p.numel() for p in result.parameters()) != augmentation_budget(source.config, split)["parameters"]:
            raise ValueError("parameter accounting failed")
        return result

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(("memory.", "reads.")))
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, destination: Path):
        """Create-only research checkpoint; never a native HF export."""
        from safetensors.torch import save_file
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        save_file({k: v.detach().contiguous() for k, v in self.state_dict().items()}, str(destination / "model.safetensors"))
        (destination / "config.json").write_text(json.dumps({
            "schema": 1, "kind": "fwt001_augmentation", "config": asdict(self.config),
            "split": self.split, "mode": self.mode,
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path):
        from safetensors.torch import load_file
        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if set(meta) != {"schema", "kind", "config", "split", "mode"} or meta["schema"] != 1 or meta["kind"] != "fwt001_augmentation":
            raise ValueError("not an FWT-001 research checkpoint")
        with torch.device("meta"):
            model = cls(Config(**meta["config"]), meta["split"], meta["mode"])
        state = load_file(str(directory / "model.safetensors"), device="cpu")
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError("checkpoint tensor inventory mismatch")
        for key, value in state.items():
            if value.shape != expected[key].shape or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"invalid checkpoint tensor: {key}")
        if len({v.dtype for v in state.values()}) != 1:
            raise ValueError("mixed checkpoint tensor dtypes")
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    def forward(self, input_ids: Tensor, *, attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None,
                past: Optional[Cache] = None, use_cache: bool = False,
                last_position_only: bool = False, compute_logits: bool = True,
                return_hidden: bool = False) -> Output:
        if type(self.read_enabled) is not bool or self.memory_accumulation_dtype is not None:
            raise ValueError("undeclared branch switch or arithmetic override")
        if not compute_logits and not return_hidden:
            raise ValueError("compute_logits=False requires return_hidden=True")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty [batch,length] int64")
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError("cache path is inference-only: use eval() and inference_mode()")
        if past is not None and not use_cache:
            raise ValueError("past requires use_cache=True")
        if last_position_only and self.training:
            raise ValueError("tail-only path is not a training objective")
        b, t = input_ids.shape
        device = input_ids.device
        if b == 0 or device != self.embed_tokens.weight.device:
            raise ValueError("input batch/device mismatch")
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError("token IDs outside vocabulary")
        if attention_mask is not None and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must contain only zero and one")
        valid = torch.ones((b, t), dtype=torch.bool, device=device) if attention_mask is None else attention_mask.bool()
        seg = torch.zeros((b, t), dtype=torch.long, device=device) if segment_ids is None else segment_ids
        for label, value in (("attention_mask", valid), ("segment_ids", seg)):
            if value.shape != input_ids.shape or value.device != device:
                raise ValueError(f"{label} shape/device mismatch")
        if seg.dtype != torch.long:
            raise ValueError("segment_ids must be int64; use unique IDs per packed document")
        if past is not None:
            self._validate_cache(past, b, device)
        if position_ids is None:
            start = 0
            if past is not None:
                slots = torch.arange(past.valid.shape[1], device=device).expand_as(past.valid)
                last = slots.masked_fill(~past.valid, -1).max(-1).values
                start = past.positions.gather(1, last.clamp_min(0)[:, None]) + 1
                start = torch.where(last[:, None] >= 0, start, 0)
            pos = (valid.long().cumsum(-1) - 1).clamp_min(0) + start
        else:
            pos = position_ids
        if pos.shape != input_ids.shape or pos.dtype != torch.long or pos.device != device:
            raise ValueError("position_ids must be matching [batch,length] int64")
        if (pos < 0).any() or (pos >= self.config.max_position_embeddings).any():
            raise ValueError("positions outside supported context")
        if t + (0 if past is None else past.positions.shape[1]) > self.config.max_position_embeddings:
            raise ValueError("physical cache/context limit exceeded; eviction is not implemented")
        if last_position_only and not bool(valid[:, -1].all()):
            raise ValueError("tail-only requires a valid last physical token; left-pad batches")
        kp = pos if past is None else torch.cat((past.positions, pos), -1)
        kv = valid if past is None else torch.cat((past.valid, valid), -1)
        ks = seg if past is None else torch.cat((past.segments, seg), -1)
        h = self.embed_tokens(input_ids)
        layer_kv = []
        memory = None
        for i, block in enumerate(self.encoder):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                # Bind the current block: a late-bound loop closure corrupts recomputation.
                def forward_block(value, block=block):
                    return block(value, pos, kp, valid, kv, seg, ks, None)[0]
                h = checkpoint(forward_block, h, use_reentrant=False)
            else:
                h, pair = block(h, pos, kp, valid, kv, seg, ks,
                                None if past is None else past.encoder[i])
                if use_cache:
                    layer_kv.append(pair)
            if i + 1 == self.split:
                memory = self.memory(h, pos, None if past is None else past.memory)
            if i >= self.split and self.read_enabled:
                read = self.reads[i - self.split]
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    delta = checkpoint(read, h, memory, pos, kp, valid, kv, seg, ks,
                                       self.mode, use_reentrant=False)
                else:
                    delta = read(h, memory, pos, kp, valid, kv, seg, ks, self.mode)
                h = h + delta
        if last_position_only:
            # Original upper self-attention still runs for every input token.
            # This saves logit computation only: no pure-CED tail-prefill claim.
            h = h[:, -1:]
        # Direct use of the input embedding guarantees one tied Parameter object.
        normalized = self.final_norm(h)
        logits = F.linear(normalized, self.embed_tokens.weight) if compute_logits else None
        cache = None
        if use_cache:
            cache = Cache(id(self), self._signature(), tuple(layer_kv), memory, kp.clone(), kv.clone(), ks.clone())
        return Output(logits, cache, normalized if return_hidden else None)


def augmentation_budget(c: Config, split: int) -> dict:
    if c.architecture != "plain" or type(split) is not int or not 0 < split < c.total_layers:
        raise ValueError("invalid augmentation budget")
    d, h = c.hidden_size, c.head_dim
    q, k = c.num_attention_heads * h, c.num_key_value_heads * h
    added = (c.total_layers - split) * (2 * d * q + d + h) + 2 * d * k + d + h
    return {"parameters": budget(c)["parameters"] + added, "added_parameters": added,
            "kv_banks": c.total_layers + 1,
            "note": "M and P match unique/trainable parameters, not FLOPs; no KV saving"}


def matrix() -> dict:
    from dataclasses import replace
    base = presets()["B28_1024_official_shape"]
    c28 = replace(base, architecture="ced", encoder_layers=14, decoder_layers=14)
    d40 = replace(base, encoder_layers=40)
    c40 = replace(c28, encoder_layers=20, decoder_layers=20)
    return {"B28": budget(base), "M28": augmentation_budget(base, 14),
            "P28": augmentation_budget(base, 14), "C28_FW_deferred": budget(c28),
            "D40_FW_deferred": budget(d40), "C40_FW_deferred": budget(c40)}
