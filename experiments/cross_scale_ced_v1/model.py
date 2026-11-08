"""Native-width two-stream causal hybrid. Read-only caches; no pure endpoint.

The aligned physical token grid is intentional. Different source/target lengths,
prompt-once decoding, eviction, quantization and cross-device sharding are not v1.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from experiments.architecture_benchmark_001.model import (
    DenseReference, KV, RMSNorm, dense_attention, rope,
)
from .config import CrossConfig, cross_budget


class MemoryWriter(nn.Module):
    def __init__(self, c: CrossConfig):
        super().__init__()
        e, d = c.encoder, c.decoder
        self.c = c
        self.norm = RMSNorm(e.hidden_size, e.rms_norm_eps)
        self.k_norm = RMSNorm(d.head_dim, d.rms_norm_eps)
        self.k_proj = nn.Linear(e.hidden_size, d.num_key_value_heads*d.head_dim, False)
        self.v_proj = nn.Linear(e.hidden_size, d.num_key_value_heads*d.head_dim, False)
        self.apply(DenseReference._initialize)

    def forward(self, x: Tensor, pos: Tensor, past: KV | None) -> KV:
        b, t, _ = x.shape
        d = self.c.decoder
        x = self.norm(x)
        def split(z):
            return z.view(b, t, d.num_key_value_heads, d.head_dim).transpose(1, 2)
        # Cross-memory positions use decoder RoPE, not the encoder's internal basis.
        k = rope(self.k_norm(split(self.k_proj(x))), pos, d.rope_theta)
        v = split(self.v_proj(x))
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        return k, v


class MemoryRead(nn.Module):
    def __init__(self, c: CrossConfig):
        super().__init__()
        self.c = c
        d = c.decoder
        self.input_norm = RMSNorm(d.hidden_size, d.rms_norm_eps)
        self.q_norm = RMSNorm(d.head_dim, d.rms_norm_eps)
        self.q_proj = nn.Linear(d.hidden_size, d.num_attention_heads*d.head_dim, False)
        self.o_proj = nn.Linear(d.num_attention_heads*d.head_dim, d.hidden_size, False)
        self.apply(DenseReference._initialize)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x, memory, pos, kp, valid, kvalid, seg, kseg):
        b, t, _ = x.shape
        d = self.c.decoder
        q = self.q_proj(self.input_norm(x)).view(b, t, d.num_attention_heads, d.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q), pos, d.rope_theta)
        if self.c.mode == "H":
            a = dense_attention(q, *memory, pos, kp, valid, kvalid, seg, kseg)
        else:
            group = d.num_attention_heads // d.num_key_value_heads
            k, v = (z[:, :, -t:].repeat_interleave(group, 1) for z in memory)
            acc = torch.float64 if q.dtype == torch.float64 else torch.float32
            gate = ((q.to(acc)*k.to(acc)).sum(-1, keepdim=True)*d.head_dim**-0.5).sigmoid()
            # Same physical token, not a softmax over one key (which kills Q/K grads).
            a = gate.to(v.dtype)*v*valid[:, None, :, None]
        return self.o_proj(a.transpose(1, 2).reshape(b, t, -1))


@dataclass(frozen=True)
class CrossCache:
    owner: int
    signature: tuple
    encoder: tuple[KV, ...]
    decoder: tuple[KV, ...]
    memory: KV
    positions: Tensor
    valid: Tensor
    segments: Tensor


@dataclass
class CrossOutput:
    logits: Tensor | None
    cache: CrossCache | None
    hidden: Tensor | None = None
    diagnostics: tuple[dict, ...] = ()


class CrossScale(nn.Module):
    def __init__(self, c: CrossConfig):
        super().__init__()
        self.config = c
        self.encoder = DenseReference(c.encoder)
        self.decoder = DenseReference(c.decoder)
        self.memory = MemoryWriter(c)
        self.reads = nn.ModuleList(MemoryRead(c) for _ in self.decoder.encoder)
        self.read_enabled = True
        self.gradient_checkpointing = False
        self._initialize_reader_queries()

    def _initialize_reader_queries(self):
        # Own copies; no parameter sharing across backbone and trainable interface.
        for reader, block in zip(self.reads, self.decoder.encoder, strict=True):
            for name in ("input_norm", "q_norm", "q_proj"):
                getattr(reader, name).load_state_dict(getattr(block, name).state_dict())

    @classmethod
    def from_sources(cls, encoder: DenseReference, decoder: DenseReference, mode="H"):
        if type(encoder) is not DenseReference or type(decoder) is not DenseReference:
            raise ValueError("exact independently qualified DenseReference backbones required")
        if torch.is_inference_mode_enabled():
            raise ValueError("construct outside inference_mode")
        c = CrossConfig(encoder.config, decoder.config, mode)
        ep, dp = next(encoder.parameters()), next(decoder.parameters())
        if ep.device != dp.device or ep.dtype != dp.dtype or ep.is_meta:
            raise ValueError("backbones must use one concrete device and dtype")
        with torch.device("meta"):
            result = cls(c)
        result.encoder, result.decoder = copy.deepcopy(encoder), copy.deepcopy(decoder)
        # Initialize only the new modules on the donor device/dtype.
        result.memory = MemoryWriter(c).to(device=ep.device, dtype=ep.dtype)
        result.reads = nn.ModuleList(MemoryRead(c).to(device=ep.device, dtype=ep.dtype)
                                    for _ in result.decoder.encoder)
        result._initialize_reader_queries()
        result.eval()
        return result

    def freeze_backbones(self):
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(("memory.", "reads.")))
        return [p for p in self.parameters() if p.requires_grad]

    def _signature(self):
        return (self.config, self.read_enabled) + tuple(
            (p.data_ptr(), p._version, str(p.dtype), str(p.device)) for p in self.parameters())

    def _validate_cache(self, past, batch, device):
        if type(past) is not CrossCache or past.owner != id(self) or past.signature != self._signature():
            raise ValueError("foreign, stale or wrong-policy cache")
        if past.positions.ndim != 2 or past.positions.shape[0] != batch or not past.positions.shape[1]:
            raise ValueError("invalid cache metadata")
        for x, dtype in ((past.positions, torch.long), (past.valid, torch.bool), (past.segments, torch.long)):
            if x.shape != past.positions.shape or x.dtype != dtype or x.device != device:
                raise ValueError("invalid cache metadata shape/dtype/device")
        limit = min(self.config.encoder.max_position_embeddings, self.config.decoder.max_position_embeddings)
        if (past.positions < 0).any() or (past.positions >= limit).any() or (past.segments < 0).any():
            raise ValueError("cache metadata out of range")
        for pairs, c, n in ((past.encoder, self.config.encoder, len(self.encoder.encoder)),
                            (past.decoder, self.config.decoder, len(self.decoder.encoder)),
                            ((past.memory,), self.config.decoder, 1)):
            if type(pairs) is not tuple or len(pairs) != n:
                raise ValueError("cache bank inventory mismatch")
            for pair in pairs:
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError("cache must contain K/V pairs")
                for x in pair:
                    expected = (batch, c.num_key_value_heads, past.positions.shape[1], c.head_dim)
                    if (x.shape != expected or x.dtype != self.decoder.embed_tokens.weight.dtype
                            or x.device != device or not torch.isfinite(x).all()):
                        raise ValueError("cache tensor shape/dtype/device/finiteness mismatch")

    def forward(self, input_ids: Tensor, *, encoder_input_ids: Tensor | None = None,
                attention_mask=None, position_ids=None, segment_ids=None,
                past: CrossCache | None = None, use_cache=False, last_position_only=False,
                compute_logits=True, return_hidden=False, diagnostics=False) -> CrossOutput:
        for flag in (self.read_enabled, self.gradient_checkpointing, use_cache,
                     last_position_only, compute_logits, return_hidden, diagnostics):
            if type(flag) is not bool:
                raise ValueError("switches must be booleans")
        if any(m.memory_accumulation_dtype is not None or m.gradient_checkpointing
               for m in (self.encoder, self.decoder)):
            raise ValueError("configure only the outer model; arithmetic overrides are forbidden")
        if not compute_logits and not return_hidden:
            raise ValueError("request logits or hidden states")
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError("caches require eval and disabled gradients")
        if past is not None and not use_cache:
            raise ValueError("past requires use_cache")
        if last_position_only and self.training:
            raise ValueError("tail-only logits are inference-only")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or 0 in input_ids.shape:
            raise ValueError("nonempty [batch,length] int64 token IDs required")
        b, t = input_ids.shape
        device = input_ids.device
        eids = input_ids if encoder_input_ids is None else encoder_input_ids
        for ids, c in ((input_ids, self.config.decoder), (eids, self.config.encoder)):
            if ids.shape != input_ids.shape or ids.dtype != torch.long or ids.device != device:
                raise ValueError("v1 requires aligned encoder/decoder physical grids")
            if (ids < 0).any() or (ids >= c.vocab_size).any():
                raise ValueError("token ID outside native vocabulary")
        ep, dp = self.encoder.embed_tokens.weight, self.decoder.embed_tokens.weight
        if ep.device != device or dp.device != device or ep.dtype != dp.dtype:
            raise ValueError("single-device, same-dtype backbones required")
        if attention_mask is not None and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("binary attention mask required")
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        seg = torch.zeros_like(input_ids) if segment_ids is None else segment_ids
        for x in (valid, seg):
            if x.shape != input_ids.shape or x.device != device:
                raise ValueError("mask/segment shape or device mismatch")
        if seg.dtype != torch.long or (seg < 0).any():
            raise ValueError("segments must be nonnegative int64; unique per packed document")
        if past is not None:
            self._validate_cache(past, b, device)
        if position_ids is None:
            start = 0
            if past is not None:
                slots = torch.arange(past.valid.shape[1], device=device).expand_as(past.valid)
                last = slots.masked_fill(~past.valid, -1).max(-1).values
                start = past.positions.gather(1, last.clamp_min(0)[:, None])+1
                start = torch.where(last[:, None] >= 0, start, 0)
            pos = (valid.long().cumsum(-1)-1).clamp_min(0)+start
        else:
            pos = position_ids
        limit = min(self.config.encoder.max_position_embeddings, self.config.decoder.max_position_embeddings)
        if pos.shape != input_ids.shape or pos.dtype != torch.long or pos.device != device:
            raise ValueError("position IDs must match physical token grid")
        if (pos < 0).any() or (pos >= limit).any() or t+(0 if past is None else past.positions.shape[1]) > limit:
            raise ValueError("position/context limit exceeded; no eviction")
        if last_position_only and not bool(valid[:, -1].all()):
            raise ValueError("last physical token must be valid")
        kp = pos if past is None else torch.cat((past.positions, pos), -1)
        kv = valid if past is None else torch.cat((past.valid, valid), -1)
        ks = seg if past is None else torch.cat((past.segments, seg), -1)
        ekv, dkv, stats = [], [], []
        ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        from torch.utils.checkpoint import checkpoint
        e = self.encoder.embed_tokens(eids)
        for i, block in enumerate(self.encoder.encoder):
            if ckpt:
                def apply_e(x, block=block):
                    return block(x, pos, kp, valid, kv, seg, ks, None)[0]
                e = checkpoint(apply_e, e, use_reentrant=False)
            else:
                e, pair = block(e, pos, kp, valid, kv, seg, ks, None if past is None else past.encoder[i])
                if use_cache:
                    ekv.append(pair)
        # Frozen encoder's final norm remains in the path; the new writer norm starts at one.
        memory = self.memory(self.encoder.final_norm(e), pos, None if past is None else past.memory)
        h = self.decoder.embed_tokens(input_ids)
        for i, (block, reader) in enumerate(zip(self.decoder.encoder, self.reads, strict=True)):
            if ckpt:
                def apply_d(x, block=block):
                    return block(x, pos, kp, valid, kv, seg, ks, None)[0]
                h = checkpoint(apply_d, h, use_reentrant=False)
            else:
                h, pair = block(h, pos, kp, valid, kv, seg, ks, None if past is None else past.decoder[i])
                if use_cache:
                    dkv.append(pair)
            if self.read_enabled:
                delta = (checkpoint(reader, h, memory, pos, kp, valid, kv, seg, ks, use_reentrant=False)
                         if ckpt else reader(h, memory, pos, kp, valid, kv, seg, ks))
                if diagnostics:
                    rms = lambda z: float(z.detach().double().square().mean().sqrt())
                    stats.append({"layer": i, "residual_rms": rms(h), "read_rms": rms(delta),
                                  "read_to_residual_rms": rms(delta)/max(rms(h), 1e-30)})
                h = h + delta
        if last_position_only:
            h = h[:, -1:]
        hidden = self.decoder.final_norm(h)
        logits = F.linear(hidden, dp) if compute_logits else None
        cache = None
        if use_cache:
            # Previous tensors are never changed in-place; append creates new banks.
            cache = CrossCache(id(self), self._signature(), tuple(ekv), tuple(dkv), memory,
                               kp.clone(), kv.clone(), ks.clone())
        return CrossOutput(logits, cache, hidden if return_hidden else None, tuple(stats))

    def verify_budget(self) -> dict:
        expected = cross_budget(self.config)
        actual = sum(p.numel() for p in self.parameters())
        if actual != expected["parameters"]:
            raise ValueError(f"unique parameter count {actual} != {expected['parameters']}")
        return expected
