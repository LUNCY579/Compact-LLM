"""D28-001: two full-width donor stacks, not a reproduction of a named release.

H/P preserve an ordinary decoder path and add memory reads after complete blocks.
C is a separate, destructive serial memory-only endpoint; there is no H -> C switch.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from experiments.architecture_benchmark_001.budget import Config, budget, presets
from experiments.architecture_benchmark_001.model import (
    DenseReference, KV, RMSNorm, SelfBlock, dense_attention, rope,
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

    def forward(self, x, memory, pos, kp, valid, kv, seg, ks, mode):
        b, t, _ = x.shape
        c = self.c
        q = self.q_proj(self.input_norm(x)).view(b, t, c.num_attention_heads, c.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q), pos, c.rope_theta)
        if mode == "H":
            a = dense_attention(q, *memory, pos, kp, valid, kv, seg, ks)
        elif mode == "P":
            # Same-token control remains contextual via the encoder and ordinary D.
            # Sigmoid rather than a one-key softmax keeps Q/K gradients meaningful.
            group = c.num_attention_heads // c.num_key_value_heads
            k, v = (z[:, :, -t:].repeat_interleave(group, 1) for z in memory)
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            gate = ((q.to(dtype) * k.to(dtype)).sum(-1, keepdim=True) * c.head_dim ** -0.5).sigmoid()
            a = gate.to(v.dtype) * v * valid[:, None, :, None]
        else:
            raise ValueError("unknown added-read mode")
        return self.o_proj(a.transpose(1, 2).reshape(b, t, -1))


@dataclass(frozen=True)
class DualCache:
    owner: int
    signature: tuple
    encoder: tuple[KV, ...]
    decoder: tuple[KV, ...]
    memory: KV
    positions: Tensor
    valid: Tensor
    segments: Tensor


@dataclass
class DualOutput:
    logits: Tensor | None
    cache: DualCache | None
    hidden: Tensor | None = None


class DualStack(DenseReference):
    def __init__(self, source_config: Config, mode: str):
        if source_config.architecture != "plain" or mode not in {"H", "P", "C"}:
            raise ValueError("ordinary source and H/P/C mode required")
        target = replace(source_config, architecture="ced", decoder_layers=source_config.encoder_layers)
        super().__init__(target)
        self.source_config, self._mode = source_config, mode
        self.read_enabled = True
        self.reads = nn.ModuleList()
        if mode != "C":
            self.decoder = nn.ModuleList(SelfBlock(target) for _ in range(target.decoder_layers))
            self.reads = nn.ModuleList(AddedRead(target) for _ in range(target.decoder_layers))

    @property
    def mode(self):
        return self._mode

    def _signature(self):
        return (self.mode, self.read_enabled) + super()._signature()

    @classmethod
    def from_source(cls, source: DenseReference, mode: str):
        if type(source) is not DenseReference or source.config.architecture != "plain":
            raise ValueError("exact ordinary DenseReference source required")
        if torch.is_inference_mode_enabled():
            raise ValueError("construct outside inference_mode to retain parameter version counters")
        with torch.device("meta"):
            result = cls(source.config, mode)
        src, state = source.state_dict(), {}
        for name in result.state_dict():
            donor = name
            if name.startswith("decoder."):
                donor = "encoder." + name[len("decoder."):]
            elif name.startswith("reads."):
                _, i, tail = name.split(".", 2)
                donor = f"encoder.{i}.{tail}"
            elif name.startswith("memory."):
                tail = name[len("memory."):]
                donor = "final_norm.weight" if tail == "norm.weight" else "encoder.0." + tail
            value = src[donor]
            state[name] = (torch.zeros_like(value) if name.startswith("reads.") and name.endswith("o_proj.weight")
                           else value.detach().clone())
        result.load_state_dict(state, strict=True, assign=True)
        result.eval()
        if sum(p.numel() for p in result.parameters()) != dual_budget(source.config, mode)["parameters"]:
            raise ValueError("unique-parameter accounting mismatch")
        return result

    def freeze_backbones(self):
        if self.mode == "C":
            raise ValueError("memory-only recovery is not the H/P engineering lane")
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(("memory.", "reads.")))
        return [p for p in self.parameters() if p.requires_grad]

    def _validate_cache(self, past, batch, device):
        if (not isinstance(past, DualCache) or past.owner != id(self)
                or past.signature != self._signature()):
            raise ValueError("foreign or stale dual-stack cache")
        if past.positions.ndim != 2 or past.positions.shape[0] != batch or not past.positions.shape[1]:
            raise ValueError("invalid cache metadata shape")
        for value, dtype in ((past.positions, torch.long), (past.valid, torch.bool), (past.segments, torch.long)):
            if value.shape != past.positions.shape or value.dtype != dtype or value.device != device:
                raise ValueError("cache metadata shape/dtype/device mismatch")
        if (past.positions < 0).any() or (past.positions >= self.config.max_position_embeddings).any():
            raise ValueError("cache position out of range")
        expected_d = 0 if self.mode == "C" else len(self.decoder)
        if len(past.encoder) != len(self.encoder) or len(past.decoder) != expected_d:
            raise ValueError("cache bank inventory mismatch")
        pairs = (*past.encoder, *past.decoder, past.memory)
        shape = (batch, self.config.num_key_value_heads, past.positions.shape[1], self.config.head_dim)
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("cache requires K/V pairs")
            for value in pair:
                if (value.shape != shape or value.dtype != self.embed_tokens.weight.dtype
                        or value.device != device or not torch.isfinite(value).all()):
                    raise ValueError("invalid cache tensor")

    def forward(self, input_ids: Tensor, *, attention_mask=None, position_ids=None,
                segment_ids=None, past: DualCache | None = None, use_cache=False,
                last_position_only=False, compute_logits=True, return_hidden=False) -> DualOutput:
        if type(self.read_enabled) is not bool or self.memory_accumulation_dtype is not None:
            raise ValueError("undeclared switch or arithmetic override")
        if self.mode == "C" and not self.read_enabled:
            raise ValueError("C has no optional memory bypass")
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
        embedded = h
        encoder_kv, decoder_kv = [], []
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        from torch.utils.checkpoint import checkpoint
        for i, block in enumerate(self.encoder):
            if checkpointing:
                def apply_encoder(value, block=block):
                    return block(value, pos, kp, valid, kv, seg, ks, None)[0]
                h = checkpoint(apply_encoder, h, use_reentrant=False)
            else:
                h, pair = block(h, pos, kp, valid, kv, seg, ks, None if past is None else past.encoder[i])
                if use_cache:
                    encoder_kv.append(pair)
        memory = self.memory(h, pos, None if past is None else past.memory)
        # H/P begin with the original token embeddings, not another stack's output.
        # C is explicitly a serial, non-function-preserving encoder -> decoder endpoint.
        if self.mode != "C":
            h = embedded
        for i, block in enumerate(self.decoder):
            if self.mode == "C":
                if checkpointing:
                    h = checkpoint(block, h, memory, pos, kp, valid, kv, seg, ks, use_reentrant=False)
                else:
                    h = block(h, memory, pos, kp, valid, kv, seg, ks)
            else:
                if checkpointing:
                    def apply_decoder(value, block=block):
                        return block(value, pos, kp, valid, kv, seg, ks, None)[0]
                    h = checkpoint(apply_decoder, h, use_reentrant=False)
                else:
                    h, pair = block(h, pos, kp, valid, kv, seg, ks, None if past is None else past.decoder[i])
                    if use_cache:
                        decoder_kv.append(pair)
                if self.read_enabled:
                    args = (h, memory, pos, kp, valid, kv, seg, ks, self.mode)
                    delta = (checkpoint(self.reads[i], *args, use_reentrant=False) if checkpointing
                             else self.reads[i](*args))
                    h = h + delta
        if last_position_only:
            h = h[:, -1:]  # logit-only saving; all blocks still run on the chunk
        normalized = self.final_norm(h)
        logits = F.linear(normalized, self.embed_tokens.weight) if compute_logits else None
        cache = None
        if use_cache:
            cache = DualCache(id(self), self._signature(), tuple(encoder_kv), tuple(decoder_kv),
                              memory, kp.clone(), kv.clone(), ks.clone())
        return DualOutput(logits, cache, normalized if return_hidden else None)

    def save(self, destination: Path):
        from safetensors.torch import save_file
        if not self.read_enabled:
            raise ValueError("diagnostic bypass is not a checkpoint configuration")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        save_file({k: v.detach().contiguous() for k, v in self.state_dict().items()},
                  str(destination / "model.safetensors"))
        meta = {"schema": 1, "kind": "D28-001", "source_config": asdict(self.source_config), "mode": self.mode}
        (destination / "config.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path):
        if torch.is_inference_mode_enabled():
            raise ValueError("load outside inference_mode to retain parameter version counters")
        from safetensors.torch import load_file
        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if set(meta) != {"schema", "kind", "source_config", "mode"} or meta["schema"] != 1 or meta["kind"] != "D28-001":
            raise ValueError("not a D28-001 checkpoint")
        with torch.device("meta"):
            model = cls(Config(**meta["source_config"]), meta["mode"])
        state = load_file(str(directory / "model.safetensors"), device="cpu")
        expected = model.state_dict()
        if set(state) != set(expected) or len({t.dtype for t in state.values()}) != 1:
            raise ValueError("checkpoint tensor inventory/dtype mismatch")
        for key, value in state.items():
            if value.shape != expected[key].shape or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"invalid checkpoint tensor: {key}")
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()


def controls(source: DenseReference, arm: str) -> DenseReference:
    """D56 identity-expanded ordinary control, or C28 split topology diagnostic.

    C28 uses the same final-norm memory convention as C56, not old receipts.
    """
    if type(source) is not DenseReference or source.config.architecture != "plain":
        raise ValueError("ordinary source required")
    if torch.is_inference_mode_enabled():
        raise ValueError("construct controls outside inference_mode")
    c, n = source.config, source.config.total_layers
    if arm == "D56":
        target = replace(c, encoder_layers=2*n)
    elif arm == "C28" and n % 2 == 0:
        target = replace(c, architecture="ced", encoder_layers=n//2, decoder_layers=n//2)
    else:
        raise ValueError("unknown control or odd source depth")
    with torch.device("meta"):
        result = DenseReference(target)
    src, state = source.state_dict(), {}
    for name in result.state_dict():
        donor, zero = name, False
        if name.startswith("encoder."):
            _, i, tail = name.split(".", 2)
            donor = f"encoder.{int(i) % n}.{tail}"
            zero = arm == "D56" and int(i) >= n and tail in {"o_proj.weight", "mlp.down_proj.weight"}
        elif name.startswith("decoder."):
            _, i, tail = name.split(".", 2)
            donor = f"encoder.{n//2 + int(i)}.{tail}"
        elif name.startswith("memory."):
            tail = name[len("memory."):]
            donor = "final_norm.weight" if tail == "norm.weight" else f"encoder.{n//2}." + tail
        value = src[donor]
        state[name] = torch.zeros_like(value) if zero else value.detach().clone()
    result.load_state_dict(state, strict=True, assign=True)
    return result.eval()


def dual_budget(c: Config, mode: str) -> dict:
    if c.architecture != "plain" or mode not in {"H", "P", "C"}:
        raise ValueError("ordinary source and H/P/C required")
    n, d, h = c.total_layers, c.hidden_size, c.head_dim
    q, k = c.num_attention_heads*h, c.num_key_value_heads*h
    if mode == "C":
        r = budget(replace(c, architecture="ced", decoder_layers=n))
        r["engineering_trainable_parameters"] = 0  # not an authorized recovery recipe
    else:
        r = budget(replace(c, encoder_layers=2*n))
        added = 2*d*k + d + h + n*(2*d*q + d + h)
        r["parameters"] += added
        r["kv_banks"] += 1
        r["engineering_trainable_parameters"] = added
        r["kv_bytes_per_token_batch1"] = 2*r["kv_banks"]*k*2
        r["kv_mib_at_8192_tokens_batch1"] = r["kv_bytes_per_token_batch1"]*8192/2**20
        # Other inherited body/bank entries would be misleading for the hybrid.
        r.pop("body_and_final_norm_parameters")
        r.pop("shared_bank_parameters")
    r["weight_gib_fp32"] = r["parameters"]*4/2**30
    r["weight_gib_bf16_analytic_only"] = r["parameters"]*2/2**30
    r["note"] = "H/P exact parameter match, not compute match; KV excludes activations; no measured fit/speed"
    return r


def matrix():
    c = presets()["B28_1024_official_shape"]
    return {"B28": budget(c), "D56": budget(replace(c, encoder_layers=56)),
            "C28": budget(replace(c, architecture="ced", encoder_layers=14, decoder_layers=14)),
            **{m+"56": dual_budget(c, m) for m in ("H", "P", "C")}}
