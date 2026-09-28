"""Explicit Qwen3-shaped actor; one traversal for rollout AND policy gradients.

Only full attention, unscaled native RoPE, one unpadded sequence and FP32/FP64.
This is a reference integration, not a qualified pretrained loader/serving engine.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from .attention import CausalRoLA, PrefixState


def finite(x: Tensor) -> None:
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError('finite FP32/FP64 required')


def rotary(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    a, b = x.chunk(2, -1)
    angle = positions.to(x.dtype)[:, None] * inv_freq.to(x.dtype)[None, :]
    return torch.cat((a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()), -1)


@dataclass(frozen=True)
class ActorCache:
    owner: int
    version: int
    parameter_versions: tuple[int, ...]
    sequence_id: str
    next_position: int
    kv: tuple[tuple[Tensor, Tensor, Tensor], ...]
    rola: tuple[tuple[int, PrefixState], ...]


class Actor(nn.Module):
    def __init__(self, backbone: nn.Module, *, sites: tuple[int, ...] = (), rank: int = 16):
        super().__init__()
        fixture = getattr(backbone, '_rng_fixture', False)
        c = backbone.config
        if not fixture and type(backbone).__name__ != 'Qwen3ForCausalLM':
            raise ValueError('stock Qwen3 or random fixture required')
        if fixture and sum(p.numel() for p in backbone.parameters()) >= 1_000_000:
            raise ValueError('fixture marker cannot admit pretrained-scale weights')
        if getattr(c, 'auto_map', None) or getattr(backbone, 'hf_device_map', None) or hasattr(backbone, 'peft_config'):
            raise ValueError('remote-code, dispatched and PEFT backbones excluded')
        if c.model_type != 'qwen3' or getattr(c, 'rope_scaling', None) is not None or getattr(c, 'use_sliding_window', False) or c.attention_dropout != 0:
            raise ValueError('only unscaled, full-attention, dropout-free Qwen3 supported')
        if getattr(c, 'layer_types', None) and any(x != 'full_attention' for x in c.layer_types):
            raise ValueError('hybrid layer types excluded')
        if any(m._forward_hooks or m._forward_pre_hooks for m in backbone.modules()):
            raise ValueError('existing hooks would be bypassed by this traversal')
        if len({p.device for p in backbone.parameters()}) != 1 or len({p.dtype for p in backbone.parameters()}) != 1:
            raise ValueError('one device/dtype required')
        for p in backbone.parameters():
            finite(p)
        h, kv, d, w = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
        if h % kv or d % 2 or len(backbone.model.layers) != c.num_hidden_layers:
            raise ValueError('invalid head/layer structure')
        if len(set(sites)) != len(sites) or any(type(i) is not int or not 0 <= i < c.num_hidden_layers for i in sites):
            raise ValueError('invalid/duplicate RoLA sites')
        for layer in backbone.model.layers:
            a = layer.self_attn
            for name, shape in (('q_proj', (h*d,w)), ('k_proj',(kv*d,w)), ('v_proj',(kv*d,w)), ('o_proj',(w,h*d))):
                projection = getattr(a, name)
                if type(projection) is not nn.Linear or tuple(projection.weight.shape) != shape:
                    raise ValueError('unsupported projection: ' + name)
            if any(tuple(getattr(a,n).weight.shape) != (d,) for n in ('q_norm','k_norm')):
                raise ValueError('native QK normalization required')
        inv = backbone.model.rotary_emb.inv_freq
        if inv.shape != (d//2,) or getattr(backbone.model.rotary_emb,'attention_scaling',1.0) != 1.0:
            raise ValueError('original unscaled rotary schedule required')
        self.backbone, self.version = backbone, 0
        weight = backbone.model.embed_tokens.weight
        self.rola = nn.ModuleDict({str(i):CausalRoLA(h,d,rank,inv).to(weight) for i in sites})

    def parameter_versions(self) -> tuple[int, ...]:
        return tuple(x._version for x in (*self.parameters(), *self.buffers()))

    def select_tail(self, count: int) -> tuple[str, ...]:
        if type(count) is not int or not 1 <= count <= len(self.backbone.model.layers):
            raise ValueError('invalid trainable tail')
        self.requires_grad_(False)
        for layer in self.backbone.model.layers[-count:]:
            layer.requires_grad_(True)
        self.rola.requires_grad_(True)
        return tuple(n for n,p in self.named_parameters() if p.requires_grad)

    def hidden(self, ids: Tensor, *, cache: ActorCache | None = None,
               use_cache: bool = False, origin: int = 0, sequence_id: str = 'uncached'):
        c, body = self.backbone.config, self.backbone.model
        if ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
            raise ValueError('one nonempty, unpadded int64 sequence required')
        if ids.device != body.embed_tokens.weight.device or bool(((ids < 0) | (ids >= c.vocab_size)).any()):
            raise ValueError('invalid token IDs/device')
        if type(origin) is not int or origin < 0 or not isinstance(sequence_id,str) or not sequence_id:
            raise ValueError('valid origin and sequence identity required')
        if cache is not None and not use_cache:
            raise ValueError('cache passed to uncached forward')
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError('cache is eval/no_grad only')
        if cache is not None:
            if origin != 0 or cache.owner != id(self) or cache.version != self.version or cache.parameter_versions != self.parameter_versions() or cache.sequence_id != sequence_id:
                raise ValueError('foreign or stale actor cache')
            if len(cache.kv) != c.num_hidden_layers or {i for i,_ in cache.rola} != {int(i) for i in self.rola}:
                raise ValueError('cache layer inventory differs')
        start = origin if cache is None else cache.next_position
        n = ids.shape[1]
        if start + n > c.max_position_embeddings:
            raise ValueError('original context bound exceeded')
        pos = torch.arange(start,start+n,device=ids.device)
        x = body.embed_tokens(ids)
        inv = body.rotary_emb.inv_freq
        h, kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        old_states = {} if cache is None else dict(cache.rola)
        next_states, next_kv = [], []
        for i,layer in enumerate(body.layers):
            z = layer.input_layernorm(x); a = layer.self_attn
            q = a.q_norm(a.q_proj(z).reshape(1,n,h,d)).transpose(1,2)
            k = a.k_norm(a.k_proj(z).reshape(1,n,kv,d)).transpose(1,2)
            v = a.v_proj(z).reshape(1,n,kv,d).transpose(1,2)
            qr, kr, values, keypos = rotary(q,pos,inv), rotary(k,pos,inv), v, pos
            if cache is not None:
                ok,ov,op = cache.kv[i]
                if ok.shape != ov.shape or ok.ndim != 4 or ok.shape[:2] != (1,kv) or ok.shape[-1] != d or op.shape != (ok.shape[2],) or op.dtype != torch.int64 or op.device != ids.device:
                    raise ValueError('malformed cached KV')
                if not op.numel() or int(op[0]) < 0 or int(op[-1])+1 != start or bool((op[1:]-op[:-1] != 1).any()):
                    raise ValueError('invalid cached positions')
                for old in (ok,ov):
                    finite(old)
                    if old.dtype != x.dtype or old.device != x.device:
                        raise ValueError('invalid cached KV dtype/device')
                kr, values, keypos = torch.cat((ok,kr),2), torch.cat((ov,v),2), torch.cat((op,pos))
            scores = qr @ kr.repeat_interleave(h//kv,1).transpose(-1,-2)
            scores = scores * getattr(a,'scaling',d**-.5)
            weights = scores.masked_fill(~(keypos[None,:] <= pos[:,None]),-torch.inf).softmax(-1)
            dense = weights @ values.repeat_interleave(h//kv,1)
            if str(i) in self.rola:
                dense,state = self.rola[str(i)](q,k,v,dense,pos[None,:],torch.ones(1,n,dtype=torch.bool,device=ids.device),sequence_id=sequence_id,state=old_states.get(i))
                next_states.append((i,state))
            x = x + a.o_proj(dense.transpose(1,2).reshape(1,n,h*d))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if use_cache:
                next_kv.append((kr.detach().clone(),values.detach().clone(),keypos.clone()))
        x = body.norm(x); finite(x)
        result = None if not use_cache else ActorCache(id(self),self.version,self.parameter_versions(),sequence_id,start+n,tuple(next_kv),tuple(next_states))
        return x,result

    def forward(self, ids, *, last_only=False, **kwargs):
        hidden, cache = self.hidden(ids, **kwargs)
        logits = self.backbone.lm_head(hidden[:,-1:] if last_only else hidden)
        finite(logits)
        return logits,cache

    def completion_logps(self, prompt: tuple[int,...], tokens: tuple[int,...], *, chunk: int = 32) -> Tensor:
        if not isinstance(prompt,tuple) or not isinstance(tokens,tuple) or not prompt or not tokens or type(chunk) is not int or not 1 <= chunk <= 256:
            raise ValueError('nonempty completion/prompt and positive chunk required')
        if any(type(t) is not int or not 0 <= t < self.backbone.config.vocab_size for t in prompt+tokens):
            raise ValueError('invalid prompt/completion tokens')
        device = self.backbone.model.embed_tokens.weight.device
        ids = torch.tensor([prompt + tokens[:-1]],dtype=torch.int64,device=device)
        hidden,_ = self.hidden(ids)
        selected = hidden[0,len(prompt)-1:]
        target = torch.tensor(tokens,dtype=torch.int64,device=device)
        out = []
        for i in range(0,len(tokens),chunk):
            logits = self.backbone.lm_head(selected[i:i+chunk]); finite(logits)
            out.append(logits.log_softmax(-1).gather(-1,target[i:i+chunk,None]).squeeze(-1))
        return torch.cat(out)
