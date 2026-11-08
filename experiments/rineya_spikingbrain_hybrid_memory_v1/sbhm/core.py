"""SBHM-001: independent correctness-first GQA/local/recurrent attention.

This is a sequential PyTorch reference, NOT a sparse/event-driven speed kernel.
Persistent state is owned by a request, never by a module. No upstream code is
vendored. Static RoPE follows the host-supplied cos/sin, only on the local path.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Memory:
    keys: torch.Tensor
    values: torch.Tensor
    matrix: torch.Tensor | None
    seen: int

    def tensors(self):
        return [x for x in (self.keys, self.values, self.matrix) if x is not None]

    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensors())

    def reset_recurrent(self):
        if self.matrix is not None:
            self.matrix = torch.zeros_like(self.matrix)


def rotate(x, cos, sin):
    if x.shape[-1] % 2:
        raise ValueError('RoPE requires an even head dimension')
    a, b = x.chunk(2, dim=-1)
    return x * cos.unsqueeze(1) + torch.cat((-b, a), -1) * sin.unsqueeze(1)


class LoRALinear(nn.Module):
    """One fixed-rank, zero-output initialization; dense base remains frozen."""
    def __init__(self, base, rank):
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError('invalid LoRA rank')
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0  # alpha == rank, fixed across all trained arms

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


class HybridAttention(nn.Module):
    """Replace ONE Qwen3 attention module; keep GQA, QK norm, source projections.

    Each KV group owns one FP32 D x D state (no MHA expansion). The hybrid
    output is local + tanh(mix) * RMS(recurrent), followed by source o_proj.
    This intra-layer pilot is an adaptation, not a SpikingBrain reproduction.
    Batch=1, unpadded, contiguous append-only sequences are intentionally the
    native contract. Padding, beams, static caches and cache cropping fail closed.
    """
    def __init__(self, source, mode='hybrid', window=128, mix_init=0.1):
        super().__init__()
        c = source.config
        if mode not in ('local', 'hybrid') or window < 1:
            raise ValueError('invalid attention mode/window')
        self.config, self.layer_idx = c, source.layer_idx
        self.head_dim = c.head_dim  # intentionally NO hidden_size // heads fallback
        self.h, self.kv = c.num_attention_heads, c.num_key_value_heads
        self.window, self.mode = window, mode
        d, width = self.head_dim, c.hidden_size
        if self.h % self.kv or d % 2:
            raise ValueError('invalid GQA / RoPE geometry')
        for name, shape in [('q_proj', (self.h*d, width)),
                            ('k_proj', (self.kv*d, width)),
                            ('v_proj', (self.kv*d, width)),
                            ('o_proj', (width, self.h*d))]:
            module = getattr(source, name)
            if tuple(module.weight.shape) != shape:
                raise ValueError(f'{name} projection mismatch: {module.weight.shape} != {shape}')
            setattr(self, name, module)
        self.q_norm, self.k_norm = source.q_norm, source.k_norm
        self.eps = c.rms_norm_eps
        if mode == 'hybrid':
            ref = source.q_proj.weight
            self.forget = nn.Linear(width, self.kv*d, bias=True,
                                    device=ref.device, dtype=ref.dtype)
            nn.init.zeros_(self.forget.weight)
            nn.init.zeros_(self.forget.bias)
            self.mix = nn.Parameter(ref.new_full((self.h,), mix_init))
            self.memory_scale = nn.Parameter(ref.new_ones(d))

    def sequence(self, hidden, cos, sin, state=None):
        b, length, _ = hidden.shape
        if b != 1 or length < 1:
            raise ValueError('only nonempty batch-one sequences are supported')
        d = self.head_dim
        q = self.q_norm(self.q_proj(hidden).view(b, length, self.h, d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).view(b, length, self.kv, d)).transpose(1, 2)
        v = self.v_proj(hidden).view(b, length, self.kv, d).transpose(1, 2)
        qr, kr = rotate(q, cos, sin), rotate(k, cos, sin)
        if state is None:
            old_k, old_v, matrix, seen = kr[:, :, :0], v[:, :, :0], None, 0
        else:
            old_k, old_v, matrix, seen = state.keys, state.values, state.matrix, state.seen
            if (old_k.shape[:2] != (b, self.kv) or old_k.shape[-1] != d or
                    old_k.shape[2] > self.window or old_k.device != hidden.device):
                raise ValueError('incompatible request state')
        if self.mode == 'hybrid':
            if matrix is None:
                matrix = torch.zeros((b, self.kv, d, d), device=hidden.device, dtype=torch.float32)
            if matrix.shape != (b, self.kv, d, d) or matrix.dtype != torch.float32:
                raise ValueError('recurrent state must be GQA-shaped FP32')
            forget = (F.logsigmoid(self.forget(hidden).float()) / 16).exp()
            forget = forget.view(b, length, self.kv, d)
        groups, outputs = self.h // self.kv, []
        for t in range(length):
            old_k = torch.cat((old_k, kr[:, :, t:t+1]), 2)[:, :, -self.window:]
            old_v = torch.cat((old_v, v[:, :, t:t+1]), 2)[:, :, -self.window:]
            # At token t all retained keys are causal, so no triangular mask.
            keys = old_k.repeat_interleave(groups, 1)
            values = old_v.repeat_interleave(groups, 1)
            scores = (qr[:, :, t:t+1].float() @ keys.float().transpose(-1, -2)) / math.sqrt(d)
            local = (scores.softmax(-1).to(values.dtype) @ values).squeeze(2)
            if self.mode == 'hybrid':
                key, val = F.relu(k[:, :, t].float()), v[:, :, t].float()
                matrix = forget[:, t, :, :, None] * matrix + key.unsqueeze(-1) * val.unsqueeze(-2)
                query = F.relu(q[:, :, t].float()).view(b, self.kv, groups, d)
                read = torch.einsum('bkgd,bkde->bkge', query, matrix).reshape(b, self.h, d)
                read = read * torch.rsqrt(read.square().mean(-1, keepdim=True) + self.eps)
                read = read * self.memory_scale.float()
                local = local + (self.mix.float().tanh()[None, :, None] * read).to(local.dtype)
            outputs.append(local)
        output = torch.stack(outputs, 1).reshape(b, length, self.h*d)
        # clone prevents window views from retaining the previous backing storage.
        memory = Memory(old_k.clone(), old_v.clone(), matrix, seen + length)
        return self.o_proj(output), memory

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        if kwargs.get('output_attentions', False):
            raise ValueError('attention-map outputs not supported by this pilot')
        if hidden_states.shape[0] != 1:
            raise ValueError('padding/beam/batched inference is not qualified')
        bank, state = None, None
        if past_key_value is not None:
            if past_key_value.__class__.__name__ != 'DynamicCache':
                raise ValueError('only Transformers 4.51.3 DynamicCache is supported')
            bank = getattr(past_key_value, '_sbhm', None)
            if bank is None:
                bank = {}
                past_key_value._sbhm = bank
            state = bank.get(self.layer_idx)
            if cache_position is None:
                raise ValueError('explicit append-only cache_position required')
            start = 0 if state is None else state.seen
            expected = torch.arange(start, start + hidden_states.shape[1], device=cache_position.device)
            if not torch.equal(cache_position, expected):
                raise ValueError('cache reorder/crop/discontinuous position not supported')
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[0:2] != (1, 1):
                raise ValueError('only standard unpadded causal masks are supported')
            start = 0 if state is None else state.seen
            for i in range(hidden_states.shape[1]):
                if not torch.all(attention_mask[0, 0, i, :start+i+1] == 0):
                    raise ValueError('padding/custom masks are unsupported')
        cos, sin = position_embeddings
        output, memory = self.sequence(hidden_states, cos, sin, state)
        if bank is not None:
            bank[self.layer_idx] = memory
        return output, None


def cache_bytes(cache):
    """Actual allocated tensor-storage bytes, deduplicated; excludes model weights."""
    tensors = []
    for name in ('key_cache', 'value_cache'):
        for value in getattr(cache, name, []):
            if torch.is_tensor(value):
                tensors.append(value)
    for memory in getattr(cache, '_sbhm', {}).values():
        tensors.extend(memory.tensors())
    storages = {}
    for x in tensors:
        storage = x.untyped_storage()
        storages[(str(x.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())
