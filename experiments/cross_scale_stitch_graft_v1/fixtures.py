"""Random tiny causal fixtures. These are NOT pretrained or native Qwen qualification."""
from __future__ import annotations

import math
import torch
from torch import nn
from .contracts import Identity, digest
from .modules import Decoder, ResidualBlock


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class TinyFFN(nn.Module):
    def __init__(self, width: int, inner: int):
        super().__init__()
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TinyBlock(ResidualBlock):
    def __init__(self, width: int, *, heads: int = 4, kv_heads: int = 2, head_dim: int = 4):
        super().__init__()
        if heads % kv_heads or head_dim % 2:
            raise ValueError("fixture GQA/RoPE dimensions invalid")
        self.width, self.heads, self.kv_heads, self.head_dim = width, heads, kv_heads, head_dim
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.q = nn.Linear(width, heads * head_dim, bias=False)
        self.k = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, width, bias=False)
        self.qnorm, self.knorm = RMSNorm(head_dim), RMSNorm(head_dim)
        self.mlp = TinyFFN(width, 3 * width)

    def attention_residual(self, h, mask, positions):
        b, t, _ = h.shape
        z = self.norm1(h)
        q = self.qnorm(self.q(z).view(b, t, self.heads, self.head_dim)).transpose(1, 2)
        k = self.knorm(self.k(z).view(b, t, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(z).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        inv = 10000 ** (-torch.arange(0, self.head_dim, 2, device=h.device, dtype=h.dtype) / self.head_dim)
        angles = positions.to(h.dtype)[..., None] * inv
        angles = torch.cat((angles, angles), -1)[:, None]
        def rotate(x):
            a, c = x.chunk(2, -1)
            return x * angles.cos() + torch.cat((-c, a), -1) * angles.sin()
        q, k = rotate(q), rotate(k)
        repeats = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + mask, dim=-1)
        attn = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return h + self.o(attn)

    def ffn_delta(self, post_attention_raw):
        return self.mlp(self.norm2(post_attention_raw))


def identity(name: str = "fixture-small") -> Identity:
    return Identity(name, "0" * 40, digest([name, "random-weights"]), digest("fixture-tokenizer"),
                    digest("fixture-template"), digest("fixture-runtime"), "float64", "dense_unmodified", True)


def decoder(width: int = 8, layers: int = 4, seed: int = 11, *, vocab: int = 23,
            heads: int = 4, kv_heads: int = 2, head_dim: int = 4) -> Decoder:
    # Isolate initialization from the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        embedding = nn.Embedding(vocab, width).double().eval().requires_grad_(False)
        blocks = [TinyBlock(width, heads=heads, kv_heads=kv_heads, head_dim=head_dim)
                  .double().eval().requires_grad_(False) for _ in range(layers)]
        norm = RMSNorm(width).double().eval().requires_grad_(False)
        head = nn.Linear(width, vocab, bias=False).double().eval().requires_grad_(False)
    name = f"fixture-w{width}-l{layers}-seed{seed}-v{vocab}-h{heads}-kv{kv_heads}-d{head_dim}"
    return Decoder(identity(name), embedding, blocks, norm, head, max_tokens=64)
