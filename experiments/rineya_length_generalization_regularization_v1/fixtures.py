"""Tiny CPU-only Qwen-shaped fixtures. Not pretrained Rineya and not a benchmark."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F


class ByteTokenizer:
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode("utf-8")]
    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 3 for i in ids if 3 <= i <= 258).decode("utf-8", errors="replace")


class FixtureRMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = 1e-6
    def forward(self, x):
        work = x.float()
        return self.weight * (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.variance_epsilon)).to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.q_norm, self.k_norm = FixtureRMSNorm(width), FixtureRMSNorm(width)
    def forward(self, x):
        q, k, v = self.q_norm(self.q_proj(x)), self.k_norm(self.k_proj(x)), self.v_proj(x)
        attention = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True)
        return self.o_proj(attention.squeeze(1))


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_layernorm = FixtureRMSNorm(width)
        self.post_attention_layernorm = FixtureRMSNorm(width)
        self.self_attn = _Attention(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyLM(nn.Module):
    def __init__(self, layers=2, width=16, vocabulary=259):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocabulary, width)
        self.model.layers = nn.ModuleList([_Block(width) for _ in range(layers)])
        self.model.norm = FixtureRMSNorm(width)
        self.lm_head = nn.Linear(width, vocabulary, bias=False)
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))
    def generate(self, input_ids, *, max_new_tokens, **kwargs):
        result = input_ids
        for _ in range(max_new_tokens):
            result = torch.cat((result, self(result).logits[:, -1].argmax(-1, keepdim=True)), -1)
        return result
