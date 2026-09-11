"""Local cross-layer KV sharing: the alternative the CED sweep points to.

The depth sweep shows sharing one encoder-final K/V bank is nearly free across 2 layers
(+0.046 nats, 6/8 pilot hits) and catastrophic across 14 (CE 12.6, worse than uniform).
The damage comes from distance, not from sharing: a layer forced to query a bank built
14 layers earlier is reading a representation it was never trained against.

So share LOCALLY instead. In each group of `group` consecutive layers, the first layer
computes K/V and the rest reuse it. KV residency drops to 1/group of the layers while every
layer keeps its own queries, output projection and MLP, and no layer ever reads a bank from
more than `group-1` layers away.

This is the same family as cross-layer attention designs; here it is applied as a
zero-training conversion of the official weights so the quality cost is measurable directly.

Exploratory. Outside the hashed experiment root; produces no receipt and opens no gate.
"""
from __future__ import annotations
import sys
import torch
from torch import Tensor, nn

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.budget import Config
from experiments.architecture_benchmark_001.model import RMSNorm, SelfBlock, dense_attention, rope


def _block_forward(blk: SelfBlock, x: Tensor, kv_override, pos: Tensor,
                   valid: Tensor, seg: Tensor):
    """SelfBlock.forward with optional externally supplied K/V. Identical arithmetic when
    kv_override is None."""
    b, t, _ = x.shape
    c = blk.c
    n = blk.input_norm(x)
    shape = lambda z, heads: z.view(b, t, heads, c.head_dim).transpose(1, 2)
    q = rope(blk.q_norm(shape(blk.q_proj(n), c.num_attention_heads)), pos, c.rope_theta)
    if kv_override is None:
        k = rope(blk.k_norm(shape(blk.k_proj(n), c.num_key_value_heads)), pos, c.rope_theta)
        v = shape(blk.v_proj(n), c.num_key_value_heads)
    else:
        k, v = kv_override
    a = dense_attention(q, k, v, pos, pos, valid, valid, seg, seg)
    x = x + blk.o_proj(a.transpose(1, 2).reshape(b, t, -1))
    return x + blk.mlp(blk.post_norm(x)), (k, v)


class LocalSharedKVReference(nn.Module):
    """Every `group` consecutive layers share the first layer's K/V. group=1 is the source."""

    def __init__(self, c: Config, group: int) -> None:
        super().__init__()
        if c.architecture != 'plain' or group < 1:
            raise ValueError('local KV sharing wraps a plain donor with group >= 1')
        self.config = c
        self.group = group
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.encoder = nn.ModuleList(SelfBlock(c) for _ in range(c.encoder_layers))
        self.final_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    @torch.no_grad()
    def load_from_plain(self, source) -> dict:
        self.load_state_dict(source.state_dict(), strict=True)
        producers = [i for i in range(self.config.encoder_layers) if i % self.group == 0]
        return {'operation': 'local_cross_layer_kv_sharing', 'group': self.group,
                'kv_producing_layers': producers,
                'kv_residency_ratio': len(producers) / self.config.encoder_layers,
                'function_preserving': self.group == 1,
                'max_bank_distance_layers': self.group - 1}

    def kv_producing_layers(self) -> int:
        return sum(1 for i in range(self.config.encoder_layers) if i % self.group == 0)

    def kv_ratio(self) -> float:
        """KV residency relative to the plain donor.

        `budget()` computes cache bytes as `2 * cache_banks * k * bytes_per_scalar`, so the
        only term that changes here is the number of resident banks. Deriving the ratio
        rather than re-deriving the constants avoids disagreeing with `budget()` about head
        count or scalar width.
        """
        return self.kv_producing_layers() / self.config.encoder_layers

    def forward(self, input_ids: Tensor, *, compute_logits: bool = True) -> Tensor:
        b, t = input_ids.shape
        device = input_ids.device
        valid = torch.ones((b, t), dtype=torch.bool, device=device)
        seg = torch.zeros((b, t), dtype=torch.long, device=device)
        pos = torch.arange(t, device=device).expand(b, t)
        h = self.embed_tokens(input_ids)
        shared = None
        for i, blk in enumerate(self.encoder):
            if i % self.group == 0:
                h, shared = _block_forward(blk, h, None, pos, valid, seg)
            else:
                h, _ = _block_forward(blk, h, shared, pos, valid, seg)
        normalized = self.final_norm(h)
        return torch.nn.functional.linear(normalized, self.embed_tokens.weight) if compute_logits else normalized
