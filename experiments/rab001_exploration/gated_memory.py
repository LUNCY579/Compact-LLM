"""Function-preserving shared-memory augmentation: the repair for CED's broken initialization.

The delivered `to_ced` REPLACES decoder self-attention with a shared encoder-final K/V read.
That discards the information flow the donor weights were trained for, and the measured
result is a model worse than uniform (CE 12.6 vs the source's 5.5, perplexity 292,267).

This variant ADDS the shared-memory read alongside an otherwise untouched stack, with the
added output projection zero-initialised. At step 0 the branch contributes exactly nothing,
so the model IS the source, bitwise. Quality is then something training can only improve on,
and the KV saving is earned afterwards by raising the gate and pruning self-attention --
rather than assumed at step 0 and paid for in destroyed quality.

Full-forward only: no KV cache path, so generation here is naive O(T^2). That is deliberate,
to keep the variant small enough to trust without reimplementing cache plumbing.
Exploratory, outside the hashed experiment root. Not a receipt; authorizes nothing.
"""
from __future__ import annotations
import sys
import torch
from torch import Tensor, nn

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.budget import Config
from experiments.architecture_benchmark_001.model import (
    RMSNorm, SelfBlock, SharedMemory, dense_attention, rope)


class MemoryReadBranch(nn.Module):
    """Additive cross-attention into the shared bank. `o_proj` starts at exactly zero."""

    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.input_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x: Tensor, memory, qp: Tensor, kp: Tensor,
                qv: Tensor, kv: Tensor, qs: Tensor, ks: Tensor) -> Tensor:
        b, t, _ = x.shape
        c = self.c
        q = self.q_proj(self.input_norm(x)).view(b, t, c.num_attention_heads, c.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q), qp, c.rope_theta)
        a = dense_attention(q, *memory, qp, kp, qv, kv, qs, ks)
        return self.o_proj(a.transpose(1, 2).reshape(b, t, -1))


class GatedMemoryReference(nn.Module):
    """All `total_layers` original blocks intact, plus a zero-initialised shared-memory read
    on every layer from `split` onward. Identical to the plain source at initialisation."""

    def __init__(self, c: Config, split: int) -> None:
        super().__init__()
        if c.architecture != 'plain' or not 0 < split < c.encoder_layers:
            raise ValueError('gated variant wraps a plain donor and needs 0 < split < depth')
        self.config = c
        self.split = split
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.encoder = nn.ModuleList(SelfBlock(c) for _ in range(c.encoder_layers))
        mem_cfg = Config(**{**c.__dict__, 'architecture': 'ced', 'encoder_layers': split,
                            'decoder_layers': c.encoder_layers - split})
        self.memory = SharedMemory(mem_cfg)
        self.reads = nn.ModuleList(MemoryReadBranch(c) for _ in range(c.encoder_layers - split))
        self.final_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    @torch.no_grad()
    def load_from_plain(self, source) -> dict:
        """Copy every original tensor verbatim; seed the branch; keep `o_proj` at zero."""
        src = source.state_dict()
        own = self.state_dict()
        copied, seeded = 0, 0
        for name in own:
            if name.startswith('memory.'):
                tail = name[len('memory.'):]
                tail = 'input_norm.weight' if tail == 'norm.weight' else tail
                own[name].copy_(src[f'encoder.{self.split}.' + tail]); seeded += 1
            elif name.startswith('reads.'):
                _, idx, tail = name.split('.', 2)
                if tail.startswith('o_proj'):
                    own[name].zero_(); seeded += 1
                else:
                    own[name].copy_(src[f'encoder.{self.split + int(idx)}.' + tail]); seeded += 1
            else:
                own[name].copy_(src[name]); copied += 1
        self.load_state_dict(own, strict=True)
        return {'operation': 'zero_gated_shared_memory_augmentation', 'split': self.split,
                'function_preserving': True, 'original_tensors_copied': copied,
                'branch_tensors_seeded': seeded,
                'o_proj_zero_initialised': True,
                'added_parameters': sum(p.numel() for p in self.reads.parameters())
                                    + sum(p.numel() for p in self.memory.parameters())}

    def forward(self, input_ids: Tensor, *, compute_logits: bool = True) -> Tensor:
        b, t = input_ids.shape
        device = input_ids.device
        valid = torch.ones((b, t), dtype=torch.bool, device=device)
        seg = torch.zeros((b, t), dtype=torch.long, device=device)
        pos = torch.arange(t, device=device).expand(b, t)
        h = self.embed_tokens(input_ids)
        memory = None
        for i, block in enumerate(self.encoder):
            h, _ = block(h, pos, pos, valid, valid, seg, seg, None)
            if i + 1 == self.split:
                memory = self.memory(h, pos, None)
            if memory is not None and i + 1 > self.split:
                h = h + self.reads[i - self.split](h, memory, pos, pos, valid, valid, seg, seg)
        normalized = self.final_norm(h)
        return torch.nn.functional.linear(normalized, self.embed_tokens.weight) if compute_logits else normalized
