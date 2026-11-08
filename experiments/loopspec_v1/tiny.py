"""Actual frozen looped causal Transformer for CPU mechanism tests.

This is randomly initialized test apparatus, NOT a trained Rineya checkpoint.
Includes a unique prelude, shared GQA/RoPE/SwiGLU core, unique coda, and separate
coda caches per readout depth. Cached and full-prefix paths are independent.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

KV = tuple[torch.Tensor, torch.Tensor]


class Block(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.width, self.heads, self.kv_heads = width, 2, 1
        self.dim = width // self.heads
        self.n1, self.n2 = nn.Parameter(torch.ones(width)), nn.Parameter(torch.ones(width))
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, self.dim, bias=False)
        self.v = nn.Linear(width, self.dim, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 2 * width, bias=False)
        self.up = nn.Linear(width, 2 * width, bias=False)
        self.down = nn.Linear(2 * width, width, bias=False)

    @staticmethod
    def norm(x, weight):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight

    def rotate(self, x, positions):
        inv = 10000.0 ** (-torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype) / self.dim)
        angles = positions[:, None].to(x.dtype) * inv[None, :]
        angles = torch.cat([angles, angles], -1)[:, None, :]
        half = self.dim // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * angles.cos() + rotated * angles.sin()

    def qkv(self, h, positions):
        x = self.norm(h, self.n1)
        q = self.rotate(self.q(x).reshape(-1, self.heads, self.dim), positions)
        k = self.rotate(self.k(x).reshape(-1, self.kv_heads, self.dim), positions)
        v = self.v(x).reshape(-1, self.kv_heads, self.dim)
        return q, k, v

    def finish(self, h, context):
        x = h + self.o(context.reshape(-1, self.width))
        n = self.norm(x, self.n2)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))

    def step(self, hidden: torch.Tensor, histories: list[list[KV]]) -> tuple[torch.Tensor, list[KV]]:
        """One real batched shared-block call, including mixed loop depths."""
        positions = torch.tensor([len(h) for h in histories], device=hidden.device)
        q, k, v = self.qkv(hidden, positions)
        batch, length = len(histories), int(positions.max()) + 1
        keys = hidden.new_zeros((batch, self.kv_heads, length, self.dim))
        values = torch.zeros_like(keys)
        mask = torch.arange(length, device=hidden.device)[None, :] <= positions[:, None]
        for i, history in enumerate(histories):
            for j, (key, value) in enumerate(history):
                keys[i, :, j], values[i, :, j] = key, value
            keys[i, :, len(history)], values[i, :, len(history)] = k[i], v[i]
        keys = keys.repeat_interleave(self.heads // self.kv_heads, dim=1)
        values = values.repeat_interleave(self.heads // self.kv_heads, dim=1)
        scores = torch.einsum("bhd,bhld->bhl", q, keys) / math.sqrt(self.dim)
        scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        context = torch.einsum("bhl,bhld->bhd", scores.softmax(-1), values)
        # Clone rows: one surviving child must not retain an entire discarded batch.
        caches = [(k[i].clone(), v[i].clone()) for i in range(batch)]
        return self.finish(hidden, context), caches

    def full(self, hidden: torch.Tensor) -> torch.Tensor:
        """Independent uncached full-sequence causal attention oracle."""
        length = hidden.shape[0]
        q, k, v = self.qkv(hidden, torch.arange(length, device=hidden.device))
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(self.dim)
        mask = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~mask[None, :, :], -torch.inf)
        context = torch.einsum("hts,shd->thd", scores.softmax(-1), v)
        return self.finish(hidden, context)


@dataclass(eq=False)
class State:
    token: int
    parent: State | None
    hidden: torch.Tensor
    position: int
    depth: int = 0
    pre: KV | None = None
    loops: dict[int, KV] = field(default_factory=dict)
    coda: dict[int, KV] = field(default_factory=dict)

    def ancestors(self) -> list[State]:
        items, p = [], self.parent
        while p is not None:
            items.append(p)
            p = p.parent
        return items[::-1]


class TinyLoopBackend(nn.Module):
    """The module stores parameters only. Request caches never persist on it."""
    def __init__(self, seed=0, total_depth=4, vocab_size=17, max_context=128, dtype=torch.float64):
        super().__init__()
        if type(total_depth) is not int or total_depth < 1:
            raise ValueError("invalid recurrent depth")
        self.total_depth, self.vocab_size, self.max_context = total_depth, vocab_size, max_context
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, 16)
            self.prelude, self.core, self.coda_block = Block(), Block(), Block()
            self.final_norm = nn.Parameter(torch.ones(16))
            self.head = nn.Linear(16, vocab_size, bias=False)
        self.to(dtype=dtype).eval().requires_grad_(False)
        self.mixed_depth_batches = 0
        self.max_batch_size = 0

    @torch.inference_mode()
    def start(self, parent: State | None, token: int) -> State:
        position = parent.position + 1 if parent is not None else 0
        if type(token) is not int or not 0 <= token < self.vocab_size or position >= self.max_context:
            raise ValueError("invalid token or context overflow")
        h = self.embedding.weight[token].clone()
        state = State(token, parent, h, position)
        history = [p.pre for p in state.ancestors()]
        h, kv = self.prelude.step(h[None, :], [history])
        state.hidden, state.pre = h[0].clone(), kv[0]
        return state

    @torch.inference_mode()
    def advance(self, states: Sequence[State]):
        if not states:
            raise ValueError("empty recurrence batch")
        if len({id(s) for s in states}) != len(states):
            raise ValueError("duplicate mutable state in recurrence batch")
        if any(s.depth >= self.total_depth for s in states):
            raise ValueError("target recurrent depth already reached")
        self.mixed_depth_batches += int(len({s.depth for s in states}) > 1)
        self.max_batch_size = max(self.max_batch_size, len(states))
        histories = []
        for s in states:
            # A missing same-depth ancestor is a correctness failure, not an
            # invitation to use the nearest available loop's cache.
            histories.append([p.loops[s.depth + 1] for p in s.ancestors()])
        hidden, caches = self.core.step(torch.stack([s.hidden for s in states]), histories)
        for i, s in enumerate(states):
            s.depth += 1
            s.hidden = hidden[i].clone()
            s.loops[s.depth] = caches[i]

    @torch.inference_mode()
    def readout(self, states: Sequence[State]) -> list[list[float]]:
        if not states:
            return []
        if any(s.depth < 1 for s in states):
            raise ValueError("readout requires a completed recurrence")
        histories = [[p.coda[s.depth] for p in s.ancestors()] for s in states]
        h, caches = self.coda_block.step(torch.stack([s.hidden for s in states]), histories)
        logits = self.head(Block.norm(h, self.final_norm))
        for i, s in enumerate(states):
            s.coda[s.depth] = caches[i]
        return logits.cpu().tolist()

    @torch.inference_mode()
    def prefill(self, prompt: tuple[int, ...], readout_depths: tuple[int, ...]):
        if (not prompt or self.total_depth not in readout_depths
                or any(type(r) is not int or not 1 <= r <= self.total_depth for r in readout_depths)):
            raise ValueError("prompt and final-depth readout required")
        state, logits = None, None
        for token in prompt:
            state = self.start(state, token)
            for r in range(1, self.total_depth + 1):
                self.advance([state])
                if r in readout_depths:
                    logits = self.readout([state])[0]
        return state, logits

    @torch.inference_mode()
    def full_logits(self, prefix: Sequence[int]) -> dict[int, list[float]]:
        if not prefix or len(prefix) > self.max_context:
            raise ValueError("invalid oracle prefix")
        h = self.prelude.full(self.embedding(torch.tensor(prefix, device=self.embedding.weight.device)))
        outputs = {}
        for r in range(1, self.total_depth + 1):
            h = self.core.full(h)
            c = self.coda_block.full(h)
            outputs[r] = self.head(Block.norm(c[-1], self.final_norm)).cpu().tolist()
        return outputs

    def cache_bytes(self, states: Sequence[State]) -> int:
        seen, total = set(), 0
        for state in states:
            while state is not None and id(state) not in seen:
                seen.add(id(state))
                caches = [state.pre] + list(state.loops.values()) + list(state.coda.values())
                for kv in caches:
                    if kv is not None:
                        total += sum(x.numel() * x.element_size() for x in kv)
                state = state.parent
        return total

    def weight_digest(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
