"""Explicit tiny random-model mechanics fixture; never a Rineya result or teacher."""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch import nn

from .backend import TorchBackend, install_lora
from .contracts import Plan
from .data import DEV_ROLES, Task


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("latin-1")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1] + self.encode(messages[0]["content"]) + [2]


class TinyBackbone(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Embedding(256, width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        den = torch.arange(1, x.shape[1]+1, device=x.device, dtype=x.dtype)[None, :, None]
        x = x.cumsum(1) / den
        return SimpleNamespace(last_hidden_state=torch.tanh(self.q_proj(x) + self.v_proj(x)))


class TinyLM(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.model = TinyBackbone(width)
        self.lm_head = nn.Linear(width, 256, bias=False)


def fixture_plan():
    return Plan(seeds=(17,), learning_rates=(.001, .002), steps=2, batch_tasks=1, replay_tasks=1,
                max_new_tokens=4, max_context_tokens=128, checkpoint_every=1, eval_samples=2,
                rank=2, alpha=4, chunk_tokens=2)


def fixture_tasks(per_role=2):
    return [Task(f"fixture:{role}:{i}", f"fixture-parent:{role}:{i}", "fixture", "fixture-v1", role,
                 f"Fixture {role} case {i}: answer 1.", "1", "1" if role in {"train", "retention_train"} else "",
                 "exact-v1") for role in sorted(DEV_ROLES) for i in range(per_role)]


def fixture_catalog(tasks):
    return {"schema": 1, "provenance": "mechanical_fixture", "tasks": [asdict(t) for t in tasks]}


def fixture_backend(plan=None, seed=17):
    plan = plan or fixture_plan()
    with torch.random.fork_rng():
        torch.manual_seed(413)
        student = TinyLM(5)
        torch.manual_seed(827)
        teacher = TinyLM(7).requires_grad_(False).eval()
    install_lora(student, plan, seed)
    return TorchBackend(student, teacher, ByteTokenizer(), plan, eos_ids=(0,))
