"""A tiny torch model with Qwen3-style MLP parameter names, and a 4-skill synthetic problem.

Used by the mask-safety and synthetic-arm tests and by scripts/synthetic_qualification.py.
Each skill s is a regression target y = tanh(W_s x); rows carry fake token ids only so the
Titan update grouping (``group_updates``) and the supervised-token accounting are exercised.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from clare.allocation import SkillData
from clare.footprint import Footprint
from clare.recipe import Recipe, group_updates
from clare.training import DeltaHost

SKILLS = ("R", "U", "S", "P")


class Block(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x)))


class Tiny(nn.Module):
    def __init__(self, hidden: int = 6, inter: int = 12, layers: int = 2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(hidden, inter) for _ in range(layers)])

    def forward(self, x):
        for b in self.model.layers:
            x = b(x)
        return x


def problem(seed: int = 0, hidden: int = 6, inter: int = 12, layers: int = 2, rows: int = 24,
            tokens: int = 10, group_tokens: int = 40):
    torch.manual_seed(seed)
    model = Tiny(hidden, inter, layers)
    fp = Footprint([(n, tuple(p.shape)) for n, p in model.named_parameters()])
    host = DeltaHost(model, fp)
    g = torch.Generator().manual_seed(seed + 1)
    data_rows, groups = {}, {}
    for s in SKILLS:
        W = torch.randn(hidden, hidden, generator=g)
        rr = []
        for _ in range(rows):
            x = torch.randn(hidden, generator=g)
            rr.append({"x": x, "y": torch.tanh(W @ x), "ids": list(range(tokens)), "supervised_tokens": hidden})
        data_rows[s] = rr
        groups[s] = group_updates(rr, group_tokens)

    def loss_fn(row):
        out = model(row["x"])
        return ((out - row["y"]) ** 2).sum(), hidden

    return model, fp, host, SkillData(data_rows, groups), loss_fn


RECIPE = Recipe(peak_lr=3e-2)
NO_CLIP = Recipe(peak_lr=3e-2, clip_norm=1e9)
