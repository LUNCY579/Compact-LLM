"""CPU reference learned depth prior; not used by the native two-depth bridge.

Adapted REINFORCE + mean-depth penalty + entropy objective. Uses negative
exp(-CE), an EMA baseline, no paper-specific variance rescaling. Not a numerical
reproduction of xLLM-Loop's optimizer or an inference stopping policy.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class DepthPrior(nn.Module):
    def __init__(self, support=(1, 2, 3), target_mean=2.0, entropy_weight=0.01,
                 mean_weight=1.0, seed=17):
        super().__init__()
        if len(support) < 3 or any(type(x) is not int or x < 1 for x in support) or list(support) != sorted(set(support)):
            raise ValueError("at least three distinct positive depths are required")
        if not min(support) < target_mean < max(support):
            raise ValueError("interior target mean required")
        if any(not math.isfinite(v) or v < 0 for v in (entropy_weight, mean_weight)):
            raise ValueError("invalid regularization")
        self.register_buffer("support", torch.tensor(support, dtype=torch.float64))
        self.logits = nn.Parameter(torch.zeros(len(support), dtype=torch.float64))
        self.register_buffer("baseline", torch.zeros((), dtype=torch.float64))
        self.register_buffer("observations", torch.zeros((), dtype=torch.int64))
        self.support_tuple = tuple(support)
        self.target_mean, self.entropy_weight, self.mean_weight = target_mean, entropy_weight, mean_weight
        self.rng = torch.Generator().manual_seed(seed)
        self.pending = None

    def sample(self) -> int:
        if self.pending is not None:
            raise RuntimeError("observe prior draw before sampling again")
        if self.logits.device.type != "cpu" or not torch.isfinite(self.logits).all():
            raise ValueError("finite CPU prior required")
        self.pending = int(torch.multinomial(self.logits.detach().softmax(0), 1, generator=self.rng))
        return int(self.support[self.pending])

    def objective(self, cross_entropy: float) -> torch.Tensor:
        if self.pending is None or not math.isfinite(cross_entropy) or cross_entropy < 0:
            raise ValueError("finite nonnegative CE and pending draw required")
        logp = self.logits.log_softmax(0)
        p = logp.exp()
        cost = -math.exp(-cross_entropy)
        advantage = cost - float(self.baseline)
        result = advantage * logp[self.pending]
        result = result + self.mean_weight * ((p * self.support).sum() - self.target_mean).square()
        result = result + self.entropy_weight * (p * logp).sum()
        with torch.no_grad():
            self.baseline.copy_(torch.tensor(cost, dtype=self.baseline.dtype) if self.observations == 0 else .95*self.baseline+.05*cost)
            self.observations.add_(1)
        self.pending = None
        return result

    def get_extra_state(self):
        if self.pending is not None:
            raise RuntimeError("save only between prior updates")
        return {"rng": self.rng.get_state(), "config": (self.support_tuple, self.target_mean, self.entropy_weight, self.mean_weight)}

    def set_extra_state(self, state):
        if tuple(state["config"]) != (self.support_tuple, self.target_mean, self.entropy_weight, self.mean_weight):
            raise ValueError("prior configuration mismatch")
        self.rng.set_state(state["rng"])
        self.pending = None
