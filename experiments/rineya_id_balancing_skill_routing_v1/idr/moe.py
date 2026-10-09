"""Sparse Top-2+ LoRA on an unchanged dense projection, with causal load updates.

Four experts are the pilot default. This is not a complete conversion to MoE.
The controller remains a Python/NumPy sidecar; use one process/device initially.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .control import CountController, ControlConfig, digest


@dataclass(frozen=True)
class MoEConfig:
    experts: int = 4
    top_k: int = 2
    rank: int = 4
    alpha: float = 4.0
    auxiliary_weight: float = 0.0

    def __post_init__(self):
        if (type(self.experts) is not int or type(self.top_k) is not int
                or type(self.rank) is not int or self.rank < 1
                or not 2 <= self.top_k <= self.experts):
            raise ValueError("require rank >= 1 and 2 <= K <= E (no Top-1 gradient ambiguity)")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be finite and nonnegative")


class SparseLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config=MoEConfig(), control=ControlConfig(),
                 identity="", strengths=None):
        super().__init__()
        if type(base) is not nn.Linear or base.weight.dtype != torch.float32:
            raise ValueError("only ordinary FP32 nn.Linear projections are supported")
        if config.auxiliary_weight and control.method != "none":
            raise ValueError("auxiliary and bias control must be separate comparison arms")
        self.base, self.config = base, config
        self.base.requires_grad_(False)
        e, r = config.experts, config.rank
        self.A = nn.Parameter(base.weight.new_empty(e, r, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(e, base.out_features, r))
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.router = nn.Linear(base.in_features, e, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        strengths = [1.0] * e if strengths is None else strengths
        strength = torch.as_tensor(strengths, device=base.weight.device, dtype=torch.float32)
        if strength.shape != (e,) or not torch.isfinite(strength).all():
            raise ValueError("invalid fixed composition strengths")
        self.register_buffer("strengths", strength)
        binding = digest({"identity": identity, "config": asdict(config),
                          "strengths": strength.tolist(),
                          "shape": list(base.weight.shape)})
        self.controller = CountController([f"expert_{i}" for i in range(e)], control, binding)
        self.token_mask = None
        self._recording = False
        self.last_auxiliary = None
        self.last_routes = None

    def begin_step(self):
        self.controller.begin("train")
        self._recording = True

    def end_step(self, success=True):
        self._recording = False
        return self.controller.commit() if success else self.controller.abort()

    def forward(self, x):
        if self.training and not self._recording:
            raise RuntimeError("begin_step before training; end_step after successful optimizer.step")
        if self.token_mask is None:
            valid = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
        else:
            valid = self.token_mask.to(device=x.device)
            if valid.dtype != torch.bool or tuple(valid.shape) != tuple(x.shape[:-1]):
                raise ValueError("padding mask does not match projection input")
        y = self.base(x)
        flat = x.reshape(-1, x.shape[-1])
        positions = valid.reshape(-1).nonzero(as_tuple=True)[0]
        selected_x = flat.index_select(0, positions)
        if len(positions) == 0:
            self.last_auxiliary = self.A.sum() * 0
            self.last_routes = torch.empty((0, self.config.top_k), device=x.device, dtype=torch.long)
            return y
        logits = self.router(selected_x)
        # Sigmoid original scores, matching the paper's score/bias separation.
        scores = torch.sigmoid(logits).clamp_min(torch.finfo(torch.float32).tiny)
        bias = torch.as_tensor(self.controller.bias.copy(), device=x.device, dtype=scores.dtype)
        indices = torch.argsort(scores.detach() + bias, dim=-1,
                                descending=True, stable=True)[:, :self.config.top_k]
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(dim=1, keepdim=True)
        update = y.new_zeros(len(positions), y.shape[-1])
        # Dispatch only selected tokens; no hidden dense all-expert evaluation.
        for i in range(self.config.experts):
            token_ids, slots = (indices == i).nonzero(as_tuple=True)
            if len(token_ids):
                local_x = selected_x.index_select(0, token_ids)
                delta = F.linear(F.linear(local_x, self.A[i]), self.B[i])
                delta = delta * (self.config.alpha / self.config.rank) * self.strengths[i]
                delta = delta * weights[token_ids, slots, None]
                update = update.index_add(0, token_ids, delta)
        counts = torch.bincount(indices.reshape(-1), minlength=self.config.experts)
        probabilities = torch.softmax(logits, dim=-1).mean(0)
        fractions = counts.to(probabilities.dtype) / (len(positions) * self.config.top_k)
        self.last_auxiliary = (self.config.auxiliary_weight * self.config.experts
                               * (fractions.detach() * probabilities).sum())
        self.last_routes = indices.detach()
        if self.training and self._recording:
            target = len(positions) * self.config.top_k / self.config.experts
            self.controller.observe(counts.detach().cpu().numpy(),
                                    [target] * self.config.experts)
        return y.reshape(-1, y.shape[-1]).index_add(0, positions, update).reshape_as(y)

    def get_extra_state(self):
        return self.controller.state_dict()

    def set_extra_state(self, state):
        self.controller.load_state_dict(state)

    def adapter_state(self):
        return {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                for k, v in self.state_dict().items() if not k.startswith("base.")}

    def load_adapter_state(self, state):
        expected = set(self.adapter_state())
        if set(state) != expected:
            raise ValueError("adapter state has missing/unexpected keys or base weights")
        # Validate identities and shapes BEFORE copying any parameter.
        probe = CountController(self.controller.expert_ids, self.controller.config, self.controller.identity)
        probe.load_state_dict(state["_extra_state"])
        current = self.adapter_state()
        for key, value in state.items():
            if key != "_extra_state" and (not torch.is_tensor(value)
                    or value.shape != current[key].shape or value.dtype != current[key].dtype
                    or not torch.isfinite(value).all()):
                raise ValueError("invalid adapter tensor")
        if not torch.equal(state["strengths"].cpu(), self.strengths.cpu()):
            raise ValueError("composition strengths cannot change on restore")
        self.load_state_dict({**self.state_dict(), **state}, strict=True)


def inject(model, paths, config=MoEConfig(), control=ControlConfig(), identity=""):
    """Atomic validation first. Explicit projection paths only; no regex surgery."""
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("nonempty unique projection paths required")
    if any(a != b and b.startswith(a + ".") for a in paths for b in paths):
        raise ValueError("nested injection paths prohibited")
    modules = [model.get_submodule(p) for p in paths]
    if any(type(m) is not nn.Linear or m.weight.dtype != torch.float32 for m in modules):
        raise ValueError("all targets must be unwrapped dense FP32 nn.Linear")
    wrappers = [SparseLoRALinear(m, config, control, identity + ":" + p)
                for p, m in zip(paths, modules)]
    model.requires_grad_(False)
    for p, wrapper in zip(paths, wrappers):
        parent, _, child = p.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, wrapper)
        wrapper.A.requires_grad_(True)
        wrapper.B.requires_grad_(True)
        wrapper.router.requires_grad_(True)
    return dict(zip(paths, wrappers))
