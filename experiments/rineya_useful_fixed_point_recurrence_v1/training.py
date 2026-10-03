"""Budgeted interface-only training mechanics, separately qualified by the caller.

No native training CLI is enabled. This engine is used by the CPU fixture suite;
future native integration must supply an approved loss closure and stage receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Callable
import torch
from torch import nn, Tensor
from .core import InjectionInterface


@dataclass(frozen=True)
class TrainingBudget:
    max_steps: int = 32
    max_seconds: float = 60.0
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    def validate(self):
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 1000:
            raise ValueError("invalid bounded step count")
        if any(not math.isfinite(x) or x <= 0 for x in (self.max_seconds, self.learning_rate, self.grad_clip)):
            raise ValueError("invalid optimizer budget")


def fit_interface(interface: InjectionInterface, frozen_model: nn.Module,
                  loss_closure: Callable[[int], tuple[Tensor, int]], *, budget: TrainingBudget,
                  train_split: str, guard: Callable[[], None] = lambda: None) -> dict:
    """No base parameters in optimizer; explicit split; token/step/time ledger.

    closure(step) returns (mean supervised loss, supervised token count).
    It must be deterministic for matched arms/seeds and not consume held-out data.
    Time/resource checks are cooperative, not OS-level hard preemption.
    """
    budget.validate()
    if train_split != "TRAIN":
        raise ValueError("optimizer may consume TRAIN only")
    if frozen_model.training or any(p.requires_grad for p in frozen_model.parameters()):
        raise ValueError("base must already be eval and frozen")
    if {id(p) for p in interface.parameters()} & {id(p) for p in frozen_model.parameters()}:
        raise ValueError("interface must remain external to canonical model")
    signatures = [(p, p._version) for p in frozen_model.parameters()]
    optimizer = torch.optim.AdamW(interface.parameters(), lr=budget.learning_rate, weight_decay=0.0)
    rows, started, token_count, attempted_tokens, forwards = [], time.perf_counter(), 0, 0, 0
    for step in range(budget.max_steps):
        guard()
        if time.perf_counter() - started >= budget.max_seconds:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, count = loss_closure(step)
        if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
            raise FloatingPointError("loss must be finite scalar with an interface gradient")
        if type(count) is not int or count < 1:
            raise ValueError("positive supervised-token count required")
        forwards += 1
        attempted_tokens += count
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(interface.parameters(), budget.grad_clip, error_if_nonfinite=True)
        if not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in interface.parameters()):
            raise RuntimeError("interface receives no nonzero gradient")
        guard()
        if time.perf_counter() - started >= budget.max_seconds:
            break
        if any(p._version != v or p.grad is not None for p, v in signatures):
            raise RuntimeError("canonical parameter mutation/gradient detected before optimizer")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in interface.parameters()):
            raise FloatingPointError("nonfinite adapter parameter")
        token_count += count
        rows.append({"step": step+1, "loss": float(loss.detach()), "gradient_norm": float(norm),
                     "blend": float(interface.blend.detach().tanh()), "supervised_tokens": count})
    if any(p._version != v or p.grad is not None for p, v in signatures):
        raise RuntimeError("canonical parameter mutation/gradient detected")
    optimizer.zero_grad(set_to_none=True)
    return {"optimizer_steps": len(rows), "supervised_tokens": token_count, "history": rows,
            "forward_evaluations": forwards, "processed_supervised_tokens": attempted_tokens,
            "wall_seconds": time.perf_counter()-started, "base_parameters_optimized": 0,
            "mutation_check": "tensor version counters; not a cryptographic weight audit"}
