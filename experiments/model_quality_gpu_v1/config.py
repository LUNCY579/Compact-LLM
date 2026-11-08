"""Frozen objective/rate screen. No architecture, optimizer-family or data sweeps."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math


ARMS = {
    f"{objective}_{name}": (objective, rate)
    for objective in ("whole", "response")
    for name, rate in (("3e-6", 3e-6), ("1e-5", 1e-5), ("3e-5", 3e-5))
}


@dataclass(frozen=True)
class Design:
    processed_budget: int = 25000
    min_budget_fraction: float = 0.90
    max_sequence: int = 2048
    update_tokens: int = 2048
    max_dev_rows: int = 64
    selection_seed: int = 17
    order_seed: int = 17
    trainable_tail: int = 20
    ce_chunk: int = 64
    warmup_fraction: float = 0.20
    decay_fraction: float = 0.20
    minimum_lr_fraction: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    checkpoint_every: int = 4
    timeout_seconds: int = 3600

    def __post_init__(self):
        for name in ("processed_budget", "max_sequence", "update_tokens", "max_dev_rows",
                     "trainable_tail", "ce_chunk", "checkpoint_every", "timeout_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("selection_seed", "order_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("nonnegative integer seeds required")
        for name in ("min_budget_fraction", "warmup_fraction", "decay_fraction",
                     "minimum_lr_fraction", "beta1", "beta2", "epsilon", "weight_decay", "clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"non-finite or invalid {name}")
        if not 0 < self.min_budget_fraction <= 1:
            raise ValueError("invalid budget tolerance")
        if not (0 < self.warmup_fraction < 1 and 0 < self.decay_fraction < 1
                and self.warmup_fraction + self.decay_fraction < 1):
            raise ValueError("WSD must contain warmup, stable and decay phases")
        if not 0 < self.minimum_lr_fraction <= 1:
            raise ValueError("invalid final rate")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1
                and self.epsilon > 0 and self.weight_decay >= 0 and self.clip_norm > 0):
            raise ValueError("invalid optimizer configuration")
        if self.max_sequence < 3 or self.max_sequence > self.update_tokens:
            raise ValueError("one full sequence must fit an update group")

    def as_dict(self) -> dict:
        return asdict(self)


def validate_native_design(design: Design) -> None:
    # Subsequent budgets/recipes require a new registered successor, not a CLI override.
    canonical = Design(selection_seed=design.selection_seed, order_seed=design.order_seed)
    if design != canonical:
        raise ValueError("native v1 supports only the frozen 25k-token six-cell design")


def rate_at(peak: float, completed_tokens: int, total_tokens: int, design: Design) -> float:
    if not math.isfinite(peak) or peak <= 0 or not 0 < completed_tokens <= total_tokens:
        raise ValueError("invalid learning-rate position")
    progress = completed_tokens / total_tokens
    if progress < design.warmup_fraction:
        return peak * progress / design.warmup_fraction
    start = 1.0 - design.decay_fraction
    if progress <= start:
        return peak
    phase = min(1.0, (progress - start) / design.decay_fraction)
    return peak * (design.minimum_lr_fraction
                   + (1.0 - design.minimum_lr_fraction) * (1.0 - math.sqrt(phase)))
