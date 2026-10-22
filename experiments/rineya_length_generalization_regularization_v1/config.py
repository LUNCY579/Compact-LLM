"""Strict, immutable single-arm plans; no production defaults are modified."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Plan:
    mode: str = "none"
    probability: float = 0.0
    detach_beta: bool = True
    detach_alpha: bool = False
    seed: int = 101
    expected_layers: int = 28
    rank: int = 8
    lora_alpha: float = 16.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    accumulation: int = 8
    train_context: int = 1024
    max_tokens: int = 524288
    max_steps: int = 512
    eval_lengths: tuple = (1024, 2048, 4096, 8192)
    max_new_tokens: int = 32
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    attention: str = "sdpa"

    def __post_init__(self):
        if self.mode not in {"none", "pre_linear", "vpad", "pre_norm_control"}:
            raise ValueError("Unknown intervention")
        for key in ("probability", "lora_alpha", "learning_rate", "weight_decay",
                    "beta1", "beta2", "warmup_fraction", "max_grad_norm"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if not 0 <= self.probability < 1 or (self.mode == "none" and self.probability != 0):
            raise ValueError("Invalid probability; mode=none requires p=0")
        for key in ("expected_layers", "rank", "accumulation", "train_context", "max_tokens",
                    "max_steps", "max_new_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("detach_beta", "detach_alpha", "gradient_checkpointing"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if min(self.learning_rate, self.lora_alpha, self.max_grad_norm) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or adapter setting")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1 and 0 <= self.warmup_fraction < 1):
            raise ValueError("Invalid beta or warmup fraction")
        lengths = tuple(self.eval_lengths)
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError("eval_lengths must be positive integers")
        if tuple(sorted(set(lengths))) != lengths:
            raise ValueError("eval_lengths must be increasing and unique")
        object.__setattr__(self, "eval_lengths", lengths)
        if self.dtype not in {"float32", "bfloat16"} or self.attention not in {"eager", "sdpa"}:
            raise ValueError("Unqualified precision or attention backend")

    def to_dict(self):
        return asdict(self)


def screening_plans(base=None):
    """Only the named activation intervention differs between A/B/C."""
    base = base or Plan()
    return {name: replace(base, mode=mode, probability=p) for name, mode, p in (
        ("A_none", "none", 0.0), ("B_pre_linear", "pre_linear", 0.1), ("C_vpad", "vpad", 0.1))}


def factorial_plans(base, reference_decay):
    if reference_decay <= 0 or not math.isfinite(reference_decay):
        raise ValueError("A positive reference decay must be preregistered")
    if base.mode not in {"pre_linear", "vpad"} or base.probability <= 0:
        raise ValueError("Select a nonzero candidate before the factorial")
    return {f"{mode}_wd{decay:g}": replace(base, mode=mode,
            probability=0 if mode == "none" else base.probability, weight_decay=decay)
            for mode in ("none", base.mode) for decay in (0.0, reference_decay)}
