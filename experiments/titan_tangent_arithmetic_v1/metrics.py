"""Paper Eq. (6)/(7) and complementary output-space diagnostics."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def paired(before: list[bool], after: list[bool]) -> dict:
    if len(before) != len(after) or not before:
        raise ValueError("nonempty matched rows required")
    if any(type(x) is not bool for x in before + after):
        raise ValueError("paired flags must be boolean")
    gains = sum(not a and b for a, b in zip(before, after))
    losses = sum(a and not b for a, b in zip(before, after))
    n = gains + losses
    # Stable exact Binomial(n,.5) two-sided tail, without overflowing 2**n.
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                        - math.lgamma(n - i + 1) - n * math.log(2))
               for i in range(min(gains, losses) + 1)) if n else 0.5
    return {"rows": len(before), "before": sum(before), "after": sum(after),
            "gains": gains, "losses": losses, "net": gains - losses,
            "exact_mcnemar_p": min(1.0, 2 * tail),
            "inference": "descriptive; non-significance is not equivalence"}


def normalized_accuracy(merged: dict[str, float], single: dict[str, float]) -> dict:
    if not merged or set(merged) != set(single):
        raise ValueError("task sets must match")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*merged.values(), *single.values()]):
        raise ValueError("invalid accuracy")
    ratios = {k: (merged[k] / single[k] if single[k] else None) for k in merged}
    return {"absolute_macro_accuracy": sum(merged.values()) / len(merged),
            "normalized_macro_accuracy": (sum(ratios.values()) / len(ratios)
                                           if all(v is not None for v in ratios.values()) else None),
            "per_task_ratios": ratios,
            "zero_single_task_denominator": any(v is None for v in ratios.values())}


def disentanglement_error(single: dict[str, list], merged: dict[str, list]) -> float:
    """Eq. (6) SUM of task-conditional means, not a row-pooled average."""
    if not single or set(single) != set(merged):
        raise ValueError("task sets differ")
    value = 0.0
    for task in single:
        a, b = single[task], merged[task]
        if not a or len(a) != len(b):
            raise ValueError("missing aligned predictions")
        value += sum(x != y for x, y in zip(a, b)) / len(a)
    return value


def response_metrics(base: Tensor, actual: Tensor, linearized: Tensor) -> dict:
    if base.shape != actual.shape or actual.shape != linearized.shape or not base.numel():
        raise ValueError("incompatible output arrays")
    if any(not torch.isfinite(t).all() for t in (base, actual, linearized)):
        raise ValueError("nonfinite outputs")
    actual_change = (actual - base).double()
    remainder = (actual - linearized).double()
    denom = actual_change.square().sum().item()
    num = remainder.square().sum().item()
    return {"change_l2": math.sqrt(denom), "remainder_l2": math.sqrt(num),
            "relative_linearization_error": math.sqrt(num / denom) if denom else None,
            "zero_change_denominator": denom == 0,
            "jvp_energy": (linearized - base).double().square().mean().item(),
            "argmax_agreement": (actual.argmax(-1) == linearized.argmax(-1)).double().mean().item()}


def interaction(base: Tensor, a: Tensor, b: Tensor, ab: Tensor) -> float:
    if not (base.shape == a.shape == b.shape == ab.shape):
        raise ValueError("interaction shape mismatch")
    return (ab.double() - a.double() - b.double() + base.double()).square().mean().sqrt().item()
