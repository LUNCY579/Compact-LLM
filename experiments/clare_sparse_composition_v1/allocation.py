"""Exact capacity accounting, deterministic global top-k, explicit boolean masks."""
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import torch


def budget(available: int, rho: float) -> int:
    if type(available) is not int or available < 0 or type(rho) not in (int, float) or not 0 < rho < 1:
        raise ValueError("invalid capacity/rho")
    if available == 0:
        return 0
    k = int((Decimal(available) * (1 - Decimal(str(rho)))).to_integral_value(rounding=ROUND_HALF_UP))
    return min(available, max(1, k))


def checked_mask(mask, size):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device.type != "cpu" or tuple(mask.shape) != (size,):
        raise ValueError("mask must be an explicit CPU bool vector of the exact size")
    return mask


def select(free, k: int, *, scores=None, seed: int = 0):
    checked_mask(free, free.numel())
    if type(k) is not int or not 0 <= k <= int(free.sum()):
        raise ValueError("invalid exact budget")
    ids = torch.where(free)[0]
    if scores is None:
        if type(seed) is not int or seed < 0:
            raise ValueError("invalid seed")
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    else:
        if scores.device.type != "cpu" or scores.shape != free.shape or not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
            raise ValueError("invalid scores")
        # Stable ties use the canonical parameter-name/row-major coordinate order.
        order = torch.argsort(scores[ids].abs(), descending=True, stable=True)
    mask = torch.zeros_like(free)
    mask[ids[order[:k]]] = True
    return mask


@dataclass
class Allocation:
    size: int
    order: tuple[str, ...]
    masks: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.size) is not int or self.size <= 0 or not self.order or len(set(self.order)) != len(self.order):
            raise ValueError("invalid allocation layout/order")
        existing = self.masks
        self.masks = {}
        for skill, mask in existing.items():
            self.claim(skill, mask, int(mask.sum()))

    @property
    def free(self):
        out = torch.ones(self.size, dtype=torch.bool)
        for mask in self.masks.values():
            out &= ~mask
        return out

    def claim(self, skill, mask, expected):
        checked_mask(mask, self.size)
        if len(self.masks) >= len(self.order) or skill != self.order[len(self.masks)]:
            raise ValueError("skill order/reassignment violation")
        if int(mask.sum()) != expected or bool((mask & ~self.free).any()):
            raise ValueError("budget or disjointness violation")
        self.masks[skill] = mask.clone()
