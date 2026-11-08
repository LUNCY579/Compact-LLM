"""Capacity schedule: how many coordinates each skill may claim.

``rho`` is the fraction of the CURRENTLY UNALLOCATED capacity that stays free after each skill.
Skill k claims ``k_t = min(|A_t|, max(1, round((1 - rho) * |A_t|)))`` coordinates, which is
CLARE's Eq. 6-7 (arXiv:2609.17026) with round-half-up made explicit. Arithmetic is exact
(Fraction from the decimal string), so budgets are reproducible integers on every platform.

One deliberate deviation from the paper: CLARE includes every coordinate tied at the threshold,
so its masks can exceed k_t. Here the budget is EXACT (ties are broken deterministically in
``select``), because the learned-vs-random comparison requires identical capacity.

For K skills the total claimed fraction is ``1 - rho**K`` (up to flooring). With K = 4:
rho 0.80 -> 59.04 %, 0.90 -> 34.39 %, 0.95 -> 18.55 %, 0.975 -> 9.63 %.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


def as_fraction(rho: str | float | Fraction) -> Fraction:
    if isinstance(rho, Fraction):
        f = rho
    elif isinstance(rho, str):
        f = Fraction(rho)
    elif isinstance(rho, float):
        # Go through repr so 0.9 means 9/10, not the binary double nearest to it.
        f = Fraction(repr(rho))
    else:
        raise TypeError(f"rho must be str/float/Fraction, got {type(rho).__name__}")
    if not (0 < f < 1):
        raise ValueError(f"rho must lie strictly in (0, 1), got {f}")
    return f


@dataclass(frozen=True)
class SkillBudget:
    skill: str
    position: int
    free_before: int
    budget: int
    free_after: int

    @property
    def fraction_of_free(self) -> float:
        return self.budget / self.free_before if self.free_before else 0.0

    def as_dict(self, total: int) -> dict:
        return {"skill": self.skill, "position": self.position, "free_before": self.free_before,
                "budget": self.budget, "free_after": self.free_after,
                "budget_fraction_of_free": self.fraction_of_free,
                "budget_fraction_of_total": self.budget / total}


def budgets(total: int, order: Sequence[str], rho: str | float | Fraction) -> list[SkillBudget]:
    if type(total) is not int or total <= 0:
        raise ValueError("total must be a positive int")
    if len(set(order)) != len(order) or not order:
        raise ValueError(f"skill order must be non-empty and unique, got {list(order)!r}")
    claim = 1 - as_fraction(rho)
    free = total
    out: list[SkillBudget] = []
    for i, skill in enumerate(order):
        if free <= 0:
            raise ValueError(f"skill {skill!r}: no free capacity left (rho={rho})")
        exact = claim * free + Fraction(1, 2)
        b = min(free, max(1, exact.numerator // exact.denominator))
        out.append(SkillBudget(skill, i, free, b, free - b))
        free -= b
    return out


def schedule_table(total: int, order: Sequence[str], rhos: Sequence[str]) -> list[dict]:
    rows = []
    for rho in rhos:
        bs = budgets(total, order, rho)
        claimed = sum(b.budget for b in bs)
        rows.append({"rho": str(rho), "budgets": [b.budget for b in bs],
                     "claimed": claimed, "claimed_fraction": claimed / total,
                     "closed_form_fraction": float(1 - as_fraction(rho) ** len(order))})
    return rows
