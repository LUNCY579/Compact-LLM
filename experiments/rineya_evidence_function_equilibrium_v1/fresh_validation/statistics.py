"""Paired, simultaneous exact conservative intervals; no null-significance gate.

For a paired difference estimate B/n-C/n, bound each multinomial component
by its marginal exact binomial interval. A union bound gives the registered
one-sided coverage without assuming independence of gains and losses.
"""
from __future__ import annotations
import math

FROZEN_GATE_SPECS = (
    ("G1", "T", "T", "source", "difference_lower", ">", .05),
    ("G2", "H", "H025", "source", "difference_lower", ">", .05),
    ("G3", "T", "independent", "source", "difference_lower", ">", .05),
    ("G4", "T", "independent", "T", "difference_lower", ">", -.03),
    ("G5", "T", "independent", "T", "conditional_damage_upper", "<", .03),
    ("G6", "H", "independent", "T", "difference_lower", ">", .05),
    ("G7", "H", "independent", "H025", "difference_lower", ">", -.03),
    ("G8", "H", "independent", "H025", "conditional_damage_upper", "<", .03),
    ("G9", "answerable", "independent", "source", "conditional_damage_upper", "<", .03),
    ("G10", "broad", "independent", "source", "conditional_damage_upper", "<", .03),
)


def registered_gates() -> list[dict]:
    fields = ("id", "population", "candidate", "comparator", "metric", "relation", "threshold")
    return [dict(zip(fields, values)) for values in FROZEN_GATE_SPECS]


def binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0:
        return 1.0
    if p == 1:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    peak = max(logs)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(v - peak) for v in logs))


def cp_upper(k: int, n: int, tail: float) -> float:
    if not (type(k) is int and type(n) is int and 0 <= k <= n and n > 0 and 0 < tail < 1):
        raise ValueError("invalid binomial inputs")
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(tail) / n)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > tail:
            lo = mid
        else:
            hi = mid
    return hi


def cp_lower(k: int, n: int, tail: float) -> float:
    if not (type(k) is int and type(n) is int and 0 <= k <= n and n > 0 and 0 < tail < 1):
        raise ValueError("invalid binomial inputs")
    return 1 - cp_upper(n - k, n, tail)


def paired(comparator: list[bool], candidate: list[bool], *, claims=10, alpha=0.025) -> dict:
    if (len(comparator) != len(candidate) or not comparator
            or any(type(v) is not bool for v in comparator + candidate)):
        raise ValueError("complete paired boolean outcomes required")
    if type(claims) is not int or claims < 1:
        raise ValueError("positive prospectively fixed claims count required")
    n = len(candidate)
    gains = sum(b and not a for a, b in zip(comparator, candidate))
    losses = sum(a and not b for a, b in zip(comparator, candidate))
    correct = sum(comparator)
    tail = alpha / (2 * claims)
    lower = cp_lower(gains, n, tail) - cp_upper(losses, n, tail)
    upper = cp_upper(gains, n, tail) - cp_lower(losses, n, tail)
    # alpha is the directional family budget: protocol allocates .025 each
    # to false PROCEED and false REJECT, for a combined .05 across ten gates.
    return {"n": n, "comparator_correct": correct, "candidate_correct": sum(candidate),
            "gains": gains, "losses": losses, "discordant": gains + losses,
            "difference": (gains - losses) / n, "difference_lower": lower,
            "difference_upper": upper,
            "conditional_damage": losses / correct if correct else None,
            "conditional_damage_upper": cp_upper(losses, correct, alpha / claims) if correct else None,
            "conditional_damage_lower": cp_lower(losses, correct, alpha / claims) if correct else None,
            "method": "marginal_Clopper_Pearson_union_bound",
            "directional_family_alpha": alpha, "claims": claims, "component_tail": tail}


def damage_power(n: int, *, true_damage=0.01, margin=0.03, claims=10, alpha=0.025) -> dict:
    lo, hi = -1, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cp_upper(mid, n, alpha / claims) < margin:
            lo = mid
        else:
            hi = mid
    return {"comparator_correct_n": n, "true_damage_assumption": true_damage,
            "margin": margin, "max_losses_to_pass": lo,
            "power": binomial_cdf(lo, n, true_damage),
            "zero_loss_upper": cp_upper(0, n, alpha / claims)}


def verdict(metrics: dict[str, dict], gates: list[dict], *, prerequisites_complete: bool) -> dict:
    """Complete registered limbs before a decision; no negative-null clause.

    This pure statistic cannot prove independent data or authorize native launch.
    The future collector must establish those prerequisites separately.
    """
    if type(prerequisites_complete) is not bool or not prerequisites_complete:
        return {"decision": "HOLD", "reason": "prerequisites incomplete"}
    canonical_ids = {g["id"] for g in registered_gates()}
    if (len(gates) != 10 or {g.get("id") for g in gates} != canonical_ids
            or set(metrics) != canonical_ids):
        return {"decision": "HOLD", "reason": "missing or extra paired gates"}
    expected = {g["id"]: g for g in registered_gates()}
    if any(g != expected[g["id"]] for g in gates):
        return {"decision": "HOLD", "reason": "gate definitions changed from frozen protocol"}
    proceed, reject = [], []
    for gate in gates:
        value = metrics[gate["id"]]
        if (value.get("directional_family_alpha") != .025 or value.get("claims") != 10
                or type(value.get("n")) is not int or value.get("n") != 20160
                or type(value.get("comparator_correct")) is not int
                or not 0 <= value["comparator_correct"] <= value["n"]):
            return {"decision": "HOLD", "reason": "unbound statistics or incomplete n"}
        if gate["metric"].startswith("conditional_damage") and value["comparator_correct"] < 900:
            return {"decision": "HOLD", "reason": "insufficient comparator-correct parents"}
        names = (("difference_lower", "difference_upper") if gate["metric"] == "difference_lower"
                 else ("conditional_damage_lower", "conditional_damage_upper"))
        bounds = [value.get(name) for name in names]
        minimum = -1 if names[0] == "difference_lower" else 0
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not minimum <= bounds[0] <= bounds[1] <= 1):
            return {"decision": "HOLD", "reason": "invalid or contradictory statistical bounds"}
        if gate["metric"] == "difference_lower":
            proceed.append(value["difference_lower"] > gate["threshold"])
            reject.append(value["difference_upper"] < gate["threshold"])
        elif gate["metric"] == "conditional_damage_upper":
            proceed.append(value["conditional_damage_upper"] < gate["threshold"])
            reject.append(value["conditional_damage_lower"] > gate["threshold"])
        else:
            raise ValueError("unregistered decision metric")
    if all(proceed):
        return {"decision": "PROCEED", "reason": "all ten registered statistical gates pass"}
    if any(reject):
        return {"decision": "REJECT", "reason": "opposite-direction simultaneous bound exceeds registered margin",
                "failed_gates": [g["id"] for g, failed in zip(gates, reject) if failed]}
    return {"decision": "HOLD", "reason": "registered margins unresolved"}
