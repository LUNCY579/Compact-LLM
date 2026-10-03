"""CPU-only, model-independent paired audits. No training or promotion authority.

Counts describe the observed bank. Probability bounds additionally require iid
sampling at the declared unit; reused selection banks do not establish generalization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Outcome:
    id: str
    group: str
    fingerprint: str
    correct: bool


def _indexed(rows: list[Outcome]) -> dict[str, Outcome]:
    result = {}
    for row in rows:
        if any(not isinstance(value, str) or not value for value in [row.id, row.group, row.fingerprint]):
            raise ValueError("id, group, and input/target fingerprint are required")
        if type(row.correct) is not bool:
            raise ValueError("correct must be a boolean; missing outcomes are not successes")
        if row.id in result:
            raise ValueError(f"duplicate id: {row.id}")
        result[row.id] = row
    if not result:
        raise ValueError("empty bank")
    return result


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if p == 0:
        return 1.0
    if p == 1:
        return float(k == n)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    high = max(terms)
    return min(1.0, math.exp(high) * math.fsum(math.exp(t - high) for t in terms))


def loss_upper_bound(losses: int, eligible: int, alpha: float = 0.05) -> float | None:
    """One-sided exact Clopper-Pearson bound, conditional on source success.

    None is uninformative, never evidence for zero forgetting. No multiple-testing
    adjustment is made here; pass a prespecified adjusted alpha if appropriate.
    """
    if type(losses) is not int or type(eligible) is not int or not 0 <= losses <= eligible:
        raise ValueError("counts must satisfy 0 <= losses <= eligible")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if eligible == 0:
        return None
    if losses == eligible:
        return 1.0
    if losses == 0:
        return -math.expm1(math.log(alpha) / eligible)
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2
        if _binomial_cdf(losses, eligible, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def exact_sign_flip_p(differences: list[int]) -> float:
    """Two-sided paired sign randomization at the supplied independent unit.

    Assumes exchangeable arm labels/sign symmetry under the null. Group members
    must stay together. Exploratory p-values on reused banks are not confirmation.
    """
    if any(type(d) is not int for d in differences):
        raise ValueError("differences must be integers")
    nonzero = [d for d in differences if d]
    target = abs(sum(nonzero))
    ways = Counter({0: 1})
    for value in nonzero:
        updated = Counter()
        for total, count in ways.items():
            updated[total + value] += count
            updated[total - value] += count
        ways = updated
    return sum(count for total, count in ways.items() if abs(total) >= target) / (2 ** len(nonzero))


def compare(source: list[Outcome], candidate: list[Outcome]) -> dict:
    baseline, arm = _indexed(source), _indexed(candidate)
    if baseline.keys() != arm.keys():
        raise ValueError("banks must contain exactly the same row IDs; no inner-join scoring")
    for key, old in baseline.items():
        new = arm[key]
        if old.group != new.group or old.fingerprint != new.fingerprint:
            raise ValueError(f"input, target, or group changed for {key}")
    lost = sorted(k for k in baseline if baseline[k].correct and not arm[k].correct)
    gained = sorted(k for k in baseline if not baseline[k].correct and arm[k].correct)
    old_correct = sum(r.correct for r in baseline.values())
    grouped = defaultdict(list)
    for key in sorted(baseline):
        grouped[baseline[key].group].append(key)
    eligible_groups = [g for g, ids in grouped.items() if any(baseline[i].correct for i in ids)]
    loss_set = set(lost)
    damaged_groups = [g for g in eligible_groups if any(i in loss_set for i in grouped[g])]
    diffs = [sum(int(arm[i].correct) - int(baseline[i].correct) for i in ids)
             for ids in grouped.values()]
    return {
        "rows": len(baseline), "groups": len(grouped),
        "source_correct": old_correct, "candidate_correct": sum(r.correct for r in arm.values()),
        "retained": old_correct - len(lost), "lost": len(lost), "gained": len(gained),
        "net": len(gained) - len(lost), "churn": len(gained) + len(lost),
        "lost_ids": lost, "gained_ids": gained,
        "observed_zero_loss": not lost,
        "row_loss_fraction": len(lost) / old_correct if old_correct else None,
        "row_loss_upper95_iid_only": loss_upper_bound(len(lost), old_correct),
        "eligible_groups": len(eligible_groups), "damaged_groups": len(damaged_groups),
        "damaged_group_ids": damaged_groups,
        "group_any_loss_upper95_iid_only": loss_upper_bound(len(damaged_groups), len(eligible_groups)),
        "row_mcnemar_p_descriptive": exact_sign_flip_p([1] * len(gained) + [-1] * len(lost)),
        "group_sign_flip_p_descriptive": exact_sign_flip_p(diffs),
        "population_no_forgetting_established": False,
    }


def sequence(stages: list[list[Outcome]]) -> list[dict]:
    """Track base, previous-stage and ever-correct losses through skill additions."""
    if len(stages) < 2:
        raise ValueError("at least a base and one trained stage are required")
    base = stages[0]
    ever = _indexed(base)
    reports = []
    for index, stage in enumerate(stages[1:], 1):
        reports.append({
            "stage": index,
            "vs_base": compare(base, stage),
            "vs_previous": compare(stages[index - 1], stage),
            "vs_ever_correct": compare(list(ever.values()), stage),
        })
        ever = {r.id: Outcome(r.id, r.group, r.fingerprint, ever[r.id].correct or r.correct)
                for r in stage}
    return reports


def composition(specialists: dict[str, list[Outcome]], combined: dict[str, list[Outcome]]) -> dict:
    """Audit each component's retained competence after activating skills together.

    Each entry is that specialist's own skill bank. This diagnoses interference;
    true joint-task generalization needs an additional reachable composed-task bank.
    """
    if not specialists or specialists.keys() != combined.keys():
        raise ValueError('all component skill banks are required')
    by_skill = {skill: compare(rows, combined[skill]) for skill, rows in specialists.items()}
    return {
        'per_skill': by_skill,
        'all_components_have_measured_competence': all(r['source_correct'] > 0 for r in by_skill.values()),
        'component_successes_lost': sum(r['lost'] for r in by_skill.values()),
        'joint_task_generalization_established': False,
    }
