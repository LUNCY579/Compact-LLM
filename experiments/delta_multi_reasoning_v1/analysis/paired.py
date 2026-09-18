"""Paired outcome analysis: repair, damage, interactions, selector utilisation.

Everything here is pure computation over recorded per-item outcomes. No model is called, so
every function is testable against fixtures, and the tests are the only way a table in the
final report can be trusted.

WHY THIS MODULE IS SHAPED THIS WAY
----------------------------------
Three lessons from this programme's own record are encoded as hard behaviour, not advice:

1. A higher total is not an improvement. SPARSE-001's k=0.25 arm scored +3 arithmetic over
   the source while LOSING 6 of the source's 47 correct answers. `paired()` therefore always
   returns repair and damage separately and never reports a net without them.

2. Non-significance is not equivalence. Every long-context arm in INT-001/SPARSE-001 was
   statistically indistinguishable from the source (p = 0.57-0.85) while losing 7-11
   source-correct rows. `paired()` exposes `damage` so a "no observed regression" claim can
   only be made when that count is actually zero, and `preservation_verdict()` refuses to say
   PASS on the strength of a large p-value alone.

3. Single-draw differences smaller than the training-draw spread are uninterpretable. REP-001
   measured spreads of 7 acquisition rows, 13 long-context rows and 13 strict-arithmetic rows
   across three draws of the SAME recipe. `noise_verdict()` compares an observed difference
   against a registered per-endpoint spread and returns WITHIN_NOISE rather than a direction.

Items are clustered by WORLD (a scenario group). Calls, tokens and paraphrases are not
independent units, so every confidence interval here resamples worlds, never items.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

# Registered single-draw spreads from REP-001, three independent draws of the RRP-015 control
# recipe. An observed difference at or below these is not distinguishable from draw noise on
# one draw per system. Keys are endpoint names used throughout this package.
REP001_SINGLE_DRAW_SPREAD: dict[str, int] = {
    "acquisition": 7,            # 228-235 of 384
    "long_context": 13,          # 144-157 of 256
    "arithmetic_strict": 13,     # 51-64 of 150
}


# --------------------------------------------------------------------------------------- #
# Paired outcomes
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Paired:
    """The 2x2 table for one ordered comparison, plus the counts that matter."""

    baseline: str
    candidate: str
    n: int
    both_correct: int
    repair: int          # baseline wrong -> candidate correct
    damage: int          # baseline correct -> candidate wrong
    both_wrong: int
    repair_ids: tuple[str, ...] = field(default_factory=tuple)
    damage_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def net_gain(self) -> int:
        return self.repair - self.damage

    @property
    def baseline_correct(self) -> int:
        return self.both_correct + self.damage

    @property
    def candidate_correct(self) -> int:
        return self.both_correct + self.repair

    @property
    def zero_observed_damage(self) -> bool:
        """The strict criterion. True only when not one baseline success was lost."""
        return self.damage == 0

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline, "candidate": self.candidate, "n": self.n,
            "both_correct": self.both_correct, "repair": self.repair,
            "damage": self.damage, "both_wrong": self.both_wrong,
            "baseline_correct": self.baseline_correct,
            "candidate_correct": self.candidate_correct,
            "net_gain": self.net_gain,
            "zero_observed_damage": self.zero_observed_damage,
            "repair_ids": list(self.repair_ids), "damage_ids": list(self.damage_ids),
            "mcnemar_exact_two_sided_p": mcnemar_exact(self.repair, self.damage),
        }


def paired(baseline_label: str, candidate_label: str,
           baseline: Mapping[str, bool], candidate: Mapping[str, bool]) -> Paired:
    """Build the paired table over the ids present in BOTH maps.

    Raises when the id sets differ, because a silently-dropped item is how a damaged case
    disappears from a comparison. The caller must decide what an absent item means and say so.
    """
    bk, ck = set(baseline), set(candidate)
    if bk != ck:
        only_b, only_c = sorted(bk - ck)[:5], sorted(ck - bk)[:5]
        raise ValueError(
            f"id sets differ: {len(bk - ck)} only in {baseline_label} (e.g. {only_b}), "
            f"{len(ck - bk)} only in {candidate_label} (e.g. {only_c}). Refusing to compare a "
            f"partial overlap -- align the populations or declare the stratum incomparable.")
    both = rep = dam = neither = 0
    rep_ids: list[str] = []
    dam_ids: list[str] = []
    for i in sorted(bk):
        b, c = bool(baseline[i]), bool(candidate[i])
        if b and c:
            both += 1
        elif not b and c:
            rep += 1
            rep_ids.append(i)
        elif b and not c:
            dam += 1
            dam_ids.append(i)
        else:
            neither += 1
    return Paired(baseline_label, candidate_label, len(bk), both, rep, dam, neither,
                  tuple(rep_ids), tuple(dam_ids))


def mcnemar_exact(repair: int, damage: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs). 1.0 when none discordant."""
    n = repair + damage
    if n == 0:
        return 1.0
    k = min(repair, damage)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------------------- #
# Noise floor and preservation
# --------------------------------------------------------------------------------------- #

def noise_verdict(endpoint: str, difference: int,
                  spreads: Mapping[str, int] = REP001_SINGLE_DRAW_SPREAD) -> str:
    """WITHIN_NOISE / EXCEEDS_NOISE / NO_REGISTERED_SPREAD for a single-draw difference.

    `difference` is the raw count difference between two systems on one draw each. The
    comparison is against the observed spread of the SAME recipe across independent training
    draws, which is the only scale on which a one-draw difference can be read.
    """
    if endpoint not in spreads:
        return "NO_REGISTERED_SPREAD"
    return "EXCEEDS_NOISE" if abs(difference) > spreads[endpoint] else "WITHIN_NOISE"


def preservation_verdict(p: Paired, margin: int | None = None) -> dict:
    """Report the strict criterion and any noninferiority criterion SEPARATELY.

    Failure to find a significant decline is not proof of preservation, and an underpowered
    sample means inconclusive rather than pass. Both are encoded here so a caller cannot
    accidentally promote a large p-value into a preservation claim.
    """
    strict = "PASS_ZERO_OBSERVED_DAMAGE" if p.damage == 0 else "FAIL_OBSERVED_DAMAGE"
    out = {
        "strict_criterion": strict,
        "damage_count": p.damage,
        "damaged_ids": list(p.damage_ids),
        "mcnemar_exact_two_sided_p": mcnemar_exact(p.repair, p.damage),
        "note": ("A large p-value is not equivalence. The strict criterion is the only one "
                 "that licenses 'no observed regressions on these registered cases'."),
    }
    if margin is None:
        out["noninferiority"] = "NOT_REGISTERED"
        return out
    out["noninferiority_margin"] = margin
    if p.net_gain >= -margin:
        # Power check: with few discordant pairs the test cannot detect the margin at all.
        out["noninferiority"] = ("PASS_WITHIN_MARGIN" if (p.repair + p.damage) >= margin
                                 else "INCONCLUSIVE_UNDERPOWERED")
    else:
        out["noninferiority"] = "FAIL_OUTSIDE_MARGIN"
    return out


# --------------------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------------------- #

def interaction(s: Callable[[str, str], float],
                sys_a: str, sys_b: str, inf_hi: str, inf_lo: str) -> dict:
    """A difference-in-differences contrast: [s(A,hi)-s(A,lo)] - [s(B,hi)-s(B,lo)].

    This is the shape of every interaction the protocol predeclares, e.g.
    Delta x MR = [s(D,MR)-s(D,O)] - [s(B,MR)-s(B,O)].

    A positive point estimate is NOT established synergy: it is one number from one draw, and
    the caller must pair it with a world-clustered interval and the component simple effects,
    which is why those are returned alongside rather than hidden.
    """
    a_hi, a_lo = s(sys_a, inf_hi), s(sys_a, inf_lo)
    b_hi, b_lo = s(sys_b, inf_hi), s(sys_b, inf_lo)
    simple_a, simple_b = a_hi - a_lo, b_hi - b_lo
    return {
        "contrast": f"[s({sys_a},{inf_hi})-s({sys_a},{inf_lo})] - [s({sys_b},{inf_hi})-s({sys_b},{inf_lo})]",
        "simple_effect_on": {sys_a: simple_a, sys_b: simple_b},
        "cells": {f"{sys_a}/{inf_hi}": a_hi, f"{sys_a}/{inf_lo}": a_lo,
                  f"{sys_b}/{inf_hi}": b_hi, f"{sys_b}/{inf_lo}": b_lo},
        "interaction": simple_a - simple_b,
        "reading": ("A positive interaction means the inference condition helps sys_a more "
                    "than sys_b. It is a point estimate from one draw and establishes nothing "
                    "on its own."),
    }


# --------------------------------------------------------------------------------------- #
# World-clustered bootstrap
# --------------------------------------------------------------------------------------- #

def cluster_bootstrap_ci(values_by_world: Mapping[str, Sequence[float]],
                         statistic: Callable[[list[float]], float] = None,
                         iters: int = 10000, alpha: float = 0.05,
                         seed: int = 20260918) -> dict:
    """Percentile CI resampling WORLDS with replacement, not items.

    Items inside a world share a scenario, and role calls inside an item share a prompt, so
    neither is an independent unit. Resampling worlds is the weakest assumption that is still
    defensible here. The seed is fixed so the interval is reproducible.
    """
    import random

    worlds = sorted(values_by_world)
    if not worlds:
        return {"point": None, "lo": None, "hi": None, "n_worlds": 0}
    stat = statistic or (lambda xs: sum(xs) / len(xs) if xs else 0.0)
    flat = [v for w in worlds for v in values_by_world[w]]
    point = stat(flat)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iters):
        pool: list[float] = []
        for _ in worlds:
            pool.extend(values_by_world[worlds[rng.randrange(len(worlds))]])
        if pool:
            draws.append(stat(pool))
    if not draws:
        return {"point": point, "lo": None, "hi": None, "n_worlds": len(worlds)}
    draws.sort()
    lo = draws[max(0, int(math.floor((alpha / 2) * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return {"point": point, "lo": lo, "hi": hi, "n_worlds": len(worlds),
            "iters": iters, "alpha": alpha, "seed": seed,
            "unit": "world (scenario group), resampled with replacement"}


# --------------------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------------------- #

def holm_bonferroni(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down over a predeclared family. Order-free and deterministic."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: dict[str, dict] = {}
    still_rejecting = True
    for idx, (name, p) in enumerate(items):
        thresh = alpha / (m - idx)
        if still_rejecting and p <= thresh:
            out[name] = {"p": p, "threshold": thresh, "reject": True}
        else:
            still_rejecting = False
            out[name] = {"p": p, "threshold": thresh, "reject": False}
    return {"alpha": alpha, "family_size": m, "results": out,
            "note": "Family must be predeclared. Adding a contrast after seeing results voids this."}


# --------------------------------------------------------------------------------------- #
# Selector utilisation and role complementarity
# --------------------------------------------------------------------------------------- #

def selector_utilisation(candidate_correct: Mapping[str, Mapping[str, bool]],
                         selected: Mapping[str, str | None],
                         final_correct: Mapping[str, bool]) -> dict:
    """Correct selections on cases with a correct available candidate, over those cases.

    `candidate_correct[item][role]` is whether that role's candidate was correct.
    `selected[item]` is the role id the selector chose, or None for abstention/empty set.

    The denominator is deliberately "cases where a correct candidate existed". Cases where no
    role was correct cannot test the selector and are reported separately rather than
    inflating or deflating the rate.
    """
    have = [i for i, c in candidate_correct.items() if any(c.values())]
    none = [i for i, c in candidate_correct.items() if not any(c.values())]
    got = [i for i in have if final_correct.get(i)]
    abstained_with_candidate = [i for i in have if selected.get(i) is None]
    oracle = len(have)
    return {
        "n_items": len(candidate_correct),
        "n_with_correct_candidate": oracle,
        "n_with_no_correct_candidate": len(none),
        "n_selected_correctly": len(got),
        "utilisation": (len(got) / oracle) if oracle else None,
        "n_abstained_despite_correct_candidate": len(abstained_with_candidate),
        "abstained_despite_correct_candidate_ids": sorted(abstained_with_candidate)[:50],
        "candidate_oracle_ceiling": oracle,
        "selector_shortfall_vs_oracle": oracle - len(got),
        "note": ("The oracle ceiling is DIAGNOSTIC. It is the score a label-aware chooser would "
                 "reach over these recorded candidates and gets no deployment credit."),
    }


def role_complementarity(candidate_correct: Mapping[str, Mapping[str, bool]]) -> dict:
    """Pairwise agreement, unique-solve counts, and the correlated-error picture."""
    items = sorted(candidate_correct)
    roles = sorted({r for c in candidate_correct.values() for r in c})
    per_role = {r: sum(1 for i in items if candidate_correct[i].get(r)) for r in roles}
    unique = {r: sum(1 for i in items
                     if candidate_correct[i].get(r)
                     and not any(candidate_correct[i].get(o) for o in roles if o != r))
              for r in roles}
    pair: dict[str, dict] = {}
    for a_i, a in enumerate(roles):
        for b in roles[a_i + 1:]:
            both = sum(1 for i in items if candidate_correct[i].get(a) and candidate_correct[i].get(b))
            aonly = sum(1 for i in items if candidate_correct[i].get(a) and not candidate_correct[i].get(b))
            bonly = sum(1 for i in items if candidate_correct[i].get(b) and not candidate_correct[i].get(a))
            neither = len(items) - both - aonly - bonly
            agree = (both + neither) / len(items) if items else None
            pair[f"{a}|{b}"] = {"both": both, f"{a}_only": aonly, f"{b}_only": bonly,
                                "neither": neither, "agreement": agree}
    any_correct = sum(1 for i in items if any(candidate_correct[i].values()))
    all_correct = sum(1 for i in items
                      if roles and all(candidate_correct[i].get(r) for r in roles))
    return {
        "roles": roles, "n_items": len(items),
        "per_role_correct": per_role,
        "uniquely_solved_by": unique,
        "pairwise": pair,
        "any_role_correct": any_correct,
        "all_roles_correct": all_correct,
        "correlated_failure": len(items) - any_correct,
        "best_single_role": max(per_role, key=per_role.get) if per_role else None,
        "oracle_minus_best_single": (any_correct - max(per_role.values())) if per_role else None,
        "reading": ("oracle_minus_best_single is the headroom differentiated roles create. If "
                    "it is ~0 the roles are redundant however different their names are."),
    }


def roles_perform_different_operations(role_outputs: Mapping[str, Mapping[str, str]]) -> dict:
    """Evidence that roles did different things, not merely carried different names.

    `role_outputs[item][role]` is the raw decoded text. Identical text across roles on the
    same item is the signal that a role prompt changed nothing.
    """
    items = sorted(role_outputs)
    roles = sorted({r for c in role_outputs.values() for r in c})
    identical_pairs: dict[str, int] = {}
    for a_i, a in enumerate(roles):
        for b in roles[a_i + 1:]:
            same = sum(1 for i in items
                       if a in role_outputs[i] and b in role_outputs[i]
                       and role_outputs[i][a] == role_outputs[i][b])
            identical_pairs[f"{a}|{b}"] = same
    all_same = sum(1 for i in items
                   if len({role_outputs[i][r] for r in roles if r in role_outputs[i]}) <= 1)
    return {
        "n_items": len(items), "roles": roles,
        "items_where_all_roles_emitted_identical_text": all_same,
        "pairwise_identical_counts": identical_pairs,
        "fraction_all_identical": (all_same / len(items)) if items else None,
        "verdict": ("ROLES_COLLAPSED" if items and all_same / len(items) > 0.9
                    else "ROLES_DIVERGE"),
        "note": ("A high identical count means the role prompts did not induce different "
                 "operations on this backend, so an MR result cannot be attributed to "
                 "differentiated reasoning."),
    }


def cost_per_net_correct(net_gain: int, cost: Mapping[str, float]) -> dict:
    """Cost per additional correct case, and an explicit refusal when net gain is not positive."""
    if net_gain <= 0:
        return {"net_gain": net_gain, "cost": dict(cost),
                "cost_per_net_correct": None,
                "verdict": ("NOT_REPORTABLE: net gain is zero or negative, so a cost-per-gain "
                            "ratio would describe a benefit that was not observed.")}
    return {"net_gain": net_gain, "cost": dict(cost),
            "cost_per_net_correct": {k: v / net_gain for k, v in cost.items()},
            "verdict": "REPORTABLE"}
