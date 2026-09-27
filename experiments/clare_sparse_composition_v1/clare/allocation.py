"""Arm orchestration: the CLARE two-stage allocation and its controls, on top of ``DeltaHost``.

Arms (CLARE-RINEYA-001 primary = A, D, E; B/C are cheap derivatives; F is config support only):

A  dense_sequential    each skill trains a zero-initialised DENSE delta over the whole footprint
                       on top of the frozen sum of earlier skills; nothing is protected.
B  independent_sum     each skill trains a dense delta from the base alone; composition = sum.
C  magnitude_sparse    B's deltas magnitude-pruned to the CLARE budgets (supports may overlap).
D  random_disjoint     CLARE budgets, strict disjoint RANDOM masks, masked retraining.
E  learned_disjoint    CLARE: L1 discovery probe on free coordinates -> top-|delta| mask ->
                       RESET -> masked retraining -> freeze.
F  shared_private      a shared learned support trainable by every skill + CLARE private masks.

For D and E each skill trains in the presence of the frozen earlier skills (CLARE Sec. 3.1: "the
next task starts from this updated adapter"); B trains every skill from the base alone.
Every arm uses the same rows, order, update groups, recipe and seed for a given skill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .deltas import Delta, DenseDelta, SparseDelta, compose, magnitude_prune
from .footprint import Footprint
from .masks import AllocationState, SkillMask
from .recipe import Recipe
from .schedule import SkillBudget, budgets
from .select import seed_key, select_random, select_top_importance
from .training import DeltaHost, LossFn, StageLog, run_stage

ARMS = ("A_dense_sequential", "B_independent_sum", "C_magnitude_sparse", "D_random_disjoint",
        "E_learned_disjoint", "F_shared_private")


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    order: tuple[str, ...]
    rho: str = "0.9"
    seed: int = 0
    study: str = "CLARE-RINEYA-001"
    discovery_fraction: float = 0.25   # E/F: probe length as a fraction of the skill's update groups
    l1_lambda: float = 0.1             # E/F: proximal threshold per step = lr_t * l1_lambda
    lambda_fallback: tuple = ()        # E: outcome-blind ladder tried if the probe leaves < 2x budget non-zeros
    shared_fraction: float = 0.0       # F only: shared support as a fraction of the footprint

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}")
        if len(set(self.order)) != len(self.order) or not self.order:
            raise ValueError("order must be non-empty and unique")
        if not 0 < self.discovery_fraction <= 1:
            raise ValueError("discovery_fraction must lie in (0, 1]")
        if self.l1_lambda < 0:
            raise ValueError("l1_lambda must be >= 0")
        if self.arm == "F_shared_private" and not 0 < self.shared_fraction < 1:
            raise ValueError("F needs 0 < shared_fraction < 1")

    def key(self, *parts) -> np.uint64:
        return seed_key(self.study, self.arm, self.seed, "|".join(self.order), *parts)


@dataclass
class SkillData:
    """Compiled training rows per skill (dicts with ids/labels/supervised_tokens) and groups."""
    rows: Mapping[str, Sequence[dict]]
    groups: Mapping[str, Sequence[Sequence[int]]]

    def discovery_groups(self, skill: str, fraction: float) -> list[Sequence[int]]:
        g = list(self.groups[skill])
        n = max(1, int(round(len(g) * fraction)))
        return g[:n]


@dataclass
class SkillRecord:
    skill: str
    position: int
    delta: Delta
    mask: SkillMask | None
    budget: SkillBudget | None
    train_log: StageLog | None
    discovery: dict | None = None
    frozen_checksum_before: str | None = None
    frozen_checksum_after: str | None = None


@dataclass
class ArmResult:
    spec: ArmSpec
    footprint: Footprint
    records: list[SkillRecord] = field(default_factory=list)
    allocation: AllocationState | None = None
    shared_mask: SkillMask | None = None

    def deltas(self) -> dict[str, Delta]:
        return {r.skill: r.delta for r in self.records}

    def prefix_sum(self, k: int) -> DenseDelta:
        """Composed delta after the first k skills (the model 'after skill k')."""
        return compose([r.delta for r in self.records[:k]])


def _train(host: DeltaHost, data: SkillData, skill: str, loss_fn: LossFn, recipe: Recipe,
           groups=None, l1: float = 0.0, guard=None) -> StageLog:
    rows = data.rows[skill]
    groups = data.groups[skill] if groups is None else groups
    log, _ = run_stage(host, rows, groups, loss_fn, recipe, l1_lambda=l1, guard=guard)
    return log


def _check_frozen(host: DeltaHost, before: str, what: str) -> str:
    after = host.frozen_checksum()
    if after != before:
        raise AssertionError(f"{what}: the frozen delta changed during training")
    return after


def run_arm(spec: ArmSpec, host: DeltaHost, data: SkillData, loss_fn: LossFn, recipe: Recipe,
            *, independent: Mapping[str, Delta] | None = None,
            on_skill: Callable[[SkillRecord, "ArmResult"], None] | None = None,
            persist: Callable[[DeltaHost, str, int], Delta] | None = None,
            resume: Mapping[str, SkillRecord] | None = None,
            guard=None) -> ArmResult:
    """Train one arm. ``independent`` supplies precomputed B deltas (for C, or to reuse A's first stage).

    ``persist(host, skill, position)`` (optional) must stream the open stage's trainable delta to
    disk and return a disk-backed Delta (``DeltaHost.save_trainable``); without it deltas are kept
    in memory, which is only appropriate for small models.

    ``resume`` maps skills already completed (in order, from saved manifests) to their records;
    they are re-claimed and skipped, so a crash loses at most the skill in progress.
    """
    take = (lambda skill, k: persist(host, skill, k)) if persist else (lambda skill, k: host.trainable_delta())
    fp = host.footprint
    res = ArmResult(spec, fp)
    sched = budgets(fp.total, spec.order, spec.rho)

    if spec.arm == "C_magnitude_sparse":
        if independent is None or set(independent) != set(spec.order):
            raise ValueError("C needs the independently trained B deltas for every skill")
        for k, (skill, b) in enumerate(zip(spec.order, sched)):
            pruned = magnitude_prune(independent[skill], b.budget, spec.key("C", skill))
            rec = SkillRecord(skill, k, pruned, pruned.mask, b, None)
            res.records.append(rec)
            if on_skill:
                on_skill(rec, res)
        return res

    strict = spec.arm in ("D_random_disjoint", "E_learned_disjoint", "F_shared_private")
    state = AllocationState(fp, strict=True) if strict else None
    res.allocation = state

    if spec.arm == "F_shared_private":
        res.shared_mask = _shared_support(spec, host, data, loss_fn, recipe, guard)
        state.claim("__shared__", res.shared_mask)
        sched = budgets(state.free_count(), spec.order, spec.rho)

    for k, skill in enumerate(spec.order):
        b = sched[k] if strict else None
        prev = [r.delta for r in res.records]
        if resume and skill in resume:
            rec = resume[skill]
            if rec.position != k or (strict and (rec.mask is None or rec.mask.count() != b.budget)):
                raise ValueError(f"resume record for {skill} does not match the registered schedule")
            if strict:
                state.claim(skill, rec.mask, expected_budget=b.budget)
            res.records.append(rec)
            continue
        if resume and any(s in resume for s in spec.order[k + 1:]):
            raise ValueError("resume records must be a prefix of the order")
        if spec.arm == "B_independent_sum":
            if independent is not None and skill in independent:
                rec = SkillRecord(skill, k, independent[skill], None, None, None)
                res.records.append(rec)
                if on_skill:
                    on_skill(rec, res)
                continue
            host.set_frozen(None)
        else:
            host.set_frozen(prev if prev else None)
        before = host.frozen_checksum()
        discovery = None

        if spec.arm in ("A_dense_sequential", "B_independent_sum"):
            host.start_dense(None)
            log = _train(host, data, skill, loss_fn, recipe, guard=guard)
            delta = take(skill, k)
            mask = None
            host.stop_trainable()
        else:
            free = state.free()
            if spec.arm == "D_random_disjoint":
                sel, rep = select_random(fp, {s.name: free.bool(s.name) for s in fp}, b.budget,
                                         spec.key("D", skill))
                mask = SkillMask.from_bool(fp, sel)
                discovery = {"rule": "random", "selection": rep.as_dict()}
            else:
                mask, discovery = _discover(spec, host, data, skill, loss_fn, recipe, free, b.budget, guard)
            state.claim(skill, mask, expected_budget=b.budget)
            train_mask = mask if res.shared_mask is None else mask.union(res.shared_mask)
            host.start_sparse(train_mask)
            log = _train(host, data, skill, loss_fn, recipe, guard=guard)
            delta = take(skill, k)
            host.stop_trainable()
        after = _check_frozen(host, before, f"{spec.arm}/{skill}")
        rec = SkillRecord(skill, k, delta, mask, b, log, discovery, before, after)
        res.records.append(rec)
        if on_skill:
            on_skill(rec, res)
    host.set_frozen(None)
    return res


def _discover(spec: ArmSpec, host: DeltaHost, data: SkillData, skill: str, loss_fn: LossFn,
              recipe: Recipe, free: SkillMask, budget: int, guard) -> tuple[SkillMask, dict]:
    """CLARE Stage 1 (probe on free coordinates, L1) -> top-|delta| selection -> RESET.

    Registered lambda gate (outcome-blind: it reads only the probe's non-zero count, never an
    accuracy): if the probe leaves fewer than 2x ``budget`` non-zero free coordinates, the probe is
    discarded (reset) and repeated with the next value of ``spec.lambda_fallback``.
    """
    fp = host.footprint
    pre_effective = host.effective_checksum()
    pre_frozen = host.frozen_checksum()
    groups = data.discovery_groups(skill, spec.discovery_fraction)
    ladder = [spec.l1_lambda] + [l for l in spec.lambda_fallback if l < spec.l1_lambda]
    attempts = []
    for lam in ladder:
        host.start_dense(free)
        log = _train(host, data, skill, loss_fn, recipe, groups=groups, l1=lam, guard=guard)
        leaked = host.outside_mask_nonzero()
        if leaked:
            raise AssertionError(f"discovery moved {leaked} coordinates outside the free set")
        nonzero_free = 0
        for s in fp:
            q = host.importance(s.name)
            nonzero_free += int(np.count_nonzero(q[free.bool(s.name)]))
        attempts.append({"l1_lambda": lam, "nonzero_free_after_probe": nonzero_free})
        if nonzero_free >= 2 * budget or lam == ladder[-1]:
            break
        host.stop_trainable()  # gate failed: discard this probe entirely and retry
    sel, rep = select_top_importance(fp, lambda sl: host.importance(sl.name), {s.name: free.bool(s.name) for s in fp},
                                     budget, spec.key("E-tiebreak", skill))
    zero_selected = 0
    for s in fp:
        zero_selected += int(np.count_nonzero(host.importance(s.name)[sel[s.name]] == 0))
    host.stop_trainable()  # RESET: discard every discovery value; the optimiser state goes with it
    post_effective = host.effective_checksum()
    if post_effective != pre_effective or host.frozen_checksum() != pre_frozen:
        raise AssertionError("discovery reset did not restore the pre-discovery model")
    mask = SkillMask.from_bool(fp, sel)
    return mask, {"rule": "learned_top_abs_delta", "l1_lambda": attempts[-1]["l1_lambda"],
                  "lambda_attempts": attempts, "lambda_gate_passed": nonzero_free >= 2 * budget,
                  "discovery_updates": log.updates, "discovery_fraction": spec.discovery_fraction,
                  "discovery_log": log.as_dict(), "free_coordinates": free.count(),
                  "nonzero_free_after_probe": nonzero_free, "selected_with_zero_importance": zero_selected,
                  "selection": rep.as_dict(), "effective_checksum_before": pre_effective,
                  "effective_checksum_after_reset": post_effective}


def _shared_support(spec: ArmSpec, host: DeltaHost, data: SkillData, loss_fn: LossFn,
                    recipe: Recipe, guard) -> SkillMask:
    """F only: a joint probe over every skill's discovery groups picks a shared support."""
    fp = host.footprint
    host.set_frozen(None)
    budget = int(round(spec.shared_fraction * fp.total))
    host.start_dense(None)
    for skill in spec.order:
        _train(host, data, skill, loss_fn, recipe, groups=data.discovery_groups(skill, spec.discovery_fraction),
               l1=spec.l1_lambda, guard=guard)
    sel, _ = select_top_importance(fp, lambda sl: host.importance(sl.name), None, budget, spec.key("F-shared"))
    host.stop_trainable()
    return SkillMask.from_bool(fp, sel)
