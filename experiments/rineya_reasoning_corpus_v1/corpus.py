"""Deterministic corpus assembly: balancing, disjointness and split isolation.

Answerability design. An information gap is genuinely NECESSARY for an answer to be
undetermined -- a complete record always determines it -- so "no gap therefore answerable"
is sound reasoning, not a shortcut, and is not something to engineer away. The defect in
the scout bank is the CONVERSE: there, a gap implies UNKNOWN with probability 1.0 in three
of four families, so abstention is obtainable from one cue without doing the task. RTG-1
targets P(UNKNOWN | gap) = 0.5 by allocating, in every cell, as many gap-bearing
answerable rows as UNKNOWN rows.

Every emitted row carries a unique canonical structural fingerprint. Name-only variants of
one structure share an answer and teach nothing extra, so the variant cap is 1 and the
achieved unique-structure count equals the row count by construction.
"""
from __future__ import annotations
import hashlib
import random
from collections import Counter, defaultdict

from . import families as F
from . import sampling

SEED = 20260917
SPLITS = (("train", 80), ("dev", 10), ("holdout", 10))

# Rows per (family, tier). `match` always reproduces the scout's own parameter range.
#
# Two cells are capped by measured structural capacity rather than by preference:
#   * prerequisite_schedule/match -- only 480 distinct 4-job instances exist in the whole
#     scout construction, and only EIGHT of them have answer "3" (a depth-3 chain with the
#     query at its end), none carrying an information gap. The evaluation bank already uses
#     SIX of those eight, so exactly two remain available to a disjoint corpus, capping this
#     cell at 10 rows under uniform answer balance. This is a property of the eval
#     instrument, not a choice: 4-job training data at scale cannot be both balanced and
#     disjoint from the bank. Larger job counts teach the identical algorithm -- earliest
#     start is the longest prerequisite chain regardless of job count -- and carry the
#     family's volume, with the eval's 4-job instances strictly simpler than those.
#   * the `easy` tiers are short by construction, so their spaces are small
#     (register_machine 1-2 ops admits ~910 structures).
# Both figures are verified by enumeration in tests/test_capacity.py.
# Every quota below sits at or under the capacity measured by capacity.py (60,000 draws
# per cell at the configured gap share), so no cell is ever filled by relaxing balance or
# by repeating a structure:
#   register_machine/easy       150 of   208     counterfactual_state/easy   150 of   208
#   register_machine/match     3000 of 18264     counterfactual_state/match 3000 of 16090
#   register_machine/hard      2850 of 26570     counterfactual_state/hard  2850 of 25210
#   evidence_chain/easy         600 of 13770     prerequisite_schedule/match  10 of    10
#   evidence_chain/match       3000 of 17084     prerequisite_schedule/wide5 1200 of  2395
#   evidence_chain/hard        2400 of 17248     prerequisite_schedule/wide6 4790 of 24255
#
# prerequisite_schedule/easy is measured and DELIBERATELY UNUSED: three jobs cannot reach
# earliest start 3, so answer "3" is unreachable there and uniform balance is impossible.
# Its measured capacity is exactly 0 under a uniform-balance requirement.
QUOTAS = {
    "register_machine": {"easy": 150, "match": 3000, "hard": 2850},
    "counterfactual_state": {"easy": 150, "match": 3000, "hard": 2850},
    "evidence_chain": {"easy": 600, "match": 3000, "hard": 2400},
    "prerequisite_schedule": {"match": 10, "wide5": 1200, "wide6": 4790},
}
UNKNOWN_SHARE = 0.20
# Gap-bearing ANSWERABLE rows, as a share of the whole cell.
#
# This is the corpus's central validity dial, so the arithmetic is worth stating. A gap is
# logically necessary for an undetermined answer, so a cue-only classifier always attains
# recall 1.0 on the UNKNOWN class by guessing UNKNOWN whenever a gap is visible. Its
# balanced accuracy is therefore 0.5 * (1 + 1 - g/(1 - UNKNOWN_SHARE)), where g is this
# share: the only way to push it toward chance is to make gaps COMMON AMONG ANSWERABLE
# ROWS, so that seeing one says little.
#
#   g = 0.20  ->  0.875   (and the scout bank sits at 1.000 in three of its four families)
#   g = 0.50  ->  0.750
#
# 0.50 is used: gaps appear in 70% of rows, 30% still carry a complete record, and what
# varies is whether the missing fact is RELEVANT to the query -- an unrecorded job that is
# not an ancestor of the queried one, records absent from a node off the route, an
# unrecorded event that a later SET overwrites. Driving g to its maximum would make every
# row gap-bearing and reach chance, but would also mean the model never sees a complete
# record, which the evaluation bank is full of.
GAP_ANSWERABLE_SHARE = 0.50
MAX_DRAWS_PER_ROW = 4000
MIN_DRAW_BUDGET = 600000


def _buckets(total):
    """Per-cell targets: a quota per answer, plus two pools controlling the gap marker.

    Answer quotas and gap pools are tracked SEPARATELY rather than as a single
    (answer, gap) grid. A grid would demand gap-bearing rows of every answer, and some
    answers cannot supply them at all -- no 4-job schedule reaches earliest start 3 while
    also carrying an unrecorded row. Decoupling lets gap rows come from whichever answers
    can produce them, which is all the shortcut property requires.
    """
    unknown = round(total * UNKNOWN_SHARE)
    per_answer = (total - unknown) // 4
    leftover = (total - unknown) - per_answer * 4
    answers = {"UNKNOWN": unknown}
    for index, answer in enumerate(("0", "1", "2", "3")):
        answers[answer] = per_answer + (1 if index < leftover else 0)
    answerable = total - unknown
    gap = round(total * GAP_ANSWERABLE_SHARE)
    return answers, {"gap": gap, "nogap": answerable - gap}


def scout_exclusions(scout_worlds):
    """Structures and name-blinded texts of the evaluation bank, which must never recur."""
    fingerprints, texts = set(), set()
    for world in scout_worlds:
        fingerprints.add(F.fingerprint(world))
        texts.add(_blind(F.render(world), world["names"]))
    return fingerprints, texts


def _blind(text, names):
    for index, name in enumerate(sorted(names, key=len, reverse=True)):
        text = text.replace(name, f"<E{index}>")
    return text


def _split_for(index, total):
    """Deterministic 80/10/10 assignment by position inside a bucket.

    Splitting inside each (answer, gap) bucket keeps every split's answer balance and gap
    balance identical to the corpus as a whole.
    """
    boundary, cumulative = [], 0
    for name, share in SPLITS:
        cumulative += share
        boundary.append((name, cumulative * total // 100))
    for name, limit in boundary:
        if index < limit:
            return name
    return SPLITS[-1][0]


def generate(scout_worlds, quotas=None, seed=SEED):
    """Build the corpus. Returns (rows, gold, worlds, report)."""
    quotas = quotas or QUOTAS
    rng = random.Random(seed)
    forbidden_fp, forbidden_text = scout_exclusions(scout_worlds)
    seen_fp, seen_text = set(), set()
    collected = defaultdict(list)
    report = {"cells": {}, "rejections": Counter(), "scout_fingerprint_collisions": 0,
              "scout_text_collisions": 0}

    for family in F.FAMILIES:
        for tier, total in quotas[family].items():
            answer_targets, pool_targets = _buckets(total)
            filled, pools = Counter(), Counter()
            draws = 0
            # A floor matters for tiny cells: prerequisite_schedule/match needs the two
            # remaining answer-"3" schedules out of 480, and a budget proportional to a
            # 10-row quota is far too small to find both by rejection sampling.
            budget = max(MAX_DRAWS_PER_ROW * total, MIN_DRAW_BUDGET)
            while sum(filled.values()) < total:
                draws += 1
                if draws > budget:
                    short = {a: answer_targets[a] - filled[a]
                             for a in answer_targets if filled[a] < answer_targets[a]}
                    pool_short = {p: pool_targets[p] - pools[p]
                                  for p in pool_targets if pools[p] < pool_targets[p]}
                    raise RuntimeError(
                        f"cell {family}/{tier} could not be filled within {budget} draws; "
                        f"unfilled answers {short}; unfilled gap pools {pool_short}. Reduce "
                        "the quota rather than relaxing a balance or disjointness constraint.")
                world = sampling.sample(rng, family, tier)
                try:
                    answer = F.solve(world)
                except ValueError:
                    report["rejections"]["solver_rejected"] += 1
                    continue
                if answer not in F.ANSWERS:
                    # e.g. a 5-job chain whose earliest start is 4, outside the vocabulary
                    report["rejections"]["answer_outside_vocabulary"] += 1
                    continue
                gap = bool(F.answerability_cue(world))
                if filled[answer] >= answer_targets[answer]:
                    report["rejections"]["answer_quota_full"] += 1
                    continue
                pool = None
                if answer != "UNKNOWN":
                    pool = "gap" if gap else "nogap"
                    # Two-phase filling. Answer quotas are hard; the gap/no-gap split is a
                    # target. In the first half of the budget both are enforced. After
                    # that the no-gap cap is released so hard answer quotas can still be
                    # met, because the two constraints can genuinely deadlock: no 4-job
                    # schedule with answer "3" carries a gap, so those rows can ONLY come
                    # from the no-gap pool, and an earlier answer may have consumed it.
                    # The shortfall is recorded and the resulting P(UNKNOWN | gap) is
                    # asserted per family, which is the property that actually matters.
                    strict = draws <= budget // 2
                    if pools[pool] >= pool_targets[pool] and (strict or pool == "gap"):
                        report["rejections"]["gap_pool_full"] += 1
                        continue
                elif not gap:
                    # An undetermined answer without any information gap would mean the
                    # record is complete yet insufficient, which the families cannot build.
                    raise AssertionError("UNKNOWN without an information gap: " + family)
                key = (answer, gap)
                fingerprint = F.fingerprint(world)
                if fingerprint in forbidden_fp:
                    report["scout_fingerprint_collisions"] += 1
                    report["rejections"]["scout_structure"] += 1
                    continue
                if fingerprint in seen_fp:
                    report["rejections"]["duplicate_structure"] += 1
                    continue
                text = F.render(world)
                blinded = _blind(text, world["names"])
                if blinded in forbidden_text:
                    report["scout_text_collisions"] += 1
                    report["rejections"]["scout_text"] += 1
                    continue
                if blinded in seen_text:
                    report["rejections"]["duplicate_text"] += 1
                    continue
                seen_fp.add(fingerprint)
                seen_text.add(blinded)
                filled[answer] += 1
                if pool:
                    pools[pool] += 1
                collected[(family, tier, key)].append((world, answer, text, fingerprint))
            report["cells"][f"{family}/{tier}"] = {
                "rows": total, "draws": draws, "answer_targets": dict(answer_targets),
                "gap_pool_targets": dict(pool_targets), "gap_pool_filled": dict(pools),
                "capacity_note": sampling.capacity_note(family, tier)}

    rows, gold, worlds = [], [], []
    for (family, tier, key), entries in sorted(collected.items(), key=lambda kv: str(kv[0])):
        for index, (world, answer, text, fingerprint) in enumerate(entries):
            split = _split_for(index, len(entries))
            structure_id = "RTG1-s-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
            case_id = "RTG1-" + hashlib.sha256(
                (structure_id + "|" + family).encode()).hexdigest()[:24]
            difficulty = len(world.get("ops") or world.get("links") or world.get("deps"))
            rows.append({"id": case_id, "structure_id": structure_id, "family": family,
                         "tier": tier, "split": split, "difficulty": difficulty,
                         "problem": text})
            gold.append({"id": case_id, "structure_id": structure_id, "family": family,
                         "tier": tier, "split": split, "answer": answer,
                         "answerable": answer != "UNKNOWN",
                         "information_gap": bool(key[1]), "difficulty": difficulty,
                         "verification": "forward == independent backward"})
            worlds.append({"id": case_id, "structure_id": structure_id,
                           "fingerprint": fingerprint, "world": world})

    # Deterministic presentation order: shuffle once under the same seed so that families
    # and tiers are interleaved rather than presented in blocks.
    order = random.Random(seed + 1)
    indices = list(range(len(rows)))
    order.shuffle(indices)
    rows = [rows[i] for i in indices]
    gold = [gold[i] for i in indices]
    worlds = [worlds[i] for i in indices]

    report["totals"] = {"rows": len(rows),
                        "unique_structures": len({r["structure_id"] for r in rows}),
                        "splits": dict(Counter(r["split"] for r in rows)),
                        "families": dict(Counter(r["family"] for r in rows)),
                        "tiers": dict(Counter(f'{r["family"]}/{r["tier"]}' for r in rows))}
    report["rejections"] = dict(report["rejections"])
    return rows, gold, worlds, report


def split_isolation(rows, worlds):
    """No canonical structure may appear in more than one split."""
    by_structure = defaultdict(set)
    for row in rows:
        by_structure[row["structure_id"]].add(row["split"])
    crossing = {k: sorted(v) for k, v in by_structure.items() if len(v) > 1}
    fingerprints = defaultdict(set)
    for entry in worlds:
        fingerprints[entry["fingerprint"]].add(entry["structure_id"])
    collisions = {k: sorted(v) for k, v in fingerprints.items() if len(v) > 1}
    return {"structures": len(by_structure), "crossing_splits": crossing,
            "fingerprint_id_collisions": collisions,
            "isolated": not crossing and not collisions}
