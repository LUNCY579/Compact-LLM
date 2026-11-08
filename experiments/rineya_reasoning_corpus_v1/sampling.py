"""Tiered world samplers.

The `match` tier reproduces the scout generator's own parameter ranges exactly, so a
`match` row using only ADD/XOR renders a byte-identical preamble to a scout item. `easy`
sits below that range as curriculum, `hard` above it as generalisation. For
`prerequisite_schedule` the tiers are job counts, because only 480 distinct 4-job
instances exist in the entire scout construction.

Samplers are deliberately biased toward constructions that BREAK the answerability cue
the scout bank made perfectly predictive; `families.answerability_cue` and the audit
measure how well that worked rather than assuming it.
"""
from __future__ import annotations

from . import families as F

# (family, tier) -> complexity range. "match" always equals the scout's own range.
TIERS = {
    "register_machine": {"easy": (1, 2), "match": (3, 6), "hard": (7, 9)},
    "counterfactual_state": {"easy": (2, 2), "match": (3, 6), "hard": (7, 9)},
    "prerequisite_schedule": {"easy": (3, 3), "match": (4, 4), "wide5": (5, 5), "wide6": (6, 6)},
    "evidence_chain": {"easy": (1, 2), "match": (3, 3), "hard": (4, 5)},
}
# Scout-identical surface: only the operation set the scout itself uses.
SCOUT_OPS = ("add", "xor")
EXTENDED_OPS = ("add", "xor", "set", "unrec")


def _ops(rng, count, extended):
    """Operation list, with SET and NOT RECORDED barred from the final position.

    The rendered question line carries no digit, so the last digit in the text is the final
    operand. A trailing `SET k` would therefore make the last digit BE the answer: measured
    at 0.497 exact-answer accuracy against a 0.251 floor before this restriction. A trailing
    NOT RECORDED would likewise make "the listing ends with NOT RECORDED" imply UNKNOWN.
    ADD and XOR carry no such leak -- the preceding value is near-uniform over Z4, so the
    result is near-uniform whatever the operand is.
    """
    kinds = EXTENDED_OPS if extended else SCOUT_OPS
    ops = []
    for index in range(count):
        choices = kinds
        if index == count - 1:
            choices = tuple(k for k in kinds if k not in ("set", "unrec")) or SCOUT_OPS
        kind = rng.choice(choices)
        ops.append([kind, None] if kind == "unrec" else [kind, rng.randrange(4)])
    return ops


def register(rng, tier, family="register_machine"):
    low, high = TIERS[family][tier]
    count = rng.randint(low, high)
    # Extended operations exist to decorrelate answerability from the "not recorded"
    # marker; roughly half of rows keep the scout's exact operation vocabulary.
    extended = rng.random() < .5
    w = {"family": family, "tier": tier,
         "initial": None if rng.random() < .4 else rng.randrange(4),
         "ops": _ops(rng, count, extended), "skip": None}
    if family == "counterfactual_state":
        w["skip"] = rng.randrange(len(w["ops"]))
    w["names"] = F.names(rng, 5)
    return w


def counterfactual(rng, tier):
    return register(rng, tier, "counterfactual_state")


def schedule(rng, tier):
    low, high = TIERS["prerequisite_schedule"][tier]
    jobs = rng.randint(low, high)
    deps = [[j for j in range(i) if rng.random() < .55] for i in range(jobs)]
    # `missing` is never job 0: it has no possible prerequisites, so an unrecorded row
    # there would carry no information. This matches the scout.
    missing = None if rng.random() < .45 else rng.randrange(1, jobs)
    return {"family": "prerequisite_schedule", "tier": tier, "deps": deps,
            "query": rng.randrange(jobs), "missing": missing,
            "names": F.names(rng, jobs)}


def evidence(rng, tier):
    low, high = TIERS["evidence_chain"][tier]
    length = rng.randint(low, high)
    nodes = length + 1 + rng.randint(1, 2)          # chain plus detached distractors
    order = rng.sample(range(nodes), nodes)
    chain = order[:length + 1]
    links = [[chain[i], chain[i + 1]] for i in range(length)]
    # A branch makes the ROUTE ambiguous while every record is present, which breaks the
    # "a missing record is the only route to UNKNOWN" regularity in the scout bank.
    spare = list(order[length + 1:])
    branch_end = None
    if spare and rng.random() < .45:
        branch_end = spare.pop()
        links.append([rng.choice(chain[:-1]), branch_end])
    terminal_nodes = {b for _, b in links} - {a for a, _ in links}
    # Two or three dated records per node. The extra time point is not cosmetic: the answer
    # is one of the values printed in the text, so "read the last printed value" scores
    # about 1/|facts| + (1 - 1/|facts|)/4. With two times the list averaged 7.9 entries and
    # that heuristic reached 0.379 against a 0.251 floor. Lengthening the list dilutes it,
    # and more superseding records is also a genuine increase in task difficulty. Node COUNT
    # is deliberately not increased instead: the backward solver enumerates node
    # permutations, so its cost grows factorially.
    times = (1, 2) if rng.random() < .4 else (1, 2, 3)
    facts = []
    for node in range(nodes):
        # Dropping records for a node OFF the route leaves the answer determined; dropping
        # them for a node the route ends at does not. Both cases are generated.
        if rng.random() < (.30 if node in terminal_nodes else .22):
            continue
        for t in times:
            if rng.random() < .10:
                continue
            facts.append([node, t, rng.randrange(4)])
    if branch_end is not None and rng.random() < .35:
        # Two independent records that happen to agree: the route is ambiguous but the
        # destination value is not. These are the rows that stop "ambiguous route" from
        # implying UNKNOWN. Records missing from a node off the route now supply most
        # gap-bearing answerable rows, so this rate no longer has to be high.
        latest = {}
        for node, t, v in facts:
            if node not in latest or t > latest[node][0]:
                latest[node] = (t, v)
        others = [n for n in terminal_nodes if n != branch_end and n in latest]
        if others and branch_end in latest:
            target = latest[others[0]][1]
            keep = latest[branch_end][0]
            facts = [row for row in facts if not (row[0] == branch_end and row[1] == keep)]
            facts.append([branch_end, keep, target])
    rng.shuffle(facts)
    return {"family": "evidence_chain", "tier": tier, "links": links,
            "query": chain[0], "facts": facts, "names": F.names(rng, nodes)}


SAMPLERS = {"register_machine": register, "counterfactual_state": counterfactual,
            "prerequisite_schedule": schedule, "evidence_chain": evidence}


def sample(rng, family, tier):
    return SAMPLERS[family](rng, tier)


def capacity_note(family, tier):
    """Structural capacity, where it is small enough to bind the corpus size.

    Only the 4-job `prerequisite_schedule` cell is genuinely capped; the figure is
    verified by exhaustive enumeration in tests/test_capacity.py.
    """
    if family == "prerequisite_schedule":
        return {"easy": 42, "match": 480, "wide5": 9920, "wide6": 387072}.get(tier)
    return None
