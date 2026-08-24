"""Corrected, balanced R7 rule/exception/override benchmark.

R7 v1 is intentionally left frozen.  This module defines a distinct generator
and a rendered-text interpreter for R7 v2.  The generator records an intended
answer in its structured oracle, while the interpreter sees only the rendered
question and program segments and executes a fixed-point rule engine.

Every valid case contains two competing override branches but activates only
one for the queried unit.  The active override is the opposite of a fired base
decision, so answering requires both forward chaining and override resolution.
The annotated evidence is a deletion-minimal proof of that final outcome.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final

from ..core.randomness import stable_rng
from .advanced import CONTEXT_LAYOUTS, ContextLayout, place_evidence
from .schema import SyntheticCase, SyntheticSegment

R7_V2_GENERATOR_VERSION: Final = "r7-v2.0.0"
R7_V2_POLICY: Final = (
    "POLICY: repeatedly apply RULE lines. A DECISION fires when its condition is present. "
    "An OVERRIDE fires only when its condition and named lower DECISION both fire. "
    "For the queried unit, the highest-priority fired status wins."
)

_QUESTION_RE = re.compile(
    r"Apply the policy to all FACT, RULE, DECISION, and OVERRIDE lines\. "
    r"Is the final status of unit ([A-Z]+) READY\? Return YES or NO\."
)
_FACT_RE = re.compile(r"FACT: unit ([A-Z]+) has ([A-Z]+)\.")
_RULE_RE = re.compile(
    r"RULE ([A-Z0-9]+): if unit ([A-Z]+) has ([A-Z]+)"
    r"(?: and unit ([A-Z]+) has ([A-Z]+))?, then unit ([A-Z]+) gains ([A-Z]+)\."
)
_DECISION_RE = re.compile(
    r"DECISION ([A-Z0-9]+) PRIORITY (\d+): if unit ([A-Z]+) has ([A-Z]+), "
    r"then status of unit ([A-Z]+) is (READY|NOT READY)\."
)
_OVERRIDE_RE = re.compile(
    r"OVERRIDE ([A-Z0-9]+) PRIORITY (\d+) OVER ([A-Z0-9]+): "
    r"if unit ([A-Z]+) has ([A-Z]+), then status of unit ([A-Z]+) is (READY|NOT READY)\."
)
_STATUS_RE = re.compile(r"status of unit [A-Z]+ is (READY|NOT READY)\.")

Atom = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _ForwardRule:
    rule_id: str
    antecedents: tuple[Atom, ...]
    consequent: Atom


@dataclass(frozen=True, slots=True)
class _Decision:
    rule_id: str
    priority: int
    antecedent: Atom
    unit: str
    status: str


@dataclass(frozen=True, slots=True)
class _Override:
    rule_id: str
    priority: int
    over_rule_id: str
    antecedent: Atom
    unit: str
    status: str


@dataclass(frozen=True, slots=True)
class R7V2Resolution:
    """Auditable result of the independent rendered-program interpreter."""

    answer: str
    final_status: str
    query_unit: str
    fired_forward_rule_ids: tuple[str, ...]
    fired_decision_ids: tuple[str, ...]
    fired_override_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Vocabulary:
    target: str
    decoy: str
    source: str
    condition: str
    relay: str
    permit: str
    route_a: str
    key_a: str
    seal_a: str
    route_b: str
    key_b: str
    seal_b: str
    decoy_signal: str


_VOCABULARIES: Final[tuple[_Vocabulary, ...]] = (
    _Vocabulary(
        "CEDAR",
        "MAPLE",
        "AMBER",
        "CLEAR",
        "BRONZE",
        "SILVER",
        "NORTH",
        "IVORY",
        "JADE",
        "SOUTH",
        "ONYX",
        "PEARL",
        "LANTERN",
    ),
    _Vocabulary(
        "ORION",
        "HARBOR",
        "COPPER",
        "CALM",
        "BIRCH",
        "QUARTZ",
        "EAST",
        "CORAL",
        "OPAL",
        "WEST",
        "FLINT",
        "TOPAZ",
        "BEACON",
    ),
    _Vocabulary(
        "LYRA",
        "FJORD",
        "CRIMSON",
        "OPEN",
        "WILLOW",
        "CRYSTAL",
        "UPPER",
        "MARBLE",
        "AZURE",
        "LOWER",
        "SLATE",
        "GARNET",
        "TORCH",
    ),
    _Vocabulary(
        "ATLAS",
        "GROVE",
        "GOLD",
        "STABLE",
        "PINE",
        "GLASS",
        "INNER",
        "CHALK",
        "MINT",
        "OUTER",
        "SHALE",
        "RUBY",
        "SIGNAL",
    ),
)


def _status(ready: bool) -> str:
    return "READY" if ready else "NOT READY"


def _answer(status: str) -> str:
    return "YES" if status == "READY" else "NO"


def _r7_v2_filler(index: int, seed: int) -> SyntheticSegment:
    labels = ("archive", "bridge", "circuit", "delta", "ember", "forest", "garden", "harbor")
    label = labels[(seed * 7 + index * 5) % len(labels)]
    return SyntheticSegment(
        segment_id=f"X{index:03d}",
        text=f"Routine bulletin {index:02d} records neutral tag {label}-{(seed + 31 * index) % 997:03d}.",
        role="filler",
    )


def _actual_depth(segments: Sequence[SyntheticSegment], evidence_ids: set[str]) -> float:
    positions = [
        index for index, segment in enumerate(segments) if segment.segment_id in evidence_ids
    ]
    return float(sum(positions) / len(positions) / max(1, len(segments) - 1))


def _render_program(
    seed: int,
) -> tuple[
    list[SyntheticSegment],
    list[SyntheticSegment],
    str,
    dict[str, object],
]:
    vocabulary = _VOCABULARIES[(seed // 4) % len(_VOCABULARIES)]
    final_ready = seed % 2 == 0
    base_ready = not final_ready
    active_branch = "A" if (seed // 2) % 2 == 0 else "B"
    branch_a_ready = final_ready if active_branch == "A" else base_ready
    branch_b_ready = final_ready if active_branch == "B" else base_ready

    common = [
        SyntheticSegment("P00", R7_V2_POLICY, "evidence"),
        SyntheticSegment(
            "P01", f"FACT: unit {vocabulary.target} has {vocabulary.source}.", "evidence"
        ),
        SyntheticSegment(
            "P02",
            f"FACT: unit {vocabulary.target} has {vocabulary.condition}.",
            "evidence",
        ),
        SyntheticSegment(
            "P03",
            f"RULE F10: if unit {vocabulary.target} has {vocabulary.source}, "
            f"then unit {vocabulary.target} gains {vocabulary.relay}.",
            "evidence",
        ),
        SyntheticSegment(
            "P04",
            f"RULE F20: if unit {vocabulary.target} has {vocabulary.relay} "
            f"and unit {vocabulary.target} has {vocabulary.condition}, "
            f"then unit {vocabulary.target} gains {vocabulary.permit}.",
            "evidence",
        ),
        SyntheticSegment(
            "P05",
            f"DECISION D30 PRIORITY 10: if unit {vocabulary.target} has {vocabulary.permit}, "
            f"then status of unit {vocabulary.target} is {_status(base_ready)}.",
            "evidence",
        ),
    ]

    branch_specs = {
        "A": (
            "P06",
            "P07",
            "P08",
            "P09",
            "A40",
            "A50",
            "A60",
            vocabulary.route_a,
            vocabulary.key_a,
            vocabulary.seal_a,
            branch_a_ready,
        ),
        "B": (
            "P10",
            "P11",
            "P12",
            "P13",
            "B40",
            "B50",
            "B60",
            vocabulary.route_b,
            vocabulary.key_b,
            vocabulary.seal_b,
            branch_b_ready,
        ),
    }
    branch_rows: dict[str, list[SyntheticSegment]] = {}
    for branch, spec in branch_specs.items():
        fact_id, first_id, second_id, override_id, r1_id, r2_id, r3_id, route, key, seal, ready = (
            spec
        )
        fact_unit = vocabulary.target if branch == active_branch else vocabulary.decoy
        role = "evidence" if branch == active_branch else "distractor"
        branch_rows[branch] = [
            SyntheticSegment(fact_id, f"FACT: unit {fact_unit} has {route}.", role),
            SyntheticSegment(
                first_id,
                f"RULE {r1_id}: if unit {vocabulary.target} has {route}, "
                f"then unit {vocabulary.target} gains {key}.",
                role,
            ),
            SyntheticSegment(
                second_id,
                f"RULE {r2_id}: if unit {vocabulary.target} has {key} "
                f"and unit {vocabulary.target} has {vocabulary.permit}, "
                f"then unit {vocabulary.target} gains {seal}.",
                role,
            ),
            SyntheticSegment(
                override_id,
                f"OVERRIDE {r3_id} PRIORITY 20 OVER D30: if unit {vocabulary.target} has {seal}, "
                f"then status of unit {vocabulary.target} is {_status(bool(ready))}.",
                role,
            ),
        ]

    # This unrelated fired decision balances READY/NOT READY surface counts in
    # every prompt.  It cannot affect the queried unit.
    decoy_rows = [
        SyntheticSegment(
            "P14",
            f"FACT: unit {vocabulary.decoy} has {vocabulary.decoy_signal}.",
            "distractor",
        ),
        SyntheticSegment(
            "P15",
            f"DECISION Q90 PRIORITY 5: if unit {vocabulary.decoy} has {vocabulary.decoy_signal}, "
            f"then status of unit {vocabulary.decoy} is {_status(final_ready)}.",
            "distractor",
        ),
        SyntheticSegment(
            "P16",
            f"DRAFT NOTE: unit {vocabulary.decoy} mentions {vocabulary.permit}, but draft notes are not FACT lines.",
            "distractor",
        ),
    ]

    active_rows = branch_rows[active_branch]
    inactive_rows = branch_rows["B" if active_branch == "A" else "A"]
    evidence = [*common, *active_rows]
    other = [*inactive_rows, *decoy_rows]
    question = (
        "Apply the policy to all FACT, RULE, DECISION, and OVERRIDE lines. "
        f"Is the final status of unit {vocabulary.target} READY? Return YES or NO."
    )
    oracle: dict[str, object] = {
        "target_unit": vocabulary.target,
        "decoy_unit": vocabulary.decoy,
        "base_status": _status(base_ready),
        "active_branch": active_branch,
        "intended_final_status": _status(final_ready),
        "intended_answer": _answer(_status(final_ready)),
        "minimal_proof_segment_ids": [segment.segment_id for segment in evidence],
        "reasoning_profile": "two_forward_chains_plus_single_named_override",
    }
    return evidence, other, question, oracle


def generate_r7_v2_case(
    seed: int,
    evidence_depth: float,
    *,
    filler_count: int = 24,
    layout: ContextLayout = "clustered",
) -> SyntheticCase:
    """Generate one deterministic corrected R7 case."""

    if layout not in CONTEXT_LAYOUTS:
        raise ValueError(f"unknown context layout: {layout!r}")
    if not 0.0 <= evidence_depth <= 1.0:
        raise ValueError("evidence depth must be in [0, 1]")
    if filler_count < 0:
        raise ValueError("filler_count must be non-negative")
    evidence, program_distractors, question, oracle = _render_program(seed)
    other = [*program_distractors, *(_r7_v2_filler(index, seed) for index in range(filler_count))]
    stable_rng(seed, "R7-v2", "other-order").shuffle(other)
    segments = place_evidence(
        evidence,
        other,
        layout=layout,
        depth=evidence_depth,
        seed=seed,
        family="R7-v2",
    )
    evidence_ids = tuple(segment.segment_id for segment in evidence)
    case = SyntheticCase(
        case_id=(
            f"R7v2-s{seed:06d}-d{round(evidence_depth * 100):03d}-l{layout.replace('_', '-')}"
        ),
        family="R7",
        seed=seed,
        requested_evidence_depth=evidence_depth,
        actual_evidence_depth=_actual_depth(segments, set(evidence_ids)),
        segments=segments,
        question=question,
        answer=str(oracle["intended_answer"]),
        evidence_segment_ids=evidence_ids,
        distractor_segment_ids=tuple(segment.segment_id for segment in program_distractors),
        stale_segment_ids=(),
        oracle=oracle,
        generator_version=R7_V2_GENERATOR_VERSION,
        metadata={
            "benchmark_version": R7_V2_GENERATOR_VERSION,
            "cluster_id": f"R7v2-s{seed:06d}",
            "context_layout": layout,
            "filler_count": filler_count,
            "latent_seed": seed,
            "synthetic": True,
            "active_branch": oracle["active_branch"],
            "minimal_proof_size": len(evidence_ids),
        },
    )
    validate_r7_v2_case(case)
    return case


def solve_r7_v2_oracle(case: SyntheticCase) -> str:
    """Read the generator's intended answer without invoking the text parser."""

    if case.generator_version != R7_V2_GENERATOR_VERSION:
        raise ValueError("not an R7 v2 case")
    answer = str(case.oracle["intended_answer"])
    if answer not in {"YES", "NO"}:
        raise ValueError("invalid structured R7 v2 answer")
    return answer


def interpret_r7_v2_segments(
    segments: Sequence[SyntheticSegment],
    question: str,
) -> R7V2Resolution:
    """Parse and execute only the rendered R7 v2 program.

    This function does not inspect a case seed, answer, oracle, role annotation,
    or segment identifier.  Segment order is semantically irrelevant.
    """

    texts = [segment.text for segment in segments]
    if texts.count(R7_V2_POLICY) != 1:
        raise ValueError("R7 v2 requires exactly one policy line")
    question_match = _QUESTION_RE.fullmatch(question)
    if question_match is None:
        raise ValueError("cannot parse R7 v2 query unit")
    query_unit = question_match.group(1)

    facts: set[Atom] = set()
    rules: list[_ForwardRule] = []
    decisions: list[_Decision] = []
    overrides: list[_Override] = []
    for text in texts:
        if fact_match := _FACT_RE.fullmatch(text):
            facts.add((fact_match.group(1), fact_match.group(2)))
            continue
        if rule_match := _RULE_RE.fullmatch(text):
            antecedents: list[Atom] = [(rule_match.group(2), rule_match.group(3))]
            if rule_match.group(4) is not None and rule_match.group(5) is not None:
                antecedents.append((rule_match.group(4), rule_match.group(5)))
            rules.append(
                _ForwardRule(
                    rule_id=rule_match.group(1),
                    antecedents=tuple(antecedents),
                    consequent=(rule_match.group(6), rule_match.group(7)),
                )
            )
            continue
        if decision_match := _DECISION_RE.fullmatch(text):
            condition_unit = decision_match.group(3)
            output_unit = decision_match.group(5)
            if condition_unit != output_unit:
                raise ValueError("R7 v2 decision changes unit identity")
            decisions.append(
                _Decision(
                    rule_id=decision_match.group(1),
                    priority=int(decision_match.group(2)),
                    antecedent=(condition_unit, decision_match.group(4)),
                    unit=output_unit,
                    status=decision_match.group(6),
                )
            )
            continue
        if override_match := _OVERRIDE_RE.fullmatch(text):
            condition_unit = override_match.group(4)
            output_unit = override_match.group(6)
            if condition_unit != output_unit:
                raise ValueError("R7 v2 override changes unit identity")
            overrides.append(
                _Override(
                    rule_id=override_match.group(1),
                    priority=int(override_match.group(2)),
                    over_rule_id=override_match.group(3),
                    antecedent=(condition_unit, override_match.group(5)),
                    unit=output_unit,
                    status=override_match.group(7),
                )
            )

    if not facts or not rules or not decisions:
        raise ValueError("R7 v2 rendered program is incomplete")
    identifiers = [rule.rule_id for rule in rules] + [rule.rule_id for rule in decisions]
    identifiers += [rule.rule_id for rule in overrides]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("R7 v2 rendered program has duplicate rule identifiers")

    closure = set(facts)
    fired_forward: list[str] = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.rule_id in fired_forward or not set(rule.antecedents) <= closure:
                continue
            fired_forward.append(rule.rule_id)
            if rule.consequent not in closure:
                closure.add(rule.consequent)
                changed = True

    fired_decisions = [rule for rule in decisions if rule.antecedent in closure]
    fired_decision_ids = {rule.rule_id for rule in fired_decisions}
    fired_overrides = [
        rule
        for rule in overrides
        if rule.antecedent in closure and rule.over_rule_id in fired_decision_ids
    ]
    overridden_ids = {rule.over_rule_id for rule in fired_overrides}
    candidates: list[tuple[int, str, str]] = [
        (rule.priority, rule.rule_id, rule.status)
        for rule in fired_decisions
        if rule.unit == query_unit and rule.rule_id not in overridden_ids
    ]
    candidates.extend(
        (rule.priority, rule.rule_id, rule.status)
        for rule in fired_overrides
        if rule.unit == query_unit
    )
    if not candidates:
        raise ValueError("R7 v2 query has no resolved decision")
    highest_priority = max(priority for priority, _, _ in candidates)
    winners = [
        (rule_id, status)
        for priority, rule_id, status in candidates
        if priority == highest_priority
    ]
    statuses = {status for _, status in winners}
    if len(statuses) != 1:
        raise ValueError("R7 v2 query has conflicting highest-priority decisions")
    final_status = statuses.pop()
    return R7V2Resolution(
        answer=_answer(final_status),
        final_status=final_status,
        query_unit=query_unit,
        fired_forward_rule_ids=tuple(sorted(fired_forward)),
        fired_decision_ids=tuple(sorted(rule.rule_id for rule in fired_decisions)),
        fired_override_ids=tuple(sorted(rule.rule_id for rule in fired_overrides)),
    )


def solve_r7_v2_from_segments(case: SyntheticCase) -> str:
    """Return the independent rendered-program answer for one R7 v2 case."""

    return interpret_r7_v2_segments(case.segments, case.question).answer


def _validate_deletion_minimal_proof(case: SyntheticCase) -> None:
    by_id = {segment.segment_id: segment for segment in case.segments}
    proof = [by_id[segment_id] for segment_id in case.evidence_segment_ids]
    proof_answer = interpret_r7_v2_segments(proof, case.question).answer
    if proof_answer != case.answer:
        raise ValueError("annotated R7 v2 evidence is not sufficient")
    for removed in case.evidence_segment_ids:
        ablated = [segment for segment in proof if segment.segment_id != removed]
        try:
            ablated_answer = interpret_r7_v2_segments(ablated, case.question).answer
        except ValueError:
            continue
        if ablated_answer == case.answer:
            raise ValueError(f"R7 v2 evidence is not deletion-minimal; {removed} is redundant")


def validate_r7_v2_case(case: SyntheticCase) -> dict[str, object]:
    """Enforce the corrected R7 v2 design and exact-proof invariants."""

    if case.family != "R7" or case.generator_version != R7_V2_GENERATOR_VERSION:
        raise ValueError("invalid R7 v2 family or generator version")
    ids = [segment.segment_id for segment in case.segments]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate R7 v2 segment identifiers")
    if not set(case.evidence_segment_ids) <= set(ids):
        raise ValueError("R7 v2 evidence annotation references a missing segment")
    if any(segment_id.startswith(("E", "D")) for segment_id in ids):
        raise ValueError("R7 v2 segment IDs must not leak evidence/distractor roles")
    expected_proof = tuple(str(value) for value in case.oracle["minimal_proof_segment_ids"])
    if case.evidence_segment_ids != expected_proof:
        raise ValueError("R7 v2 evidence annotation differs from the intended proof")
    oracle_answer = solve_r7_v2_oracle(case)
    rendered = interpret_r7_v2_segments(case.segments, case.question)
    if rendered.answer != oracle_answer or case.answer != oracle_answer:
        raise ValueError("R7 v2 structured and rendered solvers disagree")
    if len(rendered.fired_forward_rule_ids) < 4 or len(rendered.fired_override_ids) != 1:
        raise ValueError("R7 v2 does not exercise the required forward-chain/override path")
    status_counts = Counter(
        match.group(1) for segment in case.segments if (match := _STATUS_RE.search(segment.text))
    )
    if status_counts != {"READY": 2, "NOT READY": 2}:
        raise ValueError("R7 v2 READY/NOT READY surface wording is not balanced")
    _validate_deletion_minimal_proof(case)
    return {
        "case_id": case.case_id,
        "status": "PASS",
        "answer": case.answer,
        "active_branch": case.oracle["active_branch"],
        "minimal_proof_size": len(case.evidence_segment_ids),
        "fired_forward_rules": len(rendered.fired_forward_rule_ids),
        "fired_overrides": len(rendered.fired_override_ids),
    }


def relayout_r7_v2_case(
    case: SyntheticCase,
    *,
    layout: ContextLayout,
    evidence_depth: float,
) -> SyntheticCase:
    """Move the frozen proof without altering program semantics."""

    by_id = {segment.segment_id: segment for segment in case.segments}
    evidence = [by_id[segment_id] for segment_id in case.evidence_segment_ids]
    evidence_ids = set(case.evidence_segment_ids)
    other = [segment for segment in case.segments if segment.segment_id not in evidence_ids]
    segments = place_evidence(
        evidence,
        other,
        layout=layout,
        depth=evidence_depth,
        seed=case.seed,
        family="R7-v2",
    )
    updated = replace(
        case,
        case_id=(
            f"R7v2-s{case.seed:06d}-d{round(evidence_depth * 100):03d}-l{layout.replace('_', '-')}"
        ),
        requested_evidence_depth=evidence_depth,
        actual_evidence_depth=_actual_depth(segments, evidence_ids),
        segments=segments,
        metadata={**case.metadata, "context_layout": layout},
    )
    validate_r7_v2_case(updated)
    return updated


def generate_r7_v2_suite(
    seeds: Iterable[int],
    evidence_depths: Iterable[float],
    *,
    layouts: Iterable[ContextLayout] = ("clustered",),
    filler_count: int = 24,
) -> list[SyntheticCase]:
    cases = [
        generate_r7_v2_case(
            int(seed),
            float(depth),
            filler_count=filler_count,
            layout=layout,
        )
        for seed in seeds
        for depth in evidence_depths
        for layout in layouts
    ]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("R7 v2 suite parameters generated duplicate case IDs")
    return cases
