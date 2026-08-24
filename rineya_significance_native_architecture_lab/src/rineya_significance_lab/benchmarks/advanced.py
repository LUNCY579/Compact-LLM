"""Deterministic R1-cardinality and R5-R8 controlled benchmarks.

These generators extend the original R1-R4 calibration set without changing
its frozen bytes.  Every answer is produced twice: once from a structured
oracle and once by parsing only the rendered segments.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..core.randomness import stable_rng
from .schema import SyntheticCase, SyntheticSegment
from .synthetic import solve_case, solve_case_from_segments, validate_case

ADVANCED_FAMILIES = ("R5", "R6", "R7", "R8")
R1_CARDINALITIES = (2, 3, 4, 6, 8)
ContextLayout = Literal[
    "all_early",
    "all_middle",
    "all_late",
    "early_late",
    "early_middle_late",
    "clustered",
    "uniformly_distributed",
    "random",
]
CONTEXT_LAYOUTS: tuple[ContextLayout, ...] = (
    "all_early",
    "all_middle",
    "all_late",
    "early_late",
    "early_middle_late",
    "clustered",
    "uniformly_distributed",
    "random",
)
BuilderResult = tuple[
    list[SyntheticSegment],
    list[SyntheticSegment],
    list[SyntheticSegment],
    str,
    dict[str, Any],
]


def _filler(index: int, seed: int) -> SyntheticSegment:
    labels = ("atlas", "briar", "cinder", "drift", "elm", "fjord", "grove", "hearth")
    label = labels[(seed * 5 + index * 3) % len(labels)]
    return SyntheticSegment(
        segment_id=f"F{index:03d}",
        text=f"Background bulletin {index:02d} contains routine tag {label}-{(seed + 29 * index) % 991:03d}.",
        role="filler",
    )


def _nearest_free(target: int, used: set[int], total: int) -> int:
    for distance in range(total):
        for candidate in (target - distance, target + distance):
            if 0 <= candidate < total and candidate not in used:
                return candidate
    raise RuntimeError("no free context slot remains")


def _layout_positions(
    evidence_count: int,
    other_count: int,
    *,
    layout: ContextLayout,
    depth: float,
    seed: int,
    family: str,
) -> list[int]:
    if evidence_count <= 0:
        raise ValueError("at least one evidence segment is required")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("evidence depth must be in [0, 1]")
    total = evidence_count + other_count
    if layout not in CONTEXT_LAYOUTS:
        raise ValueError(f"unknown context layout: {layout!r}")
    if layout in {"all_early", "all_middle", "all_late", "clustered"}:
        center = {
            "all_early": 0.05,
            "all_middle": 0.50,
            "all_late": 0.95,
            "clustered": depth,
        }[layout]
        start = round(center * max(0, total - evidence_count))
        return list(range(start, start + evidence_count))
    if layout == "early_late":
        targets = np.linspace(0.05, 0.25, (evidence_count + 1) // 2).tolist()
        targets += np.linspace(0.75, 0.95, evidence_count // 2).tolist()
    elif layout == "early_middle_late":
        anchors = (0.08, 0.50, 0.92)
        targets = [anchors[index % len(anchors)] for index in range(evidence_count)]
    elif layout == "uniformly_distributed":
        targets = np.linspace(0.05, 0.95, evidence_count).tolist()
    else:
        rng = stable_rng(seed, family, "context-layout", layout)
        return sorted(int(value) for value in rng.choice(total, size=evidence_count, replace=False))

    used: set[int] = set()
    positions: list[int] = []
    for fraction in targets:
        slot = _nearest_free(round(float(fraction) * max(0, total - 1)), used, total)
        used.add(slot)
        positions.append(slot)
    return sorted(positions)


def place_evidence(
    evidence: Sequence[SyntheticSegment],
    other: Sequence[SyntheticSegment],
    *,
    layout: ContextLayout,
    depth: float,
    seed: int,
    family: str,
) -> tuple[SyntheticSegment, ...]:
    """Place evidence under one of the frozen context-layout controls."""

    slots = set(
        _layout_positions(
            len(evidence),
            len(other),
            layout=layout,
            depth=depth,
            seed=seed,
            family=family,
        )
    )
    evidence_iter = iter(evidence)
    other_iter = iter(other)
    rows = [next(evidence_iter) if index in slots else next(other_iter) for index in range(len(evidence) + len(other))]
    return tuple(rows)


def relayout_case(
    case: SyntheticCase,
    *,
    layout: ContextLayout,
    evidence_depth: float,
) -> SyntheticCase:
    """Re-place a solved case without changing its evidence or answer."""

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
        family=case.family,
    )
    base_id = re.sub(r"-d\d{3}(?:-l[a-z-]+)?$", "", case.case_id)
    updated = replace(
        case,
        case_id=f"{base_id}-d{round(evidence_depth * 100):03d}-l{layout.replace('_', '-')}",
        requested_evidence_depth=evidence_depth,
        actual_evidence_depth=_actual_depth(segments, evidence_ids),
        segments=segments,
        metadata={**case.metadata, "context_layout": layout},
    )
    if updated.family in ADVANCED_FAMILIES:
        validate_advanced_case(updated)
    else:
        validate_case(updated)
    return updated


def _actual_depth(segments: Sequence[SyntheticSegment], evidence_ids: set[str]) -> float:
    indices = [index for index, row in enumerate(segments) if row.segment_id in evidence_ids]
    return float(np.mean(indices) / max(1, len(segments) - 1))


def _r5(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R5")
    vaults = ("ORION", "CEDAR", "HARBOR", "LYRA")
    target = vaults[int(rng.integers(0, len(vaults)))]
    code = int(rng.integers(1000, 9999))
    wrong = code + int(rng.integers(7, 97))
    other = vaults[(vaults.index(target) + 1) % len(vaults)]
    evidence = [
        SyntheticSegment(
            "E00",
            "Authority rule: for vault codes, FINAL VERIFIED records override provisional and quoted records.",
            "evidence",
        ),
        SyntheticSegment("E01", f"FINAL VERIFIED: vault {target} access code is {code}.", "evidence"),
    ]
    stale = [SyntheticSegment("S00", f"Provisional record: vault {target} access code is {wrong}.", "stale")]
    distractors = [
        SyntheticSegment("D00", f"FINAL VERIFIED: vault {other} access code is {code}.", "distractor"),
        SyntheticSegment("D01", f"Quoted rumor: vault {target} access code is {wrong}.", "distractor"),
        SyntheticSegment("D02", f"Question index: 'What is the access code for vault {target}?'", "distractor"),
        SyntheticSegment("D03", f"The {target} vault uses an access code, but this note omits its digits.", "distractor"),
    ]
    question = f"Under the authority rule, what is vault {target}'s access code? Return one integer."
    return evidence, distractors, stale, question, {"target": target, "code": code}


def _r6(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R6")
    a, b, c = (int(value) for value in rng.integers(3, 25, size=3))
    evidence = [
        SyntheticSegment("E00", f"NUMERIC FACT: A = {a}.", "evidence"),
        SyntheticSegment("E01", f"NUMERIC FACT: B = {b}.", "evidence"),
        SyntheticSegment("E02", f"NUMERIC FACT: C = {c}.", "evidence"),
        SyntheticSegment("E03", "RULE ONE: T equals A plus B.", "evidence"),
        SyntheticSegment("E04", "RULE TWO: U equals T minus C.", "evidence"),
        SyntheticSegment("E05", "RULE THREE: RESULT equals two times U.", "evidence"),
    ]
    distractors = [
        SyntheticSegment("D00", f"DRAFT FACT: A = {a + 9}; ignore draft facts.", "distractor"),
        SyntheticSegment("D01", "UNRELATED RULE: W equals B plus C.", "distractor"),
        SyntheticSegment("D02", f"NUMERIC FACT for another case: Z = {a + b + c}.", "distractor"),
    ]
    question = "Apply RULE ONE, RULE TWO, then RULE THREE. What is RESULT? Return one integer."
    return evidence, distractors, [], question, {"a": a, "b": b, "c": c, "result": 2 * (a + b - c)}


def _r7(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R7")
    locked = bool(int(rng.integers(0, 2)))
    evidence = [
        SyntheticSegment("E00", "FACT: unit CEDAR has token A.", "evidence"),
        SyntheticSegment("E01", "RULE 1: token A implies token B.", "evidence"),
        SyntheticSegment("E02", "FACT: unit CEDAR is not on HOLD.", "evidence"),
        SyntheticSegment("E03", "RULE 2: token B together with not HOLD implies token C.", "evidence"),
        SyntheticSegment("E04", "RULE 3: token C implies status READY.", "evidence"),
        SyntheticSegment(
            "E05",
            "OVERRIDE 4: LOCKED implies status NOT READY and overrides RULE 3.",
            "evidence",
        ),
        SyntheticSegment(
            "E06",
            f"FACT: unit CEDAR is {'LOCKED' if locked else 'not LOCKED'}.",
            "evidence",
        ),
    ]
    distractors = [
        SyntheticSegment("D00", "RULE 8: token A implies status READY for unit MAPLE only.", "distractor"),
        SyntheticSegment("D01", "FACT: unit MAPLE is LOCKED.", "distractor"),
        SyntheticSegment("D02", "DRAFT RULE: HOLD might imply READY; this draft has no authority.", "distractor"),
    ]
    question = "After forward rule application and overrides, is unit CEDAR READY? Return YES or NO."
    return evidence, distractors, [], question, {"locked": locked, "ready": not locked}


def _r8(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R8")
    operation_count = int(rng.integers(7, 12))
    stack: list[int] = []
    operations: list[list[str | int | None]] = []
    evidence: list[SyntheticSegment] = []
    for index in range(operation_count):
        pop_allowed = bool(stack) and index > 2
        do_pop = pop_allowed and bool(int(rng.integers(0, 4)) == 0)
        if do_pop:
            stack.pop()
            operations.append(["POP", None])
            text = f"STACK A step {index:02d}: POP."
        else:
            value = int(rng.integers(10, 100))
            stack.append(value)
            operations.append(["PUSH", value])
            text = f"STACK A step {index:02d}: PUSH {value}."
        evidence.append(SyntheticSegment(f"E{index:02d}", text, "evidence"))
    distractors = [
        SyntheticSegment("D00", "STACK B step 00: PUSH 999.", "distractor"),
        SyntheticSegment("D01", "STACK B step 01: POP.", "distractor"),
        SyntheticSegment("D02", "A prose note says the word PUSH but is not a numbered stack operation.", "distractor"),
    ]
    question = "Execute STACK A steps in numeric order. Return its final top and size as top,size."
    return evidence, distractors, [], question, {"operations": operations, "top": stack[-1], "size": len(stack)}


def _build_advanced(family: str, seed: int) -> BuilderResult:
    builders = {"R5": _r5, "R6": _r6, "R7": _r7, "R8": _r8}
    try:
        return builders[family](seed)
    except KeyError as exc:
        raise ValueError(f"unknown advanced family {family!r}; expected one of {ADVANCED_FAMILIES}") from exc


def generate_advanced_case(
    family: str,
    seed: int,
    evidence_depth: float,
    *,
    filler_count: int = 24,
    layout: ContextLayout = "clustered",
) -> SyntheticCase:
    key = family.upper()
    evidence, distractors, stale, question, oracle = _build_advanced(key, seed)
    other = [*stale, *distractors, *(_filler(index, seed) for index in range(filler_count))]
    stable_rng(seed, key, "advanced-other-order").shuffle(other)
    segments = place_evidence(
        evidence,
        other,
        layout=layout,
        depth=evidence_depth,
        seed=seed,
        family=key,
    )
    evidence_ids = tuple(row.segment_id for row in evidence)
    layout_tag = layout.replace("_", "-")
    case_id = f"{key}-s{seed:06d}-d{round(evidence_depth * 100):03d}-l{layout_tag}"
    provisional = SyntheticCase(
        case_id=case_id,
        family=key,
        seed=seed,
        requested_evidence_depth=evidence_depth,
        actual_evidence_depth=_actual_depth(segments, set(evidence_ids)),
        segments=segments,
        question=question,
        answer="",
        evidence_segment_ids=evidence_ids,
        distractor_segment_ids=tuple(row.segment_id for row in distractors),
        stale_segment_ids=tuple(row.segment_id for row in stale),
        oracle=oracle,
        generator_version="2.0",
        metadata={
            "filler_count": filler_count,
            "synthetic": True,
            "cluster_id": f"{key}-s{seed:06d}",
            "latent_seed": seed,
            "context_layout": layout,
        },
    )
    case = replace(provisional, answer=solve_advanced_case(provisional))
    validate_advanced_case(case)
    return case


def solve_advanced_case(case: SyntheticCase) -> str:
    if case.family == "R5":
        return str(int(case.oracle["code"]))
    if case.family == "R6":
        return str(int(case.oracle["result"]))
    if case.family == "R7":
        return "YES" if bool(case.oracle["ready"]) else "NO"
    if case.family == "R8":
        return f"{int(case.oracle['top'])},{int(case.oracle['size'])}"
    raise ValueError(f"unsupported advanced family: {case.family}")


def solve_advanced_case_from_segments(case: SyntheticCase) -> str:
    texts = [segment.text for segment in case.segments]
    if case.family == "R5":
        if not any(text.startswith("Authority rule:") for text in texts):
            raise ValueError("R5 authority rule is missing")
        target_match = re.search(r"vault ([A-Z]+)'s access code", case.question)
        if target_match is None:
            raise ValueError("cannot parse R5 target")
        target = target_match.group(1)
        pattern = re.compile(r"FINAL VERIFIED: vault ([A-Z]+) access code is (\d+)\.")
        matches = [(match.group(1), int(match.group(2))) for text in texts if (match := pattern.fullmatch(text))]
        target_values = [value for vault, value in matches if vault == target]
        if len(target_values) != 1:
            raise ValueError("R5 requires exactly one final verified target record")
        return str(target_values[0])
    if case.family == "R6":
        numeric_values: dict[str, int] = {}
        pattern = re.compile(r"NUMERIC FACT: ([ABC]) = (\d+)\.")
        for text in texts:
            match = pattern.fullmatch(text)
            if match:
                numeric_values[match.group(1)] = int(match.group(2))
        required_rules = {
            "RULE ONE: T equals A plus B.",
            "RULE TWO: U equals T minus C.",
            "RULE THREE: RESULT equals two times U.",
        }
        if set(numeric_values) != {"A", "B", "C"} or not required_rules.issubset(texts):
            raise ValueError("R6 rendered program is incomplete")
        return str(2 * (numeric_values["A"] + numeric_values["B"] - numeric_values["C"]))
    if case.family == "R7":
        required = {
            "FACT: unit CEDAR has token A.",
            "RULE 1: token A implies token B.",
            "FACT: unit CEDAR is not on HOLD.",
            "RULE 2: token B together with not HOLD implies token C.",
            "RULE 3: token C implies status READY.",
            "OVERRIDE 4: LOCKED implies status NOT READY and overrides RULE 3.",
        }
        if not required.issubset(texts):
            raise ValueError("R7 rendered rule chain is incomplete")
        locked_rows = [text for text in texts if text.startswith("FACT: unit CEDAR is ") and "LOCKED" in text]
        if len(locked_rows) != 1:
            raise ValueError("R7 requires one CEDAR lock fact")
        return "NO" if locked_rows[0] == "FACT: unit CEDAR is LOCKED." else "YES"
    if case.family == "R8":
        pattern = re.compile(r"STACK A step (\d+): (PUSH (\d+)|POP)\.")
        operations: list[tuple[int, str, int | None]] = []
        for text in texts:
            match = pattern.fullmatch(text)
            if match:
                operations.append(
                    (int(match.group(1)), "PUSH" if match.group(3) else "POP", None if match.group(3) is None else int(match.group(3)))
                )
        operations.sort()
        if not operations or [index for index, _, _ in operations] != list(range(len(operations))):
            raise ValueError("R8 operation sequence is incomplete")
        stack: list[int] = []
        for _, operation, value in operations:
            if operation == "PUSH":
                if value is None:
                    raise ValueError("R8 PUSH is missing a value")
                stack.append(value)
            elif not stack:
                raise ValueError("R8 POP underflow")
            else:
                stack.pop()
        if not stack:
            raise ValueError("R8 final stack is unexpectedly empty")
        return f"{stack[-1]},{len(stack)}"
    raise ValueError(f"unsupported advanced family: {case.family}")


def _validate_structure(case: SyntheticCase, families: set[str]) -> None:
    if case.family not in families or not case.case_id:
        raise ValueError("invalid case family/id")
    if not 0 <= case.requested_evidence_depth <= 1 or not 0 <= case.actual_evidence_depth <= 1:
        raise ValueError("invalid evidence depth")
    ids = [row.segment_id for row in case.segments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate segment IDs in {case.case_id}")
    evidence = set(case.evidence_segment_ids)
    distractors = set(case.distractor_segment_ids)
    stale = set(case.stale_segment_ids)
    if not evidence or not (evidence | distractors | stale) <= set(ids):
        raise ValueError(f"annotation references missing segments in {case.case_id}")
    if evidence & distractors or evidence & stale or distractors & stale:
        raise ValueError("evidence/distractor/stale annotations must be disjoint")


def validate_advanced_case(case: SyntheticCase) -> dict[str, Any]:
    _validate_structure(case, set(ADVANCED_FAMILIES))
    oracle_answer = solve_advanced_case(case)
    rendered_answer = solve_advanced_case_from_segments(case)
    if oracle_answer != rendered_answer:
        raise ValueError(
            f"independent solver mismatch for {case.case_id}: oracle={oracle_answer!r}, rendered={rendered_answer!r}"
        )
    if case.answer and case.answer != oracle_answer:
        raise ValueError(f"solver mismatch for {case.case_id}: {oracle_answer!r} != {case.answer!r}")
    return {"case_id": case.case_id, "status": "PASS", "answer": oracle_answer}


def generate_advanced_suite(
    seeds: Iterable[int],
    evidence_depths: Iterable[float],
    *,
    families: Iterable[str] = ADVANCED_FAMILIES,
    layouts: Iterable[ContextLayout] = ("clustered",),
    filler_count: int = 24,
) -> list[SyntheticCase]:
    cases = [
        generate_advanced_case(family, int(seed), float(depth), filler_count=filler_count, layout=layout)
        for family in families
        for seed in seeds
        for depth in evidence_depths
        for layout in layouts
    ]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("advanced suite parameters generated duplicate case IDs")
    return cases


def generate_r1_cardinality_case(
    seed: int,
    evidence_depth: float,
    *,
    key_count: int,
    filler_count: int = 24,
    layout: ContextLayout = "clustered",
) -> SyntheticCase:
    """Generate an R1 case with a required cardinality from {2,3,4,6,8}."""

    if key_count not in R1_CARDINALITIES:
        raise ValueError(f"key_count must be one of {R1_CARDINALITIES}")
    rng = stable_rng(seed, "R1-cardinality", key_count)
    names = [f"K{index}" for index in range(key_count)]
    values = [int(value) for value in rng.integers(5, 80, size=key_count)]
    coefficients = [1 if index % 3 != 2 else -1 for index in range(key_count)]
    evidence = [
        SyntheticSegment(f"E{index:02d}", f"Verified register: {name} = {value}.", "evidence")
        for index, (name, value) in enumerate(zip(names, values, strict=True))
    ]
    distractors = [
        SyntheticSegment(f"D{index:02d}", f"Draft register: X{index} = {value + 11}; this is not a K-key.", "distractor")
        for index, value in enumerate(values)
    ]
    other = [*distractors, *(_filler(index, seed) for index in range(filler_count))]
    stable_rng(seed, "R1-cardinality", "other-order").shuffle(other)
    segments = place_evidence(
        evidence,
        other,
        layout=layout,
        depth=evidence_depth,
        seed=seed,
        family="R1",
    )
    terms = [name if coefficient > 0 else f"- {name}" for name, coefficient in zip(names, coefficients, strict=True)]
    question = "Compute " + " + ".join(terms).replace("+ -", "-") + ". Return one integer."
    evidence_ids = tuple(row.segment_id for row in evidence)
    provisional = SyntheticCase(
        case_id=f"R1-s{seed:06d}-k{key_count}-d{round(evidence_depth * 100):03d}-l{layout.replace('_', '-')}",
        family="R1",
        seed=seed,
        requested_evidence_depth=evidence_depth,
        actual_evidence_depth=_actual_depth(segments, set(evidence_ids)),
        segments=segments,
        question=question,
        answer="",
        evidence_segment_ids=evidence_ids,
        distractor_segment_ids=tuple(row.segment_id for row in distractors),
        stale_segment_ids=(),
        oracle={"names": names, "values": values, "coefficients": coefficients},
        generator_version="2.0",
        metadata={
            "filler_count": filler_count,
            "synthetic": True,
            "cluster_id": f"R1-s{seed:06d}-k{key_count}",
            "latent_seed": seed,
            "context_layout": layout,
            "required_fact_count": key_count,
        },
    )
    case = replace(provisional, answer=solve_case(provisional))
    if solve_case_from_segments(case) != case.answer:
        raise ValueError("R1 cardinality independent solver mismatch")
    validate_case(case)
    return case


def write_advanced_cases_jsonl(path: str | Path, cases: Iterable[SyntheticCase]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            validate_advanced_case(case)
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def read_advanced_cases_jsonl(path: str | Path) -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number} is not an object")
            case = SyntheticCase.from_dict(payload)
            validate_advanced_case(case)
            cases.append(case)
    return cases
