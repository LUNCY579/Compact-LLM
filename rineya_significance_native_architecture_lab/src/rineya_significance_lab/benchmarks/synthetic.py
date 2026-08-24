"""Deterministic R1-R4 generators with independent structured solvers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.randomness import stable_rng
from .schema import SyntheticCase, SyntheticSegment

FAMILIES = ("R1", "R2", "R3", "R4")


def _filler(index: int, seed: int) -> SyntheticSegment:
    codes = ("amber", "birch", "cobalt", "delta", "ember", "frost", "granite", "harbor")
    code = codes[(seed + index * 3) % len(codes)]
    return SyntheticSegment(
        segment_id=f"F{index:03d}",
        text=f"Archive note {index:02d} records routine marker {code}-{(seed + index * 17) % 997:03d}.",
        role="filler",
    )


def _place_evidence(
    evidence: Sequence[SyntheticSegment],
    other: Sequence[SyntheticSegment],
    depth: float,
) -> tuple[SyntheticSegment, ...]:
    if not 0.0 <= depth <= 1.0:
        raise ValueError("evidence depth must be in [0, 1]")
    total = len(evidence) + len(other)
    start = round(depth * max(0, total - len(evidence)))
    return (*other[:start], *evidence, *other[start:])


def _actual_depth(segments: Sequence[SyntheticSegment], evidence_ids: set[str]) -> float:
    positions = [index for index, row in enumerate(segments) if row.segment_id in evidence_ids]
    return float(np.mean(positions) / max(1, len(segments) - 1))


def _r1(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R1")
    key_count = int(rng.integers(2, 6))
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
    terms = [
        name if coefficient > 0 else f"- {name}"
        for name, coefficient in zip(names, coefficients, strict=True)
    ]
    question = "Compute " + " + ".join(terms).replace("+ -", "-") + ". Return one integer."
    oracle = {"names": names, "values": values, "coefficients": coefficients}
    return evidence, distractors, [], question, oracle


def _r2(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R2")
    hops = int(rng.integers(2, 9))
    directions = [("east", 1, 0), ("west", -1, 0), ("north", 0, 1), ("south", 0, -1)]
    moves = [directions[int(index)] for index in rng.integers(0, len(directions), size=hops)]
    evidence = [
        SyntheticSegment(
            f"E{index:02d}",
            f"Node N{index + 1} is one unit {direction} of node N{index}.",
            "evidence",
        )
        for index, (direction, _, _) in enumerate(moves)
    ]
    distractors = [
        SyntheticSegment(
            f"D{index:02d}",
            f"Disconnected node Z{index + 1} is one unit north of Z{index}.",
            "distractor",
        )
        for index in range(max(2, hops // 2))
    ]
    question = f"Starting N0 at (0,0), what is N{hops}'s coordinate? Return x,y."
    oracle = {"moves": [[dx, dy] for _, dx, dy in moves], "hops": hops}
    return evidence, distractors, [], question, oracle


def _r3(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R3")
    updates = int(rng.integers(3, 8))
    rooms = [int(value) for value in rng.integers(1, 30, size=updates)]
    records = [
        SyntheticSegment(
            f"U{index:02d}",
            f"At update {index}, item A was moved to room {room}.",
            "evidence" if index == updates - 1 else "stale",
        )
        for index, room in enumerate(rooms)
    ]
    evidence = [records[-1]]
    stale = records[:-1]
    distractors = [
        SyntheticSegment(f"D{index:02d}", f"At update {index}, item B was moved to room {room + 30}.", "distractor")
        for index, room in enumerate(rooms[:3])
    ]
    question = "After all numbered updates, which room contains item A? Return the room number."
    oracle = {"updates": rooms}
    return evidence, distractors, stale, question, oracle


def _r4(seed: int) -> tuple[list[SyntheticSegment], list[SyntheticSegment], list[SyntheticSegment], str, dict[str, Any]]:
    rng = stable_rng(seed, "R4")
    old_value = int(rng.integers(100, 500))
    new_value = old_value + int(rng.integers(7, 90))
    rule = SyntheticSegment(
        "E00",
        "Protocol: a later signed correction supersedes every earlier draft for the same ledger field.",
        "evidence",
    )
    correction = SyntheticSegment(
        "E01",
        f"Revision 3, signed: ledger field ALPHA is corrected to {new_value}.",
        "evidence",
    )
    stale = [
        SyntheticSegment("S00", f"Revision 1 draft: ledger field ALPHA is {old_value}.", "stale"),
        SyntheticSegment("S01", f"Revision 2 unsigned note repeats ALPHA as {old_value}.", "stale"),
    ]
    distractors = [
        SyntheticSegment("D00", f"Revision 4, signed: unrelated field BETA is {new_value + 3}.", "distractor")
    ]
    question = "What is the authoritative final value of ledger field ALPHA? Return one integer."
    oracle = {"old_value": old_value, "signed_corrections": [new_value]}
    return [rule, correction], distractors, stale, question, oracle


def generate_case(family: str, seed: int, evidence_depth: float, *, filler_count: int = 24) -> SyntheticCase:
    key = family.upper()
    builders = {"R1": _r1, "R2": _r2, "R3": _r3, "R4": _r4}
    if key not in builders:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    evidence, distractors, stale, question, oracle = builders[key](seed)
    fillers = [_filler(index, seed) for index in range(filler_count)]
    other = [*stale, *distractors, *fillers]
    stable_rng(seed, key, "other-order").shuffle(other)
    segments = _place_evidence(evidence, other, evidence_depth)
    evidence_ids = tuple(row.segment_id for row in evidence)
    case_id = f"{key}-s{seed:06d}-d{round(evidence_depth * 100):03d}"
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
        metadata={
            "filler_count": filler_count,
            "synthetic": True,
            "cluster_id": f"{key}-s{seed:06d}",
            "latent_seed": seed,
        },
    )
    answer = solve_case(provisional)
    case = replace(provisional, answer=answer)
    validate_case(case)
    return case


def solve_case(case: SyntheticCase) -> str:
    oracle = case.oracle
    if case.family == "R1":
        values = [int(value) for value in oracle["values"]]
        coefficients = [int(value) for value in oracle["coefficients"]]
        return str(
            sum(
                value * coefficient
                for value, coefficient in zip(values, coefficients, strict=True)
            )
        )
    if case.family == "R2":
        moves = [[int(axis) for axis in move] for move in oracle["moves"]]
        return f"{sum(move[0] for move in moves)},{sum(move[1] for move in moves)}"
    if case.family == "R3":
        updates = [int(value) for value in oracle["updates"]]
        if not updates:
            raise ValueError("R3 oracle has no updates")
        return str(updates[-1])
    if case.family == "R4":
        corrections = [int(value) for value in oracle["signed_corrections"]]
        if not corrections:
            raise ValueError("R4 oracle has no signed correction")
        return str(corrections[-1])
    raise ValueError(f"unsupported family: {case.family}")


def solve_case_from_segments(case: SyntheticCase) -> str:
    """Independently solve rendered segment text without reading ``oracle``."""

    texts = [segment.text for segment in case.segments]
    if case.family == "R1":
        registers: dict[str, int] = {}
        for text in texts:
            match = re.fullmatch(r"Verified register: (K\d+) = (\d+)\.", text)
            if match:
                registers[match.group(1)] = int(match.group(2))
        terms = re.findall(r"([+-]?)\s*(K\d+)", case.question.split(".", 1)[0])
        if not terms or any(name not in registers for _, name in terms):
            raise ValueError(f"cannot parse R1 rendered evidence for {case.case_id}")
        return str(sum((-1 if sign == "-" else 1) * registers[name] for sign, name in terms))
    if case.family == "R2":
        deltas = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
        moves: dict[int, tuple[int, int]] = {}
        pattern = re.compile(r"Node N(\d+) is one unit (east|west|north|south) of node N(\d+)\.")
        for text in texts:
            match = pattern.fullmatch(text)
            if not match:
                continue
            destination, direction, source = int(match.group(1)), match.group(2), int(match.group(3))
            if destination != source + 1:
                raise ValueError("R2 rendered chain is not contiguous")
            moves[source] = deltas[direction]
        target_match = re.search(r"what is N(\d+)'s coordinate", case.question, flags=re.IGNORECASE)
        if target_match is None:
            raise ValueError("cannot parse R2 target")
        target = int(target_match.group(1))
        if set(moves) != set(range(target)):
            raise ValueError(f"R2 rendered chain is incomplete for {case.case_id}")
        return f"{sum(moves[i][0] for i in range(target))},{sum(moves[i][1] for i in range(target))}"
    if case.family == "R3":
        updates: list[tuple[int, int]] = []
        pattern = re.compile(r"At update (\d+), item A was moved to room (\d+)\.")
        for text in texts:
            match = pattern.fullmatch(text)
            if match:
                updates.append((int(match.group(1)), int(match.group(2))))
        if not updates:
            raise ValueError(f"no rendered item-A updates in {case.case_id}")
        return str(max(updates)[1])
    if case.family == "R4":
        if not any("later signed correction supersedes" in text for text in texts):
            raise ValueError(f"missing rendered authority rule in {case.case_id}")
        corrections: list[tuple[int, int]] = []
        pattern = re.compile(r"Revision (\d+), signed: ledger field ALPHA is corrected to (\d+)\.")
        for text in texts:
            match = pattern.fullmatch(text)
            if match:
                corrections.append((int(match.group(1)), int(match.group(2))))
        if not corrections:
            raise ValueError(f"no rendered signed correction in {case.case_id}")
        return str(max(corrections)[1])
    raise ValueError(f"unsupported family: {case.family}")


def validate_case(case: SyntheticCase) -> dict[str, Any]:
    if case.family not in FAMILIES or not case.case_id:
        raise ValueError("invalid case family/id")
    if not 0 <= case.requested_evidence_depth <= 1 or not 0 <= case.actual_evidence_depth <= 1:
        raise ValueError("invalid evidence depth")
    ids = [row.segment_id for row in case.segments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate segment IDs in {case.case_id}")
    known = set(ids)
    evidence = set(case.evidence_segment_ids)
    distractors = set(case.distractor_segment_ids)
    stale = set(case.stale_segment_ids)
    if not evidence or not (evidence | distractors | stale) <= known:
        raise ValueError(f"annotation references missing segments in {case.case_id}")
    if evidence & distractors or evidence & stale or distractors & stale:
        raise ValueError("evidence/distractor/stale annotations must be disjoint")
    solved = solve_case(case)
    rendered_solved = solve_case_from_segments(case)
    if solved != rendered_solved:
        raise ValueError(
            f"independent solver mismatch for {case.case_id}: "
            f"oracle={solved!r}, rendered={rendered_solved!r}"
        )
    if case.answer and solved != case.answer:
        raise ValueError(f"solver mismatch for {case.case_id}: {solved!r} != {case.answer!r}")
    return {"case_id": case.case_id, "status": "PASS", "answer": solved}


def generate_suite(
    seeds: Iterable[int],
    evidence_depths: Iterable[float],
    *,
    families: Iterable[str] = FAMILIES,
    filler_count: int = 24,
) -> list[SyntheticCase]:
    cases = [
        generate_case(family, int(seed), float(depth), filler_count=filler_count)
        for family in families
        for seed in seeds
        for depth in evidence_depths
    ]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("suite parameters generated duplicate case IDs")
    return cases


def write_cases_jsonl(path: str | Path, cases: Iterable[SyntheticCase]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            validate_case(case)
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def read_cases_jsonl(path: str | Path) -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number} is not an object")
            case = SyntheticCase.from_dict(payload)
            validate_case(case)
            cases.append(case)
    return cases
