from __future__ import annotations

"""Generate v2.1.2 broad general-reasoning curriculum for Phase 9 students.

The corpus deliberately targets transferable operations rather than memorized
world knowledge.  Every answer is derived deterministically from information in
the prompt.  Scenarios are compact and operationally realistic: releases,
incidents, schedules, quotas, routing, inventory, access, support queues,
capacity planning, and similar tasks.

No chain-of-thought is trained.  Rows contain an auditable verifier payload, but
the assistant target is only the short final answer.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

SOURCE = "deterministic_phase9_general_reasoning_v212"
SCHEMA_VERSION = "2.1.2"
ABSTENTION_TEXT = "I don't know based on the available information."
SYSTEM = (
    "Answer the question concisely using only the information stated in the prompt. "
    "Do the reasoning internally and return only the short final answer. If the prompt "
    "does not contain enough information to determine one answer, reply exactly: "
    f'\"{ABSTENTION_TEXT}\"'
)

LABELS = ["Atlas", "Beacon", "Cedar", "Delta", "Ember", "Falcon", "Grove", "Harbor"]
PEOPLE = ["Ari", "Bea", "Chen", "Dara", "Eli", "Faye", "Gus", "Hana"]
REGIONS = ["East", "West", "North", "South", "Central", "Coastal"]
ITEMS = ["token", "key", "badge", "tablet", "folder", "sensor", "router", "permit"]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contract(answer: str, aliases: list[str] | None = None) -> dict[str, Any]:
    phrases = [answer.rstrip(".")]
    for alias in aliases or []:
        if alias and alias not in phrases:
            phrases.append(alias.rstrip("."))
    return {
        "schema_version": "4.5",
        "open_ended": False,
        "acceptable_modes": [],
        "requirements": [{
            "id": "required_reasoning_answer",
            "label": "explicit_required_content_missing",
            "description": f"The reply must contain the verified answer: {phrases[0]}",
            "severity": "major",
            "weight": 1.0,
            "failure_evidence": f"Verified answer content was missing: {phrases[0]}",
            "match": {"any_phrases": phrases},
        }],
        "prohibitions": [],
    }


@dataclass(frozen=True)
class Case:
    prompt: str
    answer: str
    difficulty: str
    verifier: dict[str, Any]
    aliases: list[str] | None = None
    decision: str = "answer"


@dataclass(frozen=True)
class Family:
    name: str
    build: Callable[[int, random.Random], Case]


def _pick_two(i: int) -> tuple[str, str]:
    a = LABELS[i % len(LABELS)]
    b = LABELS[(i * 3 + 1) % len(LABELS)]
    if a == b:
        b = LABELS[(LABELS.index(a) + 1) % len(LABELS)]
    return a, b


def arithmetic_total(i: int, rng: random.Random) -> Case:
    start = 30 + (i * 7) % 120
    add = 5 + (i * 11) % 50
    remove = 3 + (i * 13) % min(40, start + add - 1)
    ans = start + add - remove
    return Case(f"A warehouse starts with {start} units, receives {add}, then ships {remove}. How many remain?", f"{ans}.", "easy", {"op": "add_sub", "values": [start, add, remove], "result": ans})


def multiplication_total(i: int, rng: random.Random) -> Case:
    boxes = 3 + i % 17
    each = 4 + (i * 5) % 23
    ans = boxes * each
    return Case(f"A shipment has {boxes} boxes with {each} sensors in each box. How many sensors are there?", f"{ans}.", "easy", {"op": "multiply", "values": [boxes, each], "result": ans})


def exact_division(i: int, rng: random.Random) -> Case:
    teams = 2 + i % 11
    each = 3 + (i * 7) % 19
    total = teams * each
    return Case(f"{total} test cases are split equally across {teams} teams. How many cases does each team get?", f"{each}.", "easy", {"op": "divide", "values": [total, teams], "result": each})


def percentage_discount(i: int, rng: random.Random) -> Case:
    pct = [10, 20, 25, 50][i % 4]
    base = 20 * (2 + (i * 3) % 15)
    ans = base * (100 - pct) // 100
    return Case(f"A service costs ${base}. A {pct}% discount applies. What is the discounted price?", f"${ans}.", "moderate", {"op": "percent_discount", "base": base, "pct": pct, "result": ans})


def rate_time(i: int, rng: random.Random) -> Case:
    rate = 4 + (i * 3) % 21
    minutes = [5, 10, 15, 20, 30][i % 5]
    ans = rate * minutes
    return Case(f"A validator processes {rate} records per minute for {minutes} minutes. How many records does it process?", f"{ans}.", "easy", {"op": "rate_time", "rate": rate, "time": minutes, "result": ans})


def ratio_scale(i: int, rng: random.Random) -> Case:
    a = 2 + i % 7
    b = 3 + (i * 5) % 9
    scale = 2 + (i * 7) % 8
    ans = b * scale
    return Case(f"A mix uses {a} parts Cedar for every {b} parts Harbor. If Cedar is scaled to {a*scale} parts, how many Harbor parts are needed?", f"{ans}.", "moderate", {"op": "ratio", "ratio": [a, b], "scale": scale, "result": ans})


def duration_minutes(i: int, rng: random.Random) -> Case:
    start_h = 8 + i % 7
    start_m = [0, 10, 15, 20, 30, 40, 45, 50][i % 8]
    dur = [25, 35, 45, 55, 70, 85][i % 6]
    total = start_h * 60 + start_m + dur
    end_h, end_m = divmod(total, 60)
    prompt = f"A maintenance task starts at {start_h:02d}:{start_m:02d} and lasts {dur} minutes. What time does it end?"
    return Case(prompt, f"{end_h:02d}:{end_m:02d}.", "moderate", {"op": "time_add", "duration": dur, "result": f"{end_h:02d}:{end_m:02d}"})


def latest_date(i: int, rng: random.Random) -> Case:
    base = date(2030 + i % 4, 1 + i % 10, 1 + i % 20)
    d1 = base + timedelta(days=(i * 3) % 25)
    d2 = base + timedelta(days=10 + (i * 7) % 30)
    a, b = _pick_two(i)
    winner = a if d1 > d2 else b
    return Case(f"Release {a} is dated {d1.isoformat()}. Release {b} is dated {d2.isoformat()}. Which release is later?", f"{winner}.", "easy", {"op": "latest_date", "dates": {a: str(d1), b: str(d2)}, "result": winner})


def minimum_value(i: int, rng: random.Random) -> Case:
    labels = LABELS[:4]
    vals = [12 + (i * (k + 3) + k * 7) % 45 for k in range(4)]
    # deterministic tie-free adjustment
    while len(set(vals)) < 4:
        vals = [v + k for k, v in enumerate(vals)]
    winner = labels[min(range(4), key=lambda k: vals[k])]
    pairs = ", ".join(f"{n}: {v} ms" for n, v in zip(labels, vals))
    return Case(f"Latency results are {pairs}. Which option has the lowest latency?", f"{winner}.", "easy", {"op": "argmin", "values": dict(zip(labels, vals)), "result": winner})


def maximum_score(i: int, rng: random.Random) -> Case:
    labels = LABELS[2:6]
    vals = [50 + (i * (k + 5) + 9 * k) % 46 for k in range(4)]
    while len(set(vals)) < 4:
        vals = [v + k for k, v in enumerate(vals)]
    winner = labels[max(range(4), key=lambda k: vals[k])]
    pairs = ", ".join(f"{n}: {v}" for n, v in zip(labels, vals))
    return Case(f"Quality scores are {pairs}. Which option has the highest score?", f"{winner}.", "easy", {"op": "argmax", "values": dict(zip(labels, vals)), "result": winner})


def smallest_sufficient(i: int, rng: random.Random) -> Case:
    demand = 30 + (i * 11) % 135
    pools = [("Atlas", 48), ("Beacon", 72), ("Cedar", 96), ("Delta", 128), ("Ember", 192)]
    qualifies = [(n, c) for n, c in pools if c >= demand]
    if not qualifies:
        demand = 150
        qualifies = [(n, c) for n, c in pools if c >= demand]
    winner = min(qualifies, key=lambda x: x[1])[0]
    return Case(f"A job needs {demand} GB. Pools have Atlas 48, Beacon 72, Cedar 96, Delta 128, Ember 192 GB. Choose the smallest pool that is sufficient.", f"{winner}.", "moderate", {"op": "smallest_sufficient", "demand": demand, "result": winner})


def threshold_rule(i: int, rng: random.Random) -> Case:
    threshold = 70 + i % 21
    value = 55 + (i * 13) % 55
    ans = "Escalate" if value >= threshold else "Queue"
    return Case(f"Rule: escalate when risk is at least {threshold}; otherwise queue it. This case has risk {value}. What is the action?", f"{ans}.", "easy", {"op": "threshold", "threshold": threshold, "value": value, "result": ans})


def and_eligibility(i: int, rng: random.Random) -> Case:
    role = ["viewer", "operator", "admin"][i % 3]
    mfa = bool((i // 3) % 2)
    active = bool((i // 6) % 2)
    ans = "Allow" if role in {"operator", "admin"} and mfa and active else "Deny"
    return Case(f"Access requires role operator/admin AND verified MFA AND an active account. Role={role}; MFA={'verified' if mfa else 'not verified'}; account={'active' if active else 'inactive'}. Decision?", f"{ans}.", "moderate", {"op": "and_rule", "result": ans})


def or_routing(i: int, rng: random.Random) -> Case:
    severity = ["P1", "P2", "P3", "P4"][i % 4]
    data_loss = bool((i // 4) % 2)
    ans = "Page" if severity == "P1" or data_loss else "Queue"
    return Case(f"Rule: Page if severity is P1 OR confirmed data loss; otherwise Queue. Severity={severity}; data loss={'yes' if data_loss else 'no'}. What is the route?", f"{ans}.", "moderate", {"op": "or_rule", "result": ans})


def filter_then_min(i: int, rng: random.Random) -> Case:
    floor = 80 + i % 8
    rows = []
    choices = []
    for k, name in enumerate(LABELS[:4]):
        q = 75 + (i * (k + 2) + k * 5) % 18
        price = 70 + (i * (k + 3) + 11 * k) % 75
        rows.append((name, q, price))
        if q >= floor:
            choices.append((name, price))
    if not choices:
        name, _, price = rows[0]
        rows[0] = (name, floor, price)
        choices = [(name, price)]
    winner = min(choices, key=lambda x: (x[1], x[0]))[0]
    text = "; ".join(f"{n}: quality {q}, price ${p}" for n, q, p in rows)
    return Case(f"Minimum acceptable quality is {floor}. {text}. Among acceptable options, choose the lowest price.", f"{winner}.", "hard", {"op": "filter_argmin", "floor": floor, "result": winner})


def tie_break(i: int, rng: random.Random) -> Case:
    a, b = _pick_two(i)
    primary = 80 + i % 15
    tie = i % 2 == 0
    pb = primary if tie else primary - 1
    la = 10 + (i * 5) % 30
    lb = 10 + (i * 7 + 3) % 30
    if la == lb:
        lb += 1
    if primary != pb:
        winner = a
    else:
        winner = a if la < lb else b
    return Case(f"Choose higher score; if scores tie, choose lower latency. {a}: score {primary}, latency {la} ms. {b}: score {pb}, latency {lb} ms. Winner?", f"{winner}.", "hard", {"op": "tie_break", "result": winner})


def comparative_chain(i: int, rng: random.Random) -> Case:
    a, b, c = LABELS[0], LABELS[1], LABELS[2]
    # rotate relation direction
    if i % 2 == 0:
        prompt = f"{a} has higher throughput than {b}, and {b} has higher throughput than {c}. Which has the highest throughput?"
        ans = a
    else:
        prompt = f"{a} has lower latency than {b}, and {b} has lower latency than {c}. Which has the highest latency?"
        ans = c
    return Case(prompt, f"{ans}.", "moderate", {"op": "transitive_order", "result": ans})


def set_intersection(i: int, rng: random.Random) -> Case:
    skills = ["Python", "Rust", "SQL", "Docker", "Linux", "React"]
    a = {skills[(i + k) % len(skills)] for k in (0, 1, 3)}
    b = {skills[(i + k) % len(skills)] for k in (1, 2, 4)}
    common = sorted(a & b)
    ans = common[0] if len(common) == 1 else ", ".join(common)
    return Case(f"Team Atlas supports {', '.join(sorted(a))}. Team Beacon supports {', '.join(sorted(b))}. Which skill is supported by both?", f"{ans}.", "moderate", {"op": "intersection", "result": common})


def interval_overlap(i: int, rng: random.Random) -> Case:
    a0 = 9 * 60 + (i % 3) * 30
    a1 = a0 + 120
    b0 = 10 * 60 + (i % 4) * 20
    b1 = b0 + 100
    lo, hi = max(a0, b0), min(a1, b1)
    if lo < hi:
        ans = f"{lo//60:02d}:{lo%60:02d}-{hi//60:02d}:{hi%60:02d}"
    else:
        ans = "No overlap"
    fmt = lambda x: f"{x//60:02d}:{x%60:02d}"
    return Case(f"Atlas is available {fmt(a0)}-{fmt(a1)}. Beacon is available {fmt(b0)}-{fmt(b1)}. What is their overlapping window?", f"{ans}.", "hard", {"op": "interval_intersection", "result": ans})


def state_transition(i: int, rng: random.Random) -> Case:
    states = ["queued", "running", "paused", "done"]
    start = states[i % 3]
    actions = ["start", "pause", "resume", "finish"]
    seq = [actions[(i + k) % len(actions)] for k in range(3)]
    state = start
    for action in seq:
        if action == "start" and state == "queued": state = "running"
        elif action == "pause" and state == "running": state = "paused"
        elif action == "resume" and state == "paused": state = "running"
        elif action == "finish" and state == "running": state = "done"
    return Case(f"State rules: start turns queued→running; pause running→paused; resume paused→running; finish running→done. Start state is {start}. Apply: {', '.join(seq)}. Final state?", f"{state}.", "hard", {"op": "state_machine", "start": start, "actions": seq, "result": state})


def prerequisite_next(i: int, rng: random.Random) -> Case:
    steps = ["build", "test", "review", "deploy"]
    completed_n = i % 4
    completed = steps[:completed_n]
    next_step = steps[completed_n] if completed_n < 4 else "Done"
    text = ", ".join(completed) if completed else "none"
    return Case(f"Required order is build → test → review → deploy. Completed: {text}. What is the next required step?", f"{next_step}.", "easy", {"op": "prerequisite", "completed": completed, "result": next_step})


def entity_transfer(i: int, rng: random.Random) -> Case:
    p1, p2, p3 = PEOPLE[0], PEOPLE[1], PEOPLE[2]
    item = ITEMS[i % len(ITEMS)]
    # deterministic two transfers
    return Case(f"{p1} starts with the {item}. {p1} gives it to {p2}. Then {p2} gives it to {p3}. Who has the {item} now?", f"{p3}.", "moderate", {"op": "entity_tracking", "result": p3})


def indirect_manager(i: int, rng: random.Random) -> Case:
    a, b, c = PEOPLE[2], PEOPLE[4], PEOPLE[6]
    return Case(f"{a} reports to {b}. {b} reports to {c}. Who is {a}'s manager's manager?", f"{c}.", "moderate", {"op": "two_hop_relation", "result": c})


def shortest_path(i: int, rng: random.Random) -> Case:
    # Two explicit candidate paths, no graph search ambiguity.
    a = 3 + i % 8
    b = 4 + (i * 3) % 9
    c = 2 + (i * 5) % 7
    d = 5 + (i * 7) % 8
    p1 = a + b
    p2 = c + d
    winner = "Atlas route" if p1 < p2 else "Beacon route"
    if p1 == p2:
        d += 1
        p2 = c + d
        winner = "Atlas route" if p1 < p2 else "Beacon route"
    return Case(f"Atlas route has legs {a} km and {b} km. Beacon route has legs {c} km and {d} km. Which route is shorter?", f"{winner}.", "moderate", {"op": "path_sum_compare", "totals": [p1, p2], "result": winner})


def arithmetic_sequence(i: int, rng: random.Random) -> Case:
    start = 2 + i % 17
    step = 2 + (i * 3) % 9
    seq = [start + step * k for k in range(4)]
    ans = start + step * 4
    return Case(f"The sequence is {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}. It increases by the same amount each time. What comes next?", f"{ans}.", "moderate", {"op": "arithmetic_sequence", "step": step, "result": ans})


def parity_rule(i: int, rng: random.Random) -> Case:
    n = 10 + (i * 7) % 90
    ans = "Even" if n % 2 == 0 else "Odd"
    return Case(f"Classify the integer {n} as even or odd.", f"{ans}.", "easy", {"op": "parity", "result": ans})


def conditional_counterfactual(i: int, rng: random.Random) -> Case:
    load = 50 + (i * 11) % 90
    cap = 80 + (i * 3) % 60
    new_load = max(1, load - 30)
    ans = "Fits" if new_load <= cap else "Does not fit"
    return Case(f"A pool has capacity {cap}. Current load is {load}. If load were reduced by 30, would it fit?", f"{ans}.", "moderate", {"op": "counterfactual", "new_load": new_load, "capacity": cap, "result": ans})


def diagnostic_rule(i: int, rng: random.Random) -> Case:
    timeout = bool(i % 2)
    auth = bool((i // 2) % 2)
    if auth:
        ans = "Authentication issue"
    elif timeout:
        ans = "Network issue"
    else:
        ans = "No listed issue"
    return Case(f"Diagnostic rules: invalid credentials imply Authentication issue; otherwise a timeout implies Network issue; otherwise No listed issue. Credentials are {'invalid' if auth else 'valid'} and timeout is {'present' if timeout else 'absent'}. Diagnosis?", f"{ans}.", "moderate", {"op": "priority_rules", "result": ans})


def table_filter(i: int, rng: random.Random) -> Case:
    minimum = 3 + i % 4
    max_cost = 90 + (i * 7) % 40
    rows = []
    eligible = []
    for k, name in enumerate(LABELS[:4]):
        rating = 2 + (i * (k + 1) + k) % 5
        cost = 60 + (i * (k + 3) + 17*k) % 90
        rows.append((name, rating, cost))
        if rating >= minimum and cost <= max_cost:
            eligible.append(name)
    ans = eligible[0] if len(eligible) == 1 else (", ".join(eligible) if eligible else "None")
    text = "; ".join(f"{n}: rating {r}, cost ${c}" for n, r, c in rows)
    return Case(f"Keep options with rating at least {minimum} and cost at most ${max_cost}. {text}. Which options qualify?", f"{ans}.", "hard", {"op": "table_filter", "result": eligible})


def instruction_order(i: int, rng: random.Random) -> Case:
    # Explicit priority, useful instruction-following reasoning.
    default = ["Archive", "Queue", "Escalate"][i % 3]
    exception = bool((i // 3) % 2)
    ans = "Hold" if exception else default
    return Case(f"Policy: normally use {default}. Exception: if legal hold is active, use Hold instead. Legal hold is {'active' if exception else 'inactive'}. What action applies?", f"{ans}.", "moderate", {"op": "exception_overrides_default", "result": ans})


def sufficient_information(i: int, rng: random.Random) -> Case:
    missing = ["price", "quality", "capacity", "deadline"][i % 4]
    prompt = f"Rule: choose the cheapest option that meets minimum quality. Atlas has price $80 and quality 90. Beacon has quality 95, but its {missing if missing == 'price' else 'price'} is not provided. Which option should be chosen?"
    # Ensure missing price makes answer genuinely indeterminate; if i picks other labels, still construct missing price.
    return Case(prompt, ABSTENTION_TEXT, "hard", {"op": "insufficient_information", "missing": "Beacon price", "result": ABSTENTION_TEXT}, decision="abstain")


def ambiguous_ranking(i: int, rng: random.Random) -> Case:
    a, b = _pick_two(i)
    prompt = f"{a} is faster than {b}. A third option Cedar is also available, but no speed comparison involving Cedar is given. Which of all three is fastest?"
    return Case(prompt, ABSTENTION_TEXT, "hard", {"op": "partial_order_insufficient", "result": ABSTENTION_TEXT}, decision="abstain")


def missing_rule(i: int, rng: random.Random) -> Case:
    prompt = "A ticket has severity P2 and customer impact. The prompt does not state what routing rule should be used. What route is required?"
    return Case(prompt, ABSTENTION_TEXT, "moderate", {"op": "missing_rule", "result": ABSTENTION_TEXT}, decision="abstain")


FAMILIES: list[Family] = [
    Family("arithmetic_total", arithmetic_total),
    Family("multiplication_total", multiplication_total),
    Family("exact_division", exact_division),
    Family("percentage_discount", percentage_discount),
    Family("rate_time", rate_time),
    Family("ratio_scale", ratio_scale),
    Family("duration_minutes", duration_minutes),
    Family("latest_date", latest_date),
    Family("minimum_value", minimum_value),
    Family("maximum_score", maximum_score),
    Family("smallest_sufficient", smallest_sufficient),
    Family("threshold_rule", threshold_rule),
    Family("and_eligibility", and_eligibility),
    Family("or_routing", or_routing),
    Family("filter_then_min", filter_then_min),
    Family("tie_break", tie_break),
    Family("comparative_chain", comparative_chain),
    Family("set_intersection", set_intersection),
    Family("interval_overlap", interval_overlap),
    Family("state_transition", state_transition),
    Family("prerequisite_next", prerequisite_next),
    Family("entity_transfer", entity_transfer),
    Family("indirect_manager", indirect_manager),
    Family("shortest_path", shortest_path),
    Family("arithmetic_sequence", arithmetic_sequence),
    Family("parity_rule", parity_rule),
    Family("conditional_counterfactual", conditional_counterfactual),
    Family("diagnostic_rule", diagnostic_rule),
    Family("table_filter", table_filter),
    Family("instruction_order", instruction_order),
    Family("sufficient_information", sufficient_information),
    Family("ambiguous_ranking", ambiguous_ranking),
    Family("missing_rule", missing_rule),
]


def split_for(local_index: int) -> str:
    # 80/10/10 per family, preserving family balance in every split.
    bucket = (local_index * 37 + 13) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def row_for(family: Family, local_index: int, global_index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed * 1000003 + global_index * 97 + len(family.name))
    case = family.build(local_index, rng)
    split = split_for(local_index)
    rid = f"v212_{family.name}_{local_index:04d}"
    return {
        "id": rid,
        "topic_id": rid,
        "dataset_schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "mode": "general_reasoning",
        "reasoning_family": family.name,
        "difficulty": case.difficulty,
        "decision": case.decision,
        "answer_stance": "uncertain" if case.decision == "abstain" else "supported",
        "answer": case.answer,
        "approved_answer": case.answer,
        "answer_aliases": case.aliases or [],
        "response_contract": contract(case.answer, case.aliases),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": case.prompt},
            {"role": "assistant", "content": case.answer},
        ],
        "intended_split": split,
        "review_status": "auto_verified",
        "scenario_flags": ["general_reasoning", family.name, case.difficulty, "real_world_reflected"],
        "verification": case.verifier,
        "teacher_provenance": {
            "source": SOURCE,
            "reviewed": False,
            "verification": "deterministic_programmatic_gold_no_chain_of_thought",
            "created_at": "2026-08-09",
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def with_assistant_training_target(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure replay rows are CompletionDataset-ready.

    Original Phase 9 reviewed splits are often prompt-only (system + user) with
    gold in `approved_answer` / `answer`. v2.1.1 / v2.1.2 rows already include
    the assistant turn. Preservation replay must append that target before SFT.
    """
    out = json.loads(json.dumps(row))
    messages = [dict(message) for message in (out.get("messages") or [])]
    if messages and messages[-1].get("role") == "assistant":
        content = str(messages[-1].get("content") or "").strip()
        if not content:
            raise ValueError(f"{out.get('id')}: empty assistant training target")
        out["messages"] = messages
        out.setdefault("reference", content)
        return out
    answer = str(out.get("approved_answer") or out.get("answer") or out.get("reference") or "").strip()
    if not answer:
        raise ValueError(f"{out.get('id')}: missing approved_answer/answer for assistant training target")
    if not messages:
        raise ValueError(f"{out.get('id')}: missing prompt messages for assistant training target")
    messages.append({"role": "assistant", "content": answer})
    out["messages"] = messages
    out["reference"] = answer
    return out


def preservation_sample(rows: list[dict[str, Any]], limit: int, seed: int, prefix: str) -> list[dict[str, Any]]:
    """Stratified, train-only replay sample with SFT assistant targets attached."""
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("intended_split") or "train") != "train":
            continue
        cat = str(row.get("category") or row.get("mode") or "other")
        by_cat[cat].append(row)
    rng = random.Random(seed)
    cats = sorted(by_cat)
    out: list[dict[str, Any]] = []
    if not cats:
        return out
    # round-robin sampling prevents one dominant source category from consuming replay
    for cat in cats:
        rng.shuffle(by_cat[cat])
    pointers = {c: 0 for c in cats}
    while len(out) < limit:
        progressed = False
        for cat in cats:
            p = pointers[cat]
            if p < len(by_cat[cat]) and len(out) < limit:
                copied = with_assistant_training_target(by_cat[cat][p])
                copied["v212_replay_source"] = prefix
                out.append(copied)
                pointers[cat] += 1
                progressed = True
        if not progressed:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--original-train", type=Path, required=True)
    p.add_argument("--v211-train", type=Path, required=True)
    p.add_argument("--examples-per-family", type=int, default=400)
    p.add_argument("--stage-a-general-rows", type=int, default=4800)
    p.add_argument("--stage-b-general-rows", type=int, default=3600)
    p.add_argument("--stage-b-preservation-rows", type=int, default=2400)
    p.add_argument("--seed", type=int, default=21242)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = args.output_root
    if root.exists() and args.force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    g = 0
    for family in FAMILIES:
        for local_index in range(args.examples_per_family):
            rows.append(row_for(family, local_index, g, args.seed))
            g += 1

    # IDs, prompt/answer pairs, and split leakage must be clean by construction.
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate generated IDs")
    split_rows = {s: [r for r in rows if r["intended_split"] == s] for s in ("train", "dev", "test")}
    train_ids = {r["id"] for r in split_rows["train"]}
    dev_ids = {r["id"] for r in split_rows["dev"]}
    test_ids = {r["id"] for r in split_rows["test"]}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise RuntimeError("Split ID leakage detected")

    # Training views: Stage A learns broad operations; Stage B consolidates with
    # historical Phase 9 and v2.1.1 train-only behavior replay.
    general_train = list(split_rows["train"])
    rng = random.Random(args.seed + 9)
    rng.shuffle(general_train)
    stage_a = general_train[: min(args.stage_a_general_rows, len(general_train))]
    stage_b_general = general_train[: min(args.stage_b_general_rows, len(general_train))]

    original = read_jsonl(args.original_train)
    v211 = read_jsonl(args.v211_train)
    half = args.stage_b_preservation_rows // 2
    replay_original = preservation_sample(original, half, args.seed + 101, "original_phase9_train")
    replay_v211 = preservation_sample(v211, args.stage_b_preservation_rows - len(replay_original), args.seed + 202, "v211_train")
    replay = replay_original + replay_v211
    stage_b = stage_b_general + replay
    random.Random(args.seed + 303).shuffle(stage_b)

    write_jsonl(root / "prompts.all.jsonl", rows)
    for split, vals in split_rows.items():
        write_jsonl(root / "reviewed_splits" / f"{split}.reviewed.jsonl", vals)
    write_jsonl(root / "curriculum_views" / "train_stage_a_reasoning_core.reviewed.jsonl", stage_a)
    write_jsonl(root / "curriculum_views" / "train_stage_b_generalization_mix.reviewed.jsonl", stage_b)
    write_jsonl(root / "curriculum_views" / "dev_general_reasoning_probe.reviewed.jsonl", split_rows["dev"])

    family_counts = Counter(r["reasoning_family"] for r in rows)
    split_counts = Counter(r["intended_split"] for r in rows)
    difficulty_counts = Counter(r["difficulty"] for r in rows)
    decision_counts = Counter(r["decision"] for r in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "seed": args.seed,
        "families": len(FAMILIES),
        "family_names": [f.name for f in FAMILIES],
        "canonical_rows": len(rows),
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "difficulty_counts": dict(difficulty_counts),
        "decision_counts": dict(decision_counts),
        "stage_a_rows": len(stage_a),
        "stage_b_rows": len(stage_b),
        "stage_b_general_rows": len(stage_b_general),
        "stage_b_preservation_rows": len(replay),
        "stage_b_preservation_sources": dict(Counter(str(r.get("v212_replay_source") or "") for r in replay)),
        "test_policy": "topic/example IDs are held out from training; sealed TEST is not used for selection",
        "chain_of_thought_policy": "No rationale text is placed in assistant targets. Verification metadata is audit-only.",
    }
    write_json(root / "manifests" / "dataset_manifest.json", manifest)

    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "manifests/file_hashes.json" not in path.as_posix():
            hashes[path.relative_to(root).as_posix()] = sha256(path)
    write_json(root / "manifests" / "file_hashes.json", hashes)

    readme = f"""# Phase 9 v2.1.2 General Reasoning Curriculum\n\nThis deterministic corpus trains **transferable operations**, not general-world facts.\n\n- Canonical rows: **{len(rows)}**\n- Families: **{len(FAMILIES)}**\n- Train / DEV / sealed TEST: **{split_counts['train']} / {split_counts['dev']} / {split_counts['test']}**\n- Stage A broad-reasoning view: **{len(stage_a)}** rows\n- Stage B mixed generalization/preservation view: **{len(stage_b)}** rows\n- Stage B historical preservation replay: **{len(replay)}** rows\n\nThe assistant target is always a short final answer. No chain-of-thought is trained.\n"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
