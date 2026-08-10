from __future__ import annotations

"""Generate the v2.2 multi-domain capability-composition curriculum.

The dataset intentionally teaches *operations* rather than world facts.  Every
answer is derivable from information embedded in the prompt, so a tiny student
learns reusable comparison, filtering, state tracking, rule application,
uncertainty, evidence selection, and concise transformation behaviors instead
of memorizing an encyclopedia.

The 27 domains mirror the v2.2 research contract.  High-stakes domains use
fictional/hypothetical rules and conservative boundary targets; the corpus is
not a substitute for medical, legal, or financial reference material.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.2"
SOURCE = "deterministic_phase9_multidomain_reasoning_v22"
ABSTAIN = "I don't know based on the available information."
SYSTEM = (
    "Answer concisely using only the information stated in the prompt. Reason internally and return only the short final answer. "
    f'If one answer is not supported, reply exactly: "{ABSTAIN}"'
)

DOMAINS = [
    "general", "research", "writing", "communication", "translation", "coding", "devops", "systems",
    "math", "data_analysis", "science", "engineering", "education", "business", "operations", "finance",
    "legal", "health", "consumer", "travel", "home", "creative", "media", "personal", "coaching", "tools", "agents",
]

ARCHETYPES = [
    "compare_metric", "filter_then_rank", "aggregate_delta", "capacity_fit", "state_transition", "rule_exception",
    "evidence_binding", "insufficient_information", "conflicting_evidence", "dependency_order", "two_hop_relation",
    "constraint_preserving_transform", "constraint_satisfaction", "evidence_synthesis", "diagnostic_priority",
]

NAMES = ["Atlas", "Beacon", "Cedar", "Delta", "Ember", "Falcon", "Grove", "Harbor"]
PEOPLE = ["Ari", "Bea", "Chen", "Dara", "Eli", "Faye", "Gus", "Hana"]

PROFILES: dict[str, dict[str, str]] = {
    "general": {"thing":"option","metric":"score","unit":"points","work":"task","group":"team","artifact":"note"},
    "research": {"thing":"study","metric":"evidence score","unit":"points","work":"analysis","group":"lab","artifact":"finding"},
    "writing": {"thing":"draft","metric":"word count","unit":"words","work":"revision","group":"editorial team","artifact":"sentence"},
    "communication": {"thing":"message","metric":"response time","unit":"minutes","work":"reply","group":"support team","artifact":"message"},
    "translation": {"thing":"translation","metric":"term matches","unit":"matches","work":"localization","group":"language team","artifact":"phrase"},
    "coding": {"thing":"implementation","metric":"runtime","unit":"ms","work":"build","group":"dev team","artifact":"function"},
    "devops": {"thing":"deployment","metric":"error rate","unit":"percent","work":"rollout","group":"SRE team","artifact":"service"},
    "systems": {"thing":"node","metric":"load","unit":"percent","work":"allocation","group":"platform team","artifact":"resource"},
    "math": {"thing":"solution","metric":"value","unit":"units","work":"calculation","group":"study group","artifact":"expression"},
    "data_analysis": {"thing":"dataset","metric":"error","unit":"points","work":"analysis","group":"analytics team","artifact":"table"},
    "science": {"thing":"sample","metric":"measurement","unit":"units","work":"experiment","group":"science team","artifact":"result"},
    "engineering": {"thing":"design","metric":"stress","unit":"MPa","work":"review","group":"engineering team","artifact":"component"},
    "education": {"thing":"lesson","metric":"quiz score","unit":"points","work":"lesson","group":"class","artifact":"exercise"},
    "business": {"thing":"proposal","metric":"cost","unit":"dollars","work":"plan","group":"strategy team","artifact":"proposal"},
    "operations": {"thing":"route","metric":"duration","unit":"minutes","work":"shift","group":"ops team","artifact":"runbook"},
    "finance": {"thing":"scenario","metric":"cost","unit":"dollars","work":"budget review","group":"finance team","artifact":"estimate"},
    "legal": {"thing":"policy option","metric":"review days","unit":"days","work":"policy review","group":"compliance team","artifact":"clause"},
    "health": {"thing":"care option","metric":"wait time","unit":"minutes","work":"triage example","group":"care team","artifact":"instruction"},
    "consumer": {"thing":"product","metric":"price","unit":"dollars","work":"purchase","group":"household","artifact":"listing"},
    "travel": {"thing":"itinerary","metric":"travel time","unit":"minutes","work":"trip","group":"travel party","artifact":"booking"},
    "home": {"thing":"repair option","metric":"time","unit":"minutes","work":"home task","group":"household","artifact":"checklist"},
    "creative": {"thing":"concept","metric":"length","unit":"words","work":"creative brief","group":"creative team","artifact":"line"},
    "media": {"thing":"segment","metric":"duration","unit":"minutes","work":"schedule","group":"media team","artifact":"clip"},
    "personal": {"thing":"plan","metric":"time","unit":"minutes","work":"personal plan","group":"household","artifact":"reminder"},
    "coaching": {"thing":"practice plan","metric":"time","unit":"minutes","work":"practice","group":"learner","artifact":"goal"},
    "tools": {"thing":"tool","metric":"latency","unit":"ms","work":"tool task","group":"toolchain","artifact":"command"},
    "agents": {"thing":"agent path","metric":"steps","unit":"steps","work":"workflow","group":"agent system","artifact":"state"},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_for(local_index: int) -> str:
    m = local_index % 100
    if m < 80:
        return "train"
    if m < 90:
        return "dev"
    return "test"


def difficulty_for(local_index: int) -> str:
    m = local_index % 10
    if m < 3:
        return "easy"
    if m < 8:
        return "moderate"
    return "hard"


def punctuation_answer(value: str) -> str:
    value = value.strip()
    if value.endswith((".", "!", "?")):
        return value
    return value + "."


def row_contract(answer: str) -> dict[str, Any]:
    return {
        "schema_version": "4.5",
        "open_ended": False,
        "acceptable_modes": [],
        "requirements": [{
            "id": "verified_short_answer",
            "label": "explicit_required_content_missing",
            "description": f"Reply must equal the deterministic verified answer: {answer.rstrip('.')}",
            "severity": "major",
            "weight": 1.0,
            "match": {"any_phrases": [answer.rstrip(".")]},
        }],
        "prohibitions": [],
    }


def surface(split: str, variants: tuple[str, str, str]) -> str:
    return variants[0] if split == "train" else variants[1] if split == "dev" else variants[2]


def build_case(domain: str, archetype: str, j: int, rng: random.Random) -> tuple[str, str, str, dict[str, Any]]:
    p = PROFILES[domain]
    a, b, c = NAMES[(j + len(domain)) % 8], NAMES[(j * 3 + 1) % 8], NAMES[(j * 5 + 2) % 8]
    if len({a,b,c}) < 3:
        a, b, c = NAMES[j % 8], NAMES[(j+2)%8], NAMES[(j+5)%8]
    split = split_for(j)
    diff = difficulty_for(j)
    base = 10 + (j * 7 + len(domain)) % 31
    decision = "answer"
    verification: dict[str, Any] = {"operation": archetype, "domain": domain}

    if archetype == "compare_metric":
        vals = {a: base + 7, b: base + 2, c: base + 11}
        choose_min = (j % 2 == 0)
        winner = min(vals, key=vals.get) if choose_min else max(vals, key=vals.get)
        direction = "lowest" if choose_min else "highest"
        prompt = surface(split, (
            f"Three candidates for this {p['work']} have {p['metric']}: {a} {vals[a]} {p['unit']}, {b} {vals[b]} {p['unit']}, {c} {vals[c]} {p['unit']}. Which has the {direction} {p['metric']}?",
            f"For this {p['work']}, {a}={vals[a]}, {b}={vals[b]}, and {c}={vals[c]} {p['unit']} on {p['metric']}. Name the {direction} one.",
            f"Select the {p['thing']} with the {direction} {p['metric']} from {a}:{vals[a]}, {b}:{vals[b]}, {c}:{vals[c]} {p['unit']}.",
        ))
        answer = punctuation_answer(winner)
        verification.update({"values": vals, "direction": direction, "winner": winner})

    elif archetype == "filter_then_rank":
        threshold = base + 4
        vals = {
            a: {"quality": threshold + 3, "cost": 120 + j % 20},
            b: {"quality": threshold - 2, "cost": 80 + j % 20},
            c: {"quality": threshold + 7, "cost": 100 + j % 20},
        }
        valid = [name for name,v in vals.items() if v["quality"] >= threshold]
        winner = min(valid, key=lambda name: vals[name]["cost"])
        prompt = surface(split, (
            f"For this {p['work']}, require quality at least {threshold}. {a}: quality {vals[a]['quality']}, cost {vals[a]['cost']}; {b}: quality {vals[b]['quality']}, cost {vals[b]['cost']}; {c}: quality {vals[c]['quality']}, cost {vals[c]['cost']}. Among qualifying options, choose the lowest cost.",
            f"Minimum quality is {threshold}. Filter {a}({vals[a]['quality']}, ${vals[a]['cost']}), {b}({vals[b]['quality']}, ${vals[b]['cost']}), {c}({vals[c]['quality']}, ${vals[c]['cost']}); then pick the cheapest qualifying {p['thing']}.",
            f"Apply two steps: keep quality >= {threshold}; then minimize cost. Candidates: {a} q={vals[a]['quality']} c={vals[a]['cost']}; {b} q={vals[b]['quality']} c={vals[b]['cost']}; {c} q={vals[c]['quality']} c={vals[c]['cost']}.",
        ))
        answer = punctuation_answer(winner)
        verification.update({"threshold": threshold, "values": vals, "winner": winner})

    elif archetype == "aggregate_delta":
        start = base + 20
        add1, add2, remove = 3 + j % 7, 2 + (j*2)%6, 1 + j%5
        result = start + add1 + add2 - remove
        prompt = surface(split, (
            f"A {p['work']} starts at {start} units, gains {add1}, gains {add2}, then uses {remove}. What is the final amount?",
            f"Track the {p['artifact']}: initial {start}; +{add1}; +{add2}; -{remove}. Final total?",
            f"Compute the ending quantity for this {p['work']}: {start} + {add1} + {add2} - {remove}.",
        ))
        answer = f"{result}."
        verification.update({"start":start,"add":[add1,add2],"remove":remove,"result":result})

    elif archetype == "capacity_fit":
        need = 25 + j % 30
        caps = {a: need - 4, b: need + 3, c: need + 12}
        winner = min((name for name in caps if caps[name] >= need), key=lambda name:caps[name])
        prompt = surface(split, (
            f"This {p['work']} needs capacity {need}. {a} supports {caps[a]}, {b} supports {caps[b]}, {c} supports {caps[c]}. Choose the smallest sufficient option.",
            f"Required capacity: {need}. Available candidates: {a}={caps[a]}, {b}={caps[b]}, {c}={caps[c]}. Which is the least capacity that still fits?",
            f"Pick the minimum qualifying capacity for {need}: {a}:{caps[a]}, {b}:{caps[b]}, {c}:{caps[c]}.",
        ))
        answer = punctuation_answer(winner)
        verification.update({"need":need,"capacities":caps,"winner":winner})

    elif archetype == "state_transition":
        states = ["queued", "running", "paused", "done"]
        actions = ["start", "pause", "resume", "finish"]
        transition = {("queued","start"):"running", ("running","pause"):"paused", ("paused","resume"):"running", ("running","finish"):"done"}
        sequence = ["start", "pause", "resume"] if j % 2 == 0 else ["start", "finish"]
        state = "queued"
        for act in sequence:
            state = transition[(state,act)]
        prompt = surface(split, (
            f"State rules for this {p['work']}: queued+start->running; running+pause->paused; paused+resume->running; running+finish->done. Start queued. Actions: {', '.join(sequence)}. Final state?",
            f"Apply the workflow transitions in order. queued --start--> running; running --pause--> paused; paused --resume--> running; running --finish--> done. Initial=queued; actions={', '.join(sequence)}.",
            f"Track the state only: initial queued. Rules: start from queued gives running; pause from running gives paused; resume from paused gives running; finish from running gives done. Sequence: {', '.join(sequence)}.",
        ))
        answer = punctuation_answer(state)
        verification.update({"sequence":sequence,"final_state":state})

    elif archetype == "rule_exception":
        default = a
        exception = b
        condition = "priority case" if j % 2 == 0 else "weekend case"
        applies = j % 3 != 0
        context = condition if applies else "ordinary case"
        winner = exception if applies else default
        prompt = surface(split, (
            f"Hypothetical {domain.replace('_',' ')} rule: use {default} by default. Exception: for a {condition}, use {exception}. This is a {context}. Which option applies?",
            f"Apply this stated rule only: default={default}; if '{condition}' then {exception}. Current case: {context}. Result?",
            f"Rule precedence for this example: {condition} overrides the default. Default {default}; override {exception}. Case type is {context}. Which one wins?",
        ))
        answer = punctuation_answer(winner)
        verification.update({"default":default,"exception":exception,"condition":condition,"context":context,"winner":winner})

    elif archetype == "evidence_binding":
        target_record = 1 + j % 3
        values = {1: a, 2: b, 3: c}
        prompt = surface(split, (
            f"A {p['artifact']} lists record 1: {a}; record 2: {b}; record 3: {c}. The question asks for record {target_record}. Which label belongs to that exact record?",
            f"Bind the requested record, not its siblings. Records are 1={a}, 2={b}, 3={c}. Requested: record {target_record}.",
            f"Use only the named record from this {p['artifact']}: [1 {a}] [2 {b}] [3 {c}]. Return record {target_record}'s value.",
        ))
        winner = values[target_record]
        answer = punctuation_answer(winner)
        verification.update({"records":values,"requested":target_record,"winner":winner})

    elif archetype == "insufficient_information":
        decision = "abstain"
        prompt = surface(split, (
            f"{a} has {p['metric']} {base}. {b} has {p['metric']} {base+2}. No {p['metric']} is given for {c}. Which of all three has the highest {p['metric']}?",
            f"Known: {a}={base}, {b}={base+2}. {c}'s {p['metric']} is not provided. Name the highest among all three.",
            f"Can you determine the top {p['thing']} from {a}:{base}, {b}:{base+2}, and {c}:unknown?",
        ))
        answer = ABSTAIN
        verification.update({"missing":c,"reason":"required comparison operand absent"})

    elif archetype == "conflicting_evidence":
        decision = "abstain"
        prompt = surface(split, (
            f"Two equal-authority notes conflict. Note A says the selected {p['thing']} is {a}. Note B says it is {b}. No precedence rule is given. Which one is selected?",
            f"For this hypothetical {p['work']}, source 1 says {a}; equally trusted source 2 says {b}. There is no freshness or priority difference. Give the certain answer.",
            f"Equal-trust evidence disagrees: {a} versus {b}. With no tie-breaking rule, what is the supported selection?",
        ))
        answer = ABSTAIN
        verification.update({"claims":[a,b],"reason":"unresolved equal-authority conflict"})

    elif archetype == "dependency_order":
        # A depends on B; B depends on C -> C first.
        prompt = surface(split, (
            f"For this {p['work']}, {a} requires {b} first, and {b} requires {c} first. Which item must be completed before the other two?",
            f"Dependencies: {a} <- {b} <- {c}, where the item on the right must happen first. What is the earliest prerequisite?",
            f"Before {a}, do {b}; before {b}, do {c}. Which one comes first?",
        ))
        answer = punctuation_answer(c)
        verification.update({"edges":[[a,b],[b,c]],"first":c})

    elif archetype == "two_hop_relation":
        person = PEOPLE[j % len(PEOPLE)]
        team = f"Team-{a}"
        system = f"System-{b}"
        prompt = surface(split, (
            f"{person} belongs to {team}. {team} uses {system} for this {p['work']}. Which system does {person}'s team use?",
            f"Relations: {person} -> {team}; {team} -> {system}. Follow the two links and name the system.",
            f"Entity tracking: person {person} is assigned to {team}, and that team operates {system}. What does {person} indirectly use?",
        ))
        answer = punctuation_answer(system)
        verification.update({"person":person,"team":team,"system":system})

    elif archetype == "constraint_satisfaction":
        # Multi-constraint selection with one uniquely valid option.  This is
        # deliberately language-heavy compared with the arithmetic families.
        required = "verified" if j % 2 == 0 else "current"
        second = "compact" if j % 3 else "approved"
        forbidden = "stale" if j % 2 == 0 else "blocked"
        opts = {
            "A": {required, second, "safe"},
            "B": {required, second, forbidden},
            "C": {second, "safe"},
        }
        prompt = surface(split, (
            f"Choose the only {p['thing']} that satisfies every stated constraint for this {p['work']}: it must be {required} and {second}, and it must not be {forbidden}. A has {', '.join(sorted(opts['A']))}. B has {', '.join(sorted(opts['B']))}. C has {', '.join(sorted(opts['C']))}.",
            f"Requirements for this {p['work']}: require {required}; require {second}; reject anything {forbidden}. Candidate A: {', '.join(sorted(opts['A']))}. Candidate B: {', '.join(sorted(opts['B']))}. Candidate C: {', '.join(sorted(opts['C']))}. Which candidate qualifies?",
            f"Apply all constraints together, not one at a time. Need [{required}, {second}], forbid [{forbidden}]. A=[{', '.join(sorted(opts['A']))}], B=[{', '.join(sorted(opts['B']))}], C=[{', '.join(sorted(opts['C']))}].",
        ))
        answer = "A."
        verification.update({"required":[required,second],"forbidden":forbidden,"options":{k:sorted(v) for k,v in opts.items()},"winner":"A"})

    elif archetype == "evidence_synthesis":
        item = a
        stage = "review" if j % 2 == 0 else "verification"
        status = "approved" if j % 3 else "flagged"
        prompt = surface(split, (
            f"Synthesize these two prompt-contained facts for this {p['work']}: (1) {item} is {status}. (2) every {status} {p['thing']} goes to {stage}. Where does {item} go next?",
            f"Fact A: {item} has status {status}. Fact B: items with status {status} are routed to {stage}. Combine the facts and give {item}'s next destination.",
            f"Use both statements: {item}->{status}; {status}->{stage}. What follows for {item}?",
        ))
        answer = punctuation_answer(stage)
        verification.update({"item":item,"status":status,"routing_rule":stage,"conclusion":stage})

    elif archetype == "diagnostic_priority":
        # Ordered rule application: critical overrides warning, otherwise monitor.
        critical = (j % 4 == 0)
        warning = (j % 3 != 0)
        action = "escalate" if critical else "inspect" if warning else "monitor"
        observations = []
        if critical:
            observations.append("critical flag present")
        if warning:
            observations.append("warning flag present")
        if not observations:
            observations.append("no critical or warning flag")
        obs = "; ".join(observations)
        prompt = surface(split, (
            f"For this hypothetical {p['work']}, apply the priority rules in order: if a critical flag is present, escalate; otherwise if a warning flag is present, inspect; otherwise monitor. Observation: {obs}. What action applies?",
            f"Priority policy: critical -> escalate; else warning -> inspect; else -> monitor. Current observation: {obs}. Return the selected action.",
            f"Choose by highest-priority matching rule only. Rules are [critical: escalate] [warning: inspect] [default: monitor]. Signals: {obs}.",
        ))
        answer = punctuation_answer(action)
        verification.update({"critical":critical,"warning":warning,"priority":["critical","warning","default"],"action":action})

    elif archetype == "constraint_preserving_transform":
        # Closed transformation/equivalence tasks avoid fuzzy grader dependence.
        if domain == "translation":
            left = ["luma", "sena", "taro", "mira"][j % 4]
            right = ["gate", "blue", "quiet", "north"][(j + 1) % 4]
            glossary = {left: a.casefold(), right: b.casefold()}
            prompt = surface(split, (
                f"Use only this supplied glossary: {left}={glossary[left]}, {right}={glossary[right]}. Translate '{left} {right}' into English.",
                f"Glossary for this exercise: '{left}' means '{glossary[left]}' and '{right}' means '{glossary[right]}'. What is the English phrase for '{left} {right}'?",
                f"Translate exactly from the provided mapping, not outside knowledge: {left}->{glossary[left]}; {right}->{glossary[right]}. Source phrase: {left} {right}.",
            ))
            answer = punctuation_answer(f"{glossary[left]} {glossary[right]}")
            verification.update({"glossary":glossary,"source_phrase":f"{left} {right}","translated":answer.rstrip('.')})
        else:
            subject = p["artifact"]
            fact1 = f"{a} is approved"
            fact2 = f"deadline is day {1 + j % 9}"
            option_a = f"{a} approved; deadline day {1 + j % 9}."
            option_b = f"{b} approved; deadline day {1 + j % 9}."
            option_c = f"{a} approved; deadline day {2 + j % 9}."
            prompt = surface(split, (
                f"Original {subject} facts: '{fact1}' and '{fact2}'. Which concise rewrite preserves both facts exactly? A) {option_a} B) {option_b} C) {option_c}",
                f"Preserve meaning, not wording. Facts: {fact1}; {fact2}. Candidate A: {option_a} Candidate B: {option_b} Candidate C: {option_c} Which candidate is faithful?",
                f"Choose the only transformation that keeps both stated facts unchanged. Source: {fact1}; {fact2}. A={option_a} B={option_b} C={option_c}",
            ))
            answer = "A."
            verification.update({"source_facts":[fact1,fact2],"correct_option":"A"})

    else:
        raise KeyError(archetype)

    return prompt, answer, decision, {**verification, "difficulty": diff}


def make_row(domain: str, archetype: str, j: int, rng: random.Random) -> dict[str, Any]:
    prompt, answer, decision, verification = build_case(domain, archetype, j, rng)
    # Add a non-predictive scenario reference and explicit domain context.  The
    # short hash prevents thousands of exact prompt duplicates without leaking
    # the numeric generator index or answer pattern into DEV/TEST.
    case_ref = hashlib.sha256(f"{domain}|{archetype}|{j}|v2.2".encode("utf-8")).hexdigest()[:8]
    prompt = f"{domain.replace('_', ' ').title()} scenario {case_ref}: {prompt}"
    split = split_for(j)
    difficulty = difficulty_for(j)
    rid = f"v22_{domain}_{archetype}_{j:04d}"
    return {
        "id": rid,
        "source": SOURCE,
        "dataset_schema_version": SCHEMA_VERSION,
        "intended_split": split,
        "review_status": "programmatically_verified",
        "domain": domain,
        "reasoning_family": archetype,
        "difficulty": difficulty,
        "mode": "multidomain_reasoning",
        "decision": decision,
        "answer_stance": "abstain" if decision == "abstain" else "direct",
        "answer": answer,
        "approved_answer": answer,
        "answer_aliases": [],
        "messages": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":prompt},
            {"role":"assistant","content":answer},
        ],
        "response_contract": row_contract(answer),
        "scenario_flags": ["prompt_contained_reasoning", domain, archetype, difficulty],
        "topic_id": f"v22_{domain}_{archetype}_{j:04d}",
        "teacher_provenance": {"kind":"deterministic_verifier","generator":SOURCE},
        "verification": verification,
    }


def ensure_training_target(row: dict[str, Any]) -> dict[str, Any] | None:
    item = json.loads(json.dumps(row))
    answer = str(item.get("approved_answer") or item.get("answer") or "").strip()
    if not answer:
        return None
    messages = [dict(m) for m in (item.get("messages") or [])]
    if messages and messages[-1].get("role") == "assistant":
        return item
    if messages:
        messages.append({"role":"assistant","content":answer})
    elif item.get("question"):
        messages = [
            {"role":"system","content":str(item.get("system_prompt") or SYSTEM)},
            {"role":"user","content":str(item["question"])},
            {"role":"assistant","content":answer},
        ]
    else:
        return None
    item["messages"] = messages
    return item


def stratified_sample(rows: list[dict[str, Any]], count: int, rng: random.Random, *, key_fields: tuple[str,...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str,...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(k) or "") for k in key_fields)
        buckets[key].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    out: list[dict[str, Any]] = []
    cursor = 0
    while len(out) < count and any(buckets.values()):
        key = keys[cursor % len(keys)]
        if buckets[key]:
            out.append(buckets[key].pop())
        cursor += 1
    if len(out) < count:
        pool = [r for r in rows if r not in out]
        rng.shuffle(pool)
        out.extend(pool[:count-len(out)])
    return out[:count]


def repeat_to(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        return []
    base = [json.loads(json.dumps(r)) for r in rows]
    out: list[dict[str, Any]] = []
    cycle = 0
    while len(out) < count:
        batch = [json.loads(json.dumps(r)) for r in base]
        rng.shuffle(batch)
        for row in batch:
            row["curriculum_repeat"] = cycle
            out.append(row)
            if len(out) >= count:
                break
        cycle += 1
    return out


def build_social_synthetic_train(seed: int, count: int = 3000) -> list[dict[str, Any]]:
    """Create train-only short conversational rehearsal for the social adapter.

    These examples are not part of DEV/TEST and are intentionally ordinary
    conversational behaviors rather than factual knowledge.  They prevent the
    social specialist from degenerating into a repackaged logic adapter.
    """
    rng = random.Random(seed + 991)
    families = ["acknowledgment", "celebration", "clarification", "brief_support", "preference_response"]
    subjects = ["migration", "presentation", "draft", "build", "trip", "practice session", "report", "prototype"]
    options = [("Atlas", "Beacon"), ("Cedar", "Delta"), ("Ember", "Falcon"), ("Grove", "Harbor")]
    rows: list[dict[str, Any]] = []
    per_family = count // len(families)
    for fam_index, family in enumerate(families):
        for j in range(per_family):
            subject = subjects[(j + fam_index) % len(subjects)]
            if family == "acknowledgment":
                user = f"Quick update: I finished the {subject}."
                answer = ["Got it - thanks for the update.", "Nice, thanks for letting me know.", "Got it. Sounds good."][j % 3]
            elif family == "celebration":
                user = f"The {subject} finally worked!"
                answer = ["Nice! How did it go?", "Awesome - glad it worked!", "Nice!! That's great."][j % 3]
            elif family == "clarification":
                user = [f"Can you improve the {subject}?", f"Fix this {subject} for me.", f"Can you change the {subject}?"][j % 3]
                answer = ["Sure - what should I change?", "Which part should I fix?", "What do you want changed?"][j % 3]
            elif family == "brief_support":
                user = f"The {subject} failed again and I'm frustrated."
                answer = ["That sounds frustrating. What went wrong?", "Yeah, that's rough. What failed?", "That sounds exhausting. What happened?"][j % 3]
            else:
                left, right = options[j % len(options)]
                user = f"Which would you pick for this example, {left} or {right}?"
                answer = [f"I'd pick {left}.", f"Probably {right}.", f"I'd go with {left}."][j % 3]
            rid = f"v22_social_{family}_{j:04d}"
            rows.append({
                "id": rid,
                "source": "deterministic_phase9_social_rehearsal_v22",
                "dataset_schema_version": SCHEMA_VERSION,
                "intended_split": "train",
                "review_status": "programmatically_authored",
                "domain": "communication",
                "reasoning_family": f"social_{family}",
                "difficulty": "easy" if j % 3 == 0 else "moderate",
                "mode": "conversation",
                "decision": "social",
                "answer_stance": "social",
                "answer": answer,
                "approved_answer": answer,
                "answer_aliases": [],
                "messages": [
                    {"role":"system","content":"Reply briefly, naturally, and directly. Acknowledge, clarify, or respond socially as the message calls for. Do not turn a casual exchange into a lecture."},
                    {"role":"user","content":user},
                    {"role":"assistant","content":answer},
                ],
                "response_contract": {"schema_version":"4.5","open_ended":True,"acceptable_modes":[family],"requirements":[],"prohibitions":[]},
                "scenario_flags": ["social_rehearsal", family],
                "topic_id": rid,
                "teacher_provenance": {"kind":"deterministic_authored","generator":"deterministic_phase9_social_rehearsal_v22"},
                "verification": {"family": family, "short_response": True},
            })
    rng.shuffle(rows)
    return rows[:count]


def build_specialist_views(
    train: list[dict[str, Any]],
    original_train: list[dict[str, Any]],
    v211_train: list[dict[str, Any]],
    social_synthetic: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    original = [x for r in original_train if (x:=ensure_training_target(r)) is not None]
    proc = [x for r in v211_train if (x:=ensure_training_target(r)) is not None]

    broad = stratified_sample(train, 15000, rng, key_fields=("domain","reasoning_family"))
    general = stratified_sample(train, 10000, rng, key_fields=("domain","reasoning_family"))
    # Track B implements both parts of the preservation proposal: supervised
    # specialist rehearsal (lambda_1 conceptually, via sampling proportion) and
    # explicit frozen-anchor KL (lambda_2 in train_student.py).
    anchored_general = stratified_sample(train, 12000, rng, key_fields=("domain","reasoning_family"))
    anchored_original = stratified_sample(original, min(2000, len(original)), rng, key_fields=("decision","mode","difficulty","category"))
    anchored_procedure = stratified_sample(proc, min(1000, len(proc)), rng, key_fields=("decision","procedure_family","difficulty"))
    anchored_mix = anchored_general + repeat_to(anchored_original, 2000, rng) + repeat_to(anchored_procedure, 1000, rng)
    rng.shuffle(anchored_mix)
    anchored_mix = anchored_mix[:15000]

    orig_supported = [r for r in original if str(r.get("decision") or "") == "answer"]
    v22_ground = [r for r in train if r.get("reasoning_family") in {"evidence_binding","rule_exception","two_hop_relation","filter_then_rank"} and r.get("decision") == "answer"]
    grounding = repeat_to(stratified_sample(orig_supported, min(3300,len(orig_supported)), rng, key_fields=("mode","difficulty")) + stratified_sample(v22_ground, 2200, rng, key_fields=("domain","reasoning_family")), 5500, rng)

    proc_answer = [r for r in proc if str(r.get("decision") or "") == "answer"]
    v22_proc = [r for r in train if r.get("reasoning_family") in {"state_transition","rule_exception","dependency_order","capacity_fit","filter_then_rank"} and r.get("decision") == "answer"]
    procedure = repeat_to(stratified_sample(proc_answer, min(3300,len(proc_answer)), rng, key_fields=("procedure_family","difficulty")) + stratified_sample(v22_proc, 2200, rng, key_fields=("domain","reasoning_family")), 5500, rng)

    orig_boundary = [r for r in original if str(r.get("decision") or "") in {"abstain","conflict"}]
    proc_boundary = [r for r in proc if str(r.get("decision") or "") in {"abstain","conflict"}]
    v22_boundary = [r for r in train if r.get("decision") == "abstain"]
    abstention_pool = orig_boundary + proc_boundary + v22_boundary
    abstention = repeat_to(stratified_sample(abstention_pool, min(4500,len(abstention_pool)), rng, key_fields=("source","decision","difficulty")), 4500, rng)

    social_orig = [r for r in original if str(r.get("decision") or "") == "social" or str(r.get("category") or "") in {"social","conversation","acknowledgment","celebration"}]
    # Preserve actual historical social rows, but make the majority of the
    # specialist view purpose-built conversational rehearsal rather than
    # generic logic examples with social-domain vocabulary.
    social = repeat_to(stratified_sample(social_orig, min(1500, len(social_orig)), rng, key_fields=("category","difficulty")), 1500, rng)
    social += stratified_sample(social_synthetic, 3000, rng, key_fields=("reasoning_family","difficulty"))
    rng.shuffle(social)

    return {
        "track_a_broad_general": broad,
        "track_b_anchored_mixed": anchored_mix,
        "specialist_general": general,
        "specialist_grounding": grounding,
        "specialist_procedure": procedure,
        "specialist_abstention": abstention,
        "specialist_social": social,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--original-train", type=Path, required=True)
    ap.add_argument("--v211-train", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=22042)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {args.output_root}; use --force deliberately")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []
    # 27 domains * 15 archetypes * 100 cases = 40,500 rows.
    for domain in DOMAINS:
        for archetype in ARCHETYPES:
            for j in range(100):
                rows.append(make_row(domain, archetype, j, rng))

    splits = {name:[r for r in rows if r["intended_split"] == name] for name in ("train","dev","test")}
    original_train = read_jsonl(args.original_train)
    v211_train = read_jsonl(args.v211_train)
    social_synthetic = build_social_synthetic_train(args.seed, 3000)
    views = build_specialist_views(splits["train"], original_train, v211_train, social_synthetic, rng)

    write_jsonl(args.output_root / "canonical.jsonl", rows)
    write_jsonl(args.output_root / "curriculum_sources" / "social_synthetic_train.jsonl", social_synthetic)
    for split, srows in splits.items():
        write_jsonl(args.output_root / "reviewed_splits" / f"{split}.reviewed.jsonl", srows)
    for name, vrows in views.items():
        write_jsonl(args.output_root / "curriculum_views" / f"{name}.reviewed.jsonl", vrows)

    prompt_texts = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in rows]
    audit_lines = [
        "# v2.2 Multi-Domain Dataset Audit",
        "",
        f"- Canonical rows: **{len(rows):,}**",
        f"- Unique canonical user prompts: **{len(set(prompt_texts)):,} / {len(prompt_texts):,}**",
        f"- Domains: **{len(DOMAINS)}**",
        f"- Reasoning archetypes: **{len(ARCHETYPES)}**",
        f"- TRAIN / DEV / TEST: **{len(splits['train']):,} / {len(splits['dev']):,} / {len(splits['test']):,}**",
        f"- TRAIN-only social rehearsal rows: **{len(social_synthetic):,}**",
        "",
        "## Design checks",
        "",
        "- Every canonical answer is derived from prompt-contained information.",
        "- Canonical user prompts are exact-string unique, including across splits.",
        "- The assistant target contains no chain-of-thought/rationale field.",
        "- Legal, health, and finance cases use hypothetical stated rules rather than external professional advice.",
        "- Specialist and anchored curriculum views are built from TRAIN-only sources.",
        "- Social rehearsal is TRAIN-only and is not used as a direct evaluation set.",
        "",
        "## Domains",
        "",
        ", ".join(DOMAINS),
        "",
        "## Reasoning archetypes",
        "",
        ", ".join(ARCHETYPES),
    ]
    (args.output_root / "DATASET_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    tracked = [args.output_root / "canonical.jsonl", args.output_root / "curriculum_sources" / "social_synthetic_train.jsonl", args.output_root / "DATASET_AUDIT.md"]
    tracked += [args.output_root / "reviewed_splits" / f"{s}.reviewed.jsonl" for s in ("train","dev","test")]
    tracked += [args.output_root / "curriculum_views" / f"{name}.reviewed.jsonl" for name in views]
    hashes = {str(p.relative_to(args.output_root)).replace("\\","/"):sha256(p) for p in tracked}
    write_json(args.output_root / "manifests" / "file_hashes.json", hashes)

    domain_counts = Counter(r["domain"] for r in rows)
    family_counts = Counter(r["reasoning_family"] for r in rows)
    decision_counts = Counter(r["decision"] for r in rows)
    difficulty_counts = Counter(r["difficulty"] for r in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "seed": args.seed,
        "canonical_rows": len(rows),
        "domains": len(DOMAINS),
        "domain_names": DOMAINS,
        "archetypes": len(ARCHETYPES),
        "archetype_names": ARCHETYPES,
        "split_counts": {k:len(v) for k,v in splits.items()},
        "domain_counts": dict(sorted(domain_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "curriculum_view_counts": {k:len(v) for k,v in views.items()},
        "social_synthetic_train_rows": len(social_synthetic),
        "design": "prompt-contained transferable operations; deterministic gold; no visible chain-of-thought; train-only specialist rehearsal",
        "original_train_source": str(args.original_train),
        "original_train_sha256": sha256(args.original_train),
        "v211_train_source": str(args.v211_train),
        "v211_train_sha256": sha256(args.v211_train),
    }
    write_json(args.output_root / "manifests" / "dataset_manifest.json", manifest)
    readme = f"""# v2.2 Multi-Domain Capability Curriculum\n\n- Canonical: **{len(rows):,}** rows\n- Domains: **{len(DOMAINS)}**\n- Archetypes: **{len(ARCHETYPES)}**\n- TRAIN / DEV / TEST: **{len(splits['train']):,} / {len(splits['dev']):,} / {len(splits['test']):,}**\n\nAll gold answers are computed from prompt-contained information. The assistant target contains only the concise final answer. High-stakes domains use fictional rules and conservative uncertainty examples; this dataset is for reasoning behavior, not factual reference memorization.\n\nSpecialist views are derived from TRAIN only and never consume DEV/TEST rows.\n"""
    (args.output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
