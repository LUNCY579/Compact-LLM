"""Controlled-language task rendering and strict record parsing.

Both solvers share this syntax parser, but not their optimization algorithms.
The independent solver therefore does not constitute an independent grammar audit.
"""
from __future__ import annotations
import copy
import random
import re
from typing import Any
from .common import FAMILIES, SYSTEM_FINAL, SYSTEM_REASON, dumps, loads, digest

RULES = {
"ledger_replay": "You are reconciling a fictional inventory journal. Initial records set signed balances. In increasing tick order, use only posted entries at or before cutoff. Void and later entries have no effect. add increases account by amount; scale multiplies account by amount; transfer subtracts amount from account and adds the same signed amount to other. Negative balances and negative amounts are allowed; no overdraft restriction applies. For add and scale, other is ignored. Return the requested account balance as {\"value\":integer}.",
"register_machine": "You are checking a fictional three-register controller. Initial records set X, Y, Z. Execute instructions in increasing tick order through cutoff. All arithmetic assignments use the nonnegative residue modulo modulus. add: register := register + value. mix: register := register + 2*other + value, using old values. swap: exchange register and other. conditional: if the old other is strictly greater than threshold, add value to register; otherwise subtract value. Other fields irrelevant to an operation have no effect. Return the requested register as {\"value\":integer}.",
"temporal_route": "Plan a fictional dispatch through a complete directed-network record set. For each (src,dst), the largest tick among approved versions at or before cutoff is authoritative; drafts and later versions do not replace it. If its open field is false, that edge is closed. Unlisted edges do not exist. Costs are positive integers. Start at start, visit the ordered waypoints in order, and end at goal; revisits are allowed. A waypoint is consumed whenever it is the next required waypoint and the path reaches it, including at the starting node. Minimize total cost. Any optimal path is acceptable. Return {\"path\":[node,...],\"cost\":integer}, or {\"status\":\"IMPOSSIBLE\"} if no path exists.",
"continuous_replan": "Replan a fictional dispatch after successive approved updates. The reported start is the current location; do not reuse a stale earlier route. Only currently revealed records are available. Solve the current snapshot, not an unknown future network. For each (src,dst), select the largest tick among approved versions at or before cutoff; ignore drafts and later versions. A latest open=false closes the edge. Unlisted edges do not exist. Costs are positive. Visit ordered waypoints in order and end at goal; revisits are allowed and a waypoint at start is consumed. Minimize current total cost. Any optimal path is acceptable. Return {\"path\":[node,...],\"cost\":integer}, or {\"status\":\"IMPOSSIBLE\"}.",
"prerequisite_plan": "Plan a fictional device workflow with bits Boolean flags. Decimal masks encode flag sets: flag i has weight 2**i. Initial is the starting mask. An action is applicable only when all pre_on flags are set and every pre_off flag is unset. Its effect first unsets delete flags, then sets add flags; other flags persist. All actions and costs are listed; costs are positive integers. A goal state has every goal_on flag set and every goal_off flag unset. Minimize total cost over applicable action sequences; any optimal sequence is acceptable, including an empty one when already at the goal. Return {\"actions\":[name,...],\"cost\":integer}, or {\"status\":\"IMPOSSIBLE\"} if no sequence reaches the goal.",
"deadline_schedule": "Schedule every fictional job exactly once on one non-preemptive machine, initially available at start_time. A job may start only after its release and after all jobs in after have finished. Only one job may run at a time. It must finish at or before its deadline. Idle time is allowed. For a proposed order, start each job as early as the rules allow; delaying cannot improve feasibility or the final finish. Minimize the finish time of the last job. Any optimal order is acceptable. Return {\"order\":[name,...],\"finish\":integer}, or {\"status\":\"IMPOSSIBLE\"} if there is no feasible schedule.",
"budget_selection": "Choose a subset of fictional project items. Resolve budget and capacity independently using the latest approved limit at or before cutoff; drafts and later records do not replace approved limits. If either limit has no approved record, return {\"status\":\"UNKNOWN\"}; do not assume zero or use a draft. Otherwise, total cost must not exceed budget and total weight must not exceed capacity. Every selected item's requires must also be selected. No selected item may have a selected item in its excludes. Select each item at most once; the empty subset is allowed. Maximize total value, breaking value ties by minimum total cost. Any remaining tied subset and any item order are acceptable. Return {\"items\":[name,...],\"value\":integer,\"cost\":integer}.",
"evidence_chain": "Resolve a fictional chain of organizational records. For each (entity,key), use the largest tick among approved records at or before cutoff; drafts and later records do not replace an approved record. Starting at start, follow each field in path in the given order: its value becomes the next entity, or the final answer for the last field. If a required approved field is missing, return {\"status\":\"UNKNOWN\"}; do not infer it from a draft or a similar name. Otherwise return the final string as {\"answer\":string}.",
}
TARGET_WORDS = [512, 1024, 2048, 4096, 8192, 16384]
LAYOUTS = ("early", "middle", "late", "distributed")


def canonical_case(w: dict[str, Any]) -> dict[str, Any]:
    return {"family": w["family"], "query": w["query"], "records": sorted(w["records"], key=dumps)}


def render(w: dict[str, Any], rng: random.Random, tier: int, layout: str, *, mode: str = "final_only",
           scope: str | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if layout not in LAYOUTS or tier not in range(6):
        raise ValueError("Invalid layout or tier")
    scope = scope or f"W{rng.getrandbits(40):010x}"
    seen_ids: set[str] = set()
    def record_line(rec: dict[str, Any], rec_scope: str) -> str:
        rid = f"R{rng.getrandbits(48):012x}"
        while rid in seen_ids:
            rid = f"R{rng.getrandbits(48):012x}"
        seen_ids.add(rid)
        fields = list(rec.items())
        rng.shuffle(fields)
        return "Record " + rid + " | scope=" + dumps(rec_scope) + " | " + " | ".join(k + "=" + dumps(v) for k, v in fields)
    core = [record_line(rec, scope) for rec in w["records"]]
    rng.shuffle(core)
    header = (f"TASK {w['family']}\n{RULES[w['family']]}\n"
              "Records below use JSON values after field names. Record IDs are arbitrary; display order is not time order. "
              "Only records with exactly the requested scope belong to this task. Other scopes are separate workspaces.\n"
              f"SCOPE {dumps(scope)}\n")
    footer = "\nQUERY " + dumps(w["query"]) + "\nResolve the task from the complete supplied record set."
    current_words = len((header + "\n".join(core) + footer).split())
    distractors = []
    # Decoys have the same family grammar and overlapping values, but different scope.
    scopes = [f"W{rng.getrandbits(40):010x}" for _ in range(5)]
    scopes = [x if x != scope else x + "d" for x in scopes]
    while current_words < TARGET_WORDS[tier]:
        rec = copy.deepcopy(rng.choice(w["records"]))
        if "tick" in rec:
            rec["tick"] = rng.randint(1, 1000)
        if "amount" in rec:
            rec["amount"] = rng.randint(-30, 40)
        line = record_line(rec, rng.choice(scopes))
        distractors.append(line)
        current_words += len(line.split())
    if layout == "early":
        lines = core + distractors
    elif layout == "late":
        lines = distractors + core
    elif layout == "middle":
        k = len(distractors) // 2
        lines = distractors[:k] + core + distractors[k:]
    else:
        lines = core + distractors
        rng.shuffle(lines)
    text = header + "\n".join(lines) + footer
    core_set = set(core)
    positions = []
    offset = len(header)
    for line in lines:
        if line in core_set:
            positions.append([offset, offset + len(line)])
        offset += len(line) + 1
    if mode not in ("final_only", "reason_then_final"):
        raise ValueError("Unknown mode")
    prompt = [{"role": "system", "content": SYSTEM_FINAL if mode == "final_only" else SYSTEM_REASON},
              {"role": "user", "content": text}]
    meta = {"layout": layout, "target_words": TARGET_WORDS[tier], "input_words": len(text.split()),
            "input_characters": len(text), "token_count": None, "target_records": len(core),
            "distractor_records": len(distractors), "target_character_spans": positions,
            "scope": scope, "case_sha256": digest(canonical_case(w))}
    return prompt, meta


_LINE = re.compile(r"^Record (R[0-9a-f]+) \| (.+)$")


def parse_prompt(prompt: list[dict[str, str]]) -> dict[str, Any]:
    if len(prompt) != 2 or [m.get("role") for m in prompt] != ["system", "user"]:
        raise ValueError("Expected one system and one user message")
    if prompt[0].get("content") not in (SYSTEM_FINAL, SYSTEM_REASON):
        raise ValueError("Unrecognized system policy")
    text = prompt[1]["content"]
    if not isinstance(text, str) or len(text) > 4_000_000:
        raise ValueError("Prompt is not bounded text")
    family = scope = query = None
    records = []
    record_ids: set[str] = set()
    for line in text.splitlines():
        if line.startswith("TASK "):
            if family is not None:
                raise ValueError("Duplicate TASK")
            family = line[5:]
        elif line.startswith("SCOPE "):
            if scope is not None:
                raise ValueError("Duplicate SCOPE")
            scope = loads(line[6:])
        elif line.startswith("QUERY "):
            if query is not None:
                raise ValueError("Duplicate QUERY")
            query = loads(line[6:])
        elif line.startswith("Record "):
            match = _LINE.fullmatch(line)
            if not match or match[1] in record_ids:
                raise ValueError("Malformed or duplicate record ID")
            record_ids.add(match[1])
            record = {}
            for field in match[2].split(" | "):
                if "=" not in field:
                    raise ValueError("Malformed record field")
                key, value = field.split("=", 1)
                if key in record or not re.fullmatch("[a-z_]+", key):
                    raise ValueError("Duplicate or invalid record key")
                record[key] = loads(value)
            records.append(record)
    if family not in FAMILIES or not isinstance(scope, str) or not isinstance(query, dict):
        raise ValueError("Missing task declaration")
    if RULES[family] not in text:
        raise ValueError("Rules were changed or removed")
    selected = []
    for record in records:
        if "scope" not in record:
            raise ValueError("Record has no scope")
        rec_scope = record.pop("scope")
        if rec_scope == scope:
            selected.append(record)
    if not selected:
        raise ValueError("No target records")
    return {"family": family, "query": query, "records": selected}
