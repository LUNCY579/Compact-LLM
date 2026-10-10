"""Offline closed-world compiler; no model, tokenizer, or confirmation access.

Two teachers consume only the rendered messages. Their evaluation algorithms are
separate; shared code only parses the public payload and renders response formats.
Generator-family separation is mechanical, not a reviewed generalization claim.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import reduce
import hashlib
import json
from pathlib import Path
import random

from .common import DOMAIN_ROLE, TRAIN_COUNTS, canonical, digest, new_directory, write_json, write_jsonl


FAMILIES = {
    "train": {"reasoning": "train_register_program_v1", "transition": "train_item_event_history_v1",
              "arithmetic": "train_weighted_baskets_v1", "grounding": "train_authority_lookup_v1",
              "long_context": "train_timestamp_record_lookup_v1", "format": "train_field_transform_v1"},
    "pilot": {"reasoning": "pilot_expression_dependencies_v1", "transition": "pilot_transition_table_v1",
              "arithmetic": "pilot_euclidean_sharing_v1", "grounding": "pilot_horn_support_v1",
              "long_context": "pilot_alias_record_join_v1", "format": "pilot_filtered_roster_v1"},
}
FORMATS = ("plain", "json", "tag")
KINDS = {"train": {"reasoning": "program", "transition": "item_events", "arithmetic": "baskets",
                   "grounding": "authority", "long_context": "latest_record", "format": "field_transform"},
         "pilot": {"reasoning": "dependencies", "transition": "fsm", "arithmetic": "sharing",
                   "grounding": "horn", "long_context": "alias_join", "format": "filtered_roster"}}
BEGIN, END = "TASK_JSON\n", "\nEND_TASK"
# Prospective shape variation only; actual bound-tokenizer profiling remains a
# separate offline gate. The multiplier permutes all 32 parent length levels.
PILOT_JOIN_RECORD_COUNTS = tuple(50 + 2 * ((index * 17) % 32) for index in range(32))


def render_messages(task: dict, requested_format: str = "plain") -> list[dict]:
    if requested_format not in FORMATS:
        raise ValueError("unknown response format")
    instructions = {
        "plain": "Return the answer alone with no surrounding text.",
        "json": 'Return exactly one JSON object with one string field named "answer".',
        "tag": "Return exactly <answer>VALUE</answer> with no surrounding text.",
    }
    rules = {
        "program": "Execute register instructions in list order. Return the queried register's final integer.",
        "dependencies": "Evaluate the queried expression dependency with exact integer arithmetic.",
        "item_events": "Apply the item moves in list order. Return the queried item's final bin.",
        "fsm": "Start at initial; apply the first query.after_symbols input symbols using the transition table. Return that state.",
        "baskets": "Sum quantity times unit price for the queried basket. Return the integer total.",
        "sharing": "Divide the queried group's count among its people. Return the requested integer quotient or remainder.",
        "authority": "Only authoritative_source can establish the queried entity/attribute. Conflicting authoritative values are UNKNOWN.",
        "horn": "Infer from facts and rules. Return SUPPORTED iff the queried atom follows; otherwise UNKNOWN.",
        "latest_record": "Choose the queried entity's greatest timestamp record and return the queried field.",
        "alias_join": "Resolve the query alias to its code, then retrieve its unique record's queried field.",
        "field_transform": "Select fields in query.keys order, apply query.case upper/lower, and join with query.separator.",
        "filtered_roster": "Select records with query.tier and score >= query.minimum; order labels lexicographically using query.order. Join with commas; empty is NONE.",
    }[task["kind"]]
    rules = "Use only these closed-world inputs; lists are exhaustive. Missing required facts return UNKNOWN. " + rules
    return [{"role": "system", "content": rules},
            {"role": "user", "content": instructions[requested_format] + "\n" + BEGIN +
             canonical(task).decode("utf-8") + END}]


def parse_prompt(messages: list[dict]) -> dict:
    """Read the legitimate task/query from messages; never accept a row/world."""
    if not isinstance(messages, list) or not messages:
        raise TypeError("teachers accept a message list only")
    payloads = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("invalid rendered message")
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("message content must be text")
        if message["role"] == "user" and BEGIN in content:
            if content.count(BEGIN) != 1 or content.count(END) != 1:
                raise ValueError("ambiguous task payload")
            raw = content.split(BEGIN, 1)[1].split(END, 1)[0]
            payloads.append(json.loads(raw))
    if len(payloads) != 1 or not isinstance(payloads[0], dict) or "query" not in payloads[0]:
        raise ValueError("exactly one public task and query required")
    return payloads[0]


def format_target(answer: str, requested_format: str) -> str:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if requested_format == "plain":
        return answer
    if requested_format == "json":
        return canonical({"answer": answer}).decode("utf-8")
    if requested_format == "tag":
        return "<answer>" + answer + "</answer>"
    raise ValueError("unknown response format")


def solve_prompt_a(messages: list[dict]) -> str:
    """Forward execution, iterative closure, and direct indexed arithmetic."""
    t = parse_prompt(messages)
    k, q = t["kind"], t["query"]
    if k == "program":
        state = dict(t["initial"])
        for ins in t["instructions"]:
            dst, op, arg = ins["dst"], ins["op"], ins["arg"]
            if op == "add": state[dst] += arg
            elif op == "multiply": state[dst] *= arg
            elif op == "add_from": state[dst] += state[arg]
            elif op == "if_even_add":
                if state[dst] % 2 == 0: state[dst] += arg
            else: raise ValueError("unknown program instruction")
        return str(state[q]) if q in state else "UNKNOWN"
    if k == "dependencies":
        values, pending = dict(t["leaves"]), dict(t["expressions"])
        while pending:
            progressed = False
            for name, expr in list(pending.items()):
                left, op, right = expr
                if left in values and right in values:
                    a, b = values[left], values[right]
                    values[name] = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else None
                    if values[name] is None: raise ValueError("unknown expression operator")
                    del pending[name]; progressed = True
            if not progressed: break
        return str(values[q]) if q in values else "UNKNOWN"
    if k == "item_events":
        locations = dict(t["initial"])
        for event in t["events"]: locations[event["item"]] = event["bin"]
        return locations.get(q, "UNKNOWN")
    if k == "fsm":
        state = t["initial"]
        for symbol in t["symbols"][:q["after_symbols"]]:
            state = t["transitions"].get(state, {}).get(symbol)
            if state is None: return "UNKNOWN"
        return state
    if k == "baskets":
        if q not in t["baskets"]: return "UNKNOWN"
        return str(sum(term["quantity"] * term["price"] for term in t["baskets"][q]))
    if k == "sharing":
        group = t["groups"].get(q["group"])
        if group is None: return "UNKNOWN"
        quotient, remainder = divmod(group["count"], group["people"])
        return str({"quotient": quotient, "remainder": remainder}[q["component"]])
    if k == "authority":
        values = {f["value"] for f in t["facts"] if f["source"] == t["authoritative_source"]
                  and f["entity"] == q["entity"] and f["attribute"] == q["attribute"]}
        return next(iter(values)) if len(values) == 1 else "UNKNOWN"
    if k == "horn":
        known = set(t["facts"])
        while True:
            previous = set(known)
            for rule in t["rules"]:
                if set(rule["premises"]) <= known: known.add(rule["conclusion"])
            if known == previous: break
        return "SUPPORTED" if q in known else "UNKNOWN"
    if k == "latest_record":
        rows = [r for r in t["records"] if r["entity"] == q["entity"]]
        if not rows: return "UNKNOWN"
        return str(max(rows, key=lambda r: r["timestamp"]).get(q["field"], "UNKNOWN"))
    if k == "alias_join":
        code = t["aliases"].get(q["alias"])
        rows = [r for r in t["records"] if r["code"] == code]
        return str(rows[0].get(q["field"], "UNKNOWN")) if len(rows) == 1 else "UNKNOWN"
    if k == "field_transform":
        if any(key not in t["fields"] for key in q["keys"]): return "UNKNOWN"
        values = [t["fields"][key] for key in q["keys"]]
        values = [v.upper() if q["case"] == "upper" else v.lower() for v in values]
        return q["separator"].join(values)
    if k == "filtered_roster":
        labels = [r["label"] for r in t["records"] if r["tier"] == q["tier"] and r["score"] >= q["minimum"]]
        labels.sort(reverse=q["order"] == "desc")
        return ",".join(labels) if labels else "NONE"
    raise ValueError("unknown closed-world task kind")


def solve_prompt_b(messages: list[dict]) -> str:
    """Independent composition, recursive proof, scans, and repeated addition."""
    t = parse_prompt(messages)
    k, q = t["kind"], t["query"]
    if k == "program":
        functions = []
        for ins in t["instructions"]:
            dst, op, arg = ins["dst"], ins["op"], ins["arg"]
            def transformation(s, dst=dst, op=op, arg=arg):
                previous = s[dst]
                if op == "add": replacement = previous + arg
                elif op == "multiply": replacement = previous * arg
                elif op == "add_from": replacement = previous + s[arg]
                elif op == "if_even_add": replacement = previous + (arg if not previous & 1 else 0)
                else: raise ValueError("unknown program instruction")
                return {**s, dst: replacement}
            functions.append(transformation)
        final = reduce(lambda state, fn: fn(state), functions, t["initial"])
        return str(final[q]) if q in final else "UNKNOWN"
    if k == "dependencies":
        def evaluate(name, visited):
            if name in visited: raise ValueError("cyclic dependency")
            if name in t["leaves"]: return t["leaves"][name]
            if name not in t["expressions"]: raise KeyError(name)
            left, op, right = t["expressions"][name]
            a, b = evaluate(left, visited | {name}), evaluate(right, visited | {name})
            if op == "+": return a + b
            if op == "-": return a - b
            if op == "*": return a * b
            raise ValueError("unknown expression operator")
        try: return str(evaluate(q, set()))
        except KeyError: return "UNKNOWN"
    if k == "item_events":
        for event in reversed(t["events"]):
            if event["item"] == q: return event["bin"]
        return t["initial"].get(q, "UNKNOWN")
    if k == "fsm":
        # Compose mappings for every state, independently of a single-state walk.
        composed = {state: state for state in t["transitions"]}
        for symbol in t["symbols"][:q["after_symbols"]]:
            composed = {origin: t["transitions"].get(current, {}).get(symbol)
                        for origin, current in composed.items()}
        return composed.get(t["initial"]) or "UNKNOWN"
    if k == "baskets":
        if q not in t["baskets"]: return "UNKNOWN"
        terms = [price for r in t["baskets"][q] for price in [r["price"]] * r["quantity"]]
        total = 0
        for price in terms: total += price
        return str(total)
    if k == "sharing":
        group = t["groups"].get(q["group"])
        if group is None: return "UNKNOWN"
        remaining, allocated = group["count"], 0
        while remaining >= group["people"]:
            remaining -= group["people"]; allocated += 1
        return str(allocated if q["component"] == "quotient" else remaining)
    if k == "authority":
        answer = None
        for fact in t["facts"]:
            if (fact["entity"], fact["attribute"], fact["source"]) == (q["entity"], q["attribute"], t["authoritative_source"]):
                if answer is not None and answer != fact["value"]: return "UNKNOWN"
                answer = fact["value"]
        return answer if answer is not None else "UNKNOWN"
    if k == "horn":
        def prove(atom, path):
            if atom in t["facts"]: return True
            if atom in path: return False
            for rule in t["rules"]:
                if rule["conclusion"] == atom and all(prove(premise, path | {atom}) for premise in rule["premises"]):
                    return True
            return False
        return "SUPPORTED" if prove(q, set()) else "UNKNOWN"
    if k == "latest_record":
        current = None
        for record in t["records"]:
            if record["entity"] == q["entity"] and (current is None or record["timestamp"] > current["timestamp"]):
                current = record
        return str(current.get(q["field"], "UNKNOWN")) if current is not None else "UNKNOWN"
    if k == "alias_join":
        matches = []
        for record in t["records"]:
            names = [alias for alias, code in t["aliases"].items() if code == record["code"]]
            if q["alias"] in names: matches.append(record)
        return str(matches[0].get(q["field"], "UNKNOWN")) if len(matches) == 1 else "UNKNOWN"
    if k == "field_transform":
        answer = ""
        for key in q["keys"]:
            if key not in t["fields"]: return "UNKNOWN"
            word = t["fields"][key]
            transformed = "".join(char.upper() if q["case"] == "upper" else char.lower() for char in word)
            answer = transformed if not answer else answer + q["separator"] + transformed
        return answer
    if k == "filtered_roster":
        # Repeated selection instead of list sorting/filtering.
        eligible = {r["label"] for r in t["records"] if not (r["tier"] != q["tier"] or r["score"] < q["minimum"])}
        ordered = []
        while eligible:
            chosen = max(eligible) if q["order"] == "desc" else min(eligible)
            eligible.remove(chosen); ordered.append(chosen)
        return ",".join(ordered) or "NONE"
    raise ValueError("unknown closed-world task kind")


def _word(rng: random.Random, prefix: str = "w") -> str:
    return prefix + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz") for _ in range(6))


def _pair_tasks(split: str, domain: str, rng: random.Random, parent_index: int) -> tuple[dict, dict, str]:
    changed = parent_index % 2 == 0 or domain == "grounding"
    kind = KINDS[split][domain]
    if kind == "program":
        q = rng.choice(["x", "y"])
        first = {"kind": kind, "initial": {"x": rng.randint(1, 20), "y": rng.randint(1, 20), "z": 3}, "query": q,
                 "instructions": [{"dst": q, "op": "add", "arg": rng.randint(2, 9)},
                                  {"dst": "y" if q == "x" else "x", "op": "multiply", "arg": 2},
                                  {"dst": q, "op": "add_from", "arg": "y" if q == "x" else "x"},
                                  {"dst": q, "op": "multiply", "arg": rng.randint(2, 4)},
                                  {"dst": q, "op": "if_even_add", "arg": 3},
                                  {"dst": q, "op": "add", "arg": -rng.randint(1, 8)}]}
        second = deepcopy(first)
        if changed: second["initial"][q] += 17
        else: second["instructions"].append({"dst": "z", "op": "add", "arg": 29})
    elif kind == "dependencies":
        first = {"kind": kind, "leaves": {"a": rng.randint(1, 30), "b": rng.randint(1, 30), "c": rng.randint(2, 8)},
                 "expressions": {"d": ["a", "+", "b"], "e": ["d", "*", "c"], "f": ["e", "-", "b"]},
                 "query": rng.choice(["e", "f"])}
        second = deepcopy(first)
        if changed: second["leaves"]["a"] += 17
        else: second["expressions"]["unused"] = ["b", "+", "c"]
    elif kind == "item_events":
        items, bins = [_word(rng, "i") for _ in range(6)], ["north", "east", "west"]
        q = rng.choice(items)
        events = [{"item": rng.choice(items), "bin": rng.choice(bins)} for _ in range(13)]
        events.append({"item": q, "bin": rng.choice(bins)})
        first = {"kind": kind, "initial": {i: rng.choice(bins) for i in items}, "events": events, "query": q}
        second = deepcopy(first)
        if changed: second["events"][-1]["bin"] = bins[(bins.index(events[-1]["bin"]) + 1) % 3]
        else: second["events"].append({"item": next(i for i in items if i != q), "bin": "north"})
    elif kind == "fsm":
        states = ["s0", "s1", "s2", "s3"]
        table = {s: {"a": rng.choice(states), "b": rng.choice(states), "z": states[(i + 1) % 4]} for i, s in enumerate(states)}
        first = {"kind": kind, "transitions": table, "initial": rng.choice(states), "symbols": [rng.choice(["a", "b"]) for _ in range(12)], "query": {"after_symbols": rng.randint(4, 9)}}
        second = deepcopy(first)
        if changed:
            second["symbols"].insert(first["query"]["after_symbols"], "z")
            second["query"]["after_symbols"] += 1
        else: second["transitions"]["unreachable"] = {"a": "unreachable", "b": "unreachable", "z": "unreachable"}
    elif kind == "baskets":
        names = ["cedar", "maple"]
        first = {"kind": kind, "baskets": {name: [{"quantity": rng.randint(2, 7), "price": rng.randint(2, 19)} for _ in range(4)] for name in names}, "query": rng.choice(names)}
        second = deepcopy(first)
        basket = first["query"] if changed else next(name for name in names if name != first["query"])
        second["baskets"][basket][0]["quantity"] += 1
    elif kind == "sharing":
        names = ["orchard", "harbor"]
        q = {"group": rng.choice(names), "component": "quotient" if parent_index % 4 < 2 else "remainder"}
        first = {"kind": kind, "groups": {name: {"count": rng.randint(40, 190), "people": rng.randint(3, 11)} for name in names}, "query": q}
        second = deepcopy(first)
        group = q["group"] if changed else next(name for name in names if name != q["group"])
        second["groups"][group]["count"] += second["groups"][group]["people"] if q["component"] == "quotient" else 1
    elif kind == "authority":
        entities = [_word(rng, "e") for _ in range(5)]
        q = {"entity": rng.choice(entities), "attribute": rng.choice(["color", "zone"])}
        value = ["amber", "blue", "green", "violet"][parent_index % 4]
        facts = [{"entity": e, "attribute": "status", "value": _word(rng), "source": "charter"} for e in entities]
        facts.extend([{**q, "value": value, "source": "charter"}, {**q, "value": "rumor_only", "source": "rumor"}])
        rng.shuffle(facts)
        first = {"kind": kind, "authoritative_source": "charter", "facts": facts, "query": q}
        second = deepcopy(first)
        second["facts"] = [f for f in second["facts"] if not (f["entity"] == q["entity"] and f["attribute"] == q["attribute"] and f["source"] == "charter")]
    elif kind == "horn":
        seed, middle, goal, noise = [_word(rng, "p") for _ in range(4)]
        first = {"kind": kind, "facts": [seed, noise],
                 "rules": [{"premises": [seed], "conclusion": middle}, {"premises": [middle, noise], "conclusion": goal},
                           {"premises": ["absent_atom"], "conclusion": _word(rng, "p")}], "query": goal}
        second = deepcopy(first); second["facts"].remove(seed)
    elif kind == "latest_record":
        entities = [_word(rng, "e") for _ in range(6)]
        q = {"entity": rng.choice(entities), "field": "value"}
        records = [{"entity": rng.choice(entities), "timestamp": i + 1, "value": _word(rng, "v")} for i in range(19)]
        records.append({"entity": q["entity"], "timestamp": 30, "value": _word(rng, "v")})
        rng.shuffle(records)
        first = {"kind": kind, "records": records, "query": q}; second = deepcopy(first)
        if changed:
            for r in second["records"]:
                if r["timestamp"] == 30: r["value"] += "x"
        else: second["records"].append({"entity": next(e for e in entities if e != q["entity"]), "timestamp": 40, "value": _word(rng, "v")})
    elif kind == "alias_join":
        codes, aliases = [_word(rng, "k") for _ in range(18)], [_word(rng, "a") for _ in range(18)]
        records = [{"code": code, "city": _word(rng, "c"), "color": _word(rng, "v")} for code in codes]
        rng.shuffle(records)
        first = {"kind": kind, "aliases": dict(zip(aliases, codes)), "records": records,
                 "query": {"alias": rng.choice(aliases), "field": rng.choice(["city", "color"])}}
        second = deepcopy(first)
        if changed:
            selected = first["aliases"][first["query"]["alias"]]
            for r in second["records"]:
                if r["code"] == selected: r[first["query"]["field"]] += "x"
        else: second["aliases"][_word(rng, "a")] = codes[0]
        # Preserve the existing base task and its query/support exactly. Only
        # irrelevant legitimate records are added. An independent padding RNG
        # keeps the next parent's original generator stream unchanged.
        padding_rng = random.Random(int(digest([first, parent_index, "pilot_join_padding_v1"])[:16], 16))
        target_count = PILOT_JOIN_RECORD_COUNTS[parent_index % len(PILOT_JOIN_RECORD_COUNTS)]
        used_codes = set(codes)
        used_aliases = set(first["aliases"]) | set(second["aliases"])
        added_records, added_aliases = [], {}
        for _ in range(target_count - len(records)):
            code = _word(padding_rng, "k")
            while code in used_codes: code = _word(padding_rng, "k")
            used_codes.add(code)
            alias = _word(padding_rng, "a")
            while alias in used_aliases: alias = _word(padding_rng, "a")
            used_aliases.add(alias)
            added_records.append({"code": code, "city": _word(padding_rng, "c"), "color": _word(padding_rng, "v")})
            added_aliases[alias] = code
        order = list(range(target_count))
        padding_rng.shuffle(order)
        for task in (first, second):
            task["aliases"].update(added_aliases)
            combined = task["records"] + deepcopy(added_records)
            task["records"] = [combined[index] for index in order]
    elif kind == "field_transform":
        keys = ["title", "token", "region", "note"]
        selected = rng.sample(keys, 2)
        first = {"kind": kind, "fields": {key: _word(rng) for key in keys},
                 "query": {"keys": selected, "case": "upper" if parent_index % 2 else "lower", "separator": rng.choice(["|", ";", ":"])}}
        second = deepcopy(first)
        key = selected[0] if changed else next(key for key in keys if key not in selected)
        second["fields"][key] += "x"
    elif kind == "filtered_roster":
        minimum, tier = rng.randint(2, 7), rng.choice(["A", "B"])
        records = [{"label": _word(rng, "n"), "tier": rng.choice(["A", "B"]), "score": rng.randint(0, 10)} for _ in range(7)]
        records.append({"label": _word(rng, "n"), "tier": tier, "score": minimum})
        first = {"kind": kind, "records": records, "query": {"tier": tier, "minimum": minimum, "order": "asc" if parent_index % 2 else "desc"}}
        second = deepcopy(first)
        second["records"].append({"label": _word(rng, "n"), "tier": tier, "score": minimum if changed else minimum - 1})
    else: raise ValueError("unknown generator")
    return first, second, ("changed_answer_support_removal" if domain == "grounding" else "changed_answer_counterfactual" if changed else "unchanged_answer_distractor")


def _validate_split(split: str) -> None:
    if split == "confirmation":
        raise PermissionError("confirmation is closed; no examples will be materialized")
    if split not in FAMILIES:
        raise ValueError("split must be train or pilot")


def generate_bank(split: str, seed: int = 17) -> list[dict]:
    _validate_split(split)
    if type(seed) is not int: raise TypeError("seed must be an integer")
    rows, seen_content = [], set()
    counts = TRAIN_COUNTS if split == "train" else {domain: 64 for domain in DOMAIN_ROLE}
    for domain_index, (domain, count) in enumerate(counts.items()):
        rng = random.Random(int(digest([seed, split, domain])[:16], 16))
        for parent_index in range(count // 2):
            requested = FORMATS[(parent_index + domain_index) % 3]
            # Regeneration is solely a content-deduplication rule; no score,
            # teacher answer, or pilot performance enters this choice.
            for duplicate_attempt in range(100):
                first, second, pair_kind = _pair_tasks(split, domain, rng, parent_index)
                signatures = [digest(render_messages(task, requested)) for task in (first, second)]
                if len(set(signatures)) == 2 and not (set(signatures) & seen_content):
                    seen_content.update(signatures)
                    break
            else:
                raise RuntimeError("generator failed the mechanical content-uniqueness bound")
            pair_answers = []
            for variant, task in enumerate((first, second)):
                messages = render_messages(task, requested)
                answer_a, answer_b = solve_prompt_a(messages), solve_prompt_b(messages)
                if answer_a != answer_b: raise AssertionError("independent teachers disagree")
                pair_answers.append(answer_a)
                parent = f"{split}.{domain}.{parent_index:04d}"
                rows.append({"case_id": parent + f".v{variant}", "parent_id": parent,
                             "family_id": FAMILIES[split][domain], "domain": domain, "role": DOMAIN_ROLE[domain],
                             "messages": messages, "canonical_answer": answer_a,
                             "target": format_target(answer_a, requested), "requested_format": requested,
                             "variant_id": variant, "pair_kind": pair_kind,
                             "solver_trace": {"source": "rendered_messages_only", "teacher_a": answer_a,
                                              "teacher_b": answer_b, "agreement": True}})
            expected_changed = pair_kind.startswith("changed")
            if (pair_answers[0] != pair_answers[1]) != expected_changed:
                raise AssertionError("counterfactual/distractor does not satisfy its registered relationship")
            if domain == "grounding" and pair_answers[1] != "UNKNOWN":
                raise AssertionError("support removal must make the query unanswerable")
    # Ordering is unrelated to answers; formats never change between paired variants.
    random.Random(seed ^ (101 if split == "train" else 202)).shuffle(rows)
    validate_bank(rows, split)
    return rows


def validate_bank(rows: list[dict], split: str) -> dict:
    _validate_split(split)
    expected = TRAIN_COUNTS if split == "train" else {domain: 64 for domain in DOMAIN_ROLE}
    if Counter(row["domain"] for row in rows) != Counter(expected): raise ValueError("domain count mismatch")
    if len({row["case_id"] for row in rows}) != len(rows): raise ValueError("duplicate case IDs")
    content = [digest(row["messages"]) for row in rows]
    if len(set(content)) != len(content): raise ValueError("duplicate rendered task content")
    parents = defaultdict(list)
    for row in rows:
        if row["family_id"] != FAMILIES[split][row["domain"]]: raise ValueError("generator family mismatch")
        if row["role"] != DOMAIN_ROLE[row["domain"]]: raise ValueError("domain role mismatch")
        if solve_prompt_a(row["messages"]) != row["canonical_answer"] or solve_prompt_b(row["messages"]) != row["canonical_answer"]:
            raise ValueError("gold/teacher agreement mismatch")
        if row["target"] != format_target(row["canonical_answer"], row["requested_format"]): raise ValueError("target formatting mismatch")
        if row["case_id"] in canonical(row["messages"]).decode() or row["parent_id"] in canonical(row["messages"]).decode():
            raise ValueError("ID leaked into model inputs")
        parents[row["parent_id"]].append(row)
    for pair in parents.values():
        if len(pair) != 2 or {r["variant_id"] for r in pair} != {0, 1}: raise ValueError("pair must contain exactly two variants")
        if len({r["requested_format"] for r in pair}) != 1 or len({r["pair_kind"] for r in pair}) != 1: raise ValueError("pair contract differs")
        changed = pair[0]["canonical_answer"] != pair[1]["canonical_answer"]
        if changed != pair[0]["pair_kind"].startswith("changed"): raise ValueError("pair relationship mismatch")
    return {"rows": len(rows), "parents": len(parents), "domain_counts": dict(Counter(r["domain"] for r in rows)),
            "role_counts": dict(Counter(r["role"] for r in rows)), "format_counts": dict(Counter(r["requested_format"] for r in rows)),
            "content_duplicate_count": 0, "teacher_agreement_count": len(rows)}


def partition_audit(train: list[dict], pilot: list[dict]) -> dict:
    result = {}
    for field in ("case_id", "parent_id", "family_id"):
        common = {r[field] for r in train} & {r[field] for r in pilot}
        result[field + "_overlap_count"] = len(common)
        if common: raise ValueError("split identity/family overlap")
    overlap = {digest(r["messages"]) for r in train} & {digest(r["messages"]) for r in pilot}
    result["rendered_content_overlap_count"] = len(overlap)
    if overlap: raise ValueError("split rendered-content overlap")
    result.update(generator_mechanisms={split: KINDS[split] for split in KINDS}, overlap_review="UNREVIEWED",
                  semantic_generalization_established=False)
    return result


def _blind_query(t: dict):
    k = t["kind"]
    if k == "program": return sorted(t["initial"])[0]
    if k == "dependencies": return sorted(t["leaves"])[0]
    if k == "item_events": return sorted(t["initial"])[0]
    if k == "fsm": return {"after_symbols": 0}
    if k == "baskets": return sorted(t["baskets"])[0]
    if k == "sharing": return {"group": sorted(t["groups"])[0], "component": "quotient"}
    if k == "authority": return {"entity": t["facts"][0]["entity"], "attribute": t["facts"][0]["attribute"]}
    if k == "horn": return t["facts"][0] if t["facts"] else "absent"
    if k == "latest_record": return {"entity": t["records"][0]["entity"], "field": "value"}
    if k == "alias_join": return {"alias": sorted(t["aliases"])[0], "field": "city"}
    if k == "field_transform": return {"keys": [sorted(t["fields"])[0]], "case": "lower", "separator": "|"}
    if k == "filtered_roster": return {"tier": "A", "minimum": 0, "order": "asc"}
    raise ValueError("unknown task")


def floor_audit(rows: list[dict]) -> dict:
    """Offline diagnostic shortcuts; raw-answer scores, not model evidence."""
    output = {}
    for domain in DOMAIN_ROLE:
        subset = [r for r in rows if r["domain"] == domain]
        constant_count = max(Counter(r["canonical_answer"] for r in subset).values())
        copied, blind = 0, 0
        for row in subset:
            task = parse_prompt(row["messages"])
            copy_prediction = str(task["query"]) if isinstance(task["query"], str) else canonical(task["query"]).decode()
            copied += copy_prediction == row["canonical_answer"]
            task["query"] = _blind_query(task)
            blind += solve_prompt_b(render_messages(task)) == row["canonical_answer"]
        output[domain] = {"rows": len(subset), "retrospective_best_constant_correct": constant_count,
                          "copy_query_correct": copied, "query_blind_first_available_correct": blind,
                          "constant_accuracy": constant_count / len(subset), "copy_accuracy": copied / len(subset),
                          "query_blind_accuracy": blind / len(subset)}
    return {"evidence_kind": "OFFLINE_SYNTHETIC_SHORTCUT_DIAGNOSTIC_NOT_NATIVE",
            "status": "UNREVIEWED_NO_ACCEPTANCE_GATE", "scope": "raw answers; best constant is retrospective within domain",
            "limitation": "These three floors do not rule out other shortcuts; they are diagnostic and have no acceptance gate.",
            "domains": output}


def build_dataset(out, seed: int = 17) -> dict:
    train, pilot = generate_bank("train", seed), generate_bank("pilot", seed)
    train_check, pilot_check = validate_bank(train, "train"), validate_bank(pilot, "pilot")
    partition = partition_audit(train, pilot)
    train_floors, pilot_floors = floor_audit(train), floor_audit(pilot)
    prompts = [{"case_id": row["case_id"], "messages": row["messages"]} for row in pilot]
    references = [{key: value for key, value in row.items() if key != "messages"} for row in pilot]
    path = new_directory(out)
    write_jsonl(path / "train.jsonl", train)
    write_jsonl(path / "pilot.prompts.jsonl", prompts)
    write_jsonl(path / "pilot.references.jsonl", references)
    artifact_hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                       for name in ("train.jsonl", "pilot.prompts.jsonl", "pilot.references.jsonl")}
    manifest = {"schema": "TITAN_OFFLINE_DATA_V1", "status": "DRAFT_UNREVIEWED_NOT_FROZEN", "seed": seed,
                "train": train_check, "pilot": pilot_check, "partition": partition,
                "teacher": {"kind": "two_algorithmically_separate_prompt_only_symbolic_solvers",
                            "source_digest": digest({"data.py": Path(__file__).read_text(encoding="utf-8")}),
                            "agreement_train": len(train), "agreement_pilot": len(pilot), "model_teacher_calls": 0,
                            "independence_review": "UNREVIEWED"},
                "artifact_sha256": artifact_hashes, "corpus_digest": digest({"train": train, "pilot": pilot}),
                "floors": {"train": train_floors, "pilot": pilot_floors},
                "input_projection": "pilot prompts contain case_id and messages only; IDs are not rendered inputs",
                "targets": "concise truth targets; EOS must be appended later by the bound tokenizer",
                "deduplication": "regenerate only exact rendered-content collisions; no teacher outcome selection",
                "pilot_long_input_design": {"domain": "long_context", "record_count_levels": list(PILOT_JOIN_RECORD_COUNTS),
                                            "padding": "legitimate irrelevant aliases/records; base query/support unchanged",
                                            "rough_prompt_tokens_target": [1500, 3500], "tokenizer_profile_verified": False},
                "native_model_calls": 0, "native_authorized": False, "training_authorized": False,
                "confirmation_materialized": False, "confirmation_authorized": False,
                "native_token_caps_verified": False, "overlap_review": "UNREVIEWED", "frozen": False}
    write_json(path / "DATA_MANIFEST.json", manifest)
    return manifest
