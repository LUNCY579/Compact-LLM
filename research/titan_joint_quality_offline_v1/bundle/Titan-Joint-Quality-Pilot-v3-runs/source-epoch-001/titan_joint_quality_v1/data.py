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
              "arithmetic": "train_weighted_baskets_v1", "grounding": "train_query_selected_authority_v3",
              "long_context": "train_timestamp_record_lookup_v1", "format": "train_field_transform_v1"},
    "pilot": {"reasoning": "pilot_expression_dependencies_v1", "transition": "pilot_transition_table_v1",
              "arithmetic": "pilot_euclidean_sharing_v1", "grounding": "pilot_horn_support_v1",
              "long_context": "pilot_alias_record_join_v1", "format": "pilot_filtered_roster_v1"},
}
FORMATS = ("plain", "json", "tag")
KINDS = {"train": {"reasoning": "program", "transition": "item_events", "arithmetic": "baskets",
                   "grounding": "authority_query_source_v3", "long_context": "latest_record", "format": "field_transform"},
         "pilot": {"reasoning": "dependencies", "transition": "fsm", "arithmetic": "sharing",
                   "grounding": "horn", "long_context": "alias_join", "format": "filtered_roster"}}
BEGIN, END = "TASK_JSON\n", "\nEND_TASK"
# Prospective shape variation only; actual bound-tokenizer profiling remains a
# separate offline gate. The multiplier permutes all 32 parent length levels.
PILOT_JOIN_RECORD_COUNTS = tuple(50 + 2 * ((index * 17) % 32) for index in range(32))
V2_INPUT_ROOT = Path("C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2-runs/data-001")
V2_INPUT_SHA256 = {"train.jsonl": "6c82a98a4d66c64d15aa813dc1afa3f20898996c00f2d8590158a8eec8e1b11d",
                   "pilot.prompts.jsonl": "1ba852d8b0bb78d42c543a876e24a38987284095ff17f3ec9887bd7882d46bf8",
                   "pilot.references.jsonl": "51b3b96d8e557ddcf046efbb7a8afc81d1b49bbadd1289fb41023a7936aced2d"}
V2_DATA_SOURCE_SHA256 = "dd4a39762f805b4d2a99d9c7416cbf454c2d99244ca2f42e4e73faa17d4dba1b"
REPAIR_SOURCES = ("charter", "rumor")


def render_messages(task: dict, requested_format: str = "plain") -> list[dict]:
    if requested_format not in FORMATS:
        raise ValueError("unknown response format")
    instructions = {
        "plain": "Return the answer alone with no surrounding text.",
        "json": 'Return exactly one JSON object with one string field named "answer".',
        "tag": "Return exactly <answer>VALUE</answer> with no surrounding text.",
    }
    rules = {
        "program": "Execute register instructions in list order. Return the queried register's final integer. Annotations are irrelevant.",
        "dependencies": "Evaluate the queried expression dependency with exact integer arithmetic.",
        "item_events": "Apply the item moves in list order. Return the queried item's final bin.",
        "fsm": "Start at initial; apply the first query.after_symbols input symbols using the transition table. Return that state.",
        "baskets": "Sum quantity times unit price for the queried basket. Return the integer total.",
        "sharing": "Divide the queried group's count among its people. Return the requested integer quotient or remainder.",
        "authority": "Only authoritative_source can establish the queried entity/attribute. Conflicting authoritative values are UNKNOWN.",
        "authority_query_source_v3": "For query.entity and query.attribute use ONLY facts whose source equals query.authoritative_source. Exactly one distinct value means return that value. Missing selected-source evidence or two or more distinct selected-source values means UNKNOWN. Other sources cannot supply the answer.",
        "horn": "Infer from facts and rules. Return SUPPORTED iff the queried atom follows; otherwise UNKNOWN.",
        "latest_record": "Choose the queried entity's greatest timestamp record and return the queried field.",
        "alias_join": "Resolve the query alias to its code, then retrieve its unique record's queried city or color field.",
        "field_transform": "Select two distinct fields in query.keys order, apply query.case upper/lower, and join with query.separator (|, ; or :).",
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


def requested_format_from_messages(messages: list[dict]) -> str:
    """Bind the visible finite wrapper instruction, independently of metadata.

    This verifies our instrument's generated prefix. It does not qualify arbitrary
    natural-language instruction priority, or infer intent from target text.
    """
    parse_prompt(messages)  # Require the same unambiguous legitimate public task.
    prefixes = [message["content"].split(BEGIN, 1)[0].strip() for message in messages
                if message["role"] == "user" and BEGIN in message["content"]]
    if len(prefixes) != 1: raise ValueError("one visible wrapper instruction required")
    prefix = prefixes[0]
    if prefix == "Return the answer alone with no surrounding text.": return "plain"
    if prefix == 'Return exactly one JSON object with one string field named "answer".': return "json"
    if prefix == "Return exactly <answer>VALUE</answer> with no surrounding text.": return "tag"
    raise ValueError("visible wrapper instruction is unsupported or conflicting")


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
    if k == "authority_query_source_v3":
        matching = {fact["value"] for fact in t["facts"]
                    if (fact["entity"], fact["attribute"], fact["source"]) ==
                    (q["entity"], q["attribute"], q["authoritative_source"])}
        return next(iter(matching)) if len(matching) == 1 else "UNKNOWN"
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
    if k == "authority_query_source_v3":
        answer = None
        for fact in t["facts"]:
            if fact["entity"] != q["entity"] or fact["attribute"] != q["attribute"]:
                continue
            if fact["source"] == q["authoritative_source"]:
                if answer is not None and answer != fact["value"]:
                    return "UNKNOWN"
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


def _generate_world(split: str, domain: str, rng: random.Random, parent_index: int) -> dict:
    """Create the entire world before selecting or rendering a question."""
    kind = KINDS[split][domain]
    if kind == "program":
        registers = ("x", "y")
        operations = {name: [{"dst": name, "op": "add", "arg": rng.randint(2, 9)},
                             {"dst": name, "op": "multiply", "arg": rng.randint(2, 4)},
                             {"dst": name, "op": "add_from", "arg": "y" if name == "x" else "x"},
                             {"dst": name, "op": "multiply", "arg": 2},
                             {"dst": name, "op": "add", "arg": 2 if name == "x" else 1}]
                      for name in registers}
        instructions = []
        for stage in range(5):
            for name in rng.sample(list(registers), 2):
                instructions.append(operations[name][stage])
        return {"kind": kind, "initial": {name: rng.randint(1, 20) for name in registers},
                "instructions": instructions, "annotations": [_word(rng, "note") for _ in range(3)]}
    if kind == "dependencies":
        return {"kind": kind, "leaves": {"a": rng.randint(1, 30), "b": rng.randint(1, 30), "c": rng.randint(2, 8)},
                "expressions": {"d": ["a", "+", "b"], "e": ["d", "*", "c"], "f": ["e", "-", "b"],
                                "unused": ["b", "+", "c"]}}
    if kind == "item_events":
        items, bins = [_word(rng, "i") for _ in range(6)], ["north", "east", "west"]
        events = []
        for round_index in range(4):
            for item in rng.sample(items, len(items)):
                index = items.index(item)
                destination = bins[(index + parent_index) % 3] if round_index == 3 else rng.choice(bins)
                events.append({"item": item, "bin": destination})
        return {"kind": kind, "initial": {item: rng.choice(bins) for item in items}, "events": events}
    if kind == "fsm":
        states = ["s0", "s1", "s2", "s3"]
        return {"kind": kind, "initial": rng.choice(states),
                "transitions": {state: {symbol: rng.choice([other for other in states if other != state])
                                        for symbol in ("a", "b")} for state in states},
                "symbols": [rng.choice(["a", "b"]) for _ in range(12)]}
    if kind == "baskets":
        return {"kind": kind, "baskets": {name: [{"quantity": rng.randint(2, 7), "price": rng.randint(2, 19)}
                                              for _ in range(4)] for name in ("cedar", "maple")}}
    if kind == "sharing":
        return {"kind": kind, "groups": {name: {"count": rng.randint(40, 190), "people": rng.randint(3, 11)}
                                        for name in ("orchard", "harbor")}}
    if kind == "authority":
        entities, attributes = [_word(rng, "e") for _ in range(5)], ("status", "color", "zone")
        facts = []
        for index, entity in enumerate(entities):
            missing = (index + parent_index) % len(attributes)
            for attribute_index, attribute in enumerate(attributes):
                if attribute_index != missing:
                    facts.append({"entity": entity, "attribute": attribute, "value": _word(rng, "v"), "source": "charter"})
                facts.append({"entity": entity, "attribute": attribute, "value": _word(rng, "r"), "source": "rumor"})
        rng.shuffle(facts)
        return {"kind": kind, "authoritative_source": "charter", "facts": facts}
    if kind == "horn":
        atoms = []
        while len(atoms) < 7:
            atom = _word(rng, "p")
            if atom not in atoms: atoms.append(atom)
        root_a, root_b, mid_a, mid_b, goal_a, goal_b, neutral = atoms
        rules = [{"premises": [root_a], "conclusion": mid_a}, {"premises": [mid_a], "conclusion": goal_a},
                 {"premises": [root_b], "conclusion": mid_b}, {"premises": [mid_b], "conclusion": goal_b}]
        rng.shuffle(rules)
        facts = [rng.choice([root_a, root_b]), neutral]
        rng.shuffle(facts)
        return {"kind": kind, "facts": facts, "rules": rules}
    if kind == "latest_record":
        entities = [_word(rng, "e") for _ in range(6)]
        records = []
        for entity in entities:
            timestamp = rng.randint(1, 5)
            for _ in range(4):
                timestamp += rng.randint(1, 7)
                records.append({"entity": entity, "timestamp": timestamp, "value": _word(rng, "v")})
        rng.shuffle(records)
        return {"kind": kind, "records": records}
    if kind == "alias_join":
        count = PILOT_JOIN_RECORD_COUNTS[parent_index % len(PILOT_JOIN_RECORD_COUNTS)]
        codes, aliases = [], []
        while len(codes) < count:
            word = _word(rng, "k")
            if word not in codes: codes.append(word)
        while len(aliases) < count:
            word = _word(rng, "a")
            if word not in aliases: aliases.append(word)
        records = [{"code": code, "city": _word(rng, "c"), "color": _word(rng, "v")} for code in codes]
        rng.shuffle(records)
        return {"kind": kind, "aliases": dict(zip(aliases, codes)), "records": records}
    if kind == "field_transform":
        return {"kind": kind, "fields": {key: _word(rng) for key in ("title", "token", "region", "note")}}
    if kind == "filtered_roster":
        records = [{"label": _word(rng, "n"), "tier": "A" if index % 2 else "B", "score": rng.randint(0, 10)}
                   for index in range(8)]
        rng.shuffle(records)
        return {"kind": kind, "records": records}
    raise ValueError("unknown world generator")


def enumerate_valid_queries(world: dict) -> list:
    """Enumerate the finite prospective question grammar from public world data.

    This is independent of teachers' answer values and any model outcomes.
    Grounding includes every atom/entity/attribute in the public vocabulary,
    including queries unsupported by authoritative facts.
    """
    k = world["kind"]
    if k == "program": return sorted(world["initial"])
    if k == "dependencies": return sorted(world["expressions"])
    if k == "item_events": return sorted(world["initial"])
    if k == "fsm": return [{"after_symbols": index} for index in range(len(world["symbols"]) + 1)]
    if k == "baskets": return sorted(world["baskets"])
    if k == "sharing":
        return [{"group": group, "component": component} for group in sorted(world["groups"])
                for component in ("quotient", "remainder")]
    if k == "authority":
        entities = sorted({fact["entity"] for fact in world["facts"]})
        attributes = sorted({fact["attribute"] for fact in world["facts"]})
        return [{"entity": entity, "attribute": attribute} for entity in entities for attribute in attributes]
    if k == "authority_query_source_v3":
        return [{"entity": entity, "attribute": attribute, "authoritative_source": source}
                for entity in sorted({fact["entity"] for fact in world["facts"]})
                for attribute in sorted({fact["attribute"] for fact in world["facts"]})
                for source in world["sources"]]
    if k == "horn":
        atoms = set(world["facts"])
        for rule in world["rules"]:
            atoms.add(rule["conclusion"])
            atoms.update(rule["premises"])
        return sorted(atoms)
    if k == "latest_record":
        return [{"entity": entity, "field": "value"} for entity in sorted({record["entity"] for record in world["records"]})]
    if k == "alias_join":
        return [{"alias": alias, "field": field} for alias in sorted(world["aliases"]) for field in ("city", "color")]
    if k == "field_transform":
        keys = sorted(world["fields"])
        return [{"keys": [first, second], "case": case, "separator": separator}
                for first in keys for second in keys if first != second
                for case in ("upper", "lower") for separator in ("|", ";", ":")]
    if k == "filtered_roster":
        return [{"tier": tier, "minimum": minimum, "order": order}
                for tier in sorted({record["tier"] for record in world["records"]})
                for minimum in range(12) for order in ("asc", "desc")]
    raise ValueError("unknown query grammar")


def _pair_tasks(split: str, domain: str, rng: random.Random, parent_index: int) -> tuple[dict, dict, str, dict]:
    """Prospective whole-world generation; log every degenerate world refusal."""
    degenerate = 0
    for world_attempt in range(1, 101):
        world = _generate_world(split, domain, rng, parent_index)
        queries = enumerate_valid_queries(world)
        qualified = []
        for query in queries:
            messages = render_messages({**world, "query": query})
            answer_a, answer_b = solve_prompt_a(messages), solve_prompt_b(messages)
            if answer_a != answer_b:
                raise AssertionError("independent teachers disagree on an enumerated public query")
            qualified.append((query, answer_a))
        classes = {answer for _, answer in qualified}
        if len(classes) < 2 or (domain == "grounding" and ("UNKNOWN" not in classes or classes == {"UNKNOWN"})):
            degenerate += 1
            continue
        if world["kind"] == "program":
            # Fully counterbalance both queried registers in every world.
            first_query = queries[parent_index % 2]
            first_answer = next(answer for query, answer in qualified if query == first_query)
            second_query = next(query for query, answer in qualified if answer != first_answer)
        elif domain == "grounding":
            known = [query for query, answer in qualified if answer != "UNKNOWN"]
            unknown = [query for query, answer in qualified if answer == "UNKNOWN"]
            supported_query, unsupported_query = rng.choice(known), rng.choice(unknown)
            first_query, second_query = ((supported_query, unsupported_query) if parent_index % 2 == 0
                                         else (unsupported_query, supported_query))
        else:
            first_query, first_answer = rng.choice(qualified)
            second_query = rng.choice([query for query, answer in qualified if answer != first_answer])
        first, second = deepcopy(world), deepcopy(world)
        first["query"], second["query"] = deepcopy(first_query), deepcopy(second_query)
        audit = {"world_attempts": world_attempt, "degenerate_worlds": degenerate,
                 "valid_query_count": len(queries), "distinct_truth_count": len(classes),
                 "query_candidates_digest": digest([{"query": query, "truth": answer} for query, answer in qualified]),
                 "query_selection": "world-first finite grammar; independently verified distinct truth; before model outcomes"}
        return first, second, "changed_answer_query_switch", audit
    raise RuntimeError("prospective world-generation bound exceeded; no eligible-only corpus returned")


def _validate_split(split: str) -> None:
    if split == "confirmation":
        raise PermissionError("confirmation is closed; no examples will be materialized")
    if split not in FAMILIES:
        raise ValueError("split must be train or pilot")


def _generate_unbound_bank(split: str, seed: int = 17) -> list[dict]:
    _validate_split(split)
    if type(seed) is not int: raise TypeError("seed must be an integer")
    rows, seen_content = [], set()
    counts = TRAIN_COUNTS if split == "train" else {domain: 64 for domain in DOMAIN_ROLE}
    for domain_index, (domain, count) in enumerate(counts.items()):
        rng = random.Random(int(digest([seed, split, domain])[:16], 16))
        for parent_index in range(count // 2):
            requested = FORMATS[(parent_index + domain_index) % 3]
            world_attempts = degenerate_worlds = 0
            # Regeneration is solely a content-deduplication rule; no score,
            # teacher answer, or pilot performance enters this choice.
            for duplicate_attempt in range(100):
                first, second, pair_kind, generation = _pair_tasks(split, domain, rng, parent_index)
                world_attempts += generation["world_attempts"]
                degenerate_worlds += generation["degenerate_worlds"]
                signatures = [digest(render_messages(task, requested)) for task in (first, second)]
                if len(set(signatures)) == 2 and not (set(signatures) & seen_content):
                    seen_content.update(signatures)
                    break
            else:
                raise RuntimeError("generator failed the mechanical content-uniqueness bound")
            generation = {**generation, "world_attempts": world_attempts,
                          "degenerate_worlds": degenerate_worlds,
                          "exact_content_rejections": duplicate_attempt}
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
                             "generator_audit": deepcopy(generation),
                             "solver_trace": {"source": "rendered_messages_only", "teacher_a": answer_a,
                                              "teacher_b": answer_b, "agreement": True}})
            expected_changed = pair_kind.startswith("changed")
            if (pair_answers[0] != pair_answers[1]) != expected_changed:
                raise AssertionError("counterfactual/distractor does not satisfy its registered relationship")
            if domain == "grounding" and ("UNKNOWN" not in pair_answers or pair_answers[0] == pair_answers[1]):
                raise AssertionError("grounding query pair must cross answerability in its unchanged world")
    # Ordering is unrelated to answers; formats never change between paired variants.
    random.Random(seed ^ (101 if split == "train" else 202)).shuffle(rows)
    validate_bank(rows, split)
    return rows


def load_bound_v2_epoch() -> tuple[list[dict], list[dict]]:
    """Verified seed17 inputs only; hash and parse the same read-once bytes."""
    source = Path("C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2/titan_joint_quality_v1/data.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != V2_DATA_SOURCE_SHA256:
        raise RuntimeError("bound v2 generator source changed")
    decoded = {}
    for name, expected in V2_INPUT_SHA256.items():
        payload = (V2_INPUT_ROOT / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError("bound v2 input epoch changed: " + name)
        decoded[name] = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    prompts = {row["case_id"]: row["messages"] for row in decoded["pilot.prompts.jsonl"]}
    pilot = [{**row, "messages": prompts[row["case_id"]]} for row in decoded["pilot.references.jsonl"]]
    return decoded["train.jsonl"], pilot


def _repair_source_world(base_world: dict, seed: int, index: int) -> tuple[dict, dict, dict, dict]:
    """Actual base-world identity and fresh repair seed determine new G facts."""
    rng = random.Random(int(digest([seed, "query_selected_authority_v3", digest(base_world), index])[:16], 16))
    entities = sorted({fact["entity"] for fact in base_world["facts"]})
    attributes = rng.sample(sorted({fact["attribute"] for fact in base_world["facts"]}), 2)
    if len(entities) < 2: raise ValueError("eligible base authority world requires multiple entities")
    target = {"entity": rng.choice(entities), "attribute": rng.choice(attributes)}
    known_source = REPAIR_SOURCES[0 if index % 4 in (0, 3) else 1]
    unknown_source = next(source for source in REPAIR_SOURCES if source != known_source)
    unknown_kind = "missing" if index % 2 == 0 else "conflict"
    facts = []
    used_values = set()
    def value():
        result = _word(rng, "v")  # Identical source-independent alphabet/prefix/distribution.
        while result in used_values: result = _word(rng, "v")
        used_values.add(result)
        return result
    for entity in entities:
        for attribute in attributes:
            for source in REPAIR_SOURCES:
                selected = (entity, attribute) == (target["entity"], target["attribute"])
                if selected and source == unknown_source and unknown_kind == "missing": continue
                facts.append({"entity": entity, "attribute": attribute, "source": source, "value": value()})
                if selected and source == unknown_source and unknown_kind == "conflict":
                    facts.append({"entity": entity, "attribute": attribute, "source": source, "value": value()})
    rng.shuffle(facts)
    world = {"kind": "authority_query_source_v3", "sources": list(REPAIR_SOURCES), "facts": facts}
    queries = [{**target, "authoritative_source": source} for source in REPAIR_SOURCES]
    if index % 2: queries.reverse()
    return world, queries[0], queries[1], {"known_source": known_source, "unknown_source": unknown_source,
                                          "unknown_kind": unknown_kind, "base_world_digest": digest(base_world)}


def transform_grounding_rows(base_train: list[dict], *, seed: int, base_source_sha256=V2_DATA_SOURCE_SHA256) -> list[dict]:
    """G-only deterministic transformation of an actual eligible base TRAIN draw.

    No base draw is generated here. Current generate_bank binds only seed17;
    future fresh draws must be supplied, source/input-bound and owner-reviewed.
    """
    if type(seed) is not int: raise TypeError("repair seed must be an integer")
    if base_source_sha256 != V2_DATA_SOURCE_SHA256: raise ValueError("eligible base generator source must be explicitly pinned")
    if Counter(row["domain"] for row in base_train) != Counter(TRAIN_COUNTS): raise ValueError("complete eligible base TRAIN draw required")
    groups = defaultdict(list)
    for row in base_train:
        if row["domain"] == "grounding": groups[row["parent_id"]].append(row)
    if len(groups) != 150 or any(len(pair) != 2 for pair in groups.values()): raise ValueError("complete150 base grounding parents required")
    replacements = {}
    base_digest = digest(base_train)
    for index, (parent, pair) in enumerate(sorted(groups.items())):
        pair = sorted(pair, key=lambda row: row["variant_id"])
        tasks = [parse_prompt(row["messages"]) for row in pair]
        worlds = [{key: value for key, value in task.items() if key != "query"} for task in tasks]
        if worlds[0] != worlds[1] or worlds[0]["kind"] != "authority": raise ValueError("eligible base authority pair/world differs")
        if {row["variant_id"] for row in pair} != {0, 1} or len({row["requested_format"] for row in pair}) != 1: raise ValueError("base pair geometry/wrapper differs")
        world, first_q, second_q, trace = _repair_source_world(worlds[0], seed, index)
        candidates = enumerate_valid_queries(world)
        truths = []
        for candidate in candidates:
            public = render_messages({**world, "query": candidate})
            a, b = solve_prompt_a(public), solve_prompt_b(public)
            if a != b: raise AssertionError("selected-source query grammar teacher disagreement")
            truths.append({"query": candidate, "truth": a})
        generation = {"world_attempts": 1, "degenerate_worlds": 0, "exact_content_rejections": 0,
                      "valid_query_count": len(candidates), "distinct_truth_count": len({value["truth"] for value in truths}),
                      "query_candidates_digest": digest(truths), "query_selection": "G-only selected-source switch before model outcomes"}
        for old, query in zip(pair, (first_q, second_q)):
            messages = render_messages({**world, "query": query}, old["requested_format"])
            answer_a, answer_b = solve_prompt_a(messages), solve_prompt_b(messages)
            if answer_a != answer_b: raise AssertionError("new selected-source teachers disagree")
            replacements[old["case_id"]] = {**deepcopy(old), "family_id": FAMILIES["train"]["grounding"],
                "messages": messages, "canonical_answer": answer_a, "target": format_target(answer_a, old["requested_format"]),
                "pair_kind": "changed_answer_selected_authority_switch",
                "generator_audit": deepcopy(generation),
                "solver_trace": {"source": "rendered_messages_only", "teacher_a": answer_a, "teacher_b": answer_b, "agreement": True},
                "grounding_repair_trace": {**trace, "repair_seed": seed, "base_input_digest": base_digest,
                                           "base_source_sha256": base_source_sha256, "fresh_draw_creation_verified": False}}
    output = [replacements.get(row["case_id"], deepcopy(row)) for row in base_train]
    validate_bank(output, "train")
    return output


def generate_bank(split: str, seed: int = 17) -> list[dict]:
    _validate_split(split)
    if type(seed) is not int: raise TypeError("seed must be an integer")
    if seed != 17: raise ValueError("v3 is bound to the existing seed17 input epoch; independent-draw generation/integration is pending")
    train, pilot = load_bound_v2_epoch()
    rows = transform_grounding_rows(train, seed=seed) if split == "train" else pilot
    validate_bank(rows, split)
    return rows


def source_blind_projection(messages: list[dict]) -> list[dict]:
    task = deepcopy(parse_prompt(messages))
    if task["kind"] != "authority_query_source_v3": raise ValueError("selected-authority instrument required")
    task.pop("sources", None)
    task["query"].pop("authoritative_source")
    for fact in task["facts"]: fact.pop("source")
    prefix = next(message["content"].split(BEGIN, 1)[0] for message in messages if message["role"] == "user")
    return [deepcopy(messages[0]), {"role": "user", "content": prefix + BEGIN + canonical(task).decode("utf-8") + END}]


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
        if requested_format_from_messages(row["messages"]) != row["requested_format"]:
            raise ValueError("visible wrapper instruction differs from requested_format metadata")
        task = parse_prompt(row["messages"])
        world = {key: value for key, value in task.items() if key != "query"}
        if digest(task["query"]) not in {digest(query) for query in enumerate_valid_queries(world)}:
            raise ValueError("query is outside the independently enumerated public grammar")
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
        tasks = [parse_prompt(row["messages"]) for row in pair]
        worlds = [{key: value for key, value in task.items() if key != "query"} for task in tasks]
        selected_source = tasks[0]["kind"] == "authority_query_source_v3"
        expected_kind = "changed_answer_selected_authority_switch" if selected_source else "changed_answer_query_switch"
        if worlds[0] != worlds[1] or not changed or pair[0]["pair_kind"] != expected_kind:
            raise ValueError("every paired parent requires the same world with different true query answers")
        if selected_source:
            left, right = tasks[0]["query"], tasks[1]["query"]
            if (left["entity"], left["attribute"]) != (right["entity"], right["attribute"]) or left["authoritative_source"] == right["authoritative_source"]:
                raise ValueError("selected-source pair must hold entity/attribute fixed and cross named authority")
            if "UNKNOWN" not in {row["canonical_answer"] for row in pair} or source_blind_projection(pair[0]["messages"]) != source_blind_projection(pair[1]["messages"]):
                raise ValueError("source-blind projection/truth certificate differs")
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
    if k == "dependencies": return sorted(t["expressions"])[0]
    if k == "item_events": return sorted(t["initial"])[0]
    if k == "fsm": return {"after_symbols": 0}
    if k == "baskets": return sorted(t["baskets"])[0]
    if k == "sharing": return {"group": sorted(t["groups"])[0], "component": "quotient"}
    if k == "authority": return {"entity": t["facts"][0]["entity"], "attribute": t["facts"][0]["attribute"]}
    if k == "authority_query_source_v3": return enumerate_valid_queries({key: value for key, value in t.items() if key != "query"})[0]
    if k == "horn": return t["facts"][0] if t["facts"] else "absent"
    if k == "latest_record": return {"entity": t["records"][0]["entity"], "field": "value"}
    if k == "alias_join": return {"alias": sorted(t["aliases"])[0], "field": "city"}
    if k == "field_transform": return {"keys": sorted(t["fields"])[:2], "case": "lower", "separator": "|"}
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


def generation_audit(rows: list[dict]) -> dict:
    parents = {}
    for row in rows:
        value = row["generator_audit"]
        if row["parent_id"] in parents and parents[row["parent_id"]]["audit"] != value:
            raise ValueError("paired generation audit differs")
        parents[row["parent_id"]] = {"domain": row["domain"], "audit": value}
    by_domain = {}
    for domain in DOMAIN_ROLE:
        audits = [value["audit"] for value in parents.values() if value["domain"] == domain]
        if not audits: continue
        by_domain[domain] = {"parents": len(audits), "world_attempts": sum(a["world_attempts"] for a in audits),
                             "degenerate_worlds": sum(a["degenerate_worlds"] for a in audits),
                             "exact_content_rejections": sum(a["exact_content_rejections"] for a in audits),
                             "valid_queries_min_max": [min(a["valid_query_count"] for a in audits), max(a["valid_query_count"] for a in audits)],
                             "distinct_truths_min_max": [min(a["distinct_truth_count"] for a in audits), max(a["distinct_truth_count"] for a in audits)]}
    return {"prospective_generation": True, "eligible_only_subsetting": False, "domains": by_domain,
            "refusal_rule": "stop after 100 degenerate worlds or 100 exact-content retries; never return a partial corpus"}


def _structural_queryblind_predictions(world: dict) -> dict[str, str]:
    """Fixed stronger nuisance controls; no query or reference is available."""
    if "query" in world: raise ValueError("question must be removed before structural prediction")
    k = world["kind"]
    predictions = {}
    first_query = enumerate_valid_queries(world)[0]
    predictions["first_valid_query_executor"] = solve_prompt_b(render_messages({**world, "query": first_query}))
    if k == "item_events": predictions["fixed_event13_bin"] = world["events"][13]["bin"]
    if k == "authority":
        qualifying = [fact for fact in world["facts"] if fact["source"] == world["authoritative_source"] and fact["attribute"] != "status"]
        predictions["unique_nonstatus_authoritative_else_UNKNOWN"] = qualifying[0]["value"] if len(qualifying) == 1 else "UNKNOWN"
        predictions["old_factcount7_else_UNKNOWN"] = next((fact["value"] for fact in qualifying), "UNKNOWN") if len(world["facts"]) == 7 else "UNKNOWN"
    if k == "authority_query_source_v3":
        predictions["first_fact_value_sourceblind"] = world["facts"][0]["value"]
    if k == "latest_record":
        matches = [record for record in world["records"] if record["timestamp"] == 30]
        predictions["fixed_timestamp30_first_value"] = matches[0]["value"] if matches else "UNKNOWN"
        predictions["global_latest_value"] = max(world["records"], key=lambda record: record["timestamp"])["value"]
    if k == "fsm":
        z = [index for index, symbol in enumerate(world["symbols"]) if symbol == "z"]
        predictions["lone_z_prefix_executor"] = solve_prompt_b(render_messages({**world, "query": {"after_symbols": z[0] + 1}})) if len(z) == 1 else "UNKNOWN"
        predictions["complete_input_executor"] = solve_prompt_b(render_messages({**world, "query": {"after_symbols": len(world["symbols"])}}))
    if k == "alias_join":
        marked = [record[field] for record in world["records"] for field in ("city", "color") if len(record[field]) == 8]
        predictions["lone_eight_character_value"] = marked[0] if len(marked) == 1 else "UNKNOWN"
        predictions["first_record_city"] = world["records"][0]["city"]
    if k == "filtered_roster":
        anchor = world["records"][7]
        for order in ("asc", "desc"):
            query = {"tier": anchor["tier"], "minimum": anchor["score"], "order": order}
            predictions["record7_inferred_query_" + order] = solve_prompt_b(render_messages({**world, "query": query}))
    return predictions


def counterdirection_audit(rows: list[dict]) -> dict:
    """Mechanical query/support diagnostics; predictions read messages only.

    References enter correctness comparisons after prediction. The paired upper
    bound applies to a deterministic query-blind raw-answer policy, not models
    that read queries, randomized model repetitions, or general task ability.
    """
    program_rows = [row for row in rows if parse_prompt(row["messages"])["kind"] == "program"]
    horn_rows = [row for row in rows if parse_prompt(row["messages"])["kind"] == "horn"]
    result = {"evidence_kind": "OFFLINE_INSTRUMENT_MECHANICS_NOT_NATIVE", "native_model_calls": 0,
              "acceptance_thresholds_changed": False}
    all_domains = {}
    for domain in DOMAIN_ROLE:
        selected = [row for row in rows if row["domain"] == domain]
        if not selected: continue
        parents, controls = defaultdict(list), Counter()
        for row in selected:
            task = parse_prompt(row["messages"])
            world = {key: value for key, value in task.items() if key != "query"}
            for name, prediction in _structural_queryblind_predictions(world).items():
                controls[name] += prediction == row["canonical_answer"]
            parents[row["parent_id"]].append(row)
        certified = 0
        for pair in parents.values():
            tasks = [parse_prompt(row["messages"]) for row in pair]
            worlds = [{key: value for key, value in task.items() if key != "query"} for task in tasks]
            certified += (len(pair) == 2 and worlds[0] == worlds[1] and
                          pair[0]["canonical_answer"] != pair[1]["canonical_answer"] and
                          pair[0]["requested_format"] == pair[1]["requested_format"])
        all_domains[domain] = {"rows": len(selected), "parents": len(parents), "certified_same_world_query_pairs": certified,
                               "deterministic_queryblind_pair_accuracy_upper_bound": .5 if certified == len(parents) else None,
                               "structural_control_correct_counts": dict(controls)}
    result["all_domains"] = all_domains
    selected_authority = [row for row in rows if parse_prompt(row["messages"])["kind"] == "authority_query_source_v3"]
    if selected_authority:
        groups, known_sources, unknown_sources, kinds = defaultdict(list), Counter(), Counter(), Counter()
        old_prefix_correct = 0
        for row in selected_authority:
            task = parse_prompt(row["messages"])
            groups[row["parent_id"]].append(row)
            selected = task["query"]["authoritative_source"]
            (unknown_sources if row["canonical_answer"] == "UNKNOWN" else known_sources)[selected] += 1
            kinds[row["grounding_repair_trace"]["unknown_kind"]] += row["variant_id"] == 0
            q = task["query"]
            values = {fact["value"] for fact in task["facts"] if fact["entity"] == q["entity"] and fact["attribute"] == q["attribute"] and fact["value"].startswith("v")}
            prediction = next(iter(values)) if len(values) == 1 else "UNKNOWN"
            old_prefix_correct += prediction == row["canonical_answer"]
        certified = sum(source_blind_projection(pair[0]["messages"]) == source_blind_projection(pair[1]["messages"])
                        and pair[0]["canonical_answer"] != pair[1]["canonical_answer"] for pair in groups.values())
        result["selected_authority"] = {"rows": len(selected_authority), "parents": len(groups),
            "sourceblind_certified_pairs": certified, "deterministic_sourceblind_pair_accuracy_upper_bound": .5 if certified == len(groups) else None,
            "known_source_counts": dict(known_sources), "UNKNOWN_source_counts": dict(unknown_sources),
            "unknown_kind_parent_counts": dict(kinds), "old_value_prefix_control_correct": old_prefix_correct,
            "admission": "UNADMITTED_MECHANICS_ONLY; no model/source-aware acquisition claim"}
    if program_rows:
        scores = Counter()
        unique_inference = 0
        query_counts = Counter()
        parents = defaultdict(list)
        for row in program_rows:
            task = parse_prompt(row["messages"])
            query_counts[task["query"]] += 1
            parents[row["parent_id"]].append(row)
            blinded = deepcopy(task)
            del blinded["query"]
            frequencies = Counter(instruction["dst"] for instruction in blinded["instructions"])
            tied = sorted(register for register, count in frequencies.items() if count == max(frequencies.values()))
            unique_inference += len(tied) == 1
            selectors = {"most_written_lex_tie": tied[0], "first_dst": blinded["instructions"][0]["dst"],
                         "last_dst": blinded["instructions"][-1]["dst"], "lex_first_register": sorted(blinded["initial"])[0],
                         "lex_last_register": sorted(blinded["initial"])[-1]}
            for name, inferred in selectors.items():
                prediction = solve_prompt_b(render_messages({**blinded, "query": inferred}))
                scores[name] += prediction == row["canonical_answer"]
        same_world_different_answer = 0
        for pair in parents.values():
            tasks = [parse_prompt(row["messages"]) for row in pair]
            worlds = [{key: value for key, value in task.items() if key != "query"} for task in tasks]
            same_world_different_answer += (len(pair) == 2 and worlds[0] == worlds[1]
                                           and pair[0]["canonical_answer"] != pair[1]["canonical_answer"])
        result["program"] = {"rows": len(program_rows), "parents": len(parents), "query_counts": dict(query_counts),
                             "unique_most_written_dst_rows": unique_inference,
                             "same_world_different_answer_pairs": same_world_different_answer,
                             "query_blind_pair_accuracy_upper_bound": .5 if same_world_different_answer == len(parents) else None,
                             "shortcut_correct_counts": dict(scores)}
    if horn_rows:
        scores, fact_counts, rule_counts, query_lengths, directions = Counter(), Counter(), Counter(), Counter(), Counter()
        parents = defaultdict(list)
        for row in horn_rows:
            task = parse_prompt(row["messages"])
            parents[row["parent_id"]].append(row)
            fact_counts[len(task["facts"])] += 1
            rule_counts[len(task["rules"])] += 1
            query_lengths[len(task["query"])] += 1
            # Three non-proof controls: cardinality, question-name hash and
            # arbitrary lexical ordering of the provided support atom vs query.
            supplied_roots = [atom for atom in task["facts"]
                              if any(atom in rule["premises"] for rule in task["rules"])]
            predictions = {"fact_count_two": "SUPPORTED" if len(task["facts"]) == 2 else "UNKNOWN",
                           "question_name_hash_parity": "SUPPORTED" if int(digest(task["query"])[-1], 16) % 2 else "UNKNOWN",
                           "support_atom_lex_before_query": "SUPPORTED" if supplied_roots and min(supplied_roots) < task["query"] else "UNKNOWN"}
            for name, prediction in predictions.items():
                scores[name] += prediction == row["canonical_answer"]
        query_pairs = support_pairs = 0
        for pair in parents.values():
            pair = sorted(pair, key=lambda row: row["variant_id"])
            tasks = [parse_prompt(row["messages"]) for row in pair]
            directions[pair[0]["pair_kind"] + ":" + pair[0]["canonical_answer"] + "->" + pair[1]["canonical_answer"]] += 1
            worlds = [{key: value for key, value in task.items() if key != "query"} for task in tasks]
            query_pairs += worlds[0] == worlds[1] and tasks[0]["query"] != tasks[1]["query"]
            support_pairs += tasks[0]["rules"] == tasks[1]["rules"] and tasks[0]["query"] == tasks[1]["query"] and set(tasks[0]["facts"]) != set(tasks[1]["facts"])
        result["horn"] = {"rows": len(horn_rows), "parents": len(parents), "fact_counts": dict(fact_counts),
                          "rule_counts": dict(rule_counts), "query_atom_name_lengths": dict(query_lengths),
                          "same_world_query_pairs": query_pairs, "same_query_support_swap_pairs": support_pairs,
                          "answer_directions": dict(directions), "shortcut_correct_counts": dict(scores),
                          "name_control_limit": "These fixed name controls do not exhaust all lexical heuristics; new instrument review remains required."}
    return result


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
                "instrument_revision": 3,
                "input_epoch": {"v2_open_root": str(V2_INPUT_ROOT), "input_sha256": V2_INPUT_SHA256,
                                "base_generator_source_sha256": V2_DATA_SOURCE_SHA256,
                                "seed17_only": True, "independent_draw_generation_integration": "PENDING",
                                "preserved": "all non-G TRAIN rows and all PILOT rows, including public inputs/targets/order",
                                "transform": "G-only query-selected-authority replacement; new family and isolated base-world/seed RNG"},
                "train": train_check, "pilot": pilot_check, "partition": partition,
                "teacher": {"kind": "two_algorithmically_separate_prompt_only_symbolic_solvers",
                            "source_digest": digest({"data.py": Path(__file__).read_text(encoding="utf-8")}),
                            "agreement_train": len(train), "agreement_pilot": len(pilot), "model_teacher_calls": 0,
                            "independence_review": "UNREVIEWED"},
                "artifact_sha256": artifact_hashes, "corpus_digest": digest({"train": train, "pilot": pilot}),
                "floors": {"train": train_floors, "pilot": pilot_floors},
                "counterdirection_audit": {"train": counterdirection_audit(train), "pilot": counterdirection_audit(pilot)},
                "generation_audit": {"train": generation_audit(train), "pilot": generation_audit(pilot)},
                "input_projection": "pilot prompts contain case_id and messages only; IDs are not rendered inputs",
                "targets": "concise truth targets; EOS must be appended later by the bound tokenizer",
                "deduplication": "regenerate only exact rendered-content collisions; no teacher outcome selection",
                "pilot_long_input_design": {"domain": "long_context", "record_count_levels": list(PILOT_JOIN_RECORD_COUNTS),
                                            "padding": "complete world before query selection; equal facts/order in both variants",
                                            "rough_prompt_tokens_target": [1500, 3500], "tokenizer_profile_verified": False},
                "native_model_calls": 0, "native_authorized": False, "training_authorized": False,
                "confirmation_materialized": False, "confirmation_authorized": False,
                "native_token_caps_verified": False, "overlap_review": "UNREVIEWED", "frozen": False}
    write_json(path / "DATA_MANIFEST.json", manifest)
    return manifest
