"""Independent, immutable, deterministic unsealed banks for the two tracks.

The primary bank has 1,024 distinct scenarios. Linked robustness variants live
in a separate file and never inflate the primary independent sample count.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any

from .guards import guard_non_sealed_path, require_within
from .hashing import canonical_sha256, sha256_file, write_json
from .scoring import canonical_answer, score_response, scorer_sha256

GENERATOR_VERSION = "adaptive_banks_v2.2"
TRACKS = ("native_armr", "cross_scale")
ROLES = ("calibration", "architecture_dev", "confirmatory_dev")
FAMILIES = ("relational", "counterfactual", "compositional", "mathematical",
            "symbolic_logic", "retrieval_multihop", "ordinary_qa_preservation", "instruction_evidence")
TEMPLATE_NAMES = {
    "relational": ("total_order", "spatial_chain", "ancestor_chain", "nested_containment"),
    "counterfactual": ("intervened_causal_chain", "replacement_arithmetic", "overridden_rule", "changed_route"),
    "compositional": ("successive_maps", "string_transform", "affine_composition", "list_pipeline"),
    "mathematical": ("inventory_balance", "unit_rate", "integer_remainder", "fraction_of_quantity"),
    "symbolic_logic": ("boolean_expression", "set_intersection", "exclusive_or", "quantified_rule"),
    "retrieval_multihop": ("two_table_join", "dated_update", "linked_records", "distractor_retrieval"),
    "ordinary_qa_preservation": ("appointment_lookup", "object_location", "weather_action", "menu_lookup"),
    "instruction_evidence": ("json_fields", "ordered_output", "insufficient_evidence", "conflicting_evidence"),
}

# These are proposition rewrites, not prefixes claiming to be paraphrases.
_PARAPHRASES = {
    "relational": [(r"(\S+) is taller than (\S+)\.", r"\1 has greater height than \2.")],
    "counterfactual": [(r"lamp equals relay; relay equals switch", "the lamp copies the relay value, and the relay copies the switch value"),
                       (r"The switch is initially", "The initial switch value is"),
                       (r"Intervene to set relay to", "Force the relay value to")],
    "compositional": [(r"Map ([FG]) sends (\S+) to (\S+), and (\S+) to (\S+)\.",
                       r"Function \1 maps input \2 onto \3; it maps input \4 onto \5.")],
    "mathematical": [(r"Shop (\S+) starts with (\d+) pens, receives (\d+), and sells (\d+)\.",
                      r"There are initially \2 pens at shop \1. Another \3 arrive, and \4 are sold.")],
    "symbolic_logic": [(r"For system (\S+),", r"The truth assignments in system \1 are")],
    "retrieval_multihop": [(r"Employee (\S+) belongs to team (\S+)\.", r"\1 is a member of team \2."),
                           (r"Team (\S+) uses room (\S+)\.", r"The room assigned to team \1 is \2.")],
    "ordinary_qa_preservation": [(r"(\S+)'s dental appointment is at (\S+)\.", r"The dental appointment for \1 is scheduled for \2.")],
    "instruction_evidence": [(r"Order (\S+) has quantity (\d+)\.", r"The quantity for order \1 is \2.")],
}


def semantic_answer(row: dict) -> Any:
    """Independently solve rendered propositions; never read answer/scorer/latent.

    This separate parser/solver intentionally does not call the random builder.
    It validates every generated instance, including linked semantic changes.
    It is an oracle for this controlled grammar, not a general English solver.
    """
    text = "\n".join(row["facts"])
    question = row["question"]
    template = row["template_id"].split(".", 1)[1]

    def match(pattern: str, value: str = text):
        found = re.search(pattern, value)
        if found is None:
            raise ValueError(f"unparsed oracle proposition for {template}: {pattern}")
        return found.groups()

    def integer(pattern: str, value: str = text):
        return tuple(map(int, match(pattern, value)))

    if template == "total_order":
        edges = re.findall(r"(\S+) (?:is taller than|has greater height than) (\S+)\.", text)
        edges += [(b, a) for a, b in re.findall(r"(\S+) is shorter than (\S+)\.", text)]
        first, second = {a for a, _ in edges}, {b for _, b in edges}
        candidates = first - second if "tallest" in question else second - first
        if len(candidates) != 1:
            raise ValueError("order is not uniquely determined")
        return candidates.pop()
    if template == "spatial_chain":
        edges = re.findall(r"(\S+) is (\d+) meters (east|west) of (\S+)\.", text)
        graph = defaultdict(list)
        for a, amount, direction, b in edges:
            delta = int(amount) * (1 if direction == "east" else -1)
            graph[b].append((a, delta)); graph[a].append((b, -delta))
        target, origin = match(r"What is (\S+)'s signed eastward displacement from (\S+),", question)
        coordinates, pending = {origin: 0}, [origin]
        while pending:
            node = pending.pop()
            for child, delta in graph[node]:
                if child not in coordinates:
                    coordinates[child] = coordinates[node] + delta; pending.append(child)
        return str(coordinates[target])
    if template == "ancestor_chain":
        parents = {child: parent for parent, child in re.findall(r"(\S+) is a parent of (\S+)\.", text)}
        if "great-grandparent" in question:
            node, = match(r"great-grandparent of (\S+)\?", question); hops = 3
        else:
            hops, node = match(r"ancestor (\d+) generations before (\S+)\?", question); hops = int(hops)
        for _ in range(hops):
            node = parents[node]
        return node
    if template == "nested_containment":
        edges = dict(re.findall(r"(?:Object|Box|Crate) (\S+) is inside (?:box|crate|room) (\S+)\.", text))
        node, = match(r"contains object (\S+)\?", question)
        while node in edges:
            node = edges[node]
        return node
    if template == "intervened_causal_chain":
        value, = match(r"(?:Intervene to set relay to|Force the relay value to) ([01])", text)
        if not ("lamp equals relay" in text or "lamp copies the relay value" in text):
            raise ValueError("missing causal lamp equation")
        return value
    if template == "replacement_arithmetic":
        boxes, = integer(r"makes (\d+) boxes"); replacement, = integer(r"items per box by (\d+)")
        return str(sum(replacement for _ in range(boxes)))
    if template == "overridden_rule":
        state, = match(r"counterfactually the override is (active|inactive)")
        return "yes" if state == "inactive" else "no"
    if template == "changed_route":
        a, b, c = match(r"original route from (\S+) goes to (\S+), then (\S+)\.")
        origin, old, new_origin, new = match(r"first edge (\S+)->(\S+) by (\S+)->(\S+)\.")
        d, e = match(r"route from (\S+) continues to (\S+)\.")
        routes = {a: b, b: c, d: e}
        if origin != new_origin or routes[origin] != old:
            raise ValueError("invalid counterfactual edge replacement")
        routes[origin] = new
        start, = match(r"hops from (\S+) end", question)
        return routes[routes[start]]
    if template == "successive_maps":
        entries = re.findall(r"Map ([FG]) sends (\S+) to (\S+), and (\S+) to (\S+)\.", text)
        entries += re.findall(r"Function ([FG]) maps input (\S+) onto (\S+); it maps input (\S+) onto (\S+)\.", text)
        maps = {f: {a: b, c: d} for f, a, b, c, d in entries}
        start, = match(r"G\(F\((\S+)\)\)", question)
        return maps["G"][maps["F"][start]]
    if template == "string_transform":
        word, = match(r"input at station \S+ is ([a-z]+)\.")
        return "".join(reversed(list(word)[:-2]))
    if template == "affine_composition":
        multiplier, offset, subtract = integer(r"F\(t\)=(\d+)\*t\+(\d+) and G\(t\)=t-(\d+)\.")
        value, = integer(r"G\(F\((\d+)\)\)", question)
        return str(sum(value for _ in range(multiplier)) + offset - subtract)
    if template == "list_pipeline":
        values, = match(r"list in job \S+ is ([\d,]+)\.")
        even = [v for v in map(int, values.split(",")) if divmod(v, 2)[1] == 0]
        return sorted(v + v for v in even)
    if template == "inventory_balance":
        if "There are initially" in text:
            start, receive, sold = integer(r"initially (\d+) pens at shop \S+\. Another (\d+) arrive, and (\d+) are sold")
        else:
            start, receive, sold = integer(r"starts with (\d+) pens, receives (\d+), and sells (\d+)")
        return str(start - sold + receive)
    if template == "unit_rate":
        rate, = integer(r"makes (\d+) units per minute"); minutes, = integer(r"in (\d+) minutes", question)
        return str(sum(rate for _ in range(minutes)))
    if template == "integer_remainder":
        count, = integer(r"holds (\d+) counters"); group, = integer(r"groups of (\d+) counters")
        return str(count - (count // group) * group)
    if template == "fraction_of_quantity":
        total, numerator, denominator = integer(r"harvested (\d+) apples and sold (\d+)/(\d+) of them")
        return str(total // denominator * numerator)
    if template == "boolean_expression":
        assignments = dict(re.findall(r"([PQR])=(true|false)", text))
        truth = {p for p, value in assignments.items() if value == "true"}
        return str("R" in truth or {"P", "Q"} & truth == {"P"}).lower()
    if template == "set_intersection":
        left, right = match(r"set A=(\[[\d, ]*\]) and set B=(\[[\d, ]*\])")
        return sorted(v for v in json.loads(left) if v in json.loads(right))
    if template == "exclusive_or":
        p, q = match(r"input P is (true|false) and input Q is (true|false)")
        return str(sum(value == "true" for value in (p, q)) == 1).lower()
    if template == "quantified_rule":
        edges = dict(re.findall(r"Every (\S+) is a (\S+)\.", text))
        exclusions = set(re.findall(r"No (\S+) is a (\S+)\.", text))
        obj, category = match(r"Object (\S+) is a (\S+)\.")
        _, target = match(r"(?:Can|Must) (\S+) (?:be a|be a) (\S+)(?: under|\?)", question)
        membership = {category}
        while category in edges:
            category = edges[category]; membership.add(category)
        if "Must" in question:
            return "yes" if target in membership else "no"
        return "no" if any((m, target) in exclusions for m in membership) else "yes"
    if template == "two_table_join":
        employees = dict(re.findall(r"Employee (\S+) belongs to team (\S+)\.", text) + re.findall(r"(\S+) is a member of team (\S+)\.", text))
        rooms = dict(re.findall(r"Team (\S+) uses room (\S+)\.", text) + re.findall(r"The room assigned to team (\S+) is (\S+)\.", text))
        employee, = match(r"employee (\S+)'s team", question)
        return rooms[employees[employee]]
    if template == "dated_update":
        updates = [(int(day), place) for day, place in re.findall(r"On day (\d+), parcel \S+ (?:was in|moved to) (\S+)\.", text)]
        return max(updates)[1]
    if template == "linked_records":
        pointers = dict(re.findall(r"Record (\S+) points to (\S+)\.", text))
        values = dict(re.findall(r"Record (\S+) has value (\d+)\.", text))
        start, = match(r"pointers from (\S+)\.", question)
        return values[pointers[pointers[start]]]
    if template == "distractor_retrieval":
        records = dict(re.findall(r"Record (\S+) has code (\d+)\.", text))
        key, = match(r"code for record (\S+)\?", question)
        return records[key]
    if template == "appointment_lookup":
        records = dict(re.findall(r"(\S+)'s dental appointment is at (\d+:\d+)\.", text) + re.findall(r"The dental appointment for (\S+) is scheduled for (\d+:\d+)\.", text))
        person, = match(r"time is (\S+)'s dental", question)
        return records[person]
    if template == "object_location":
        records = dict(re.findall(r"(\S+) put the keys in the (\S+) and left them there", text))
        person, = match(r"Where are (\S+)'s keys", question)
        return records[person]
    if template == "weather_action":
        forecast, = match(r"forecast is (\S+)\.")
        rules = dict((weather, item) for item, weather in re.findall(r"(\w+) for (rain|snow|sun)", text))
        return rules[forecast]
    if template == "menu_lookup":
        price, = match(r"salad at (\d+) credits")
        return price
    if template == "json_fields":
        if "The quantity for" in text:
            order, quantity = match(r"quantity for order (\S+) is (\d+)\.")
        else:
            order, quantity = match(r"Order (\S+) has quantity (\d+)\.")
        return {"order": order, "quantity": int(quantity)}
    if template == "ordered_output":
        words, = match(r"Tokens: ([^.]*)\.")
        tokens = words.split()
        return " ".join([tokens[1], tokens[0]])
    if template == "insufficient_evidence":
        records = {(person, color): number for person, number, color in re.findall(r"(\S+) owns (\d+) (blue|red) marbles", text)}
        person, = match(r"red marbles does (\S+) own", question)
        return records.get((person, "red"), "unknown")
    if template == "conflicting_evidence":
        values = re.findall(r"Record \d+: locker \S+ is (open|closed)\.", text)
        if len(values) != 2:
            raise ValueError("incomplete conflict records")
        return "conflict" if len(set(values)) > 1 else "consistent"
    raise ValueError(f"unregistered semantic oracle template {template}")


def _construct(family: str, template: int, rng: random.Random, tag: str) -> tuple[list[str], str, Any, bool]:
    """Build a semantic instance and compute its answer before text rendering."""
    names = [f"{tag}{c}" for c in "ABCDE"]
    a, b, c, d, e = names
    x, y, z = rng.sample(range(3, 48), 3)
    integer = lambda value: str(int(value))
    if family == "relational":
        if template == 0:
            return [f"{a} is taller than {b}.", f"{b} is taller than {c}.", f"{c} is taller than {d}."], f"Who is the {'tallest' if x % 2 else 'shortest'}?", a if x % 2 else d, False
        if template == 1:
            return [f"{a} is {x} meters east of {b}.", f"{b} is {y} meters west of {c}."], f"What is {a}'s signed eastward displacement from {c}, in meters?", integer(x-y), False
        if template == 2:
            return [f"{a} is a parent of {b}.", f"{b} is a parent of {c}.", f"{c} is a parent of {d}."], f"Who is the great-grandparent of {d}?", a, False
        return [f"Object {a} is inside box {b}.", f"Box {b} is inside crate {c}.", f"Crate {c} is inside room {d}."], f"Which named room contains object {a}?", d, False
    if family == "counterfactual":
        if template == 0:
            switch, override = rng.choice([0, 1]), rng.choice([0, 1])
            return [f"Device {a}: lamp equals relay; relay equals switch.", f"The switch is initially {switch}.", f"Intervene to set relay to {override}, leaving the switch unchanged."], "What is the lamp value after this intervention?", integer(override), False
        if template == 1:
            return [f"Workshop {a} makes {x} boxes with {y} items per box.", f"Counterfactually replace the items per box by {z}; keep the box count fixed."], "What would the total number of items be?", integer(x*z), False
        if template == 2:
            active = rng.choice([True, False])
            return [f"Under rule {a}, a badge normally opens the door.", f"An emergency override blocks everyone when active, even badge holders.", f"The badge holder arrives; counterfactually the override is {'active' if active else 'inactive'}."], "Does the door open? Answer yes or no.", "no" if active else "yes", False
        return [f"The original route from {a} goes to {b}, then {c}.", f"Replace only the first edge {a}->{b} by {a}->{d}.", f"The route from {d} continues to {e}."], f"After replacement, where do exactly two hops from {a} end?", e, False
    if family == "compositional":
        if template == 0:
            return [f"Map F sends {a} to {b}, and {d} to {e}.", f"Map G sends {b} to {c}, and {e} to {a}."], f"What is G(F({a}))?", c, False
        if template == 1:
            word = "".join(rng.sample("abcdefghjkmnpqrstuvwxyz", 6))
            return [f"The input at station {a} is {word}.", "First reverse the string; then remove the first two characters of the reversed string."], "What string remains?", word[::-1][2:], False
        if template == 2:
            return [f"At station {a}, F(t)={x}*t+{y} and G(t)=t-{z}."], f"What is G(F({z}))?", integer(x*z+y-z), False
        values = rng.sample(range(1, 40), 7)
        result = sorted(v * 2 for v in values if v % 2 == 0)
        return [f"The list in job {a} is {','.join(map(str, values))}.", "Keep even values, double them, then sort ascending."], "Return the resulting JSON array.", result, True
    if family == "mathematical":
        if template == 0:
            return [f"Shop {a} starts with {x+y} pens, receives {z}, and sells {y}."], "How many pens remain?", integer(x+z), False
        if template == 1:
            return [f"Machine {a} makes {x} units per minute at a constant rate."], f"How many units does it make in {y} minutes?", integer(x*y), False
        if template == 2:
            return [f"Bin {a} holds {x*y+z} counters.", f"Put as many complete groups of {y} counters aside as possible."], "How many counters remain outside complete groups?", integer(z%y), False
        denominator, numerator = rng.choice([3, 4, 5, 6, 8]), rng.choice([1, 2])
        return [f"Farm {a} harvested {x*denominator} apples and sold {numerator}/{denominator} of them."], "How many apples were sold?", integer(x*numerator), False
    if family == "symbolic_logic":
        if template == 0:
            p, q, r = [rng.choice([True, False]) for _ in range(3)]
            return [f"For system {a}, P={str(p).lower()}, Q={str(q).lower()}, R={str(r).lower()}."], "Evaluate (P AND NOT Q) OR R. Answer true or false.", str((p and not q) or r).lower(), False
        if template == 1:
            left, right = set(rng.sample(range(1, 16), 6)), set(rng.sample(range(1, 16), 6))
            return [f"For job {a}, set A={sorted(left)} and set B={sorted(right)}."], "Return A intersect B as an ascending JSON array.", sorted(left & right), True
        if template == 2:
            p, q = rng.choice([True, False]), rng.choice([True, False])
            return [f"In circuit {a}, input P is {str(p).lower()} and input Q is {str(q).lower()}.", "The output is true exactly when one input is true and the other is false."], "What is the output? Answer true or false.", str(p != q).lower(), False
        negative = rng.choice([True, False])
        return [f"Every {a} is a {b}.", f"No {b} is a {c}.", f"Object {d} is a {a}."], f"Can {d} be a {c} under these rules? Answer yes or no." if negative else f"Must {d} be a {b}? Answer yes or no.", "no" if negative else "yes", False
    if family == "retrieval_multihop":
        if template == 0:
            return [f"Employee {a} belongs to team {b}.", f"Employee {d} belongs to team {e}.", f"Team {b} uses room {c}.", f"Team {e} uses room {d}."], f"Which room does employee {a}'s team use?", c, False
        if template == 1:
            facts = [f"On day {x}, parcel {a} was in {b}.", f"On day {x+y}, parcel {a} moved to {c}.", f"On day {x+y+z}, parcel {a} moved to {d}."]
            rng.shuffle(facts)
            return facts, f"Where was parcel {a} immediately after the latest dated event?", d, False
        if template == 2:
            return [f"Record {a} points to {b}.", f"Record {b} points to {c}.", f"Record {c} has value {z}.", f"Record {d} has value {y}."], f"Follow two pointers from {a}. What value is in the reached record?", integer(z), False
        facts = [f"Record {tag}Q{i} has code {rng.randrange(100,999)}." for i in range(12)]
        facts.insert(rng.randrange(13), f"Record {a} has code {x*100+y}.")
        return facts, f"What is the code for record {a}?", integer(x*100+y), False
    if family == "ordinary_qa_preservation":
        if template == 0:
            time = f"{rng.randrange(1,13)}:{rng.choice(['00','15','30','45'])}"
            return [f"{a}'s dental appointment is at {time}.", f"{b}'s appointment is tomorrow."], f"At what time is {a}'s dental appointment?", time, False
        if template == 1:
            place = rng.choice(["drawer", "cupboard", "backpack", "desk", "basket"])
            return [f"{a} put the keys in the {place} and left them there.", f"{b} put a book on the floor."], f"Where are {a}'s keys?", place, False
        if template == 2:
            weather = rng.choice(["rain", "snow", "sun"])
            action = {"rain": "umbrella", "snow": "boots", "sun": "sunhat"}[weather]
            return [f"For trip {a}, the forecast is {weather}.", "The packing rule is umbrella for rain, boots for snow, and sunhat for sun."], "Which single item does the rule specify?", action, False
        price = x + y
        return [f"Cafe {a}'s menu lists soup at {x} credits, salad at {price} credits, and tea at {z} credits."], "What is the listed price of salad in credits?", integer(price), False
    if family == "instruction_evidence":
        if template == 0:
            return [f"Order {a} has quantity {x}."], "Return only a JSON object with keys order and quantity, using the order identifier and numeric quantity.", {"order": a, "quantity": x}, True
        if template == 1:
            words = rng.sample([a, b, c, d, e], 3)
            return [f"Tokens: {' '.join(words)}."], "Output only the second token followed by the first token, separated by one space.", f"{words[1]} {words[0]}", False
        if template == 2:
            return [f"{a} owns {x} blue marbles.", f"{b} owns {y} red marbles.", "No other facts are supplied."], f"How many red marbles does {a} own? If not determined, answer unknown.", "unknown", False
        return [f"Two equally authoritative records describe the same current state of locker {a}.", f"Record 1: locker {a} is open.", f"Record 2: locker {a} is closed.", "Neither record has priority."], "Return conflict if the records contradict, otherwise consistent.", "conflict", False
    raise ValueError(f"unknown family {family}")


def generate_rows(track: str, role: str, seed: int = 20260905, count: int | None = None) -> list[dict]:
    if track not in TRACKS or role not in ROLES:
        raise ValueError("unknown track or unregistered data role")
    count = count if count is not None else 1024 if role == "confirmatory_dev" else 128
    if count <= 0 or count % 8:
        raise ValueError("count must be a positive multiple of eight family strata")
    rows = []
    for family in FAMILIES:
        for index in range(count // 8):
            # Template 3 never appears in calibration/architecture development.
            template = index % (4 if role == "confirmatory_dev" else 3)
            identity = canonical_sha256([GENERATOR_VERSION, seed, track, role, family, index])
            rng = random.Random(int(identity, 16))
            tag = ("N" if track == "native_armr" else "X") + identity[:7]
            facts, question, answer, json_answer = _construct(family, template, rng, tag)
            prompt = "\n".join(facts + [question, "Give only the requested answer, with no explanation."])
            # Bind version/seed/content namespace into the immutable identity;
            # another prospective bank must never reuse an old ID for new text.
            scenario_id = f"{track}_{role}_{family}_{identity[:12]}_{index:04d}"
            scorer = {"type": "json_exact", "value": answer} if json_answer else {"type": "exact", "answers": [str(answer)]}
            row = {"schema_version": GENERATOR_VERSION, "example_id": scenario_id + "_original",
                   "case_id": scenario_id + "_original", "scenario_id": scenario_id,
                   "family": family, "split": role, "track": track, "prompt": prompt,
                   "scorer": scorer, "variant": "original", "template_id": family + "." + TEMPLATE_NAMES[family][template],
                   "template_unseen_in_engineering": template == 3, "prompt_sha256": canonical_sha256(prompt),
                   "latent_sha256": canonical_sha256({"facts": facts, "question": question, "answer": answer}),
                   "max_new_tokens": 32, "facts": facts, "question": question, "entity_tag": tag,
                   "alignment_partition": "holdout" if role == "calibration" and index % 4 == 3 else "fit" if role == "calibration" else None}
            row["answer"] = canonical_answer(row)
            rows.append(row)
    validate_rows(rows)
    return rows


def diagnostic_rows(originals: list[dict], count: int = 128) -> list[dict]:
    if count != 128 or len(originals) != 1024:
        raise ValueError("registered diagnostics require 128 variants from 1024 originals")
    result = []
    reorder_template = {"relational": 0, "counterfactual": 3, "compositional": 0,
                        "mathematical": 2, "symbolic_logic": 3, "retrieval_multihop": 0,
                        "ordinary_qa_preservation": 0, "instruction_evidence": 2}
    for family in FAMILIES:
        group = [r for r in originals if r["family"] == family]
        for index in range(16):
            variant = ("paraphrase", "irrelevant_context", "reordered_evidence", "renamed_entities")[index % 4]
            template = 0 if variant == "paraphrase" else reorder_template[family] if variant == "reordered_evidence" else index % 4
            if family == "relational":
                variant = ("paraphrase", "changed_relation_direction", "longer_chain", "reordered_evidence")[index % 4]
                template = 2 if variant == "longer_chain" else 0
            # One distinct original per diagnostic, with template chosen by the
            # transformation's semantic preconditions. Never relabel a preamble.
            source = group[index * 4 + template]
            row = json.loads(json.dumps(source))
            facts, question = row["facts"], row["question"]
            if variant == "paraphrase":
                for pattern, replacement in _PARAPHRASES[family]:
                    facts = [re.sub(pattern, replacement, fact) for fact in facts]
                if facts == source["facts"]:
                    raise AssertionError("paraphrase did not reword propositions")
            elif variant == "irrelevant_context":
                facts.insert(0, "Unrelated note: a calendar hangs on an office wall; this changes none of the following facts.")
            elif variant == "reordered_evidence":
                facts.reverse()
                if facts == source["facts"]:
                    raise AssertionError("reordering requires multiple distinct fact records")
            elif variant == "renamed_entities":
                old, new = source["entity_tag"], source["entity_tag"] + "Renamed"
                facts = [value.replace(old, new) for value in facts]
                question = question.replace(old, new)
                row["entity_tag"] = new
            # Some repeated surface probes are replaced by stronger semantic
            # diagnostics. Each remains linked to its source scenario.
            if variant == "changed_relation_direction":
                facts = [re.sub(r"(\S+) is taller than (\S+)\.", r"\2 is shorter than \1.", fact)
                         for fact in source["facts"]]
            elif variant == "longer_chain":
                facts = list(source["facts"]) + [f"{source['entity_tag']}D is a parent of {source['entity_tag']}E."]
                question = f"Who is the ancestor 4 generations before {source['entity_tag']}E?"
            elif family == "counterfactual" and index in (0, 4):
                variant = "counterfactual_change"
                facts = [re.sub(r"set relay to ([01])", lambda m: "set relay to " + str(1-int(m[1])), fact)
                         for fact in source["facts"]]
            row.update(example_id=source["scenario_id"] + "_" + variant,
                       case_id=source["scenario_id"] + "_" + variant, variant=variant,
                       facts=facts, question=question, primary_eligible=False,
                       source_example_id=source["example_id"], source_prompt_sha256=source["prompt_sha256"])
            target = semantic_answer(row)
            row["scorer"] = {"type": "json_exact", "value": target} if source["scorer"]["type"] == "json_exact" else {"type": "exact", "answers": [str(target)]}
            row["answer"] = canonical_answer(row)
            row["prompt"] = "\n".join(facts + [question, "Give only the requested answer, with no explanation."])
            row["prompt_sha256"] = canonical_sha256(row["prompt"])
            # scenario_id and latent_sha256 deliberately identify the unchanged
            # source allocation, including counterfactual diagnostic interventions.
            result.append(row)
    validate_rows(result)
    return result


def validate_rows(rows: list[dict]) -> dict:
    required = {"schema_version", "example_id", "case_id", "scenario_id", "family", "split", "track", "prompt", "scorer", "answer", "variant", "template_id", "facts", "question", "prompt_sha256", "latent_sha256", "alignment_partition"}
    ids, prompts, scenario_roles, scenario_partitions, latent_scenarios = set(), set(), {}, {}, {}
    if not rows:
        raise ValueError("empty bank")
    for row in rows:
        if not required <= row.keys() or row["example_id"] != row["case_id"]:
            raise ValueError("invalid dataset schema")
        if row["track"] not in TRACKS or row["split"] not in ROLES or row["family"] not in FAMILIES:
            raise ValueError("invalid track, role, or family")
        if row["schema_version"] != GENERATOR_VERSION or row["template_id"] not in {row["family"] + "." + value for value in TEMPLATE_NAMES[row["family"]]}:
            raise ValueError("unregistered generator version or family template")
        if not isinstance(row["facts"], list) or not row["facts"] or any(not isinstance(v, str) or not v.strip() for v in row["facts"]) or not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("nonempty rendered facts and question required")
        rendered = "\n".join(row["facts"] + [row["question"], "Give only the requested answer, with no explanation."])
        if row["prompt"] != rendered:
            raise ValueError("prompt does not match oracle-visible facts and question")
        if row["example_id"] in ids or row["prompt_sha256"] in prompts:
            raise ValueError("duplicate example ID or prompt")
        ids.add(row["example_id"]); prompts.add(row["prompt_sha256"])
        role = (row["track"], row["split"])
        if scenario_roles.setdefault(row["scenario_id"], role) != role:
            raise ValueError("scenario crosses a track or split")
        partition = row["alignment_partition"]
        if row["split"] == "calibration":
            expected_partition = "holdout" if int(row["scenario_id"].rsplit("_", 1)[1]) % 4 == 3 else "fit"
            if partition != expected_partition:
                raise ValueError("calibration scenario violates frozen fit/holdout boundary")
        elif partition is not None:
            raise ValueError("only calibration can enter alignment partitions")
        if scenario_partitions.setdefault(row["scenario_id"], partition) != partition:
            raise ValueError("linked variants cross fit/holdout boundary")
        if latent_scenarios.setdefault(row["latent_sha256"], row["scenario_id"]) != row["scenario_id"]:
            raise ValueError("one latent scenario has multiple independent IDs")
        unseen = row["template_id"].endswith("." + TEMPLATE_NAMES[row["family"]][3])
        if row["template_unseen_in_engineering"] != unseen or unseen and row["split"] != "confirmatory_dev":
            raise ValueError("held-out template crosses engineering boundary")
        if row["prompt_sha256"] != canonical_sha256(row["prompt"]):
            raise ValueError("prompt hash mismatch")
        if row["answer"] != canonical_answer(row) or not score_response(row["answer"], row["scorer"])["correct"]:
            raise ValueError("canonical answer fails its frozen scorer")
        # This check catches a consistently corrupted gold answer AND scorer.
        target = semantic_answer(row)
        oracle_canonical = json.dumps(target, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if row["scorer"]["type"] == "json_exact" else str(target)
        if row["answer"] != oracle_canonical:
            raise ValueError("gold answer contradicts independent rendered-fact oracle")
        if row["variant"] == "original" and row["latent_sha256"] != canonical_sha256({"facts": row["facts"], "question": row["question"], "answer": target}):
            raise ValueError("original latent hash mismatch")
    return {"examples": len(rows), "scenarios": len(scenario_roles),
            "families": dict(sorted(Counter(row["family"] for row in rows).items())),
            "templates": dict(sorted(Counter(row["template_id"] for row in rows).items()))}


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def build_banks(root: Path, seed: int = 20260905) -> dict[str, str]:
    root = guard_non_sealed_path(Path(root))
    generator_hash = sha256_file(Path(__file__))
    all_ids, all_prompts, all_latents = set(), set(), set()
    manifests = {}
    for track in TRACKS:
        directory = root / track
        directory.mkdir(parents=True, exist_ok=False)
        datasets = {}
        for role in ROLES:
            rows = generate_rows(track, role, seed)
            ids = {r["scenario_id"] for r in rows}
            prompts = {r["prompt_sha256"] for r in rows}
            latents = {r["latent_sha256"] for r in rows}
            if ids & all_ids or prompts & all_prompts or latents & all_latents:
                raise ValueError("cross-track or cross-role scenario overlap")
            all_ids |= ids; all_prompts |= prompts; all_latents |= latents
            path = directory / (role + ".jsonl")
            _write_rows(path, rows)
            datasets[role] = {"path": path.name, "sha256": sha256_file(path), **validate_rows(rows)}
            if role == "confirmatory_dev":
                diagnostics = diagnostic_rows(rows)
                path = directory / "confirmatory_diagnostics.jsonl"
                _write_rows(path, diagnostics)
                datasets["confirmatory_diagnostics"] = {"path": path.name, "sha256": sha256_file(path), **validate_rows(diagnostics), "independent_primary_n": 0}
        manifest = {"schema_version": GENERATOR_VERSION, "track": track, "seed": seed, "status": "FROZEN_UNSEALED",
                    "sealed_access": False, "generator_sha256": generator_hash, "scorer_sha256": scorer_sha256(),
                    "datasets": datasets, "calibration_holdout_rule": "scenario index modulo 4 equals 3",
                    "primary_role": "confirmatory_dev", "primary_independent_scenarios": 1024,
                    "diagnostics_count_as_primary": False, "unseen_engineering_template_index": 3,
                    "answer_validation": "independent rendered-proposition parser/solver over every row; 32 hand-authored oracle fixtures",
                    "diagnostic_variants": dict(sorted(Counter(r["variant"] for r in diagnostics).items()))}
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = directory / "manifest.json"
        write_json(manifest_path, manifest)
        manifests[track] = str(manifest_path)
    return manifests


def load_frozen_dataset(manifest_path: Path, *, track: str, role: str) -> list[dict]:
    # Resolve and reject all disallowed names BEFORE opening either file.
    manifest_path = guard_non_sealed_path(Path(manifest_path))
    if track not in TRACKS or role not in (*ROLES, "confirmatory_diagnostics"):
        raise ValueError("unregistered track or role")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256", None)
    if canonical_sha256(manifest) != expected:
        raise ValueError("manifest hash mismatch")
    if manifest.get("status") != "FROZEN_UNSEALED" or manifest.get("track") != track or manifest.get("sealed_access") is not False:
        raise ValueError("wrong track or unfrozen data")
    if manifest.get("scorer_sha256") != scorer_sha256() or manifest.get("generator_sha256") != sha256_file(Path(__file__)):
        raise ValueError("scorer/generator hash mismatch; version and refreeze prospectively")
    info = manifest["datasets"][role]
    path = guard_non_sealed_path(manifest_path.parent / info["path"])
    require_within(path, manifest_path.parent)
    if sha256_file(path) != info["sha256"]:
        raise ValueError("dataset hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = validate_rows(rows)
    if any(counts[key] != info[key] for key in ("examples", "scenarios", "families", "templates")):
        raise ValueError("manifest counts mismatch")
    expected_split = "confirmatory_dev" if role == "confirmatory_diagnostics" else role
    if any(row["track"] != track or row["split"] != expected_split for row in rows):
        raise ValueError("row crosses a track or role")
    if role == "confirmatory_dev" and any(row["variant"] != "original" for row in rows):
        raise ValueError("diagnostic variants cannot enter primary bank")
    return rows


def alignment_prompts(manifest_path: Path, *, partition: str) -> list[dict]:
    """Bridge-facing interface intentionally exposes no answers or scorers."""
    if partition not in {"fit", "holdout"}:
        raise ValueError("invalid alignment partition")
    rows = load_frozen_dataset(manifest_path, track="cross_scale", role="calibration")
    return [{key: row[key] for key in ("example_id", "scenario_id", "family", "prompt", "prompt_sha256")}
            for row in rows if row["alignment_partition"] == partition]
