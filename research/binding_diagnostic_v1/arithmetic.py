"""Fresh arithmetic diagnostic construction; standard library and no file I/O.

Ninety-six parent worlds each contribute two numerical variants crossed with
two response forms. Parent worlds, not the 384 related views, are statistical
units. Policy messages contain task facts and a response instruction only.
Gold, grouping factors and exact typed targets are separate.

The targets adapt new tasks to the UNCHANGED frozen retention grammar. This
module never loads that grammar, a tokenizer, a model or any previous prompts.
The explicit clock/duration numeric forms request minute-of-day/elapsed minutes;
their contrast includes representation conversion, not only surface formatting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import random
import re

SCHEMA = 1
RECORD = "FRESH_ARITHMETIC_DIAGNOSTIC_V1"
SYSTEM = "Solve the stated arithmetic exactly. Follow the requested answer form. Exclude explanations."
FAMILIES = ("percentage_count_currency", "ratio_quantity", "clock_duration")
RESPONSE_TYPES = ("numeric_value_only", "unit_explicit")
SUBTYPES = ("percentage_count", "percentage_currency", "ratio_mass", "ratio_volume", "clock_arrival", "elapsed_duration")
FROZEN_SCORER = {
    "path": "C:/Users/clclc/BusinessProject/ChatBubble-binding-mixture-pilot-20260920/experiments/architecture_benchmark_001/retention.py",
    "sha256": "c52b6a48cb811e3c272793a66765e82152fe170e8307eea153c3b1b6c5c58322",
    "policy_axis": "explicit count/money/quantity: permissive permits bare values, strict requires type annotation; numeric tasks are number targets; clock/duration rules are unchanged",
}

PREFIXES = ("Aven", "Brel", "Corda", "Dres", "Elvi", "Fenn", "Garo", "Hesta", "Ivor", "Juna", "Kelm", "Luro")
SUFFIXES = ("brook", "cove", "dell", "field", "grove", "haven", "mere", "ridge")
NOUNS = ("resin disc", "ceramic pin", "linen tab", "brass clip")
INSTRUCTIONS = {
    "count": {
        "numeric_value_only": "Reply with only the exact count as a number. Omit the counted noun and all other words.",
        "unit_explicit": "Reply with the exact count followed by the counted noun. Exclude explanations.",
    },
    "money": {
        "numeric_value_only": "Reply with only the numerical value in dollars. Omit the currency symbol and all words. Do not round.",
        "unit_explicit": "Reply with the exact dollar amount, including a dollar symbol or currency word. Exclude explanations.",
    },
    "quantity": {
        "numeric_value_only": "Reply with only the numerical value in the requested unit. Omit all unit words. Use an integer, terminating decimal, or exact fraction; do not round.",
        "unit_explicit": "Reply with the exact quantity followed by the requested unit. Use an integer, terminating decimal, or exact fraction; do not round.",
    },
    "clock": {
        "numeric_value_only": "Reply with only the integer number of minutes since midnight for the arrival time. Omit all words.",
        "unit_explicit": "Reply with a clock time, using AM/PM or 24-hour HH:MM. Exclude elapsed durations and explanations.",
    },
    "duration": {
        "numeric_value_only": "Reply with only the integer number of elapsed minutes. Omit all words.",
        "unit_explicit": "Reply with the elapsed duration as a number followed by minutes. Exclude clock times and explanations.",
    },
}

PATTERNS = {
    "percentage_count": re.compile(r"At (?P<context>[A-Za-z]+), an inventory has (?P<total>\d+) (?P<noun>[a-z]+ [a-z]+)s\. Exactly (?P<percent>\d+) percent are marked violet\. How many (?P=noun)s are marked violet\?"),
    "percentage_currency": re.compile(r"At (?P<context>[A-Za-z]+), a price is (?P<total>\d+) dollars\. A rebate is exactly (?P<percent>\d+) percent of that price\. Determine the rebate amount in dollars\."),
    "ratio_mass": re.compile(r"At (?P<context>[A-Za-z]+), a coating rule uses (?P<quantity>\d+) grams of powder for every (?P<base>\d+) panels\. The same rule is used for (?P<request>\d+) panels\. How many grams of powder are needed\?"),
    "ratio_volume": re.compile(r"At (?P<context>[A-Za-z]+), an irrigation rule assigns (?P<quantity>\d+) liters of water to every (?P<base>\d+) planters\. Apply that rule to (?P<request>\d+) planters\. How many liters of water are needed\?"),
    "clock_arrival": re.compile(r"At (?P<context>[A-Za-z]+), a courier departs at (?P<start>\d{2}:\d{2}) on a 24-hour clock\. The trip lasts (?P<elapsed>\d+) minutes\. Determine the arrival clock time\. If midnight is crossed, use the time on the following day\."),
    "elapsed_duration": re.compile(r"At (?P<context>[A-Za-z]+), a drying interval begins at (?P<start>\d{2}:\d{2}) and ends at (?P<end>\d{2}:\d{2}) on the (?P<day>same|next) day\. The clock readings use a 24-hour format\. Determine the elapsed time in minutes\."),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _rational(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _number_text(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    if 100 % value.denominator == 0:
        cents = value.numerator * (100 // value.denominator)
        return f"{cents // 100}.{cents % 100:02d}".rstrip("0")
    return _rational(value)


def _clock(value):
    require(type(value) is int and 0 <= value < 1440, "invalid minute of day")
    return f"{value // 60:02d}:{value % 60:02d}"


def _parse_clock(value):
    require(re.fullmatch(r"\d{2}:\d{2}", value) is not None, "invalid clock syntax")
    hour, minute = map(int, value.split(":"))
    require(0 <= hour <= 23 and 0 <= minute <= 59, "invalid 24-hour clock value")
    return 60 * hour + minute


def parse_policy_input(prompt):
    """Decode only the two policy messages; recover an exact independent oracle.

    This parser accepts no gold or metadata arguments. All operands, requested
    answer types and day transitions are recovered from full-matched text.
    """
    require(isinstance(prompt, list) and len(prompt) == 2 and
            all(isinstance(row, dict) and set(row) == {"role", "content"} for row in prompt), "exactly two role/content messages required")
    require(prompt[0] == {"role": "system", "content": SYSTEM} and prompt[1]["role"] == "user" and
            isinstance(prompt[1]["content"], str), "unregistered arithmetic system/user shape")
    pieces = prompt[1]["content"].split("\n\n")
    require(len(pieces) == 2, "arithmetic facts and response instruction must be separate")
    facts, instruction = pieces
    matches = [(kind, pattern.fullmatch(facts)) for kind, pattern in PATTERNS.items()]
    matches = [(kind, match) for kind, match in matches if match is not None]
    require(len(matches) == 1, "facts do not match one independent arithmetic decoder")
    subtype, match = matches[0]; parameters = match.groupdict()
    for key in ("total", "percent", "quantity", "base", "request", "elapsed"):
        if key in parameters:
            parameters[key] = int(parameters[key])
            require(parameters[key] > 0, "arithmetic operands must be positive")
    if subtype.startswith("percentage"):
        require(parameters["percent"] < 100, "percentage must lie strictly between zero and one hundred")
        value = Fraction(parameters["total"] * parameters["percent"], 100)
        family = FAMILIES[0]
        kind = "count" if subtype == "percentage_count" else "money"
        if kind == "count":
            require(parameters["noun"] in NOUNS and value.denominator == 1, "count oracle must be a whole registered entity count")
        unit = parameters["noun"] if kind == "count" else "dollar"
    elif subtype.startswith("ratio"):
        value = Fraction(parameters["request"], parameters["base"]) * parameters["quantity"]
        family, kind = FAMILIES[1], "quantity"
        unit = "gram" if subtype == "ratio_mass" else "liter"
    elif subtype == "clock_arrival":
        start = _parse_clock(parameters["start"])
        require(parameters["elapsed"] < 1440, "clock interval exceeds one day")
        value = Fraction((start + parameters["elapsed"]) % 1440)
        family, kind, unit = FAMILIES[2], "clock", "minute_of_day"
    else:
        start, end = _parse_clock(parameters["start"]), _parse_clock(parameters["end"])
        duration = end - start + (1440 if parameters["day"] == "next" else 0)
        require(0 < duration < 1440, "duration must be positive and shorter than one day")
        value = Fraction(duration)
        family, kind, unit = FAMILIES[2], "duration", "minute"
    response = [key for key, text in INSTRUCTIONS[kind].items() if instruction == text]
    require(len(response) == 1, "response instruction mismatches the task's answer type")
    return {"family": family, "subtype": subtype, "context": parameters["context"], "parameters": parameters,
            "response_type": response[0], "explicit_kind": kind, "answer_unit": unit, "exact_value": _rational(value)}


def typed_targets(parsed):
    """Adapt new oracles to frozen target schemas without changing the grammar."""
    value = Fraction(parsed["exact_value"]); kind = parsed["explicit_kind"]
    if parsed["response_type"] == "numeric_value_only":
        target = {"kind": "number", "value": _rational(value), "bare_allowed": True}
        return {policy: dict(target) for policy in ("permissive", "strict")}
    if kind == "clock":
        require(value.denominator == 1, "clock target must be an integer minute")
        return {policy: {"kind": "clock", "minute_of_day": int(value)} for policy in ("permissive", "strict")}
    if kind == "duration":
        return {policy: {"kind": "duration", "value": _rational(value * 60), "bare_allowed": False}
                for policy in ("permissive", "strict")}
    result = {}
    for policy in ("permissive", "strict"):
        target = {"kind": kind, "value": _rational(value), "bare_allowed": policy == "permissive"}
        if kind == "count":
            noun = parsed["answer_unit"]
            target["unit_aliases"] = [noun, noun + "s"]
        elif kind == "quantity":
            target["unit"] = parsed["answer_unit"]
        require(kind in ("count", "quantity", "money"), "unknown explicit target kind")
        result[policy] = target
    return result


def reference_text(parsed):
    value = Fraction(parsed["exact_value"]); text = _number_text(value)
    if parsed["response_type"] == "numeric_value_only":
        return text
    kind = parsed["explicit_kind"]
    if kind == "clock":
        return _clock(int(value))
    if kind == "money":
        return "$" + text
    unit = "minute" if kind == "duration" else parsed["answer_unit"]
    return text + " " + unit + ("" if value == 1 else "s")


def _world_rng(seed, world):
    encoded = f"fresh-arithmetic-diagnostic-v1:{seed}:{world}".encode("ascii")
    return random.Random(int.from_bytes(hashlib.sha256(encoded).digest(), "big"))


def _make_world(index, context, rng):
    """Construct latent arithmetic independently of parse_policy_input's decoder."""
    family_index, within = divmod(index, 32); subtype = SUBTYPES[family_index * 2 + within % 2]
    world = {"family": FAMILIES[family_index], "subtype": subtype, "context": context}
    variants = []
    if subtype.startswith("percentage"):
        percent = rng.choice((10, 15, 20, 25, 30, 40, 50, 60, 75))
        noun = NOUNS[(within // 2) % len(NOUNS)]
        for variant in (0, 1):
            total = (2 * rng.randint(8, 39) + variant) * (20 if subtype == "percentage_count" else 1)
            exact = Fraction(percent, 100) * total
            if subtype == "percentage_count":
                facts = f"At {context}, an inventory has {total} {noun}s. Exactly {percent} percent are marked violet. How many {noun}s are marked violet?"
                kind, unit = "count", noun
            else:
                facts = f"At {context}, a price is {total} dollars. A rebate is exactly {percent} percent of that price. Determine the rebate amount in dollars."
                kind, unit = "money", "dollar"
            variants.append((facts, kind, unit, exact))
    elif subtype.startswith("ratio"):
        base, quantity = rng.choice((3, 4, 5, 6, 7, 8, 9)), rng.randint(4, 29)
        for variant in (0, 1):
            request = 2 * rng.randint(5, 21) + variant
            exact = Fraction(quantity * request, base)
            if subtype == "ratio_mass":
                facts = f"At {context}, a coating rule uses {quantity} grams of powder for every {base} panels. The same rule is used for {request} panels. How many grams of powder are needed?"
                unit = "gram"
            else:
                facts = f"At {context}, an irrigation rule assigns {quantity} liters of water to every {base} planters. Apply that rule to {request} planters. How many liters of water are needed?"
                unit = "liter"
            variants.append((facts, "quantity", unit, exact))
    elif subtype == "clock_arrival":
        boundary = (within // 2) % 4
        start = (23 * 60 + rng.randint(15, 35) if boundary == 0 else
                 11 * 60 + rng.randint(25, 40) if boundary == 1 else
                 rng.randint(6, 9) * 60 + rng.randint(10, 40) if boundary == 2 else
                 rng.randint(15, 19) * 60 + rng.randint(10, 40))
        elapsed = rng.randint(65, 155)
        for variant in (0, 1):
            departure, duration = start + 7 * variant, elapsed + 11 * variant
            exact = Fraction((departure + duration) % 1440)
            facts = f"At {context}, a courier departs at {_clock(departure)} on a 24-hour clock. The trip lasts {duration} minutes. Determine the arrival clock time. If midnight is crossed, use the time on the following day."
            variants.append((facts, "clock", "minute_of_day", exact))
    else:
        overnight = (within // 2) % 2 == 0
        start = 23 * 60 + rng.randint(20, 35) if overnight else rng.randint(6, 18) * 60 + rng.randint(5, 35)
        elapsed = rng.randint(65, 155)
        for variant in (0, 1):
            begin, duration = start + 7 * variant, elapsed + 11 * variant
            end = (begin + duration) % 1440; day = "next" if overnight else "same"
            facts = f"At {context}, a drying interval begins at {_clock(begin)} and ends at {_clock(end)} on the {day} day. The clock readings use a 24-hour format. Determine the elapsed time in minutes."
            variants.append((facts, "duration", "minute", Fraction(duration)))
    return world, variants


def build_rows(seed=31):
    """Return a deterministic JSON-serializable bank in memory; write nothing.

    policy_inputs: id, parent_id, prompt (two role/content messages).
    gold: id, parent_id, family, subtype, numeric_variant, response_type,
          reference, targets{permissive,strict}, oracle{exact_value,unit,kind}.
    metadata: one row per world with context, six-way count type, template ID,
              paired row IDs, and explicit limited provenance scope.
    """
    require(type(seed) is int and 0 <= seed < 2**63, "seed must be a nonnegative 63-bit integer")
    names = [prefix + suffix for prefix in PREFIXES for suffix in SUFFIXES]
    random.Random(seed).shuffle(names)
    policy, gold, metadata = [], [], []
    for index, context in enumerate(names):
        parent = f"fresh-arithmetic-v1-s{seed}-world{index:03d}"
        world, variants = _make_world(index, context, _world_rng(seed, index))
        ids = []
        for variant, (facts, kind, unit, exact) in enumerate(variants):
            for response in RESPONSE_TYPES:
                view_key = hashlib.sha256(f"{parent}|{variant}|{response}".encode("ascii")).hexdigest()[:20]
                identifier = f"{parent}-view-{view_key}"
                prompt = [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": facts + "\n\n" + INSTRUCTIONS[kind][response]}]
                generated = {"explicit_kind": kind, "answer_unit": unit, "exact_value": _rational(exact), "response_type": response}
                policy.append({"id": identifier, "parent_id": parent, "prompt": prompt})
                gold.append({"id": identifier, "parent_id": parent, "family": world["family"], "subtype": world["subtype"],
                             "numeric_variant": variant, "response_type": response, "reference": reference_text(generated),
                             "targets": typed_targets(generated), "oracle": {"exact_value": _rational(exact), "unit": unit, "kind": kind}})
                ids.append(identifier)
        metadata.append({"parent_id": parent, **world, "world_index": index, "count_type": world["subtype"],
                         "template_id": "fresh-arithmetic-v1/" + world["subtype"], "rows": 4, "numeric_variants": 2,
                         "response_types": list(RESPONSE_TYPES), "row_ids": ids,
                         "independence_unit": "parent_world; four correlated views",
                         "construction": "independent seeded operands and newly authored context templates; no model-output filtering",
                         "external_overlap_status": "UNVERIFIED; no prior retention prompts or reserved raw sources read"})
    bundle = {"schema": SCHEMA, "record": RECORD, "seed": seed,
              "policy_inputs": policy, "gold": gold, "metadata": metadata}
    validate_rows(bundle)
    return bundle


def validate_rows(bundle):
    """Validate full geometry and reproduce gold from text independently."""
    require(set(bundle) == {"schema", "record", "seed", "policy_inputs", "gold", "metadata"} and
            bundle["schema"] == SCHEMA and bundle["record"] == RECORD and type(bundle["seed"]) is int,
            "invalid arithmetic bundle contract")
    policy, gold, meta = bundle["policy_inputs"], bundle["gold"], bundle["metadata"]
    require(len(policy) == len(gold) == 384 and len(meta) == 96, "requires 96 worlds and exactly 384 views")
    inputs = {row["id"]: row for row in policy}; answers = {row["id"]: row for row in gold}
    worlds = {row["parent_id"]: row for row in meta}
    require(len(inputs) == len(answers) == 384 and inputs.keys() == answers.keys() and len(worlds) == 96,
            "duplicate or mismatched arithmetic identities")
    require(len({row["context"] for row in meta}) == 96 and
            Counter(row["family"] for row in meta) == {name: 32 for name in FAMILIES} and
            Counter(row["subtype"] for row in meta) == {name: 16 for name in SUBTYPES}, "world subtype/context geometry differs")
    grouped = defaultdict(list)
    for key, row in inputs.items():
        require(set(row) == {"id", "parent_id", "prompt"}, "gold or analysis fields leaked into policy input")
        answer = answers[key]
        require(set(answer) == {"id", "parent_id", "family", "subtype", "numeric_variant", "response_type", "reference", "targets", "oracle"}, "gold schema differs")
        require(row["parent_id"] == answer["parent_id"] and row["parent_id"] in worlds and
                type(answer["numeric_variant"]) is int and answer["numeric_variant"] in (0, 1), "row/parent/numerical variant differs")
        parsed = parse_policy_input(row["prompt"]); world = worlds[row["parent_id"]]
        require(parsed["context"] == world["context"] and parsed["family"] == answer["family"] == world["family"] and
                parsed["subtype"] == answer["subtype"] == world["subtype"] and parsed["response_type"] == answer["response_type"], "policy text/grouping metadata disagree")
        require(answer["oracle"] == {"exact_value": parsed["exact_value"], "unit": parsed["answer_unit"], "kind": parsed["explicit_kind"]} and
                answer["targets"] == typed_targets(parsed) and answer["reference"] == reference_text(parsed), "independent text oracle/typed target disagrees with gold")
        grouped[row["parent_id"]].append((answer, parsed))
    require(grouped.keys() == worlds.keys(), "unrepresented arithmetic world")
    for parent, values in grouped.items():
        world = worlds[parent]
        require(len(values) == 4 and {(v[0]["numeric_variant"], v[0]["response_type"]) for v in values} ==
                {(variant, response) for variant in (0, 1) for response in RESPONSE_TYPES}, "world is not the full two-by-two view crossing")
        require(set(world["row_ids"]) == {v[0]["id"] for v in values} and len(world["row_ids"]) == 4 and world["rows"] == 4 and
                world["numeric_variants"] == 2 and world["response_types"] == list(RESPONSE_TYPES) and world["count_type"] == world["subtype"], "world membership/count metadata differs")
        by_variant = {}
        for variant in (0, 1):
            pair = [parsed for answer, parsed in values if answer["numeric_variant"] == variant]
            require({k: v for k, v in pair[0].items() if k != "response_type"} ==
                    {k: v for k, v in pair[1].items() if k != "response_type"}, "response forms changed arithmetic facts")
            by_variant[variant] = pair[0]["parameters"]
        require(by_variant[0] != by_variant[1], "numerical variants reused identical operands")
    return {"worlds": 96, "views": 384, "worlds_by_family": dict(Counter(row["family"] for row in meta)),
            "worlds_by_subtype": dict(Counter(row["subtype"] for row in meta)), "independent_text_oracle": "PASS",
            "policy_gold_separation": "PASS", "model_calls": 0, "tokenizer_calls": 0, "human_review": "NOT_PERFORMED"}
