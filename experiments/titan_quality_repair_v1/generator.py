"""Prospectively versioned quality tasks; no model, Torch, or bank I/O.

Calling build_split('final') materializes new prospective final membership. The
authorized offline compiler can construct it and audit independent golds, leakage,
and token masks before any model outcomes, then seal it externally. Native model
evaluation/access requires the separately authorized final stage. No old sealed
bank is read. Arithmetic memberships use a role-aware hash partition, so changing
names or prompt styles cannot move train arithmetic into evaluation.
"""
from __future__ import annotations

import copy
import random

from . import oracles, schema

VERSION = "titan-quality-data-1"
GENERATOR_SEED = 20261005
MAX_ATTEMPTS_PER_WORLD = 10000
LEDGER_RANGE = (-9999, 9999)
REGISTER_OPERAND_RANGE = (-96, 96)
OLD_LEDGER_OPERAND_RANGE = (-9, 12)
OLD_REGISTER_OPERAND_RANGE = (-8, 11)
H_NATIVE = {
    "native_a": "Solve the problem silently and give only the concise final answer. Use only the supplied facts; do not show chain-of-thought or invent missing information.",
    "native_b": "Use the stated facts to reach the result. Return only the compact conclusion, not the reasoning steps.",
}
LEDGER_RULES = ("Apply one event. If status is not posted, preserve all balances. add adds amount to account. "
                "scale multiplies account by amount. transfer subtracts amount from account and adds it to other simultaneously.")
REGISTER_RULES = ("Registers are integers modulo 97. Apply one instruction. add adds value to register. "
                  "conditional adds value if other is greater than threshold, otherwise subtracts value. "
                  "Reduce the updated register modulo 97 and preserve the others.")
SCENARIOS = ("supported", "relevant_missing", "irrelevant_missing", "relevant_conflict",
             "incomplete_cue_supported", "complete_cue_missing", "irrelevant_conflict", "complete_cue_conflict")


def design_metadata():
    return {"version": VERSION, "generator_seed": GENERATOR_SEED,
            "families": list(schema.FAMILIES), "styles": list(schema.STYLES),
            "ledger_operand_range": list(LEDGER_RANGE), "ledger_balance_range": list(LEDGER_RANGE),
            "old_ledger_amount_support": list(OLD_LEDGER_OPERAND_RANGE),
            "register_operand_range": list(REGISTER_OPERAND_RANGE),
            "old_register_value_support": list(OLD_REGISTER_OPERAND_RANGE),
            "register_initial_state": [0, 96], "register_threshold": [0, 96],
            "register_support_expansion": "operand outside documented -8..11 OR modulo wrap outside the preserved PARENTQ no-wrap prototype",
            "partition_rule": "sha256(role-aware-operative-key) modulo 10: train 0..5; dev 6..7; final 8..9",
            "development": {"independent_worlds_per_family": 64, "attached_variants_per_repair_family": 16, "prompts": 512},
            "final": {"independent_worlds_per_family": 512, "attached_variants_per_repair_family": 128, "prompts": 4096},
            "generation_exhaustion": "STOP after 10000 rejected candidate groups; never drop or replace a required stratum",
            "old_support_interpretation": "ledger amounts and register values have documented old support; no invented old ledger balance bound",
            "training_dose": "2000 including exactly 400 source-provenanced existing replay examples for 20% recipes"}


def operative_number_key(family, payload):
    """Same touched-input/output roles as preserved PARENTQ keys, independent solver."""
    if family in {"ledger", "ledger_replay"}:
        before, event, after = payload["before"], payload["event"], oracles.ledger(payload)
        a, kind = event["account"], event["kind"]
        if kind in {"add", "scale"}:
            return ("ledger_replay", kind, event["status"], before[a], event["amount"], after[a])
        other = event["other"]
        return ("ledger_replay", kind, event["status"], before[a], before[other], event["amount"], after[a], after[other])
    if family in {"register", "register_machine"}:
        before, event, after = payload["before"], payload["event"], oracles.register(payload)
        r = event["register"]
        if event["kind"] == "add":
            return ("register_machine", "add", before[r], event["value"], after[r])
        other = event["other"]
        return ("register_machine", "conditional", before[r], before[other], event["threshold"], event["value"],
                before[other] > event["threshold"], after[r])
    if family == "arithmetic":
        return ("arithmetic", payload["operator"], payload["a"], payload["b"], oracles.arithmetic(payload))
    if family == "instruction":
        return (family, payload["record"], sorted(payload["fields"]), oracles.instruction(payload))
    return (family, payload["question"], sorted(payload["facts"], key=schema.canonical))


def operative_key(family, payload):
    return schema.canonical(operative_number_key(family, payload))


def partition_of(key):
    bucket = int(schema.digest(key), 16) % 10
    return "train" if bucket < 6 else "dev" if bucket < 8 else "final"


def _rng(split, family, difficulty):
    return random.Random(int(schema.digest([VERSION, GENERATOR_SEED, split, family, difficulty]), 16))


def _sign(index):
    # Exact marginal sign balance for all registered even family counts, with
    # positive and negative cases under every contract in a diverse recipe.
    return 1 if ((index // 4) + (index % 4) // 2) % 2 == 0 else -1


def _scenario(index):
    # Every full 32-row block crosses all eight scenarios with four contracts.
    # Even/odd scenario encodes supported/unsupported; marginal support remains
    # exactly balanced for the registered 64/400/500/512 family counts.
    return 2 * ((index // 8 + index % 4) % 4) + ((index // 4 + (index % 4) // 2) % 2)


def _fact_payload(family, index, rng, scenario=None):
    scene = SCENARIOS[_scenario(index) if scenario is None else scenario]
    value = rng.randint(1000, 999999) * _sign(index)
    subject = "parcel-" + str(rng.randint(1000000, 999999999))
    field = "quantity" if family == "answerability" else "dock"
    answer = value if family == "answerability" else "Dock " + str(abs(value))
    alternate = answer + 1 if isinstance(answer, int) else answer + " annex"
    facts = [{"subject": subject, "field": field, "value": answer},
             {"subject": subject, "field": "colour", "value": "blue"}]
    if scene in {"relevant_missing", "complete_cue_missing"}:
        facts = facts[1:]
    elif scene == "irrelevant_missing":
        facts = facts[:1]
    elif scene in {"relevant_conflict", "complete_cue_conflict"}:
        facts.append({"subject": subject, "field": field, "value": alternate})
    elif scene == "irrelevant_conflict":
        facts.append({"subject": subject, "field": "colour", "value": "red"})
    cue = "These notes are complete." if scene.startswith("complete_cue") else (
        "The notes may be incomplete." if scene == "incomplete_cue_supported" else
        "The parcel colour is not recorded." if scene == "irrelevant_missing" else "")
    return {"facts": facts, "question": {"subject": subject, "field": field}, "cue": cue,
            "irrelevant_field_missing": scene == "irrelevant_missing", "scenario": scene,
            "latent_sign": "positive" if value > 0 else "negative"}


def _payload(family, index, count, rng):
    if family == "ledger":
        amount = rng.randint(1, 9999) * _sign(index)
        return {"before": {name: rng.randint(*LEDGER_RANGE) for name in ("A", "B", "C", "D")},
                "event": {"kind": ("add", "scale", "transfer")[(index // 2) % 3], "account": "A", "other": "B",
                          "amount": amount, "status": "posted", "tick": rng.randint(1, 99)}}
    if family == "register":
        conditional_count = (count // 2 // 4) * 4
        conditional = index < conditional_count
        other = rng.randint(1, 95)
        true = index % 2 == 0
        return {"before": {"X": rng.randint(0, 96), "Y": other, "Z": rng.randint(0, 96)},
                "event": {"kind": "conditional" if conditional else "add", "register": "X", "other": "Y",
                          "value": rng.randint(1, 96) * _sign(index), "threshold": other - 1 if true else other,
                          "tick": rng.randint(1, 99)}}
    if family in {"answerability", "grounded_qa", "concise_grounded"}:
        return _fact_payload(family, index, rng)
    if family == "arithmetic":
        return {"a": rng.randint(1, 9999) * _sign(index), "b": rng.randint(-9999, 9999),
                "operator": ("add", "subtract", "multiply")[index % 3]}
    if family == "instruction":
        return {"record": {"code": "C" + str(rng.randint(10000, 999999999)), "count": rng.randint(-9999, 9999),
                           "ignored": "Do not copy this field"}, "fields": ["code", "count"]}
    raise ValueError("STOP_UNKNOWN_GENERATOR_FAMILY")


def _counterfactual(family, first, index):
    second = copy.deepcopy(first)
    if family == "ledger":
        first["event"]["amount"] = abs(first["event"]["amount"])
        second["event"]["amount"] = -first["event"]["amount"]
    elif family == "register":
        if first["event"]["kind"] == "conditional":
            sign = 1 if (index // 2) % 2 == 0 else -1
            first["event"]["value"] = abs(first["event"]["value"]) * sign
            second["event"]["value"] = first["event"]["value"]
            other = first["before"][first["event"]["other"]]
            first["event"]["threshold"], second["event"]["threshold"] = other - 1, other
        else:
            first["event"]["value"] = abs(first["event"]["value"])
            second["event"]["value"] = -first["event"]["value"]
    else:
        q = first["question"]
        relevant = lambda fact: fact["subject"] == q["subject"] and fact["field"] == q["field"]
        # Start each hard pair with independently supported facts, then change
        # only queried evidence. Complete/incomplete language opposes the gold.
        facts = [fact for fact in first["facts"] if relevant(fact)]
        if not facts:
            raise ValueError("STOP_HARD_PAIR_WITHOUT_SUPPORTED_START")
        pair_case = (index // 2) % 4
        if pair_case in {0, 2}:
            second["facts"] = [fact for fact in first["facts"] if not relevant(fact)]
            second["scenario"] = "complete_cue_missing" if pair_case == 2 else "relevant_missing"
        else:
            value = facts[0]["value"]
            alternate = value + 1 if isinstance(value, int) else value + " annex"
            second["facts"] = copy.deepcopy(first["facts"]) + [{**q, "value": alternate}]
            second["scenario"] = "complete_cue_conflict" if pair_case == 3 else "relevant_conflict"
        second["cue"] = "These notes are complete." if pair_case in {2, 3} else ""
        second["irrelevant_field_missing"] = first["irrelevant_field_missing"]
        if (index // 4) % 2:
            return second, first
    return first, second


def _contract(family, style):
    if family in {"ledger", "register", "instruction"}:
        return {"kind": "json_object", "abstention": "UNKNOWN"}
    if family in {"grounded_qa", "concise_grounded"} or style == "natural":
        return {"kind": "text", "abstention": schema.ABSTENTION}
    return {"kind": "integer", "abstention": "UNKNOWN"}


def _system(family, style, contract):
    lead = H_NATIVE.get(style, "Answer concisely using only the supplied information." if style == "canonical"
                        else "Please use these records to give the requested result briefly.")
    rules = LEDGER_RULES if family == "ledger" else REGISTER_RULES if family == "register" else (
        "Copy exactly the requested fields from the record." if family == "instruction" else
        "Use only stated facts. Conflicting values are equally admissible. Missing relevant facts cannot be invented. "
        "A claim that notes are complete or incomplete adds no facts.")
    output = {"json_object": "Return only the complete answer as a JSON object, without Markdown or prose.",
              "integer": "Return only one decimal integer.",
              "text": "Return only the concise final answer, without explaining the reasoning."}[contract["kind"]]
    if family == "concise_grounded":
        output += ' For a supported location, use exactly "SUBJECT is at LOCATION." replacing the placeholders.'
    return " ".join((lead, rules, output, "If the requested answer is not supported, reply exactly: " + contract["abstention"]))


def render_prompt(row):
    """User text only; callers use row.system separately with exact chat templating."""
    p, family = row["payload"], row["family"]
    if row.get("is_replay"):
        return row["user"]
    if family in {"ledger", "register"}:
        return schema.canonical({"before": p["before"], "event": p["event"]})
    if family == "arithmetic":
        op = {"add": "+", "subtract": "-", "multiply": "*"}[p["operator"]]
        return f'Calculate ({p["a"]}) {op} ({p["b"]}).'
    if family == "instruction":
        return "Record: " + schema.canonical(p["record"]) + ". Return only these fields: " + ", ".join(p["fields"]) + "."
    facts = "\n".join(f'- {f["subject"]} {f["field"]}: {f["value"]}' for f in p["facts"])
    q = p["question"]
    return f'Notes:\n{facts}\n{p.get("cue", "")}\nWhat is the {q["field"]} of {q["subject"]}?'


def _row(family, payload, split, style, pair_id=None, world_id=None):
    answer = oracles.solve(family, payload)
    key = operative_key(family, payload)
    world_id = world_id or "world-" + schema.digest([VERSION, key])[:24]
    contract = _contract(family, style)
    payload = copy.deepcopy(payload)
    payload["expected_kind"] = "object" if isinstance(answer, dict) else "integer" if family in {"answerability", "arithmetic"} else "text"
    expansion = (not OLD_LEDGER_OPERAND_RANGE[0] <= payload["event"]["amount"] <= OLD_LEDGER_OPERAND_RANGE[1]
                 if family == "ledger" else False)
    wraps, operand_expansion = False, False
    if family == "register":
        e = payload["event"]
        delta = e["value"]
        if e["kind"] == "conditional" and payload["before"][e["other"]] <= e["threshold"]:
            delta = -delta
        raw = payload["before"][e["register"]] + delta
        wraps = not 0 <= raw < 97
        operand_expansion = not OLD_REGISTER_OPERAND_RANGE[0] <= e["value"] <= OLD_REGISTER_OPERAND_RANGE[1]
        expansion = wraps or operand_expansion
    strata = {"scenario": payload.get("scenario", "transition" if family in {"ledger", "register"} else "retention"),
              "generator_version": VERSION, "membership_partition": partition_of(key)}
    if family in {"ledger", "register"}:
        e = payload["event"]
        strata.update(event_kind=e["kind"], operand_sign="positive" if e.get("amount", e.get("value")) > 0 else "negative")
        if family == "register" and e["kind"] == "conditional":
            strata["conditional_branch"] = payload["before"][e["other"]] > e["threshold"]
        strata["support_domain"] = "expanded_operand" if expansion else "documented_operand_support"
        if family == "ledger":
            strata["balance_support"] = "historical_bound_unestablished_prospective_broad_integer_domain"
        else:
            strata["wrap"] = wraps
            strata["operand_support_expansion"] = operand_expansion
            strata["support_domain"] = ("expanded_operand_and_modulo_wrap" if wraps and operand_expansion else
                                        "modulo_wrap_outside_parentq_prototype" if wraps else
                                        "expanded_operand" if operand_expansion else "documented_operand_support_no_wrap")
    elif "scenario" in payload:
        strata.update(operand_sign=payload["latent_sign"], relevant_missing=payload["scenario"] in {"relevant_missing", "complete_cue_missing"},
                      irrelevant_missing=payload["irrelevant_field_missing"], relevant_conflict=payload["scenario"] in {"relevant_conflict", "complete_cue_conflict"},
                      misleading_completeness_cue=payload["scenario"].startswith(("complete_cue", "incomplete_cue")))
    row_id = world_id + "-" + style + ("-" + schema.digest(key)[:8] if pair_id else "")
    row = {"schema_version": schema.SCHEMA_VERSION, "row_id": row_id, "world_id": world_id,
           "pair_id": pair_id, "split": split, "family": family, "variant_id": style,
           "system": _system(family, style, contract), "user": "pending", "answerable": answer is not None,
           "expected": answer, "contract": contract, "payload": payload, "operative_key": key,
           "support_expansion": expansion, "strata": strata}
    row["user"] = render_prompt(row)
    schema.validate_row(row)
    oracles.verify_gold(row)
    return row


def _normalized_keys(keys):
    return {key if isinstance(key, str) else schema.canonical(key) for key in keys}


def _family_rows(family, count, split, difficulty="regular", coverage="canonical", exposed_keys=()):
    rng, seen, rows = _rng(split, family, difficulty), _normalized_keys(exposed_keys), []
    step = 2 if difficulty == "counterfactual" else 1
    if count % step:
        raise ValueError("STOP_INCOMPLETE_COUNTERFACTUAL_PAIR")
    for index in range(0, count, step):
        for attempt in range(MAX_ATTEMPTS_PER_WORLD):
            p = _payload(family, index, count, rng)
            if step == 2:
                if family in {"answerability", "grounded_qa"}:
                    p = _fact_payload(family, index, rng, scenario=2 * ((index // 2) % 4))
                payloads = _counterfactual(family, p, index)
            else:
                payloads = (p,)
            keys = [operative_key(family, payload) for payload in payloads]
            if len(set(keys)) != step or any(key in seen or partition_of(key) != split for key in keys):
                continue
            pair_id = "pair-" + schema.digest(sorted(keys))[:24] if step == 2 else None
            world_id = "world-" + schema.digest([VERSION, sorted(keys)])[:24] if step == 2 else None
            for offset, payload in enumerate(payloads):
                style = schema.STYLES[(index + offset) % 4] if coverage == "diverse" else "canonical"
                rows.append(_row(family, payload, split, style, pair_id, world_id))
            seen.update(keys)
            break
        else:
            raise ValueError(f"STOP_GENERATION_EXHAUSTED:{family}:{split}:index={index}")
    return rows


def build_split(split, exposed_keys=()):
    if split not in {"dev", "final"}:
        raise ValueError("STOP_EVALUATION_SPLIT_MUST_BE_DEV_OR_FINAL")
    count = schema.DEVELOPMENT_WORLDS_PER_FAMILY if split == "dev" else schema.FINAL_WORLDS_PER_FAMILY
    rows = []
    for family in schema.FAMILIES:
        canonical = _family_rows(family, count, split, exposed_keys=exposed_keys)
        rows.extend(canonical)
        if family in schema.REPAIR_FAMILIES:
            for index, parent in enumerate(_variant_parents(family, canonical, count // 4)):
                offset = schema.REPAIR_FAMILIES.index(family) * (count // 4)
                style = schema.STYLES[1 + (offset + index) % 3]
                rows.append(_row(family, parent["payload"], split, style, world_id=parent["world_id"]))
    expected = 512 if split == "dev" else 4096
    if len(rows) != expected or len({r["row_id"] for r in rows}) != expected:
        raise ValueError("STOP_EVALUATION_MEMBERSHIP_COUNT")
    return rows


def _variant_parents(family, canonical, count):
    """Fixed stratum quotas make attached variants cover failures, not just wins."""
    if family in {"answerability", "grounded_qa"}:
        quotas = [(scene, count // 8) for scene in SCENARIOS]
        key = lambda row: row["strata"]["scenario"]
    elif family == "register":
        quotas = [(("add", sign, None), count // 4) for sign in ("positive", "negative")]
        quotas += [(("conditional", sign, branch), count // 8)
                   for sign in ("positive", "negative") for branch in (True, False)]
        key = lambda row: (row["strata"]["event_kind"], row["strata"]["operand_sign"], row["strata"].get("conditional_branch"))
    else:
        base, remaining = divmod(count, 6)
        quotas = [((kind, sign), base + int(2 * i < remaining))
                  for i, kind in enumerate(("add", "scale", "transfer")) for sign in ("positive", "negative")]
        key = lambda row: (row["strata"]["event_kind"], row["strata"]["operand_sign"])
    selected = []
    for stratum, quota in quotas:
        members = [row for row in canonical if key(row) == stratum]
        if len(members) < quota:
            raise ValueError("STOP_ATTACHED_VARIANT_STRATUM_UNAVAILABLE")
        selected.extend(members[:quota])
    if len(selected) != count:
        raise ValueError("STOP_ATTACHED_VARIANT_QUOTA_COUNT")
    return selected


def _verify_replay(row):
    schema.validate_row(row)
    p = row.get("provenance", {})
    required = {"source_path", "source_sha256", "source_line", "source_id", "target_type"}
    if not required <= p.keys() or p["target_type"] != "HUMAN_OR_DETERMINISTIC_EXISTING_TRAIN_GOLD":
        raise ValueError("STOP_REPLAY_SOURCE_PROVENANCE")
    if row["family"] not in {"ledger", "register"} or not row.get("replay_target"):
        raise ValueError("STOP_REPLAY_REQUIRES_EXISTING_RETENTION_TARGET")
    if len(p["source_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in p["source_sha256"]):
        raise ValueError("STOP_REPLAY_SOURCE_DIGEST")
    if type(p["source_line"]) is not int or p["source_line"] < 1:
        raise ValueError("STOP_REPLAY_SOURCE_LINE")
    if p.get("gold_verified") is not True:
        raise ValueError("STOP_REPLAY_GOLD_NOT_INDEPENDENTLY_VERIFIED")
    oracles.verify_gold({**row, "is_replay": True})


def training_rows(recipe, seed, replay_rows):
    if seed not in schema.SEEDS:
        raise ValueError("STOP_UNREGISTERED_TRAINING_SEED")
    coverage, difficulty, replay_fraction = recipe["prompt_coverage"], recipe["difficulty"], recipe["replay_fraction"]
    if coverage not in {"canonical", "diverse"} or difficulty not in {"regular", "counterfactual"} or replay_fraction not in {0.0, 0.2}:
        raise ValueError("STOP_UNREGISTERED_RECIPE_FACTOR")
    replay_count = schema.REPLAY_EXAMPLES if replay_fraction == 0.2 else 0
    repair_count = schema.TRAIN_EXAMPLES - replay_count
    excluded = [row["operative_key"] for row in replay_rows]
    rows = []
    for family in schema.REPAIR_FAMILIES:
        rows.extend(_family_rows(family, repair_count // 4, "train", difficulty, coverage, excluded))
    if replay_count:
        if not replay_rows:
            raise ValueError("STOP_REPLAY_SOURCE_POOL_EMPTY")
        replay_families = ("ledger", "register")
        by_family = {family: [] for family in replay_families}
        for row in replay_rows:
            _verify_replay(row)
            by_family[row["family"]].append(row)
        if any(not group for group in by_family.values()):
            raise ValueError("STOP_REPLAY_MISSING_REGISTER_OR_LEDGER_FAMILY")
        for family in by_family:
            by_family[family].sort(key=lambda row: schema.canonical(row["provenance"]))
        source_counts = {}
        for index in range(replay_count):
            family = replay_families[index % 2]
            source = by_family[family][(index // 2) % len(by_family[family])]
            row = copy.deepcopy(source)
            source_key = schema.canonical(row["provenance"])
            occurrence = source_counts.get(source_key, 0) + 1
            source_counts[source_key] = occurrence
            row.update(split="replay", is_replay=True, row_id="replay-" + schema.digest([source_key, occurrence])[:24])
            row["strata"].update(replay_occurrence=occurrence, repeated_replay_source=occurrence > 1)
            rows.append(row)
    random.Random(seed).shuffle(rows)
    if len(rows) != schema.TRAIN_EXAMPLES or sum(bool(r.get("is_replay")) for r in rows) != replay_count:
        raise ValueError("STOP_TRAINING_REALIZED_MIXTURE")
    return rows


def target_reply(row):
    schema.validate_row(row)
    if row.get("is_replay"):
        _verify_replay(row)
        return row["replay_target"]
    oracles.verify_gold(row)
    if not row["answerable"]:
        return row["contract"]["abstention"]
    answer = row["expected"]
    return schema.canonical(answer) if isinstance(answer, dict) else str(answer)


target_text = target_reply
