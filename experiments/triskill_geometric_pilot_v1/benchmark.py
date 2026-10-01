"""Small, generated evaluation fixtures; never training data.

Only stdlib is used. No legacy banks are read, and no model is loaded. Signatures
prove DEVELOPMENT/SCREEN disjointness inside this file, not global exclusion
against protected, training, or spent banks. See BENCHMARK_MATRIX.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


MOD = 97
UNKNOWN = "UNKNOWN"
LEDGER_SYSTEM = (
    "Apply one event to the supplied account balances. If status is not posted, leave all "
    "balances unchanged. add: add amount to account. scale: multiply account by amount. "
    "transfer: subtract amount from account and add it to other, simultaneously. Return only "
    "the complete resulting balances as a JSON object."
)
REGISTER_SYSTEM = (
    "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap "
    "exchanges register and other simultaneously. add adds value to register. mix adds (2 "
    "times other plus value) to register. conditional adds value if other is greater than "
    "threshold, otherwise subtracts value. Reduce updated numeric registers modulo 97; "
    "preserve other registers. Return only the complete resulting registers as a JSON object."
)
ANSWERABILITY_SYSTEM = (
    "Use only the supplied complete notes. Return exactly one integer if the requested value "
    "is established. If it cannot be determined, return exactly UNKNOWN. Do not infer "
    "unstated facts."
)
TRANSFER_SYSTEM = (
    "Use only the complete records and rules. Registers are integers modulo 97 (0..96). "
    "Missing initial state admits all 97 values; conflicting records admit each listed value "
    "equally. Apply every instruction in order to each admissible value, reducing modulo 97 "
    "after each step. Return only the common final integer, or UNKNOWN if final values differ."
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def signature(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkRow:
    row_id: str
    split: str
    family: str
    task_kind: str
    system: str
    user: str
    expected: str
    output_kind: str
    world_signature: str
    state_signature: str
    rule_signature: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transition_oracle(family: str, payload: Mapping[str, Any]) -> dict[str, int]:
    """Original T contracts, independently implemented without reading bank rows."""
    result = dict(payload["before"])
    event = payload["event"]
    if family == "ledger_replay":
        if event.get("status") != "posted":
            return result
        register, amount = event["account"], event["amount"]
        if event["kind"] == "add":
            result[register] += amount
        elif event["kind"] == "scale":
            result[register] *= amount
        elif event["kind"] == "transfer":
            result[register] -= amount
            result[event["other"]] += amount
        else:
            raise ValueError("unknown ledger instruction")
        return result
    if family != "register_machine":
        raise ValueError("unsupported transition family")
    register, other = event["register"], event.get("other")
    kind, value = event["kind"], event.get("value", 0)
    if kind == "swap":
        result[register], result[other] = result[other], result[register]
    elif kind == "add":
        result[register] = (result[register] + value) % MOD
    elif kind == "mix":
        result[register] = (result[register] + 2 * result[other] + value) % MOD
    elif kind == "conditional":
        delta = value if result[other] > event["threshold"] else -value
        result[register] = (result[register] + delta) % MOD
    else:
        raise ValueError("unknown register instruction")
    return result


def affine_outcomes(initial: Sequence[int] | None,
                    rules: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Enumerate a finite domain, avoiding assumptions about missing state."""
    states = set(range(MOD)) if initial is None else set(initial)
    if not states or any(type(value) is not int or not 0 <= value < MOD for value in states):
        raise ValueError("initial states must be a nonempty subset of 0..96")
    for scale, offset in rules:
        if type(scale) is not int or type(offset) is not int:
            raise ValueError("rules require integer coefficients")
        states = {(scale * value + offset) % MOD for value in states}
    return tuple(sorted(states))


def _transition(split: str, family: str, payload: dict[str, Any], world: str) -> BenchmarkRow:
    system = LEDGER_SYSTEM if family == "ledger_replay" else REGISTER_SYSTEM
    operative = dict(payload["event"])
    operative.pop("tick", None)
    return BenchmarkRow(
        f"{split}-T-{family}", split, "state_tracking", "transition_parent", system,
        canonical(payload), canonical(transition_oracle(family, payload)), "json_object",
        signature(world), signature(payload["before"]), signature(operative),
        {"parent": "T", "native_contract": True, "transition_family": family,
         "membership": "generated_local_split_only", "training": False},
    )


def _answerability(split: str, world: str, known: bool, number: int) -> BenchmarkRow:
    value_note = f"The {world} seal has code {number}." if known else f"The {world} crate has code {number}."
    user = (
        f"Complete notes for {world}: {value_note} The seal is stored beside the crate. "
        f"What is the {world} seal's code?"
    )
    return BenchmarkRow(
        f"{split}-H-{'answerable' if known else 'missing'}", split, "answerability_abstention",
        "answerability_parent", ANSWERABILITY_SYSTEM, user, str(number) if known else UNKNOWN,
        "integer_or_unknown", signature(world), signature({"world": world, "notes": value_note}),
        signature({"world": world, "query": "seal_code", "complete_notes": True}),
        {"parent": "H", "native_contract": False, "format_modified_diagnostic": True,
         "answerable": known, "membership": "generated_local_split_only", "training": False},
    )


def _transfer(split: str, world: str, initial: Sequence[int] | None,
              rules: Sequence[tuple[int, int]], task: str) -> BenchmarkRow:
    names = [f"{world.upper()}_{index + 1}" for index in range(len(rules))]
    if initial is None:
        first_note = f"Record A: the initial {world} register value is not recorded."
        last_note = f"Record B: a separate marker has value 23; it supplies no initial {world} value."
    else:
        first_note = f"Record A: the initial {world} register value is {initial[0]}."
        last_note = f"Record B: the initial {world} register value is {initial[-1]}."
    definitions = "\n".join(
        f"{name}(x) = ({scale} * x + ({offset})) modulo 97."
        for name, (scale, offset) in zip(names, rules)
    )
    sequence = ", then ".join(names)
    user = (
        f"Complete records for {world}:\n{first_note}\nRule definitions:\n{definitions}\n"
        f"Instruction log: apply {sequence} to the {world} register.\n{last_note}\n"
        f"What is the final {world} register value?"
    )
    outcomes = affine_outcomes(initial, rules)
    expected = str(outcomes[0]) if len(outcomes) == 1 else UNKNOWN
    known_control = task == "known_chain"
    return BenchmarkRow(
        f"{split}-C-{task}", split, "state_tracking" if known_control else "skill_composition",
        "known_state_control" if known_control else "third_skill_transfer", TRANSFER_SYSTEM,
        user, expected, "integer_or_unknown", signature(world),
        signature({"register": world, "initial": initial}), signature(list(rules)),
        {"parent": None, "new_task_structure": not known_control,
         "transfer_endpoint": not known_control, "initial_missing": initial is None,
         "initial_conflict": initial is not None and len(set(initial)) > 1,
         "reset_present": any(scale % MOD == 0 for scale, _ in rules),
         "outcome_count": len(outcomes), "rules": [list(rule) for rule in rules],
         "initial": None if initial is None else list(initial), "answerable": len(outcomes) == 1,
         "membership": "generated_local_split_only", "training": False},
    )


def _build(split: str) -> tuple[BenchmarkRow, ...]:
    if split == "development":
        ledger = {"before": {"A": 14, "B": -27, "C": 8, "D": 31},
                  "event": {"account": "C", "amount": -9, "kind": "transfer", "other": "A",
                            "status": "posted", "tick": 41}}
        register = {"before": {"X": 19, "Y": 62, "Z": 8},
                    "event": {"kind": "mix", "register": "Y", "other": "X", "value": -7}}
        transfers = (
            ("amber", None, ((3, 1), (0, 4), (2, -2)), "missing_erased"),
            ("brass", (1, 4), ((2, -3), (0, 5), (-1, 2)), "conflict_erased"),
            ("cobalt", None, ((3, 2), (1, -1)), "missing_survives"),
            ("dusk", (96, 3), ((1, 6), (2, -4)), "conflict_survives"),
        )
        h_worlds, h_numbers = ("Birch", "Cedar"), (6, 28)
    elif split == "screen":
        ledger = {"before": {"A": -39, "B": 17, "C": 46, "D": -11},
                  "event": {"account": "B", "amount": -13, "kind": "transfer", "other": "D",
                            "status": "posted", "tick": 73}}
        register = {"before": {"X": 71, "Y": 23, "Z": 44},
                    "event": {"kind": "conditional", "register": "X", "other": "Z",
                              "value": 16, "threshold": 52}}
        transfers = (
            ("ember", None, ((2, -1), (0, 7), (-1, 3)), "missing_erased"),
            ("flint", (95, 95), ((-1, 4), (3, -6), (2, 1)), "known_chain"),
            ("garnet", None, ((2, 3), (-1, 5)), "missing_survives"),
            ("hazel", (2, 94), ((2, 1), (-1, 4), (1, -2)), "conflict_survives"),
        )
        h_worlds, h_numbers = ("Juniper", "Maple"), (9, 42)
    else:
        raise ValueError("split must be development or screen")
    return (
        _transition(split, "ledger_replay", ledger, f"{split}-ledger"),
        _transition(split, "register_machine", register, f"{split}-register"),
        _answerability(split, h_worlds[0], True, h_numbers[0]),
        _answerability(split, h_worlds[1], False, h_numbers[1]),
        *(_transfer(split, *specification) for specification in transfers),
    )


def build_development() -> tuple[BenchmarkRow, ...]:
    """Scorer/mechanics fixtures. No optimizer, gradient, or third-skill training use."""
    return _build("development")


def build_screen() -> tuple[BenchmarkRow, ...]:
    """Exactly eight frozen, generated screening rows, not protected confirmation."""
    return _build("screen")


def build_conjunction_controls() -> tuple[BenchmarkRow, ...]:
    """Optional DEVELOPMENT-only tasks; not part of the native eight-row budget.

    The missing fact has no dependency on the state calculation. This is ordinary
    conjunction and must not be scored as transfer to the new task structure.
    """
    rows = []
    for world, initial, scale, offset in (("ivory", 8, 3, 4), ("jasper", 17, -1, 6)):
        final = (scale * initial + offset) % MOD
        user = (
            f"Complete notes: the initial {world} register is {initial}. Apply x := "
            f"({scale} * x + ({offset})) modulo 97. The witness's code is not recorded, "
            "and has no effect on this register. Return only a JSON object with keys state "
            "(the final integer) and fact (the witness's code, or UNKNOWN if absent)."
        )
        rows.append(BenchmarkRow(
            f"development-conjunction-{world}", "development", "skill_composition",
            "parent_task_conjunction", ANSWERABILITY_SYSTEM.replace(
                "Return exactly one integer if the requested value is established. If it cannot be "
                "determined, return exactly UNKNOWN.", "Follow the requested JSON output schema."),
            user, canonical({"state": final, "fact": UNKNOWN}), "json_object", signature(world),
            signature({"register": world, "initial": initial}), signature((scale, offset)),
            {"new_task_structure": False, "transfer_endpoint": False,
             "training": False, "membership": "generated_local_split_only"},
        ))
    return tuple(rows)


def validate_local_membership() -> dict[str, Any]:
    development, screen = build_development(), build_screen()
    checks = {}
    for field in ("row_id", "world_signature", "state_signature", "rule_signature"):
        a = {getattr(row, field) for row in development}
        b = {getattr(row, field) for row in screen}
        checks[field] = not (a & b)
    return {"local_disjoint": all(checks.values()), "checks": checks,
            "legacy_exclusion_verified": False, "protected_confirmation": False,
            "development_n": len(development), "screen_n": len(screen)}


def _atom(text: str, strict: bool) -> str | None:
    text = text.strip()
    if text == UNKNOWN or (not strict and text.upper() == UNKNOWN):
        return UNKNOWN
    expression = r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)" if strict else r"[+-]?[0-9]+"
    if re.fullmatch(expression, text):
        return str(int(text))
    return None


def _same_typed_json(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_typed_json(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_json(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def _same_schema(actual: Any, expected: Any) -> bool:
    """Full output keys and exact JSON value types, without requiring right values."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_schema(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_schema(a, e) for a, e in zip(actual, expected)
        )
    return True


def _json_load(text: str) -> Any:
    """Reject duplicate keys and nonstandard NaN/Infinity accepted by json.loads."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def _semantic_text(text: str) -> str:
    """Unwrap only a complete fenced answer or complete labeled answer.

    Prefix prose is not discarded: it might contain a competing answer. This
    diagnostic deliberately rejects reasoning plus a final-answer line.
    """
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json|text)?\s*\n?(.*?)\n?```", stripped, re.I | re.S)
    if fence:
        return fence.group(1).strip()
    labels = list(re.finditer(r"(?:^|\n)\s*(?:final\s+)?(?:answer|output|value)\s*[:=]\s*", stripped, re.I))
    if len(labels) == 1 and not stripped[:labels[0].start()].strip():
        suffix = stripped[labels[0].end():].strip()
        fenced_suffix = re.fullmatch(r"```(?:json|text)?\s*\n?(.*?)\n?```", suffix, re.I | re.S)
        return fenced_suffix.group(1).strip() if fenced_suffix else suffix
    return stripped


def score_response(row: BenchmarkRow, response: str) -> dict[str, Any]:
    """Strict correctness is primary. Wrapper semantics is a separate diagnostic."""
    if not isinstance(response, str):
        raise TypeError("response must be text")
    strict_value, semantic_value = None, None
    if row.output_kind == "json_object":
        expected = _json_load(row.expected)
        try:
            strict_value = _json_load(response.strip())
        except (ValueError, TypeError):
            pass
        try:
            semantic_value = _json_load(_semantic_text(response))
        except (ValueError, TypeError):
            pass
        strict_correct = _same_typed_json(strict_value, expected)
        semantic_correct = _same_typed_json(semantic_value, expected)
        strict_valid = _same_schema(strict_value, expected)
    elif row.output_kind == "integer_or_unknown":
        strict_value = _atom(response, strict=True)
        semantic_text = _semantic_text(response)
        semantic_value = _atom(semantic_text, strict=False)
        if semantic_value is None:
            try:
                wrapped = _json_load(semantic_text)
                if isinstance(wrapped, dict) and set(wrapped) == {"answer"}:
                    value = wrapped["answer"]
                    if type(value) is int or isinstance(value, str):
                        semantic_value = _atom(str(value), strict=False)
            except (ValueError, TypeError):
                pass
        strict_correct = strict_value == row.expected
        semantic_correct = semantic_value == row.expected
        strict_valid = strict_value is not None
    else:
        raise ValueError("unsupported output schema")
    return {"row_id": row.row_id, "strict_valid": strict_valid,
            "strict_correct": strict_correct, "semantic_correct": semantic_correct,
            "strict_value": strict_value, "semantic_value": semantic_value,
            "expected": row.expected}


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Descriptive 95% binomial interval; eight rows cannot confirm a capability."""
    if total < 0 or not 0 <= correct <= total:
        raise ValueError("invalid count")
    if total == 0:
        return None
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def compare_outputs(rows: Sequence[BenchmarkRow], source: Mapping[str, str],
                    candidate: Mapping[str, str]) -> dict[str, Any]:
    """Paired damage and gain; absent calls stay unmeasured, empty text fails.

    A caller records a measured failed/invalid response as an explicit empty
    string. Missing map keys never produce comparator gains or losses.
    """
    row_ids = [row.row_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row IDs")
    records, grouped, families, transitions, answerability = [], {}, {}, {}, {}
    for row in rows:
        left = score_response(row, source.get(row.row_id, ""))
        right = score_response(row, candidate.get(row.row_id, ""))
        source_measured, candidate_measured = row.row_id in source, row.row_id in candidate
        paired = source_measured and candidate_measured
        gain = paired and not left["strict_correct"] and right["strict_correct"]
        damage = paired and left["strict_correct"] and not right["strict_correct"]
        record = {"row_id": row.row_id, "family": row.family, "task_kind": row.task_kind,
                  "source_correct": left["strict_correct"] if source_measured else None,
                  "candidate_correct": right["strict_correct"] if candidate_measured else None,
                  "source_semantic_correct": left["semantic_correct"] if source_measured else None,
                  "candidate_semantic_correct": right["semantic_correct"] if candidate_measured else None,
                  "source_strict_valid": left["strict_valid"] if source_measured else None,
                  "candidate_strict_valid": right["strict_valid"] if candidate_measured else None,
                  "strict_format_damage": paired and left["strict_valid"] and not right["strict_valid"],
                  "source_correct_format_damage": paired and left["strict_correct"] and not right["strict_valid"],
                  "gain": gain, "source_correct_damage": damage,
                  "source_missing": not source_measured, "candidate_missing": not candidate_measured,
                  "paired_measured": paired}
        if row.output_kind == "integer_or_unknown":
            expected_unknown = row.expected == UNKNOWN
            record["answerable"] = not expected_unknown
            record["candidate_false_abstention"] = (candidate_measured and not expected_unknown
                                                      and right["semantic_value"] == UNKNOWN)
            record["candidate_unsupported_answer"] = (candidate_measured and expected_unknown and right["semantic_value"] is not None
                                                       and right["semantic_value"] != UNKNOWN)
            answerability.setdefault("unanswerable" if expected_unknown else "answerable", []).append(record)
        records.append(record)
        grouped.setdefault(row.task_kind, []).append(record)
        families.setdefault(row.family, []).append(record)
        if "transition_family" in row.metadata:
            transitions.setdefault(row.metadata["transition_family"], []).append(record)

    def summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        source_n = sum(not item["source_missing"] for item in items)
        candidate_n = sum(not item["candidate_missing"] for item in items)
        paired_n = sum(item["paired_measured"] for item in items)
        source_correct = sum(item["source_correct"] is True for item in items)
        candidate_correct = sum(item["candidate_correct"] is True for item in items)
        paired_source_correct = sum(item["source_correct"] is True and item["paired_measured"] for item in items)
        gains = sum(item["gain"] for item in items)
        damage = sum(item["source_correct_damage"] for item in items)
        discordant = gains + damage
        mcnemar_p = (min(1.0, 2 * sum(math.comb(discordant, index) for index in
                       range(min(gains, damage) + 1)) / (2 ** discordant)) if discordant else 1.0)
        return {"n": n, "source_measured_n": source_n, "candidate_measured_n": candidate_n,
                "paired_measured_n": paired_n, "source_correct": source_correct,
                "candidate_correct": candidate_correct, "paired_source_correct": paired_source_correct,
                "gains": gains, "source_correct_damage": damage, "net_gain": gains - damage,
                "source_correct_damage_rate": damage / paired_source_correct if paired_source_correct else None,
                "source_correct_damage_interval_95": wilson_interval(damage, paired_source_correct),
                "candidate_accuracy_interval_95": wilson_interval(candidate_correct, candidate_n),
                "candidate_false_abstentions": sum(item.get("candidate_false_abstention", False) for item in items),
                "candidate_unsupported_answers": sum(item.get("candidate_unsupported_answer", False) for item in items),
                "source_strict_valid_n": sum(item["source_strict_valid"] is True for item in items),
                "candidate_strict_valid_n": sum(item["candidate_strict_valid"] is True for item in items),
                "strict_format_damage": sum(item["strict_format_damage"] for item in items),
                "source_correct_format_damage": sum(item["source_correct_format_damage"] for item in items),
                "exact_paired_mcnemar_p_two_sided": mcnemar_p if paired_n else None}

    return {"overall": summary(records), "by_task_kind": {key: summary(value) for key, value in grouped.items()},
            "by_family": {key: summary(value) for key, value in families.items()},
            "by_transition_family": {key: summary(value) for key, value in transitions.items()},
            "by_answerability": {key: summary(value) for key, value in answerability.items()},
            "rows": records, "uncertainty": "Descriptive only; tiny generated screen, no confirmation.",
            "membership": validate_local_membership()}


if __name__ == "__main__":
    print(json.dumps({"membership": validate_local_membership(),
                      "screen": [row.to_dict() for row in build_screen()]}, indent=2))
