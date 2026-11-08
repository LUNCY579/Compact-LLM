"""Small, generated evaluation fixtures; never training data.

Only stdlib is used. No legacy banks are read, and no model is loaded. Signatures
prove DEVELOPMENT/SCREEN disjointness inside this file, not global exclusion
against protected, training, or spent banks. See BENCHMARK_MATRIX.md.

Section H amendment (AMENDMENT_SECTIONH.md): `build_screen()` and
`build_development()` now return the rebuilt rows. T rows sit inside T's trained
operand support and carry `tick`; H and chain rows use the programme's registered
answerability contract and are credited by the published boundary rule; chain rows
balance answerable and abstention golds and include disguised resets. The original
eight-row screen stays reproducible through `build_legacy_screen()`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
ATOM_OUTPUT_KINDS = ("integer_or_unknown", "answer_json", "answer_tag", "fenced_atom")
FORMAT_CONTRACTS = {
    "integer_or_unknown": (
        "Return only one integer in decimal notation, or exactly UNKNOWN. "
        "No labels, prose, tags, JSON, or code fences."
    ),
    "answer_json": (
        'Return only a JSON object with exactly one key, "answer". Its value must be '
        'an integer, or the string "UNKNOWN". No other keys, prose, or code fences.'
    ),
    "answer_tag": (
        "Return only <answer> followed by one integer in decimal notation or exactly "
        "UNKNOWN, followed by </answer>. No other tags, prose, JSON, or code fences."
    ),
    "fenced_atom": (
        "Return only a three-line text code fence: the first line is ```text, the "
        "second line is one integer in decimal notation or exactly UNKNOWN, and the "
        "third line is ```. No labels, prose, tags, or JSON."
    ),
}

# Section H contracts. ABSTAIN_PHRASE and ANSWERABILITY_SYSTEM_H are verbatim from the
# HSKILL answerable rows that measured H (HSKILL_VIEWS_V2/SOUP_V3); a CPU test checks it.
ABSTAIN_PHRASE = "I don't know based on the available information."
ABSTAIN_CONTRACT = f'If one answer is not supported, reply exactly: "{ABSTAIN_PHRASE}"'
ANSWERABILITY_SYSTEM_H = (
    "Answer concisely using only the information stated in the prompt. Reason internally "
    "and return only the short final answer. " + ABSTAIN_CONTRACT
)
TRANSFER_SYSTEM_H = (
    "Use only the complete records and rules. Registers are integers modulo 97 (0..96). "
    "Missing initial state admits all 97 values; conflicting records admit each listed value "
    "equally. Apply every instruction in order to each admissible value, reducing modulo 97 "
    "after each step. Reason internally and return only the common final integer. "
    + ABSTAIN_CONTRACT
)
ANSWER_OR_ABSTAIN = "answer_or_abstain"
# T's trained and measured support (T_BANK_V3X/V4 and delta_T training): ledger amounts
# -9..12 over accounts A-D; register values -8..11 over X/Y/Z, values 0..96; every event
# carries the keys below, including tick.
T_SUPPORT = {
    "ledger_replay": {"accounts": ("A", "B", "C", "D"), "amount": (-9, 12), "tick": (1, 99),
                      "event_keys": ("account", "amount", "kind", "other", "status", "tick")},
    "register_machine": {"registers": ("X", "Y", "Z"), "value": (-8, 11), "tick": (1, 99),
                         "state": (0, 96), "threshold": (0, 96),
                         "event_keys": ("kind", "other", "register", "threshold", "tick", "value")},
}
# Published verbatim in rineya-dataset-retention-records/hskill-views-002/boundary_rule.py
# (sha256 below). Copied so the native run reads no external bank; a CPU test checks equality.
BOUNDARY_RULE_SHA256 = "416cfb6aec8aa28240752d3db2fc1b8af85115c4962a873582765c009adc3bf0"
ABSTAIN_PATTERNS = (
    r"\?",
    r"\bi (?:can ?not|cannot|can't)\b",
    r"\bi (?:do not|don't) (?:have|know|see)\b",
    r"\b(?:unable|not able) to\b",
    r"\bneed (?:more|the|additional|to know)\b",
    r"\b(?:not|isn't|is not) (?:clear|certain|sure|enough|possible|safe)\b",
    r"\b(?:possibly|unclear|uncertain|may be|might be|it depends)\b",
    r"\b(?:ask|consult|contact|call|seek|check with) (?:a|an|your|the|emergency|911)\b",
    r"\b(?:do not|don't) guess\b",
    r"\bno[,.]? ",
    r"\bcannot (?:help|provide|verify|diagnose|disclose|assist)\b",
    r"\bplease (?:provide|share|upload|clarify|specify)\b",
)
_ABSTAIN = tuple(re.compile(pattern) for pattern in ABSTAIN_PATTERNS)
# Registered extensions (section H): non-determination wordings H's own training prompts
# request ("say that the available information is insufficient"), the base model's literal
# UNKNOWN and common phrasings of "the final values differ". Applied to the same normalized
# first line as the published patterns. On abstention-gold rows a first line that also
# contains an integer is never credited (a hedged number is still an answer).
EXTRA_ABSTAIN_PATTERNS = (
    r"\binsufficient\b",
    r"\b(?:cannot|can ?not|can't) (?:be )?determined?\b",
    r"\b(?:not determined|undetermined|underdetermined|indeterminate)\b",
    r"\bunknown\b",
    r"\bvalues? (?:differ|disagree)\b",
    r"\bmultiple (?:final |possible )?values\b",
)
_EXTRA_ABSTAIN = tuple(re.compile(pattern) for pattern in EXTRA_ABSTAIN_PATTERNS)
_INTEGER = re.compile(r"[0-9]")

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


def _build_legacy(split: str) -> tuple[BenchmarkRow, ...]:
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


def build_legacy_development() -> tuple[BenchmarkRow, ...]:
    """The original TRISKILL-GEO-001 development fixtures (format-variant sources)."""
    return _build_legacy("development")


def build_legacy_screen() -> tuple[BenchmarkRow, ...]:
    """The original eight-row screen, kept reproducible; superseded by section H."""
    return _build_legacy("screen")


def t_support_violations(row: BenchmarkRow) -> list[str]:
    """Where a T row leaves T's trained/measured operand support (empty = inside).

    Register rows must also not need a modulo-97 wrap: T_BANK_V3X has no wrapping add row
    and TT scores 0/3 on wrapping conditionals (0/5 for every arm on V4 wrapping adds).
    """
    family = row.metadata["transition_family"]
    support, payload = T_SUPPORT[family], json.loads(row.user)
    event, before = payload["event"], payload["before"]
    problems = []
    if tuple(sorted(event)) != support["event_keys"]:
        problems.append(f"event keys {sorted(event)}")
    if not support["tick"][0] <= event.get("tick", -1) <= support["tick"][1]:
        problems.append("tick")
    if family == "ledger_replay":
        if tuple(sorted(before)) != support["accounts"]:
            problems.append("accounts")
        if not support["amount"][0] <= event["amount"] <= support["amount"][1]:
            problems.append("amount")
    else:
        if tuple(sorted(before)) != support["registers"]:
            problems.append("registers")
        if not support["value"][0] <= event["value"] <= support["value"][1]:
            problems.append("value")
        if not all(support["state"][0] <= value <= support["state"][1] for value in before.values()):
            problems.append("state")
        if not support["threshold"][0] <= event["threshold"] <= support["threshold"][1]:
            problems.append("threshold")
        if event["kind"] in ("add", "conditional", "mix"):
            register, value = before[event["register"]], event["value"]
            if event["kind"] == "add":
                raw = register + value
            elif event["kind"] == "mix":
                raw = register + 2 * before[event["other"]] + value
            else:
                raw = register + (value if before[event["other"]] > event["threshold"] else -value)
            if not 0 <= raw < MOD:
                problems.append("modulo wrap")
    return problems


def _transition_h(split: str, index: int, family: str, payload: dict[str, Any]) -> BenchmarkRow:
    system = LEDGER_SYSTEM if family == "ledger_replay" else REGISTER_SYSTEM
    operative = dict(payload["event"])
    operative.pop("tick", None)
    world = f"h-{split}-T{index}-{family}"
    return BenchmarkRow(
        world, split, "state_tracking", "transition_parent", system,
        canonical(payload), canonical(transition_oracle(family, payload)), "json_object",
        signature(world), signature(payload["before"]), signature(operative),
        {"parent": "T", "native_contract": True, "transition_family": family,
         "event_kind": payload["event"]["kind"], "in_trained_support": True,
         "membership": "generated_local_split_only", "training": False, "amendment": "section_h"},
    )


def _answerability_h(split: str, world: str, kind: str, number: int) -> BenchmarkRow:
    """kind: 'answerable', 'distractor' (only the crate's code is given) or 'cue'
    (the notes state the seal's code was not recorded, the cue H was trained on)."""
    if kind == "answerable":
        value_note = f"The {world} seal has code {number}."
    elif kind == "distractor":
        value_note = f"The {world} crate has code {number}."
    elif kind == "cue":
        value_note = f"The {world} crate has code {number}. The seal's code was not recorded."
    else:
        raise ValueError("unknown answerability row kind")
    user = f"Notes for {world}: {value_note} The seal is stored beside the crate. What is the {world} seal's code?"
    known = kind == "answerable"
    return BenchmarkRow(
        f"h-{split}-H-{world.lower()}", split, "answerability_abstention", "answerability_parent",
        ANSWERABILITY_SYSTEM_H, user, str(number) if known else UNKNOWN, ANSWER_OR_ABSTAIN,
        signature(world), signature({"world": world, "notes": value_note}),
        signature({"world": world, "query": "seal_code"}),
        {"parent": "H", "native_contract": True, "contract": "hskill_answerable_v2", "h_row_kind": kind,
         "answerable": known, "strict_reply": str(number) if known else ABSTAIN_PHRASE,
         "membership": "generated_local_split_only", "training": False, "amendment": "section_h"},
    )


def chain_step(value: int, rule: Sequence[Any]) -> int:
    """One chain rule: affine (a, b) -> a*x + b, or ("if_above", t, v): T's conditional
    update applied to the chain register itself: x + v if x > t, otherwise x - v."""
    if len(rule) == 3 and rule[0] == "if_above":
        _, threshold, delta = rule
        return (value + delta) % MOD if value > threshold else (value - delta) % MOD
    scale, offset = rule
    return (scale * value + offset) % MOD


def chain_outcomes(initial: Sequence[int] | None, rules: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    """Enumerate every admissible start through the chain (finite domain, no assumptions)."""
    states = set(range(MOD)) if initial is None else set(initial)
    if not states or any(type(value) is not int or not 0 <= value < MOD for value in states):
        raise ValueError("initial states must be a nonempty subset of 0..96")
    for rule in rules:
        if not (len(rule) == 2 and all(type(item) is int for item in rule)) and not (
                len(rule) == 3 and rule[0] == "if_above" and all(type(item) is int for item in rule[1:])):
            raise ValueError(f"unsupported chain rule {rule!r}")
        states = {chain_step(value, rule) for value in states}
    return tuple(sorted(states))


def _rule_text(name: str, rule: Sequence[Any]) -> str:
    if len(rule) == 3:
        _, threshold, delta = rule
        return (f"{name}(x) = (x + ({delta})) modulo 97 if x is greater than {threshold}; "
                f"otherwise (x - ({delta})) modulo 97.")
    return f"{name}(x) = ({rule[0]} * x + ({rule[1]})) modulo 97."


def _transfer_h(split: str, world: str, initial: Sequence[int] | None,
                rules: Sequence[Sequence[Any]], known_control: bool = False) -> BenchmarkRow:
    names = [f"{world.upper()}_{index + 1}" for index in range(len(rules))]
    if initial is None:
        first_note = f"Record A: the initial {world} register value is not recorded."
        last_note = f"Record B: a separate marker has value 23; it supplies no initial {world} value."
    else:
        first_note = f"Record A: the initial {world} register value is {initial[0]}."
        last_note = f"Record B: the initial {world} register value is {initial[-1]}."
    definitions = "\n".join(_rule_text(name, rule) for name, rule in zip(names, rules))
    user = (
        f"Records for {world}:\n{first_note}\nRule definitions:\n{definitions}\n"
        f"Instruction log: apply {', then '.join(names)} to the {world} register.\n{last_note}\n"
        f"What is the final {world} register value?"
    )
    outcomes = chain_outcomes(initial, rules)
    answerable = len(outcomes) == 1
    expected = str(outcomes[0]) if answerable else UNKNOWN
    if known_control and (initial is None or len(set(initial)) != 1):
        raise ValueError("known-state controls need one recorded initial value")
    affine = [rule for rule in rules if len(rule) == 2]
    reset = any(scale % MOD == 0 for scale, _ in affine)
    return BenchmarkRow(
        f"h-{split}-C-{world}", split, "state_tracking" if known_control else "skill_composition",
        "known_state_control" if known_control else "third_skill_transfer", TRANSFER_SYSTEM_H,
        user, expected, ANSWER_OR_ABSTAIN, signature(world),
        signature({"register": world, "initial": initial}), signature([list(rule) for rule in rules]),
        {"parent": None, "new_task_structure": not known_control,
         "transfer_endpoint": not known_control, "initial_missing": initial is None,
         "initial_conflict": initial is not None and len(set(initial)) > 1,
         "reset_present": reset,
         "reset_visible": any(scale == 0 for scale, _ in affine),
         "reset_disguised": any(scale != 0 and scale % MOD == 0 for scale, _ in affine),
         "decoy_multiplier": any(scale not in (1, -1) and scale % MOD in (1, MOD - 1) for scale, _ in affine),
         "conditional_present": any(len(rule) == 3 for rule in rules),
         "collision_answerable": answerable and not reset and not known_control,
         "outcome_count": len(outcomes), "rules": [list(rule) for rule in rules],
         "initial": None if initial is None else list(initial), "answerable": answerable,
         "strict_reply": expected if answerable else ABSTAIN_PHRASE,
         "membership": "generated_local_split_only", "training": False, "amendment": "section_h"},
    )


# Hand-chosen literals; no RNG. Chain rows: 4 answerable (a visible 0*x reset, a disguised
# 291*x reset, and two conflicts merged by T-style conditional collisions) and 4 abstention
# rows (decoys 195 = 1 and 96 = -1 mod 97, and two non-colliding conditionals), balanced
# missing/conflicting starts, plus 2 known-state controls (one through a conditional).
_SECTION_H = {
    "development": {
        "t": (("ledger_replay", {"before": {"A": 12, "B": 30, "C": -8, "D": 17},
                                 "event": {"account": "D", "amount": 5, "kind": "add", "other": "B",
                                           "status": "posted", "tick": 26}}),
              ("register_machine", {"before": {"X": 9, "Y": 64, "Z": 31},
                                    "event": {"kind": "swap", "other": "Z", "register": "X",
                                              "threshold": 20, "tick": 71, "value": 4}})),
        "h": (("Alder", "answerable", 31), ("Bramble", "cue", 8)),
        "chains": (("acorn", None, ((2, 3), (0, 15), (3, -1)), False),
                   ("bloom", (81, 25), (("if_above", 60, -28), (2, 1)), False),
                   ("cinder", None, ((-1, 7), ("if_above", 30, 5)), False),
                   ("dune", (14, 45), (("if_above", 20, 6), (98, 5)), False)),
    },
    "screen": {
        "t": (("ledger_replay", {"before": {"A": 23, "B": -14, "C": 37, "D": 5},
                                 "event": {"account": "C", "amount": 8, "kind": "transfer", "other": "A",
                                           "status": "posted", "tick": 58}}),
              ("ledger_replay", {"before": {"A": -6, "B": 19, "C": 11, "D": -27},
                                 "event": {"account": "B", "amount": 3, "kind": "scale", "other": "D",
                                           "status": "posted", "tick": 17}}),
              ("register_machine", {"before": {"X": 40, "Y": 61, "Z": 13},
                                    "event": {"kind": "add", "other": "Z", "register": "Y",
                                              "threshold": 61, "tick": 34, "value": 11}}),
              ("register_machine", {"before": {"X": 52, "Y": 7, "Z": 70},
                                    "event": {"kind": "conditional", "other": "Z", "register": "X",
                                              "threshold": 45, "tick": 82, "value": -6}})),
        "h": (("Larch", "answerable", 14), ("Myrtle", "answerable", 63),
              ("Nutmeg", "distractor", 27), ("Osier", "cue", 50)),
        "chains": (("onyx", None, ((3, 5), (0, 11), (2, -4)), False),
                   ("pyrite", None, ((2, 7), (291, 30), (-1, 9)), False),
                   ("quartz", None, ((3, -2), (195, 6)), False),
                   ("rowan", None, (("if_above", 40, 9), (2, 1)), False),
                   ("sorrel", (70, 30), (("if_above", 50, -20), (3, 4)), False),
                   ("tansy", (40, 16), ((2, 1), ("if_above", 60, -24), (1, 5)), False),
                   ("umber", (8, 33), ((96, 4), (2, -1)), False),
                   ("vetch", (40, 75), (("if_above", 50, 10), (2, 3)), False),
                   ("willow", (44, 44), ((2, 9), (-1, 3)), True),
                   ("yarrow", (15, 15), (("if_above", 10, 7), (3, -2)), True)),
    },
}


def _build_section_h(split: str) -> tuple[BenchmarkRow, ...]:
    spec = _SECTION_H[split]
    return (
        *(_transition_h(split, index + 1, family, payload) for index, (family, payload) in enumerate(spec["t"])),
        *(_answerability_h(split, world, kind, number) for world, kind, number in spec["h"]),
        *(_transfer_h(split, world, initial, rules, known) for world, initial, rules, known in spec["chains"]),
    )


def build_development() -> tuple[BenchmarkRow, ...]:
    """Section H scorer/mechanics fixtures. No optimizer, gradient, or third-skill training use."""
    return _build_section_h("development")


def build_screen() -> tuple[BenchmarkRow, ...]:
    """Eighteen frozen generated section H rows: 4 T, 4 H, 8 dependent chains, 2 known controls."""
    return _build_section_h("screen")


def build_format_variants() -> tuple[BenchmarkRow, ...]:
    """DEVELOPMENT-only format fixtures, outside the frozen native screen.

    Four contracts share each existing H/C semantic case and gold atom. These
    are correlated variants, not 24 independent ability observations. No native
    format panel is authorized or measured; no operator, layer, angle, or stopping
    choice may use their C outcomes. Native use requires a separate protocol.
    """
    rows = []
    for source in build_legacy_development():
        if source.output_kind != "integer_or_unknown":
            continue
        if source.task_kind == "answerability_parent":
            old_contract = (
                "Return exactly one integer if the requested value is established. "
                "If it cannot be determined, return exactly UNKNOWN."
            )
            semantic_contract = (
                "The answer is the established requested integer; if the requested "
                "value cannot be determined, the answer is UNKNOWN. "
            )
        else:
            old_contract = (
                "Return only the common final integer, or UNKNOWN if final values differ."
            )
            semantic_contract = (
                "The answer is the common final integer if all admissible final values "
                "agree; otherwise the answer is UNKNOWN. "
            )
        if source.system.count(old_contract) != 1:
            raise ValueError("format source has an unexpected output contract")
        for output_kind in ATOM_OUTPUT_KINDS:
            contract = FORMAT_CONTRACTS[output_kind]
            rows.append(replace(
                source, row_id=f"{source.row_id}-format-{output_kind}",
                system=source.system.replace(old_contract, semantic_contract + contract),
                output_kind=output_kind,
                metadata={**source.metadata,
                          "semantic_case_id": source.row_id,
                          "format_source_row_id": source.row_id,
                          "format_variant": output_kind,
                          "strict_format_contract": contract,
                          "format_panel_scoring_unit": "semantic_case_cluster",
                          "native_format_panel": "unmeasured_not_authorized",
                          "operator_selection_allowed": False},
            ))
    return tuple(rows)


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
    """Section H screen is disjoint from its development rows and from every legacy row.

    T-row signatures exclude tick, so equal operative events cannot hide behind a new tick.
    """
    development, screen = build_development(), build_screen()
    legacy = (*build_legacy_development(), *build_legacy_screen(), *build_conjunction_controls())
    checks, legacy_checks = {}, {}
    for field in ("row_id", "world_signature", "state_signature", "rule_signature"):
        a = {getattr(row, field) for row in development}
        b = {getattr(row, field) for row in screen}
        c = {getattr(row, field) for row in legacy}
        checks[field] = not (a & b)
        legacy_checks[field] = not ((a | b) & c)
    return {"local_disjoint": all(checks.values()), "checks": checks,
            "legacy_rows_disjoint": all(legacy_checks.values()), "legacy_checks": legacy_checks,
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
    tagged = re.fullmatch(r"<answer>(.*?)</answer>", stripped, re.S)
    if tagged:
        return tagged.group(1).strip()
    fence = re.fullmatch(r"```(?:json|text)?\s*\n?(.*?)\n?```", stripped, re.I | re.S)
    if fence:
        return fence.group(1).strip()
    labels = list(re.finditer(r"(?:^|\n)\s*(?:final\s+)?(?:answer|output|value)\s*[:=]\s*", stripped, re.I))
    if len(labels) == 1 and not stripped[:labels[0].start()].strip():
        suffix = stripped[labels[0].end():].strip()
        fenced_suffix = re.fullmatch(r"```(?:json|text)?\s*\n?(.*?)\n?```", suffix, re.I | re.S)
        return fenced_suffix.group(1).strip() if fenced_suffix else suffix
    return stripped


def _strict_answer_atom(output_kind: str, response: str) -> str | None:
    """Validate a whole-answer contract, without accepting competing claims."""
    stripped = response.strip()
    if output_kind == "integer_or_unknown":
        return _atom(stripped, strict=True)
    if output_kind == "answer_json":
        try:
            wrapped = _json_load(stripped)
        except (ValueError, TypeError):
            return None
        if type(wrapped) is dict and set(wrapped) == {"answer"}:
            value = wrapped["answer"]
            if type(value) is int:
                return str(value)
            if value == UNKNOWN and type(value) is str:
                return UNKNOWN
        return None
    if output_kind == "answer_tag":
        wrapped = re.fullmatch(r"<answer>(.*?)</answer>", stripped, re.S)
    elif output_kind == "fenced_atom":
        wrapped = re.fullmatch(r"```text\r?\n([^\r\n]*)\r?\n```", stripped)
    else:
        raise ValueError("unsupported atom output schema")
    return _atom(wrapped.group(1), strict=True) if wrapped else None


def boundary_canonical(text: str) -> str:
    """boundary_rule.canonical: lowercase, collapse whitespace, strip trailing . ! ?"""
    return re.sub(r"\s+", " ", str(text).strip().lower()).rstrip(".!?").strip()


def first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


def _normalize_apostrophes(text: str) -> str:
    return str(text).replace("\u2019", "'").replace("\u2018", "'")


def credited_abstention(line: str) -> bool:
    """boundary_rule.abstains on the first line, plus the registered extensions.

    Typographic apostrophes are normalized first (an extension; boundary_rule does not).
    The constant-abstainer floor (screen_floors) is scored by this same predicate.
    """
    normalized = re.sub(r"\s+", " ", _normalize_apostrophes(line).strip().lower())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in (*_ABSTAIN, *_EXTRA_ABSTAIN))


def _credited_answer_or_abstain(row: BenchmarkRow, text: str) -> tuple[bool, bool]:
    line = first_line(text)
    abstained = credited_abstention(line)
    if row.expected == UNKNOWN:
        return abstained and not _INTEGER.search(line), abstained
    return (not abstained) and boundary_canonical(line) == row.expected, abstained


def _strip_one_period(text: str) -> str:
    stripped = text.strip()
    return stripped[:-1].rstrip() if stripped.endswith(".") else stripped


def score_response(row: BenchmarkRow, response: str) -> dict[str, Any]:
    """Score semantic correctness separately from row-specific strict formatting.

    A semantic wrapper must contain the whole answer; prefix prose, duplicate
    keys, additional answers, and mixed claims are never silently discarded.
    `credited` is the registered primary endpoint: for T rows exact typed JSON after
    stripping whitespace and at most one trailing period; for answer_or_abstain rows the
    published boundary rule (abstention) or canonical exact match (answers).
    """
    if not isinstance(response, str):
        raise TypeError("response must be text")
    strict_value, semantic_value = None, None
    if row.output_kind == ANSWER_OR_ABSTAIN:
        credited, abstained = _credited_answer_or_abstain(row, response)
        semantic_text = _semantic_text(response)
        semantic_correct, semantic_abstained = _credited_answer_or_abstain(row, semantic_text)
        stripped = response.strip()
        strict_valid = stripped == ABSTAIN_PHRASE or _atom(stripped, strict=True) not in (None, UNKNOWN)
        strict_value = (UNKNOWN if stripped == ABSTAIN_PHRASE else _atom(stripped, strict=True)) if strict_valid else None
        semantic_value = UNKNOWN if semantic_abstained else _atom(boundary_canonical(first_line(semantic_text)), strict=False)
        return {"row_id": row.row_id, "strict_valid": strict_valid,
                "strict_correct": stripped == row.metadata["strict_reply"],
                "semantic_correct": semantic_correct, "credited": credited, "abstained": abstained,
                "strict_value": strict_value, "semantic_value": semantic_value,
                "expected": row.expected}
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
        try:
            credited = _same_typed_json(_json_load(_strip_one_period(response)), expected)
        except (ValueError, TypeError):
            credited = False
    elif row.output_kind in ATOM_OUTPUT_KINDS:
        strict_value = _strict_answer_atom(row.output_kind, response)
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
        credited = strict_correct  # legacy contract rows: strict was the registered primary
    else:
        raise ValueError("unsupported output schema")
    return {"row_id": row.row_id, "strict_valid": strict_valid,
            "strict_correct": strict_correct, "semantic_correct": semantic_correct,
            "credited": credited, "strict_value": strict_value, "semantic_value": semantic_value,
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


def _valid_count(value: int) -> bool:
    return type(value) is int and value >= 0


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Stdlib binomial CDF using log probabilities, never 2**n division."""
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_factorial_n = math.lgamma(n + 1)
    logs = [log_factorial_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * log_p + (n - i) * log_q for i in range(k + 1)]
    largest = max(logs)
    result = math.exp(largest) * math.fsum(math.exp(value - largest) for value in logs)
    return min(1.0, max(0.0, result))


def exact_paired_mcnemar(gains: int, losses: int) -> float:
    """Two-sided exact conditional binomial McNemar p-value.

    Conditional on discordance, the null assigns gain/loss probability 1/2.
    No discordance returns 1; absence of measured pairs is handled by the caller.
    Three gains and zero losses gives 0.25, not confirmation.
    """
    if not _valid_count(gains) or not _valid_count(losses):
        raise ValueError("invalid paired counts")
    discordant = gains + losses
    return min(1.0, 2 * _binomial_cdf(min(gains, losses), discordant, 0.5)) if discordant else 1.0


def clopper_pearson_interval(correct: int, total: int,
                             alpha: float = 0.05) -> list[float] | None:
    """Exact binomial interval by 64-step monotone tail inversion.

    Coverage concerns independent samples from a common Bernoulli distribution;
    for our fixed generated fixtures this is a descriptive uncertainty diagnostic.
    """
    if not _valid_count(correct) or not _valid_count(total) or correct > total:
        raise ValueError("invalid binomial count")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if total == 0:
        return None
    tail = alpha / 2
    lower, upper = 0.0, 1.0
    if correct:
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            # P_p(X >= correct) = P_(1-p)(X <= total-correct).
            if _binomial_cdf(total - correct, total, 1 - mid) < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if correct < total:
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if _binomial_cdf(correct, total, mid) > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return [lower, upper]


def paired_net_rate_interval(gains: int, losses: int, total: int,
                              alpha: float = 0.05) -> dict[str, Any]:
    """Conservative simultaneous binomial bounds for p_gain - p_loss.

    Each marginal Clopper-Pearson interval has confidence 1-alpha/2. Bonferroni
    gives joint coverage >=1-alpha under iid paired-case sampling, regardless
    of dependence between gains and losses. Difference bounds are deliberately
    broad. Correlated format variants require semantic-case cluster analysis;
    a fixed tiny generated screen provides no population or retention claim.
    """
    if any(not _valid_count(value) for value in (gains, losses, total)) or gains + losses > total:
        raise ValueError("invalid paired counts")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    gain_interval = clopper_pearson_interval(gains, total, alpha / 2)
    loss_interval = clopper_pearson_interval(losses, total, alpha / 2)
    net_interval = ([max(-1.0, gain_interval[0] - loss_interval[1]),
                     min(1.0, gain_interval[1] - loss_interval[0])]
                    if total else None)
    return {"net_rate": (gains - losses) / total if total else None,
            "interval": net_interval,
            "gain_probability_interval": gain_interval,
            "loss_probability_interval": loss_interval,
            "confidence_level_under_iid": 1 - alpha,
            "method": "Bonferroni simultaneous Clopper-Pearson marginal difference bounds",
            "interpretation": "descriptive for fixed fixtures; coverage requires iid paired cases"}


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
        semantic_gain = paired and not left["semantic_correct"] and right["semantic_correct"]
        semantic_damage = paired and left["semantic_correct"] and not right["semantic_correct"]
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
                  "semantic_gain": semantic_gain, "source_semantic_correct_damage": semantic_damage,
                  "format_repair_only": gain and left["semantic_correct"] and right["semantic_correct"],
                  "source_credited": left["credited"] if source_measured else None,
                  "candidate_credited": right["credited"] if candidate_measured else None,
                  "source_missing": not source_measured, "candidate_missing": not candidate_measured,
                  "paired_measured": paired}
        if row.output_kind == ANSWER_OR_ABSTAIN:
            expected_unknown = row.expected == UNKNOWN
            record["answerable"] = not expected_unknown
            record["candidate_false_abstention"] = candidate_measured and not expected_unknown and right["abstained"]
            record["candidate_unsupported_answer"] = (candidate_measured and expected_unknown and not right["abstained"]
                                                       and right["semantic_value"] is not None)
            answerability.setdefault("unanswerable" if expected_unknown else "answerable", []).append(record)
        elif row.output_kind in ATOM_OUTPUT_KINDS:
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
        def paired_endpoint(source_key: str, candidate_key: str) -> dict[str, Any]:
            table = {f"n{left}{right}": sum(
                item["paired_measured"] and item[source_key] is bool(left)
                and item[candidate_key] is bool(right) for item in items)
                for left in (0, 1) for right in (0, 1)}
            endpoint_gains, endpoint_losses = table["n01"], table["n10"]
            uncertainty = paired_net_rate_interval(endpoint_gains, endpoint_losses, paired_n)
            return {**table, "gains": endpoint_gains, "losses": endpoint_losses,
                    "net_gain": endpoint_gains - endpoint_losses,
                    "paired_net_rate": uncertainty["net_rate"],
                    "paired_net_rate_interval_95": uncertainty["interval"],
                    "paired_net_rate_uncertainty": uncertainty,
                    "exact_mcnemar_p_two_sided": exact_paired_mcnemar(endpoint_gains, endpoint_losses)
                    if paired_n else None}

        strict_paired = paired_endpoint("source_correct", "candidate_correct")
        semantic_paired = paired_endpoint("source_semantic_correct", "candidate_semantic_correct")
        credited_paired = paired_endpoint("source_credited", "candidate_credited")
        return {"n": n, "source_measured_n": source_n, "candidate_measured_n": candidate_n,
                "credited_paired": credited_paired,
                "paired_measured_n": paired_n, "source_correct": source_correct,
                "candidate_correct": candidate_correct, "paired_source_correct": paired_source_correct,
                "gains": gains, "source_correct_damage": damage, "net_gain": gains - damage,
                "paired_net_rate": strict_paired["paired_net_rate"],
                "strict_paired": strict_paired, "semantic_paired": semantic_paired,
                "source_semantic_correct": sum(item["source_semantic_correct"] is True for item in items),
                "candidate_semantic_correct": sum(item["candidate_semantic_correct"] is True for item in items),
                "semantic_gains": semantic_paired["gains"],
                "source_semantic_correct_damage": semantic_paired["losses"],
                "semantic_net_gain": semantic_paired["net_gain"],
                "semantic_paired_net_rate": semantic_paired["paired_net_rate"],
                "format_repair_only": sum(item["format_repair_only"] for item in items),
                "source_correct_damage_rate": damage / paired_source_correct if paired_source_correct else None,
                "source_correct_damage_interval_95": wilson_interval(damage, paired_source_correct),
                "candidate_accuracy_interval_95": wilson_interval(candidate_correct, candidate_n),
                "candidate_false_abstentions": sum(item.get("candidate_false_abstention", False) for item in items),
                "candidate_unsupported_answers": sum(item.get("candidate_unsupported_answer", False) for item in items),
                "source_strict_valid_n": sum(item["source_strict_valid"] is True for item in items),
                "candidate_strict_valid_n": sum(item["candidate_strict_valid"] is True for item in items),
                "strict_format_damage": sum(item["strict_format_damage"] for item in items),
                "source_correct_format_damage": sum(item["source_correct_format_damage"] for item in items),
                "exact_paired_mcnemar_p_two_sided": strict_paired["exact_mcnemar_p_two_sided"],
                "semantic_exact_paired_mcnemar_p_two_sided": semantic_paired["exact_mcnemar_p_two_sided"],
                "accuracy_damage_interval_interpretation": "descriptive Wilson intervals, not paired net-gain intervals"}

    return {"overall": summary(records), "by_task_kind": {key: summary(value) for key, value in grouped.items()},
            "by_family": {key: summary(value) for key, value in families.items()},
            "by_transition_family": {key: summary(value) for key, value in transitions.items()},
            "by_answerability": {key: summary(value) for key, value in answerability.items()},
            "rows": records, "uncertainty": "Descriptive only; tiny generated screen, no confirmation.",
            "membership": validate_local_membership()}


SCREEN_ARMS = ("BASE", "T", "H", "TH", "IDENTITY", "ROTATION", "REVERSE", "RANDOM", "PROMPT")
CANDIDATE_ARM = "ROTATION"
J_COMPARATORS = ("BASE", "T", "H", "TH", "IDENTITY", "RANDOM", "PROMPT", "REVERSE")


def row_groups(rows: Sequence[BenchmarkRow]) -> dict[str, list[BenchmarkRow]]:
    kinds = {"T": "transition_parent", "H": "answerability_parent",
             "J": "third_skill_transfer", "K": "known_state_control"}
    return {group: [row for row in rows if row.task_kind == kind] for group, kind in kinds.items()}


def required_coverage(rows: Sequence[BenchmarkRow]) -> dict[str, list[str]]:
    """Which rows each arm must have generated before a verdict beyond a stop exists."""
    groups = row_groups(rows)
    every = [row.row_id for row in rows]
    chains = [row.row_id for row in (*groups["J"], *groups["K"])]
    coverage = {arm: list(every) for arm in SCREEN_ARMS}
    coverage["PROMPT"] = chains
    return coverage


def parent_gate(rows: Sequence[BenchmarkRow], scores: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Registered parent transport gate on the 4 T and 4 H rows (credited endpoint).

    T passes with >=2 of 4 credited and more than BASE. H passes with more credited H rows
    than BASE and >=1 credited abstention on a missing-value row. When BASE already scores
    every row of a parent's group the parent cannot be shown to transport on this screen;
    that is a separate stop (`STOP_SCREEN_PARENT_CEILING`), not evidence of a missing parent.
    Parent damage on BASE-credited rows is reported, not gating.
    """
    groups = row_groups(rows)
    def credited(arm: str, group: str) -> int:
        return sum(bool(scores[arm][row.row_id]["credited"]) for row in groups[group])
    def lost(arm: str, group: str) -> list[str]:
        return [row.row_id for row in groups[group]
                if scores["BASE"][row.row_id]["credited"] and not scores[arm][row.row_id]["credited"]]
    t_n, t_base = credited("T", "T"), credited("BASE", "T")
    h_n, h_base = credited("H", "H"), credited("BASE", "H")
    h_abstain = sum(bool(scores["H"][row.row_id]["credited"]) for row in groups["H"] if row.expected == UNKNOWN)
    t_pass, h_pass = t_n >= 2 and t_n > t_base, h_n > h_base and h_abstain >= 1
    stop = None
    for parent, passed, base_n, group in (("T", t_pass, t_base, "T"), ("H", h_pass, h_base, "H")):
        if not passed and stop is None:
            ceiling = base_n == len(groups[group])
            stop = f"STOP_SCREEN_PARENT_CEILING:{parent}" if ceiling else f"STOP_SCREEN_NO_PARENT_SIGNAL:{parent}"
    return {"T": {"credited": t_n, "base_credited": t_base, "n": len(groups["T"]), "passed": t_pass,
                  "base_correct_lost": lost("T", "T")},
            "H": {"credited": h_n, "base_credited": h_base, "n": len(groups["H"]), "passed": h_pass,
                  "missing_row_abstentions": h_abstain, "base_correct_lost": lost("H", "H")},
            "passed": t_pass and h_pass, "stop": stop}


def _reset_scan_value(row: BenchmarkRow, recognize) -> str:
    """Answer as if every affine rule whose scale `recognize` accepts were a reset (and
    conditionals were ignored for answerability); abstain when no such rule exists."""
    rules = row.metadata["rules"]
    last = max((index for index, rule in enumerate(rules) if len(rule) == 2 and recognize(rule[0])), default=None)
    if last is None:
        return ABSTAIN_PHRASE
    value = rules[last][1] % MOD
    for rule in rules[last + 1:]:
        value = chain_step(value, rule)
    return str(value)


def screen_floors(rows: Sequence[BenchmarkRow]) -> dict[str, dict[str, int]]:
    """Credited scores of shortcut and constant policies on dependent (J) and control (K) rows.

    Reset scans read answerability off the multipliers alone and push the believed reset
    constant through the remaining rules with correct arithmetic, so they are upper bounds
    for readers that recognize those surface forms. `affine_reset_oracle` recognizes every
    multiple of 97 correctly but ignores conditional collisions.
    """
    groups = row_groups(rows)
    def value_from(initial: Sequence[int] | None, rules: Sequence[Sequence[Any]]) -> str:
        outcomes = chain_outcomes(initial, rules)
        return str(outcomes[0]) if len(outcomes) == 1 else ABSTAIN_PHRASE
    def oracle(row: BenchmarkRow) -> str:
        return row.metadata["strict_reply"]
    policies = {
        "always_abstain": lambda row: ABSTAIN_PHRASE,
        "copy_marker_23": lambda row: "23",
        "visible_zero_cue": lambda row: _reset_scan_value(row, lambda scale: scale == 0),
        "literal_0_or_97_cue": lambda row: _reset_scan_value(row, lambda scale: scale in (0, 97)),
        "magnitude_ge_97_cue": lambda row: _reset_scan_value(row, lambda scale: scale == 0 or abs(scale) >= 97),
        "affine_reset_oracle": lambda row: _reset_scan_value(row, lambda scale: scale % MOD == 0),
        "first_record_only": lambda row: (ABSTAIN_PHRASE if row.metadata["initial_missing"]
                                          else value_from(row.metadata["initial"][:1], row.metadata["rules"])),
        "abstain_missing_solve_conflict": lambda row: ABSTAIN_PHRASE if row.metadata["initial_missing"] else oracle(row),
        "copy_last_reset_offset": lambda row: next(
            (str(rule[1] % MOD) for rule in reversed(row.metadata["rules"]) if len(rule) == 2 and rule[0] % MOD == 0),
            ABSTAIN_PHRASE),
        "oracle": oracle,
    }
    result = {}
    for name, policy in policies.items():
        result[name] = {group: sum(score_response(row, policy(row))["credited"] for row in groups[group])
                        for group in ("J", "K")}
    best_constant = max(range(MOD), key=lambda value: sum(
        score_response(row, str(value))["credited"] for row in groups["J"]))
    result["best_constant_integer"] = {
        "value": best_constant,
        **{group: sum(score_response(row, str(best_constant))["credited"] for row in groups[group])
           for group in ("J", "K")}}
    return result


def decide_screen(rows: Sequence[BenchmarkRow], outputs: Mapping[str, Mapping[str, str]],
                  mechanics: Mapping[str, Any], stopped_by: str | None = None) -> dict[str, Any]:
    """The single registered screening verdict (AMENDMENT_SECTIONH.md section 6).

    Ladder in execution order, first match wins. A mechanics value of None means the
    stage was never reached (`STOPPED_BEFORE:<stage>`, with `stopped_by`), never a failure.
      1. padding: False -> STOP_NATIVE_MECHANICS_PADDING; None -> STOPPED_BEFORE:padding
      2. the parent gate (needs BASE/T/H parent rows; else STOPPED_BEFORE:parent_gate)
      3. liveness: False -> STOP_MECHANICS_HOOK_LIVENESS; None -> STOPPED_BEFORE:liveness
      4. the identity canary as soon as TH and IDENTITY share rows
      5. missing coverage -> INCOMPLETE_COVERAGE
      6. UNDETECTABLE_ON_J: ROTATION text equals TH on every dependent (J) row
      7. SCREEN_SIGNAL_EXPLORATORY if S1-S6 all hold, otherwise NO_SIGNAL.
    No verdict is confirmation, and NO_SIGNAL is not evidence of absence (low power).
    """
    by_id = {row.row_id: row for row in rows}
    clean = {arm: {row_id: text for row_id, text in arm_outputs.items() if row_id in by_id}
             for arm, arm_outputs in outputs.items()}
    unknown = sorted({row_id for arm_outputs in outputs.values() for row_id in arm_outputs if row_id not in by_id})
    scores = {arm: {row_id: score_response(by_id[row_id], text) for row_id, text in arm_outputs.items()}
              for arm, arm_outputs in clean.items()}
    groups = row_groups(rows)
    j_ids = [row.row_id for row in groups["J"]]
    verdict: dict[str, Any] = {"confirmation": False, "floors": screen_floors(rows), "reasons": [],
                               "stopped_by": stopped_by, "ignored_unknown_row_ids": unknown}
    def j_pairs():
        rot, th = scores.get(CANDIDATE_ARM, {}), scores.get("TH", {})
        paired = [row_id for row_id in j_ids if row_id in rot and row_id in th]
        gains = [row_id for row_id in paired if rot[row_id]["credited"] and not th[row_id]["credited"]]
        losses = [row_id for row_id in paired if th[row_id]["credited"] and not rot[row_id]["credited"]]
        return paired, gains, losses
    def finish(status: str) -> dict[str, Any]:
        paired, gains, losses = j_pairs()
        verdict["status"] = status
        verdict["rotation_vs_TH_on_J"] = {"paired": len(paired), "gains": len(gains), "losses": len(losses),
                                          "exact_mcnemar_p_two_sided": exact_paired_mcnemar(len(gains), len(losses))
                                          if paired else None}
        return verdict
    for stage, key, failure in (("padding", "padding_parity_ok", "STOP_NATIVE_MECHANICS_PADDING"),):
        if mechanics.get(key) is False:
            return finish(failure)
        if mechanics.get(key) is not True:
            return finish(f"STOPPED_BEFORE:{stage}")
    parents = {row.row_id for group in ("T", "H") for row in groups[group]}
    if any(not parents <= set(scores.get(arm, {})) for arm in ("BASE", "T", "H")):
        return finish("STOPPED_BEFORE:parent_gate")
    gate = parent_gate(rows, scores)
    verdict["parent_gate"] = gate
    if not gate["passed"]:
        return finish(gate["stop"])
    if mechanics.get("liveness_ok") is False:
        return finish("STOP_MECHANICS_HOOK_LIVENESS")
    if mechanics.get("liveness_ok") is not True:
        return finish("STOPPED_BEFORE:liveness")
    shared = set(clean.get("TH", {})) & set(clean.get("IDENTITY", {}))
    if any(clean["IDENTITY"][row_id] != clean["TH"][row_id] for row_id in shared):
        return finish("STOP_IDENTITY_CANARY")
    missing = {arm: sorted(set(ids) - set(clean.get(arm, {})))
               for arm, ids in required_coverage(rows).items()}
    missing = {arm: ids for arm, ids in missing.items() if ids}
    if missing:
        verdict["missing_coverage"] = missing
        return finish("INCOMPLETE_COVERAGE")
    verdict["j_credited"] = {arm: sum(bool(scores[arm][row_id]["credited"]) for row_id in j_ids)
                             for arm in SCREEN_ARMS}
    verdict["texts_differ_from_TH"] = {
        arm: {"J": sum(clean[arm][row_id] != clean["TH"][row_id] for row_id in j_ids),
              "all": sum(clean[arm][row_id] != clean["TH"][row_id] for row_id in clean[arm])}
        for arm in SCREEN_ARMS}
    if verdict["texts_differ_from_TH"][CANDIDATE_ARM]["J"] == 0:
        verdict["reasons"].append("ROTATION changed no greedy token on any dependent row")
        return finish("UNDETECTABLE_ON_J")
    rot, th = scores[CANDIDATE_ARM], scores["TH"]
    _, gains, losses = j_pairs()
    # A counted gain must change the decision or value, not the format (TH semantically
    # wrong too), and must not merely restore a parent's own ability (T or H credited).
    format_repairs = [row_id for row_id in gains if th[row_id]["semantic_correct"]]
    parent_repairs = [row_id for row_id in gains if row_id not in format_repairs
                      and (scores["T"][row_id]["credited"] or scores["H"][row_id]["credited"])]
    signal_gains = [row_id for row_id in gains if row_id not in format_repairs and row_id not in parent_repairs]
    answerable_signal_gains = [row_id for row_id in signal_gains if by_id[row_id].expected != UNKNOWN]
    answerable_losses = [row_id for row_id in losses if by_id[row_id].expected != UNKNOWN]
    retention_ids = [row.row_id for group in ("T", "H", "K") for row in groups[group]]
    retention_losses = [row_id for row_id in retention_ids if th[row_id]["credited"] and not rot[row_id]["credited"]]
    best_comparator = max(verdict["j_credited"][arm] for arm in J_COMPARATORS)
    abstain_floor = verdict["floors"]["always_abstain"]["J"]
    conditions = {
        "S1_above_every_comparator": verdict["j_credited"][CANDIDATE_ARM] > best_comparator,
        "S2_net_signal_gain_vs_TH": len(signal_gains) >= 1 and len(signal_gains) > len(losses),
        "S3_no_answerable_chain_loss_vs_TH": not answerable_losses,
        "S4_no_parent_or_known_control_loss_vs_TH": not retention_losses,
        "S5_gains_a_computed_answer": len(answerable_signal_gains) >= 1,
        "S6_above_always_abstain_floor": verdict["j_credited"][CANDIDATE_ARM] > abstain_floor,
    }
    verdict.update({"conditions": conditions, "vs_TH": {
        "gains": gains, "signal_gains": signal_gains, "format_repairs": format_repairs,
        "parent_repairs": parent_repairs, "answerable_signal_gains": answerable_signal_gains,
        "losses": losses, "answerable_losses": answerable_losses, "retention_losses": retention_losses},
        "best_comparator_j_credited": best_comparator,
        "reverse_sign_specific": verdict["j_credited"][CANDIDATE_ARM] > verdict["j_credited"]["REVERSE"]})
    verdict["reasons"] = [name for name, held in conditions.items() if not held]
    return finish("SCREEN_SIGNAL_EXPLORATORY" if all(conditions.values()) else "NO_SIGNAL")


if __name__ == "__main__":
    print(json.dumps({"membership": validate_local_membership(), "floors": screen_floors(build_screen()),
                      "screen": [row.to_dict() for row in build_screen()]}, indent=2))
