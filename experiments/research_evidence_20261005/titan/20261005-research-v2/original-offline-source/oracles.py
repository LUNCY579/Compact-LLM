"""Independent deterministic gold computations. No model or historical bank reads.

The two implementations intentionally use different control/data structures. Gold
validation checks both, including supplied replay rows with registered provenance.
"""
from __future__ import annotations
import json
from .schema import typed_equal


def ledger(payload):
    state, event = dict(payload["before"]), payload["event"]
    if event["status"] != "posted":
        return state
    a, value = event["account"], event["amount"]
    if event["kind"] == "add":
        state[a] += value
    elif event["kind"] == "scale":
        state[a] *= value
    elif event["kind"] == "transfer":
        state[a] -= value
        state[event["other"]] += value
    else:
        raise ValueError("STOP_UNKNOWN_LEDGER_EVENT")
    return state


def ledger_check(payload):
    before, event = payload["before"], payload["event"]
    if event["status"] != "posted":
        return {key: val for key, val in before.items()}
    kind, target, operand = event["kind"], event["account"], event["amount"]
    if kind not in {"add", "scale", "transfer"}:
        raise ValueError("STOP_UNKNOWN_LEDGER_EVENT")
    out = {}
    for account, balance in before.items():
        if kind == "scale" and account == target:
            out[account] = balance * operand
        else:
            delta = operand if kind == "add" and account == target else 0
            if kind == "transfer":
                delta = (int(account == event["other"]) - int(account == target)) * operand
            out[account] = balance + delta
    return out


def register(payload):
    state, event = dict(payload["before"]), payload["event"]
    target, value = event["register"], event["value"]
    if event["kind"] == "conditional":
        if state[event["other"]] <= event["threshold"]:
            value = -value
    elif event["kind"] != "add":
        raise ValueError("STOP_UNKNOWN_REGISTER_EVENT")
    state[target] = (state[target] + value) % 97
    return state


def register_check(payload):
    before, event = payload["before"], payload["event"]
    if event["kind"] not in {"add", "conditional"}:
        raise ValueError("STOP_UNKNOWN_REGISTER_EVENT")
    multiplier = 1
    if event["kind"] == "conditional":
        multiplier = 2 * int(before[event["other"]] > event["threshold"]) - 1
    return {name: ((value + multiplier * event["value"]) % 97
                   if name == event["register"] else value)
            for name, value in before.items()}


def fact_answer(payload):
    question = payload["question"]
    values = [fact["value"] for fact in payload["facts"]
              if fact["subject"] == question["subject"] and fact["field"] == question["field"]]
    if not values:
        return None
    return values[0] if all(value == values[0] for value in values) else None


def fact_answer_check(payload):
    table = {}
    for fact in payload["facts"]:
        table.setdefault((fact["subject"], fact["field"]), set()).add(fact["value"])
    q = payload["question"]
    alternatives = table.get((q["subject"], q["field"]), set())
    return next(iter(alternatives)) if len(alternatives) == 1 else None


def arithmetic(payload):
    a, b, op = payload["a"], payload["b"], payload["operator"]
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    raise ValueError("STOP_UNKNOWN_ARITHMETIC_OPERATOR")


def arithmetic_check(payload):
    a, b = payload["a"], payload["b"]
    funcs = {"add": lambda: sum((a, b)), "subtract": lambda: sum((a, -b)),
             "multiply": lambda: a * b}
    if payload["operator"] not in funcs:
        raise ValueError("STOP_UNKNOWN_ARITHMETIC_OPERATOR")
    return funcs[payload["operator"]]()


def instruction(payload):
    return {key: payload["record"][key] for key in payload["fields"]}


def instruction_check(payload):
    wanted = frozenset(payload["fields"])
    return dict((key, value) for key, value in payload["record"].items() if key in wanted)


def concise(payload):
    value = fact_answer(payload)
    return None if value is None else f'{payload["question"]["subject"]} is at {value}.'


def concise_check(payload):
    value = fact_answer_check(payload)
    return None if value is None else "{} is at {}.".format(payload["question"]["subject"], value)


ORACLES = {"ledger": (ledger, ledger_check), "register": (register, register_check),
           "answerability": (fact_answer, fact_answer_check), "grounded_qa": (fact_answer, fact_answer_check),
           "arithmetic": (arithmetic, arithmetic_check), "instruction": (instruction, instruction_check),
           "concise_grounded": (concise, concise_check)}


def solve(family, payload):
    if family not in ORACLES:
        raise ValueError("STOP_UNKNOWN_ORACLE_FAMILY")
    first, second = ORACLES[family]
    answer = first(payload)
    if not typed_equal(answer, second(payload)):
        raise ValueError("STOP_INDEPENDENT_ORACLE_DISAGREEMENT")
    return answer


def verify_gold(row):
    # Existing human/deterministic replay gold has its own source verification and
    # cannot be silently regenerated by this new synthetic task oracle.
    if row.get("is_replay"):
        if not row.get("replay_target") or not row.get("provenance"):
            raise ValueError("STOP_REPLAY_WITHOUT_EXISTING_GOLD")
        answer = solve(row["family"], row["payload"])
        try:
            stored = json.loads(row["replay_target"])
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("STOP_REPLAY_GOLD_IS_NOT_JSON") from error
        if not typed_equal(stored, answer) or not typed_equal(answer, row["expected"]) or not row["answerable"]:
            raise ValueError("STOP_REPLAY_GOLD_OR_ANSWERABILITY_MISMATCH")
        return True
    answer = solve(row["family"], row["payload"])
    if not typed_equal(answer, row["expected"]) or (answer is not None) != row["answerable"]:
        raise ValueError("STOP_GOLD_OR_ANSWERABILITY_MISMATCH")
    return True
