"""Candidate output contracts for a successor multi-reasoning instrument.

Registered prospectively. MRS-1's frozen JSON contract is carried unchanged as the
C0 control; every other contract is a new candidate. Contract selection is decided
on FORMAT COMPLIANCE ONLY. Answer correctness is recorded for description and is
explicitly forbidden as a selection input (see PROBE_PREREG.md).
"""
from __future__ import annotations
import json
import re

ANSWERS = ("0", "1", "2", "3", "UNKNOWN")

# MRS-1 frozen system prompt, reproduced verbatim as the control.
MRS1_SYSTEM = ('Solve only from the supplied problem. Return one JSON object with "answer" '
               'equal to a string 0, 1, 2, 3, or UNKNOWN if the facts do not determine a unique answer. '
               'You may add a short "state" string (at most 20 words). Do not use outside knowledge.')

_CLOSED = ('Answer only from the supplied problem. The answer is 0, 1, 2, or 3, '
           'or UNKNOWN if the facts do not determine a unique answer. Do not use outside knowledge.')


def _strict_json(text, role):
    data = json.loads(text.strip())
    allowed = {"answer", "state", "sufficient_evidence", "candidate_valid", "issues"}
    if not isinstance(data, dict) or not set(data) <= allowed or data.get("answer") not in ANSWERS:
        raise ValueError("invalid answer schema")
    required = {"R2": "sufficient_evidence", "R3": "candidate_valid"}.get(role)
    if required and required not in data:
        raise ValueError("checker field absent")
    if required and type(data[required]) is not bool:
        raise ValueError("checker flag must be boolean")
    return data


def _bare(text, role):
    token = text.strip()
    if token not in ANSWERS:
        raise ValueError("not a bare in-vocabulary answer")
    return {"answer": token}


_LINE = re.compile(r"\AANSWER:[ ]?(0|1|2|3|UNKNOWN)\Z")
_LINE_FLAG = re.compile(r"\AANSWER:[ ]?(0|1|2|3|UNKNOWN)\n(SUFFICIENT|VALID):[ ]?(YES|NO)\Z")


def _line(text, role):
    body = text.strip()
    if role in ("R2", "R3"):
        m = _LINE_FLAG.match(body)
        if not m:
            raise ValueError("two-line checker form not matched")
        field = "sufficient_evidence" if role == "R2" else "candidate_valid"
        expect = "SUFFICIENT" if role == "R2" else "VALID"
        if m.group(2) != expect:
            raise ValueError("wrong checker field name")
        return {"answer": m.group(1), field: m.group(3) == "YES"}
    m = _LINE.match(body)
    if not m:
        raise ValueError("single-line form not matched")
    return {"answer": m.group(1)}


_TAG = re.compile(r"\A<answer>(0|1|2|3|UNKNOWN)</answer>\Z")


def _tag(text, role):
    m = _TAG.match(text.strip())
    if not m:
        raise ValueError("tag form not matched")
    return {"answer": m.group(1)}


# A prefilled assistant turn is a deployable, label-blind decoding procedure: the
# same fixed bytes are prepended for every item, model and arm.
CONTRACTS = {
    "C0_JSON_MRS1": {
        "system": MRS1_SYSTEM,
        "prefill": "",
        "parse": _strict_json,
        "role_suffix": {"R2": ' Also return boolean "sufficient_evidence".',
                        "R3": ' Also return boolean "candidate_valid".'},
        "note": "MRS-1 frozen contract, carried verbatim as the control.",
    },
    "C1_BARE": {
        "system": _CLOSED + " Reply with the answer only: 0, 1, 2, 3, or UNKNOWN. No other text.",
        "prefill": "",
        "parse": _bare,
        "role_suffix": {"R2": "", "R3": ""},
        "note": "Bare token. Cannot carry a checker flag; checker roles collapse to their answer.",
    },
    "C2_LINE": {
        "system": _CLOSED + ' Reply in exactly this form and nothing else:\nANSWER: <0|1|2|3|UNKNOWN>',
        "prefill": "",
        "parse": _line,
        "role_suffix": {
            "R2": "\n\nReply in exactly this form and nothing else:\nANSWER: <0|1|2|3|UNKNOWN>\nSUFFICIENT: <YES|NO>",
            "R3": "\n\nReply in exactly this form and nothing else:\nANSWER: <0|1|2|3|UNKNOWN>\nVALID: <YES|NO>"},
        "note": "Line form; carries one checker boolean.",
    },
    "C3_JSON_PREFILL": {
        "system": MRS1_SYSTEM,
        "prefill": '{"answer": "',
        "parse": _strict_json,
        "role_suffix": {"R2": ' Also return boolean "sufficient_evidence".',
                        "R3": ' Also return boolean "candidate_valid".'},
        "note": "MRS-1 JSON schema with a fixed prefilled assistant opening.",
    },
    "C4_LINE_PREFILL": {
        "system": _CLOSED + ' Reply in exactly this form and nothing else:\nANSWER: <0|1|2|3|UNKNOWN>',
        "prefill": "ANSWER:",
        "parse": _line,
        "role_suffix": {
            "R2": "\n\nReply in exactly this form and nothing else:\nANSWER: <0|1|2|3|UNKNOWN>\nSUFFICIENT: <YES|NO>",
            "R3": "\n\nReply in exactly this form and nothing else:\nANSWER: <0|1|2|3|UNKNOWN>\nVALID: <YES|NO>"},
        "note": "Line form with a fixed prefilled assistant opening.",
    },
}

# Role instructions reproduced from the MRS-1 frozen prompt set so the probe measures
# the OUTPUT CONTRACT, not a change of reasoning instruction.
ROLES = {
    "R0": "Solve the problem normally. Give the final answer.",
    "R2": ("Audit the premises independently: identify every fact needed for the query, "
           "including missing initial values, missing dependencies, and superseding records. "
           "If multiple answers remain possible, answer UNKNOWN."),
    "R3": ("Check mechanical consistency independently. Build a compact table of intermediate "
           "states or earliest legal start times; check each transition, prerequisite, "
           "counterfactual deletion and latest-record rule. Recompute the answer."),
}


def parse(contract_id, text, role, terminated):
    spec = CONTRACTS[contract_id]
    result = {"parsed": {}, "format_failure": False, "terminated": terminated,
              "nontermination": not terminated}
    try:
        if not terminated:
            raise ValueError("nontermination")
        result["parsed"] = spec["parse"](text, role)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        result.update(format_failure=True, parse_error=str(exc))
    return result


def messages(contract_id, role, problem):
    spec = CONTRACTS[contract_id]
    content = ROLES[role] + spec["role_suffix"].get(role, "") + "\n\nPROBLEM\n" + problem
    return [{"role": "system", "content": spec["system"]}, {"role": "user", "content": content}]
