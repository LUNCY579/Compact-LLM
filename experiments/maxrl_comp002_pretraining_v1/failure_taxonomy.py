"""Diagnostic strict-failure subtypes and RS response classes (Amendment 1 §8.4 c-d).

Descriptive only: nothing here changes the strict endpoint (``corpus.reward``)
or the bounded content endpoint (``content_scoring``). Rules are fixed in the
Amendment 1 addendum of STRICT_CONTENT_CONTRACT.md.
"""
from __future__ import annotations

import json

from experiments.maxrl_comp002_pretraining_v1.content_scoring import (
    EXPRESSION_FAMILY, PARSE_ERRORS, is_int_list, parse_candidate, row_family,
    score, score_row, strip_wrappers)
from experiments.maxrl_composable_skills_v2.contracts import validate
from experiments.maxrl_composable_skills_v2.verifier import concrete

SUBTYPES = ("LENGTH_CAP", "EMPTY", "STRING_ENCODED_ANSWER", "NON_INT_ELEMENTS", "ALIAS_STRINGS",
            "KEY_VARIANT", "PROSE_WRAPPED", "MULTI_OBJECT", "OTHER")
RS_CLASSES = ("UNPARSEABLE", "UNKNOWN_ANSWER", "NOT_INT_LIST", "RS_PASS", "FULL_RESOLVED_SET",
              "RESOLVED_WRONG", "R_BLIND_S_VALID", "OTHER_WRONG")
KEY_VARIANTS = frozenset({"answers", "result", "output", "final_answer"})
_DECODER = json.JSONDecoder()  # permissive: shape detection only, never scoring


def well_typed(answer, family: str | None) -> bool:
    """The answer type a contract admits: int list or UNKNOWN (composition),
    string (countdown), int list (the seven array-answer legacy families).

    "Int list" is ``content_scoring.is_int_list`` (clarification C4), imported
    rather than restated, so ``[]`` is never well typed.
    """
    if family is None:
        return is_int_list(answer) or (type(answer) is str and answer == "UNKNOWN")
    if family == EXPRESSION_FAMILY:
        return type(answer) is str
    return is_int_list(answer)


def _json_sequence(text: str) -> list | None:
    """Every top-level JSON value when ``text`` is only JSON values and whitespace."""
    values, at = [], 0
    while True:
        while at < len(text) and text[at].isspace():
            at += 1
        if at == len(text):
            return values
        try:
            value, at = _DECODER.raw_decode(text, at)
        except (ValueError, RecursionError):
            return None
        values.append(value)


def _embeds_container(text: str) -> bool:
    for at, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = _DECODER.raw_decode(text, at)
        except (ValueError, RecursionError):
            continue
        if type(value) in (list, dict):
            return True
    return False


def subtype_strict_failure(row: dict, response: str, stop_reason: str | None) -> str | None:
    """One type for a strict failure outside the content normalisations, else None.

    None, before any type is tried (addendum A4, narrowed by A7 / Amendment 1
    clarification C6), for:

    - a strict pass;
    - a response whose content normalization is ``multiple_candidates_in_object``
      (``parse_candidate`` reports multiple candidates). That category is the
      row's one label, whatever the type of its ``answer`` value and whatever
      the stop reason: ``{"answer":null,"alt":[2]}`` is not OTHER, and
      ``{"answer":"[1]","alt":[2]}`` is not STRING_ENCODED_ANSWER;
    - a single parsed candidate of the contract's answer type (``well_typed``):
      a format-only pass or a well-typed wrong answer, named by its
      normalization.

    Scorer receipts count the second and third kinds in
    ``strict_failures_without_subtype``; the ``normalization`` counts tell them
    apart. Every other strict failure gets exactly one type from SUBTYPES,
    first match in that order.
    """
    if type(response) is not str:
        raise ValueError("response must be a string")
    if stop_reason is not None and type(stop_reason) is not str:
        raise ValueError("stop_reason must be a string or None")
    if score_row(row, response)["strict_pass"]:
        return None
    family = row_family(row)
    try:
        answer, _, multiple = parse_candidate(response, family)
    except PARSE_ERRORS:
        pass
    else:
        if multiple or well_typed(answer, family):
            return None
    if stop_reason == "length":
        return "LENGTH_CAP"
    if not response.strip():
        return "EMPTY"
    try:
        text, _ = strip_wrappers(response)
    except PARSE_ERRORS:
        text = response.strip()
    sequence = _json_sequence(text)
    whole = sequence[0] if sequence is not None and len(sequence) == 1 else None
    if type(whole) is dict:
        value = whole.get("answer")
    else:
        value = whole
    if type(value) is str:
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            decoded = None
        if type(decoded) in (list, dict):
            return "STRING_ENCODED_ANSWER"
    spec = row["verifier"]["spec"]
    aliases = set(spec["lookup"]["links"]) if family is None and "lookup" in spec else set()
    if type(value) is list:
        if any(type(item) is not int and not (type(item) is str and item in aliases) for item in value):
            return "NON_INT_ELEMENTS"
        if any(type(item) is str and item in aliases for item in value):
            return "ALIAS_STRINGS"
    if type(whole) is dict and len(whole) == 1:
        (key,) = whole
        if (key.casefold() == "answer" and key != "answer") or key in KEY_VARIANTS:
            return "KEY_VARIANT"
    if sequence is None and _embeds_container(text):
        return "PROSE_WRAPPED"
    if sequence is not None and len(sequence) >= 2:
        return "MULTI_OBJECT"
    return "OTHER"


def _require_rs(spec: dict) -> None:
    validate(spec)
    if "lookup" not in spec or "selection" not in spec or "updates" in spec or "precedence" in spec:
        raise ValueError("classify_rs needs an RS spec (lookup + selection only)")


def classify_rs(spec: dict, response: str) -> str:
    """Exactly one RS response class, first match in RS_CLASSES order.

    The candidate is the content-contract candidate, the one H3b's e uses
    (clarification C5): classes UNPARSEABLE, UNKNOWN_ANSWER and NOT_INT_LIST
    are exactly the responses whose H3b candidate is None.
    """
    _require_rs(spec)
    try:
        answer, _, multiple = parse_candidate(response)
    except PARSE_ERRORS:
        return "UNPARSEABLE"
    if multiple:
        return "UNPARSEABLE"
    if type(answer) is str and answer == "UNKNOWN":
        return "UNKNOWN_ANSWER"
    if not is_int_list(answer):
        return "NOT_INT_LIST"
    if score(spec, response)["content_pass"]:
        return "RS_PASS"
    resolved, values = concrete(spec)
    resolved = resolved or set()  # an unresolvable lookup resolves nothing
    chosen = set(answer)
    if resolved and chosen == resolved:
        return "FULL_RESOLVED_SET"
    if chosen <= resolved:
        return "RESOLVED_WRONG"
    k, target = spec["selection"]["k"], spec["selection"]["target"]
    if (len(answer) == k and len(chosen) == k and all(0 <= i < len(values) for i in answer)
            and sum(values[i] for i in answer) == target):
        return "R_BLIND_S_VALID"
    return "OTHER_WRONG"
