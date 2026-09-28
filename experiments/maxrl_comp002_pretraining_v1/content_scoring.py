"""Bounded content diagnosis using the unchanged MAXRL-COMP-002 verifiers.

Composition specs are rechecked by ``maxrl_composable_skills_v2.verifier.score``
and legacy specs by ``maxrl_verifiable_data_v1.verifier.score``. Strict outcomes
are never changed. The rules are STRICT_CONTENT_CONTRACT.md and its Amendment 1
addendum.
"""
from __future__ import annotations

import ast
import json
import warnings

from experiments.maxrl_composable_skills_v2.corpus import reward as strict_reward
from experiments.maxrl_composable_skills_v2.verifier import score as strict_score
from experiments.maxrl_verifiable_data_v1.specs import FAMILIES as LEGACY_FAMILIES
from experiments.maxrl_verifiable_data_v1.verifier import score as legacy_strict_score

MAX_CHARS = 4096
UNPARSEABLE = "unparseable_under_content_contract"
MULTIPLE = "multiple_candidates_in_object"
PARSE_ERRORS = (ValueError, TypeError, UnicodeError, OverflowError, RecursionError)
EXPRESSION_FAMILY = "countdown"  # the one legacy family whose answer is a string
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _nonfinite(value):
    raise ValueError("nonfinite JSON value: " + value)


def strip_wrappers(response: str) -> tuple[str, list[str]]:
    """Remove one outer code fence and one trailing period; return (text, wrappers)."""
    if type(response) is not str or len(response) > MAX_CHARS:
        raise ValueError("invalid or oversized response")
    text = response.strip()
    wrappers = []
    if text.startswith("```"):
        first, sep, rest = text.partition("\n")
        if not sep or first not in ("```", "```json") or not rest.endswith("\n```"):
            raise ValueError("invalid code fence")
        text = rest[:-4].strip()
        wrappers.append("code_fence")
    if text.endswith("."):
        text = text[:-1].rstrip()
        wrappers.append("trailing_period")
    if not text:
        raise ValueError("empty candidate")
    return text, wrappers


def is_int_list(value) -> bool:
    """A non-empty list whose elements all have type int (bools excluded).

    The package's one definition of "int list" (Amendment 1 section 15,
    clarification C4; STRICT_CONTENT_CONTRACT.md addendum A7). The A1/A2
    candidate rules, the subtyper's answer types, the RS classes and H3b's
    candidate and e all rest on it; other modules import it rather than restate
    it. ``[]``, ``[True]``, ``[1.0]``, int subclasses and tuples are not int
    lists. The verifiers' own range-checked predicates belong to the unchanged
    strict endpoint and are a different thing.
    """
    return type(value) is list and bool(value) and all(type(item) is int for item in value)


def is_expression(value) -> bool:
    """A string the countdown grammar admits, with at least one operator.

    Leaves are ints 1..100 and nodes are binary + - * /, as in the legacy
    verifier's ``expression_value``; the number multiset is not checked.
    """
    if type(value) is not str or not 0 < len(value) <= 512:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    if not isinstance(root.body, ast.BinOp):
        return False
    for node in ast.walk(root.body):
        if type(node) in _OPERATORS:
            continue
        if type(node) is ast.BinOp and type(node.op) in _OPERATORS:
            continue
        if type(node) is ast.Constant and type(node.value) is int and 1 <= node.value <= 100:
            continue
        return False
    return True


def _is_candidate(value, family: str | None) -> bool:
    if family is None:
        return is_int_list(value) or (type(value) is str and value == "UNKNOWN")
    if family == EXPRESSION_FAMILY:
        return is_expression(value)
    return is_int_list(value)


def _holds_candidate(roots, family: str | None) -> bool:
    """True if any root holds, at any depth, a candidate answer for the contract."""
    stack = list(roots)
    while stack:
        item = stack.pop()
        if _is_candidate(item, family):
            return True
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
    return False


def parse_candidate(response: str, family: str | None = None) -> tuple[object, list[str], bool]:
    """Return (answer, categories, multiple_candidates) for the one complete candidate.

    ``family`` None selects the composition contract; a legacy family name
    selects the legacy contract (no bare UNKNOWN; bare list only for families
    whose answer is a JSON array). Raises one of PARSE_ERRORS when the response
    holds no single candidate.
    """
    if family is not None and family not in LEGACY_FAMILIES:
        raise ValueError("unknown legacy family")
    text, wrappers = strip_wrappers(response)
    if family is None and text == "UNKNOWN":
        return "UNKNOWN", wrappers + ["bare_unknown"], False
    value = json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite)
    if type(value) is list and family != EXPRESSION_FAMILY:
        return value, wrappers + ["bare_list"], False
    if type(value) is dict and "answer" in value:
        extras = [field for key, field in value.items() if key != "answer"]
        if extras:
            wrappers.append("extra_object_fields")
        return value["answer"], wrappers + ["answer_object"], _holds_candidate(extras, family)
    raise ValueError("candidate has no single answer")


def _content(verify, response: str, family: str | None) -> tuple[bool, str]:
    try:
        answer, categories, multiple = parse_candidate(response, family)
        if multiple:
            return False, MULTIPLE
        normalized = json.dumps({"answer": answer}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return bool(verify(normalized)), "+".join(categories)
    except PARSE_ERRORS:
        return False, UNPARSEABLE


def _result(strict: bool, content: bool, category: str) -> dict:
    if strict and not content:
        raise AssertionError("strict verifier success lost under content normalization")
    return {"strict_pass": strict, "content_pass": content,
            "normalization": category,
            "format_only_failure": bool(content and not strict)}


def score(spec: dict, response: str) -> dict:
    """Return strict and bounded-content correctness without changing reward."""
    strict = bool(strict_score(spec, response))
    content, category = _content(lambda text: strict_score(spec, text), response, None)
    return _result(strict, content, category)


def score_legacy(family: str, problem: dict, response: str) -> dict:
    """Strict and bounded-content correctness for one legacy (guard) problem."""
    if family not in LEGACY_FAMILIES:
        raise ValueError("unknown legacy family")
    strict = bool(legacy_strict_score(family, problem, response))
    content, category = _content(lambda text: legacy_strict_score(family, problem, text), response, family)
    return _result(strict, content, category)


def row_family(row: dict) -> str | None:
    """None for a composition row, else the legacy family (validates the row kind)."""
    kind = row.get("kind") if type(row) is dict else None
    verifier = row.get("verifier") if type(row) is dict else None
    if type(verifier) is not dict or verifier.get("kind") != kind:
        raise ValueError("row/verifier kind mismatch")
    if kind == "composition":
        return None
    if kind == "legacy":
        spec = verifier.get("spec")
        if type(spec) is not dict or set(spec) != {"family", "problem"} or spec["family"] not in LEGACY_FAMILIES:
            raise ValueError("invalid legacy verifier spec")
        return spec["family"]
    raise ValueError("unknown row kind")


def score_row(row: dict, response: str) -> dict:
    """Score one canonical row; strict_pass is asserted equal to ``corpus.reward``."""
    family = row_family(row)
    spec = row["verifier"]["spec"]
    if family is None:
        result = score(spec, response)
    else:
        result = score_legacy(family, spec["problem"], response)
    if result["strict_pass"] != bool(strict_reward(row, response)):
        raise AssertionError("strict endpoint disagrees with corpus.reward")
    return result
