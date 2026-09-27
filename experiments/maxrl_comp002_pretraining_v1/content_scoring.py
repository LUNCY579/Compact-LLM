"""Bounded content diagnosis using the unchanged MAXRL-COMP-002 verifier."""
from __future__ import annotations

import json

from experiments.maxrl_composable_skills_v2.verifier import score as strict_score

MAX_CHARS = 4096


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _nonfinite(value):
    raise ValueError("nonfinite JSON value: " + value)


def _candidate(response: str):
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
    if text == "UNKNOWN":
        return "UNKNOWN", wrappers + ["bare_unknown"]
    value = json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite)
    if type(value) is list:
        return value, wrappers + ["bare_list"]
    if type(value) is dict and "answer" in value:
        if set(value) != {"answer"}:
            wrappers.append("extra_object_fields")
        return value["answer"], wrappers + ["answer_object"]
    raise ValueError("candidate has no single answer")


def score(spec: dict, response: str) -> dict:
    """Return strict and bounded-content correctness without changing reward."""
    strict = bool(strict_score(spec, response))
    try:
        answer, categories = _candidate(response)
        normalized = json.dumps({"answer": answer}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        content = bool(strict_score(spec, normalized))
        category = "+".join(categories)
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        content = False
        category = "unparseable_under_content_contract"
    if strict and not content:
        raise AssertionError("strict verifier success lost under content normalization")
    return {"strict_pass": strict, "content_pass": content,
            "normalization": category,
            "format_only_failure": bool(content and not strict)}
