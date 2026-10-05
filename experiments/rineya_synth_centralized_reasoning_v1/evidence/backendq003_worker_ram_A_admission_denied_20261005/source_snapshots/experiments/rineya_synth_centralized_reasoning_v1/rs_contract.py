"""Bounded public R/S contract; CPU fixtures, never a native launch permit.

Semantics/bounds follow maxrl_composable_skills_v2/contracts.py 2.0.0.
Only the R/S subset is implemented locally; no bank or model imports.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

VERSION = "synth-rs-contract-1"
MAX_BYTES = 65536
SYSTEM = ('Solve using only the stated data and rules. Return exactly one JSON object '
          'with the single key "answer": a nonempty list of distinct zero-based item '
          'indices satisfying all requirements, or the string "UNKNOWN" only when '
          'no such list exists. Any valid list is accepted. No explanations or extra keys.')
INTRO = ("Complete this dispatch task.", "Find one valid plan for these items.",
         "Return a certificate meeting every requirement.", "Resolve this finite planning instance.")
R_RULE = ('Resolve EVERY requested alias through links until an integer item index is reached. '
          'Repeated resolved items count once. A missing link or cycle in any requested '
          'chain means no valid answer exists. Other aliases are distractors.')
S_RULE = ('Choose exactly k distinct eligible item indices whose INITIAL values sum to target. '
          'There is no preferred subset.')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid bounded JSON value") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError("nonfinite JSON")


def strict_json(text: str) -> Any:
    require(type(text) is str and len(text) <= MAX_BYTES, "wrong type or excessive characters")
    require(utf8_size(text) <= MAX_BYTES, "excessive bytes")
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_constant)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid bounded JSON") from exc


def utf8_size(text: str) -> int:
    try:
        return len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("invalid UTF-8 text") from exc


def isint(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def alias(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,19}", value) is not None


def validate_spec(spec: Any) -> str:
    require(type(spec) is dict and "initial" in spec and
            set(spec) <= {"initial", "lookup", "selection"}, "invalid problem fields")
    values = spec["initial"]
    require(type(values) is list and 6 <= len(values) <= 15 and
            all(isint(v, -100, 100) for v in values), "invalid initial values")
    n = len(values)
    sig = ("R" if "lookup" in spec else "") + ("S" if "selection" in spec else "")
    require(bool(sig), "no active skills")
    if "R" in sig:
        lookup = spec["lookup"]
        require(type(lookup) is dict and set(lookup) == {"links", "requested"}, "lookup fields")
        links = lookup["links"]
        require(type(links) is dict and 1 <= len(links) <= 80, "lookup map size")
        require(all(alias(k) and (alias(v) or isint(v, 0, n - 1))
                    for k, v in links.items()), "invalid alias/link")
        requested = lookup["requested"]
        require(type(requested) is list and 1 <= len(requested) <= n and
                all(alias(k) for k in requested), "invalid requested aliases")
    if "S" in sig:
        selection = spec["selection"]
        require(type(selection) is dict and set(selection) == {"k", "target"}, "selection fields")
        require(isint(selection["k"], 1, min(5, n)) and
                isint(selection["target"], -50000, 50000), "invalid selection limits")
    return sig


def render(spec: dict, template: int = 0) -> list[dict[str, str]]:
    sig = validate_spec(spec)
    require(isint(template, 0, 3), "invalid template")
    rules = [R_RULE if "R" in sig else "All items are eligible; no reference lookup is needed.",
             S_RULE if "S" in sig else "Return ALL resolved items, each exactly once.",
             "Order does not matter. There are no updates, filtering, or precedence rules."]
    if template % 2 == 0:
        block = "SPEC\n" + canonical(spec) + "\nEND_SPEC"
    else:
        block = "DATA_FIELDS\n" + "\n".join(k + " = " + canonical(spec[k])
                  for k in ("initial", "lookup", "selection") if k in spec) + "\nEND_DATA_FIELDS"
    text = INTRO[template] + "\n" + "\n".join(rules) + "\n" + block
    require(utf8_size(text) <= MAX_BYTES, "overlength prompt")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def parse_visible(messages: Any) -> dict:
    require(type(messages) is list and len(messages) == 2 and
            messages[0] == {"role": "system", "content": SYSTEM}, "policy boundary")
    user = messages[1]
    require(type(user) is dict and set(user) == {"role", "content"} and
            user["role"] == "user" and type(user["content"]) is str, "user boundary")
    text = user["content"]
    require(utf8_size(text) <= MAX_BYTES, "overlength prompt")
    try:
        if "\nSPEC\n" in text:
            spec = strict_json(text.split("\nSPEC\n", 1)[1].removesuffix("\nEND_SPEC"))
        else:
            spec = {}
            block = text.split("\nDATA_FIELDS\n", 1)[1].removesuffix("\nEND_DATA_FIELDS")
            for line in block.splitlines():
                key, value = line.split(" = ", 1)
                require(key not in spec, "duplicate field")
                spec[key] = strict_json(value)
        require(any(messages == render(spec, t) for t in range(4)), "rules or rendering drift")
        return spec
    except (IndexError, KeyError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid visible task") from exc


def parse_answer(text: str, n: int) -> list[int] | str:
    answer = strict_json(text)
    require(type(answer) is dict and set(answer) == {"answer"}, "answer fields")
    answer = answer["answer"]
    if type(answer) is str:
        require(answer == "UNKNOWN", "invalid abstention")
        return answer
    require(type(answer) is list and 1 <= len(answer) <= n and
            all(isint(v, 0, n - 1) for v in answer) and
            len(set(answer)) == len(answer), "invalid answer indices")
    return sorted(answer)


def format_answer(answer: list[int] | str) -> str:
    return canonical({"answer": sorted(answer) if type(answer) is list else answer})
