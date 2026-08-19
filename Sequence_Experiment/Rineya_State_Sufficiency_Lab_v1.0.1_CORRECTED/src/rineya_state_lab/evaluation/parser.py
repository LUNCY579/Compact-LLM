from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils import normalize_casefold, normalize_ws


_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:i\s+)?cannot\s+(?:answer|solve|help|comply)\b",
        r"\b(?:i\s+)?can't\s+(?:answer|solve|help|comply)\b",
        r"\bunable\s+to\s+(?:answer|solve|help)\b",
        r"\bnot\s+enough\s+information\b",
        r"\bas\s+an\s+ai\b",
    ]
]


@dataclass
class Candidate:
    value: Any
    source: str
    explicit: bool


@dataclass
class ResponseParse:
    raw: str
    status: str
    strict_final: Any = None
    lenient_final: Any = None
    final_source: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    conflict: bool = False
    refusal: bool = False
    empty: bool = False
    json_object: dict[str, Any] | None = None
    json_route: str | None = None
    json_repaired: bool = False
    structured: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json|JSON)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : index + 1])
                    break
    return objects


def _try_json(candidate: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value, "json", False
    except Exception:
        pass

    # Conservative repair: trailing commas and Python-literal dictionaries only.
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    if repaired != candidate:
        try:
            value = json.loads(repaired)
            if isinstance(value, dict):
                return value, "json_trailing_comma_repair", True
        except Exception:
            pass
    try:
        value = ast.literal_eval(candidate)
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return value, "python_literal_repair", True
    except Exception:
        pass
    return None, None, False


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    stripped = _strip_fence(text)
    candidates = [stripped, *_balanced_json_objects(stripped)]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        value, route, repaired = _try_json(candidate)
        if value is not None:
            return value, route, repaired
    return None, None, False


def _extract_tagged(text: str) -> list[Candidate]:
    results: list[Candidate] = []
    patterns = [
        (r"<final(?:_answer)?>\s*(.*?)\s*</final(?:_answer)?>", "final_tag"),
        (r"<answer>\s*(.*?)\s*</answer>", "answer_tag"),
        (r"\\boxed\s*\{([^{}]+)\}", "latex_boxed"),
        (r"(?im)^\s*(?:\d+[.)]\s*)?(?:final\s+answer|answer|result|output)\s*[:=]\s*(.+?)\s*$", "labeled_line"),
        (r"(?im)^\s*(?:\d+[.)]\s*)?(?:therefore|thus|so)[, ]+(?:the\s+)?answer\s+is\s+(.+?)\s*$", "conclusion_sentence"),
    ]
    for pattern, source in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL if "tag" in source else 0):
            value = normalize_ws(match.group(1))
            if value:
                results.append(Candidate(value, source, True))
    return results


def _json_final_candidates(value: dict[str, Any]) -> list[Candidate]:
    results: list[Candidate] = []
    for key in ["final_answer", "answer", "result", "final", "output"]:
        if key in value and value[key] is not None:
            results.append(Candidate(value[key], f"json.{key}", True))
    return results


def _last_line_candidate(text: str) -> Candidate | None:
    lines = [normalize_ws(line) for line in text.splitlines() if normalize_ws(line)]
    if not lines:
        return None
    last = lines[-1]
    if len(last) > 500:
        return None
    return Candidate(last, "last_nonempty_line", False)


def _numeric_tail_candidate(text: str) -> Candidate | None:
    # Used only as a lenient fallback; units and symbols are retained when possible.
    pattern = re.compile(
        r"(?<![\w.])(?:[$£€]\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?(?:\s*%|\s*[A-Za-z°²³]+(?:/[A-Za-z°²³]+)?)?\s*[.!?]?\s*$"
    )
    match = pattern.search(text.strip())
    if not match:
        return None
    return Candidate(match.group(0).strip().rstrip(".!?"), "numeric_tail", False)


def _canonical_candidate(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_casefold(value)


def parse_response(text: Any) -> ResponseParse:
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return ResponseParse(raw=raw, status="EMPTY", empty=True)
    refusal = any(pattern.search(stripped) for pattern in _REFUSAL_PATTERNS)
    json_object, json_route, repaired = extract_json_object(stripped)
    candidates: list[Candidate] = []
    if json_object is not None:
        candidates.extend(_json_final_candidates(json_object))
        # A structured transition JSON object with final_answer=null is not itself a final answer.
        # Do not reinterpret numbers inside the JSON as a lenient final candidate.
    else:
        candidates.extend(_extract_tagged(stripped))
        last = _last_line_candidate(stripped)
        if last is not None:
            candidates.append(last)
        numeric = _numeric_tail_candidate(stripped)
        if numeric is not None and all(numeric.value != candidate.value for candidate in candidates):
            candidates.append(numeric)

    explicit = [candidate for candidate in candidates if candidate.explicit]
    strict = explicit[0] if explicit else None
    lenient = strict or (candidates[0] if candidates else None)
    explicit_values = {_canonical_candidate(candidate.value) for candidate in explicit}
    conflict = len(explicit_values) > 1
    warnings: list[str] = []
    if conflict:
        warnings.append("Multiple explicit final-answer candidates disagree")
        strict = None
    if repaired:
        warnings.append("JSON required conservative repair")
    status = "OK"
    if refusal:
        status = "REFUSAL"
    elif conflict:
        status = "CONFLICTING_FINALS"
    elif lenient is None:
        status = "NO_ANSWER_FOUND"
    elif strict is None:
        status = "LENIENT_ONLY"
    return ResponseParse(
        raw=raw,
        status=status,
        strict_final=strict.value if strict else None,
        lenient_final=lenient.value if lenient else None,
        final_source=(strict or lenient).source if (strict or lenient) else None,
        candidates=[asdict(candidate) for candidate in candidates],
        conflict=conflict,
        refusal=refusal,
        json_object=json_object,
        json_route=json_route,
        json_repaired=repaired,
        warnings=warnings,
    )


_KEY_ALIASES = {
    "next_reasoning_step": ["next_reasoning_step", "next_step", "reasoning_step", "step", "action"],
    "updated_compact_state": ["updated_compact_state", "next_state", "updated_state", "state"],
    "done": ["done", "complete", "completed", "is_done", "finished"],
    "final_answer": ["final_answer", "answer", "result", "final"],
}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1", "done", "complete", "completed"}:
            return True
        if lowered in {"false", "no", "0", "continue", "incomplete"}:
            return False
    return None


def parse_stateful_response(text: Any) -> ResponseParse:
    parsed = parse_response(text)
    obj = parsed.json_object
    structured: dict[str, Any] = {
        "next_reasoning_step": None,
        "updated_compact_state": None,
        "done": None,
        "final_answer": None,
    }
    if obj is not None:
        for canonical, aliases in _KEY_ALIASES.items():
            for key in aliases:
                if key in obj:
                    structured[canonical] = obj[key]
                    break
        structured["done"] = _coerce_bool(structured["done"])
        if structured["final_answer"] is not None:
            parsed.strict_final = structured["final_answer"]
            parsed.lenient_final = structured["final_answer"]
            parsed.final_source = "json.final_answer"
        missing = [key for key in _KEY_ALIASES if structured[key] is None and key != "final_answer"]
        inconsistent_final = structured["done"] is True and structured["final_answer"] is None
        if missing or inconsistent_final:
            problems = list(missing)
            if inconsistent_final:
                problems.append("final_answer required when done=true")
            parsed.warnings.append(f"Structured output is missing or invalid fields: {problems}")
            parsed.status = "PARTIAL_STRUCTURED_OUTPUT"
        elif structured["done"] is False and structured["final_answer"] is None:
            # A valid non-final transition intentionally has no final answer.
            parsed.status = "OK" if not parsed.json_repaired else "OK_REPAIRED_JSON"
    else:
        # Lenient recovery from labels, while never claiming strict contract compliance.
        patterns = {
            "next_reasoning_step": r"(?im)^\s*(?:next reasoning step|next step|reasoning step)\s*[:=]\s*(.+)$",
            "updated_compact_state": r"(?im)^\s*(?:updated compact state|next state|updated state)\s*[:=]\s*(.+)$",
            "done": r"(?im)^\s*(?:done|complete|completed)\s*[:=]\s*(.+)$",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, parsed.raw)
            if match:
                structured[key] = match.group(1).strip()
        structured["done"] = _coerce_bool(structured["done"])
        structured["final_answer"] = parsed.lenient_final
        if any(value is not None for value in structured.values()):
            parsed.warnings.append("Structured fields recovered from unstructured text")
            if parsed.status == "NO_ANSWER_FOUND":
                parsed.status = "LENIENT_STRUCTURED_RECOVERY"
    parsed.structured = structured
    return parsed
