from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.rstrip(" .,!?:;")


def _final_number(value: str) -> float | None:
    matches = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.replace(",", "")
    )
    return float(matches[-1]) if matches else None


def _stem_token(token: str) -> str:
    value = token.casefold().strip("_'’")
    if value.endswith("'s") or value.endswith("’s"):
        value = value[:-2]
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


def _content_tokens(value: str) -> list[str]:
    stop = {"a", "an", "the", "it", "is", "are", "to", "of", "and"}
    return [
        stemmed
        for token in re.findall(r"[\w'’]+", normalize_text(value), flags=re.UNICODE)
        if (stemmed := _stem_token(token)) and stemmed not in stop
    ]


def score_response(response: str, scorer: dict[str, Any]) -> dict[str, Any]:
    kind = scorer["type"]
    normalized = normalize_text(response)
    raw_folded = unicodedata.normalize("NFKC", response).casefold()
    passed = False
    details: dict[str, Any] = {}
    if kind == "exact":
        answers = [normalize_text(str(value)) for value in scorer["answers"]]
        passed = normalized in answers
        details["normalized_answers"] = answers
    elif kind == "numeric":
        actual = _final_number(response)
        expected = float(scorer["value"])
        tolerance = float(scorer.get("tolerance", 0.0))
        passed = actual is not None and math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=tolerance
        )
        details.update(actual=actual, expected=expected, tolerance=tolerance)
    elif kind == "contains_all":
        required = [normalize_text(str(value)) for value in scorer["values"]]
        passed = all(value in normalized for value in required)
        details["required"] = required
    elif kind == "keywords":
        expected_tokens = _content_tokens(str(scorer["phrase"]))
        actual_tokens = set(_content_tokens(response))
        passed = bool(expected_tokens) and all(
            token in actual_tokens for token in expected_tokens
        )
        details.update(
            expected_keywords=expected_tokens,
            matched_keywords=sorted(set(expected_tokens) & actual_tokens),
        )
    elif kind == "choice":
        allowed = [normalize_text(str(value)) for value in scorer["allowed"]]
        tokens = [normalize_text(value) for value in re.findall(r"[\w'-]+", response)]
        recognized = [token for token in tokens if token in allowed]
        actual = recognized[-1] if recognized else None
        expected = normalize_text(str(scorer["value"]))
        passed = actual == expected
        details.update(actual=actual, expected=expected, allowed=allowed)
    elif kind == "regex":
        pattern = str(scorer["pattern"])
        passed = re.fullmatch(pattern, response.strip(), flags=re.IGNORECASE) is not None
        details["pattern"] = pattern
    elif kind == "json_exact":
        try:
            actual_json = json.loads(response.strip())
        except json.JSONDecodeError:
            actual_json = None
        expected_json = scorer["value"]
        passed = actual_json == expected_json
        details.update(actual=actual_json, expected=expected_json)
    elif kind == "constraints":
        words = re.findall(r"\b[\w'-]+\b", response, flags=re.UNICODE)
        minimum = int(scorer.get("minimum_words", 0))
        maximum = int(scorer.get("maximum_words", 10**9))
        required = [normalize_text(str(value)) for value in scorer.get("contains", [])]
        forbidden = [
            unicodedata.normalize("NFKC", str(value)).casefold()
            for value in scorer.get("forbids", [])
        ]
        passed = (
            minimum <= len(words) <= maximum
            and all(value in normalized for value in required)
            and all(value not in raw_folded for value in forbidden)
        )
        details.update(
            word_count=len(words),
            minimum_words=minimum,
            maximum_words=maximum,
            required=required,
            forbidden=forbidden,
        )
    else:
        raise ValueError(f"unsupported scorer: {kind}")
    return {"correct": bool(passed), "score": 1.0 if passed else 0.0, **details}


def aggregate_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    family_scores: dict[str, list[float]] = defaultdict(list)
    scenario_scores: dict[str, list[float]] = defaultdict(list)
    invalid = 0
    total_latency = 0.0
    generated_tokens = 0
    eos_emitted = 0
    format_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = float(row["score"]["score"])
        family_scores[row["family"]].append(score)
        scenario_scores[row["scenario_id"]].append(score)
        invalid += int(bool(row.get("invalid_generation")))
        total_latency += float(row.get("latency_seconds", 0.0))
        generated_tokens += int(row.get("generated_tokens", 0))
        eos_emitted += int(bool(row.get("eos_emitted")))
        if row.get("format_score") is not None:
            format_value = float(row["format_score"]["score"])
            format_scores[row["family"]].append(format_value)
    families = {
        family: {
            "correct": int(sum(values)),
            "items": len(values),
            "score": sum(values) / len(values),
        }
        for family, values in sorted(family_scores.items())
    }
    scenario_means = [sum(values) / len(values) for values in scenario_scores.values()]
    return {
        "items": len(rows),
        "correct": int(sum(row["score"]["score"] for row in rows)),
        "overall_score": (
            sum(row["score"]["score"] for row in rows) / len(rows) if rows else 0.0
        ),
        "worst_family_score": min(
            (details["score"] for details in families.values()), default=0.0
        ),
        "scenario_macro_score": (
            sum(scenario_means) / len(scenario_means) if scenario_means else 0.0
        ),
        "families": families,
        "scenario_count": len(scenario_scores),
        "invalid_generations": invalid,
        "invalid_generation_rate": invalid / len(rows) if rows else 0.0,
        "wall_seconds": total_latency,
        "generated_tokens": generated_tokens,
        "eos_emitted": eos_emitted,
        "completion_rate": eos_emitted / len(rows) if rows else 0.0,
        "format_adherence": {
            "overall": (
                sum(value for values in format_scores.values() for value in values)
                / sum(len(values) for values in format_scores.values())
                if format_scores
                else None
            ),
            "families": {
                family: sum(values) / len(values)
                for family, values in sorted(format_scores.items())
            },
            "items": sum(len(values) for values in format_scores.values()),
        },
        "tokens_per_second": generated_tokens / total_latency if total_latency > 0 else None,
    }
