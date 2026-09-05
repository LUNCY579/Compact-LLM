"""Frozen strict scoring adapters; no last-number/keyword/LLM judging."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from .hashing import canonical_sha256, sha256_file

_LEGACY_PATH = Path(__file__).resolve().parents[3] / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/rineya_cross_scale/evaluation/scoring.py"
_spec = importlib.util.spec_from_file_location("_adaptive_legacy_scoring", _LEGACY_PATH)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
normalize_text = _legacy.normalize_text
SCORER_VERSION = "strict_exact_or_json_v2.2"


def scorer_sha256() -> str:
    return canonical_sha256({"version": SCORER_VERSION, "adapter": sha256_file(Path(__file__)),
                             "legacy": sha256_file(_LEGACY_PATH)})


def canonical_answer(case: dict[str, Any]) -> str:
    scorer = case["scorer"]
    _validate_scorer(scorer)
    if scorer["type"] == "exact" and len(scorer["answers"]) == 1:
        return str(scorer["answers"][0])
    if scorer["type"] == "json_exact":
        return json.dumps(scorer["value"], sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    raise ValueError("only a single exact answer or exact JSON is licensed")


def _validate_scorer(scorer: dict[str, Any]) -> None:
    if not isinstance(scorer, dict):
        raise ValueError("scorer must be an object")
    if scorer.get("type") == "exact":
        answers = scorer.get("answers")
        if set(scorer) != {"type", "answers"} or not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str) or not normalize_text(answers[0]):
            raise ValueError("exact scorer requires one nonempty textual answer")
    elif scorer.get("type") == "json_exact":
        if set(scorer) != {"type", "value"}:
            raise ValueError("JSON scorer requires only one canonical value")
        def check(value):
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("JSON object keys must be strings")
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError("unregistered JSON value type")
        check(scorer["value"])
        canonical_sha256(scorer["value"])
    else:
        raise ValueError("unregistered scorer; only exact and json_exact are licensed")


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def score_response(response: str, scorer: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, str):
        raise TypeError("response must be text")
    _validate_scorer(scorer)
    if scorer.get("type") == "exact":
        if not isinstance(scorer.get("answers"), list) or len(scorer["answers"]) != 1:
            raise ValueError("exact scorer requires one unambiguous canonical answer")
        result = _legacy.score_response(response, scorer)
    elif scorer.get("type") == "json_exact":
        try:
            actual = json.loads(response.strip(), object_pairs_hook=_no_duplicate_keys,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            # Canonical JSON comparison distinguishes true from 1, unlike Python ==.
            correct = canonical_sha256(actual) == canonical_sha256(scorer["value"])
        except (ValueError, TypeError):
            actual, correct = None, False
        result = {"correct": correct, "score": float(correct), "actual": actual}
    else:
        raise ValueError("unregistered scorer; only exact and json_exact are licensed")
    return {**result, "ambiguous": False, "score_confidence": "deterministic",
            "scorer_version": SCORER_VERSION}
