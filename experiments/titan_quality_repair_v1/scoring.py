"""Typed response scoring and complete-coverage aggregation, without model imports.

Gold comes from deterministic world oracles.  These functions report measurements;
calling them with fixtures never establishes native model capability.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .schema import ABSTENTION, FAMILIES, REPAIR_FAMILIES, canonical, validate_row


def _typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _typed_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _typed_shape(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _typed_shape(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_shape(a, b) for a, b in zip(actual, expected))
    return True


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON: {value}")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
        return value, isinstance(value, dict)
    except (ValueError, TypeError):
        return None, False


def _unwrap(text):
    """Allow one explicit answer wrapper for semantic scoring, never extra prose."""
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json|text)?\s*\n(.*?)\n```", text, flags=re.S)
        return match.group(1).strip() if match else text
    match = re.fullmatch(r"(?:answer|final answer|result)\s*:\s*(.+)", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text


_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_ABSTENTIONS = {
    "unknown", "i don't know", "i do not know", "cannot determine", "can't determine",
    "not enough information", "insufficient information", "not provided", "not recorded",
    "not stated", "not available", "no supported answer", "unanswerable",
    "i don't know based on the available information",
    "i cannot determine the answer from the supplied information",
    "the answer is not provided", "the answer is not recorded",
    "the answer is not stated", "the information is insufficient",
    "cannot determine from the supplied facts", "cannot determine from the available information",
    "the answer cannot be determined from the provided information",
    "the answer cannot be determined from the supplied facts",
    "there is not enough information to answer", "the information needed is not provided",
    "the requested value is not recorded", "the requested information is not provided",
}


def _abstained(text, declared):
    # A leading numerical answer is an answer even if later lines disclaim it.
    if re.match(r"\s*(?:answer\s*:\s*)?-?[0-9]+(?:\s|[.,]|$)", text, flags=re.I):
        return False
    normalized = re.sub(r"\s+", " ", text.strip()).lower().rstrip(".!")
    declared_normalized = re.sub(r"\s+", " ", declared.strip()).lower().rstrip(".!")
    # Full-string matching prevents 'unknown, but the answer is 7' from passing.
    return bool(normalized) and (normalized == declared_normalized or normalized in _ABSTENTIONS)


def score_response(row, text):
    """Return independent semantic and requested-contract correctness flags."""
    validate_row(row)
    if row["answerable"] and type(row["expected"]) not in (int, str, dict):
        raise ValueError("STOP_INVALID_SUPPORTED_ORACLE_TYPE")
    if not isinstance(text, str):
        raise ValueError("STOP_RESPONSE_NOT_TEXT")
    raw = text.strip()
    unwrapped = _unwrap(raw)
    contract = row["contract"]
    kind = contract["kind"]
    declared = contract.get("abstention", ABSTENTION)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("STOP_INVALID_ABSTENTION_CONTRACT")
    abstained = _abstained(raw, declared)
    if abstained:
        semantic = not row["answerable"]
        strict = semantic and raw == declared
        valid = raw == declared
    else:
        expected = row["expected"]
        if kind == "json_object":
            strict_value, valid = _json(raw)
            semantic_value, semantic_valid = _json(unwrapped)
            # Full expected keys/types are required even in semantic scoring.
            strict = valid and _typed_equal(strict_value, expected)
            semantic = semantic_valid and _typed_equal(semantic_value, expected)
            if row["answerable"] and isinstance(expected, dict):
                valid = valid and _typed_shape(strict_value, expected)
        elif kind == "integer":
            valid = bool(_INTEGER.fullmatch(raw))
            semantic_valid = bool(_INTEGER.fullmatch(unwrapped))
            strict = valid and type(expected) is int and int(raw) == expected
            semantic = semantic_valid and type(expected) is int and int(unwrapped) == expected
        else:
            valid = bool(raw) and "\n" not in raw
            # A narrow deterministic text oracle; no subjective/model judging.
            expected_text = canonical(expected) if isinstance(expected, dict) else str(expected).strip() if expected is not None else ""
            if isinstance(expected, dict):
                semantic_value, semantic_valid = _json(unwrapped)
                semantic = semantic_valid and _typed_equal(semantic_value, expected)
            elif type(expected) is int:
                semantic = bool(_INTEGER.fullmatch(unwrapped)) and int(unwrapped) == expected
            else:
                semantic = bool(expected_text) and unwrapped.casefold().rstrip(".") == expected_text.casefold().rstrip(".")
            strict = valid and raw == expected_text
        semantic = bool(row["answerable"] and semantic)
        strict = bool(row["answerable"] and strict)
    return {
        "semantic_correct": bool(semantic), "strict_correct": bool(strict),
        "abstained": bool(abstained), "unsupported_answer": bool(raw) and not row["answerable"] and not abstained,
        "unnecessary_abstention": row["answerable"] and abstained, "format_valid": bool(valid),
        "no_answer": not bool(raw),
    }


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(items):
    n = len(items)
    answerable = sum(item["answerable"] for item in items)
    unsupported = n - answerable
    counts = {name: sum(item[name] for item in items) for name in (
        "semantic_correct", "strict_correct", "abstained", "unsupported_answer",
        "unnecessary_abstention", "format_valid", "no_answer")}
    return {
        "n": n, "answerable_n": answerable, "unanswerable_n": unsupported,
        **counts,
        "semantic_accuracy": _rate(counts["semantic_correct"], n),
        "strict_accuracy": _rate(counts["strict_correct"], n),
        "supported_answer_accuracy": _rate(sum(item["semantic_correct"] for item in items if item["answerable"]), answerable),
        "unsupported_answer_rate": _rate(counts["unsupported_answer"], unsupported),
        "unnecessary_abstention_rate": _rate(counts["unnecessary_abstention"], answerable),
        "coverage": _rate(n - counts["abstained"] - counts["no_answer"], n),
        "format_valid_rate": _rate(counts["format_valid"], n),
    }


def aggregate_rows(rows, outputs):
    """Reject incomplete coverage; variants are diagnostic and never independent N.

    outputs is a mapping of row_id -> text, or an iterable of {row_id, response}.
    The latter is preferred at file boundaries because duplicate IDs remain visible.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("STOP_EMPTY_EVALUATION")
    identifiers = [row["row_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("STOP_DUPLICATE_ROW_ID")
    output_map = {}
    items = outputs.items() if isinstance(outputs, dict) else (
        (item["row_id"], item["response"]) for item in outputs)
    for row_id, response in items:
        if row_id in output_map:
            raise ValueError("STOP_DUPLICATE_OUTPUT_ID")
        if not isinstance(response, str):
            raise ValueError("STOP_RESPONSE_NOT_TEXT")
        output_map[row_id] = response
    if set(output_map) != set(identifiers):
        raise ValueError("STOP_OUTPUT_COVERAGE_MISMATCH")
    primary = defaultdict(list)
    variants = []
    canonical_worlds = set()
    split_set = set()
    for row in rows:
        score = score_response(row, output_map[row["row_id"]])
        item = {"row_id": row["row_id"], "world_id": row["world_id"], "family": row["family"],
                "variant_id": row["variant_id"], "answerable": row["answerable"],
                "support_expansion": row["support_expansion"], **score}
        split_set.add(row["split"])
        if row["variant_id"] == "canonical":
            if row["world_id"] in canonical_worlds:
                raise ValueError("STOP_DUPLICATE_CANONICAL_WORLD")
            canonical_worlds.add(row["world_id"])
            primary[row["family"]].append(item)
        else:
            variants.append(item)
    if len(split_set) != 1:
        raise ValueError("STOP_MIXED_EVALUATION_SPLITS")
    primary_map = {item["world_id"]: item for family in primary.values() for item in family}
    if not primary_map or any(item["world_id"] not in primary_map or
                              item["family"] != primary_map[item["world_id"]]["family"] or
                              item["answerable"] != primary_map[item["world_id"]]["answerable"]
                              for item in variants):
        raise ValueError("STOP_VARIANT_WITHOUT_MATCHED_CANONICAL_WORLD")
    variant_keys = [(item["world_id"], item["variant_id"]) for item in variants]
    if len(set(variant_keys)) != len(variant_keys):
        raise ValueError("STOP_DUPLICATE_WORLD_VARIANT")
    all_primary = [item for family in FAMILIES for item in primary[family]]
    repair = [item for item in all_primary if item["family"] in REPAIR_FAMILIES]
    expanded = [item for item in all_primary if item["support_expansion"]]
    regular = [item for item in all_primary if not item["support_expansion"]]
    return {
        "schema_version": 1, "split": next(iter(split_set)), "measurement_only": True,
        "native_capability_claim": False,
        "primary": {family: sorted(primary[family], key=lambda item: item["world_id"]) for family in FAMILIES},
        "families": {family: _metrics(primary[family]) for family in FAMILIES},
        "overall": _metrics(all_primary), "repair": _metrics(repair),
        "support_expansion": _metrics(expanded), "original_support": _metrics(regular),
        "variants": {"items": variants, "metrics": _metrics(variants),
                     "independent_n": 0, "attached_world_n": len({item["world_id"] for item in variants}),
                     "canonical_disagreements": sum(
                         item["semantic_correct"] != primary_map[item["world_id"]]["semantic_correct"] for item in variants)},
    }
