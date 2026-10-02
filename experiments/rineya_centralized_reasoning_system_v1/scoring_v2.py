"""Versioned evaluator: semantic fields and strict format have separate outcomes.

No model/runtime imports. Projection only reads explicit unique status/value
fields, never guesses from prose or gold. Operational failure is never abstention.
Original scorer and epoch001 receipts retain their original contract.
"""
from __future__ import annotations

from collections import Counter
import json
import math

VERSION = "CRS-SEMANTIC-STATUS-SCORER-v2"
STATUSES = ("SUPPORTED", "REFUTED", "UNKNOWN", "CONTRADICTORY")
MAX_RESPONSE_BYTES = 4096
MAX_INTEGER = 1_000_000


def _finite(value, depth=0):
    if depth > 16:
        raise ValueError("response nesting ceiling")
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("string JSON keys required")
        for v in value.values():
            _finite(v, depth + 1)
    elif type(value) is list:
        for v in value:
            _finite(v, depth + 1)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("JSON types required")


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON response key")
        out[key] = value
    return out


def _bad_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def parse_response(raw):
    if type(raw) is str:
        if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("response byte ceiling")
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_bad_constant)
    else:
        value = raw
    _finite(value)
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("response byte ceiling")
    return value


def _projection(value):
    if type(value) is not dict or not {"status", "value"} <= value.keys():
        return None
    status, answer = value["status"], value["value"]
    if type(status) is not str or status not in STATUSES:
        return None
    if status == "SUPPORTED":
        valid = ((type(answer) is str and bool(answer)) or
                 (type(answer) is int and abs(answer) <= MAX_INTEGER))
    else:
        valid = answer is None
    return {"status": status, "value": answer} if valid else None


def _format(value):
    if _projection(value) is None or set(value) not in ({"status", "value"}, {"status", "value", "evidence_ids"}):
        return False
    if "evidence_ids" in value:
        refs = value["evidence_ids"]
        if type(refs) is not list or any(type(x) is not str or not x for x in refs) or len(refs) != len(set(refs)):
            return False
    return True


def validate_gold(gold):
    if type(gold) is not dict or set(gold) != {"status", "value"} or _projection(gold) is None:
        raise ValueError("corrupt evaluator gold")
    parse_response(gold)
    return gold


def score_one(answer, gold):
    validate_gold(gold)
    missing = answer is None
    parse_error = None
    try:
        value = parse_response(answer)
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError) as exc:
        value, parse_error = None, type(exc).__name__
    operational_error = type(value) is dict and "error" in value and value["error"] is not None
    projected = None if operational_error else _projection(value)
    format_valid = bool(not operational_error and parse_error is None and _format(value))
    semantic = bool(projected is not None and projected == gold)
    abstains = bool(projected is not None and projected["status"] in ("UNKNOWN", "CONTRADICTORY"))
    positive = gold["status"] == "SUPPORTED"
    determinate = gold["status"] in ("SUPPORTED", "REFUTED")
    return {"semantic_exact": semantic, "strict_exact": bool(semantic and format_valid),
            "format_valid": format_valid, "status_correct": bool(projected is not None and projected["status"] == gold["status"]),
            "value_correct": bool(projected is not None and type(projected["value"]) is type(gold["value"]) and projected["value"] == gold["value"]),
            "answerable": positive, "determinate": determinate, "abstains": abstains,
            "erroneous_abstention": bool(positive and abstains), "determinate_erroneous_abstention": bool(determinate and abstains),
            "unsupported_answer": bool(projected is not None and projected["status"] == "SUPPORTED" and not semantic),
            "operational_error": bool(operational_error), "missing": missing,
            "malformed": bool(not missing and not operational_error and not format_valid),
            "parse_error": parse_error, "semantic_projection": projected}


def score(predictions, gold):
    if type(predictions) is not dict or type(gold) is not dict or not gold:
        raise ValueError("prediction and nonempty private gold mappings required")
    if any(type(k) is not str or not k for k in gold) or any(type(k) is not str or not k for k in predictions):
        raise ValueError("nonempty case ID strings required")
    for target in gold.values():
        validate_gold(target)
    rows = {cid: score_one(predictions.get(cid), target) for cid, target in gold.items()}
    booleans = ("semantic_exact", "strict_exact", "format_valid", "status_correct", "value_correct",
                "answerable", "determinate", "abstains", "erroneous_abstention", "determinate_erroneous_abstention",
                "unsupported_answer", "operational_error", "missing", "malformed")
    counts = {name: sum(row[name] for row in rows.values()) for name in booleans}
    matrix = {state: {} for state in STATUSES}
    for cid, target in gold.items():
        row = rows[cid]
        label = ("ERROR" if row["operational_error"] else "MISSING" if row["missing"] else
                 row["semantic_projection"]["status"] if row["semantic_projection"] is not None else "MALFORMED")
        cell = matrix[target["status"]]
        cell[label] = cell.get(label, 0) + 1
    def conditional(flag, numerator="semantic_exact"):
        denominator = counts[flag]
        return sum(r[flag] and r[numerator] for r in rows.values()) / denominator if denominator else None
    extra = sorted(set(predictions) - set(gold))
    return {"schema_version": 2, "scorer_version": VERSION, "scheduled_cases": len(gold),
            "observed_cases": sum(cid in predictions for cid in gold), "extra_prediction_ids": extra,
            "membership_valid": not extra, "all_scheduled_accounted": True, "counts": counts, "rows": rows,
            "semantic_accuracy": counts["semantic_exact"] / len(gold), "strict_accuracy": counts["strict_exact"] / len(gold),
            "format_valid_rate": counts["format_valid"] / len(gold), "answerable_accuracy": conditional("answerable"),
            "answerable_scope": "SUPPORTED positive-value tasks only", "determinate_accuracy": conditional("determinate"),
            "determinate_scope": "SUPPORTED and explicit REFUTED tasks",
            "erroneous_abstention_rate": counts["erroneous_abstention"] / counts["answerable"] if counts["answerable"] else None,
            "determinate_erroneous_abstention_rate": counts["determinate_erroneous_abstention"] / counts["determinate"] if counts["determinate"] else None,
            "gold_status_counts": dict(Counter(g["status"] for g in gold.values())), "status_confusion": matrix,
            "calibration": None, "provenance_truth_graded": False,
            "scope": "explicit bounded unique JSON status/value semantics; prose semantics and native efficacy unqualified"}
