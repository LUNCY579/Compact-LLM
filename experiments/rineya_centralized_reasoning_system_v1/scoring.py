"""CRS evaluator-only semantic/format scorer; never imported by deployed runtime."""
from __future__ import annotations

from collections import Counter

STATUSES = ("SUPPORTED", "REFUTED", "UNKNOWN", "CONTRADICTORY")


def _valid(answer):
    if type(answer) is not dict or set(answer) not in ({"status", "value"}, {"status", "value", "evidence_ids"}):
        return False
    if "evidence_ids" in answer:
        refs = answer["evidence_ids"]
        if type(refs) is not list or any(type(ref) is not str or not ref for ref in refs) or len(refs) != len(set(refs)):
            return False
    if type(answer["status"]) is not str or answer["status"] not in STATUSES:
        return False
    return ((type(answer["value"]) is str and bool(answer["value"])) or
            (type(answer["value"]) is int and abs(answer["value"]) <= 1000000)) if answer["status"] == "SUPPORTED" else answer["value"] is None


def score_one(answer, gold):
    if not _valid(gold):
        raise ValueError("corrupt evaluator gold, not a model error")
    operational_error = type(answer) is dict and "error" in answer
    valid = _valid(answer)
    semantic = valid and (answer["status"], answer["value"]) == (gold["status"], gold["value"])
    abstains = valid and answer["status"] in ("UNKNOWN", "CONTRADICTORY")
    answerable = gold["status"] == "SUPPORTED"
    return {"semantic_exact": bool(semantic), "format_valid": bool(valid),
            "status_correct": bool(valid and answer["status"] == gold["status"]),
            "value_correct": bool(valid and answer["value"] == gold["value"]),
            "answerable": answerable, "abstains": bool(abstains),
            "erroneous_abstention": bool(answerable and abstains),
            "unsupported_answer": bool(valid and answer["status"] == "SUPPORTED" and not semantic),
            "operational_error": bool(operational_error), "missing": answer is None}


def score(predictions, gold):
    """Preserve every scheduled gold case, including absent/failed predictions.

    Gold is a private case-ID mapping. Runtime failure dictionaries may carry
    'error'; they are failed observations with their full denominator retained.
    An unavailable scorer must be a host failure; this API never invents grades.
    """
    if type(predictions) is not dict or type(gold) is not dict or not gold:
        raise ValueError("nonempty gold and prediction mappings required")
    if any(type(k) is not str or not k for k in gold) or any(not _valid(g) for g in gold.values()):
        raise ValueError("corrupt evaluator gold mapping")
    extra = sorted(set(predictions) - set(gold))
    rows = {cid: score_one(predictions.get(cid), target) for cid, target in gold.items()}
    count = len(rows)
    answerable = sum(r["answerable"] for r in rows.values())
    unknown = sum(g["status"] == "UNKNOWN" for g in gold.values())
    status_matrix = {status: {} for status in STATUSES}
    for cid, target in gold.items():
        prediction = predictions.get(cid)
        label = prediction["status"] if _valid(prediction) else "MISSING" if prediction is None else "ERROR" if type(prediction) is dict and "error" in prediction else "MALFORMED"
        cell = status_matrix[target["status"]]
        cell[label] = cell.get(label, 0) + 1
    counts = {key: sum(row[key] for row in rows.values()) for key in next(iter(rows.values()))}
    return {"schema_version": 1, "scheduled_cases": count, "observed_cases": sum(cid in predictions for cid in gold),
            "extra_prediction_ids": extra, "all_scheduled_accounted": True, "counts": counts,
            "semantic_accuracy": counts["semantic_exact"] / count, "strict_accuracy": counts["semantic_exact"] / count,
            "format_valid_rate": counts["format_valid"] / count,
            "answerable_accuracy": sum(r["semantic_exact"] and r["answerable"] for r in rows.values()) / answerable if answerable else None,
            "erroneous_abstention_rate": counts["erroneous_abstention"] / answerable if answerable else None,
            "unknown_correct_rate": sum(rows[cid]["semantic_exact"] for cid, g in gold.items() if g["status"] == "UNKNOWN") / unknown if unknown else None,
            "gold_status_counts": dict(Counter(g["status"] for g in gold.values())), "status_confusion": status_matrix, "rows": rows,
            "claim_scope": "controlled-grammar CPU/development instrument; no native quality or retention certificate",
            "semantic_format_limitation": "semantic equivalence is exact structured status/value; prose equivalence unqualified"}
