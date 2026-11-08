"""Strict single-field parser. No substring extraction, coercion or label-based repair.

`repair_count` is retained at 0 for record-shape compatibility with the MRS-1
analysis functions: MRS-2 registers NO deterministic parser repair at all, so any
nonzero value would be a defect.
"""
from .core import ANSWERS


def parse(text, role, terminated=True):
    result = {"parsed": {}, "format_failure": False, "repair_count": 0,
              "terminated": terminated, "nontermination": not terminated}
    try:
        if not terminated:
            raise ValueError("nontermination")
        token = text.strip()
        if token not in ANSWERS:
            raise ValueError("not a bare in-vocabulary answer")
        result["parsed"] = {"answer": token}
    except (ValueError, TypeError) as exc:
        result.update(format_failure=True, parse_error=str(exc))
    return result


def answer(record):
    return record["parsed"].get("answer") if not record["format_failure"] and record["terminated"] else None


def majority(records):
    from collections import Counter
    values = [answer(r) for r in records]
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    largest = max(counts.values())
    # Registered tie policy, carried from MRS-1: earliest valid role in the frozen order.
    return next(v for v in values if v is not None and counts[v] == largest)
