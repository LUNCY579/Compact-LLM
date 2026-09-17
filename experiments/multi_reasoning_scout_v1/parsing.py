"""No substring extraction, label-based repair, coercion or fence stripping."""
from .core import ANSWERS, decode


def parse(text, role, terminated=True):
    result = {"parsed": {}, "format_failure": False, "repair_count": 0,
              "terminated": terminated, "nontermination": not terminated}
    try:
        data = decode(text.strip())
        allowed = {"answer", "state", "sufficient_evidence", "candidate_valid", "issues"}
        if not isinstance(data, dict) or not set(data) <= allowed or data.get("answer") not in ANSWERS:
            raise ValueError("invalid answer schema")
        if "state" in data and (not isinstance(data["state"], str) or len(data["state"].split()) > 20):
            raise ValueError("state is not a short string")
        for key in ("sufficient_evidence", "candidate_valid"):
            if key in data and type(data[key]) is not bool:
                raise ValueError("checker flag must be boolean")
        if "issues" in data and (not isinstance(data["issues"], list)
                                  or any(not isinstance(x, str) for x in data["issues"])):
            raise ValueError("issues must be strings")
        required = {"R2": "sufficient_evidence", "R3": "candidate_valid",
                    "M2_check": "sufficient_evidence", "M2_constraint": "candidate_valid"}.get(role)
        if required and (required not in data or "issues" not in data):
            raise ValueError("checker fields absent")
        result["parsed"] = data
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
    # Registered tie policy: earliest valid role in the frozen order.
    return next(v for v in values if v is not None and counts[v] == largest)
