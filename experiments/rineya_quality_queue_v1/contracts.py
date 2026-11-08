"""Offline evidence and preservation contracts; no execution permission is granted."""

from __future__ import annotations

import math
from typing import Any


QUALITY_EVIDENCE_CLASSES = frozenset(
    {"historical", "native", "exploratory", "confirmed"}
)
METRIC_UNITS = frozenset({"count", "rate", "tokens", "seconds"})
GATE_STATUSES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_MEASURED"})
REQUIRED_GATES = (
    "acquisition",
    "retention",
    "reliable_answering",
    "termination",
    "response_length",
    "native_qualification",
    "independent_replication",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    # JSON numbers are ints or floats; booleans must not become numeric evidence.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _quality_evidence(evidence_id: Any, classifications: dict[str, str]) -> None:
    _require(_nonempty_text(evidence_id), "evidence_id must be nonempty")
    _require(evidence_id in classifications, "unknown evidence_id: " + evidence_id)
    classification = classifications[evidence_id]
    _require(
        isinstance(classification, str) and classification in QUALITY_EVIDENCE_CLASSES,
        "fixture or preparation evidence cannot support a quality measurement",
    )


def validate_metric(metric: dict[str, Any], evidence_classifications: dict[str, str]) -> None:
    """Validate measurement structure; source pointer/anchor resolution is external.

    Unmeasured values must be explicitly null, preserving unknown denominators.
    Historical/native labels describe evidence scope; this check does not attest
    that a model actually ran or that its result generalizes.
    """
    _require(isinstance(metric, dict), "metric must be an object")
    _require(isinstance(evidence_classifications, dict), "evidence classifications must be an object")
    _require(_nonempty_text(metric.get("name")), "metric name must be nonempty")
    _require(metric.get("status") in ("MEASURED", "NOT_MEASURED"), "invalid metric status")
    _require("numerator" in metric and "denominator" in metric, "metric must declare numerator and denominator")

    if metric["status"] == "NOT_MEASURED":
        _require(
            metric["numerator"] is None and metric["denominator"] is None,
            "NOT_MEASURED requires null numerator and denominator",
        )
        return

    numerator = metric["numerator"]
    denominator = metric["denominator"]
    _require(_finite_number(numerator) and numerator >= 0, "numerator must be finite and nonnegative")
    _require(_finite_number(denominator) and denominator > 0, "denominator must be finite and positive")
    _require(isinstance(metric.get("unit"), str) and metric["unit"] in METRIC_UNITS, "invalid metric unit")
    if metric["unit"] == "count":
        _require(numerator <= denominator, "count numerator exceeds denominator")

    _quality_evidence(metric.get("evidence_id"), evidence_classifications)
    _require(
        _nonempty_text(metric.get("json_pointer")) or _nonempty_text(metric.get("text_anchor")),
        "MEASURED requires a nonempty json_pointer or text_anchor",
    )
    if "denominator_evidence_id" in metric:
        _quality_evidence(metric["denominator_evidence_id"], evidence_classifications)
    for field in ("denominator_json_pointer", "denominator_text_anchor"):
        if field in metric:
            _require(_nonempty_text(metric[field]), field + " must be nonempty")


def preservation_decision(gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Require all common and registered extra gates; never pool a tradeoff.

    Eligibility means only that the supplied statuses permit a separate review.
    This function grants no native execution, release, or promotion permission.
    """
    _require(isinstance(gates, list), "gates must be a list")
    gate_ids: set[str] = set()
    blocking_gates: list[str] = []
    for gate in gates:
        _require(isinstance(gate, dict), "gate must be an object")
        gate_id = gate.get("id")
        _require(_nonempty_text(gate_id), "gate id must be nonempty")
        _require(gate_id not in gate_ids, "duplicate gate id: " + gate_id)
        _require(
            isinstance(gate.get("status"), str) and gate["status"] in GATE_STATUSES,
            "invalid gate status: " + gate_id,
        )
        gate_ids.add(gate_id)
        if gate["status"] != "PASS":
            blocking_gates.append(gate_id)

    missing_gates = [gate_id for gate_id in REQUIRED_GATES if gate_id not in gate_ids]
    return {
        "status": "HOLD" if blocking_gates or missing_gates else "ELIGIBLE_FOR_SEPARATE_REVIEW",
        "blocking_gates": blocking_gates,
        "missing_gates": missing_gates,
        "native_execution_authorized": False,
        "release_permitted": False,
        "promotion_permitted": False,
    }
