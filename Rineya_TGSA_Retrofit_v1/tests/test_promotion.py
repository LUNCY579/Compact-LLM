from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from rineya_tgsa.promotion import (
    MANDATORY_PROMOTION_GATES,
    promotion_failures,
    validate_candidate,
    validate_ledger,
)

LAB = Path(__file__).resolve().parents[1]


def _candidate() -> dict:
    return {
        "id": "candidate-a",
        "candidate": "Candidate A",
        "classification": "ELIGIBLE",
        "source_branch": "research/example",
        "source_commit": "1" * 40,
        "research_status": "GO",
        "research_evidence": ["evidence/result.json"],
        "parent_checkpoint": "parent",
        "parent_hash": "2" * 64,
        "retrofit_configuration": {"coefficient": 0.1},
        "parent_score": 10.0,
        "retrofit_score": 11.0,
        "delta": 1.0,
        "fidelity_status": "PASS",
        "regression_status": "PASS",
        "latency_delta": 0.0,
        "throughput_delta": 0.0,
        "memory_delta": 0.0,
        "artifact_hash": "3" * 64,
        "demo_status": "PASS",
        "gates": {name: True for name in MANDATORY_PROMOTION_GATES},
        "decision": "PROMOTE",
        "reason": "Every mandatory gate passed.",
    }


def test_promotion_is_a_strict_logical_and() -> None:
    candidate = _candidate()
    assert promotion_failures(candidate) == []
    assert validate_candidate(candidate) == []

    for gate in MANDATORY_PROMOTION_GATES:
        failed = deepcopy(candidate)
        failed["gates"][gate] = False
        errors = validate_candidate(failed)
        assert any(gate in error for error in errors), gate


def test_missing_gate_never_counts_as_pass() -> None:
    candidate = _candidate()
    del candidate["gates"]["artifact_integrity_pass"]
    errors = validate_candidate(candidate)
    assert any("artifact_integrity_pass" in error for error in errors)


def test_promotion_requires_a_strict_positive_delta() -> None:
    candidate = _candidate()
    candidate["retrofit_score"] = candidate["parent_score"]
    candidate["delta"] = 0.0
    errors = validate_candidate(candidate)
    assert any("strictly positive" in error for error in errors)


def test_negative_research_cannot_be_deferred_or_promoted() -> None:
    candidate = _candidate()
    candidate["classification"] = "NEGATIVE"
    candidate["decision"] = "DEFER"
    assert any("NEGATIVE research" in error for error in validate_candidate(candidate))


def test_current_ledger_passes_schema_and_semantic_validation() -> None:
    ledger = json.loads((LAB / "evidence" / "candidate_ledger.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (LAB / "schemas" / "candidate_ledger.schema.json").read_text(encoding="utf-8")
    )
    schema_errors = list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(ledger)
    )
    assert schema_errors == []
    assert validate_ledger(ledger) == []
    assert not any(candidate["decision"] == "PROMOTE" for candidate in ledger["candidates"])
