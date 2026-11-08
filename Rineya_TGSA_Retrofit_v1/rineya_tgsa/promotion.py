"""Fail-closed validation for production-retrofit candidate records.

This module deliberately does not decide whether research is positive. It
checks that an operator has recorded that decision and prevents a candidate
from being labelled ``PROMOTE`` unless every production gate is explicitly
true. Missing evidence is not truthy evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

LEDGER_SCHEMA_VERSION = "rineya.retrofit-candidate-ledger.v1"

MANDATORY_PROMOTION_GATES = (
    "positive_source_evidence",
    "parent_identity_verified",
    "conversion_fidelity_pass",
    "baked_effect_verified",
    "strict_quality_gain",
    "no_blocking_regression",
    "production_budget_pass",
    "artifact_integrity_pass",
    "demo_smoke_pass",
    "provenance_complete",
)

CLASSIFICATIONS = {
    "ELIGIBLE",
    "WAITING_FOR_EVIDENCE",
    "NEGATIVE",
    "INCOMPATIBLE",
    "ALREADY_INTEGRATED",
}
DECISIONS = {"PROMOTE", "REJECT", "DEFER", "ALREADY_INTEGRATED"}

REQUIRED_CANDIDATE_FIELDS = {
    "id",
    "candidate",
    "classification",
    "source_branch",
    "source_commit",
    "research_status",
    "research_evidence",
    "parent_checkpoint",
    "parent_hash",
    "retrofit_configuration",
    "parent_score",
    "retrofit_score",
    "delta",
    "fidelity_status",
    "regression_status",
    "latency_delta",
    "throughput_delta",
    "memory_delta",
    "artifact_hash",
    "demo_status",
    "gates",
    "decision",
    "reason",
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def promotion_failures(candidate: Mapping[str, Any]) -> list[str]:
    """Return every reason the strict promotion predicate is not satisfied."""

    gates = candidate.get("gates")
    if not isinstance(gates, Mapping):
        return ["gates must be an object"]
    failures = [name for name in MANDATORY_PROMOTION_GATES if gates.get(name) is not True]
    extras = sorted(set(gates) - set(MANDATORY_PROMOTION_GATES))
    if extras:
        failures.append(f"unknown promotion gates: {', '.join(extras)}")
    return failures


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_candidate(candidate: Mapping[str, Any]) -> list[str]:
    """Validate one candidate without mutating it."""

    errors: list[str] = []
    missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(candidate))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
        return errors

    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not candidate_id:
        errors.append("id must be a non-empty string")
    classification = candidate.get("classification")
    if classification not in CLASSIFICATIONS:
        errors.append(f"unsupported classification: {classification!r}")
    decision = candidate.get("decision")
    if decision not in DECISIONS:
        errors.append(f"unsupported decision: {decision!r}")

    commit = candidate.get("source_commit")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        errors.append("source_commit must be an immutable 40-character lowercase git SHA")
    evidence = candidate.get("research_evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(value, str) and value for value in evidence
    ):
        errors.append("research_evidence must be a non-empty list of artifact paths")

    for field in ("parent_hash", "artifact_hash"):
        value = candidate.get(field)
        if value is not None and (not isinstance(value, str) or not _HASH_RE.fullmatch(value)):
            errors.append(f"{field} must be null or a 64-character lowercase SHA-256")

    scores = [candidate.get(name) for name in ("parent_score", "retrofit_score", "delta")]
    if any(value is not None for value in scores):
        if not all(_is_finite_number(value) for value in scores):
            errors.append("parent_score, retrofit_score, and delta must be all finite numbers or all null")
        elif not math.isclose(scores[1] - scores[0], scores[2], rel_tol=0.0, abs_tol=1e-12):
            errors.append("delta does not equal retrofit_score - parent_score")

    gate_failures = promotion_failures(candidate)
    if decision == "PROMOTE":
        if classification != "ELIGIBLE":
            errors.append("PROMOTE requires classification ELIGIBLE")
        if gate_failures:
            errors.append("PROMOTE blocked by: " + ", ".join(gate_failures))
        if not _is_finite_number(candidate.get("delta")) or candidate["delta"] <= 0:
            errors.append("PROMOTE requires a finite, strictly positive quality delta")
        if candidate.get("artifact_hash") is None:
            errors.append("PROMOTE requires an artifact_hash")
    elif classification == "ELIGIBLE" and not gate_failures:
        errors.append("an ELIGIBLE candidate with every gate true must be recorded as PROMOTE")

    if classification == "NEGATIVE" and decision != "REJECT":
        errors.append("NEGATIVE research must be recorded as REJECT")
    if classification in {"WAITING_FOR_EVIDENCE", "INCOMPATIBLE"} and decision != "DEFER":
        errors.append(f"{classification} must be recorded as DEFER")
    if classification == "ALREADY_INTEGRATED" and decision != "ALREADY_INTEGRATED":
        errors.append("ALREADY_INTEGRATED classification requires the matching decision")
    return errors


def validate_ledger(ledger: Mapping[str, Any]) -> list[str]:
    """Validate the complete machine-readable candidate ledger."""

    errors: list[str] = []
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {ledger.get('schema_version')!r}")
    candidates = ledger.get("candidates")
    if not isinstance(candidates, list):
        return errors + ["candidates must be a list"]

    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"candidates[{index}] must be an object")
            continue
        candidate_id = str(candidate.get("id", f"index-{index}"))
        if candidate_id in seen:
            errors.append(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        errors.extend(f"{candidate_id}: {error}" for error in validate_candidate(candidate))
    return errors
