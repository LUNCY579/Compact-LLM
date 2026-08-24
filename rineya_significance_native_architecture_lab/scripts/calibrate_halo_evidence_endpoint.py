#!/usr/bin/env python3
"""Freeze and screen an answer-blind evidence-retention endpoint definition."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import cast

import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.evidence_endpoints import (  # noqa: E402
    EVIDENCE_RETENTION_ENDPOINT_VERSION,
    PRIMARY_ENDPOINT_NAME,
    EvidenceRetentionEndpointScore,
    score_serialized_evidence_retention,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402

CONFIG_PATH = LAB_ROOT / "configs" / "halo_evidence_endpoint_v1.yaml"
OUTPUT_PATH = LAB_ROOT / "results" / "statistics" / "halo_evidence_endpoint_calibration_v1.json"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        raise ValueError("cannot summarize an empty endpoint vector")
    ordered = sorted(values)
    return {
        "count": len(values),
        "distinct_value_count": len(set(values)),
        "minimum": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "maximum": ordered[-1],
        "fraction_at_zero": sum(value == 0.0 for value in values) / len(values),
        "fraction_at_one": sum(value == 1.0 for value in values) / len(values),
    }


def _screen(summary: Mapping[str, object], criteria: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "minimum_distinct_values": int(summary["distinct_value_count"])
        >= int(criteria["minimum_distinct_values"]),
        "maximum_fraction_at_zero": float(summary["fraction_at_zero"])
        <= float(criteria["maximum_fraction_at_zero"]),
        "maximum_fraction_at_one": float(summary["fraction_at_one"])
        <= float(criteria["maximum_fraction_at_one"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _source_path(config: Mapping[str, object]) -> Path:
    calibration = _mapping(config["calibration"], name="calibration")
    path = (REPOSITORY_ROOT / str(calibration["source_cases"])).resolve()
    if REPOSITORY_ROOT.resolve() not in path.parents:
        raise ValueError("calibration source must remain inside the current repository")
    return path


def calibrate(config_path: Path = CONFIG_PATH) -> dict[str, object]:
    """Calibrate from evidence mappings only; policy labels and answers are not accessed."""

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(loaded, name="config")
    endpoint = _mapping(config["endpoint"], name="endpoint")
    if endpoint.get("version") != EVIDENCE_RETENTION_ENDPOINT_VERSION:
        raise ValueError("config endpoint version does not match the scoring implementation")
    primary = _mapping(endpoint["primary"], name="endpoint.primary")
    if primary.get("name") != PRIMARY_ENDPOINT_NAME:
        raise ValueError("config primary endpoint does not match the scoring implementation")

    calibration = _mapping(config["calibration"], name="calibration")
    source = _source_path(config)
    observed_source_hash = sha256_file(source)
    if observed_source_hash != calibration["expected_source_sha256"]:
        raise ValueError("calibration source hash does not match the frozen config")

    scores: list[EvidenceRetentionEndpointScore] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), name=f"source row {line_number}")
            evidence = _mapping(row.get("evidence"), name=f"source row {line_number}.evidence")
            scores.append(score_serialized_evidence_retention(evidence))

    metric_values = {
        "layer_fact_exact_completion_rate": [
            score.layer_fact_exact_completion_rate for score in scores
        ],
        "layer_fact_macro_token_recall": [
            score.layer_fact_macro_token_recall for score in scores
        ],
        "layer_balanced_minimum_fact_token_recall": [
            score.layer_balanced_minimum_fact_token_recall for score in scores
        ],
        "minimum_layer_fact_token_recall": [
            score.minimum_layer_fact_token_recall for score in scores
        ],
        "strict_all_required_fact_tokens_available_in_every_evaluated_layer": [
            float(score.strict_all_required_fact_tokens_available_in_every_evaluated_layer)
            for score in scores
        ],
    }
    criteria = _mapping(
        calibration["non_saturation_screen"],
        name="calibration.non_saturation_screen",
    )
    summaries = {name: _summary(values) for name, values in metric_values.items()}
    screens = {name: _screen(summary, criteria) for name, summary in summaries.items()}
    primary_passed = bool(screens[PRIMARY_ENDPOINT_NAME]["passed"])
    strict_name = "strict_all_required_fact_tokens_available_in_every_evaluated_layer"

    permitted = list(cast(Sequence[object], calibration["permitted_fields"]))
    forbidden = list(cast(Sequence[object], calibration["forbidden_for_endpoint_selection"]))
    payload = {
        "schema_version": "1.0",
        "artifact_id": "halo_evidence_endpoint_calibration_v1",
        "status": "PASS" if primary_passed else "FAIL_NON_SATURATION_SCREEN",
        "classification": "EXPLORATORY_OUTCOME_BLIND_ENDPOINT_CALIBRATION",
        "endpoint_freeze": {
            "version": EVIDENCE_RETENTION_ENDPOINT_VERSION,
            "primary_endpoint": PRIMARY_ENDPOINT_NAME,
            "definition_sha256": _canonical_sha256(endpoint),
            "config_sha256": sha256_file(config_path),
            "scope": "future evaluations only",
            "confirmatory_status": "NOT_REGISTERED",
        },
        "source": {
            "path": source.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": observed_source_hash,
            "evidence_record_count": len(scores),
            "pooling": calibration["pooling"],
        },
        "selection_blinding_audit": {
            "fields_accessed": ["evidence"],
            "permitted_evidence_subfields": permitted,
            "fields_forbidden_and_not_accessed": forbidden,
            "condition_labels_used": False,
            "answer_outcomes_used": False,
        },
        "non_saturation_criteria": dict(criteria),
        "candidate_summaries": summaries,
        "candidate_screens": screens,
        "decision": {
            "selected_primary": PRIMARY_ENDPOINT_NAME if primary_passed else None,
            "strict_diagnostic_retained": True,
            "strict_diagnostic_screen_passed": bool(screens[strict_name]["passed"]),
            "rationale": (
                "Exact fact completion is averaged over layer/fact cells, preserving an "
                "atomic and threshold-free definition while avoiding the saturated global "
                "conjunction. Partial-token candidates remain sensitivity diagnostics."
            ),
        },
        "claim_limit": (
            "This freezes an exploratory retention endpoint using existing pilot traces; "
            "it is not a confirmatory result and was not selected using answer outcomes."
        ),
    }
    return payload


def main() -> int:
    payload = calibrate()
    atomic_write_json(OUTPUT_PATH, payload)
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT_PATH)}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

