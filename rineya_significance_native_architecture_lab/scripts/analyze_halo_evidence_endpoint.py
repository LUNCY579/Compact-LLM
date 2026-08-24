#!/usr/bin/env python3
"""Apply the frozen endpoint to the existing Halo pilot as post-hoc evidence only."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import cast

import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.clustered import (  # noqa: E402
    clustered_paired_analysis,
)
from rineya_significance_lab.evaluation.evidence_endpoints import (  # noqa: E402
    EVIDENCE_RETENTION_ENDPOINT_VERSION,
    PRIMARY_ENDPOINT_NAME,
    EvidenceRetentionEndpointScore,
    score_serialized_evidence_retention,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402

CONFIG_PATH = LAB_ROOT / "configs" / "halo_evidence_endpoint_v1.yaml"
CALIBRATION_PATH = (
    LAB_ROOT / "results" / "statistics" / "halo_evidence_endpoint_calibration_v1.json"
)
OUTPUT_PATH = LAB_ROOT / "results" / "statistics" / "halo_retention_endpoint_analysis_v1.json"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty endpoint vector")
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _metric(score: EvidenceRetentionEndpointScore, name: str) -> float:
    if name == "layer_fact_exact_completion_rate":
        return score.layer_fact_exact_completion_rate
    if name == "layer_fact_macro_token_recall":
        return score.layer_fact_macro_token_recall
    if name == "layer_balanced_minimum_fact_token_recall":
        return score.layer_balanced_minimum_fact_token_recall
    if name == "minimum_layer_fact_token_recall":
        return score.minimum_layer_fact_token_recall
    raise KeyError(f"unsupported endpoint metric: {name}")


def analyze(
    config_path: Path = CONFIG_PATH,
    calibration_path: Path = CALIBRATION_PATH,
) -> dict[str, object]:
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), name="config")
    endpoint = _mapping(config["endpoint"], name="endpoint")
    calibration_config = _mapping(config["calibration"], name="calibration")
    calibration_artifact = _mapping(
        json.loads(calibration_path.read_text(encoding="utf-8")),
        name="calibration artifact",
    )
    freeze = _mapping(calibration_artifact["endpoint_freeze"], name="endpoint_freeze")
    definition_sha256 = _canonical_sha256(endpoint)
    if endpoint.get("version") != EVIDENCE_RETENTION_ENDPOINT_VERSION:
        raise ValueError("endpoint version does not match the scoring implementation")
    if freeze.get("definition_sha256") != definition_sha256:
        raise ValueError("calibration artifact does not freeze the active endpoint definition")
    if calibration_artifact.get("status") != "PASS":
        raise ValueError("calibration artifact did not pass its non-saturation screen")

    source = (REPOSITORY_ROOT / str(calibration_config["source_cases"])).resolve()
    source_hash = sha256_file(source)
    if source_hash != calibration_config["expected_source_sha256"]:
        raise ValueError("pilot source hash does not match the frozen config")

    rows: list[tuple[str, str, EvidenceRetentionEndpointScore]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), name=f"source row {line_number}")
            case_id = str(row.get("case_id", ""))
            condition = str(row.get("condition", ""))
            if not case_id or not condition:
                raise ValueError("pilot application requires case_id and condition")
            evidence = _mapping(row.get("evidence"), name=f"source row {line_number}.evidence")
            rows.append((case_id, condition, score_serialized_evidence_retention(evidence)))

    expected_conditions = ("B2", "B3", "B4")
    indexed = {(case_id, condition): score for case_id, condition, score in rows}
    case_ids = sorted({case_id for case_id, _, _ in rows})
    for case_id in case_ids:
        missing = [condition for condition in expected_conditions if (case_id, condition) not in indexed]
        if missing:
            raise ValueError(f"case {case_id!r} is missing conditions {missing}")
    if len(indexed) != len(case_ids) * len(expected_conditions):
        raise ValueError("pilot contains duplicate or unexpected condition rows")

    metric_names = (
        PRIMARY_ENDPOINT_NAME,
        "layer_fact_macro_token_recall",
        "layer_balanced_minimum_fact_token_recall",
        "minimum_layer_fact_token_recall",
    )
    by_condition: dict[str, dict[str, object]] = defaultdict(dict)
    for condition in expected_conditions:
        for metric_name in metric_names:
            values = [_metric(indexed[(case_id, condition)], metric_name) for case_id in case_ids]
            by_condition[condition][metric_name] = _summary(values)
        by_condition[condition]["strict_case_success_count"] = sum(
            indexed[(case_id, condition)].strict_all_required_fact_tokens_available_in_every_evaluated_layer
            for case_id in case_ids
        )

    comparisons: dict[str, object] = {}
    for control in ("B2", "B4"):
        for metric_name in metric_names:
            result = clustered_paired_analysis(
                case_ids,
                [_metric(indexed[(case_id, control)], metric_name) for case_id in case_ids],
                [_metric(indexed[(case_id, "B3")], metric_name) for case_id in case_ids],
                minimum_clusters=8,
                bootstrap_samples=10_000,
                seed=17,
            )
            comparisons[f"B3_minus_{control}__{metric_name}"] = asdict(result)

    return {
        "schema_version": "1.0",
        "artifact_id": "halo_retention_endpoint_analysis_v1",
        "status": "PASS",
        "classification": "EXPLORATORY_POSTHOC_NO_ANSWER_OUTCOMES",
        "endpoint_freeze": {
            "version": EVIDENCE_RETENTION_ENDPOINT_VERSION,
            "primary_endpoint": PRIMARY_ENDPOINT_NAME,
            "definition_sha256": definition_sha256,
            "calibration_artifact_sha256": sha256_file(calibration_path),
        },
        "source": {
            "path": source.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": source_hash,
            "cluster_count": len(case_ids),
            "row_count": len(rows),
        },
        "analysis_access_audit": {
            "fields_accessed": ["case_id", "condition", "evidence"],
            "answer_outcomes_accessed": False,
            "correctness_comparisons_performed": False,
        },
        "by_condition": dict(by_condition),
        "clustered_paired_comparisons": comparisons,
        "claim_limit": (
            "Endpoint selection and these policy comparisons reuse the existing pilot. "
            "They are exploratory, are not confirmatory evidence, and do not show that "
            "retention mediated answer correctness."
        ),
    }


def main() -> int:
    payload = analyze()
    atomic_write_json(OUTPUT_PATH, payload)
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
