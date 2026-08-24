#!/usr/bin/env python3
"""Verify registered endpoint-validation v2 execution against its freeze."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit(output: Path) -> dict[str, Any]:
    registration_path = (
        LAB_ROOT / "results" / "registrations" / "halo_endpoint_validation_v2.json"
    )
    config_path = LAB_ROOT / "configs" / "halo_endpoint_validation_v2_registered.yaml"
    run = LAB_ROOT / "results" / "runs" / "halo_endpoint_validation_v2_registered"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((run / "statistics.json").read_text(encoding="utf-8"))
    cases = _jsonl(run / "cases.jsonl")
    registered_ids = {str(value) for value in registration["selected_case_ids"]}
    observed_ids = {str(row["case_id"]) for row in cases}
    observed_conditions = {str(row["condition"]) for row in cases}
    per_case = {
        case_id: sorted(str(row["condition"]) for row in cases if row["case_id"] == case_id)
        for case_id in sorted(observed_ids)
    }
    source_paths = {
        "dataset_manifest": LAB_ROOT
        / "results"
        / "registrations"
        / "source_snapshots"
        / "halo_endpoint_validation_v2_dataset_manifest.preseal.json",
        "inference": LAB_ROOT
        / "results"
        / "benchmarks"
        / "halo_endpoint_validation_v2"
        / "inference.jsonl",
        "evaluation": LAB_ROOT
        / "results"
        / "benchmarks"
        / "halo_endpoint_validation_v2"
        / "evaluation.jsonl",
        "capability_manifest": LAB_ROOT
        / "results"
        / "registrations"
        / "source_snapshots"
        / "halo_endpoint_validation_v2_capability_manifest.preseal.json",
        "capability_summary": LAB_ROOT
        / "results"
        / "runs"
        / "halo_endpoint_validation_v2_capability"
        / "summary.json",
        "capability_cases": LAB_ROOT
        / "results"
        / "runs"
        / "halo_endpoint_validation_v2_capability"
        / "cases.jsonl",
    }
    current_source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    checks = {
        "registration_precedes_run": _time(registration["registered_at"])
        < _time(manifest["created_at"]),
        "registered_config_hash_exact": registration["registered_config_sha256"]
        == sha256_file(config_path),
        "registered_source_hashes_exact": registration["source_hashes"]
        == current_source_hashes,
        "selected_case_ids_exact": registered_ids == observed_ids,
        "conditions_exact": observed_conditions == {"B2", "B3", "B4"},
        "three_conditions_per_case": all(
            values == ["B2", "B3", "B4"] for values in per_case.values()
        ),
        "expected_row_count": len(cases) == 36,
        "primary_endpoint_exact": statistics["primary_endpoint"]
        == registration["primary_endpoint"]["primary"],
        "primary_contrast_present": "B3_minus_B2_primary_endpoint"
        in statistics["comparisons"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"registered execution audit failed: {checks}")
    primary = statistics["comparisons"]["B3_minus_B2_primary_endpoint"]
    answer = statistics["comparisons"]["B3_minus_B2_answer_correct"]
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "REGISTERED_LOCAL_PROSPECTIVE_ENDPOINT_VALIDATION_COMPLETE",
        "registration_id": registration["registration_id"],
        "checks": checks,
        "source_hashes": current_source_hashes,
        "source_manifest_snapshot_role": (
            "exact pre-integrity-seal bytes hashed by the registration; raw inference, "
            "evaluation, capability summary, and cases remain live source files"
        ),
        "run_artifacts": {
            "manifest_sha256": sha256_file(run / "manifest.json"),
            "cases_sha256": sha256_file(run / "cases.jsonl"),
            "statistics_sha256": sha256_file(run / "statistics.json"),
            "summary_sha256": sha256_file(run / "summary.json"),
        },
        "primary_result": primary,
        "secondary_answer_result": answer,
        "interpretation": (
            "Frozen endpoint v1 prospectively separates B3 from B2 on this locally "
            "registered 12-cluster set; answer accuracy does not improve."
        ),
        "claim_limit": (
            "Local registration, difficulty-calibrated templates, and 12 clusters; not a "
            "publication preregistration or broad population claim."
        ),
    }
    atomic_write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT
        / "results"
        / "registrations"
        / "halo_endpoint_validation_v2_execution.json",
    )
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
