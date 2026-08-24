#!/usr/bin/env python3
"""Audit the prospective endpoint dataset and B0 stop decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import FrozenEvidenceAnnotation  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(output: Path) -> dict[str, Any]:
    dataset = LAB_ROOT / "results" / "benchmarks" / "halo_endpoint_validation_v1"
    capability = LAB_ROOT / "results" / "runs" / "halo_endpoint_validation_capability_v1"
    inference = _jsonl(dataset / "inference.jsonl")
    evaluation = _jsonl(dataset / "evaluation.jsonl")
    if len(inference) != 24 or len(evaluation) != 24:
        raise RuntimeError("prospective endpoint views must each contain 24 rows")
    if [row["case_id"] for row in inference] != [row["case_id"] for row in evaluation]:
        raise RuntimeError("prospective endpoint views do not align by case_id")
    forbidden = {"answer", "accepted_answers", "oracle", "evidence_annotation"}
    if any(forbidden & set(row) for row in inference):
        raise RuntimeError("prospective inference view exposes answer/evidence fields")
    for prompt, key in zip(inference, evaluation, strict=True):
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
        if annotation.prompt_sha256 != prompt["prompt_sha256"]:
            raise RuntimeError("prospective evidence annotation prompt hash mismatch")
    clusters = {str(row["cluster_id"]) for row in inference}
    prior_clusters: set[str] = set()
    for path in (
        LAB_ROOT / "results" / "benchmarks" / "halo_calibration_v3" / "inference.jsonl",
        LAB_ROOT / "results" / "benchmarks" / "halo_r1_r8_ladder_v1" / "inference.jsonl",
    ):
        prior_clusters.update(str(row["cluster_id"]) for row in _jsonl(path))
    overlap = sorted(clusters & prior_clusters)
    if overlap:
        raise RuntimeError(f"prospective clusters overlap prior endpoint/ladder data: {overlap}")
    summary = json.loads((capability / "summary.json").read_text(encoding="utf-8"))
    cases = _jsonl(capability / "cases.jsonl")
    if len(cases) != 24 or bool(summary["advance_gate_passed"]):
        raise RuntimeError("prospective capability gate state is inconsistent")
    compressed_candidates = [
        path
        for path in (LAB_ROOT / "results" / "runs").glob("halo_endpoint_validation_v1_*")
        if path.name != capability.name
    ]
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "PROSPECTIVE_ENDPOINT_VALIDATION_STOPPED_AT_B0_GATE",
        "dataset": {
            "cases": len(inference),
            "independent_clusters": len(clusters),
            "cluster_overlap_with_prior_data": overlap,
            "inference_sha256": sha256_file(dataset / "inference.jsonl"),
            "evaluation_sha256": sha256_file(dataset / "evaluation.jsonl"),
            "manifest_sha256": sha256_file(dataset / "manifest.json"),
        },
        "capability": {
            "correct": summary["correct_clusters"],
            "clusters": summary["independent_clusters"],
            "by_family": summary["by_family"],
            "advance_rule": summary["advance_rule"],
            "advance_gate_passed": summary["advance_gate_passed"],
            "summary_sha256": sha256_file(capability / "summary.json"),
            "cases_sha256": sha256_file(capability / "cases.jsonl"),
        },
        "compressed_endpoint_run_present": bool(compressed_candidates),
        "compressed_endpoint_run_paths": [path.name for path in compressed_candidates],
        "decision": "STOP_AND_REDESIGN_TASK_DIFFICULTY_WITH_NEW_VERSION",
        "claim_limit": (
            "Only independent dataset construction and full-cache capability were evaluated. "
            "No B2/B3/B4 endpoint outcome was generated."
        ),
    }
    if result["compressed_endpoint_run_present"]:
        raise RuntimeError("a compressed endpoint run exists despite the failed B0 gate")
    atomic_write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "results" / "statistics" / "halo_endpoint_validation_gate_v1.json",
    )
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
