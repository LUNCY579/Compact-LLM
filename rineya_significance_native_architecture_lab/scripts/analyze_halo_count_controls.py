#!/usr/bin/env python3
"""Analyze count controls and rebuild causally valid boundary evidence rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import FrozenEvidenceAnnotation  # noqa: E402
from rineya_significance_lab.evaluation.boundary_evidence import (  # noqa: E402
    score_observed_boundary_evidence,
)
from rineya_significance_lab.evaluation.count_causal import (  # noqa: E402
    analyze_count_controls,
)
from rineya_significance_lab.evaluation.failure_taxonomy import (  # noqa: E402
    compressed_retention_failure_labels,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402
from rineya_significance_lab.runtime import boundary_resident_positions  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _boundary_rows(
    unscored: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Mapping[str, Any]],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in unscored:
        case_id = str(row["case_id"])
        annotation = FrozenEvidenceAnnotation.from_dict(evaluation[case_id]["evidence_annotation"])
        diagnostic_path = run_dir / str(row["boundary_diagnostics"])
        with np.load(diagnostic_path, allow_pickle=False) as arrays:
            diagnostics = {name: arrays[name].copy() for name in arrays.files}
        for raw_summary in row["boundary_summaries"]:
            summary = dict(raw_summary)
            resident = boundary_resident_positions(
                diagnostics,
                boundary_index=int(summary["boundary_index"]),
                layer_count=int(summary["layer_count"]),
            )
            scored = score_observed_boundary_evidence(
                annotation,
                resident,
                true_length=int(summary["true_length"]),
            )
            result.append(
                {
                    "schema_version": "1.0",
                    "case_id": case_id,
                    "cluster_id": row["cluster_id"],
                    "task_family": row["family"],
                    "condition": row["condition"],
                    "count_transform": row["count_transform"],
                    **summary,
                    "boundary_evidence": scored.to_dict(),
                    "annotation_sha256": _canonical_sha256(
                        evaluation[case_id]["evidence_annotation"]
                    ),
                }
            )
    return result


def _summarize_boundaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["boundary_evidence"]["status"]) for row in rows)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        payload = row["boundary_evidence"]
        evidence = payload["evidence"]
        if evidence is not None:
            grouped[(str(row["condition"]), int(row["boundary_index"]))].append(
                float(evidence["layer_fact_exact_completion_rate"])
            )
    by_condition_boundary: dict[str, dict[str, Any]] = defaultdict(dict)
    for (condition, boundary), values in sorted(grouped.items()):
        by_condition_boundary[condition][str(boundary)] = {
            "scored_rows": len(values),
            "mean_layer_fact_exact_completion_rate": sum(values) / len(values),
        }
    return {
        "rows": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "by_condition_and_boundary": dict(sorted(by_condition_boundary.items())),
        "future_or_boundary_crossing_facts_counted_as_evicted": False,
    }


def analyze(run_dir: Path, evaluation_path: Path) -> dict[str, Any]:
    run = run_dir.resolve(strict=True)
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise RuntimeError("count-control inference must be complete before analysis")
    cases = _jsonl(run / "cases.jsonl")
    for row in cases:
        row["failure_labels"] = compressed_retention_failure_labels(
            correct=bool(row["correct"]),
            evidence=row["evidence"],
        )
        row["failure_label_scope"] = (
            "descriptive_final_retained_state_not_causal_attribution"
        )
    _write_jsonl_atomic(run / "cases.jsonl", cases)
    unscored = _jsonl(run / "predictions.unscored.jsonl")
    evaluation = {str(row["case_id"]): row for row in _jsonl(evaluation_path.resolve(strict=True))}
    if {str(row["case_id"]) for row in unscored} - set(evaluation):
        raise RuntimeError("evaluation view is missing one or more completed cases")

    count_analysis = analyze_count_controls(cases, artifact_root=run)
    boundary_rows = _boundary_rows(unscored, evaluation, run_dir=run)
    _write_jsonl_atomic(run / "boundary_evidence.jsonl", boundary_rows)
    failures = Counter(
        label
        for row in cases
        for label in row.get("failure_labels", [])
    )
    failures_by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    for row in cases:
        failures_by_condition[str(row["condition"])].update(row.get("failure_labels", []))
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_COUNT_CAUSAL_ANALYSIS",
        "source": {
            "run_id": run.name,
            "cases_sha256": sha256_file(run / "cases.jsonl"),
            "unscored_predictions_sha256": sha256_file(run / "predictions.unscored.jsonl"),
            "boundary_diagnostics_index_sha256": sha256_file(run / "boundary_diagnostics.jsonl"),
            "evaluation_sha256": sha256_file(evaluation_path),
        },
        "count_causal": count_analysis,
        "control_validity": {
            "true_realized_count": "VALID_REFERENCE",
            "constant_count": "VALID_DENOMINATOR_ABLATION",
            "age_derived_count": "VALID_DETERMINISTIC_SCHEDULE_CONTROL",
            "head_averaged_count": "VALID_HEAD_SPECIFICITY_CONTROL",
            "delayed_count": "VALID_DELAYED_STATE_CONTROL",
            "noisy_count": "VALID_SEEDED_NOISE_CONTROL",
            "shuffled_count": (
                "ELIGIBLE_POOL_CONFOUNDED: global C multiset is preserved, but values cross "
                "protected sink/recent boundaries"
            ),
            "wrong_token_count": (
                "ELIGIBLE_POOL_CONFOUNDED: global circular permutation crosses protected "
                "sink/recent boundaries"
            ),
        },
        "boundary_evidence": {
            **_summarize_boundaries(boundary_rows),
            "artifact": "boundary_evidence.jsonl",
            "sha256": sha256_file(run / "boundary_evidence.jsonl"),
        },
        "failure_distribution": {
            "overall": dict(sorted(failures.items())),
            "by_condition": {
                condition: dict(sorted(values.items()))
                for condition, values in sorted(failures_by_condition.items())
            },
        },
        "claim_limit": (
            "Exploratory, B0-selected calibration analysis. The age/head equivalence is exact "
            "for this schedule; no population, answer-quality, or confirmatory claim is made."
        ),
    }
    atomic_write_json(run / "statistics.json", result)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})["statistics"] = "statistics.json"
    manifest["artifacts"]["boundary_evidence"] = "boundary_evidence.jsonl"
    note = (
        "Boundary evidence excludes required facts not fully observed at that boundary; "
        "future facts are never labeled evicted."
    )
    if note not in manifest.setdefault("notes", []):
        manifest["notes"].append(note)
    atomic_write_json(manifest_path, manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v1",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=LAB_ROOT / "results" / "benchmarks" / "halo_calibration_v3" / "evaluation.jsonl",
    )
    args = parser.parse_args()
    result = analyze(args.run, args.evaluation)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
