"""Offline enrollment/decision CLI. No native execution entry point exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from measurement_enrollment import check_enrollment, freeze_enrollment
from quality_decision import evaluate_quality
from readiness_packet import parse_json, read_selected, safe_path, seal, validate_quality_policy

ROOT = Path(__file__).resolve().parent


def _load(relative):
    raw, ref = read_selected(ROOT, relative)
    return parse_json(raw), ref


def _save(relative, value):
    path = safe_path(ROOT, relative, must_exist=False)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    _, ref = read_selected(ROOT, relative)
    return ref


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("enroll", "decide"))
    parser.add_argument("--cohort", required=True, help="Relative path inside external handoff directory")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--policy", default="QUALITY_SCREEN_POLICY_001.json")
    parser.add_argument("--candidates")
    parser.add_argument("--enrollment")
    parser.add_argument("--enrollment-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        cohort, cohort_ref = _load(args.cohort)
        baseline, baseline_ref = _load(args.baseline)
        policy, policy_ref = _load(args.policy)
        validate_quality_policy(policy)
        required_sources = list(policy["required_retention_sources"])
        if any(row["role"] == "retention" and row["source"] == "long_context" for row in cohort["rows"]):
            required_sources.append("long_context")
        if args.operation == "enroll":
            if args.candidates or args.enrollment or args.enrollment_sha256:
                raise ValueError("enrollment freezes M0 before candidate inputs; no candidate/enrollment options")
            value = freeze_enrollment(cohort, baseline, baseline_id=args.baseline_id, required_sources=required_sources)
        else:
            if not args.candidates or not args.enrollment or not args.enrollment_sha256:
                raise ValueError("decision requires candidates and independently selected frozen enrollment file hash")
            raw, enrollment_ref = read_selected(ROOT, args.enrollment, expected_sha256=args.enrollment_sha256)
            check_enrollment(parse_json(raw), cohort, baseline, baseline_id=args.baseline_id, required_sources=required_sources)
            candidates, candidates_ref = _load(args.candidates)
            result = evaluate_quality(baseline, candidates, cohort, policy)
            value = seal({"schema": "synth-offline-quality-decision-v1", "result": result,
                "inputs": {"cohort": cohort_ref, "baseline": baseline_ref, "policy": policy_ref,
                           "candidates": candidates_ref, "enrollment": enrollment_ref},
                "native_grade_import_admitted": False, "native_execution_authorized": False,
                "training_authorized": False, "confirmation_authorized": False, "promotion_authorized": False})
        ref = _save(args.output, value)
        print(json.dumps({"status": "OFFLINE_COMPUTATION_ONLY", "file": ref,
                          "native_execution_authorized": False}, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "BLOCKED_OR_INVALID_INPUT", "reason": str(error),
                          "native_execution_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
