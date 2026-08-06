from __future__ import annotations

"""Select a warm SFT diagnostic candidate using Oracle capability evidence."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.phase_9_student_distillation.grounded_recovery_metrics import (
    capability_components,
    capability_score,
    behavior_coverage_issues,
    behavior_success_issues,
    integrity_issues,
    nondegeneracy_issues,
)


def candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be name=summary.json")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank grounded-recovery warm SFT candidates.")
    parser.add_argument("--initialization-summary", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=candidate_spec, default=[], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialization = read_json(args.initialization_summary)
    init_score = capability_score(initialization)
    initialization_issues = sorted(set([
        *integrity_issues(initialization),
        *behavior_coverage_issues(initialization),
    ]))
    rows: list[dict[str, Any]] = []
    for name, path in args.candidate:
        summary = read_json(path)
        score = capability_score(summary)
        issues = [
            *(["initialization_reference_invalid"] if initialization_issues else []),
            *integrity_issues(summary),
            *behavior_coverage_issues(summary),
            *nondegeneracy_issues(summary),
            *behavior_success_issues(summary),
        ]
        improvement = None if init_score is None or score is None else score - init_score
        if improvement is None or improvement <= 0.0:
            issues.append("no_oracle_capability_improvement_over_initialization")
        packet = (summary.get("conditions") or {}).get("oracle_rag") or {}
        rows.append({
            "name": name,
            "summary": str(path),
            "model": summary.get("model"),
            "adapter": summary.get("adapter"),
            "capability_score": score,
            "initialization_score": init_score,
            "improvement": improvement,
            "components": capability_components(summary),
            "issues": sorted(set(issues)),
            "valid": not issues,
            "unsupported_confident_answer_rate": packet.get("unsupported_confident_answer_rate"),
            "median_completion_tokens": packet.get("median_completion_tokens"),
            "mean_generation_seconds": packet.get("mean_generation_seconds"),
        })
    valid = [row for row in rows if row["valid"]]
    all_scores = [row["capability_score"] for row in valid if row["capability_score"] is not None]
    if not valid or not all_scores or max(all_scores) <= 0.0:
        selected = None
        status = "NO_VALID_QUALITY_FRONTIER"
    else:
        selected = max(
            valid,
            key=lambda row: (
                float(row["capability_score"]),
                float(row["improvement"] or 0.0),
                -float(row["unsupported_confident_answer_rate"] or 0.0),
                -float(row["median_completion_tokens"] or 0.0),
                -float(row["mean_generation_seconds"] or 0.0),
            ),
        )
        status = "SFT_CANDIDATE_SELECTED"
    packet = {
        "phase": "9_student_distillation",
        "schema_version": "1.3",
        "pipeline_revision": "phase9_grounded_recovery_v1_7_2",
        "status": status,
        "initialization_summary": str(args.initialization_summary),
        "initialization_issues": initialization_issues,
        "candidates": rows,
        "selected": selected,
        "selection_policy": "valid Oracle capability improvement first; no ranking when all candidates are degenerate",
        "human_decision": "PENDING",
    }
    write_json(args.out, packet)
    lines = ["# Phase 9 warm-SFT diagnostic sweep", "", f"- Status: `{status}`", ""]
    for row in rows:
        lines.extend([
            f"## {row['name']}",
            "",
            f"- Capability score: `{row['capability_score']}`",
            f"- Improvement over initialization: `{row['improvement']}`",
            f"- Valid: `{row['valid']}`",
            f"- Issues: `{row['issues'] or 'none'}`",
            "",
        ])
    lines.extend(["## Selected", "", f"`{selected['name'] if selected else 'none'}`", ""])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(packet, indent=2))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
