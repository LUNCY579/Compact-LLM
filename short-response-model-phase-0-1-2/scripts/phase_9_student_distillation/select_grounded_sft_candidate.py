from __future__ import annotations

"""Select a warm SFT diagnostic candidate using Oracle capability evidence.

v1.7.7 policy note
------------------
Stage B is an *untrained* warm-initialization baseline. Decode/prompt-integrity
failures observed on that weak baseline are valuable diagnostics, but they must
not poison otherwise valid Stage D SFT candidates. Candidate integrity remains
strict: every SFT candidate is judged on its own prompt/decode integrity,
coverage, non-degeneracy, behavior success, and capability improvement.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

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


SELECTION_POLICY_REVISION = "phase9_sft_selection_gate_v1_7_7"


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


def build_selection_packet(
    initialization: dict[str, Any],
    candidates: Iterable[tuple[str, Path, dict[str, Any]]],
    *,
    initialization_summary: str,
) -> dict[str, Any]:
    """Build the deterministic Stage D selection packet.

    Stage-B prompt/decode integrity failures are deliberately **warnings** here.
    They describe the untrained baseline and do not invalidate trained SFT
    candidates. Initialization *coverage* failures remain blocking because a
    missing behavior denominator makes the numeric reference unsuitable for a
    fair capability-improvement comparison.

    Every candidate is still rejected for its own integrity/coverage,
    degeneracy, behavior failure, or lack of improvement over initialization.
    """

    init_score = capability_score(initialization)
    initialization_integrity_warnings = sorted(set(integrity_issues(initialization)))
    initialization_blocking_issues = sorted(set(behavior_coverage_issues(initialization)))
    initialization_issues = sorted(
        set([*initialization_integrity_warnings, *initialization_blocking_issues])
    )

    rows: list[dict[str, Any]] = []
    for name, path, summary in candidates:
        score = capability_score(summary)
        issues = [
            *(["initialization_reference_invalid"] if initialization_blocking_issues else []),
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

    return {
        "phase": "9_student_distillation",
        "schema_version": "1.4",
        "pipeline_revision": "phase9_large_corrective_curriculum_v1_7_5",
        "selection_policy_revision": SELECTION_POLICY_REVISION,
        "status": status,
        "initialization_summary": initialization_summary,
        # Backward-compatible aggregate diagnostics. Do not interpret this field
        # alone as a selection veto; see blocking issues vs warnings below.
        "initialization_issues": initialization_issues,
        "initialization_blocking_issues": initialization_blocking_issues,
        "initialization_integrity_warnings": initialization_integrity_warnings,
        "initialization_integrity_is_selection_blocking": False,
        "candidates": rows,
        "selected": selected,
        "selection_policy": (
            "candidate-own integrity/coverage/nondegeneracy/behavior success + positive Oracle capability "
            "improvement; Stage-B prompt/decode integrity is diagnostic-only and cannot poison trained candidates; "
            "initialization behavior-coverage failures remain blocking"
        ),
        "human_decision": "PENDING",
    }


def render_report(packet: dict[str, Any]) -> str:
    status = packet["status"]
    init_warnings = packet.get("initialization_integrity_warnings") or []
    init_blockers = packet.get("initialization_blocking_issues") or []
    lines = [
        "# Phase 9 warm-SFT diagnostic sweep",
        "",
        f"- Status: `{status}`",
        f"- Selection policy revision: `{packet.get('selection_policy_revision')}`",
        "",
        "## Stage B initialization reference",
        "",
        f"- Blocking reference issues: `{init_blockers or 'none'}`",
        f"- Non-blocking integrity warnings: `{init_warnings or 'none'}`",
        "- Policy: Stage-B prompt/decode integrity failures are diagnostic warnings and do not invalidate trained SFT candidates.",
        "",
    ]
    for row in packet["candidates"]:
        lines.extend([
            f"## {row['name']}",
            "",
            f"- Capability score: `{row['capability_score']}`",
            f"- Improvement over initialization: `{row['improvement']}`",
            f"- Valid: `{row['valid']}`",
            f"- Issues: `{row['issues'] or 'none'}`",
            "",
        ])
    selected = packet.get("selected")
    lines.extend(["## Selected", "", f"`{selected['name'] if selected else 'none'}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    initialization = read_json(args.initialization_summary)
    candidates = [(name, path, read_json(path)) for name, path in args.candidate]
    packet = build_selection_packet(
        initialization,
        candidates,
        initialization_summary=str(args.initialization_summary),
    )
    write_json(args.out, packet)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(packet), encoding="utf-8")
    print(json.dumps(packet, indent=2))
    if packet["selected"] is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
