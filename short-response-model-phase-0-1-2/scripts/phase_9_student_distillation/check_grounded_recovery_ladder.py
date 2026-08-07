from __future__ import annotations

"""Assess the staged Phase 9 grounded-recovery capability ladder."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Phase 9 grounded-recovery ladder.")
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--initialization-summary", type=Path, required=True)
    parser.add_argument("--overfit-initialization-summary", type=Path, default=None)
    parser.add_argument("--overfit-trained-summary", type=Path, default=None)
    parser.add_argument("--overfit-training-manifest", type=Path, default=None)
    parser.add_argument("--sft-summary", type=Path, default=None)
    parser.add_argument("--minimum-overfit-score", type=float, default=0.80)
    parser.add_argument("--minimum-overfit-improvement", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--require-overfit-pass", action="store_true")
    parser.add_argument("--require-oracle-improvement", action="store_true")
    return parser.parse_args()


def summary_packet(path: Path | None) -> dict[str, Any] | None:
    return read_json(path) if path is not None and path.exists() else None


def stage_summary(name: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"stage": name, "status": "NOT_RUN", "score": None, "issues": ["summary_missing"]}
    issues = [*integrity_issues(summary)]
    score = capability_score(summary)
    return {
        "stage": name,
        "status": "PASS" if not issues and score is not None else "FAIL",
        "score": score,
        "components": capability_components(summary),
        "issues": issues,
        "summary_model": summary.get("model"),
        "summary_adapter": summary.get("adapter"),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_overfit_score <= 1.0:
        raise ValueError("--minimum-overfit-score must be in [0,1]")
    teacher = read_json(args.teacher_summary)
    initialization = read_json(args.initialization_summary)
    overfit_init = summary_packet(args.overfit_initialization_summary)
    overfit_trained = summary_packet(args.overfit_trained_summary)
    sft = summary_packet(args.sft_summary)

    stage_a = stage_summary("A_teacher_oracle", teacher)
    stage_a["issues"] = sorted(set([
        *stage_a["issues"],
        *behavior_coverage_issues(teacher),
        *nondegeneracy_issues(teacher),
        *behavior_success_issues(teacher),
    ]))
    stage_a["status"] = "PASS" if not stage_a["issues"] and stage_a["score"] is not None else "FAIL"
    stage_b = stage_summary("B_warm_initialization_oracle", initialization)
    stage_b["issues"] = sorted(set([*stage_b["issues"], *behavior_coverage_issues(initialization)]))
    stage_b["status"] = "BASELINE_RECORDED" if not stage_b["issues"] and stage_b["score"] is not None else "FAIL"

    stage_c: dict[str, Any] = {"stage": "C_tiny_overfit", "status": "NOT_RUN"}
    if overfit_init is not None and overfit_trained is not None:
        baseline_score = capability_score(overfit_init)
        trained_score = capability_score(overfit_trained)
        improvement = None if baseline_score is None or trained_score is None else trained_score - baseline_score
        issues = [
            *integrity_issues(overfit_trained),
            *behavior_coverage_issues(overfit_trained),
            *nondegeneracy_issues(overfit_trained),
            *behavior_success_issues(overfit_trained),
        ]
        manifest: dict[str, Any] = {}
        if args.overfit_training_manifest is not None and args.overfit_training_manifest.exists():
            manifest = read_json(args.overfit_training_manifest)
            if int(manifest.get("optimizer_steps") or 0) <= 0:
                issues.append("optimizer_steps_not_positive")
            if not manifest.get("artifact_tree_sha256"):
                issues.append("trained_artifact_hash_missing")
            if manifest.get("artifact_tree_sha256") == manifest.get("student_initialization_tree_sha256"):
                issues.append("trained_artifact_hash_matches_initialization")
        else:
            issues.append("overfit_training_manifest_missing")
        if trained_score is None or trained_score < args.minimum_overfit_score:
            issues.append("overfit_score_below_explicit_wiring_target")
        if improvement is None or improvement < args.minimum_overfit_improvement:
            issues.append("overfit_improvement_below_explicit_wiring_target")
        stage_c = {
            "stage": "C_tiny_overfit",
            "status": "PASS" if not issues else "FAIL",
            "baseline_score": baseline_score,
            "trained_score": trained_score,
            "improvement": improvement,
            "minimum_score": args.minimum_overfit_score,
            "minimum_improvement": args.minimum_overfit_improvement,
            "criterion_scope": "technical wiring test only; not a production quality threshold",
            "issues": sorted(set(issues)),
        }

    stage_d: dict[str, Any] = {"stage": "D_full_oracle_sft", "status": "NOT_RUN"}
    if sft is not None:
        init_score = capability_score(initialization)
        sft_score = capability_score(sft)
        improvement = None if init_score is None or sft_score is None else sft_score - init_score
        issues = [
            *integrity_issues(sft),
            *behavior_coverage_issues(sft),
            *nondegeneracy_issues(sft),
            *behavior_success_issues(sft),
        ]
        if improvement is None or improvement <= 0.0:
            issues.append("sft_did_not_improve_oracle_capability_over_initialization")
        stage_d = {
            "stage": "D_full_oracle_sft",
            "status": "PASS" if not issues else "FAIL",
            "initialization_score": init_score,
            "sft_score": sft_score,
            "improvement": improvement,
            "components": capability_components(sft),
            "issues": sorted(set(issues)),
        }

    stage_e_status = "UNLOCKED" if stage_d.get("status") == "PASS" else "BLOCKED"
    packet = {
        "phase": "9_student_distillation",
        "schema_version": "1.4",
        "pipeline_revision": "phase9_rag_reasoning_integration_v1_7_3",
        "stages": {
            "A": stage_a,
            "B": stage_b,
            "C": stage_c,
            "D": stage_d,
            "E": {
                "stage": "E_real_retrieval",
                "status": stage_e_status,
                "reason": "full Oracle-RAG SFT works" if stage_e_status == "UNLOCKED" else "full Oracle-RAG SFT has not been proven",
            },
        },
        "kd_unlocked": stage_d.get("status") == "PASS",
        "real_rag_unlocked": stage_d.get("status") == "PASS",
        "architecture_sweep_unlocked": stage_c.get("status") == "PASS" and stage_d.get("status") == "PASS",
        "human_decision": "PENDING",
    }
    write_json(args.json_out, packet)
    lines = [
        "# Phase 9 grounded-recovery capability ladder",
        "",
        "This ladder isolates teacher/prompt validity, warm initialization, training wiring, Oracle evidence use, and only then real retrieval.",
        "",
    ]
    for key in ("A", "B", "C", "D", "E"):
        stage = packet["stages"][key]
        lines.extend([
            f"## Stage {key} — {stage.get('stage')}",
            "",
            f"- Status: `{stage.get('status')}`",
            f"- Score: `{stage.get('score', stage.get('trained_score', stage.get('sft_score', 'n/a')) )}`",
            f"- Issues: `{stage.get('issues') or 'none'}`",
            "",
        ])
    lines.extend([
        "## Unlocks",
        "",
        f"- KD unlocked: `{packet['kd_unlocked']}`",
        f"- Real-RAG evaluation unlocked: `{packet['real_rag_unlocked']}`",
        f"- Broad architecture sweep unlocked: `{packet['architecture_sweep_unlocked']}`",
        "- Human decision: `PENDING`",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(packet, indent=2))
    if args.require_overfit_pass and stage_c.get("status") != "PASS":
        raise SystemExit(2)
    if args.require_oracle_improvement and stage_d.get("status") != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
