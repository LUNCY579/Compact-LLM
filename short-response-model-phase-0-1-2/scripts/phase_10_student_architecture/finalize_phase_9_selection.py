from __future__ import annotations

"""Materialize and register the exact Phase 9 student approved for Phase 10.

Phase 9 may emit either legacy candidate names (``sft``, ``kd``) or hybrid
profile names such as ``kd_warm_344m``.  This handoff therefore validates the
requested name against the assessment packet instead of hard-coding a finite
choice list.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.check_human_gate import has_human_pass
from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the Phase 9 student handoff.")
    parser.add_argument("--assessment", type=Path, default=ROOT / "reports" / "PHASE_9_ASSESSMENT.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports" / "PHASE_9_STOP_GATE.md")
    parser.add_argument(
        "--candidate",
        required=True,
        help=(
            "Exact candidate key from PHASE_9_ASSESSMENT.json, for example kd, "
            "kd_warm_344m, or sft_narrow_130m."
        ),
    )
    parser.add_argument(
        "--role",
        choices=("deployment_candidate", "intermediate_teacher", "research_control"),
        required=True,
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--allow-research-control-promotion",
        action="store_true",
        help=(
            "Required when a Phase 9 candidate marked as a research control is "
            "intentionally handed off with a non-research role."
        ),
    )
    parser.add_argument(
        "--materialized-output",
        type=Path,
        default=ROOT / "artifacts" / "phase_9_student_distillation" / "accepted_student",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "references" / "phase_9_student_reference.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def candidate_is_research_control(assessment: dict[str, Any], candidate: str) -> bool:
    explicit = {str(name) for name in assessment.get("research_controls") or []}
    return candidate in explicit


def validate_candidate_handoff(
    assessment: dict[str, Any],
    *,
    candidate: str,
    role: str,
    allow_research_control_promotion: bool,
) -> str:
    candidate_summaries = assessment.get("candidates") or {}
    summary_path_value = candidate_summaries.get(candidate)
    if not summary_path_value:
        available = ", ".join(sorted(str(name) for name in candidate_summaries)) or "none"
        raise ValueError(
            f"Candidate {candidate!r} is not present in the Phase 9 assessment. "
            f"Available candidates: {available}"
        )

    is_control = candidate_is_research_control(assessment, candidate)
    if is_control and role != "research_control" and not allow_research_control_promotion:
        raise ValueError(
            f"Candidate {candidate!r} is marked as a Phase 9 research control. "
            "Use --role research_control, or explicitly add "
            "--allow-research-control-promotion after separate language-readiness review."
        )
    if not is_control and role == "research_control":
        # A human may conservatively classify a production-track candidate as research-only.
        return str(summary_path_value)
    return str(summary_path_value)


def materialize(
    model_path: Path,
    adapter_path: Path | None,
    output: Path,
    *,
    overwrite: bool,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Materialized Phase 9 student exists; use --overwrite: {output}")
        shutil.rmtree(output)
    if not model_path.exists():
        raise FileNotFoundError(f"Phase 9 candidate model is unavailable: {model_path}")
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"Phase 9 candidate adapter is unavailable: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map=None,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    # Fresh reload makes the handoff a real checkpoint rather than a dangling training artifact.
    AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map=None,
    )
    return output


def main() -> None:
    args = parse_args()
    gate_path = resolve_repo_path(args.gate)
    assessment_path = resolve_repo_path(args.assessment)
    materialized_output = resolve_repo_path(args.materialized_output)
    manifest_path = resolve_repo_path(args.manifest)

    if not has_human_pass(gate_path):
        raise SystemExit(f"Phase 9 gate lacks explicit human PASS: {gate_path}")
    if not assessment_path.exists():
        raise FileNotFoundError(assessment_path)

    assessment = read_json(assessment_path)
    summary_path_value = validate_candidate_handoff(
        assessment,
        candidate=args.candidate,
        role=args.role,
        allow_research_control_promotion=args.allow_research_control_promotion,
    )
    summary_path = resolve_repo_path(summary_path_value)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = read_json(summary_path)
    raw_model_path = str(summary.get("model") or "").strip()
    raw_adapter_path = str(summary.get("adapter") or "").strip() or None
    if not raw_model_path:
        raise ValueError(f"Candidate summary does not identify a model: {summary_path}")
    model_path = resolve_repo_path(raw_model_path)
    adapter_path = resolve_repo_path(raw_adapter_path) if raw_adapter_path else None

    materialized = materialize(
        model_path,
        adapter_path,
        materialized_output,
        overwrite=args.overwrite,
    )
    research_control = candidate_is_research_control(assessment, args.candidate)
    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "phase": "9_student_distillation",
        "candidate": args.candidate,
        "candidate_class": "research_control" if research_control else "production_track",
        "role": args.role,
        "reason": args.reason,
        "allow_research_control_promotion": bool(args.allow_research_control_promotion),
        "phase_9_recommended_candidate": assessment.get("recommended_candidate"),
        "selected_was_automated_recommendation": assessment.get("recommended_candidate") == args.candidate,
        "source_assessment": str(assessment_path),
        "source_assessment_sha256": sha256(assessment_path),
        "source_gate": str(gate_path),
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "source_model": raw_model_path,
        "resolved_source_model": str(model_path),
        "source_adapter": raw_adapter_path,
        "resolved_source_adapter": str(adapter_path) if adapter_path else None,
        "selected_model": str(materialized),
        "selected_model_tree_sha256": sha256_tree(materialized),
        "assessment": (assessment.get("assessments") or {}).get(args.candidate),
        "rollback": {
            "source_model": raw_model_path,
            "source_adapter": raw_adapter_path,
        },
    }
    write_json(manifest_path, payload)
    print(manifest_path)


if __name__ == "__main__":
    main()
