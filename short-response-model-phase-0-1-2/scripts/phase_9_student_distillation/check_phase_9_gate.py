from __future__ import annotations

"""Build the Phase 9 relative-opportunity report and pending human gate."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import (
    evaluate_opportunity,
    resource_reduction,
)
from scripts.phase_9_student_distillation.phase9_metrics import (
    bounded_absolute_quality_comparison,
    summary_readiness_issues,
)


def parse_candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Candidate must be name=path, received {value!r}"
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            f"Candidate must be name=path, received {value!r}"
        )
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 9 teacher/SFT/KD evaluations.")
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        type=parse_candidate_spec,
        help="Repeatable candidate summary as name=path (preferred for hybrid suites).",
    )
    parser.add_argument("--sft-summary", type=Path, default=None, help="Legacy alias for candidate sft=...")
    parser.add_argument("--kd-summary", type=Path, default=None, help="Legacy alias for candidate kd=...")
    parser.add_argument("--on-policy-summary", type=Path, default=None, help="Legacy alias for candidate on_policy=...")
    parser.add_argument("--preference-summary", type=Path, default=None, help="Legacy alias for candidate preference=...")
    parser.add_argument(
        "--research-control",
        action="append",
        default=[],
        help="Candidate name that must not be recommended as a production artifact.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "PHASE_9_STOP_GATE.md")
    parser.add_argument("--json-out", type=Path, default=ROOT / "reports" / "PHASE_9_ASSESSMENT.json")
    parser.add_argument(
        "--unsupported-confident-answer-cap",
        type=float,
        default=None,
        help="Optional calibrated product-safety constraint. Unset by default; never invent a universal threshold.",
    )
    return parser.parse_args()


def real_rag_rate(summary: dict[str, Any], name: str) -> float | None:
    packet = (summary.get("conditions") or {}).get("real_rag") or {}
    value = packet.get(name)
    return None if value is None else float(value)


def assess_candidate(teacher: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    gain = resource_reduction(teacher.get("model_weight_bytes"), candidate.get("model_weight_bytes"))
    assessment = evaluate_opportunity(
        teacher,
        candidate,
        resource_gain=gain,
        resource_name="deployed_model_weight_bytes",
    )
    return {
        **assessment,
        "teacher_weight_bytes": teacher.get("model_weight_bytes"),
        "candidate_weight_bytes": candidate.get("model_weight_bytes"),
        "real_rag_supported_correctness": real_rag_rate(candidate, "supported_answer_correctness"),
        "real_rag_faithfulness": real_rag_rate(candidate, "evidence_faithfulness"),
        "real_rag_unsupported_confident_answer_rate": real_rag_rate(
            candidate, "unsupported_confident_answer_rate"
        ),
        "real_rag_supported_case_count": ((candidate.get("conditions") or {}).get("real_rag") or {}).get("supported_case_count"),
        "real_rag_supported_correct_count": ((candidate.get("conditions") or {}).get("real_rag") or {}).get("supported_correct_count"),
        "real_rag_unsupported_case_count": ((candidate.get("conditions") or {}).get("real_rag") or {}).get("unsupported_case_count"),
        "real_rag_correct_abstention_count": ((candidate.get("conditions") or {}).get("real_rag") or {}).get("correct_abstention_count"),
        "evaluation_valid": candidate.get("evaluation_valid", True),
        "research_ready": candidate.get("research_ready", False),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6f}"
    return str(value)


def objective_prefix(name: str) -> str | None:
    for prefix in ("progressive_kd", "direct_kd", "on_policy", "preference", "sft", "kd"):
        if name == prefix or name.startswith(f"{prefix}_"):
            return prefix
    return None


def profile_suffix(name: str) -> str | None:
    prefix = objective_prefix(name)
    if prefix is None:
        return None
    if name == prefix:
        return ""
    return name[len(prefix) + 1 :]


def is_research_control_name(name: str, explicit: set[str]) -> bool:
    if name in explicit:
        return True
    return "narrow_" in name


def build_quality_comparisons(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    quality_comparisons: dict[str, Any] = {}
    # Legacy single-profile comparisons.
    if "kd" in candidates and "sft" in candidates:
        quality_comparisons["kd_vs_sft"] = bounded_absolute_quality_comparison(candidates["sft"], candidates["kd"])
    if "on_policy" in candidates and "kd" in candidates:
        quality_comparisons["on_policy_vs_kd"] = bounded_absolute_quality_comparison(
            candidates["kd"], candidates["on_policy"]
        )
    if "preference" in candidates:
        predecessor = candidates.get("on_policy") or candidates.get("kd") or candidates.get("sft")
        if predecessor is not None:
            quality_comparisons["preference_vs_predecessor"] = bounded_absolute_quality_comparison(
                predecessor, candidates["preference"]
            )

    # Per-profile objective pairs and cross-profile matched-objective comparisons.
    by_objective: dict[str, dict[str, str]] = {}
    for name in candidates:
        objective = objective_prefix(name)
        profile = profile_suffix(name)
        if objective is None or profile is None:
            continue
        by_objective.setdefault(objective, {})[profile] = name

    for profile, sft_name in by_objective.get("sft", {}).items():
        for kd_objective in ("progressive_kd", "direct_kd", "kd"):
            kd_name = by_objective.get(kd_objective, {}).get(profile)
            if kd_name:
                key = f"{kd_name}_vs_{sft_name}" if profile else f"{kd_objective}_vs_sft"
                if key not in quality_comparisons:
                    quality_comparisons[key] = bounded_absolute_quality_comparison(
                        candidates[sft_name], candidates[kd_name]
                    )
        progressive_name = by_objective.get("progressive_kd", {}).get(profile)
        direct_name = by_objective.get("direct_kd", {}).get(profile)
        if progressive_name and direct_name:
            quality_comparisons[f"{progressive_name}_vs_{direct_name}"] = bounded_absolute_quality_comparison(
                candidates[direct_name], candidates[progressive_name]
            )

    for objective, profile_map in by_objective.items():
        names = sorted(profile_map.items(), key=lambda item: item[0])
        for index, (left_profile, left_name) in enumerate(names):
            for right_profile, right_name in names[index + 1 :]:
                if not left_profile or not right_profile:
                    continue
                key = f"{right_name}_vs_{left_name}"
                quality_comparisons[key] = bounded_absolute_quality_comparison(
                    candidates[left_name], candidates[right_name]
                )
    return quality_comparisons


def collect_candidates(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, path in args.candidate:
        if name in paths:
            raise ValueError(f"Duplicate candidate name: {name}")
        paths[name] = path
    for name, path in (
        ("sft", args.sft_summary),
        ("kd", args.kd_summary),
        ("on_policy", args.on_policy_summary),
        ("preference", args.preference_summary),
    ):
        if path is None:
            continue
        if name in paths:
            raise ValueError(f"Candidate {name} provided both via --candidate and legacy flag")
        paths[name] = path
    if not paths:
        raise ValueError("Provide at least one --candidate name=path or legacy --sft-summary")
    return paths


def main() -> None:
    args = parse_args()
    if args.unsupported_confident_answer_cap is not None and not 0.0 <= args.unsupported_confident_answer_cap <= 1.0:
        raise ValueError("--unsupported-confident-answer-cap must be in [0,1]")
    teacher = read_json(args.teacher_summary)
    candidate_paths = collect_candidates(args)
    candidates = {name: read_json(path) for name, path in candidate_paths.items()}
    research_controls = {str(name) for name in args.research_control}
    for name in candidates:
        if is_research_control_name(name, research_controls):
            research_controls.add(name)

    assessments = {name: assess_candidate(teacher, summary) for name, summary in candidates.items()}
    quality_comparisons = build_quality_comparisons(candidates)

    safety_blockers: dict[str, list[str]] = {}
    for name, assessment in assessments.items():
        blockers: list[str] = []
        rate = assessment.get("real_rag_unsupported_confident_answer_rate")
        if args.unsupported_confident_answer_cap is not None:
            if rate is None:
                blockers.append("real-RAG unsupported-confident-answer rate is missing")
            elif float(rate) > args.unsupported_confident_answer_cap:
                blockers.append(
                    f"calibrated product cap exceeded: {rate:.6f} > {args.unsupported_confident_answer_cap:.6f}"
                )
        safety_blockers[name] = blockers

    readiness_blockers = {
        name: summary_readiness_issues(summary)
        for name, summary in candidates.items()
    }

    production_eligible = [
        name
        for name, assessment in assessments.items()
        if assessment.get("eligible")
        and not safety_blockers[name]
        and not readiness_blockers[name]
        and name not in research_controls
    ]
    research_eligible = [
        name
        for name, assessment in assessments.items()
        if assessment.get("eligible") and not safety_blockers[name] and name in research_controls
    ]
    recommended = max(
        production_eligible,
        key=lambda name: (
            assessments[name].get("opportunity_margin", float("-inf")),
            candidates[name].get("mean_quality_score", float("-inf")),
        ),
        default=None,
    )
    recommended_research = max(
        research_eligible,
        key=lambda name: (
            assessments[name].get("opportunity_margin", float("-inf")),
            candidates[name].get("mean_quality_score", float("-inf")),
        ),
        default=None,
    )
    packet = {
        "teacher_summary": str(args.teacher_summary),
        "candidates": {name: str(path) for name, path in candidate_paths.items()},
        "assessments": assessments,
        "quality_comparisons": quality_comparisons,
        "unsupported_confident_answer_cap": args.unsupported_confident_answer_cap,
        "safety_blockers": safety_blockers,
        "readiness_blockers": readiness_blockers,
        "research_controls": sorted(research_controls),
        "recommended_candidate": recommended,
        "recommended_research_control": recommended_research,
        "human_decision": "PENDING",
    }
    write_json(args.json_out, packet)

    lines = [
        "# Phase 9 — Student distillation stop gate",
        "",
        "## Automated status",
        "",
        f"- Recommended production candidate: `{recommended or 'none'}`",
        f"- Best research-control candidate: `{recommended_research or 'none'}`",
        "- Human decision: `PENDING`",
        "- The recommendation uses the shared proportional resource-gain versus quality-opportunity-loss policy.",
        "- Narrow random-init profiles are research controls and are never auto-recommended for production.",
        "- Production recommendation also requires non-degenerate functional evidence: applicable cases, at least one correct supported answer, and at least one correct abstention when those case types exist.",
        "- Cross-candidate quality comparisons use bounded absolute deltas, preventing division-by-zero explosions when a reference metric is zero.",
        "- No universal grounding or unsupported-answer threshold is invented. A product safety cap is enforced only when explicitly calibrated and supplied.",
        "",
        "## Candidate assessments",
        "",
    ]
    for name, assessment in assessments.items():
        control_note = "research_control" if name in research_controls else "production_eligible_track"
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Track class: `{control_note}`",
                f"- Weight reduction versus teacher: `{fmt(assessment.get('resource_gain_fraction'))}`",
                f"- Quality opportunity loss: `{fmt(assessment.get('quality_opportunity_loss_fraction'))}`",
                f"- Opportunity margin: `{fmt(assessment.get('opportunity_margin'))}`",
                f"- Proportionally worthwhile: `{assessment.get('eligible')}`",
                f"- Real-RAG supported correctness: `{fmt(assessment.get('real_rag_supported_correctness'))}`",
                f"- Real-RAG evidence faithfulness: `{fmt(assessment.get('real_rag_faithfulness'))}`",
                f"- Real-RAG unsupported confident answer rate: `{fmt(assessment.get('real_rag_unsupported_confident_answer_rate'))}`",
                f"- Real-RAG supported successes/cases: `{assessment.get('real_rag_supported_correct_count')}` / `{assessment.get('real_rag_supported_case_count')}`",
                f"- Real-RAG safe abstentions/cases: `{assessment.get('real_rag_correct_abstention_count')}` / `{assessment.get('real_rag_unsupported_case_count')}`",
                f"- Evaluation research-ready: `{assessment.get('research_ready')}`",
                f"- Functional readiness blockers: `{readiness_blockers[name] or 'none'}`",
                f"- Product safety blockers: `{safety_blockers[name] or 'none'}`",
                "",
            ]
        )
    if quality_comparisons:
        lines.extend(["## Training and cross-profile comparisons", ""])
        for name, comparison in quality_comparisons.items():
            lines.append(
                f"- `{name}`: decision `{comparison.get('decision')}`, net quality change "
                f"`{fmt(comparison.get('opportunity_margin'))}` via `{comparison.get('comparison_method', 'shared')}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Human review required",
            "",
            "Review raw oracle/real/adversarial outputs, retrieval misses, false abstentions, unsupported claims, role-play behavior, latency, total process memory, and quantized-runtime evidence before recording PASS.",
            "A PASS should name the exact student artifact and whether it is approved as a production candidate, research-only control, or intermediate teacher.",
            "Do not promote a narrow random-init student as a release model without separate language-readiness evidence.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Assessment: {args.json_out}")
    print(f"Stop gate: {args.out}")


if __name__ == "__main__":
    main()
