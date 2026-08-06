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
    evaluate_quality_improvement,
    resource_reduction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 9 teacher/SFT/KD evaluations.")
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--kd-summary", type=Path, default=None)
    parser.add_argument("--on-policy-summary", type=Path, default=None)
    parser.add_argument("--preference-summary", type=Path, default=None)
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
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.unsupported_confident_answer_cap is not None and not 0.0 <= args.unsupported_confident_answer_cap <= 1.0:
        raise ValueError("--unsupported-confident-answer-cap must be in [0,1]")
    teacher = read_json(args.teacher_summary)
    candidates: dict[str, dict[str, Any]] = {"sft": read_json(args.sft_summary)}
    for name, path in (
        ("kd", args.kd_summary),
        ("on_policy", args.on_policy_summary),
        ("preference", args.preference_summary),
    ):
        if path is not None:
            candidates[name] = read_json(path)

    assessments = {name: assess_candidate(teacher, summary) for name, summary in candidates.items()}
    quality_comparisons: dict[str, Any] = {}
    if "kd" in candidates:
        quality_comparisons["kd_vs_sft"] = evaluate_quality_improvement(candidates["sft"], candidates["kd"])
    if "on_policy" in candidates and "kd" in candidates:
        quality_comparisons["on_policy_vs_kd"] = evaluate_quality_improvement(
            candidates["kd"], candidates["on_policy"]
        )
    if "preference" in candidates:
        predecessor = candidates.get("on_policy") or candidates.get("kd") or candidates["sft"]
        quality_comparisons["preference_vs_predecessor"] = evaluate_quality_improvement(
            predecessor, candidates["preference"]
        )

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

    eligible = [
        name for name, assessment in assessments.items()
        if assessment.get("eligible") and not safety_blockers[name]
    ]
    recommended = max(
        eligible,
        key=lambda name: (
            assessments[name].get("opportunity_margin", float("-inf")),
            candidates[name].get("mean_quality_score", float("-inf")),
        ),
        default=None,
    )
    packet = {
        "teacher_summary": str(args.teacher_summary),
        "candidates": {name: str(path) for name, path in {
            "sft": args.sft_summary,
            "kd": args.kd_summary,
            "on_policy": args.on_policy_summary,
            "preference": args.preference_summary,
        }.items() if path is not None},
        "assessments": assessments,
        "quality_comparisons": quality_comparisons,
        "unsupported_confident_answer_cap": args.unsupported_confident_answer_cap,
        "safety_blockers": safety_blockers,
        "recommended_candidate": recommended,
        "human_decision": "PENDING",
    }
    write_json(args.json_out, packet)

    lines = [
        "# Phase 9 — Student distillation stop gate",
        "",
        "## Automated status",
        "",
        f"- Recommended candidate: `{recommended or 'none'}`",
        "- Human decision: `PENDING`",
        "- The recommendation uses the shared proportional resource-gain versus quality-opportunity-loss policy.",
        "- No universal grounding or unsupported-answer threshold is invented. A product safety cap is enforced only when explicitly calibrated and supplied.",
        "",
        "## Candidate assessments",
        "",
    ]
    for name, assessment in assessments.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Weight reduction versus teacher: `{fmt(assessment.get('resource_gain_fraction'))}`",
                f"- Quality opportunity loss: `{fmt(assessment.get('quality_opportunity_loss_fraction'))}`",
                f"- Opportunity margin: `{fmt(assessment.get('opportunity_margin'))}`",
                f"- Proportionally worthwhile: `{assessment.get('eligible')}`",
                f"- Real-RAG supported correctness: `{fmt(assessment.get('real_rag_supported_correctness'))}`",
                f"- Real-RAG evidence faithfulness: `{fmt(assessment.get('real_rag_faithfulness'))}`",
                f"- Real-RAG unsupported confident answer rate: `{fmt(assessment.get('real_rag_unsupported_confident_answer_rate'))}`",
                f"- Product safety blockers: `{safety_blockers[name] or 'none'}`",
                "",
            ]
        )
    if quality_comparisons:
        lines.extend(["## Same-architecture training comparisons", ""])
        for name, comparison in quality_comparisons.items():
            lines.append(
                f"- `{name}`: decision `{comparison.get('decision')}`, net quality change "
                f"`{fmt(comparison.get('opportunity_margin'))}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Human review required",
            "",
            "Review raw oracle/real/adversarial outputs, retrieval misses, false abstentions, unsupported claims, role-play behavior, latency, total process memory, and quantized-runtime evidence before recording PASS.",
            "A PASS should name the exact student artifact and whether it is approved as a production candidate, research-only control, or intermediate teacher.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Assessment: {args.json_out}")
    print(f"Stop gate: {args.out}")


if __name__ == "__main__":
    main()
