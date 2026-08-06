from __future__ import annotations

"""Build the proportional Phase 9B assessment and explicit human stop gate."""

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
from scripts.phase_9b_memory_reconstruction.phase9b_common import PIPELINE_REVISION

EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Phase 9B memory-reconstruction gate.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase_9b_memory_reconstruction.json")
    parser.add_argument("--candidate", action="append", default=[], help="name=model_path[,adapter_path]")
    parser.add_argument("--assessment", type=Path, default=ROOT / "reports" / "PHASE_9B_ASSESSMENT.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports" / "PHASE_9B_STOP_GATE.md")
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _gain(reference: Any, candidate: Any) -> float | None:
    left, right = _finite(reference), _finite(candidate)
    if left is None or right is None:
        return None
    if right <= left:
        return 0.0
    if abs(left) <= EPSILON:
        return min(1.0, right - left)
    return min(1.0, (right - left) / abs(left))


def _reduction(reference: Any, candidate: Any) -> float | None:
    left, right = _finite(reference), _finite(candidate)
    if left is None or right is None:
        return None
    if right >= left:
        return 0.0
    if abs(left) <= EPSILON:
        return 0.0
    return min(1.0, (left - right) / abs(left))


def _increase(reference: Any, candidate: Any) -> float | None:
    left, right = _finite(reference), _finite(candidate)
    if left is None or right is None:
        return None
    if right <= left:
        return 0.0
    if abs(left) <= EPSILON:
        return min(1.0, right - left)
    return min(1.0, (right - left) / abs(left))


def _weighted(values: dict[str, float | None], weights: dict[str, Any]) -> tuple[float, dict[str, float]]:
    available = {
        key: max(0.0, float(weights.get(key, 0.0)))
        for key, value in values.items()
        if value is not None and float(weights.get(key, 0.0)) > 0.0
    }
    total = sum(available.values())
    normalized = {key: weight / total for key, weight in available.items()} if total > 0 else {}
    score = sum(normalized[key] * float(values[key] or 0.0) for key in normalized)
    return score, normalized


def _artifact_assignments(values: list[str]) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--candidate must be name=model_path[,adapter_path]")
        name, payload = value.split("=", 1)
        parts = [part.strip() for part in payload.split(",")]
        model = parts[0] if parts else ""
        adapter = parts[1] if len(parts) > 1 and parts[1] else None
        if not name.strip() or not model:
            raise ValueError("--candidate must identify a non-empty name and model path")
        result[name.strip()] = {"model": model, "adapter": adapter}
    return result


def assess(summary: dict[str, Any], config: dict[str, Any], artifacts: dict[str, dict[str, str | None]]) -> dict[str, Any]:
    strategies = summary.get("strategies") or {}
    pairwise = summary.get("pairwise") or {}
    raw = strategies.get("raw_replay") or {}
    prompted = strategies.get("prompted_reconstruction") or {}
    peak = summary.get("peak_allocated_vram_bytes") or {}
    evaluations: dict[str, Any] = {}
    benefit_weights = (config.get("evaluation") or {}).get("benefit_weights") or {}
    cost_weights = (config.get("evaluation") or {}).get("cost_weights") or {}

    for name in ("prompted_reconstruction", "trained_reconstruction", "direct_candidate"):
        candidate = strategies.get(name)
        if not isinstance(candidate, dict):
            continue
        candidate_pair = pairwise.get(name) or {}
        raw_pair = pairwise.get("raw_replay") or {}
        decision_reference = prompted.get("reconstruction_decision_accuracy") if name == "trained_reconstruction" else None
        source_reference = prompted.get("source_id_recall") if name == "trained_reconstruction" else None
        benefit_components = {
            "functional_success_gain": _gain(raw.get("functional_success_rate"), candidate.get("functional_success_rate")),
            "negative_transfer_reduction": _reduction(
                raw_pair.get("negative_transfer_rate"), candidate_pair.get("negative_transfer_rate")
            ),
            "decision_accuracy_gain": _gain(decision_reference, candidate.get("reconstruction_decision_accuracy")),
            "grounding_gain": _gain(source_reference, candidate.get("source_id_recall")),
        }
        cost_components = {
            "latency_increase": _increase(
                raw.get("mean_total_generation_seconds"), candidate.get("mean_total_generation_seconds")
            ),
            "prompt_token_increase": _increase(
                raw.get("mean_total_prompt_tokens"), candidate.get("mean_total_prompt_tokens")
            ),
            "generated_token_increase": _increase(
                raw.get("mean_total_generated_tokens"), candidate.get("mean_total_generated_tokens")
            ),
            "peak_vram_increase": _increase(peak.get("baseline"), peak.get("candidate")),
        }
        benefit, effective_benefit_weights = _weighted(benefit_components, benefit_weights)
        cost, effective_cost_weights = _weighted(cost_components, cost_weights)
        technical_reasons: list[str] = []
        if int(candidate.get("case_count") or 0) <= 0:
            technical_reasons.append("candidate has zero evaluated cases")
        if candidate.get("functional_success_rate") is None:
            technical_reasons.append("candidate functional success is missing")
        if name != "direct_candidate" and candidate.get("reconstruction_case_count", 0) <= 0:
            technical_reasons.append("candidate has no reconstruction cases")
        if summary.get("evaluation_valid") is False:
            technical_reasons.append("evaluation summary is invalid")
        margin = benefit - cost
        eligible = not technical_reasons and margin > 0.0
        if name == "prompted_reconstruction":
            artifact = {
                "model": summary.get("baseline_model"),
                "adapter": summary.get("baseline_adapter"),
            }
        else:
            artifact = artifacts.get(name) or {
                "model": summary.get("candidate_model"),
                "adapter": summary.get("candidate_adapter"),
            }
        evaluations[name] = {
            "strategy": name,
            "artifact": artifact,
            "benefit_components": benefit_components,
            "effective_benefit_weights": effective_benefit_weights,
            "combined_reconstruction_benefit": benefit,
            "cost_components": cost_components,
            "effective_cost_weights": effective_cost_weights,
            "combined_runtime_cost": cost,
            "opportunity_margin": margin,
            "eligible": eligible,
            "decision": "PROPORTIONALLY_WORTHWHILE" if eligible else "NOT_PROPORTIONALLY_WORTHWHILE",
            "technical_valid": not technical_reasons,
            "technical_invalidity_reasons": technical_reasons,
            "functional_success_rate": candidate.get("functional_success_rate"),
            "negative_transfer_rate": candidate_pair.get("negative_transfer_rate"),
            "raw_failure_rescue_rate": candidate_pair.get("raw_failure_rescue_rate"),
            "raw_success_harm_rate": candidate_pair.get("raw_success_harm_rate"),
            "reconstruction_decision_accuracy": candidate.get("reconstruction_decision_accuracy"),
            "reconstruction_parse_valid_rate": candidate.get("reconstruction_parse_valid_rate"),
        }

    ranked = sorted(
        evaluations,
        key=lambda name: (
            bool(evaluations[name]["eligible"]),
            float(evaluations[name]["opportunity_margin"]),
            float(evaluations[name].get("functional_success_rate") or 0.0),
        ),
        reverse=True,
    )
    recommended = ranked[0] if ranked and evaluations[ranked[0]]["eligible"] else None
    direct = evaluations.get("direct_candidate") or {}
    return {
        "schema_version": "1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "source_summary": str(summary.get("benchmark")),
        "research_ready": bool(summary.get("research_ready")),
        "smoke_test": bool(summary.get("smoke_test")),
        "reference_strategy": "raw_replay",
        "strategies": evaluations,
        "ranked_candidates": ranked,
        "recommended_candidate": recommended,
        "phase_10_direct_compatibility": {
            "evaluated": bool(direct),
            "eligible": bool(direct.get("eligible")),
            "functional_success_rate": direct.get("functional_success_rate"),
            "reason": (
                "The same Phase 9B checkpoint was evaluated in one-pass direct-memory mode."
                if direct else "direct_candidate was not evaluated"
            ),
        },
        "automated_status": (
            "RECOMMEND_HUMAN_REVIEW"
            if recommended and summary.get("research_ready")
            else "NO_AUTOMATIC_RECOMMENDATION"
        ),
        "human_decision_required": True,
    }


def gate_markdown(assessment: dict[str, Any], summary_path: Path) -> str:
    lines = [
        "# Phase 9B stop gate — contextual memory reconstruction",
        "",
        f"- Source summary: `{summary_path}`",
        f"- Research ready: `{assessment['research_ready']}`",
        f"- Automated status: `{assessment['automated_status']}`",
        f"- Recommended candidate: `{assessment.get('recommended_candidate')}`",
        "",
        "## Proportional comparisons",
        "",
    ]
    for name in assessment["ranked_candidates"]:
        packet = assessment["strategies"][name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Reconstruction benefit: `{packet['combined_reconstruction_benefit']}`",
                f"- Runtime/token cost: `{packet['combined_runtime_cost']}`",
                f"- Opportunity margin: `{packet['opportunity_margin']}`",
                f"- Decision: `{packet['decision']}`",
                f"- Functional success: `{packet.get('functional_success_rate')}`",
                f"- Negative-transfer rate: `{packet.get('negative_transfer_rate')}`",
                f"- Raw failure rescue/harm: `{packet.get('raw_failure_rescue_rate')}` / `{packet.get('raw_success_harm_rate')}`",
                f"- Decision accuracy / parse-valid: `{packet.get('reconstruction_decision_accuracy')}` / `{packet.get('reconstruction_parse_valid_rate')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Human decision",
            "",
            "Status: **PENDING HUMAN REVIEW**",
            "",
            "A PASS must name the exact strategy and artifact, confirm that source-state comparisons and final answers were inspected, "
            "and decide whether the checkpoint is approved for Phase 10 direct compression, retained only for the two-pass runtime, or rejected.",
            "",
            "Human decision: [ ] PASS  [ ] FAIL",
            "Selected strategy:",
            "Selected artifact:",
            "Reviewer:",
            "Reason:",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    config = read_json(args.config)
    assessment = assess(summary, config, _artifact_assignments(args.candidate))
    assessment["evaluation_summary"] = str(args.summary)
    args.assessment.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.assessment, assessment)
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text(gate_markdown(assessment, args.summary), encoding="utf-8")
    print(args.assessment)
    print(args.gate)


if __name__ == "__main__":
    main()
