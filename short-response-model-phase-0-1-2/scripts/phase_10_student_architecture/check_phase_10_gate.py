from __future__ import annotations

"""Create the Phase 10 candidate registry, recommendation, and human gate."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, resource_reduction
from scripts.phase_10_student_architecture.phase10_common import candidate_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 10 architecture-selection gate.")
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=summary.json")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "artifacts" / "phase_10_student_architecture" / "candidate_registry.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "PHASE_10_CANDIDATE_REPORT.md",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=ROOT / "reports" / "PHASE_10_STOP_GATE.md",
    )
    parser.add_argument("--unsupported-confident-answer-cap", type=float, default=None)
    return parser.parse_args()


def value(summary: dict[str, Any], condition: str, key: str) -> float | None:
    raw = ((summary.get("conditions") or {}).get(condition) or {}).get(key)
    return None if raw is None else float(raw)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else str(value)
    return str(value)


def structural_manifest_for(model_path: Any) -> tuple[Path | None, dict[str, Any] | None]:
    if not model_path:
        return None, None
    model = Path(str(model_path))
    candidates = [model / "phase_10_candidate_manifest.json"]
    if model.name == "model" and model.parent.name == "recovery":
        candidates.append(model.parents[1] / "raw" / "phase_10_candidate_manifest.json")
    for path in candidates:
        if path.exists():
            return path, read_json(path)
    return None, None


def main() -> None:
    args = parse_args()
    if args.unsupported_confident_answer_cap is not None and not 0 <= args.unsupported_confident_answer_cap <= 1:
        raise ValueError("--unsupported-confident-answer-cap must be in [0,1]")
    reference = read_json(args.reference_summary)
    candidates: dict[str, Any] = {}
    for raw in args.candidate:
        name, path = candidate_assignment(raw)
        if name in candidates:
            raise ValueError(f"Duplicate candidate: {name}")
        summary = read_json(path)
        gain = resource_reduction(reference.get("model_weight_bytes"), summary.get("model_weight_bytes"))
        assessment = evaluate_opportunity(
            reference,
            summary,
            resource_gain=gain,
            resource_name="deployed_model_weight_bytes",
        )
        blockers: list[str] = []
        unsupported_rate = value(summary, "real_rag", "unsupported_confident_answer_rate")
        if args.unsupported_confident_answer_cap is not None:
            if unsupported_rate is None:
                blockers.append("missing real-RAG unsupported-confident-answer rate")
            elif unsupported_rate > args.unsupported_confident_answer_cap:
                blockers.append(
                    f"calibrated product cap exceeded: {unsupported_rate:.6f} > "
                    f"{args.unsupported_confident_answer_cap:.6f}"
                )
        model_path = summary.get("model")
        adapter_path = summary.get("adapter")
        if not model_path:
            blockers.append("summary does not identify a model checkpoint")
        structural_manifest_path, structural_manifest = structural_manifest_for(model_path)
        candidates[name] = {
            "name": name,
            "summary": str(path),
            "model": model_path,
            "adapter": adapter_path,
            "model_weight_bytes": summary.get("model_weight_bytes"),
            "parameter_count": (
                summary.get("parameter_count")
                or ((structural_manifest or {}).get("parameters_after") or {}).get("total")
            ),
            "structural_manifest": str(structural_manifest_path) if structural_manifest_path else None,
            "target": (structural_manifest or {}).get("target"),
            "operations": (structural_manifest or {}).get("operations"),
            "assessment": assessment,
            "real_rag": {
                "supported_answer_correctness": value(summary, "real_rag", "supported_answer_correctness"),
                "evidence_faithfulness": value(summary, "real_rag", "evidence_faithfulness"),
                "unsupported_confident_answer_rate": unsupported_rate,
                "correct_abstention_rate": value(summary, "real_rag", "correct_abstention_rate"),
            },
            "technical_or_product_blockers": blockers,
            "eligible": bool(assessment.get("eligible")) and not blockers,
        }
    if not candidates:
        raise ValueError("No Phase 10 candidates supplied")
    eligible = [packet for packet in candidates.values() if packet["eligible"]]
    recommended = max(
        eligible,
        key=lambda packet: (
            float(packet["assessment"].get("opportunity_margin", float("-inf"))),
            -float(packet.get("model_weight_bytes") or float("inf")),
        ),
        default=None,
    )
    registry = {
        "schema_version": "1.0",
        "phase": "10_student_architecture",
        "reference_summary": str(args.reference_summary),
        "reference_model": reference.get("model"),
        "reference_adapter": reference.get("adapter"),
        "reference_model_weight_bytes": reference.get("model_weight_bytes"),
        "candidates": candidates,
        "recommended_candidate": recommended["name"] if recommended else None,
        "unsupported_confident_answer_cap": args.unsupported_confident_answer_cap,
        "human_decision": "PENDING",
    }
    write_json(args.registry, registry)

    report_lines = [
        "# Phase 10 — Student architecture candidate report",
        "",
        f"Reference summary: `{args.reference_summary}`",
        f"Recommended candidate: `{registry['recommended_candidate'] or 'none'}`",
        "",
        "The recommendation uses the shared relative-opportunity policy. Raw structural and recovered candidates are compared by measured deployed weight reduction versus proportional RAG/behavior quality loss. No private absolute quality floor is introduced.",
        "",
    ]
    for name, packet in candidates.items():
        assessment = packet["assessment"]
        report_lines.extend(
            [
                f"## {name}",
                "",
                f"- Model: `{packet['model']}`",
                f"- Adapter: `{packet['adapter'] or 'none'}`",
                f"- Weight bytes: `{packet['model_weight_bytes']}`",
                f"- Parameter count: `{packet['parameter_count']}`",
                f"- Structural manifest: `{packet['structural_manifest'] or 'none'}`",
                f"- Target: `{packet['target'] or 'not declared'}`",
                f"- Resource gain: `{fmt(assessment.get('resource_gain_fraction'))}`",
                f"- Quality opportunity loss: `{fmt(assessment.get('quality_opportunity_loss_fraction'))}`",
                f"- Opportunity margin: `{fmt(assessment.get('opportunity_margin'))}`",
                f"- Real-RAG supported correctness: `{fmt(packet['real_rag']['supported_answer_correctness'])}`",
                f"- Real-RAG evidence faithfulness: `{fmt(packet['real_rag']['evidence_faithfulness'])}`",
                f"- Unsupported confident answer rate: `{fmt(packet['real_rag']['unsupported_confident_answer_rate'])}`",
                f"- Eligible: `{packet['eligible']}`",
                f"- Blockers: `{packet['technical_or_product_blockers'] or 'none'}`",
                "",
            ]
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    gate_lines = [
        "# Phase 10 stop gate",
        "",
        "## Automated status",
        "",
        f"- Recommended candidate: `{registry['recommended_candidate'] or 'none'}`",
        "- Human decision: `PENDING`",
        "",
        "## Human review required",
        "",
        "Review raw and recovered outputs, architecture manifests, oracle/real/adversarial RAG behavior, unsupported claims, false abstentions, role-play quality, actual checkpoint bytes, latency, VRAM/RSS, and rollback loading.",
        "",
        "A PASS must name the exact candidate and classify it as deployment candidate, intermediate teacher, or research-only control. After approval, run `finalize_phase_10_selection.py` to materialize the selected full checkpoint and create `artifacts/references/phase_10_student_reference.json`.",
    ]
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text("\n".join(gate_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Registry: {args.registry}")
    print(f"Report: {args.report}")
    print(f"Gate: {args.gate}")


if __name__ == "__main__":
    main()
