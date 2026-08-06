from __future__ import annotations

"""Create the Phase 11 production-runtime registry and pending release gate."""

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
    parser = argparse.ArgumentParser(description="Build the Phase 11 production release gate.")
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=summary.json")
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--answerability-calibration", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "artifacts" / "phase_11_production_rag" / "release_candidate_registry.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "PHASE_11_RELEASE_CANDIDATE_REPORT.md",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=ROOT / "reports" / "PHASE_11_STOP_GATE.md",
    )
    parser.add_argument("--unsupported-confident-answer-cap", type=float, default=None)
    return parser.parse_args()


def memory_bytes(summary: dict[str, Any]) -> int | float | None:
    value = summary.get("total_process_memory_bytes")
    if value is None:
        value = summary.get("process_rss_bytes")
    if value is None:
        value = summary.get("model_weight_bytes")
    return value


def metric(summary: dict[str, Any], key: str) -> float | None:
    raw = ((summary.get("conditions") or {}).get("real_rag") or {}).get(key)
    return None if raw is None else float(raw)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    if not args.snapshot_manifest.exists():
        raise FileNotFoundError(args.snapshot_manifest)
    if not args.answerability_calibration.exists():
        raise FileNotFoundError(args.answerability_calibration)
    if args.unsupported_confident_answer_cap is not None and not 0 <= args.unsupported_confident_answer_cap <= 1:
        raise ValueError("--unsupported-confident-answer-cap must be in [0,1]")

    reference = read_json(args.reference_summary)
    reference_memory = memory_bytes(reference)
    candidates: dict[str, Any] = {}
    for raw in args.candidate:
        name, path = candidate_assignment(raw)
        summary = read_json(path)
        candidate_memory = memory_bytes(summary)
        gain = resource_reduction(reference_memory, candidate_memory)
        assessment = evaluate_opportunity(
            reference,
            summary,
            resource_gain=gain,
            resource_name="total_process_memory_bytes",
        )
        blockers: list[str] = []
        unsupported_rate = metric(summary, "unsupported_confident_answer_rate")
        if args.unsupported_confident_answer_cap is not None:
            if unsupported_rate is None:
                blockers.append("missing unsupported-confident-answer evidence")
            elif unsupported_rate > args.unsupported_confident_answer_cap:
                blockers.append(
                    f"calibrated product cap exceeded: {unsupported_rate:.6f} > "
                    f"{args.unsupported_confident_answer_cap:.6f}"
                )
        if not summary.get("model"):
            blockers.append("runtime summary does not identify the deployment artifact")
        runtime = summary.get("runtime") or {}
        if runtime.get("median_total_seconds") is None:
            blockers.append("missing end-to-end runtime measurement")
        candidates[name] = {
            "name": name,
            "summary": str(path),
            "model": summary.get("model"),
            "runtime_backend": summary.get("runtime_backend"),
            "tokenizer_source": summary.get("tokenizer_source"),
            "model_weight_bytes": summary.get("model_weight_bytes"),
            "total_process_memory_bytes": candidate_memory,
            "knowledge_index_bytes": summary.get("knowledge_index_bytes"),
            "assessment": assessment,
            "runtime": runtime,
            "real_rag": {
                "supported_answer_correctness": metric(summary, "supported_answer_correctness"),
                "evidence_faithfulness": metric(summary, "evidence_faithfulness"),
                "correct_abstention_rate": metric(summary, "correct_abstention_rate"),
                "unsupported_confident_answer_rate": unsupported_rate,
            },
            "blockers": blockers,
            "eligible": bool(assessment.get("eligible")) and not blockers,
        }
    if not candidates:
        raise ValueError("No Phase 11 deployment candidates supplied")
    eligible = [packet for packet in candidates.values() if packet["eligible"]]
    recommended = max(
        eligible,
        key=lambda packet: (
            float(packet["assessment"].get("opportunity_margin", float("-inf"))),
            -float((packet.get("runtime") or {}).get("median_total_seconds") or float("inf")),
        ),
        default=None,
    )
    registry = {
        "schema_version": "1.0",
        "phase": "11_production_rag",
        "reference_summary": str(args.reference_summary),
        "reference_model": reference.get("model"),
        "reference_total_process_memory_bytes": reference_memory,
        "snapshot_manifest": str(args.snapshot_manifest),
        "answerability_calibration": str(args.answerability_calibration),
        "unsupported_confident_answer_cap": args.unsupported_confident_answer_cap,
        "candidates": candidates,
        "recommended_candidate": recommended["name"] if recommended else None,
        "human_decision": "PENDING",
    }
    write_json(args.registry, registry)

    lines = [
        "# Phase 11 — Production RAG release candidate report",
        "",
        f"- Reference runtime: `{args.reference_summary}`",
        f"- Knowledge snapshot: `{args.snapshot_manifest}`",
        f"- Answerability calibration: `{args.answerability_calibration}`",
        f"- Recommended deployment candidate: `{registry['recommended_candidate'] or 'none'}`",
        "",
        "The primary Phase 11 opportunity is measured total-process memory reduction. End-to-end latency, retrieval/index overhead, grounding, abstention, and quality remain visible dimensions in the proportional comparison. No candidate advances on idealized INT4 arithmetic alone.",
        "",
    ]
    for name, packet in candidates.items():
        assessment = packet["assessment"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Artifact: `{packet['model']}`",
                f"- Runtime backend: `{packet['runtime_backend']}`",
                f"- Serialized model bytes: `{packet['model_weight_bytes']}`",
                f"- Total process memory bytes: `{packet['total_process_memory_bytes']}`",
                f"- Index bytes: `{packet['knowledge_index_bytes']}`",
                f"- Resource gain: `{fmt(assessment.get('resource_gain_fraction'))}`",
                f"- Quality opportunity loss: `{fmt(assessment.get('quality_opportunity_loss_fraction'))}`",
                f"- Opportunity margin: `{fmt(assessment.get('opportunity_margin'))}`",
                f"- Median end-to-end seconds: `{fmt((packet['runtime'] or {}).get('median_total_seconds'))}`",
                f"- Real-RAG supported correctness: `{fmt(packet['real_rag']['supported_answer_correctness'])}`",
                f"- Evidence faithfulness: `{fmt(packet['real_rag']['evidence_faithfulness'])}`",
                f"- Correct abstention: `{fmt(packet['real_rag']['correct_abstention_rate'])}`",
                f"- Unsupported confident answer rate: `{fmt(packet['real_rag']['unsupported_confident_answer_rate'])}`",
                f"- Eligible: `{packet['eligible']}`",
                f"- Blockers: `{packet['blockers'] or 'none'}`",
                "",
            ]
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    gate_lines = [
        "# Phase 11 stop gate",
        "",
        "## Automated status",
        "",
        f"- Recommended candidate: `{registry['recommended_candidate'] or 'none'}`",
        "- Human decision: `PENDING`",
        "",
        "## Human commercial release decision",
        "",
        "Review the exact artifact, raw outputs, retrieval misses, post-grounding replacements, unsupported claims, false abstentions, high-risk/complex routing, role-play behavior, quantization regressions, total RSS/VRAM, end-to-end latency, package hashes, offline startup, and rollback loading.",
        "",
        "After approval, change the Human decision line to `PASS`, name the exact candidate in your notes, and run `package_phase_11_release.py`. A dense deployment remains valid when quantization or an alternative runtime produces no measured net benefit.",
    ]
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text("\n".join(gate_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Registry: {args.registry}")
    print(f"Report: {args.report}")
    print(f"Gate: {args.gate}")


if __name__ == "__main__":
    main()
