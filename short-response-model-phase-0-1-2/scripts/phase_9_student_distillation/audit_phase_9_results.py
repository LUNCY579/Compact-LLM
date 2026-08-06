from __future__ import annotations

"""Audit Phase 9 outputs before retraining or promotion.

This tool separates prompt/evaluation corruption from teacher-data failure and
student-learning failure. It can inspect legacy evaluations, but legacy rows
without schema-1.2 prompt/decode evidence are explicitly marked unverifiable.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, read_jsonl, write_json
from scripts.phase_9_student_distillation.phase9_metrics import safe_rate

EXPECTED_CONDITIONS = {
    "closed_book",
    "oracle_rag",
    "real_rag",
    "adversarial_rag",
    "adversarial_passthrough_rag",
}


def evaluation_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Evaluation must be name=directory_or_summary.json")
    name, raw = value.split("=", 1)
    if not name.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("Evaluation must be name=directory_or_summary.json")
    return name.strip(), Path(raw.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw Phase 9 RAG results and diagnose the failed stage.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--evaluation", type=evaluation_spec, action="append", default=[], required=True)
    parser.add_argument("--teacher-name", default="teacher")
    parser.add_argument("--json-out", type=Path, default=ROOT / "reports" / "PHASE_9_DIAGNOSTIC_AUDIT.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "PHASE_9_DIAGNOSTIC_AUDIT.md")
    parser.add_argument("--require-usable-teacher", action="store_true")
    parser.add_argument("--maximum-examples", type=int, default=8)
    return parser.parse_args()


def resolve_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / "summary.json", path / "results.jsonl"
    if path.name == "summary.json":
        return path, path.with_name("results.jsonl")
    raise ValueError(f"Evaluation path must be a directory or summary.json: {path}")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def recompute_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row.get("case_class") == "supported"]
    unsupported = [row for row in rows if row.get("case_class") == "unsupported"]
    conflicts = [row for row in rows if row.get("case_class") == "conflict"]
    prompt_supported = [row for row in rows if row.get("prompt_support_applicable")]
    generated = [row for row in rows if row.get("stop_reason") != "external_gate"]
    adversarial = [row for row in rows if row.get("adversarial_applicable")]
    return {
        "case_count": len(rows),
        "supported_case_count": len(supported),
        "supported_answer_correctness": safe_rate(sum(bool(row.get("supported_answer_correct")) for row in supported), len(supported)),
        "unsupported_case_count": len(unsupported),
        "correct_abstention_rate": safe_rate(sum(bool(row.get("abstained")) for row in unsupported), len(unsupported)),
        "unsupported_confident_answer_rate": safe_rate(sum(not bool(row.get("abstained")) for row in unsupported), len(unsupported)),
        "conflict_detection_rate": safe_rate(sum(bool(row.get("conflict_response")) for row in conflicts), len(conflicts)),
        "prompt_support_recall_at_k": safe_rate(sum(bool(row.get("support_in_prompt")) for row in prompt_supported), len(prompt_supported)),
        "prompt_integrity_failure_rate": safe_rate(sum(not bool((row.get("prompt_integrity") or {}).get("integrity_ok")) for row in rows), len(rows)),
        "decode_integrity_failure_rate": safe_rate(sum(not bool((row.get("decode_integrity") or {}).get("integrity_ok")) for row in generated), len(generated)),
        "injection_resistance_rate": safe_rate(sum(bool(row.get("adversarial_success")) for row in adversarial), len(adversarial)),
    }


def close_enough(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    a, b = finite(left), finite(right)
    if a is None or b is None:
        return left == right
    return abs(a - b) <= 1e-8


def suspicious_decode(row: dict[str, Any]) -> list[str]:
    issues = list((row.get("decode_integrity") or {}).get("issues") or [])
    answer = str(row.get("answer") or "")
    raw = str(row.get("raw_answer") or "")
    if int(row.get("generated_tokens") or row.get("generated_token_count") or 0) > 0 and not raw.strip():
        issues.append("nonzero generated tokens but empty raw answer")
    if "\ufffd" in answer or "\ufffd" in raw:
        issues.append("replacement character in output")
    return sorted(set(issues))


def audit_one(name: str, path: Path, benchmark_ids: set[str], maximum_examples: int) -> dict[str, Any]:
    summary_path, results_path = resolve_paths(path)
    problems: list[str] = []
    if not summary_path.exists():
        return {"name": name, "path": str(path), "problems": [f"missing {summary_path}"], "usable": False}
    if not results_path.exists():
        return {"name": name, "path": str(path), "problems": [f"missing {results_path}"], "usable": False}
    summary = read_json(summary_path)
    rows = read_jsonl(results_path)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = Counter(str(row.get("evaluation_id") or "") for row in rows)
    duplicate_ids = sorted(key for key, count in ids.items() if key and count > 1)
    if duplicate_ids:
        problems.append(f"duplicate evaluation IDs: {duplicate_ids[:5]}")
    for row in rows:
        by_condition[str(row.get("condition") or "unknown")].append(row)
    condition_names = set(by_condition)
    missing_conditions = sorted(EXPECTED_CONDITIONS - condition_names)
    if missing_conditions:
        problems.append(f"missing conditions: {missing_conditions}")
    coverage: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        observed = {str(row.get("id")) for row in condition_rows}
        missing = sorted(benchmark_ids - observed)
        extra = sorted(observed - benchmark_ids)
        coverage[condition] = {"observed": len(observed), "missing_ids": missing, "extra_ids": extra}
        if missing:
            problems.append(f"{condition} missing {len(missing)} benchmark IDs")
        if extra:
            problems.append(f"{condition} has {len(extra)} unknown IDs")

    schema = str(summary.get("schema_version") or "legacy")
    prompt_evidence_verifiable = schema >= "1.2" and all("prompt_integrity" in row for row in rows)
    decode_evidence_verifiable = schema >= "1.2" and all(
        row.get("stop_reason") == "external_gate" or "decode_integrity" in row for row in rows
    )
    if not prompt_evidence_verifiable:
        problems.append("prompt integrity is unverifiable; rerun evaluation with schema 1.2 or newer")
    if not decode_evidence_verifiable:
        problems.append("decode integrity is unverifiable; rerun evaluation with schema 1.2 or newer")

    prompt_failures: list[dict[str, Any]] = []
    decode_failures: list[dict[str, Any]] = []
    support_removed: list[dict[str, Any]] = []
    functional_failures: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt_integrity") or {}
        if prompt_evidence_verifiable and not bool(prompt.get("integrity_ok")):
            prompt_failures.append({"evaluation_id": row.get("evaluation_id"), "issues": prompt.get("integrity_issues") or prompt.get("issues")})
        decode_issues = suspicious_decode(row)
        if decode_evidence_verifiable and decode_issues:
            decode_failures.append({"evaluation_id": row.get("evaluation_id"), "issues": decode_issues, "answer": row.get("answer")})
        if row.get("support_retrieved") and row.get("prompt_support_applicable") and not row.get("support_in_prompt"):
            support_removed.append({
                "evaluation_id": row.get("evaluation_id"),
                "support_document_ids": row.get("support_document_ids"),
                "retrieved_document_ids": row.get("retrieved_document_ids"),
                "prompted_document_ids": row.get("prompted_document_ids"),
            })
        if row.get("condition") in {"oracle_rag", "real_rag"}:
            klass = row.get("case_class")
            failed = (
                (klass == "supported" and not row.get("supported_answer_correct")) or
                (klass == "unsupported" and not row.get("abstained")) or
                (klass == "conflict" and not row.get("conflict_response"))
            )
            if failed:
                functional_failures.append({
                    "evaluation_id": row.get("evaluation_id"),
                    "class": klass,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "reference": row.get("reference"),
                    "gate_reason": row.get("gate_reason"),
                    "support_retrieved": row.get("support_retrieved"),
                    "support_in_prompt": row.get("support_in_prompt"),
                    "functional_correctness": row.get("functional_correctness"),
                })

    if prompt_failures:
        problems.append(f"{len(prompt_failures)} prompt-integrity failures")
    if decode_failures:
        problems.append(f"{len(decode_failures)} decode-integrity failures")
    if support_removed:
        problems.append(f"{len(support_removed)} cases retrieved required support but omitted it from the prompt")

    recomputed: dict[str, Any] = {}
    summary_mismatches: list[str] = []
    for condition, condition_rows in sorted(by_condition.items()):
        packet = recompute_condition(condition_rows)
        recomputed[condition] = packet
        official = (summary.get("conditions") or {}).get(condition) or {}
        for field, value in packet.items():
            if field in official and not close_enough(official.get(field), value):
                summary_mismatches.append(f"{condition}.{field}: summary={official.get(field)!r}, raw={value!r}")
    if summary_mismatches:
        problems.append(f"summary/results mismatch in {len(summary_mismatches)} fields")

    real = recomputed.get("real_rag") or ((summary.get("conditions") or {}).get("real_rag") or {})
    teacher_usable = (
        finite(real.get("supported_answer_correctness")) is not None
        and float(real.get("supported_answer_correctness") or 0.0) > 0.0
        and (
            int(real.get("unsupported_case_count") or 0) == 0
            or float(real.get("correct_abstention_rate") or 0.0) > 0.0
        )
        and not prompt_failures
        and not decode_failures
        and prompt_evidence_verifiable
        and decode_evidence_verifiable
    )
    technical_pipeline_failure = bool(
        duplicate_ids or missing_conditions or prompt_failures or decode_failures or support_removed or summary_mismatches
    )
    return {
        "name": name,
        "path": str(path),
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "schema_version": schema,
        "row_count": len(rows),
        "coverage": coverage,
        "problems": problems,
        "prompt_evidence_verifiable": prompt_evidence_verifiable,
        "decode_evidence_verifiable": decode_evidence_verifiable,
        "prompt_failures": prompt_failures[:maximum_examples],
        "decode_failures": decode_failures[:maximum_examples],
        "support_removed_after_retrieval": support_removed[:maximum_examples],
        "functional_failures": functional_failures[:maximum_examples],
        "summary_mismatches": summary_mismatches[:maximum_examples],
        "recomputed_conditions": recomputed,
        "teacher_usable": teacher_usable,
        "technical_pipeline_failure": technical_pipeline_failure,
        "usable": not technical_pipeline_failure and prompt_evidence_verifiable and decode_evidence_verifiable,
    }


def diagnose(audits: dict[str, dict[str, Any]], teacher_name: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    teacher = audits.get(teacher_name)
    if teacher is None:
        return "INSUFFICIENT_EVIDENCE", [f"teacher evaluation {teacher_name!r} was not supplied"]
    if not teacher.get("prompt_evidence_verifiable") or not teacher.get("decode_evidence_verifiable"):
        return "EVALUATION_RERUN_REQUIRED", ["legacy results cannot prove what prompt or decoded text the model actually saw"]
    if teacher.get("technical_pipeline_failure"):
        return "PIPELINE_INTEGRITY_FAILURE", list(teacher.get("problems") or [])
    if not teacher.get("teacher_usable"):
        return "TEACHER_OR_DATA_FAILURE", ["teacher did not demonstrate both supported answering and required abstention"]
    failed_students = []
    for name, packet in audits.items():
        if name == teacher_name:
            continue
        real = packet.get("recomputed_conditions", {}).get("real_rag", {})
        if packet.get("technical_pipeline_failure"):
            return "PIPELINE_INTEGRITY_FAILURE", [f"{name}: {item}" for item in packet.get("problems") or []]
        if float(real.get("supported_answer_correctness") or 0.0) <= 0.0 or (
            int(real.get("unsupported_case_count") or 0) > 0 and float(real.get("correct_abstention_rate") or 0.0) <= 0.0
        ):
            failed_students.append(name)
    if failed_students:
        return "STUDENT_TRAINING_FAILURE", ["teacher path is usable but these students failed the RAG contract: " + ", ".join(failed_students)]
    return "NO_BLOCKING_FAILURE_FOUND", ["teacher, prompt/decode pipeline, and supplied students have non-degenerate functional evidence"]


def markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Phase 9 diagnostic audit",
        "",
        f"- Diagnosis: `{packet['diagnosis']}`",
        f"- Reasons: `{packet['diagnosis_reasons']}`",
        "",
        "## Evaluations",
        "",
    ]
    for name, audit in packet["evaluations"].items():
        real = audit.get("recomputed_conditions", {}).get("real_rag", {})
        lines.extend([
            f"### {name}",
            "",
            f"- Schema: `{audit.get('schema_version')}`",
            f"- Prompt/decode evidence verifiable: `{audit.get('prompt_evidence_verifiable')}` / `{audit.get('decode_evidence_verifiable')}`",
            f"- Technical pipeline failure: `{audit.get('technical_pipeline_failure')}`",
            f"- Real-RAG supported correctness: `{real.get('supported_answer_correctness')}`",
            f"- Real-RAG correct abstention: `{real.get('correct_abstention_rate')}`",
            f"- Real-RAG unsupported confident answers: `{real.get('unsupported_confident_answer_rate')}`",
            f"- Problems: `{audit.get('problems') or 'none'}`",
            "",
        ])
        failures = audit.get("functional_failures") or []
        if failures:
            lines.extend(["Representative functional failures:", ""])
            for item in failures:
                lines.append(
                    f"- `{item.get('evaluation_id')}` — Q: {item.get('question')} — A: {item.get('answer')}"
                )
            lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "`EVALUATION_RERUN_REQUIRED` means existing artifacts may be retained, but their evaluations must be regenerated with schema 1.2 or newer (schema 1.3 is current) before diagnosing training.",
        "`PIPELINE_INTEGRITY_FAILURE` means fix prompt construction, evidence insertion, serialization, or decoding before retraining.",
        "`TEACHER_OR_DATA_FAILURE` means the teacher/evidence path itself has not established the target behavior.",
        "`STUDENT_TRAINING_FAILURE` means the teacher path is usable and student training/capacity/curriculum is the next suspect.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    benchmark = read_jsonl(args.benchmark)
    benchmark_ids = {str(row.get("id")) for row in benchmark}
    if not benchmark_ids:
        raise ValueError("Benchmark is empty")
    audits = {
        name: audit_one(name, path, benchmark_ids, args.maximum_examples)
        for name, path in args.evaluation
    }
    diagnosis, reasons = diagnose(audits, args.teacher_name)
    packet = {
        "phase": "9_student_distillation",
        "schema_version": "1.0",
        "benchmark": str(args.benchmark),
        "teacher_name": args.teacher_name,
        "diagnosis": diagnosis,
        "diagnosis_reasons": reasons,
        "evaluations": audits,
    }
    write_json(args.json_out, packet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(packet), encoding="utf-8")
    print(f"Diagnosis: {diagnosis}")
    print(f"Audit JSON: {args.json_out}")
    print(f"Audit report: {args.out}")
    if args.require_usable_teacher and diagnosis in {
        "EVALUATION_RERUN_REQUIRED", "PIPELINE_INTEGRITY_FAILURE", "TEACHER_OR_DATA_FAILURE", "INSUFFICIENT_EVIDENCE"
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
