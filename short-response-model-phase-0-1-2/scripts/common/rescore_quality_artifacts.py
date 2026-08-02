from __future__ import annotations

"""Re-score existing generated replies under the current evaluator without regeneration.

The utility joins benchmark records by id, preserves runtime/generation fields,
replaces only quality-derived fields, and writes a fresh results.jsonl and
summary.json. It never edits the source evidence unless --in-place is used.
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
    normalize_text,
)

QUALITY_ROW_KEYS = {
    # Current Evaluator-v4.3 fields. Remove them before each re-score so repeated
    # migrations cannot preserve stale quality evidence.
    "quality_evaluation_version", "response_contract", "checks", "legacy_checks",
    "passed_checks", "total_checks", "legacy_heuristic_score", "heuristic_score",
    "quality_score", "quality_pass", "quality_pass_is_diagnostic_only",
    "quality_hard_fail", "quality_dimensions", "quality_weights", "case_issue_cost",
    "intelligibility_score", "intelligibility_pass", "intelligibility_diagnostics",
    "mechanical_integrity_score", "contract_fulfillment_score", "contract_validation",
    "scoring_issue_cost", "contract_issue_cost", "non_contract_issue_cost",
    "functional_fulfillment_score", "response_mode_fulfillment_score",
    "prompt_function_alignment_score", "prompt_function_expectations",
    "category_scoring_used", "category_contract_inference_used",
    "detected_response_mode_confidences",
    "contract_domain_scores", "detected_response_modes", "observable_quality_pass",
    "echo_similarity", "reference_importance", "reference_token_f1",
    "reference_fuzzy_similarity", "reference_anchor_recall", "reference_alignment_score",
    "quality_defects", "critical_failures", "major_issues", "minor_issues",
    "review_diagnostics", "semantic_warnings", "semantic_severity", "semantic_pass",
    # Superseded v2/v3/v3.1 fields. These are stripped rather than translated;
    # v4.3 recomputes all official evidence from the stored reply and explicit contract.
    "semantic_integrity_diagnostics", "semantic_coherence_score",
    "prompt_relevance_score", "intent_fulfillment_score",
    "semantic_task_adequacy_score", "semantic_logic_score", "semantic_context_score",
    "semantic_safety_score", "semantic_anchor_preservation_score",
    "content_anchor_score", "task_compliance_score", "surface_naturalness_score",
}



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def normalize_benchmark(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    messages = [dict(message) for message in record.get("messages", [])]
    reference = record.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    metadata = record.get("metadata") or {}
    normalized["messages"] = messages
    normalized["reference"] = reference
    normalized["category"] = str(record.get("category") or metadata.get("reply_class") or "uncategorized")
    normalized.setdefault("expected", {})
    return normalized


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * p + 0.999999)))
    return float(ordered[index])


def aggregate_full(rows: list[dict[str, Any]], source_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    source_summary = source_summary or {}
    visible = [float(row.get("visible_generated_tokens", row.get("generated_tokens", 0))) for row in rows]
    actual = [float(row.get("generated_tokens", row.get("visible_generated_tokens", 0))) for row in rows]
    times = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    replies = Counter(normalize_text(str(row.get("reply", ""))) for row in rows)
    stop_reasons = Counter(str(row.get("stop_reason", "unknown")) for row in rows)
    base = aggregate_quality_rows(rows)
    return {
        **{key: value for key, value in source_summary.items() if key not in base},
        "case_count": len(rows),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "rescored_from_quality_evaluation_version": source_summary.get("quality_evaluation_version"),
        "median_reply_tokens": statistics.median(visible) if visible else 0.0,
        "p90_reply_tokens": percentile(visible, 0.90) if visible else 0.0,
        "median_actual_generated_tokens": statistics.median(actual) if actual else 0.0,
        "p90_actual_generated_tokens": percentile(actual, 0.90) if actual else 0.0,
        "one_sentence_violation_rate": statistics.fmean(1.0 if row.get("one_sentence_violation") else 0.0 for row in rows),
        "response_control_trim_rate": statistics.fmean(1.0 if row.get("response_control_trimmed") else 0.0 for row in rows),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "mean_generation_seconds": statistics.fmean(times) if times else 0.0,
        "p90_generation_seconds": percentile(times, 0.90) if times else 0.0,
        "peak_vram_gb": max((float(row.get("peak_vram_gb", 0.0)) for row in rows), default=0.0),
        "duplicate_reply_count": sum(count - 1 for count in replies.values() if count > 1),
        "duplicate_reply_rate": sum(count - 1 for count in replies.values() if count > 1) / max(1, len(rows)),
        **base,
    }


def rescore(
    benchmark: Path,
    results: Path,
    output_dir: Path,
    quality_config: Path,
) -> tuple[Path, Path]:
    benchmark_rows = [normalize_benchmark(row) for row in read_jsonl(benchmark)]
    by_id = {str(row["id"]): row for row in benchmark_rows}
    source_rows = read_jsonl(results)
    missing = [str(row.get("id")) for row in source_rows if str(row.get("id")) not in by_id]
    if missing:
        raise ValueError(f"Result ids missing from benchmark: {missing[:10]}")
    profile = build_corpus_profile(benchmark_rows)
    config = load_quality_config(quality_config)
    rescored: list[dict[str, Any]] = []
    for source in source_rows:
        record = by_id[str(source["id"])]
        reply = str(source.get("reply", ""))
        token_count = int(source.get("visible_generated_tokens", source.get("generated_tokens", max(1, len(reply.split())))))
        quality = evaluate_response(
            record,
            reply,
            token_count,
            visible_model_tokens=token_count,
            config=config,
            corpus_profile=profile,
        )
        preserved = {key: value for key, value in source.items() if key not in QUALITY_ROW_KEYS}
        rescored.append({
            **preserved,
            "id": record["id"],
            "category": record["category"],
            "messages": record["messages"],
            "reference": record.get("reference"),
            **quality,
        })
    source_summary_path = results.parent / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    summary = aggregate_full(rescored, source_summary)
    summary["quality_policy_name"] = config.raw.get("policy_name")
    summary["quality_config"] = str(quality_config)
    summary["quality_config_sha256"] = hashlib.sha256(quality_config.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_out = output_dir / "results.jsonl"
    with result_out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_out = output_dir / "summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    migration = {
        "source_results": str(results),
        "source_summary": str(source_summary_path) if source_summary_path.exists() else None,
        "benchmark": str(benchmark),
        "quality_config": str(quality_config),
        "target_quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "case_count": len(rescored),
        "output_results": str(result_out),
        "output_summary": str(summary_out),
    }
    (output_dir / "RESCORE_MANIFEST.json").write_text(json.dumps(migration, indent=2), encoding="utf-8")
    return result_out, summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Re-score stored generations with Quality Evaluator v{QUALITY_EVALUATION_VERSION}.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--quality-config", type=Path, default=ROOT / "configs/quality_evaluation.json")
    args = parser.parse_args()
    if args.in_place and args.output_dir:
        raise SystemExit("Use either --in-place or --output-dir, not both.")
    output = args.results.parent if args.in_place else (args.output_dir or args.results.parent.with_name(args.results.parent.name + f"_quality_v{QUALITY_EVALUATION_VERSION.replace(chr(46), chr(95))}"))
    result_out, summary_out = rescore(args.benchmark, args.results, output, args.quality_config)
    print(result_out)
    print(summary_out)


if __name__ == "__main__":
    main()
