from __future__ import annotations

"""Re-evaluate terminal-formatted generation logs under the current evaluator.

This utility is read-only. It is useful when a user has copied evaluator output
from a terminal but has not exported the corresponding results.jsonl file.
"""

import argparse
import ast
import json
import re
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
)

LINE_RE = re.compile(
    r"^\[(?P<index>\d+)/(?P<total>\d+)\]\s+(?P<id>[^:]+):\s+"
    r"(?P<reply>'(?:\\.|[^'])*')\s+quality=(?P<quality>[0-9.]+)\s+legacy=(?P<legacy>[0-9.]+)"
    r"(?:\s+MAJOR=(?P<major>\[[^\]]*\]))?(?:\s+WARN=(?P<warn>\[[^\]]*\]))?\s*$"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    messages = [dict(item) for item in row.get("messages", [])]
    reference = row.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    value["messages"] = messages
    value["reference"] = reference
    value["category"] = str(row.get("category") or (row.get("metadata") or {}).get("reply_class") or "uncategorized")
    return value


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE_RE.match(line.strip())
        if not match:
            raise ValueError(f"Unrecognized terminal line {line_number}: {line[:160]}")
        major = ast.literal_eval(match.group("major")) if match.group("major") else []
        warnings = ast.literal_eval(match.group("warn")) if match.group("warn") else []
        rows.append({
            "index": int(match.group("index")),
            "total": int(match.group("total")),
            "id": match.group("id"),
            "reply": ast.literal_eval(match.group("reply")),
            "previous_quality_score": float(match.group("quality")),
            "previous_legacy_score": float(match.group("legacy")),
            "previous_major_issues": major,
            "previous_warnings": warnings,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Review copied terminal generation outputs under Evaluator v4.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    benchmark_rows = [normalize_record(row) for row in read_jsonl(args.benchmark)]
    by_id = {str(row["id"]): row for row in benchmark_rows}
    logged = parse_log(args.log)
    missing = [row["id"] for row in logged if row["id"] not in by_id]
    if missing:
        raise ValueError(f"Logged ids missing from benchmark: {missing[:10]}")

    config = load_quality_config()
    profile = build_corpus_profile(benchmark_rows)
    reviewed: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for source in logged:
        record = by_id[source["id"]]
        result = evaluate_response(
            record,
            source["reply"],
            max(1, len(source["reply"].split())),
            visible_model_tokens=max(1, len(source["reply"].split())),
            config=config,
            corpus_profile=profile,
        )
        new_labels = result.get("critical_failures", []) + result.get("major_issues", []) + result.get("minor_issues", [])
        reviewed.append({
            **source,
            "category": record["category"],
            "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
            "quality_score": result["quality_score"],
            "quality_defects": result["quality_defects"],
            "scored_labels": new_labels,
            "review_diagnostics": result["review_diagnostics"],
            "score_change": result["quality_score"] - source["previous_quality_score"],
        })
        quality_rows.append({"id": source["id"], "category": record["category"], **result})

    aggregate = aggregate_quality_rows(quality_rows)
    previous_major_counts = Counter(label for row in logged for label in row["previous_major_issues"])
    previous_warning_counts = Counter(label for row in logged for label in row["previous_warnings"])
    new_label_counts = Counter(label for row in reviewed for label in row["scored_labels"])
    prior_major_cases = sum(1 for row in logged if row["previous_major_issues"])
    prior_warning_only_cases = sum(1 for row in logged if not row["previous_major_issues"] and row["previous_warnings"])
    v4_defect_cases = sum(1 for row in reviewed if row["scored_labels"])

    payload = {
        "source_log": str(args.log),
        "benchmark": str(args.benchmark),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "case_count": len(reviewed),
        "previous_major_case_count": prior_major_cases,
        "previous_warning_only_case_count": prior_warning_only_cases,
        "v4_scored_defect_case_count": v4_defect_cases,
        "previous_major_label_counts": dict(sorted(previous_major_counts.items())),
        "previous_warning_label_counts": dict(sorted(previous_warning_counts.items())),
        "v4_scored_label_counts": dict(sorted(new_label_counts.items())),
        "v4_aggregate": aggregate,
        "cases": reviewed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raised = sorted(reviewed, key=lambda row: row["score_change"], reverse=True)[:15]
    lowered = sorted(reviewed, key=lambda row: row["score_change"])[:15]
    lines = [
        "# Evaluator v4 review of the supplied 150 terminal outputs",
        "",
        "This is a read-only comparison. It does not modify official Phase 2B artifacts.",
        "",
        f"- Cases: **{len(reviewed)}**",
        f"- Previous major-label cases: **{prior_major_cases}**",
        f"- Previous warning-only cases: **{prior_warning_only_cases}**",
        f"- Evaluator v4 scored-defect cases: **{v4_defect_cases}**",
        f"- Evaluator v4 critical-defect rate: **{aggregate['critical_defect_case_rate']:.2%}**",
        f"- Evaluator v4 contract-violation rate: **{aggregate['contract_violation_case_rate']:.2%}**",
        f"- Evaluator v4 mean quality: **{aggregate['mean_quality_score']:.3f}**",
        "",
        "## Evaluator v4 scored labels",
        "",
    ]
    if new_label_counts:
        lines.extend(f"- `{label}`: {count}" for label, count in sorted(new_label_counts.items(), key=lambda item: (-item[1], item[0])))
    else:
        lines.append("- None")
    lines.extend(["", "## Largest corrected upward changes", "", "| ID | Previous | v4 | v4 labels | Reply |", "|---|---:|---:|---|---|"])
    for row in raised:
        labels = ", ".join(row["scored_labels"]) or "none"
        reply = row["reply"].replace("|", "\\|")
        lines.append(f"| {row['id']} | {row['previous_quality_score']:.3f} | {row['quality_score']:.3f} | {labels} | {reply} |")
    lines.extend(["", "## Largest downward or unchanged strict-contract cases", "", "| ID | Previous | v4 | v4 labels | Reply |", "|---|---:|---:|---|---|"])
    for row in lowered:
        labels = ", ".join(row["scored_labels"]) or "none"
        reply = row["reply"].replace("|", "\\|")
        lines.append(f"| {row['id']} | {row['previous_quality_score']:.3f} | {row['quality_score']:.3f} | {labels} | {reply} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The large reduction in major labels is intentional: open conversational alternatives are no longer judged through lexical responsiveness. Remaining penalties identify explicit missing facts/boundaries or concrete output corruption. Human review remains required before the corrected frontier is approved.",
    ])
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_json)
    print(args.out_report)


if __name__ == "__main__":
    main()
