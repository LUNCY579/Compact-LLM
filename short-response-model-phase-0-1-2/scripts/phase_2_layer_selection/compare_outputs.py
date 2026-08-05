from __future__ import annotations

"""Compare reference and candidate generated replies case by case."""

import argparse
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio as fuzzy_ratio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import atomic_write_text, read_jsonl, sha256, write_json
from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _index_unique(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        if "id" not in row:
            raise ValueError(f"{label} row {position} has no stable id")
        case_id = str(row["id"])
        if case_id in indexed:
            raise ValueError(f"{label} results contain duplicate id {case_id!r}")
        indexed[case_id] = row
    return indexed


def compare_rows(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref_by_id = _index_unique(reference, label="reference")
    cand_by_id = _index_unique(candidate, label="candidate")
    ref_ids = set(ref_by_id)
    cand_ids = set(cand_by_id)
    missing_from_candidate = sorted(ref_ids - cand_ids)
    extra_in_candidate = sorted(cand_ids - ref_ids)
    if missing_from_candidate or extra_in_candidate:
        raise ValueError(
            "Reference and candidate evidence do not cover the same stable IDs. "
            f"Missing from candidate: {missing_from_candidate[:8]}; "
            f"extra in candidate: {extra_in_candidate[:8]}"
        )
    common = sorted(ref_ids)
    if not common:
        raise ValueError("Reference and candidate results are empty.")

    rows: list[dict[str, Any]] = []
    for case_id in common:
        ref = ref_by_id[case_id]
        cand = cand_by_id[case_id]
        ref_reply = str(ref.get("reply", ""))
        cand_reply = str(cand.get("reply", ""))
        ref_critical = set(ref.get("critical_failures") or [])
        cand_critical = set(cand.get("critical_failures") or [])
        ref_major = set(ref.get("major_issues") or [])
        cand_major = set(cand.get("major_issues") or [])
        ref_hard = set(ref.get("critical_failures") or []) if ref.get("quality_hard_fail") else set()
        cand_hard = set(cand.get("critical_failures") or []) if cand.get("quality_hard_fail") else set()
        rows.append(
            {
                "id": case_id,
                "category": cand.get("category", ref.get("category", "uncategorized")),
                "reference_reply": ref_reply,
                "candidate_reply": cand_reply,
                "exact_match": normalize(ref_reply) == normalize(cand_reply),
                "fuzzy_similarity": float(fuzzy_ratio(normalize(ref_reply), normalize(cand_reply))) / 100.0,
                "quality_score_delta": float(cand.get("quality_score", cand.get("heuristic_score", 0.0))) - float(ref.get("quality_score", ref.get("heuristic_score", 0.0))),
                "heuristic_score_delta": float(cand.get("quality_score", cand.get("heuristic_score", 0.0))) - float(ref.get("quality_score", ref.get("heuristic_score", 0.0))),
                "generated_token_delta": int(cand.get("generated_tokens", 0)) - int(ref.get("generated_tokens", 0)),
                "new_critical_failures": sorted(cand_critical - ref_critical),
                "new_major_issues": sorted(cand_major - ref_major),
                "new_hard_failures": sorted(cand_hard - ref_hard),
            }
        )

    summary = {
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "case_count": len(rows),
        "technical_valid": True,
        "reference_case_count": len(reference),
        "candidate_case_count": len(candidate),
        "missing_candidate_ids": [],
        "extra_candidate_ids": [],
        "changed_reply_count": sum(1 for row in rows if not row["exact_match"]),
        "exact_reply_match_rate": statistics.mean(1.0 if row["exact_match"] else 0.0 for row in rows),
        "mean_fuzzy_reply_similarity": statistics.mean(float(row["fuzzy_similarity"]) for row in rows),
        "mean_quality_score_delta": statistics.mean(float(row["quality_score_delta"]) for row in rows),
        "mean_heuristic_score_delta": statistics.mean(float(row["quality_score_delta"]) for row in rows),
        "mean_generated_token_delta": statistics.mean(float(row["generated_token_delta"]) for row in rows),
        "new_hard_failure_count": sum(len(row["new_hard_failures"]) for row in rows),
        "new_critical_failure_count": sum(len(row["new_critical_failures"]) for row in rows),
        "new_major_issue_count": sum(len(row["new_major_issues"]) for row in rows),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two evaluator results.jsonl files.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def escape_cell(value: Any) -> str:
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    args = parse_args()
    summary, rows = compare_rows(read_jsonl(args.reference), read_jsonl(args.candidate))
    summary["reference_results_sha256"] = sha256(args.reference)
    summary["candidate_results_sha256"] = sha256(args.candidate)
    write_json(args.out_json, {"summary": summary, "cases": rows})

    changed = [row for row in rows if not row["exact_match"]]
    changed.sort(key=lambda row: (len(row["new_hard_failures"]), len(row["new_critical_failures"]), len(row["new_major_issues"]), -row["quality_score_delta"], -row["fuzzy_similarity"]), reverse=True)
    lines = []
    for row in changed:
        flags = ", ".join(row["new_hard_failures"] + row["new_critical_failures"] + row["new_major_issues"]) or "-"
        lines.append(
            f"| {escape_cell(row['id'])} | {escape_cell(row['category'])} | "
            f"{row['quality_score_delta']:+.3f} | {row['fuzzy_similarity']:.3f} | "
            f"{escape_cell(row['reference_reply'])} | {escape_cell(row['candidate_reply'])} | {escape_cell(flags)} |"
        )
    if not lines:
        lines.append("| None | - | +0.000 | 1.000 | - | - | - |")

    report = f"""# PHASE 2B OUTPUT DEVIATION\n\n## Summary\n\n- Cases compared: {summary['case_count']}\n- Changed replies: {summary['changed_reply_count']}\n- Exact-reply match rate: {summary['exact_reply_match_rate']:.3f}\n- Mean fuzzy similarity: {summary['mean_fuzzy_reply_similarity']:.3f}\n- Mean quality-score delta: {summary['mean_quality_score_delta']:+.4f}\n- New hard failures: {summary['new_hard_failure_count']}
- New critical failures: {summary['new_critical_failure_count']}\n- New major issues: {summary['new_major_issue_count']}\n\nThis report is a trend diagnostic. Different wording is not automatically a regression.\n\n## Changed cases\n\n| ID | Category | Quality delta | Similarity | Full reference | Candidate | New flags |\n|---|---|---:|---:|---|---|---|\n{chr(10).join(lines)}\n"""
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out_report, report)
    print(args.out_json)
    print(args.out_report)


if __name__ == "__main__":
    main()
