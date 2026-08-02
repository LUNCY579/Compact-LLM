from __future__ import annotations

"""Evaluate the human-adjudicated Evaluator-v4.1 legacy packet and write a report."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Create a Quality Evaluator v{QUALITY_EVALUATION_VERSION} adjudication report.")
    parser.add_argument("--packet", type=Path, default=ROOT / "data/evaluation/quality_v4_adjudication.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/QUALITY_V4_ADJUDICATION_REPORT.md")
    parser.add_argument("--strict", action="store_true", help="Fail when scored labels differ from the approved human adjudication.")
    args = parser.parse_args()

    source_rows = read_jsonl(args.packet)
    records = [
        {
            "id": row["id"],
            "category": row.get("group", "adjudication"),
            "messages": [{"role": "user", "content": row["prompt"]}],
            "reference": row.get("reference"),
            "response_contract": row.get("response_contract") or {
                "schema_version": QUALITY_EVALUATION_VERSION, "open_ended": True,
                "min_tokens": 1, "max_tokens": 32,
                "requirements": [], "prohibitions": [],
            },
        }
        for row in source_rows
    ]
    corpus = build_corpus_profile(records)
    config = load_quality_config()
    evaluated: list[dict[str, Any]] = []
    mismatches: list[str] = []

    for source, record in zip(source_rows, records):
        result = evaluate_response(
            record,
            source["reply"],
            max(1, len(str(source["reply"]).split())),
            config=config,
            corpus_profile=corpus,
        )
        actual_labels = sorted(set(result.get("critical_failures", []) + result.get("major_issues", []) + result.get("minor_issues", [])))
        expected_labels = sorted(set(source.get("expected_scored_labels") or []))
        actual_diagnostics = sorted({item.get("label") for item in result.get("review_diagnostics", [])})
        expected_diagnostics = sorted(set(source.get("expected_review_diagnostics") or []))
        labels_match = actual_labels == expected_labels
        diagnostics_present = set(expected_diagnostics).issubset(actual_diagnostics)
        if not labels_match or not diagnostics_present:
            mismatches.append(str(source["id"]))
        evaluated.append({
            **source,
            "result": result,
            "actual_labels": actual_labels,
            "actual_diagnostics": actual_diagnostics,
            "labels_match": labels_match,
            "diagnostics_present": diagnostics_present,
        })

    lines = [
        f"# Quality Evaluator v{QUALITY_EVALUATION_VERSION} adjudication report",
        "",
        "This packet records approved human expectations for concrete output defects and explicit response-contract violations.",
        "Reference wording and review-only diagnostics never create a scored defect.",
        "",
        f"- Cases: **{len(evaluated)}**",
        f"- Exact scored-label matches: **{len(evaluated) - len(mismatches)}/{len(evaluated)}**",
        f"- Mismatches: **{len(mismatches)}**",
        "",
        "| ID | Group | Expected scored labels | Actual scored labels | Score | Review diagnostics | Result | Reply |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in evaluated:
        result = row["result"]
        expected = ", ".join(row.get("expected_scored_labels") or []) or "none"
        actual = ", ".join(row["actual_labels"]) or "none"
        diagnostics = ", ".join(row["actual_diagnostics"]) or "none"
        status = "PASS" if row["labels_match"] and row["diagnostics_present"] else "MISMATCH"
        reply = str(row["reply"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['id']} | {row['group']} | {expected} | {actual} | {result['quality_score']:.3f} | {diagnostics} | {status} | {reply} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Acceptable informal, alternate, multilingual, and low-reference-overlap replies remain defect-free.",
        "- Mechanical collapse receives concrete labels such as `repeated_token_loop`, `encoding_replacement_character`, or `broken_clause_sequence`.",
        "- Strict-task failures receive the exact missing requirement label authored in the benchmark contract.",
        "- No catch-all responsiveness, vocabulary, or polarity label is permitted.",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)
    if args.strict and mismatches:
        raise SystemExit(f"Adjudication mismatch in: {', '.join(mismatches)}")


if __name__ == "__main__":
    main()
