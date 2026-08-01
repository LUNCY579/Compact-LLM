from __future__ import annotations

"""Evaluate the policy calibration packet without inventing human labels."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import (
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Quality Evaluator v3 calibration report.")
    parser.add_argument("--packet", type=Path, default=ROOT / "data/evaluation/quality_v3_calibration.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/QUALITY_V3_CALIBRATION_REPORT.md")
    args = parser.parse_args()
    rows = read_jsonl(args.packet)
    records = []
    for row in rows:
        record = {
            "id": row["id"],
            "category": row.get("category", "calibration"),
            "messages": [{"role": "user", "content": row["prompt"]}],
            "response_contract": row.get(
                "response_contract",
                {"open_ended": True, "min_tokens": 1, "max_tokens": 32},
            ),
        }
        if row.get("reference") is not None:
            record["reference"] = row["reference"]
        records.append(record)
    corpus = build_corpus_profile(records)
    config = load_quality_config()
    evaluated = []
    for source, record in zip(rows, records):
        result = evaluate_response(record, source["reply"], max(1, len(source["reply"].split())), config=config, corpus_profile=corpus)
        evaluated.append({**source, "result": result})

    lines = [
        "# Quality Evaluator v3 calibration report", "",
        "This report is deterministic policy evidence, not a substitute for pending human ranks.", "",
        "| ID | Group | Policy expectation | Score | Severity | Reply |", "|---|---|---|---:|---|---|",
    ]
    for row in evaluated:
        result = row["result"]
        reply = str(row["reply"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['group']} | {row['policy_expectation']} | "
            f"{result['quality_score']:.3f} | {result['semantic_severity']} | {reply} |"
        )
    lines.extend(["", "## Pending human calibration", "", "All `human_rank` values remain `PENDING` until a reviewer records them."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
