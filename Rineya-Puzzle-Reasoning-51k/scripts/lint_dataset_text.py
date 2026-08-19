#!/usr/bin/env python3
"""Lint user-visible dataset text for common synthetic-generation defects."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

PATTERNS: dict[str, re.Pattern[str]] = {
    "bad_count_plural": re.compile(
        r"\b1 (statements|codes|assignments|matchings|orders|worlds|moves|days|"
        r"positions|solutions|candidates|facts|rules|labels|clues)\b",
        re.IGNORECASE,
    ),
    "bad_article": re.compile(r"\ba (orange|amber|ivory|owl|apple)\b", re.IGNORECASE),
    "bad_exactly_one_agreement": re.compile(
        r"Exactly 1\b[^.?!]{0,100}\bare true\b", re.IGNORECASE
    ),
    "one_remain": re.compile(r"\b1 remain\b", re.IGNORECASE),
    "double_space": re.compile(r" {2,}"),
    "duplicate_word": re.compile(r"\b([A-Za-z]+)\s+\1\b", re.IGNORECASE),
    "control_character": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"),
}


def visible_fields(row: dict[str, Any]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = [
        ("prompt", str(row.get("prompt", ""))),
        ("compact_state", str(row.get("compact_state", ""))),
        ("final_answer", str(row.get("final_answer", ""))),
    ]
    fields.extend(
        (f"reasoning_steps[{i}]", str(text))
        for i, text in enumerate(row.get("reasoning_steps", []))
    )
    return fields


def lint(root: Path, max_examples: int = 20) -> dict[str, Any]:
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in PATTERNS}
    checked_rows = 0
    files = sorted(glob.glob(str(root / "data" / "*.jsonl")))
    for filename in files:
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                checked_rows += 1
                row = json.loads(line)
                for field_name, text in visible_fields(row):
                    for check_name, pattern in PATTERNS.items():
                        match = pattern.search(text)
                        if match and len(hits[check_name]) < max_examples:
                            hits[check_name].append(
                                {
                                    "id": row.get("id"),
                                    "file": Path(filename).name,
                                    "line": line_number,
                                    "field": field_name,
                                    "match": match.group(0),
                                    "text": text,
                                }
                            )
    failure_count = sum(len(examples) for examples in hits.values())
    return {
        "status": "PASS" if failure_count == 0 else "FAIL",
        "checked_rows": checked_rows,
        "failure_count": failure_count,
        "checks": {name: len(examples) for name, examples in hits.items()},
        "examples": {name: examples for name, examples in hits.items() if examples},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = lint(args.root, args.max_examples)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
