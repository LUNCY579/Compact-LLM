#!/usr/bin/env python3
"""Validate structural invariants of the Rineya long-context suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            assert isinstance(row, dict), f"{path}:{line_no}: row is not an object"
            rows.append(row)
    return rows


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("data/rineya_long_context_demo_full.jsonl"),
    )
    args = parser.parse_args()

    rows = load_jsonl(args.full)
    assert len(rows) == 256, f"expected 256 rows, found {len(rows)}"
    assert len({row["id"] for row in rows}) == len(rows), "duplicate ids"
    assert len({hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in rows}) == len(rows), "duplicate prompts"

    forbidden = ("harry potter", "hermione granger", "sherlock holmes", "john watson")
    for row in rows:
        context_lower = row["context"].lower()
        assert not any(term in context_lower for term in forbidden), f"franchise term in {row['id']}"
        assert row["context_words"] == words(row["context"]), f"bad word count in {row['id']}"
        assert 800 <= row["context_words"] <= 1300, f"unexpected context length in {row['id']}"
        assert row["prompt"].endswith("\n\nANSWER"), f"bad prompt ending in {row['id']}"
        assert words(row["answer"]) <= 12, f"answer too long in {row['id']}"

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["scenario_id"]][row["variant"]] = row

    assert len(grouped) == 64, f"expected 64 scenario packs, found {len(grouped)}"
    for scenario_id, pack in grouped.items():
        assert set(pack) == {"base", "paraphrase", "counterfactual", "unanswerable"}, scenario_id
        assert pack["base"]["context"] == pack["paraphrase"]["context"], scenario_id
        assert pack["base"]["question"] != pack["paraphrase"]["question"], scenario_id
        assert pack["base"]["answer"] == pack["paraphrase"]["answer"], scenario_id
        assert pack["base"]["context"] != pack["counterfactual"]["context"], scenario_id
        assert pack["base"]["answer"] != pack["counterfactual"]["answer"], scenario_id
        assert pack["unanswerable"]["answer"] == "UNKNOWN", scenario_id
        assert pack["base"]["context"] != pack["unanswerable"]["context"], scenario_id

    summary = {
        "status": "PASS",
        "rows": len(rows),
        "scenario_packs": len(grouped),
        "unique_contexts": len({row["context_sha256"] for row in rows}),
        "context_words": {
            "min": min(row["context_words"] for row in rows),
            "median": statistics.median(row["context_words"] for row in rows),
            "mean": round(statistics.mean(row["context_words"] for row in rows), 2),
            "max": max(row["context_words"] for row in rows),
        },
        "splits": dict(Counter(row["split"] for row in rows)),
        "variants": dict(Counter(row["variant"] for row in rows)),
        "domains": dict(Counter(row["domain"] for row in rows)),
        "task_types": dict(Counter(row["task_type"] for row in rows)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
