#!/usr/bin/env python3
"""Score main-split predictions against gold final assistant turns."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import atomic_write_json, exact_match, infer_stratum, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score {id,response} predictions on SFT gold targets.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="")
    return parser.parse_args()


def gold_target(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{row.get('id')}: expected a final assistant target")
    return str(messages[-1].get("content") or "")


def summarize(bucket: dict[str, list[int]]) -> dict[str, Any]:
    return {
        key: {
            "correct": value[0],
            "total": value[1],
            "accuracy": (value[0] / value[1]) if value[1] else 0.0,
        }
        for key, value in sorted(bucket.items())
    }


def score(records_path: Path, predictions_path: Path, arm: str) -> dict[str, Any]:
    records = {str(row["id"]): row for row in iter_jsonl(records_path)}
    predictions = {str(row["id"]): row for row in iter_jsonl(predictions_path)}
    missing = sorted(set(records) - set(predictions))
    unexpected = sorted(set(predictions) - set(records))
    exact_by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trim_by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    exact_correct = 0
    trim_correct = 0
    total = 0
    failures: list[dict[str, Any]] = []
    for rid, row in records.items():
        pred = predictions.get(rid)
        if pred is None:
            continue
        expected = gold_target(row)
        observed = str(pred.get("response") or "")
        stratum = infer_stratum(row) or str((row.get("metadata") or {}).get("context_stratum") or "unknown")
        exact_ok = exact_match(observed, expected, "strict")
        trim_ok = exact_match(observed, expected, "trim")
        total += 1
        exact_correct += int(exact_ok)
        trim_correct += int(trim_ok)
        exact_by_stratum[stratum][1] += 1
        exact_by_stratum[stratum][0] += int(exact_ok)
        trim_by_stratum[stratum][1] += 1
        trim_by_stratum[stratum][0] += int(trim_ok)
        if not trim_ok and len(failures) < 50:
            failures.append({
                "id": rid,
                "expected": expected,
                "response": observed,
                "context_stratum": stratum,
            })
    return {
        "arm": arm,
        "records": str(records_path),
        "predictions": str(predictions_path),
        "total": total,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "exact": {
            "correct": exact_correct,
            "total": total,
            "accuracy": (exact_correct / total) if total else 0.0,
            "by_stratum": summarize(exact_by_stratum),
        },
        "trim": {
            "correct": trim_correct,
            "total": total,
            "accuracy": (trim_correct / total) if total else 0.0,
            "by_stratum": summarize(trim_by_stratum),
        },
        "first_50_trim_failures": failures,
    }


def main() -> None:
    args = parse_args()
    report = score(args.records, args.predictions, args.arm)
    atomic_write_json(args.output, report)
    print(
        json_line(report)
    )


def json_line(report: dict[str, Any]) -> str:
    import json

    return json.dumps(
        {
            "arm": report["arm"],
            "exact": report["exact"]["accuracy"],
            "trim": report["trim"]["accuracy"],
            "total": report["total"],
            "missing": report["missing"],
        }
    )


if __name__ == "__main__":
    main()
