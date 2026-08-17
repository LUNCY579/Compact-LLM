#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.mean(values),
        "over_32": sum(v > 32 for v in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure final-assistant target lengths with the exact local tokenizer.")
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True, help="Local model/tokenizer directory or Hugging Face identifier.")
    parser.add_argument("--allow-remote", action="store_true", help="Permit downloading when the tokenizer is not local.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers in the training environment first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=not args.allow_remote,
    )
    lengths: list[int] = []
    by_class: dict[str, list[int]] = defaultdict(list)
    with args.sft.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                raise SystemExit(f"Line {line_no} does not end in an assistant target.")
            target = messages[-1]["content"]
            length = len(tokenizer.encode(target, add_special_tokens=False))
            lengths.append(length)
            by_class[row.get("metadata", {}).get("reply_class", "unknown")].append(length)
    if not lengths:
        raise SystemExit("No rows found.")

    report = {
        "sft": str(args.sft.resolve()),
        "tokenizer": args.tokenizer,
        "overall": summarize(lengths),
        "by_reply_class": {key: summarize(value) for key, value in sorted(by_class.items())},
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
