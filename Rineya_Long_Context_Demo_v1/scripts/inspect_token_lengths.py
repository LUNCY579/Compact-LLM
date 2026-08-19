#!/usr/bin/env python3
"""Measure exact prompt token lengths with the tokenizer used by a local model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
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
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def render_chat(tokenizer: Any, row: dict[str, Any], enable_thinking: bool | None = None) -> str:
    messages = row.get("messages", [])
    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"system", "user"}
        and "content" in message
    ]
    if messages and getattr(tokenizer, "chat_template", None):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    if messages:
        return "\n\n".join(
            f"{message['role'].upper()}\n{message['content']}" for message in messages
        ) + "\n\nASSISTANT\n"
    return str(row["prompt"])


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="Local model/tokenizer directory or HF id")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to Qwen3 chat templates.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    rows = load_jsonl(args.input)

    per_item: list[dict[str, Any]] = []
    for row in rows:
        rendered = render_chat(
            tokenizer,
            row,
            enable_thinking=False if args.disable_thinking else None,
        )
        count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        per_item.append(
            {
                "id": row["id"],
                "tokens": count,
                "length_bucket": row.get("length_bucket"),
                "needle_profile": row.get("needle_profile"),
            }
        )

    counts = [item["tokens"] for item in per_item]
    inferred = getattr(tokenizer, "model_max_length", None)
    if not isinstance(inferred, int) or inferred >= 10_000_000:
        inferred = None
    max_context = args.max_context or inferred

    overflow_ids: list[str] = []
    if max_context:
        overflow_ids = [
            item["id"]
            for item in per_item
            if item["tokens"] + args.max_new_tokens > max_context
        ]

    bucket_stats: dict[str, dict[str, float | int]] = {}
    for bucket in sorted({str(item["length_bucket"]) for item in per_item}):
        values = [item["tokens"] for item in per_item if str(item["length_bucket"]) == bucket]
        bucket_stats[bucket] = {
            "items": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "mean": round(statistics.mean(values), 2),
            "p90": percentile(values, 0.90),
            "max": max(values),
        }

    report = {
        "items": len(per_item),
        "tokenizer": args.tokenizer,
        "max_context": max_context,
        "max_new_tokens": args.max_new_tokens,
        "tokens": {
            "min": min(counts),
            "median": statistics.median(counts),
            "mean": round(statistics.mean(counts), 2),
            "p90": percentile(counts, 0.90),
            "max": max(counts),
        },
        "by_length_bucket": bucket_stats,
        "overflow_count": len(overflow_ids),
        "overflow_ids": overflow_ids,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
