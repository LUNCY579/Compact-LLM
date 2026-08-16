#!/usr/bin/env python3
"""Stub evaluator for Qwen3-native think-then-short decoding.

Future behavior (not enabled yet):
  - Prompt with enable_thinking=True
  - Allow enough max_new_tokens for the think channel
  - Parse think vs visible answer
  - Score answer only (exact match / short-response rules)
  - Report thinking_tokens and answer_tokens separately
  - Fail rows whose answer exceeds max_answer_tokens
  - Use non-sealed probes only

This stub documents the contract and refuses GPU work until evaluation is enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/think_then_short.json"))
    parser.add_argument("--model", type=str, required=True, help="Checkpoint or adapter under this lab")
    parser.add_argument("--probe", type=Path, required=True, help="Converted think-then-short JSONL probe")
    parser.add_argument("--output-dir", type=Path, required=True, help="Lab-local eval output directory")
    parser.add_argument("--enable-eval", action="store_true", help="Reserved; evaluation is not implemented yet")
    args = parser.parse_args()

    cfg = read_json(args.config)
    eval_cfg = cfg.get("evaluation_future") or {}
    thinking_cfg = cfg.get("thinking") or {}

    plan = {
        "status": "not_enabled",
        "enable_thinking": bool(thinking_cfg.get("enable_thinking", True)),
        "score_answer_only": bool(eval_cfg.get("score_answer_only", True)),
        "report_thinking_tokens": bool(eval_cfg.get("report_thinking_tokens", True)),
        "report_answer_tokens": bool(eval_cfg.get("report_answer_tokens", True)),
        "max_thinking_tokens": thinking_cfg.get("max_thinking_tokens"),
        "max_answer_tokens": thinking_cfg.get("max_answer_tokens"),
        "model": args.model,
        "probe": str(args.probe),
        "output_dir": str(args.output_dir),
        "notes": eval_cfg.get("notes") or [],
        "firewall": [
            "Score the post-think answer only",
            "Do not use official sealed benchmarks",
            "Do not write into official Phase artifacts/",
            "Keep GPU exclusive when this evaluator is eventually enabled",
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.probe.exists():
        raise FileNotFoundError(args.probe)
    if "sealed" in str(args.probe).casefold():
        raise ValueError("think-then-short evaluator refuses sealed probes")
    if args.enable_eval or bool(eval_cfg.get("enabled")):
        raise RuntimeError(
            "think-then-short evaluation is scaffolded but not enabled yet. "
            "Unset --enable-eval / config.evaluation_future.enabled until decode+score is implemented."
        )
    raise SystemExit(
        "evaluate_think_then_short.py: not enabled yet. "
        "Convert probes with convert_to_think_then_short.py; score answer only when enabled."
    )


if __name__ == "__main__":
    main()
