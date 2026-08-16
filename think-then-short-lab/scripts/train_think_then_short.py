#!/usr/bin/env python3
"""Stub trainer for Qwen3-native think-then-short SFT.

Future behavior (not enabled yet):
  - Initialize from the configured greedy-soup / frozen-anchor checkpoint
  - Render assistant targets with apply_chat_template(..., enable_thinking=True)
  - Map record.thinking -> think channel; record.answer -> short visible reply
  - Upweight answer tokens similar to official first_answer_token_weight
  - Write adapters only under this lab's artifacts/ directory

This stub documents the contract and refuses GPU work until training is enabled.
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
    parser.add_argument("--train-file", type=Path, required=True, help="Converted think-then-short JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Lab-local output directory under artifacts/")
    parser.add_argument("--student", type=str, default=None, help="Override init checkpoint path")
    parser.add_argument("--enable-training", action="store_true", help="Reserved; training is not implemented yet")
    args = parser.parse_args()

    cfg = read_json(args.config)
    training_cfg = cfg.get("training_future") or {}
    thinking_cfg = cfg.get("thinking") or {}
    student_cfg = cfg.get("student") or {}

    plan = {
        "status": "not_enabled",
        "enable_thinking": bool(thinking_cfg.get("enable_thinking", True)),
        "mechanism": thinking_cfg.get("mechanism"),
        "max_thinking_tokens": thinking_cfg.get("max_thinking_tokens"),
        "max_answer_tokens": thinking_cfg.get("max_answer_tokens"),
        "student_init": args.student or student_cfg.get("default_init"),
        "train_file": str(args.train_file),
        "output_dir": str(args.output_dir),
        "lora": training_cfg.get("lora", True),
        "answer_token_upweight": training_cfg.get("answer_token_upweight"),
        "notes": training_cfg.get("notes") or [],
        "firewall": [
            "Do not write into official Phase artifacts/",
            "Do not flip enable_thinking in short-response-model-phase-0-1-2",
            "Do not consume sealed benchmarks",
            "Keep GPU exclusive when this trainer is eventually enabled",
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.train_file.exists():
        raise FileNotFoundError(args.train_file)
    if args.enable_training or bool(training_cfg.get("enabled")):
        raise RuntimeError(
            "think-then-short training is scaffolded but not enabled yet. "
            "Unset --enable-training / config.training_future.enabled until the LoRA trainer is implemented."
        )
    raise SystemExit(
        "train_think_then_short.py: not enabled yet. "
        "Convert datasets with convert_to_think_then_short.py and keep artifacts under this lab only."
    )


if __name__ == "__main__":
    main()
