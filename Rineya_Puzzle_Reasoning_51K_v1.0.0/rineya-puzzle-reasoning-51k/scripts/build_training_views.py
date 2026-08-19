#!/usr/bin/env python3
"""Convert base rows into common SFT/distillation training views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_REASONED = (
    "Solve the reasoning problem carefully. Use the reasoning channel to verify each constraint, "
    "then give a very short final answer in the requested format."
)
SYSTEM_SHORT = "Solve the problem and return only the short final answer requested by the user."


def reasoned_target(row: dict, tags: str) -> str:
    reasoning = "\n".join(f"{i+1}. {step}" for i, step in enumerate(row["reasoning_steps"]))
    if tags == "think":
        return f"<think>\n{reasoning}\n</think>\n{row['final_answer']}"
    if tags == "reasoning":
        return f"<reasoning>\n{reasoning}\n</reasoning>\n<final>{row['final_answer']}</final>"
    return f"Reasoning:\n{reasoning}\nFinal: {row['final_answer']}"


def convert(row: dict, view: str, tags: str):
    meta = {
        "id": row["id"], "family": row["family"], "difficulty": row["difficulty"],
        "split": row["split"], "answer_type": row["answer_type"],
    }
    if view == "think_then_short":
        return {**meta, "messages": [
            {"role": "system", "content": SYSTEM_REASONED},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": reasoned_target(row, tags)},
        ]}
    if view == "short_only":
        return {**meta, "messages": [
            {"role": "system", "content": SYSTEM_SHORT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["final_answer"]},
        ]}
    if view == "dual_fields":
        return {**meta,
            "messages": [
                {"role": "system", "content": SYSTEM_REASONED},
                {"role": "user", "content": row["prompt"]},
            ],
            "analysis_target": "\n".join(row["reasoning_steps"]),
            "final_target": row["final_answer"],
            "compact_state": row["compact_state"],
        }
    if view == "preference_pair":
        # Useful for DPO/ORPO after SFT: concise correct answer is preferred over
        # an otherwise correct but unnecessarily verbose visible response.
        verbose = reasoned_target(row, "none")
        return {**meta,
            "prompt": row["prompt"],
            "chosen": row["final_answer"],
            "rejected": verbose,
            "preference_reason": "correct_concise_final_over_visible_reasoning",
        }
    raise ValueError(view)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--view", choices=["think_then_short", "short_only", "dual_fields", "preference_pair"], required=True)
    p.add_argument("--reasoning-tags", choices=["think", "reasoning", "none"], default="think")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(convert(row, args.view, args.reasoning_tags), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"view": args.view, "rows": count, "output": str(args.output)}, indent=2))

if __name__ == "__main__":
    main()
