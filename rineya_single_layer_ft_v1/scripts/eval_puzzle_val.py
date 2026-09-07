#!/usr/bin/env python3
"""Generate puzzle predictions and score with Puzzle-51K evaluate_predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    abstention_overrides,
    assert_gate_disabled,
    atomic_write_json,
    load_yaml,
    read_jsonl,
    resolve_core_path,
    resolve_repo_path,
    write_jsonl,
)

PUZZLE_SCRIPTS = REPO_ROOT / "Rineya-Puzzle-Reasoning-51k" / "scripts"
if str(PUZZLE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PUZZLE_SCRIPTS))

import evaluate_predictions as puzzle_eval  # noqa: E402
import generate_dataset as g  # noqa: E402


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(
    model_path: Path,
    adapter_dir: Path | None,
    *,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **abstention_overrides(),
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    assert_gate_disabled(model)
    return model, tokenizer


def generate_predictions(
    *,
    model: Any,
    tokenizer: Any,
    gold_rows: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, str]]:
    device = next(model.parameters()).device
    preds: list[dict[str, str]] = []
    for row in gold_rows:
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the reasoning problem carefully. Use the reasoning channel "
                    "to verify each constraint, then give a very short final answer "
                    "in the requested format."
                ),
            },
            {"role": "user", "content": row["prompt"]},
        ]
        rendered = apply_chat_template(tokenizer, messages)
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_ids = out[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        preds.append({"id": row["id"], "prediction": text})
    return preds


def score_predictions(gold_path: Path, pred_path: Path) -> dict[str, Any]:
    gold = {row["id"]: row for row in read_jsonl(gold_path)}
    preds: dict[str, str] = {}
    for row in read_jsonl(pred_path):
        preds[row["id"]] = str(row.get("prediction", row.get("output", "")))

    correct = 0
    total = len(gold)
    by_family: dict[str, list[int]] = {}
    for rid, row in gold.items():
        pred = puzzle_eval.extract_final(preds.get(rid, ""))
        norm = g.normalize_answer(pred)
        ok = any(norm == g.normalize_answer(a) for a in row["accepted_answers"])
        correct += int(ok)
        bucket = by_family.setdefault(row["family"], [0, 0])
        bucket[0] += int(ok)
        bucket[1] += 1
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_family": {
            name: {"correct": c, "total": n, "accuracy": round(c / n, 6) if n else 0.0}
            for name, (c, n) in sorted(by_family.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_lora_puzzle_smoke.yaml",
    )
    parser.add_argument(
        "--condition",
        choices=("baseline", "layer8_lora", "layer8_qlora"),
        required=True,
    )
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = args.output_dir or resolve_repo_path(cfg["train"]["output_dir"])
    gold_default = None
    if cfg.get("eval", {}).get("puzzle_gold"):
        gold_default = resolve_repo_path(cfg["eval"]["puzzle_gold"])
    gold_path = args.gold or gold_default or (output_root / "views" / "val_easy_gold.jsonl")
    if not gold_path.exists():
        # Fall back to prior puzzle smoke gold.
        alt = resolve_repo_path(
            "artifacts/rineya_single_layer_ft_v1/layer8_lora_puzzle_smoke/views/val_easy_gold.jsonl"
        )
        if alt.exists():
            gold_path = alt
        else:
            raise FileNotFoundError(gold_path)

    adapter_dir = None
    if args.condition in {"layer8_lora", "layer8_qlora"}:
        adapter_dir = args.adapter_dir or (output_root / "adapter")
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)

    dtype = torch.float16
    model_path = resolve_core_path(cfg["model"]["path"])
    model, tokenizer = load_model(model_path, adapter_dir, dtype=dtype)
    gold_rows = read_jsonl(gold_path)
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        gold_rows=gold_rows,
        max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
    )

    cond_dir = output_root / "eval" / args.condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cond_dir / "predictions.jsonl"
    write_jsonl(pred_path, preds)
    report = score_predictions(gold_path, pred_path)
    report["condition"] = args.condition
    report["adapter_dir"] = str(adapter_dir) if adapter_dir else None
    report["gold_path"] = str(gold_path)
    report["n_predictions"] = len(preds)
    atomic_write_json(cond_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
