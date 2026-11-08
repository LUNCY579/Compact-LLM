#!/usr/bin/env python3
"""Greedy long-context predictions with durable resume-by-id JSONL output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (
    LAB_ROOT,
    append_jsonl,
    iter_jsonl,
    load_config,
    resolve_repo_path,
    stage_paths,
)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict long-context answers as {id, response} JSONL.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "stage_a_qlora.yaml")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--model-id", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--torch-dtype", default=None)
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = list(row.get("messages") or [])
    if not messages:
        raise ValueError(f"{row.get('id')}: missing messages")
    if messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        rid = row.get("id")
        if isinstance(rid, str):
            done.add(rid)
    return done


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    decode = config.get("decode") or {}
    training = config.get("training") or {}
    model_id = resolve_repo_path(args.model_id or config["paths"]["model_id"])
    max_new_tokens = int(args.max_new_tokens or decode.get("max_new_tokens") or 32)
    max_length = int(args.max_sequence_length or training.get("max_sequence_length") or 4096)
    dtype_name = str(args.torch_dtype or decode.get("torch_dtype") or "auto")
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    records = list(iter_jsonl(args.records))
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    done = load_completed(args.output) if args.resume else set()
    if not args.resume and args.output.exists():
        raise FileExistsError(f"{args.output} exists; pass --resume to continue")

    tokenizer = AutoTokenizer.from_pretrained(str(model_id))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_id),
        torch_dtype=dtype_map[dtype_name],
        device_map="auto",
    )
    if args.adapter is not None:
        if PeftModel is None:
            raise RuntimeError("Install peft before loading an adapter.")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    device = next(model.parameters()).device

    remaining = [row for row in records if str(row.get("id")) not in done]
    print(f"predict records={len(records)} resume_done={len(done)} remaining={len(remaining)}")
    for index, row in enumerate(remaining, 1):
        rid = str(row["id"])
        messages = prompt_messages(row)
        rendered = apply_chat_template(tokenizer, messages)
        encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(encoded) > max_length:
            encoded = encoded[-max_length:]
        inputs = {
            "input_ids": torch.tensor([encoded], dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, len(encoded), dtype=torch.long, device=device),
        }
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(decode.get("do_sample", False)),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        append_jsonl(
            args.output,
            {
                "id": rid,
                "response": response,
                "context_stratum": metadata.get("context_stratum"),
                "prompt_tokens": len(encoded),
            },
        )
        if index % 10 == 0 or index == len(remaining):
            print(f"[{index}/{len(remaining)}] {rid}")


if __name__ == "__main__":
    main()
