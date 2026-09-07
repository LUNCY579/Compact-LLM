#!/usr/bin/env python3
"""Generate CC0 predictions and score exact/trim by stratum (intact vs adapter)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
HALO_SCRIPTS = REPO_ROOT / "rineya_halo_long_context_sft_v1" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    atomic_write_json,
    load_yaml,
    read_jsonl,
    resolve_repo_path,
    write_jsonl,
)

_spec = importlib.util.spec_from_file_location("halo_common", HALO_SCRIPTS / "common.py")
assert _spec is not None and _spec.loader is not None
halo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(halo)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(model_path: Path, adapter_dir: Path | None, dtype: torch.dtype) -> tuple[Any, Any]:
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
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def gold_target(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    return str(messages[-1].get("content") or "")


def score_rows(
    records: list[dict[str, Any]],
    preds: dict[str, str],
) -> dict[str, Any]:
    exact_by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trim_by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    exact_correct = 0
    trim_correct = 0
    total = 0
    per_item: list[dict[str, Any]] = []
    for row in records:
        rid = str(row["id"])
        if rid not in preds:
            continue
        expected = gold_target(row)
        pred = preds[rid]
        stratum = str((row.get("metadata") or {}).get("context_stratum") or "unknown")
        exact = halo.exact_match(pred, expected, "strict")
        trim = halo.exact_match(pred, expected, "trim")
        exact_correct += int(exact)
        trim_correct += int(trim)
        total += 1
        exact_by[stratum][0] += int(exact)
        exact_by[stratum][1] += 1
        trim_by[stratum][0] += int(trim)
        trim_by[stratum][1] += 1
        per_item.append({"id": rid, "stratum": stratum, "exact": exact, "trim": trim})

    def summarize(bucket: dict[str, list[int]]) -> dict[str, Any]:
        return {
            key: {
                "correct": value[0],
                "total": value[1],
                "accuracy": (value[0] / value[1]) if value[1] else 0.0,
            }
            for key, value in sorted(bucket.items())
        }

    return {
        "exact_accuracy": (exact_correct / total) if total else 0.0,
        "trim_accuracy": (trim_correct / total) if total else 0.0,
        "exact_correct": exact_correct,
        "trim_correct": trim_correct,
        "total": total,
        "exact_by_stratum": summarize(exact_by),
        "trim_by_stratum": summarize(trim_by),
        "per_item": per_item,
    }


def apply_stratum_caps(
    records: list[dict[str, Any]],
    caps: dict[str, int] | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not caps:
        return records
    import random

    rng = random.Random(seed)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        stratum = str((row.get("metadata") or {}).get("context_stratum") or "unknown")
        by[stratum].append(row)
    kept: list[dict[str, Any]] = []
    for stratum, rows in by.items():
        rows = sorted(rows, key=lambda r: str(r["id"]))
        cap = caps.get(stratum)
        if cap is None:
            kept.extend(rows)
        elif int(cap) <= 0:
            kept.extend(rows)
        elif len(rows) > int(cap):
            kept.extend(rng.sample(rows, int(cap)))
        else:
            kept.extend(rows)
    kept.sort(key=lambda r: str(r["id"]))
    return kept


def left_truncate_ids(input_ids: list[int], max_tokens: int) -> list[int]:
    if max_tokens <= 0 or len(input_ids) <= max_tokens:
        return input_ids
    return input_ids[-max_tokens:]


def generate_predictions(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_new_tokens: int,
    max_prompt_tokens: int,
    resume_path: Path | None,
) -> list[dict[str, str]]:
    done: dict[str, str] = {}
    if resume_path is not None and resume_path.exists():
        for row in read_jsonl(resume_path):
            done[str(row["id"])] = str(row["response"])

    device = next(model.parameters()).device
    out_rows: list[dict[str, str]] = []
    for i, row in enumerate(records, 1):
        rid = str(row["id"])
        if rid in done:
            out_rows.append({"id": rid, "response": done[rid]})
            continue
        messages = list(row["messages"][:-1])
        rendered = apply_chat_template(tokenizer, messages)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        ids = left_truncate_ids(ids, max_prompt_tokens)
        encoded = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=device),
        }
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_ids = generated[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        out_rows.append({"id": rid, "response": text})
        if resume_path is not None:
            with resume_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"id": rid, "response": text}, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            print(f"predicted {i}/{len(records)}", flush=True)
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_qlora_cc0_8k.yaml",
    )
    parser.add_argument("--condition", choices=("baseline", "layer8_qlora"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = resolve_repo_path(cfg["train"]["output_dir"])
    views = output_root / "views"
    records_path = views / ("val_all_strata.jsonl" if args.split == "val" else "test_all_strata.jsonl")
    records = read_jsonl(records_path)
    caps_key = "val_per_stratum_cap" if args.split == "val" else "test_per_stratum_cap"
    caps = cfg.get("eval", {}).get(caps_key)
    records = apply_stratum_caps(
        records, caps, seed=int(cfg["train"]["seed"]) + (1 if args.split == "val" else 2)
    )
    if args.limit > 0:
        records = records[: args.limit]

    adapter_dir = None
    if args.condition == "layer8_qlora":
        adapter_dir = args.adapter_dir or (output_root / "adapter")
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)

    out_dir = output_root / "eval_cc0" / args.split / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if not args.resume and pred_path.exists():
        pred_path.unlink()

    model_path = resolve_repo_path(cfg["model"]["path"])
    model, tokenizer = load_model(model_path, adapter_dir, torch.float16)
    preds_list = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
        max_prompt_tokens=int(cfg["eval"].get("max_prompt_tokens") or cfg["data"]["max_sequence_length"]),
        resume_path=pred_path if args.resume else None,
    )
    if not args.resume:
        write_jsonl(pred_path, preds_list)
    # Re-read resume file so scoring covers all written rows.
    if args.resume and pred_path.exists():
        preds_list = [{"id": r["id"], "response": r["response"]} for r in read_jsonl(pred_path)]
    preds = {row["id"]: row["response"] for row in preds_list}
    report = score_rows(records, preds)
    report["condition"] = args.condition
    report["split"] = args.split
    report["adapter_dir"] = str(adapter_dir) if adapter_dir else None
    report["n_records"] = len(records)
    report["stratum_caps"] = caps
    atomic_write_json(out_dir / "report.json", report)
    slim = {k: v for k, v in report.items() if k != "per_item"}
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
