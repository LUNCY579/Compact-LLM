from __future__ import annotations

"""Measure target-token NLL on the same 50-case structural probe.

Generation heuristics can saturate at 1.0. Teacher-forced negative log
likelihood supplies a continuous contrast signal without claiming that lower
NLL alone proves better human-facing behavior.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json


def apply_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate assistant-target NLL on an SFT-style JSONL probe.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None  # type: ignore[assignment]

    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable and --require-cuda was requested.")

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    if args.adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("Install peft before loading an adapter.")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    output_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_tokens = 0
    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{args.input}: record {row.get('id', index)} must end with an assistant target.")
        prompt_messages = messages[:-1]
        prompt_text = apply_template(tokenizer, prompt_messages, generation=True)
        full_text = apply_template(tokenizer, messages, generation=False)
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_sequence_length,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_sequence_length,
        )["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        # Token t is predicted by logits at t-1. Only assistant-target tokens
        # starting at prompt length participate in the metric.
        shifted_logits = logits[:, :-1, :].float()
        shifted_targets = input_ids[:, 1:]
        start = max(0, len(prompt_ids) - 1)
        target_logits = shifted_logits[:, start:, :]
        target_ids = shifted_targets[:, start:]
        token_losses = F.cross_entropy(
            target_logits.reshape(-1, target_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        )
        case_tokens = int(token_losses.numel())
        case_loss_sum = float(token_losses.sum().item())
        case_mean = case_loss_sum / max(1, case_tokens)
        total_loss += case_loss_sum
        total_tokens += case_tokens
        output_rows.append(
            {
                "id": row.get("id", f"row_{index}"),
                "target_tokens": case_tokens,
                "mean_target_nll": case_mean,
                "target_perplexity": math.exp(min(20.0, case_mean)),
            }
        )
        print(f"[{index:03d}/{len(rows):03d}] {output_rows[-1]['id']}: nll={case_mean:.5f}")

    if total_tokens == 0:
        raise RuntimeError("No assistant target tokens were evaluated.")
    mean_nll = total_loss / total_tokens
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        args.output_dir / "summary.json",
        {
            "model_id": args.model_id,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "input": str(args.input),
            "case_count": len(output_rows),
            "target_token_count": total_tokens,
            "mean_target_nll": mean_nll,
            "target_perplexity": math.exp(min(20.0, mean_nll)),
            "interpretation_boundary": (
                "NLL is a continuous sensitivity signal for saturated generation scores. "
                "It does not replace generated-output, category, safety, or human review."
            ),
        },
    )
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
