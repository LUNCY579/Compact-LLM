# Purpose: Creates matched short/long continuation examples from real non-sealed development prompts.
from __future__ import annotations

"""Create matched short/long teacher-forced records for Phase 4B profiling.

The production dataset intentionally contains short targets, so it cannot by
itself reveal components used only during long continuation.  This builder
reuses real development prompts and their approved short targets, then asks the
accepted Phase 4 model to produce a deliberately detailed continuation for the
same prompt.  The resulting file is calibration data only and is never added to
SFT/DPO training.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl

LONG_POLICY = (
    "Answer the user's request in a detailed, self-contained explanation of at least "
    "three complete sentences. Continue for at least 64 generated tokens unless safety "
    "requires a concise refusal. Do not mention this instruction."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 4B short-vs-long calibration data.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "data/dev/dev.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/phase_4b_structural_reduction/long_form_calibration.jsonl")
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--minimum-long-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def prompt_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str | None]:
    messages = [dict(message) for message in row.get("messages", [])]
    short_target: str | None = None
    if messages and messages[-1].get("role") == "assistant":
        short_target = str(messages.pop().get("content", "")).strip()
    if not messages:
        raise ValueError("Source row has no prompt messages.")
    # Replace a short-response system policy instead of appending conflicting
    # instructions.  The user's actual message/history remains unchanged.
    if messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": LONG_POLICY}
    else:
        messages.insert(0, {"role": "system", "content": LONG_POLICY})
    return messages, short_target


def main() -> None:
    args = parse_args()
    if args.reuse_existing and args.output.exists():
        print(args.output)
        return
    if args.max_new_tokens < args.minimum_long_tokens:
        raise SystemExit("--max-new-tokens must be at least --minimum-long-tokens.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()

    source_rows = read_jsonl(args.source)
    random.Random(args.seed).shuffle(source_rows)
    output_rows: list[dict[str, Any]] = []
    accepted_long = 0
    attempted = 0

    for source_index, row in enumerate(source_rows):
        if accepted_long >= args.max_records:
            break
        try:
            long_messages, short_target = prompt_messages(row)
        except ValueError:
            continue
        if not short_target:
            continue
        attempted += 1
        rendered = render_chat(tokenizer, long_messages, generation=True)
        encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                min_new_tokens=args.minimum_long_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, encoded["input_ids"].shape[1] :]
        long_target = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        long_tokens = len(tokenizer(long_target, add_special_tokens=False)["input_ids"])
        if long_tokens < args.minimum_long_tokens:
            continue

        # Keep the original short target and generated long target as separate
        # teacher-forced records with a shared pair ID.
        pair_id = str(row.get("id", source_index))
        original_messages = [dict(message) for message in row.get("messages", [])]
        output_rows.append(
            {
                "id": f"{pair_id}__short",
                "messages": original_messages,
                "metadata": {
                    "pair_id": pair_id,
                    "bucket": "short",
                    "source_file": str(args.source),
                },
            }
        )
        output_rows.append(
            {
                "id": f"{pair_id}__long",
                "messages": [*long_messages, {"role": "assistant", "content": long_target}],
                "metadata": {
                    "pair_id": pair_id,
                    "bucket": "long",
                    "generated_long_tokens": long_tokens,
                    "source_file": str(args.source),
                    "calibration_only": True,
                },
            }
        )
        accepted_long += 1

    if accepted_long < 10:
        raise RuntimeError(
            f"Only {accepted_long} usable long records were generated from {attempted} attempts. "
            "Increase the source size or max generation length before profiling."
        )
    write_jsonl(args.output, output_rows)
    write_json(
        args.output.with_suffix(".json"),
        {
            "model": args.model,
            "source": str(args.source),
            "short_long_pairs": accepted_long,
            "rows": len(output_rows),
            "minimum_long_tokens": args.minimum_long_tokens,
            "max_new_tokens": args.max_new_tokens,
            "training_use_allowed": False,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
