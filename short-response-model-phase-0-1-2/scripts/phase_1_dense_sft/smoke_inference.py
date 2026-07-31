from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.response_control import (
    RESPONSE_CONTROL_MODES,
    build_stopping_criteria,
    finalize_response,
    infer_stop_reason,
)

# Default to the smallest Qwen3 checkpoint used for the first local
# feasibility test. CLI arguments can override it.
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


# Expose the model, prompt, generation cap, and optional JSON output
# without requiring edits to the source file.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local short-response smoke test.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default="I finally finished the project.")
    parser.add_argument(
        "--system-prompt",
        default=(
            "Reply like a casual text message. "
            "Use only a few words or one short sentence."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--response-control", choices=RESPONSE_CONTROL_MODES, default="one_sentence"
    )
    parser.add_argument("--min-new-tokens-before-boundary", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


# CUDA launches work asynchronously. Synchronization ensures the measured
# time includes completed GPU work rather than only command submission.
def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Run one deterministic end-to-end inference and report basic resource use.
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix PyTorch/GPU setup first.")

    # Reset the per-process peak counter so the reported VRAM belongs to
    # this smoke test rather than earlier GPU operations.
    torch.cuda.reset_peak_memory_stats()
    # Load the tokenizer and model from the same checkpoint so token IDs,
    # special tokens, and the chat template remain compatible.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    # Evaluation mode disables training-only behavior such as dropout.
    model.eval()

    # Represent the request as structured chat roles before applying the
    # model-specific prompt format.
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    # Render Qwen's official chat template and explicitly disable its
    # optional thinking mode for the short-response workload.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Convert text to tensors and move them to the model's device.
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

    stopping_criteria, boundary_criterion = build_stopping_criteria(
        tokenizer,
        int(inputs.input_ids.shape[1]),
        args.response_control,
        args.min_new_tokens_before_boundary,
    )
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria

    sync_cuda()
    started = time.perf_counter()
    # Inference mode avoids gradient tracking, reducing memory and overhead.
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    sync_cuda()
    elapsed = time.perf_counter() - started

    # model.generate returns prompt + completion. Slice away the prompt so
    # decoding and token counts describe only the newly generated reply.
    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    finalized = finalize_response(raw_reply, args.response_control)
    reply = finalized.text
    stop_reason = infer_stop_reason(
        new_tokens.tolist(),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        criterion=boundary_criterion,
    )
    # Capture enough metadata to confirm local execution and compare later
    # smoke tests.
    result = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "reply": reply,
        "raw_reply": finalized.raw_text,
        "response_control_mode": args.response_control,
        "response_control_trimmed": finalized.trimmed,
        "stop_reason": stop_reason,
        "sentence_count": finalized.sentence_count,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "generated_tokens": int(new_tokens.shape[0]),
        "visible_generated_tokens": len(tokenizer(reply, add_special_tokens=False)["input_ids"]),
        "generation_seconds": elapsed,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "cuda_device": torch.cuda.get_device_name(0),
    }

    print(f"Reply: {reply}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Generated tokens: {result['generated_tokens']}")
    print(f"Stop reason: {result['stop_reason']}")
    print(f"Generation time: {elapsed:.3f} seconds")
    print(f"Peak allocated VRAM: {result['peak_vram_gb']} GB")

    # Persist a machine-readable result only when requested.
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# Standard script entry point.
if __name__ == "__main__":
    main()
