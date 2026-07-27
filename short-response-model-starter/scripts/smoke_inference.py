from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


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
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix PyTorch/GPU setup first.")

    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync_cuda()
    elapsed = time.perf_counter() - started

    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    result = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "reply": reply,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "generated_tokens": int(new_tokens.shape[0]),
        "generation_seconds": elapsed,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "cuda_device": torch.cuda.get_device_name(0),
    }

    print(f"Reply: {reply}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Generated tokens: {result['generated_tokens']}")
    print(f"Generation time: {elapsed:.3f} seconds")
    print(f"Peak allocated VRAM: {result['peak_vram_gb']} GB")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
