#!/usr/bin/env python3
"""Run a local Hugging Face causal language model on the prompt-only JSONL files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"{path}:{line_no}: every row must be an object with an id")
            rows.append(row)
    return rows


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in load_jsonl(path):
        if row.get("status", "ok") == "ok":
            completed.add(str(row["id"]))
    return completed


def render_chat(tokenizer: Any, row: dict[str, Any], enable_thinking: bool | None = None) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
            if isinstance(message, dict)
            and message.get("role") in {"system", "user"}
            and "content" in message
        ]
    else:
        messages = []

    if messages and getattr(tokenizer, "chat_template", None):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    if messages:
        parts = [f"{message['role'].upper()}\n{message['content']}" for message in messages]
        return "\n\n".join(parts) + "\n\nASSISTANT\n"

    if "prompt" in row:
        return str(row["prompt"])
    raise ValueError(f"Row {row.get('id')} has neither messages nor prompt")


def infer_context_limit(model: Any, tokenizer: Any, override: int | None) -> int | None:
    if override:
        return override

    candidates: list[int] = []
    config_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(config_limit, int) and 0 < config_limit < 10_000_000:
        candidates.append(config_limit)

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        candidates.append(tokenizer_limit)

    return min(candidates) if candidates else None


def input_device(model: Any):
    device = model.get_input_embeddings().weight.device
    if str(device) == "meta":
        device = model.device
    return device


def synchronize_if_needed(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face model id")
    parser.add_argument("--input", type=Path, required=True, help="Prompt-only JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-context", type=int, help="Override inferred context limit")
    parser.add_argument(
        "--on-overflow",
        choices=("error", "skip"),
        default="error",
        help="Never silently truncate an evaluation prompt",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to Qwen3 chat templates so the 24-token budget is not consumed by <think>.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional PEFT adapter directory loaded onto --model (no full merge).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Override checkpoint dtype at load time.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load NF4 4-bit weights (needed for Qwen3-4B/8B on 8GB GPUs).",
    )
    parser.add_argument(
        "--max-gpu-memory",
        help='Cap GPU weight placement, e.g. "3GiB". Remainder goes to CPU. Use when 4-bit is unavailable.',
    )
    parser.add_argument(
        "--max-cpu-memory",
        default="48GiB",
        help="CPU memory budget when --max-gpu-memory is set.",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Install requirements first: pip install -r requirements.txt"
        ) from exc

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("Install bitsandbytes and a CUDA transformers build for --load-in-4bit") from exc
        compute = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype_map[args.torch_dtype]
    if args.max_gpu_memory:
        model_kwargs["max_memory"] = {0: args.max_gpu_memory, "cpu": args.max_cpu_memory}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )

    if args.adapter:
        adapter = args.adapter.expanduser().resolve()
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"Missing adapter_config.json in {adapter}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Install peft to use --adapter") from exc
        model = PeftModel.from_pretrained(model, str(adapter))

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    context_limit = infer_context_limit(model, tokenizer, args.max_context)
    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(args.output) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"

    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            row_id = str(row["id"])
            if row_id in completed:
                continue

            try:
                rendered = render_chat(
                    tokenizer,
                    row,
                    enable_thinking=False if args.disable_thinking else None,
                )
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_tokens = int(encoded["input_ids"].shape[-1])

                if context_limit and input_tokens + args.max_new_tokens > context_limit:
                    message = (
                        f"{row_id}: {input_tokens} input tokens + "
                        f"{args.max_new_tokens} generated tokens exceed context limit "
                        f"{context_limit}"
                    )
                    if args.on_overflow == "error":
                        raise RuntimeError(message)
                    result = {
                        "id": row_id,
                        "status": "skipped_context_overflow",
                        "prediction": "",
                        "input_tokens": input_tokens,
                        "context_limit": context_limit,
                        "error": message,
                    }
                else:
                    device = input_device(model)
                    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

                    synchronize_if_needed(torch)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    synchronize_if_needed(torch)
                    latency = time.perf_counter() - started

                    prompt_length = int(encoded["input_ids"].shape[-1])
                    continuation = generated[0, prompt_length:]
                    prediction = tokenizer.decode(
                        continuation,
                        skip_special_tokens=True,
                    ).strip()

                    result = {
                        "id": row_id,
                        "status": "ok",
                        "prediction": prediction,
                        "input_tokens": input_tokens,
                        "generated_tokens": int(continuation.shape[-1]),
                        "latency_seconds": round(latency, 6),
                        "seed": args.seed,
                        "decoding": "greedy",
                    }

            except Exception as exc:
                if not args.continue_on_error:
                    raise
                result = {
                    "id": row_id,
                    "status": "error",
                    "prediction": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }

            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {row_id}: {result['status']}")


if __name__ == "__main__":
    main()
