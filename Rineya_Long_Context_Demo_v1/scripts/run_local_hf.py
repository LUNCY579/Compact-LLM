#!/usr/bin/env python3
"""Run a local Hugging Face causal language model on the prompt-only JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

LORA_PROJECTION_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
LAYER_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def load_adapter_scale_profile(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate a deterministic LoRA scale-profile recipe."""

    raw = path.read_bytes()
    try:
        profile = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid adapter scale profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise TypeError("Adapter scale profile must be a JSON object")

    allowed = {
        "schema_version",
        "label",
        "default_multiplier",
        "layer_multipliers",
        "module_multipliers",
    }
    unknown = sorted(set(profile) - allowed)
    if unknown:
        raise ValueError(f"Unknown adapter scale profile fields: {unknown}")
    if str(profile.get("schema_version", "1.0")) != "1.0":
        raise ValueError("Adapter scale profile schema_version must be '1.0'")

    def finite_number(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{field} must be a finite number")
        return result

    default_multiplier = finite_number(
        profile.get("default_multiplier", 1.0), "default_multiplier"
    )
    layer_values = profile.get("layer_multipliers", {})
    if not isinstance(layer_values, dict):
        raise TypeError("layer_multipliers must be an object")
    layer_multipliers: dict[str, float] = {}
    for key, value in layer_values.items():
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid layer_multipliers key: {key!r}") from exc
        if layer < 0 or str(layer) != str(key):
            raise ValueError(f"Invalid layer_multipliers key: {key!r}")
        layer_multipliers[str(layer)] = finite_number(
            value, f"layer_multipliers.{layer}"
        )

    module_values = profile.get("module_multipliers", {})
    if not isinstance(module_values, dict):
        raise TypeError("module_multipliers must be an object")
    bad_modules = sorted(set(module_values) - LORA_PROJECTION_NAMES)
    if bad_modules:
        raise ValueError(f"Unknown module_multipliers keys: {bad_modules}")
    module_multipliers = {
        key: finite_number(value, f"module_multipliers.{key}")
        for key, value in module_values.items()
    }

    normalized = {
        "schema_version": "1.0",
        "label": str(profile.get("label", path.stem)),
        "default_multiplier": default_multiplier,
        "layer_multipliers": layer_multipliers,
        "module_multipliers": module_multipliers,
    }
    return normalized, hashlib.sha256(raw).hexdigest()


def adapter_scale_for_module(
    module_name: str, global_scale: float, profile: dict[str, Any] | None
) -> tuple[float, int | None, str]:
    """Resolve the absolute LoRA scale factor for one named projection module."""

    layer_match = LAYER_NAME_RE.search(module_name)
    layer = int(layer_match.group(1)) if layer_match else None
    projection = module_name.rsplit(".", 1)[-1]
    multiplier = 1.0
    if profile is not None:
        multiplier = float(profile["default_multiplier"])
        if layer is not None:
            multiplier = float(
                profile["layer_multipliers"].get(str(layer), multiplier)
            )
        multiplier *= float(profile["module_multipliers"].get(projection, 1.0))
    return global_scale * multiplier, layer, projection


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
        "--adapter-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the active LoRA adapter contribution by this finite factor. "
            "The base checkpoint is unchanged; 0 disables the adapter and 1 uses "
            "its configured scale."
        ),
    )
    parser.add_argument(
        "--adapter-scale-profile",
        type=Path,
        help=(
            "Optional JSON recipe containing default, per-layer, and per-projection "
            "multipliers applied on top of --adapter-scale."
        ),
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

    if not math.isfinite(args.adapter_scale):
        parser.error("--adapter-scale must be finite")
    if args.adapter is None and args.adapter_scale != 1.0:
        parser.error("--adapter-scale requires --adapter")
    if args.adapter is None and args.adapter_scale_profile is not None:
        parser.error("--adapter-scale-profile requires --adapter")

    adapter_scale_profile: dict[str, Any] | None = None
    adapter_scale_profile_sha256: str | None = None
    if args.adapter_scale_profile is not None:
        profile_path = args.adapter_scale_profile.expanduser().resolve(strict=True)
        try:
            adapter_scale_profile, adapter_scale_profile_sha256 = (
                load_adapter_scale_profile(profile_path)
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

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
            from peft.tuners.lora.layer import LoraLayer
        except ImportError as exc:
            raise SystemExit("Install peft to use --adapter") from exc
        model = PeftModel.from_pretrained(model, str(adapter))
        scaled_layers = 0
        observed_layers: set[int] = set()
        observed_modules: set[str] = set()
        effective_scales: list[float] = []
        for module_name, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            effective_scale, layer, projection = adapter_scale_for_module(
                module_name, args.adapter_scale, adapter_scale_profile
            )
            if layer is not None:
                observed_layers.add(layer)
            observed_modules.add(projection)
            for adapter_name in module.active_adapters:
                module.set_scale(adapter_name, effective_scale)
            effective_scales.append(effective_scale)
            scaled_layers += 1
        if scaled_layers == 0:
            raise SystemExit("The loaded adapter contains no LoRA layers to scale")
        if adapter_scale_profile is not None:
            requested_layers = {
                int(value) for value in adapter_scale_profile["layer_multipliers"]
            }
            missing_layers = sorted(requested_layers - observed_layers)
            if missing_layers:
                raise SystemExit(
                    f"Adapter scale profile references missing LoRA layers: {missing_layers}"
                )
            requested_modules = set(adapter_scale_profile["module_multipliers"])
            missing_modules = sorted(requested_modules - observed_modules)
            if missing_modules:
                raise SystemExit(
                    "Adapter scale profile references missing LoRA modules: "
                    f"{missing_modules}"
                )

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
                    if args.adapter:
                        result["adapter_scale"] = args.adapter_scale
                        result["adapter_effective_scale_min"] = min(effective_scales)
                        result["adapter_effective_scale_max"] = max(effective_scales)
                        if adapter_scale_profile is not None:
                            result["adapter_scale_profile"] = adapter_scale_profile["label"]
                            result["adapter_scale_profile_sha256"] = (
                                adapter_scale_profile_sha256
                            )

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
