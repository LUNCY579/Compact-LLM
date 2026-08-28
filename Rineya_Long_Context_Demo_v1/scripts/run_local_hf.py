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


UNKNOWN_SURFACE_FORMS = (
    "UNKNOWN",
    "unknown",
    "Unknown",
    " UNKNOWN",
    " unknown",
    " Unknown",
)


def unknown_continuation_tokens(tokenizer: Any) -> list[dict[str, Any]]:
    """First-token ids for the abstention surface forms, de-duplicated in order."""

    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for surface in UNKNOWN_SURFACE_FORMS:
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if not encoded:
            continue
        token_id = int(encoded[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        entries.append(
            {
                "surface": surface,
                "token_id": token_id,
                "single_token": len(encoded) == 1,
            }
        )
    if not entries:
        raise RuntimeError("Tokenizer produced no UNKNOWN continuation tokens")
    return entries


def step_confidence(torch: Any, step_logits: Any, chosen_id: int) -> dict[str, Any]:
    """Greedy-step diagnostics computed in float32 over the full vocabulary."""

    logprobs = torch.log_softmax(step_logits.to(torch.float32), dim=-1)
    probs = logprobs.exp()
    entropy = float(-(probs * torch.nan_to_num(logprobs, neginf=0.0)).sum())
    top2 = torch.topk(logprobs, 2)
    top1_logprob = float(top2.values[0])
    top2_logprob = float(top2.values[1])
    return {
        "token_logprob": float(logprobs[chosen_id]),
        "top1_logprob": top1_logprob,
        "top2_logprob": top2_logprob,
        "logprob_margin": top1_logprob - top2_logprob,
        "prob_margin": float(top2.values[0].exp() - top2.values[1].exp()),
        "entropy": entropy,
        "logprobs": logprobs,
    }


def summarize_confidence(
    *,
    torch: Any,
    tokenizer: Any,
    step_logits: Any,
    continuation: Any,
    top_k: int,
    unknown_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-row confidence signals derived from the single greedy decode pass."""

    steps = min(len(step_logits), int(continuation.shape[-1]))
    if steps == 0:
        raise RuntimeError("Generation produced no scored steps")

    eos_id = tokenizer.eos_token_id
    eos_ids = set(eos_id) if isinstance(eos_id, (list, tuple, set)) else {eos_id}

    content_length = steps
    eos_emitted = False
    for position in range(steps):
        if int(continuation[position]) in eos_ids:
            content_length = position
            eos_emitted = True
            break

    per_step: list[dict[str, Any]] = []
    first_extras: dict[str, Any] | None = None
    for position in range(steps):
        chosen_id = int(continuation[position])
        stats = step_confidence(torch, step_logits[position][0], chosen_id)
        logprobs = stats.pop("logprobs")
        if position == 0:
            ranked = torch.topk(logprobs, min(top_k, int(logprobs.shape[-1])))
            first_extras = {
                "top_k": [
                    {
                        "token_id": int(token_id),
                        "token": tokenizer.decode([int(token_id)]),
                        "logprob": float(logprob),
                    }
                    for token_id, logprob in zip(ranked.indices, ranked.values)
                ],
                "unknown_logprobs": {
                    entry["token_id"]: float(logprobs[entry["token_id"]])
                    for entry in unknown_tokens
                },
            }
        stats["position"] = position
        stats["token_id"] = chosen_id
        stats["is_eos"] = chosen_id in eos_ids
        per_step.append(stats)

    if first_extras is None:
        raise RuntimeError("First decode step was not scored")

    unknown_logprobs = first_extras["unknown_logprobs"]
    variant_logprob = float(
        torch.logsumexp(torch.tensor(list(unknown_logprobs.values())), dim=0)
    )
    exact_id = unknown_tokens[0]["token_id"]

    content_steps = per_step[:content_length]
    if content_steps:
        token_logprobs = [step["token_logprob"] for step in content_steps]
        margins = [step["logprob_margin"] for step in content_steps]
        entropies = [step["entropy"] for step in content_steps]
        answer = {
            "sum_logprob": sum(token_logprobs),
            "mean_logprob": sum(token_logprobs) / len(token_logprobs),
            "min_logprob": min(token_logprobs),
            "mean_logprob_margin": sum(margins) / len(margins),
            "min_logprob_margin": min(margins),
            "mean_entropy": sum(entropies) / len(entropies),
            "max_entropy": max(entropies),
        }
    else:
        answer = None

    first = per_step[0]
    return {
        "scored_steps": steps,
        "content_tokens": content_length,
        "eos_emitted": eos_emitted,
        "first_token": {
            "token_id": first["token_id"],
            "token": tokenizer.decode([first["token_id"]]),
            "is_eos": first["is_eos"],
            "token_logprob": first["token_logprob"],
            "top1_logprob": first["top1_logprob"],
            "top2_logprob": first["top2_logprob"],
            "logprob_margin": first["logprob_margin"],
            "prob_margin": first["prob_margin"],
            "entropy": first["entropy"],
            "top_k": first_extras["top_k"],
        },
        "unknown": {
            "exact_token_id": exact_id,
            "exact_logprob": unknown_logprobs[exact_id],
            "variant_logprob": variant_logprob,
            "per_token_logprob": {
                str(token_id): value for token_id, value in unknown_logprobs.items()
            },
        },
        "answer": answer,
        "per_step": [
            {
                "position": step["position"],
                "token_id": step["token_id"],
                "is_eos": step["is_eos"],
                "token_logprob": step["token_logprob"],
                "logprob_margin": step["logprob_margin"],
                "entropy": step["entropy"],
            }
            for step in per_step
        ],
    }


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
        "--emit-logprobs",
        action="store_true",
        help=(
            "Opt-in: record greedy-decode confidence signals (per-step logprob, "
            "margin, entropy, and first-step UNKNOWN mass) next to each "
            "prediction. Decoding is unchanged."
        ),
    )
    parser.add_argument(
        "--logprob-top-k",
        type=int,
        default=20,
        help="First-step candidates recorded when --emit-logprobs is set.",
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
    if args.logprob_top_k < 1:
        parser.error("--logprob-top-k must be positive")

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

    unknown_tokens = (
        unknown_continuation_tokens(tokenizer) if args.emit_logprobs else []
    )

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

                    generate_kwargs: dict[str, Any] = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "use_cache": True,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    if args.emit_logprobs:
                        generate_kwargs["return_dict_in_generate"] = True
                        generate_kwargs["output_logits"] = True

                    synchronize_if_needed(torch)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**encoded, **generate_kwargs)
                    synchronize_if_needed(torch)
                    latency = time.perf_counter() - started

                    step_logits = None
                    if args.emit_logprobs:
                        step_logits = generated.logits
                        generated = generated.sequences

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
                    if args.emit_logprobs:
                        with torch.inference_mode():
                            result["confidence"] = summarize_confidence(
                                torch=torch,
                                tokenizer=tokenizer,
                                step_logits=step_logits,
                                continuation=continuation,
                                top_k=args.logprob_top_k,
                                unknown_tokens=unknown_tokens,
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
