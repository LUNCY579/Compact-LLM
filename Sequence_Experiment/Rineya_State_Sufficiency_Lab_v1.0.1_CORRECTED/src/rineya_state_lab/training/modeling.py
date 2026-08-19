from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any


def _dtype_from_name(name: str, torch: Any) -> Any:
    value = name.casefold()
    if value == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}")
    return mapping[value]


def _supported_kwargs(callable_object: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_object)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_tokenizer(model_name_or_path: str, config: dict[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Training dependencies are missing. Install with: pip install -e ".[train]"') from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        local_files_only=bool(config.get("local_files_only", False)),
        use_fast=bool(config.get("use_fast_tokenizer", True)),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad, EOS, or unknown token; refusing to add an untrained token")
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_with_adapter(model_name_or_path: str, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError('Training dependencies are missing. Install with: pip install -e ".[train]"') from exc

    dtype = _dtype_from_name(str(config.get("dtype", "auto")), torch)
    quant = config.get("quantization", {}) or {}
    load_in_4bit = bool(quant.get("load_in_4bit", False))
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "local_files_only": bool(config.get("local_files_only", False)),
        "low_cpu_mem_usage": bool(config.get("low_cpu_mem_usage", True)),
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    if load_in_4bit:
        compute_dtype = _dtype_from_name(str(quant.get("compute_dtype", config.get("dtype", "auto"))), torch)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank >= 0:
            model_kwargs["device_map"] = {"": local_rank}
        elif config.get("device_map") is not None:
            model_kwargs["device_map"] = config.get("device_map")
    else:
        model_kwargs["torch_dtype"] = dtype
        if config.get("device_map") is not None:
            model_kwargs["device_map"] = config.get("device_map")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": bool(config.get("gradient_checkpointing_reentrant", False))}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora = config.get("lora", {}) or {}
    target_modules = lora.get("target_modules", "all-linear")
    lora_kwargs: dict[str, Any] = {
        "r": int(lora.get("r", 16)),
        "lora_alpha": int(lora.get("alpha", 32)),
        "lora_dropout": float(lora.get("dropout", 0.05)),
        "bias": str(lora.get("bias", "none")),
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "modules_to_save": lora.get("modules_to_save"),
        "use_rslora": bool(lora.get("use_rslora", False)),
        "use_dora": bool(lora.get("use_dora", False)),
        "init_lora_weights": lora.get("init_lora_weights", True),
    }
    lora_config = LoraConfig(**_supported_kwargs(LoraConfig, lora_kwargs))
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "base_model": model_name_or_path,
        "dtype": str(dtype),
        "load_in_4bit": load_in_4bit,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total if total else 0.0,
        "lora": lora_kwargs,
    }
    return model, metadata
