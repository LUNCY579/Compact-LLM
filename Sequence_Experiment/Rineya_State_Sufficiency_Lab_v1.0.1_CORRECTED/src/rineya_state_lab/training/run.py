from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

from ..config import load_config, require
from ..data.layout import DatasetLayout, discover_dataset_root
from ..utils import (
    atomic_write_json,
    atomic_write_text,
    environment_manifest,
    seed_everything,
    sha256_file,
    sha256_text,
    utc_now_iso,
)
from .dataset import AssistantOnlyCollator, JsonlChatDataset
from .modeling import load_model_with_adapter, load_tokenizer
from .trainer_ext import make_weighted_trainer_class


def _supported_training_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments)
    supported = set(signature.parameters)
    result = {key: value for key, value in kwargs.items() if key in supported}
    if "eval_strategy" not in supported and "evaluation_strategy" in supported and "eval_strategy" in kwargs:
        result["evaluation_strategy"] = kwargs["eval_strategy"]
    return result


def _trainer_tokenizer_kwarg(trainer_class: type, tokenizer: Any) -> dict[str, Any]:
    signature = inspect.signature(trainer_class.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {}


def _resolve_precision(training: dict[str, Any]) -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False
    requested = str(training.get("mixed_precision", "auto")).casefold()
    if requested in {"no", "none", "fp32"}:
        return False, False
    if requested in {"bf16", "bfloat16"}:
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError("BF16 was requested, but the current CUDA device does not report BF16 support")
        return True, False
    if requested in {"fp16", "float16"}:
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 training was requested, but CUDA is unavailable")
        return False, True
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    if torch.cuda.is_available():
        return False, True
    return False, False


def _model_card(config: dict[str, Any], metadata: dict[str, Any], exposure: dict[str, Any]) -> str:
    return f"""# {config['run']['name']} Adapter

This adapter was produced by Rineya State Sufficiency Lab for the compact-state versus full-context reasoning-supervision experiment.

## Training representation

- View: `{config['data']['view']}`
- Base model: `{metadata['base_model']}`
- Trainable parameters: `{metadata['trainable_parameters']:,}`
- Total model parameters while loaded: `{metadata['total_parameters']:,}`
- Selected rows: `{exposure['rows']:,}`
- Unique source problems: `{exposure['unique_problems']:,}`
- Target-token accounting: `{exposure['token_count_mode']}`

## Intended use

Use this adapter only with the same immutable base model and tokenizer revision recorded in `run_manifest.json`. Compare it against the matched adapter with the repository evaluation harness; do not interpret training loss alone as evidence for the state-sufficiency hypothesis.

## Limitations

The bundled dataset is procedurally generated and deterministic. Results on it do not establish general reasoning ability without additional out-of-distribution and manually authored evaluations.
"""


def run_training(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError('Training dependencies are missing. Install with: pip install -e ".[train]"') from exc

    config = load_config(config_path, overrides)
    run_cfg = require(config, "run")
    data_cfg = require(config, "data")
    model_cfg = require(config, "model")
    train_cfg = require(config, "training")
    seed = int(run_cfg.get("seed", 42))
    seed_everything(seed)

    output_dir = Path(str(run_cfg["output_dir"])).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(model_cfg["name_or_path"])
    tokenizer = load_tokenizer(model_name, model_cfg)
    dataset_root = discover_dataset_root(data_cfg["root"])
    layout = DatasetLayout(dataset_root)
    view = str(data_cfg["view"])
    source_kind = "full" if view == "full_context" else "transition"
    train_split = str(data_cfg.get("train_split", "train"))
    eval_split = str(data_cfg.get("eval_split", "dev"))
    train_file = layout.full_file(train_split) if source_kind == "full" else layout.transition_file(train_split)
    eval_file = layout.full_file(eval_split) if source_kind == "full" else layout.transition_file(eval_split)

    common_dataset_kwargs = {
        "tokenizer": tokenizer,
        "view": view,
        "max_length": int(data_cfg.get("max_length", 1024)),
        "max_target_tokens": data_cfg.get("max_target_tokens"),
        "seed": seed,
        "train_on_prompt": bool(data_cfg.get("train_on_prompt", False)),
        "cache_size": int(data_cfg.get("token_cache_size", 2048)),
    }
    train_dataset = JsonlChatDataset(
        train_file,
        max_examples=data_cfg.get("max_train_examples"),
        one_per_problem=bool(data_cfg.get("one_transition_per_problem", False)),
        preserve_trajectory_order=bool(data_cfg.get("preserve_trajectory_order", False)),
        inverse_problem_weight=bool(data_cfg.get("inverse_problem_weight", source_kind == "transition")),
        **common_dataset_kwargs,
    )
    eval_dataset = JsonlChatDataset(
        eval_file,
        max_examples=data_cfg.get("max_eval_examples"),
        one_per_problem=bool(data_cfg.get("eval_one_transition_per_problem", False)),
        preserve_trajectory_order=True,
        inverse_problem_weight=bool(data_cfg.get("inverse_problem_weight", source_kind == "transition")),
        **common_dataset_kwargs,
    )
    if not len(train_dataset):
        raise ValueError("Training selection contains zero examples")

    exposure = train_dataset.exposure_stats(
        exact=bool(data_cfg.get("exact_token_accounting", False)),
        sample_size=int(data_cfg.get("token_accounting_sample", 2048)),
    )
    atomic_write_json(output_dir / "data_exposure.json", exposure)

    model, model_metadata = load_model_with_adapter(model_name, model_cfg)
    bf16, fp16 = _resolve_precision(train_cfg)
    args_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1.0)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "logging_strategy": "steps",
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "eval_strategy": str(train_cfg.get("eval_strategy", "steps")),
        "eval_steps": int(train_cfg.get("eval_steps", 200)),
        "save_strategy": str(train_cfg.get("save_strategy", "steps")),
        "save_steps": int(train_cfg.get("save_steps", 200)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", False)),
        "metric_for_best_model": str(train_cfg.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": bool(train_cfg.get("greater_is_better", False)),
        "bf16": bf16,
        "fp16": fp16,
        "tf32": bool(train_cfg.get("tf32", True)),
        "gradient_checkpointing": bool(model_cfg.get("gradient_checkpointing", True)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "optim": str(train_cfg.get("optim", "adamw_torch_fused")),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(train_cfg.get("dataloader_pin_memory", True)),
        "dataloader_drop_last": bool(train_cfg.get("dataloader_drop_last", False)),
        "remove_unused_columns": False,
        "report_to": train_cfg.get("report_to", []),
        "seed": seed,
        "data_seed": int(run_cfg.get("data_seed", seed)),
        "save_safetensors": True,
        "torch_compile": bool(train_cfg.get("torch_compile", False)),
        "ddp_find_unused_parameters": bool(train_cfg.get("ddp_find_unused_parameters", False)),
    }
    training_args = TrainingArguments(**_supported_training_args(args_kwargs))
    trainer_class = make_weighted_trainer_class(
        preserve_order=bool(data_cfg.get("preserve_trajectory_order", False))
    )
    collator = AssistantOnlyCollator(tokenizer, pad_to_multiple_of=data_cfg.get("pad_to_multiple_of", 8))
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        **_trainer_tokenizer_kwarg(trainer_class, tokenizer),
    )

    resolved_yaml = __import__("yaml").safe_dump(config, sort_keys=False, allow_unicode=True)
    atomic_write_text(output_dir / "resolved_config.yaml", resolved_yaml)
    manifest = {
        "schema_version": "1.0",
        "run_name": run_cfg["name"],
        "started_at": utc_now_iso(),
        "status": "RUNNING",
        "config_sha256": sha256_text(resolved_yaml),
        "train_file": str(train_file),
        "train_file_sha256": sha256_file(train_file),
        "eval_file": str(eval_file),
        "eval_file_sha256": sha256_file(eval_file),
        "dataset_root": str(dataset_root),
        "model": model_metadata,
        "exposure": exposure,
        "environment": environment_manifest(),
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)

    resume = train_cfg.get("resume_from_checkpoint", "auto")
    resume_value: str | bool | None
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        resume_value = get_last_checkpoint(str(checkpoints_dir))
    elif resume in {False, None, "none", "false"}:
        resume_value = None
    elif resume is True:
        resume_value = True
    else:
        resume_value = str(Path(str(resume)).expanduser().resolve())

    try:
        result = trainer.train(resume_from_checkpoint=resume_value)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        metrics = dict(result.metrics)
        metrics["train_samples"] = len(train_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        atomic_write_json(output_dir / "trainer_metrics.json", metrics)
        atomic_write_text(output_dir / "MODEL_CARD.md", _model_card(config, model_metadata, exposure))
        manifest.update({"status": "COMPLETED", "completed_at": utc_now_iso(), "metrics": metrics})
        atomic_write_json(output_dir / "run_manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "FAILED",
                "failed_at": utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(output_dir / "run_manifest.json", manifest)
        raise
