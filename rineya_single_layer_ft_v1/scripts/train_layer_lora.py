#!/usr/bin/env python3
"""Train LoRA/QLoRA on a single Supa decoder layer. Vault never overwritten."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    abstention_overrides,
    assert_gate_disabled,
    atomic_write_json,
    load_yaml,
    read_jsonl,
    resolve_core_path,
    resolve_repo_path,
)

ATTENTION_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGET_SUFFIXES = ("gate_proj", "up_proj", "down_proj")
TARGET_SUFFIXES = ATTENTION_TARGET_SUFFIXES + MLP_TARGET_SUFFIXES
OVERFLOW_POLICIES = ("drop", "left-context", "error")


class RowOverflowError(RuntimeError):
    pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_selected_layers(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        layers = sorted({int(x) for x in value})
    else:
        layers = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if not layers:
        raise ValueError("selected_layers must be a non-empty list of integers")
    return layers


def discover_target_modules(
    model: torch.nn.Module,
    selected_layers: list[int],
    target_module_group: str = "both",
) -> list[str]:
    if target_module_group == "attention":
        allowed = ATTENTION_TARGET_SUFFIXES
    elif target_module_group == "mlp":
        allowed = MLP_TARGET_SUFFIXES
    elif target_module_group == "both":
        allowed = TARGET_SUFFIXES
    else:
        raise ValueError(f"unsupported target_module_group: {target_module_group}")

    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(allowed):
            continue
        wrapped = f".{name}."
        if not any(f".layers.{layer_id}." in wrapped for layer_id in selected_layers):
            continue
        targets.append(name)

    matched: set[int] = set()
    for name in targets:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is not None:
            matched.add(int(match.group(1)))
    missing = sorted(set(selected_layers) - matched)
    if missing:
        raise ValueError(f"no LoRA targets for layers {missing}; matched={sorted(matched)}")
    if not targets:
        raise RuntimeError("no LoRA target modules found")
    return targets


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def validate_adapter_directory(path: Path) -> None:
    config = path / "adapter_config.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not config.exists():
        raise RuntimeError(f"Adapter save incomplete; missing {config}")
    if not any(c.exists() and c.stat().st_size > 0 for c in weights):
        raise RuntimeError(f"Adapter save incomplete under {path}")


def publish_adapter_directory(staging_dir: Path, adapter_dir: Path) -> None:
    backup_dir = adapter_dir.with_name(
        f".{adapter_dir.name}.backup-{os.getpid()}-{uuid.uuid4().hex}"
    )
    moved_existing = False
    try:
        if adapter_dir.exists():
            adapter_dir.rename(backup_dir)
            moved_existing = True
        staging_dir.rename(adapter_dir)
    except BaseException:
        if moved_existing and backup_dir.exists() and not adapter_dir.exists():
            backup_dir.rename(adapter_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


class LazyChatSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy tokenization with overflow drop (Halo Stage-A pattern)."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        *,
        overflow_policy: str = "drop",
        label: str = "dataset",
    ) -> None:
        if overflow_policy not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow_policy must be one of {OVERFLOW_POLICIES}")
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overflow_policy = overflow_policy
        self.label = label
        self.indices = list(range(len(records)))
        self.dropped_indices: list[int] = []
        self.truncated_rows = 0
        if overflow_policy == "drop":
            self._select_fitting_rows()

    def _render(self, row: dict[str, Any]) -> tuple[list[int], list[int]]:
        messages = row["messages"]
        prompt_text = apply_chat_template(
            self.tokenizer, messages[:-1], add_generation_prompt=True
        )
        full_text = apply_chat_template(
            self.tokenizer, messages, add_generation_prompt=False
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            raise RowOverflowError("empty assistant turn")
        return prompt_ids, full_ids

    def prompt_token_count(self, index: int) -> int:
        row = self.records[index]
        prompt_text = apply_chat_template(
            self.tokenizer, row["messages"][:-1], add_generation_prompt=True
        )
        return len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    def _select_fitting_rows(self) -> None:
        keep: list[int] = []
        for position, index in enumerate(self.indices, 1):
            if self.prompt_token_count(index) < self.max_length:
                keep.append(index)
            else:
                self.dropped_indices.append(index)
            if position % 500 == 0:
                print(
                    f"[{self.label}] overflow scan {position}/{len(self.records)} "
                    f"kept={len(keep)} dropped={len(self.dropped_indices)}",
                    flush=True,
                )
        self.indices = keep

    def audit(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rows_in_file": len(self.records),
            "rows_used": len(self.indices),
            "rows_dropped_for_overflow": len(self.dropped_indices),
            "overflow_policy": self.overflow_policy,
            "max_sequence_length": self.max_length,
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[self.indices[index]]
        prompt_ids, full_ids = self._render(row)
        answer_ids = full_ids[len(prompt_ids) :]
        overflow = len(full_ids) - self.max_length
        if overflow > 0:
            if self.overflow_policy == "error":
                raise RowOverflowError(
                    f"[{self.label}] needs {len(full_ids)} > {self.max_length}"
                )
            kept_prompt = prompt_ids[overflow:]
            if not kept_prompt:
                raise RowOverflowError(f"[{self.label}] answer alone does not fit")
            self.truncated_rows += 1
            prompt_ids = kept_prompt
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class CausalChatCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [x["input_ids"] for x in features], batch_first=True, padding_value=self.pad_token_id
            ),
            "attention_mask": pad_sequence(
                [x["attention_mask"] for x in features], batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(
                [x["labels"] for x in features], batch_first=True, padding_value=-100
            ),
        }


def build_model(cfg: dict[str, Any]) -> tuple[Any, Any, list[str]]:
    model_cfg = cfg["model"]
    adapter_cfg = cfg["adapter"]
    train_cfg = cfg["train"]
    model_path = str(resolve_repo_path(model_cfg["path"]))
    use_qlora = bool(adapter_cfg.get("qlora", False))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for layer-8 training")

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": "auto",
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    if use_qlora:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        dtype = str(model_cfg.get("dtype", "float16")).lower()
        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
        model_kwargs["torch_dtype"] = torch_dtype

    model_kwargs.update(abstention_overrides())
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        )

    selected = parse_selected_layers(adapter_cfg["selected_layers"])
    targets = discover_target_modules(
        model, selected, adapter_cfg.get("target_module_group", "both")
    )
    lora_config = LoraConfig(
        r=int(adapter_cfg["r"]),
        lora_alpha=int(adapter_cfg["alpha"]),
        lora_dropout=float(adapter_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # The training loss must see the same ungated logits the evaluators do.
    assert_gate_disabled(model)
    return model, tokenizer, targets


def resolve_default_files(cfg: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    views = output_dir / "views"
    name = str(cfg.get("experiment", {}).get("name", ""))
    if "cc0" in name:
        return views / "train_8k.jsonl", views / "val_all_strata.jsonl"
    return views / "train_easy_think_then_short.jsonl", views / "val_easy_think_then_short.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_lora_puzzle_smoke.yaml",
    )
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    adapter_cfg = cfg["adapter"]

    output_dir = args.output_dir or resolve_repo_path(train_cfg["output_dir"])
    default_train, default_dev = resolve_default_files(cfg, output_dir)
    train_file = args.train_file or default_train
    dev_file = args.dev_file or default_dev

    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} missing; build views first")

    seed = int(train_cfg["seed"])
    seed_everything(seed)

    train_rows = read_jsonl(train_file)
    for i, row in enumerate(train_rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{train_file}:{i} has no messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{train_file}:{i} must end with assistant")

    # Dev for Trainer loss only: prefer 8K-only subsample when present to save time.
    if Path(dev_file).exists():
        all_dev = read_jsonl(dev_file)
        eight_k = [
            r
            for r in all_dev
            if str((r.get("metadata") or {}).get("context_stratum", "")).lower() == "8k"
        ]
        # Cap eval set size for 8K training speed.
        rng = random.Random(seed)
        pool = eight_k if eight_k else all_dev
        if len(pool) > 128:
            pool = rng.sample(pool, 128)
            pool.sort(key=lambda r: str(r.get("id")))
        dev_rows = pool
    else:
        dev_rows = []

    model, tokenizer, targets = build_model(cfg)
    max_len = int(data_cfg["max_sequence_length"])
    overflow = str(data_cfg.get("overflow_policy", "drop"))
    train_data = LazyChatSFTDataset(
        train_rows, tokenizer, max_len, overflow_policy=overflow, label="train"
    )
    if not len(train_data):
        raise RuntimeError("all train rows dropped for overflow")
    eval_data = (
        LazyChatSFTDataset(
            dev_rows, tokenizer, max_len, overflow_policy=overflow, label="dev"
        )
        if dev_rows
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "overflow_audit.json", {
        "train": train_data.audit(),
        "dev": eval_data.audit() if eval_data is not None else None,
    })
    max_steps = int(train_cfg["max_steps"]) if args.max_steps is None else int(args.max_steps)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["epochs"]),
        max_steps=max_steps,
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=0.0,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps" if eval_data is not None else "no",
        save_strategy="steps",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=CausalChatCollator(tokenizer.pad_token_id),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()

    adapter_dir = output_dir / "adapter"
    staging_dir = output_dir / f".adapter.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        trainer.save_model(str(staging_dir))
        tokenizer.save_pretrained(str(staging_dir))
        validate_adapter_directory(staging_dir)
        publish_adapter_directory(staging_dir, adapter_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    manifest = {
        "experiment": cfg.get("experiment", {}),
        "model_path": str(resolve_core_path(cfg["model"]["path"])),
        "selected_layers": adapter_cfg["selected_layers"],
        "target_modules": targets,
        "trainable_parameter_count": int(trainable),
        "total_parameter_count_loaded": int(total),
        "qlora": bool(adapter_cfg.get("qlora", False)),
        "lora": {
            "r": adapter_cfg["r"],
            "alpha": adapter_cfg["alpha"],
            "dropout": adapter_cfg["dropout"],
        },
        "train_file": str(train_file),
        "dev_file": str(dev_file),
        "train_examples": len(train_data),
        "dev_examples": len(eval_data) if eval_data is not None else 0,
        "overflow_audit": train_data.audit(),
        "train_metrics": result.metrics,
        "peak_allocated_vram_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "adapter_dir": str(adapter_dir),
        "max_sequence_length": max_len,
    }
    atomic_write_json(output_dir / "training_manifest.json", manifest)
    print(json.dumps({"adapter_dir": str(adapter_dir), "trainable": trainable, "train_used": len(train_data)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
