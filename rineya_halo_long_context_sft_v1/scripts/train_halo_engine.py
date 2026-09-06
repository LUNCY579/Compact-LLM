#!/usr/bin/env python3
"""Halo QLoRA SFT engine with lazy tokenization (safe for long-context mixes)."""
from __future__ import annotations

import argparse
import json
import os
import random
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

from common import atomic_write_json, validate_adapter_directory  # noqa: E402

ATTENTION_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGET_SUFFIXES = ("gate_proj", "up_proj", "down_proj")
TARGET_SUFFIXES = ATTENTION_TARGET_SUFFIXES + MLP_TARGET_SUFFIXES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lazy-tokenizing Halo QLoRA SFT trainer.")
    p.add_argument("--model-id", required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--dev-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--selected-layers", default="all")
    p.add_argument("--target-module-group", choices=("both", "attention", "mlp"), default="both")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-sequence-length", type=int, default=4096)
    p.add_argument(
        "--overflow-policy",
        choices=OVERFLOW_POLICIES,
        default="drop",
        help=(
            "What to do with rows whose prompt leaves no room for the assistant "
            "turn. drop: exclude them (default). left-context: keep the answer "
            "and trim the front of the context. error: fail fast."
        ),
    )
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--run-label", default="")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--resume-from-checkpoint", default="")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no} has no messages list.")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_no} must end with an assistant target.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no usable training records.")
    return rows


def parse_selected_layers(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not layers:
        raise ValueError("--selected-layers must be 'all' or contain integers.")
    return layers


def discover_target_modules(
    model: torch.nn.Module,
    selected_layers: list[int] | None,
    target_module_group: str = "both",
) -> list[str]:
    if target_module_group == "attention":
        allowed_suffixes = ATTENTION_TARGET_SUFFIXES
    elif target_module_group == "mlp":
        allowed_suffixes = MLP_TARGET_SUFFIXES
    else:
        allowed_suffixes = TARGET_SUFFIXES
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(allowed_suffixes):
            continue
        if selected_layers is not None:
            wrapped = f".{name}."
            if not any(f".layers.{layer_id}." in wrapped for layer_id in selected_layers):
                continue
        targets.append(name)
    if not targets:
        raise RuntimeError("No LoRA target modules were found.")
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


def publish_adapter_directory(staging_dir: Path, adapter_dir: Path) -> None:
    backup_dir = adapter_dir.with_name(f".{adapter_dir.name}.backup-{os.getpid()}-{uuid.uuid4().hex}")
    moved_existing = False
    try:
        if adapter_dir.exists():
            os.replace(adapter_dir, backup_dir)
            moved_existing = True
        os.replace(staging_dir, adapter_dir)
    except BaseException:
        if moved_existing and backup_dir.exists() and not adapter_dir.exists():
            os.replace(backup_dir, adapter_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


OVERFLOW_POLICIES = ("drop", "left-context", "error")


class RowOverflowError(RuntimeError):
    """A row cannot be encoded without destroying its supervised answer."""


class LazyChatSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenize on demand so long-context mixes do not stall in __init__.

    Right-truncating a chat row at ``max_length`` removes the assistant turn,
    because the answer sits at the end. A row encoded that way carries zero
    supervised tokens: it contributes no gradient, and an evaluation set made
    entirely of such rows divides by zero and reports ``eval_loss = NaN``.

    This class refuses to emit a fully-masked row. ``overflow_policy`` decides
    what happens instead:

    ``drop``          exclude the row from the dataset (default). Honest for
                      needle-in-haystack data, where trimming the context can
                      remove the very fact the answer depends on and would
                      train the model to invent one.
    ``left-context``  keep the answer and trim tokens off the front of the
                      prompt. Uses every row, but can silently delete the
                      needle - only appropriate when the context is redundant.
    ``error``         raise on the first overflowing row.
    """

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
            raise ValueError(
                f"overflow_policy must be one of {OVERFLOW_POLICIES}, got {overflow_policy!r}"
            )
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
        full_text = apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "The chat template does not render the prompt as a prefix of the full "
                "conversation, so prompt-length label masking would mislabel tokens."
            )
        if len(full_ids) <= len(prompt_ids):
            raise RowOverflowError("Row has an empty assistant turn")
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
            if position % 2000 == 0:
                print(
                    f"[{self.label}] overflow scan {position}/{len(self.indices)} "
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
                    f"[{self.label}] row needs {len(full_ids)} tokens but "
                    f"max_sequence_length is {self.max_length}"
                )
            kept_prompt = prompt_ids[overflow:]
            if not kept_prompt:
                raise RowOverflowError(
                    f"[{self.label}] the assistant turn alone ({len(answer_ids)} tokens) "
                    f"does not fit in max_sequence_length={self.max_length}"
                )
            self.truncated_rows += 1
            prompt_ids = kept_prompt

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if all(label == -100 for label in labels):
            raise RowOverflowError(
                f"[{self.label}] encoded row carries no supervised tokens"
            )
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


def build_model(args: argparse.Namespace) -> tuple[Any, Any, list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float32 if args.allow_cpu else "auto",
        "device_map": "cpu" if args.allow_cpu else "auto",
    }
    if args.qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires CUDA.")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    selected_layers = parse_selected_layers(args.selected_layers)
    targets = discover_target_modules(model, selected_layers, args.target_module_group)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, targets


def main() -> None:
    args = parse_args()
    if args.allow_cpu and args.qlora:
        raise ValueError("QLoRA requires CUDA; omit --qlora when using --allow-cpu.")
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train rows from {args.train_file}", flush=True)
    train_rows = read_jsonl(args.train_file)
    print(f"Loaded train_rows={len(train_rows)}", flush=True)
    dev_rows = read_jsonl(args.dev_file) if args.dev_file.exists() else []
    print(f"Loaded dev_rows={len(dev_rows)}", flush=True)

    model, tokenizer, targets = build_model(args)
    first_param = next(model.parameters())
    print(
        f"Trainer device={first_param.device} allow_cpu={args.allow_cpu} "
        f"cuda_available={torch.cuda.is_available()} qlora={args.qlora} lazy_tokenize=True",
        flush=True,
    )

    train_data = LazyChatSFTDataset(
        train_rows,
        tokenizer,
        args.max_sequence_length,
        overflow_policy=args.overflow_policy,
        label="train",
    )
    eval_data = (
        LazyChatSFTDataset(
            dev_rows,
            tokenizer,
            args.max_sequence_length,
            overflow_policy=args.overflow_policy,
            label="dev",
        )
        if dev_rows
        else None
    )

    dataset_audit = [train_data.audit()] + ([eval_data.audit()] if eval_data else [])
    for entry in dataset_audit:
        share = (
            entry["rows_dropped_for_overflow"] / entry["rows_in_file"]
            if entry["rows_in_file"]
            else 0.0
        )
        print(
            f"[{entry['label']}] rows_in_file={entry['rows_in_file']} "
            f"used={entry['rows_used']} dropped_for_overflow="
            f"{entry['rows_dropped_for_overflow']} ({share:.1%}) "
            f"policy={entry['overflow_policy']} max_len={entry['max_sequence_length']}",
            flush=True,
        )
        if share >= 0.10:
            print(
                f"[{entry['label']}] WARNING: {share:.1%} of rows exceed "
                f"--max-sequence-length {entry['max_sequence_length']}. On a "
                "long-context mix this discards exactly the longest examples. "
                "Raise the limit or accept a short-context-biased run.",
                flush=True,
            )
    if len(train_data) == 0:
        raise SystemExit(
            "No training rows survive the overflow policy; raise --max-sequence-length"
        )
    if eval_data is not None and len(eval_data) == 0:
        raise SystemExit(
            "Every dev row overflows --max-sequence-length, so eval_loss would be "
            "NaN (0 supervised tokens). Raise --max-sequence-length, choose "
            "--overflow-policy left-context, or supply a shorter dev file."
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_data is not None else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        fp16=not args.allow_cpu and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=not args.allow_cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        use_cpu=bool(args.allow_cpu),
        dataloader_pin_memory=not args.allow_cpu,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
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
    resume = str(args.resume_from_checkpoint or "").strip()
    if resume.lower() == "latest":
        resume_from: str | bool | None = True
    elif resume:
        resume_from = resume
    else:
        resume_from = None
    print(f"Starting train resume_from_checkpoint={resume_from!r}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_from)

    adapter_dir = args.output_dir / "adapter"
    staging_dir = args.output_dir / f".adapter.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        trainer.save_model(str(staging_dir))
        tokenizer.save_pretrained(str(staging_dir))
        validate_adapter_directory(staging_dir)
        publish_adapter_directory(staging_dir, adapter_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameter_count = sum(p.numel() for p in model.parameters())
    adapter_bytes = sum(child.stat().st_size for child in adapter_dir.rglob("*") if child.is_file())
    manifest = {
        "model_id": args.model_id,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "selected_layers": args.selected_layers,
        "qlora": args.qlora,
        "lazy_tokenize": True,
        "target_module_group": args.target_module_group,
        "target_module_count": len(targets),
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count_loaded": int(total_parameter_count),
        "adapter_serialized_bytes": int(adapter_bytes),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_sequence_length": args.max_sequence_length,
        "overflow_policy": args.overflow_policy,
        "dataset_audit": dataset_audit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "save_total_limit": args.save_total_limit,
        "run_label": args.run_label,
        "seed": args.seed,
        "train_examples": len(train_data),
        "dev_examples": len(eval_data) if eval_data is not None else 0,
        "train_metrics": result.metrics,
        "peak_allocated_vram_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
    }
    atomic_write_json(args.output_dir / "training_manifest.json", manifest)
    print(f"Adapter saved to: {adapter_dir}", flush=True)
    print(f"Manifest saved to: {args.output_dir / 'training_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
