from __future__ import annotations

import argparse
import json
import random
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

# Default base checkpoint for the first dense-inference SFT reference.
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
# LoRA may be attached to attention projections (q/k/v/o) and gated-MLP
# projections (gate/up/down), matching the project's initial experiment.
TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# Expose all important training and adapter settings so experiments can
# be reproduced without editing source code.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the all-layer LoRA/QLoRA SFT reference or a controlled "
            "selected-layer adapter for later layer scans."
        )
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--train-file", type=Path, default=Path("data/train/train.jsonl"))
    p.add_argument("--dev-file", type=Path, default=Path("data/dev/dev.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/dense_sft"))
    p.add_argument(
        "--selected-layers",
        default="all",
        help='Use "all" or a comma-separated list such as "12" or "10,11".',
    )
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p.parse_args()


# Seed Python, NumPy, CPU PyTorch, and every CUDA device to reduce
# avoidable run-to-run variation.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Load versioned SFT records and reject files that cannot provide a valid
# assistant training target.
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Create versioned SFT records before training."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no} has no messages list.")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_no} must end with an assistant target.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no usable training records.")
    return rows


# `None` represents all transformer layers. A list limits LoRA placement
# to explicit layer IDs for the later controlled layer scan.
def parse_selected_layers(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not layers:
        raise ValueError("--selected-layers must be 'all' or contain integers.")
    return layers


# Inspect the loaded model rather than assuming exact module paths. Return
# full module names so PEFT can attach adapters only where requested.
def discover_target_modules(
    model: torch.nn.Module,
    selected_layers: list[int] | None,
) -> list[str]:
    targets: list[str] = []
    for name, module in model.named_modules():
        # LoRA in this experiment targets linear projection modules only.
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(TARGET_SUFFIXES):
            continue
        # During selective experiments, discard matching projections that
        # do not belong to one of the requested transformer layers.
        if selected_layers is not None:
            wrapped = f".{name}."
            if not any(f".layers.{layer_id}." in wrapped for layer_id in selected_layers):
                continue
        targets.append(name)
    if not targets:
        raise RuntimeError(
            "No LoRA target modules were found. Inspect model.named_modules() and "
            "update TARGET_SUFFIXES or the layer-name pattern."
        )
    return targets


# Convert chat records into causal-language-model tensors while training
# only on the final assistant answer.
class ChatSFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        for row in records:
            messages = row["messages"]
            # Separate the context/prompt from the final assistant target.
            prompt_messages = messages[:-1]
            # Render the prompt with a generation marker, then render the
            # complete conversation containing the target answer.
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(full_ids) <= len(prompt_ids):
                continue
            # Transformers ignores label value -100. Masking prompt tokens
            # makes the loss teach only the desired assistant continuation.
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.examples:
            raise ValueError("No usable examples remained after tokenization/truncation.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


# Batch examples have different lengths, so this collator pads inputs,
# masks, and labels consistently.
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


# Load tokenizer/base model, apply optional 4-bit QLoRA preparation,
# discover targets, and attach trainable LoRA adapters.
def build_model(args: argparse.Namespace) -> tuple[Any, Any, list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Decoder-only tokenizers often lack a dedicated padding token; EOS is
    # a standard safe fallback for padded batch positions.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
    # QLoRA keeps the frozen base in 4-bit NF4 form while training the small
    # adapter, reducing weight memory on a consumer GPU.
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
    # KV caching helps generation but conflicts with training and gradient
    # checkpointing, so it is disabled for this training process.
    model.config.use_cache = False
    # Recompute selected activations during backward passes to trade speed
    # for lower training memory.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    # Resolve either the all-layer reference or a controlled subset before
    # creating the PEFT configuration.
    selected_layers = parse_selected_layers(args.selected_layers)
    targets = discover_target_modules(model, selected_layers)
    # These parameters define adapter capacity, scaling, dropout, and the
    # exact projection modules that receive trainable low-rank updates.
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


# Orchestrate data loading, adapter construction, Trainer execution, and
# reproducibility-artifact output.
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix PyTorch/GPU setup first.")

    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file) if args.dev_file.exists() else []
    model, tokenizer, targets = build_model(args)
    # Tokenize training and optional development records with identical rules.
    train_data = ChatSFTDataset(train_rows, tokenizer, args.max_sequence_length)
    eval_data = ChatSFTDataset(dev_rows, tokenizer, args.max_sequence_length) if dev_rows else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face Trainer manages optimization, accumulation, checkpointing,
    # periodic development loss, precision, and reproducible seeds.
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
        report_to=[],
        remove_unused_columns=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=CausalChatCollator(tokenizer.pad_token_id),
    )

    # Record the training run's peak allocated VRAM for later comparisons
    # between all-layer and selected-layer adapters.
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    # Save only the PEFT adapter and tokenizer, not another full copy of the
    # frozen base checkpoint.
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Persist the model, data paths, selected layers, targets, seed, metrics,
    # and peak memory needed to reproduce and compare this run.
    manifest = {
        "model_id": args.model_id,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "selected_layers": args.selected_layers,
        "qlora": args.qlora,
        "target_module_count": len(targets),
        "target_modules": targets,
        "seed": args.seed,
        "train_examples": len(train_data),
        "dev_examples": len(eval_data) if eval_data is not None else 0,
        "train_metrics": result.metrics,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Manifest saved to: {args.output_dir / 'training_manifest.json'}")


# Standard script entry point.
if __name__ == "__main__":
    main()
