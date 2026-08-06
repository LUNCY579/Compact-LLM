from __future__ import annotations

"""Train a Phase 9 student with RAG-aware SFT or same-tokenizer online KD."""

import argparse
import json
import math
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
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import environment, read_jsonl, replace_path_with_retry, sha256, write_json
from scripts.phase_1_dense_sft.train_dense_sft import discover_target_modules
from scripts.phase_9_student_distillation.phase9_common import (
    model_parameter_breakdown,
    tokenizer_fingerprint,
    tree_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny RAG student with SFT or online logit KD.")
    parser.add_argument("--student", required=True, help="Student initialization checkpoint.")
    parser.add_argument("--teacher", default=None, help="Required for objective=kd.")
    parser.add_argument("--objective", choices=("sft", "kd"), default="sft")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-completion-tokens", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-label-weight", type=float, default=0.25)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--initial-adapter", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-module-group", choices=("both", "attention", "mlp"), default="both")
    parser.add_argument("--teacher-4bit", action="store_true")
    parser.add_argument("--memory-efficient-completion-logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--allow-random-control", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class CompletionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
        max_completion_tokens: int,
        default_hard_weight: float,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        self.skipped: list[str] = []
        for row in rows:
            messages = row.get("messages") or []
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(f"{row.get('id')} must end with an assistant training target")
            prompt_text = apply_chat_template(tokenizer, messages[:-1], generation=True)
            full_text = apply_chat_template(tokenizer, messages, generation=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                # Chat templates should be prefix-stable. Failing loudly avoids
                # training on incorrectly masked system/user tokens.
                raise ValueError(f"Tokenizer chat template is not prefix-stable for {row.get('id')}")
            completion_ids = full_ids[len(prompt_ids) :]
            if max_completion_tokens > 0:
                completion_ids = completion_ids[:max_completion_tokens]
            if not completion_ids:
                self.skipped.append(str(row.get("id")))
                continue
            prompt_budget = max_length - len(completion_ids)
            if prompt_budget <= 0:
                self.skipped.append(str(row.get("id")))
                continue
            prompt_ids = prompt_ids[-prompt_budget:]
            input_ids = torch.tensor([*prompt_ids, *completion_ids], dtype=torch.long)
            labels = torch.tensor([-100] * len(prompt_ids) + completion_ids, dtype=torch.long)
            override = row.get("hard_label_weight")
            if override is None:
                override = (row.get("metadata") or {}).get("hard_label_weight")
            hard_weight = default_hard_weight if override is None else float(override)
            if not 0.0 <= hard_weight <= 1.0:
                raise ValueError(f"{row.get('id')} hard_label_weight must be in [0,1]")
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": labels,
                    "hard_label_weight": torch.tensor(hard_weight, dtype=torch.float32),
                }
            )
        if not self.examples:
            raise ValueError("No usable examples remained after completion-preserving truncation")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass
class CompletionCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [row["input_ids"] for row in features], batch_first=True, padding_value=self.pad_token_id
            ),
            "attention_mask": pad_sequence(
                [row["attention_mask"] for row in features], batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(
                [row["labels"] for row in features], batch_first=True, padding_value=-100
            ),
            "hard_label_weight": torch.stack([row["hard_label_weight"] for row in features]),
        }


def selected_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_position_selection: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return completion prediction logits and their target token IDs.

    For batch size one, Qwen3-style ``logits_to_keep`` prevents materializing a
    [sequence, 151k-vocabulary] tensor for prompt positions. Other models fall
    back to the ordinary full-logit path without changing the objective.
    """

    if use_position_selection and input_ids.shape[0] == 1:
        target_positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
        target_positions = target_positions[target_positions > 0]
        prediction_positions = target_positions - 1
        targets = labels[0].index_select(0, target_positions)
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=prediction_positions,
            )
            logits = outputs.logits
            if logits.ndim == 3:
                logits = logits[0]
            if logits.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    "logits_to_keep returned an unexpected sequence dimension: "
                    f"{tuple(logits.shape)} for {targets.shape[0]} targets"
                )
            return logits, targets
        except TypeError:
            pass

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    return shifted_logits[mask], shifted_labels[mask]


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        teacher: Any | None,
        objective: str,
        temperature: float,
        memory_efficient_completion_logits: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.objective = objective
        self.temperature = float(temperature)
        self.memory_efficient_completion_logits = memory_efficient_completion_logits
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        del num_items_in_batch
        model_device = next(model.parameters()).device
        hard_weights = inputs.pop("hard_label_weight").to(model_device)
        labels = inputs.pop("labels").to(model_device)
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)
        student_logits, targets = selected_forward(
            model,
            input_ids,
            attention_mask,
            labels,
            use_position_selection=self.memory_efficient_completion_logits,
        )
        if targets.numel() == 0:
            raise ValueError("Batch contains no supervised completion tokens")
        hard_loss = F.cross_entropy(student_logits.float(), targets)

        if self.objective == "sft":
            loss = hard_loss
            output_packet = {"logits": student_logits, "hard_loss": hard_loss.detach()}
        else:
            if self.teacher is None:
                raise RuntimeError("KD objective requires a teacher")
            teacher_device = next(self.teacher.parameters()).device
            with torch.no_grad():
                teacher_logits, teacher_targets = selected_forward(
                    self.teacher,
                    input_ids.to(teacher_device),
                    attention_mask.to(teacher_device),
                    labels.to(teacher_device),
                    use_position_selection=self.memory_efficient_completion_logits,
                )
            if not torch.equal(targets.detach().cpu(), teacher_targets.detach().cpu()):
                raise RuntimeError("Teacher/student target positions are not aligned")
            temperature = self.temperature
            teacher_probs = F.softmax(teacher_logits.float().to(student_logits.device) / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
            hard_weight = float(hard_weights.mean().detach().cpu())
            loss = hard_weight * hard_loss + (1.0 - hard_weight) * kd_loss
            output_packet = {
                "logits": student_logits,
                "hard_loss": hard_loss.detach(),
                "kd_loss": kd_loss.detach(),
            }
        return (loss, output_packet) if return_outputs else loss


def validate_args(args: argparse.Namespace) -> None:
    if args.objective == "kd" and not args.teacher:
        raise ValueError("--teacher is required for objective=kd")
    if args.objective == "sft" and args.teacher:
        raise ValueError("Do not provide --teacher for objective=sft")
    if args.lora and args.initial_adapter is not None:
        raise ValueError("Use either --lora or --initial-adapter, not both")
    if not 0.0 <= args.hard_label_weight <= 1.0:
        raise ValueError("--hard-label-weight must be in [0,1]")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.per_device_batch_size != 1 and args.memory_efficient_completion_logits:
        raise ValueError(
            "Memory-efficient completion-position logits currently require per-device batch size 1. "
            "Use gradient accumulation for a larger effective batch."
        )
    if args.max_sequence_length <= 1 or args.max_completion_tokens <= 0:
        raise ValueError("Sequence/completion limits must be positive")


def ensure_random_control_acknowledged(student: str, allowed: bool) -> None:
    manifest = Path(student) / "student_creation_manifest.json"
    if not manifest.exists():
        return
    value = json.loads(manifest.read_text(encoding="utf-8"))
    state = str(value.get("initialization_state") or "")
    if state.startswith("random_control") and not allowed:
        raise RuntimeError(
            "This student is a randomly initialized architecture control. Pass --allow-random-control only for "
            "the explicit research track; do not mistake it for a production-ready warm start."
        )


def build_student(args: argparse.Namespace) -> tuple[Any, Any, str, list[str] | None]:
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    targets: list[str] | None = None
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.initial_adapter is not None:
        if not args.initial_adapter.exists():
            raise FileNotFoundError(args.initial_adapter)
        model = PeftModel.from_pretrained(model, str(args.initial_adapter), is_trainable=True)
        artifact_type = "continued_peft_adapter"
    elif args.lora:
        targets = discover_target_modules(model, None, args.target_module_group)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )
        artifact_type = "peft_adapter"
    else:
        artifact_type = "full_checkpoint"
    return model, tokenizer, artifact_type, targets


def build_teacher(args: argparse.Namespace, student_tokenizer: Any) -> tuple[Any, str | None]:
    if args.objective != "kd":
        return None, None
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher_tokenizer.pad_token = teacher_tokenizer.pad_token or teacher_tokenizer.eos_token
    student_fingerprint = tokenizer_fingerprint(student_tokenizer)
    teacher_fingerprint = tokenizer_fingerprint(teacher_tokenizer)
    if teacher_fingerprint != student_fingerprint:
        raise RuntimeError(
            "Direct vocabulary-level KL requires exactly identical tokenizers. Use sequence-level SFT for a "
            "cross-tokenizer challenger, or a separately validated span-alignment trainer."
        )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.teacher_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--teacher-4bit requires CUDA and bitsandbytes")
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **kwargs)
    teacher.config.use_cache = False
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, teacher_fingerprint


def publish_saved_model(trainer: Trainer, tokenizer: Any, output_dir: Path, artifact_type: str) -> Path:
    destination = output_dir / ("adapter" if "adapter" in artifact_type else "model")
    staging = output_dir / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    backup = output_dir / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        trainer.save_model(str(staging))
        tokenizer.save_pretrained(staging)
        if artifact_type == "full_checkpoint":
            if not (staging / "config.json").exists():
                raise RuntimeError("Full student save is missing config.json")
            if not any(staging.glob("*.safetensors")) and not any(staging.glob("pytorch_model*.bin")):
                raise RuntimeError("Full student save contains no weight file")
        else:
            if not (staging / "adapter_config.json").exists():
                raise RuntimeError("Adapter save is missing adapter_config.json")
        if destination.exists():
            replace_path_with_retry(destination, backup)
            moved = True
        replace_path_with_retry(staging, destination)
    except BaseException:
        if moved and backup.exists() and not destination.exists():
            replace_path_with_retry(backup, destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return destination


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_random_control_acknowledged(args.student, args.allow_random_control)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    seed_everything(args.seed)
    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file) if args.dev_file and args.dev_file.exists() else []
    model, tokenizer, artifact_type, targets = build_student(args)
    teacher, teacher_fingerprint = build_teacher(args, tokenizer)
    train_dataset = CompletionDataset(
        train_rows,
        tokenizer,
        max_length=args.max_sequence_length,
        max_completion_tokens=args.max_completion_tokens,
        default_hard_weight=args.hard_label_weight,
    )
    eval_dataset = (
        CompletionDataset(
            dev_rows,
            tokenizer,
            max_length=args.max_sequence_length,
            max_completion_tokens=args.max_completion_tokens,
            default_hard_weight=args.hard_label_weight,
        )
        if dev_rows
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
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
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = DistillationTrainer(
        model=model,
        teacher=teacher,
        objective=args.objective,
        temperature=args.temperature,
        memory_efficient_completion_logits=args.memory_efficient_completion_logits,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    artifact = publish_saved_model(trainer, tokenizer, args.output_dir, artifact_type)
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    breakdown = model_parameter_breakdown(model)
    manifest = {
        "phase": "9_student_distillation",
        "objective": args.objective,
        "student_initialization": args.student,
        "teacher": args.teacher,
        "artifact_type": artifact_type,
        "artifact_path": str(artifact),
        "artifact_bytes": tree_bytes(artifact),
        "student_tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "teacher_tokenizer_fingerprint": teacher_fingerprint,
        "direct_logit_alignment_verified": args.objective != "kd" or teacher_fingerprint == tokenizer_fingerprint(tokenizer),
        "train_file": str(args.train_file),
        "train_sha256": sha256(args.train_file),
        "dev_file": str(args.dev_file) if args.dev_file else None,
        "dev_sha256": sha256(args.dev_file) if args.dev_file and args.dev_file.exists() else None,
        "parameter_breakdown": breakdown,
        "target_module_count": len(targets) if targets is not None else None,
        "temperature": args.temperature if args.objective == "kd" else None,
        "hard_label_weight": args.hard_label_weight,
        "memory_efficient_completion_logits": args.memory_efficient_completion_logits,
        "training_metrics": result.metrics,
        "peak_allocated_vram_bytes": peak_vram,
        "skipped_train_ids": train_dataset.skipped,
        "environment": environment(),
    }
    write_json(args.output_dir / "training_manifest.json", manifest)
    print(f"Saved {artifact_type}: {artifact}")
    print(f"Training manifest: {args.output_dir / 'training_manifest.json'}")


if __name__ == "__main__":
    main()
