from __future__ import annotations

"""Train Phase 3 DPO from either an SFT adapter or a materialized reduced model.

The original all-layer branch continues training its accepted SFT adapter. A
Phase 2B reduced profile has its SFT recovery already merged into a full
checkpoint, so this script attaches a fresh LoRA adapter with the same default
projection coverage before DPO. Both paths produce ordinary compatible PEFT
adapters for Phase 4 soups.
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import environment, read_jsonl, write_json
from scripts.phase_1_dense_sft.train_dense_sft import discover_target_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO adapter from an SFT or reduced checkpoint.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help=(
            "Existing trainable SFT adapter. Omit for a materialized Phase 2B "
            "checkpoint; a fresh LoRA adapter will be attached."
        ),
    )
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/preferences/train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=ROOT / "data/preferences/dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase_3_dpo")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-module-group", choices=("both", "attention", "mlp"), default="both")
    return parser.parse_args()


def preference_dataset(path: Path) -> Dataset:
    return Dataset.from_list(
        [{key: row[key] for key in ("prompt", "chosen", "rejected")} for row in read_jsonl(path)]
    )


def build_trainable_model(args: argparse.Namespace):
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    if args.sft_adapter is not None:
        if not args.sft_adapter.exists():
            raise FileNotFoundError(args.sft_adapter)
        model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
        initialization = "continued_existing_sft_adapter"
        targets = None
    else:
        targets = discover_target_modules(base, None, args.target_module_group)
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(base, config)
        initialization = "fresh_lora_on_materialized_phase_2b_model"
    return model, initialization, targets


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model, initialization, targets = build_trainable_model(args)

    config = DPOConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        seed=args.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=preference_dataset(args.train_file),
        eval_dataset=preference_dataset(args.dev_file),
    )
    metrics = trainer.train().metrics
    output = args.output_dir / "adapter"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(
        args.output_dir / "training_manifest.json",
        {
            "model_id": args.model_id,
            "sft_adapter": str(args.sft_adapter) if args.sft_adapter is not None else None,
            "initialization": initialization,
            "target_module_group": args.target_module_group,
            "target_module_count": len(targets) if targets is not None else None,
            "seed": args.seed,
            "metrics": metrics,
            "environment": environment(),
            "compatibility_rule": (
                "Phase 4 ingredients must use the same model_id, adapter structure, "
                "target modules, rank, and parameter shapes."
            ),
        },
    )
    print("Saved", output)


if __name__ == "__main__":
    main()
