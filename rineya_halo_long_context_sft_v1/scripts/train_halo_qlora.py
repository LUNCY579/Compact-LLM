#!/usr/bin/env python3
"""Resume-safe Halo QLoRA SFT wrapper around the Phase 1 dense trainer."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (
    LAB_ROOT,
    REPO_ROOT,
    latest_checkpoint,
    list_checkpoints,
    load_config,
    python_executable,
    resolve_repo_path,
    stage_paths,
    validate_adapter_directory,
)

# Lazy-tokenizing engine avoids Phase-1 eager ChatSFTDataset stalls on 30k+ long rows.
ENGINE = Path(__file__).resolve().parent / "train_halo_engine.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Halo QLoRA Stage A with HF checkpoint resume.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "stage_a_qlora.yaml")
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Checkpoint directory, or 'latest' for the newest checkpoint-* under the run output dir.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--status", action="store_true", help="Print checkpoints and adapter status, then exit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def status_text(output_dir: Path) -> str:
    lines = [f"output_dir={output_dir}"]
    checkpoints = list_checkpoints(output_dir)
    if checkpoints:
        latest = checkpoints[-1]
        lines.append(f"checkpoints={len(checkpoints)} latest={latest.name}")
        for path in checkpoints[-5:]:
            lines.append(f"  {path.name}")
    else:
        lines.append("checkpoints=none")
    adapter = output_dir / "adapter"
    if adapter.exists():
        try:
            validate_adapter_directory(adapter)
            lines.append("adapter=valid")
        except RuntimeError as exc:
            lines.append(f"adapter=incomplete ({exc})")
    else:
        lines.append("adapter=missing")
    manifest = output_dir / "training_manifest.json"
    lines.append(f"training_manifest={'yes' if manifest.exists() else 'no'}")
    return "\n".join(lines)


def build_command(config: dict, args: argparse.Namespace) -> list[str]:
    paths = stage_paths(config)
    training = config["training"]
    train_file = args.train_file or paths["train_mix"]
    dev_file = args.dev_file or paths["val_trainer"]
    output_dir = args.output_dir or paths["output_dir"]
    if not Path(train_file).exists():
        raise FileNotFoundError(f"train mix missing: {train_file}. Run build_stage_mix.py first.")
    model_id = resolve_repo_path(config["paths"]["model_id"])
    if not model_id.exists():
        raise FileNotFoundError(f"Halo checkpoint missing: {model_id}")
    if not ENGINE.exists():
        raise FileNotFoundError(f"Halo train engine missing: {ENGINE}")

    max_steps = training["max_steps"] if args.max_steps is None else args.max_steps
    save_steps = training["save_steps"] if args.save_steps is None else args.save_steps
    eval_steps = training["eval_steps"] if args.eval_steps is None else args.eval_steps
    max_seq = training["max_sequence_length"] if args.max_sequence_length is None else args.max_sequence_length

    command = [
        python_executable(),
        str(ENGINE),
        "--model-id",
        str(model_id),
        "--train-file",
        str(train_file),
        "--dev-file",
        str(dev_file),
        "--output-dir",
        str(output_dir),
        "--selected-layers",
        str(training.get("selected_layers", "all")),
        "--target-module-group",
        str(training.get("target_module_group", "both")),
        "--lora-rank",
        str(training.get("lora_rank", 16)),
        "--lora-alpha",
        str(training.get("lora_alpha", 32)),
        "--lora-dropout",
        str(training.get("lora_dropout", 0.05)),
        "--learning-rate",
        str(training.get("learning_rate", 0.0002)),
        "--epochs",
        str(training.get("epochs", 1.0)),
        "--max-steps",
        str(max_steps),
        "--per-device-batch-size",
        str(training.get("per_device_batch_size", 1)),
        "--gradient-accumulation-steps",
        str(training.get("gradient_accumulation_steps", 8)),
        "--max-sequence-length",
        str(max_seq),
        "--warmup-ratio",
        str(training.get("warmup_ratio", 0.03)),
        "--weight-decay",
        str(training.get("weight_decay", 0.0)),
        "--logging-steps",
        str(training.get("logging_steps", 5)),
        "--save-steps",
        str(save_steps),
        "--eval-steps",
        str(eval_steps),
        "--save-total-limit",
        str(training.get("save_total_limit", 3)),
        "--seed",
        str(config.get("seed", 11)),
        "--run-label",
        str(training.get("run_label", "halo_long_context_stage_a_qlora")),
    ]
    if training.get("qlora") and not args.allow_cpu:
        command.append("--qlora")
    if training.get("gradient_checkpointing"):
        command.append("--gradient-checkpointing")
    if args.allow_cpu:
        command.append("--allow-cpu")
    resume = str(args.resume_from_checkpoint or "").strip()
    if resume:
        if resume.lower() == "latest":
            found = latest_checkpoint(Path(output_dir))
            if found is None:
                raise FileNotFoundError(
                    f"No checkpoint-* under {output_dir}. Start a fresh run instead of resume."
                )
            command.extend(["--resume-from-checkpoint", "latest"])
        else:
            command.extend(["--resume-from-checkpoint", resume])
    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = stage_paths(config)
    output_dir = Path(args.output_dir) if args.output_dir else paths["output_dir"]
    if args.status:
        print(status_text(output_dir))
        return
    command = build_command(config, args)
    print("Halo vault is never written. Adapter output:", output_dir / "adapter")
    print("Command:\n  " + " ".join(command))
    if args.dry_run:
        return
    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    adapter = output_dir / "adapter"
    if adapter.exists():
        validate_adapter_directory(adapter)
        print(f"Adapter valid: {adapter}")
    print(status_text(output_dir))


if __name__ == "__main__":
    main()
