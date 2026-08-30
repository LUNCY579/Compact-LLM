#!/usr/bin/env python3
"""Fixed-step adapter warm start for a real Rineya TGSA + MTP model.

Input is the Rineya family SFT JSONL format: the final assistant message is the
target and ``messages[:-1]`` is the prompt.  The script masks prompt loss,
trains the selected combined warm-start stage, and saves only newly introduced
TGSA/MTP tensors.  It is a development tool, not a confirmatory evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from rineya_tgsa.config import RineyaTGSAConfig
from rineya_tgsa.mtp_adapter import (
    CombinedRetrofitConfig,
    adapter_only_state_dict,
    apply_combined_stage,
    convert_and_attach_tgsa_mtp,
    state_sha256,
)


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("at least one converted layer is required")
    return layers


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")
    return mapping[value]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_sft_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            messages = value.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: messages must contain prompt and target")
            last = messages[-1]
            if not isinstance(last, dict) or last.get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: last message must be assistant")
            rows.append(value)
    if not rows:
        raise ValueError("training JSONL is empty")
    return rows


def tokenize_row(tokenizer: object, row: Mapping[str, object], max_length: int) -> dict[str, object]:
    messages = row["messages"]
    prompt_messages = messages[:-1]  # type: ignore[index]
    prompt_ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = list(int(token) for token in prompt_ids)
    full_ids = list(int(token) for token in full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"row {row.get('id')}: prompt chat-template tokens are not a prefix of the full row; "
            "loss masking would be ambiguous"
        )
    original_length = len(full_ids)
    full_ids = full_ids[:max_length]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            f"row {row.get('id')}: max_length={max_length} removes the entire assistant target"
        )
    prompt_length = min(len(prompt_ids), len(full_ids) - 1)
    return {
        "id": str(row.get("id", "unknown")),
        "input_ids": full_ids,
        "prompt_length": prompt_length,
        "original_length": original_length,
        "truncated": original_length > max_length,
    }


def collate(rows: Sequence[Mapping[str, object]], pad_token_id: int) -> tuple[Tensor, Tensor, Tensor]:
    length = max(len(row["input_ids"]) for row in rows)  # type: ignore[arg-type]
    input_ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), length), dtype=torch.bool)
    target = torch.zeros((len(rows), length), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)  # type: ignore[arg-type]
        input_ids[index, : len(ids)] = ids
        attention[index, : len(ids)] = True
        target[index, int(row["prompt_length"]) : len(ids)] = True
    return input_ids, attention, target


def batch_stream(
    rows: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
    seed: int,
) -> Iterable[list[Mapping[str, object]]]:
    rng = random.Random(seed)
    order = list(range(len(rows)))
    cursor = len(order)
    while True:
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        yield [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--stage", choices=("adapters", "attention", "converted_layers", "all"), default="adapters")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-offset", type=int, default=4)
    parser.add_argument("--trajectory-rank", type=int, default=32)
    parser.add_argument("--mtp-rank", type=int, default=64)
    parser.add_argument("--mtp-loss-weight", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise SystemExit("steps, batch-size, and gradient-accumulation must be positive")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:
        raise SystemExit("transformers with Qwen3 support is required") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_rows = load_sft_rows(args.train_jsonl)
    tokenized = [tokenize_row(tokenizer, row, args.max_length) for row in raw_rows]
    if args.batch_size > len(tokenized):
        raise SystemExit("batch-size may not exceed the number of tokenized rows")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    ).to(args.device)
    source_config = model.config.to_dict()
    tgsa_config = RineyaTGSAConfig.from_source_config(
        source_config,
        trajectory_rank=args.trajectory_rank,
        condition_q=True,
        condition_k=False,
        condition_v=False,
    )
    combined = CombinedRetrofitConfig(
        max_offset=args.max_offset,
        mtp_rank=args.mtp_rank,
        mtp_loss_weight=args.mtp_loss_weight,
        initialization_seed=args.seed,
    )
    wrapper, conversion_manifest = convert_and_attach_tgsa_mtp(
        model,
        tgsa_config=tgsa_config,
        layers=args.layers,
        combined_config=combined,
    )
    wrapper.to(args.device)
    stage_report = apply_combined_stage(wrapper, args.stage, converted_layers=args.layers)
    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    stream = iter(batch_stream(tokenized, batch_size=args.batch_size, seed=args.seed))
    trace: list[dict[str, object]] = []
    started = time.perf_counter()
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.steps):
        accumulated_loss = 0.0
        example_ids: list[str] = []
        for _ in range(args.gradient_accumulation):
            rows = next(stream)
            example_ids.extend(str(row["id"]) for row in rows)
            input_ids, attention, target = collate(rows, tokenizer.pad_token_id)
            output = wrapper(
                input_ids=input_ids.to(args.device),
                attention_mask=attention.to(args.device),
                target_mask=target.to(args.device),
                compute_joint_loss=True,
                use_cache=False,
            )
            assert output.loss is not None
            (output.loss / args.gradient_accumulation).backward()
            accumulated_loss += float(output.loss.detach().float()) / args.gradient_accumulation
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 0 or (step + 1) % max(1, args.steps // 20) == 0 or step + 1 == args.steps:
            trace.append(
                {
                    "step": step + 1,
                    "loss": accumulated_loss,
                    "gradient_norm_pre_clip": gradient_norm,
                    "example_ids_sha256": hashlib.sha256(
                        json.dumps(example_ids, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                }
            )

    adapter_state = adapter_only_state_dict(wrapper)
    checkpoint = args.output_dir / "tgsa_mtp_adapter.pt"
    torch.save(adapter_state, checkpoint)
    trace_path = args.output_dir / "training_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "rineya.tgsa-mtp-adapter-training.v1",
        "status": "complete",
        "model_dir": str(args.model_dir),
        "source_config": source_config,
        "train_jsonl": str(args.train_jsonl),
        "train_jsonl_sha256": sha256_file(args.train_jsonl),
        "rows": len(tokenized),
        "truncated_rows": sum(bool(row["truncated"]) for row in tokenized),
        "converted_layers": args.layers,
        "stage": stage_report,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "conversion": conversion_manifest,
        "adapter_checkpoint": checkpoint.name,
        "adapter_checkpoint_sha256": sha256_file(checkpoint),
        "adapter_state_sha256": state_sha256(adapter_state),
        "adapter_tensor_count": len(adapter_state),
        "training_trace": trace_path.name,
        "training_trace_sha256": sha256_file(trace_path),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_scope": (
            "fixed-step development warm start on user-supplied Rineya SFT rows; no held-out "
            "quality, complementarity, or wall-clock claim is produced by this script"
        ),
    }
    atomic_json(args.output_dir / "training_manifest.json", manifest)
    print(f"saved {len(adapter_state)} adapter tensors to {checkpoint}")


if __name__ == "__main__":
    main()
