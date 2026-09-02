"""Frozen-backbone trainer for FWA tensors only."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from rineya_fwa.arms import apply_arm, count_trainable
from rineya_fwa.loading import freeze_backbone, load_supa, load_tokenizer
from rineya_fwa.paths import ARTIFACTS, assert_readonly_target
from rineya_fwa.provenance import write_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_id_list(ids) -> list[int]:
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_row(
    tokenizer: Any,
    row: dict[str, Any],
    max_len: int,
    *,
    supervise_answer_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Tokenise one chat row into input_ids / labels / attention_mask.

    ``supervise_answer_only`` restricts the loss to the assistant's content
    tokens. By default the trailing ``<|im_end|>`` and newline are supervised
    too; for the retrieval task those are free to predict, so two of every three
    supervised tokens carry no signal and the reported loss is ~3x smaller than
    the retrieval cross-entropy it is taken to represent. (P1/P2's 2e-4 "final
    loss" is diluted this way.)
    """
    messages = list(row["messages"])
    prompt = messages[:-1]
    full = messages
    try:
        prompt_ids = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        full_ids = tokenizer.apply_chat_template(
            full, add_generation_prompt=False, tokenize=True, enable_thinking=False
        )
    except TypeError:
        prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        full_ids = tokenizer.apply_chat_template(full, add_generation_prompt=False, tokenize=True)
    prompt_ids = _as_id_list(prompt_ids)
    full_ids = _as_id_list(full_ids)
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []
    labels = list(full_ids)
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100
    if supervise_answer_only:
        answer = str(row["messages"][-1].get("content") or "")
        n_answer = len(tokenizer.encode(answer, add_special_tokens=False))
        for i in range(n_prompt + n_answer, len(labels)):
            labels[i] = -100
    if all(v == -100 for v in labels):
        labels[-1] = full_ids[-1]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
    }


def _peak_vram() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def train_arm(
    *,
    arm: str,
    layers: list[int],
    train_file: Path,
    output_dir: Path,
    seed: int,
    rank: int = 16,
    max_steps: int = 64,
    max_sequence_length: int = 1024,
    learning_rate: float = 1e-4,
    grad_accum: int = 8,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    nan_abort_factor: float = 10.0,
    device: str | None = None,
    init_alpha: float = 0.05,
    beta_raw_init: float = 0.0,
    lambda_raw_init: float = -2.0,
    train_rows: list[dict[str, Any]] | None = None,
    gradient_checkpointing: bool = False,
    supervise_answer_only: bool = True,
) -> dict[str, Any]:
    seed_everything(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer()
    model = load_supa(device=device, eval_mode=False)
    freeze_backbone(model)
    handle = apply_arm(
        model,
        arm,
        layers=layers,
        rank=rank,
        init_alpha=init_alpha,
        beta_raw_init=beta_raw_init,
        lambda_raw_init=lambda_raw_init,
    )
    if handle is not None:
        for block in handle.blocks():
            for param in block.parameters():
                param.requires_grad = True
        model.eval()
        for wrapper in handle.wrappers.values():
            wrapper.block.train()
        params = handle.trainable_parameters()
        optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        if gradient_checkpointing:
            # Both configs declared gradient_checkpointing: true, but no run ever
            # enabled it. The backbone is frozen, so the checkpointed blocks need
            # a grad-requiring input to build a graph at all.
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        params = []
        optimizer = None

    rows = list(train_rows) if train_rows is not None else read_jsonl(train_file)
    rng = random.Random(seed)
    rng.shuffle(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    assert_readonly_target(output_dir / "dummy")
    t0 = time.perf_counter()
    history: list[dict[str, Any]] = []
    tokens_seen = 0
    supervised_tokens = 0
    examples_seen = 0
    aborted = None
    first_loss: float | None = None
    step = 0
    accum = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    skipped_nonfinite = 0
    while step < max_steps and optimizer is not None:
        row = rows[examples_seen % len(rows)]
        examples_seen += 1
        batch = encode_row(
            tokenizer, row, max_sequence_length,
            supervise_answer_only=supervise_answer_only,
        )
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        labels = batch["labels"].unsqueeze(0).to(device)
        attn = batch["attention_mask"].unsqueeze(0).to(device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            skipped_nonfinite += 1
            if skipped_nonfinite > 16:
                aborted = "non_finite_loss"
                break
            continue
        loss_value = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = loss_value
        if first_loss and first_loss > 0 and loss_value > nan_abort_factor * max(first_loss, 0.5) and step > 3:
            aborted = "divergence"
            break
        (loss / grad_accum).backward()
        accum += 1
        tokens_seen += int(attn.sum().item())
        supervised_tokens += int((labels != -100).sum().item())
        if accum % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            history.append({"step": step, "loss": loss_value})
            if step >= max_steps:
                break

    wall = time.perf_counter() - t0
    adapter_path = output_dir / "fwa_adapter.pt"
    if handle is not None:
        payload = {
            "arm": arm,
            "layers": layers,
            "rank": rank,
            "state_dict": {
                f"layer{idx}." + name: tensor.detach().cpu()
                for idx, wrapper in handle.wrappers.items()
                for name, tensor in wrapper.block.state_dict().items()
            },
        }
        assert_readonly_target(adapter_path)
        torch.save(payload, adapter_path)

    manifest = {
        "arm": arm,
        "layers": layers,
        "rank": rank,
        "seed": seed,
        "device": device,
        "optimizer_steps": step,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "wall_seconds": wall,
        "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
        "peak_vram_bytes": _peak_vram(),
        "trainable_parameters": count_trainable(handle),
        "loss_history": history,
        "first_loss": first_loss,
        "last_loss": history[-1]["loss"] if history else None,
        "aborted": aborted,
        "skipped_nonfinite": skipped_nonfinite,
        "adapter_path": str(adapter_path) if handle is not None else None,
        "max_sequence_length": max_sequence_length,
        "checkpoint_bytes": adapter_path.stat().st_size if adapter_path.exists() else 0,
        "init_alpha": init_alpha,
        "beta_raw_init": beta_raw_init,
        "lambda_raw_init": lambda_raw_init,
        "learning_rate": learning_rate,
        "grad_accum": grad_accum,
        "train_rows_source": "in_memory" if train_rows is not None else str(train_file),
        "n_train_rows": len(rows),
        "gradient_checkpointing": gradient_checkpointing,
        "supervise_answer_only": supervise_answer_only,
        "supervised_tokens": supervised_tokens,
        "final_alpha": (
            float(next(iter(handle.blocks())).alpha.detach()) if handle is not None else None
        ),
    }
    write_json(output_dir / "training_manifest.json", manifest, config=manifest)
    if handle is not None:
        handle.uninstall()
    # Release every GPU reference before returning. The P2 seed-33 OOM happened
    # because the model stayed pinned across sequential train+eval loads.
    del handle
    del optimizer
    del params
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def load_adapter(handle, adapter_path: Path) -> None:
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    for idx, wrapper in handle.wrappers.items():
        prefix = f"layer{idx}."
        subset = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        wrapper.block.load_state_dict(subset)
