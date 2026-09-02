"""Frozen-backbone trainer for FWA tensors only."""

from __future__ import annotations

import json
import random
import time
from collections import deque
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


def tail_loss(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor | None:
    """Cross-entropy over the supervised tokens only, without materialising the
    full-length logit tensor.

    Passing ``labels`` to the model computes logits at every position: at
    T=1152 that is 1152 x 151936 values, ~350 MB in fp16 plus an fp32 copy in
    the loss and a gradient of the same size — roughly 1.4 GB — to supervise a
    single answer token. Since the supervised span is at the end of the
    sequence, we ask for just enough trailing positions to cover it.

    Numerically identical to the model's own loss: the logit at position p
    predicts the token at p+1, and the abstention bias is applied inside
    ``forward`` either way.
    """
    sup = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if sup.numel() == 0:
        return None
    total = int(input_ids.shape[1])
    first_pred = int(sup.min()) - 1
    if first_pred < 0:  # nothing before the first supervised token to predict it
        return None
    keep = total - first_pred
    out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep)
    logits = out.logits[0]  # (keep, V), covering positions [total-keep, total)
    idx = sup - 1 - (total - keep)
    return torch.nn.functional.cross_entropy(
        logits.index_select(0, idx).float(), labels[0].index_select(0, sup)
    )


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
    # Accepted for config compatibility and RECORDED AS UNUSED. There is no LR
    # scheduler in this trainer, so the P1/P2 configs' `warmup_ratio: 0.03`
    # never took effect. Kept as a parameter rather than deleted so the manifest
    # can say so explicitly instead of the config implying a warmup that is not
    # there.
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
    recent: deque[float] = deque(maxlen=16)
    baseline: float | None = None
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
        loss = tail_loss(model, input_ids, attn, labels)
        if loss is None or not torch.isfinite(loss):
            skipped_nonfinite += 1
            if skipped_nonfinite > 16:
                aborted = "non_finite_loss"
                break
            continue
        loss_value = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = loss_value
        # Divergence is a property of the trend, not of one example.
        #
        # The old guard aborted when a SINGLE example exceeded
        # nan_abort_factor * max(first_loss, 0.5). That was survivable when the
        # loss was diluted across the answer plus <|im_end|> plus a newline and
        # never spiked, but supervising only the answer token makes it a
        # per-example retrieval cross-entropy: one confidently-wrong item scores
        # 5-15 nats as a matter of course. It fired on example 35 of a 400-example
        # run and stopped arm B at 8 of 100 steps with aborted="divergence".
        #
        # Compare a trailing window against the opening window instead.
        recent.append(loss_value)
        if baseline is None and len(recent) == recent.maxlen:
            baseline = sum(recent) / len(recent)
        if (
            baseline is not None
            and len(recent) == recent.maxlen
            and step > 8
            and sum(recent) / len(recent) > nan_abort_factor * max(baseline, 0.5)
        ):
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
        "loss_baseline_window_mean": baseline,
        "loss_recent_window_mean": (sum(recent) / len(recent)) if recent else None,
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
        "warmup_ratio_declared_but_unused": warmup_ratio,
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
