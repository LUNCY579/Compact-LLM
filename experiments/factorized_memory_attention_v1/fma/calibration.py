"""Frozen-backbone teacher calibration for factorized Memory Attention."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from .memory import FactorizedMemoryBank, freeze_backbone_for_calibration


@dataclass(frozen=True)
class CalibrationStats:
    steps: int
    examples: int
    tokens: int
    initial_loss: float
    final_loss: float
    trainable_parameters: int
    dataset_sha256: str
    data_ids: tuple[str, ...]
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("calibration dataset is empty")
    return rows


def encode_rows(rows: list[dict], tokenizer, *, max_length: int) -> tuple[list[Tensor], tuple[str, ...]]:
    encoded: list[Tensor] = []
    data_ids: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", f"row-{index:06d}"))
        data_ids.append(row_id)
        if "input_ids" in row:
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
        elif "text" in row:
            ids = tokenizer(
                str(row["text"]),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]
        else:
            raise ValueError(f"{row_id}: requires input_ids or text")
        if ids.ndim != 1 or not ids.numel():
            raise ValueError(f"{row_id}: input_ids must be a non-empty 1D sequence")
        encoded.append(ids[:max_length])
    return encoded, tuple(data_ids)


def _pad_batch(items: list[Tensor], pad_token_id: int) -> tuple[Tensor, Tensor]:
    width = max(int(item.numel()) for item in items)
    ids = torch.full((len(items), width), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, item in enumerate(items):
        ids[index, : item.numel()] = item
        mask[index, : item.numel()] = 1
    return ids, mask


@contextmanager
def capture_attention_inputs(model, layers: tuple[int, ...]):
    captures: dict[int, Tensor] = {}
    handles = []
    for layer_idx in layers:
        module = model.model.layers[layer_idx].self_attn

        def hook(_module, args, kwargs, *, idx=layer_idx):
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                raise RuntimeError(f"layer {idx}: attention input hidden states unavailable")
            captures[idx] = hidden.detach()

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def native_qwen3_teacher(attn, hidden: Tensor) -> tuple[Tensor, Tensor]:
    """Return (standard V, content K) with native Qwen3 k_norm, before RoPE."""
    hidden_shape = (*hidden.shape[:-1], -1, attn.head_dim)
    standard_value = attn.v_proj(hidden)
    key_pre_rope = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).reshape(
        *hidden.shape[:-1], -1
    )
    return standard_value, key_pre_rope


def calibrate_qwen3_memory(
    model,
    tokenizer,
    bank: FactorizedMemoryBank,
    dataset_path: str | Path,
    *,
    steps: int = 100,
    batch_size: int = 1,
    lr: float = 1e-3,
    max_length: int = 512,
    seed: int = 20260925,
) -> CalibrationStats:
    """Fit only U/P/Norm to R_l = XW_V,l - K_l; never update the backbone."""
    if steps <= 0 or batch_size <= 0 or lr <= 0 or max_length <= 0:
        raise ValueError("steps, batch_size, lr and max_length must be positive")

    torch.manual_seed(seed)
    layers = bank.layers
    freeze = freeze_backbone_for_calibration(model, bank)
    if freeze["backbone_trainable"] != 0:
        raise RuntimeError("backbone freeze contract failed")

    model.eval()
    rows = load_jsonl_rows(dataset_path)
    items, data_ids = encode_rows(rows, tokenizer, max_length=max_length)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer needs pad_token_id or eos_token_id")

    optimizer = torch.optim.AdamW(bank.parameters(), lr=lr)
    device = next(model.parameters()).device
    losses: list[float] = []
    token_count = 0
    example_count = 0

    with capture_attention_inputs(model, layers) as captures:
        for step in range(steps):
            start = (step * batch_size) % len(items)
            batch_items = [items[(start + i) % len(items)] for i in range(batch_size)]
            ids, mask = _pad_batch(batch_items, pad_token_id)
            ids = ids.to(device)
            mask = mask.to(device)
            captures.clear()

            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            if set(captures) != set(layers):
                raise RuntimeError(
                    f"capture mismatch: got {sorted(captures)}, expected {list(layers)}"
                )

            loss = torch.zeros((), device=device, dtype=torch.float32)
            for layer_idx in layers:
                attn = model.model.layers[layer_idx].self_attn
                standard, key_pre_rope = native_qwen3_teacher(attn, captures[layer_idx])
                target = (standard - key_pre_rope).detach()
                predicted = bank.memory(ids, layer_idx)
                valid = mask.bool().unsqueeze(-1)
                squared = (predicted.float() - target.float()).square()
                loss = loss + squared.masked_select(valid.expand_as(squared)).mean()
            loss = loss / len(layers)

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite calibration loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            token_count += int(mask.sum())
            example_count += len(batch_items)

    return CalibrationStats(
        steps=steps,
        examples=example_count,
        tokens=token_count,
        initial_loss=losses[0],
        final_loss=losses[-1],
        trainable_parameters=freeze["memory_trainable"],
        dataset_sha256=sha256_file(dataset_path),
        data_ids=data_ids,
        seed=seed,
    )
