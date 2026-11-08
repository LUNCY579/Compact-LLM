from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import StreamingSafetensorsWriter, checkpoint_fingerprint
from .hashing import atomic_write_json, canonical_sha256, sha256_file
from .leakage import guard_non_sealed_path


class FisherError(RuntimeError):
    pass


def accumulate_squared_gradients(
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
    accumulator: dict[str, torch.Tensor],
) -> int:
    used = 0
    for name, parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float().cpu()
        if name not in accumulator:
            accumulator[name] = torch.zeros_like(gradient, dtype=torch.float32)
        accumulator[name].add_(gradient.square())
        used += 1
    return used


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )


def estimate_diagonal_fisher(
    *,
    model_path: Path,
    calibration_path: Path,
    destination: Path,
    limit: int = 32,
    max_length: int = 512,
    threads: int = 16,
    trust_remote_code: bool = False,
    abstain_logit_bias: float | None = None,
    max_shard_bytes: int = 1536 * 1024 * 1024,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    calibration_path = guard_non_sealed_path(calibration_path)
    destination = destination.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Fisher destination exists: {destination}")
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    rows = [
        json.loads(line)
        for line in calibration_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    if not rows:
        raise FisherError("empty calibration selection")
    torch.set_num_threads(max(1, int(threads)))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
    )
    model.to("cpu")
    model.train(False)
    model.config.use_cache = False
    if abstain_logit_bias is not None:
        if not hasattr(model.config, "abstain_logit_bias"):
            raise FisherError("abstain bias override requested for unsupported model")
        model.config.abstain_logit_bias = float(abstain_logit_bias)
    named_parameters = list(model.named_parameters(remove_duplicate=True))
    accumulator: dict[str, torch.Tensor] = {}
    losses = []
    token_counts = []
    started = time.perf_counter()
    for row in rows:
        rendered = _render_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        labels = inputs["input_ids"].clone()
        model.zero_grad(set_to_none=True)
        output = model(**inputs, labels=labels)
        loss = output.loss
        if not torch.isfinite(loss).item():
            raise FisherError(f"non-finite calibration loss: {row['case_id']}")
        loss.backward()
        used = accumulate_squared_gradients(named_parameters, accumulator)
        if used == 0:
            raise FisherError("no gradients contributed to Fisher estimate")
        losses.append(float(loss.item()))
        token_counts.append(int(inputs["input_ids"].numel()))
    for value in accumulator.values():
        value.div_(len(rows))
        if not torch.isfinite(value).all().item():
            raise FisherError("non-finite Fisher accumulator")
    writer = StreamingSafetensorsWriter(destination, max_shard_bytes)
    tensor_stats = {}
    for name in sorted(accumulator):
        value = accumulator[name]
        writer.add(name, value)
        tensor_stats[name] = {
            "shape": list(value.shape),
            "minimum": float(value.min().item()),
            "maximum": float(value.max().item()),
            "mean": float(value.mean().item()),
            "zero_fraction": float((value == 0).sum().item() / max(1, value.numel())),
        }
    output_index = writer.finalize()
    fingerprint = checkpoint_fingerprint(destination)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "diagonal_empirical_fisher",
        "objective": "prompt-token causal language-model NLL",
        "model_path": str(model_path),
        "model_fingerprint": checkpoint_fingerprint(model_path),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "case_ids_sha256": canonical_sha256([row["case_id"] for row in rows]),
        "examples": len(rows),
        "tokens": sum(token_counts),
        "max_length": max_length,
        "thinking": "disabled",
        "device": "cpu",
        "dtype": "float32",
        "threads": threads,
        "cpu": platform.processor(),
        "loss_mean": sum(losses) / len(losses),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "elapsed_seconds": time.perf_counter() - started,
        "output_index": output_index,
        "output_fingerprint": fingerprint,
        "tensor_count": len(tensor_stats),
        "tensor_stats_sha256": canonical_sha256(tensor_stats),
    }
    atomic_write_json(destination / "fisher_tensor_stats.json", tensor_stats)
    atomic_write_json(destination / "fisher_manifest.json", manifest)
    return manifest
