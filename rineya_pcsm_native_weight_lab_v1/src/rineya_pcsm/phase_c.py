"""Resource-bounded real-model one-step Phase-C primitives."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import encode_chat_example
from .hashing import canonical_json_hash, sha256_file
from .manifest import atomic_write_json, environment_manifest, utc_now


@dataclass(frozen=True)
class EncodedExample:
    row_id: str
    input_ids: torch.Tensor
    labels: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.input_ids.numel())


def encode_row(
    tokenizer: Any, row: dict[str, Any], maximum_tokens: int
) -> EncodedExample:
    row_id = str(row["id"])
    input_ids, labels = encode_chat_example(
        tokenizer,
        row["messages"],
        maximum_tokens=maximum_tokens,
        example_id=row_id,
    )
    return EncodedExample(
        row_id=row_id,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def load_native_model(
    model_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, Any]:
    """Load the true dense model; quantized adapters are prohibited in Phase C."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = cast(torch.nn.Module, loaded_model)
    model.to(device)
    model.train()
    cast(Any, model).config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        cast(Any, model).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(model, "enable_input_require_grads"):
        cast(Any, model).enable_input_require_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def resolve_parameter(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    parameters = dict(model.named_parameters())
    if name not in parameters:
        raise KeyError(f"Model has no parameter named {name}")
    parameter = parameters[name]
    if parameter.ndim != 2:
        raise ValueError(f"PCSM v1 target must be a matrix: {name}")
    return parameter


def _loss_for_example(
    model: torch.nn.Module, example: EncodedExample, device: torch.device
) -> tuple[torch.Tensor, int]:
    input_ids = example.input_ids.unsqueeze(0).to(device)
    labels = example.labels.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    label_positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).reshape(-1)
    if label_positions.numel() == 0 or bool((label_positions == 0).any()):
        raise RuntimeError("Example has no valid shifted assistant targets")
    logit_positions = label_positions - 1
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=logit_positions,
    )
    targets = labels[0].index_select(0, label_positions)
    logits = output.logits[0].float()
    if logits.shape[0] != targets.numel():
        raise RuntimeError("Model did not honor sparse logits_to_keep positions")
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss, int(targets.numel())


def aggregate_target_gradient(
    model: torch.nn.Module,
    target_name: str,
    examples: list[EncodedExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute a mean FP32 task gradient without taking an optimizer step."""

    if not examples:
        raise ValueError("At least one calibration example is required")
    target = resolve_parameter(model, target_name)
    target.requires_grad_(True)
    accumulated = torch.zeros_like(target, dtype=torch.float32, device="cpu")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for example in examples:
            model.zero_grad(set_to_none=True)
            loss, supervised_tokens = _loss_for_example(model, example, device)
            loss.backward()  # type: ignore[no-untyped-call]
            if target.grad is None:
                raise RuntimeError(f"Target received no gradient: {target_name}")
            gradient = target.grad.detach().float().cpu()
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"Non-finite target gradient: {target_name}")
            accumulated.add_(gradient)
            rows.append(
                {
                    "row_id": example.row_id,
                    "tokens": example.token_count,
                    "supervised_tokens": supervised_tokens,
                    "loss": float(loss.detach().cpu()),
                    "gradient_l2": float(torch.linalg.vector_norm(gradient)),
                }
            )
            del loss, gradient
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        target.requires_grad_(False)
        model.zero_grad(set_to_none=True)
    accumulated.div_(len(examples))
    report = {
        "target": target_name,
        "examples": rows,
        "mean_gradient_l2": float(torch.linalg.vector_norm(accumulated)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return accumulated, report


@torch.no_grad()
def evaluate_examples(
    model: torch.nn.Module,
    examples: list[EncodedExample],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("At least one evaluation example is required")
    model.eval()
    rows: list[dict[str, Any]] = []
    weighted_loss = 0.0
    token_count = 0
    started = time.perf_counter()
    try:
        for example in examples:
            loss, supervised_tokens = _loss_for_example(model, example, device)
            value = float(loss.detach().cpu())
            rows.append(
                {
                    "row_id": example.row_id,
                    "tokens": example.token_count,
                    "supervised_tokens": supervised_tokens,
                    "loss": value,
                }
            )
            weighted_loss += value * supervised_tokens
            token_count += supervised_tokens
    finally:
        model.train()
    return {
        "mean_nll": weighted_loss / token_count,
        "supervised_tokens": token_count,
        "examples": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def save_gradient_cache(
    output_directory: Path,
    *,
    name: str,
    gradient: torch.Tensor,
    report: dict[str, Any],
    model_path: Path,
    target_name: str,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    tensor_path = output_directory / f"{name}.safetensors"
    save_file({"gradient": gradient.contiguous().float()}, str(tensor_path))
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "kind": "phase_c_calibration_gradient_no_optimizer_step",
        "name": name,
        "target": target_name,
        "model": {
            "path": str(model_path.resolve()),
            "weight_sha256": sha256_file(model_path / "model.safetensors"),
            "config_sha256": sha256_file(model_path / "config.json"),
        },
        "tensor": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "shape": list(gradient.shape),
            "dtype": str(gradient.dtype),
        },
        "report": report,
        "environment": environment_manifest(),
    }
    manifest["cache_sha256"] = canonical_json_hash(manifest)
    atomic_write_json(output_directory / f"{name}.json", manifest)
    return manifest
