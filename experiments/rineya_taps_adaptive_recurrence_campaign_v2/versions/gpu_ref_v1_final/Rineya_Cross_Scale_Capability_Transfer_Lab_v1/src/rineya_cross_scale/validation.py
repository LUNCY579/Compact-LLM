from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import CheckpointReader, checkpoint_fingerprint
from .evaluation.scoring import normalize_text
from .hashing import atomic_write_json, canonical_sha256


class ValidationError(RuntimeError):
    pass


def short_period_repetition(token_ids: list[int], maximum_period: int = 4) -> bool:
    if len(token_ids) < 6:
        return False
    for period in range(1, min(maximum_period, len(token_ids) // 2) + 1):
        if all(token_ids[index] == token_ids[index % period] for index in range(len(token_ids))):
            return True
    return False


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def validate_checkpoint(
    *,
    checkpoint_path: Path,
    template_path: Path,
    output_path: Path,
    expected_parameter_count: int = 596_049_920,
    expected_vocab_size: int = 151_936,
    threads: int = 16,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    template_path = template_path.expanduser().resolve()
    torch.set_num_threads(max(1, int(threads)))
    structural_errors: list[str] = []
    tensor_records: list[dict[str, Any]] = []
    with CheckpointReader(template_path) as template, CheckpointReader(
        checkpoint_path
    ) as candidate:
        missing = sorted(set(template.keys) - set(candidate.keys))
        unexpected = sorted(set(candidate.keys) - set(template.keys))
        if missing:
            structural_errors.append(f"missing keys: {missing[:10]}")
        if unexpected:
            structural_errors.append(f"unexpected keys: {unexpected[:10]}")
        for key in sorted(set(template.keys) & set(candidate.keys)):
            target = template.descriptor(key)
            value = candidate.get_tensor(key)
            if tuple(value.shape) != target.shape:
                structural_errors.append(
                    f"shape mismatch {key}: {tuple(value.shape)} vs {target.shape}"
                )
            nan_count = int(torch.isnan(value).sum().item()) if value.is_floating_point() else 0
            inf_count = int(torch.isinf(value).sum().item()) if value.is_floating_point() else 0
            if nan_count or inf_count:
                structural_errors.append(
                    f"non-finite {key}: nan={nan_count}, inf={inf_count}"
                )
            tensor_records.append(
                {
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                }
            )

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model.to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameter_count:
        structural_errors.append(
            f"parameter count {parameter_count} != {expected_parameter_count}"
        )
    if int(model.config.vocab_size) != expected_vocab_size:
        structural_errors.append(
            f"vocab size {model.config.vocab_size} != {expected_vocab_size}"
        )
    prompt = _render_prompt(tokenizer, "Reply with exactly the word READY.")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        output = model(**inputs)
        logits_finite = bool(torch.isfinite(output.logits).all().item())
        generations = []
        for _ in range(2):
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            generations.append(generated[0, inputs["input_ids"].shape[1] :].tolist())
    deterministic = generations[0] == generations[1]
    generation_text = tokenizer.decode(generations[0], skip_special_tokens=True)
    requested_response_correct = normalize_text(generation_text) == "ready"
    repetition_collapse = short_period_repetition(generations[0])
    tied = False
    try:
        tied = (
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        )
    except Exception:
        tied = False
    eos_id = tokenizer.eos_token_id
    eos_valid = isinstance(eos_id, int) and 0 <= eos_id < expected_vocab_size
    if not logits_finite:
        structural_errors.append("forward logits are non-finite")
    if not deterministic:
        structural_errors.append("greedy generation is non-deterministic")
    if not requested_response_correct:
        structural_errors.append(
            f"exact smoke response is incoherent or noncompliant: {generation_text!r}"
        )
    if repetition_collapse:
        structural_errors.append("generation collapsed into a short-period token loop")
    if not tied:
        structural_errors.append("tied embedding storage contract is not active")
    if not eos_valid:
        structural_errors.append("EOS token ID is invalid")
    eos_emitted = eos_id in generations[0] if eos_valid else False
    if not eos_emitted:
        structural_errors.append("EOS was not emitted in the exact-response smoke")
    loader_missing = list(loading_info.get("missing_keys") or [])
    loader_unexpected = list(loading_info.get("unexpected_keys") or [])
    loader_mismatched = list(loading_info.get("mismatched_keys") or [])
    if loader_missing:
        structural_errors.append(f"loader missing keys: {loader_missing}")
    if loader_unexpected:
        structural_errors.append(f"loader unexpected keys: {loader_unexpected}")
    if loader_mismatched:
        structural_errors.append(f"loader mismatched keys: {loader_mismatched}")
    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GO" if not structural_errors else "NO_GO",
        "checkpoint_path": str(checkpoint_path),
        "template_path": str(template_path),
        "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint_path),
        "tensor_count": len(tensor_records),
        "tensor_records_sha256": canonical_sha256(tensor_records),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "loader_missing_keys": loader_missing,
        "loader_unexpected_keys": loader_unexpected,
        "loader_mismatched_keys": loader_mismatched,
        "parameter_count": parameter_count,
        "expected_parameter_count": expected_parameter_count,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "load_seconds": load_seconds,
        "device": "cpu",
        "cpu": platform.processor(),
        "dtype": "float32",
        "logits_finite": logits_finite,
        "logits_shape": list(output.logits.shape),
        "deterministic_generation": deterministic,
        "generation_token_ids": generations[0],
        "generation_text": generation_text,
        "requested_response_correct": requested_response_correct,
        "short_period_repetition": repetition_collapse,
        "eos_token_id": eos_id,
        "eos_configuration_valid": eos_valid,
        "eos_emitted_in_smoke": eos_emitted,
        "tied_embeddings_active": tied,
        "errors": structural_errors,
    }
    atomic_write_json(output_path, report)
    if structural_errors:
        raise ValidationError("; ".join(structural_errors))
    return report
