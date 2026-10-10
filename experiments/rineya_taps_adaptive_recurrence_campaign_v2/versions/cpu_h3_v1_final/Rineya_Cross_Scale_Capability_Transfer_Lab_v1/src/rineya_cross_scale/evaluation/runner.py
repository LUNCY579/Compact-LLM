from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..checkpoint import checkpoint_fingerprint
from ..hashing import atomic_write_json, sha256_file
from ..leakage import (
    append_access_ledger,
    authorize_sealed_path,
    begin_sealed_access,
    guard_non_sealed_path,
)
from .scoring import aggregate_predictions, score_response


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class EvaluationError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"invalid JSONL line {line_number}: {path}") from error
            rows.append(row)
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationError("duplicate evaluation case IDs")
    return rows


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "summary": completed.stdout.strip(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": repr(error)}


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


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


def _invalid_generation(response: str, token_ids: list[int]) -> tuple[bool, str | None]:
    if not response.strip():
        return True, "empty"
    if len(token_ids) >= 8:
        most_common = max(token_ids.count(value) for value in set(token_ids))
        if most_common / len(token_ids) >= 0.8:
            return True, "token_repetition"
    return False, None


def run_evaluation(
    *,
    model_path: Path,
    dataset_path: Path,
    output_directory: Path,
    split_name: str,
    dtype_name: str = "float32",
    threads: int = 16,
    max_context_tokens: int = 4096,
    seed: int = 17,
    trust_remote_code: bool = False,
    case_limit: int | None = None,
    sealed_authorization: Path | None = None,
    expected_candidate_hash: str | None = None,
    abstain_logit_bias: float | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if dtype_name not in DTYPES:
        raise EvaluationError(f"unsupported dtype: {dtype_name}")
    model_path = model_path.expanduser().resolve()
    model_fingerprint = checkpoint_fingerprint(model_path)
    model_hash = model_fingerprint["digest"]
    if sealed_authorization is None:
        dataset_path = guard_non_sealed_path(dataset_path)
        access_mode = "development"
    else:
        if expected_candidate_hash is None:
            raise EvaluationError("sealed evaluation requires expected candidate hash")
        dataset_path = authorize_sealed_path(
            dataset_path, sealed_authorization, expected_candidate_hash
        )
        if expected_candidate_hash != model_hash:
            raise EvaluationError("loaded model fingerprint differs from frozen candidate")
        access_mode = "sealed_single_run"
    dataset_hash = sha256_file(dataset_path)
    rows = _read_jsonl(dataset_path)
    if case_limit is not None:
        rows = rows[: max(0, int(case_limit))]
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "predictions.jsonl"
    partial_path = output_directory / "predictions.partial.jsonl"
    manifest_path = output_directory / "run_manifest.json"
    summary_path = output_directory / "summary.json"
    ledger_path = output_directory.parent / "access_ledger.jsonl"
    run_identity = {
        "model_hash": model_hash,
        "dataset_sha256": dataset_hash,
        "split": split_name,
        "dtype": dtype_name,
        "threads": threads,
        "max_context_tokens": max_context_tokens,
        "seed": seed,
        "trust_remote_code": trust_remote_code,
        "case_limit": case_limit,
        "abstain_logit_bias": abstain_logit_bias,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_identity") != run_identity:
            raise EvaluationError("output directory belongs to a different run identity")
    elif predictions_path.exists() or partial_path.exists():
        raise EvaluationError("predictions exist without a matching run manifest")

    completed: dict[str, dict[str, Any]] = {}
    resume_path = partial_path if partial_path.exists() else predictions_path
    if resume and resume_path.exists():
        for row in _read_jsonl(resume_path):
            completed[row["case_id"]] = row
    elif not resume and (predictions_path.exists() or partial_path.exists()):
        raise FileExistsError(f"evaluation output already exists: {output_directory}")
    selected_case_ids = {row["case_id"] for row in rows}
    if predictions_path.exists() and resume:
        if set(completed) == selected_case_ids and summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise EvaluationError(
            "final predictions exist but do not exactly cover the selected cases"
        )
    if sealed_authorization is not None:
        begin_sealed_access(
            sealed_authorization,
            dataset_sha256=dataset_hash,
            candidate_hash=model_hash,
            output_directory=output_directory,
        )

    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(max(1, min(4, int(threads))))
    except RuntimeError:
        pass
    random.seed(seed)
    torch.manual_seed(seed)
    gpu_before = _gpu_snapshot()
    started = time.perf_counter()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=DTYPES[dtype_name],
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
    )
    model.to("cpu")
    model.eval()
    original_abstain_logit_bias = getattr(model.config, "abstain_logit_bias", None)
    if abstain_logit_bias is not None:
        if original_abstain_logit_bias is None:
            raise EvaluationError(
                "abstain_logit_bias override requested for a model without that field"
            )
        model.config.abstain_logit_bias = float(abstain_logit_bias)
    load_seconds = time.perf_counter() - load_started
    run_manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "run_identity": run_identity,
        "model_path": str(model_path),
        "model_fingerprint": model_fingerprint,
        "dataset_path": str(dataset_path),
        "dataset_items_selected": len(rows),
        "device": "cpu",
        "cpu": platform.processor(),
        "gpu_before": gpu_before,
        "load_seconds": load_seconds,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "resolved_attention_implementation": getattr(
            model.config, "_attn_implementation", None
        ),
        "thinking": "disabled",
        "decoding": "greedy",
        "temperature": 0,
        "batch_size": 1,
        "resume": resume,
        "rss_after_load_bytes": _process_rss_bytes(),
        "abstain_logit_bias_control": {
            "original": original_abstain_logit_bias,
            "resolved": getattr(model.config, "abstain_logit_bias", None),
        },
    }
    atomic_write_json(manifest_path, run_manifest)
    append_access_ledger(
        ledger_path,
        dataset_path=dataset_path,
        dataset_sha256=dataset_hash,
        mode=access_mode,
        model_hash=model_hash,
        details={"split": split_name, "output": str(output_directory)},
    )

    output_mode = "a" if partial_path.exists() and resume else "w"
    with partial_path.open(output_mode, encoding="utf-8", newline="\n") as output:
        for case in rows:
            if case["case_id"] in completed:
                continue
            rendered = _render_prompt(tokenizer, case["prompt"])
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=max_context_tokens,
                add_special_tokens=False,
            )
            input_ids = inputs["input_ids"].to("cpu")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to("cpu")
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=int(case["max_new_tokens"]),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            latency = time.perf_counter() - generation_started
            new_ids = generated[0, input_ids.shape[1] :].tolist()
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            invalid, invalid_reason = _invalid_generation(response, new_ids)
            record = {
                "schema_version": "1.0",
                "case_id": case["case_id"],
                "scenario_id": case["scenario_id"],
                "transformation_id": case["transformation_id"],
                "family": case["family"],
                "prompt_sha256": case["prompt_sha256"],
                "response": response,
                "response_sha256": __import__("hashlib").sha256(
                    response.encode("utf-8")
                ).hexdigest(),
                "score": score_response(response, case["scorer"]),
                "format_score": (
                    score_response(response, case["format_scorer"])
                    if case.get("format_scorer") is not None
                    else None
                ),
                "invalid_generation": invalid,
                "invalid_reason": invalid_reason,
                "input_tokens": int(input_ids.numel()),
                "generated_tokens": len(new_ids),
                "eos_emitted": tokenizer.eos_token_id in new_ids,
                "latency_seconds": latency,
                "metadata": case.get("metadata", {}),
            }
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())
            completed[case["case_id"]] = record

    os.replace(partial_path, predictions_path)
    ordered_predictions = [completed[case["case_id"]] for case in rows]
    summary = aggregate_predictions(ordered_predictions)
    summary.update(
        {
            "schema_version": "1.0",
            "model_hash": model_hash,
            "dataset_sha256": dataset_hash,
            "split": split_name,
            "elapsed_seconds_including_load": time.perf_counter() - started,
            "load_seconds": load_seconds,
            "predictions_sha256": sha256_file(predictions_path),
            "rss_final_bytes": _process_rss_bytes(),
        }
    )
    atomic_write_json(summary_path, summary)
    run_manifest["status"] = "complete"
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["summary_sha256"] = sha256_file(summary_path)
    run_manifest["predictions_sha256"] = sha256_file(predictions_path)
    atomic_write_json(manifest_path, run_manifest)
    return summary
