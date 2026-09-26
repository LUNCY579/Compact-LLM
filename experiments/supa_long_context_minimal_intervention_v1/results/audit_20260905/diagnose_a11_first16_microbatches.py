#!/usr/bin/env python3
"""Bounded A1.1 CUDA diagnostic; execution belongs to the reviewing root task.

Requires CUDA_LAUNCH_BLOCKING=1 and the recorded native allocator policy from
the caller. Reuses the frozen slcmi curriculum, adapter, batch, checkpointing,
and gate helpers. Runs exactly the first 16 bounded-curriculum microbatches if
they succeed, with fp16 autocast and fixed GradScaler scale 4096 / accumulation
8. No optimizer is constructed, no optimizer/scaler update occurs, gradients
are cleared after each group of 8, and no evaluation or checkpoint save runs.

This diagnoses forward/backward execution only. The second group deliberately
retains initial parameters; it is not a substitute for a training receipt.
All outputs are confined to a fresh directory under this audit directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
AUDIT_ROOT = Path(__file__).resolve().parent
LAB_ROOT = AUDIT_ROOT.parents[1]
REPO_ROOT = LAB_ROOT.parents[1]
CORE_PATH = (
    REPO_ROOT.parent / "Chat-Bubble" / "Rineya_Published_Cores"
    / "Rineya-Supa-A1.1-1.1.0-devfreeze"
)
ARM_A_ROOT = LAB_ROOT / "results" / "phase1" / "arm_A"
MAX_MICROBATCHES = 16
GRAD_ACCUM = 8
SCALE = 4096
SEED = 17
MAX_LENGTH = 3840
TRAIN_PACKS = 600
ALLOCATOR_POLICY = "backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8"
CANONICAL_FILES = (
    "CORE.json", "FILES.json", "config.json", "generation_config.json",
    "merges.txt", "model.safetensors", "modeling_supa_a11.py",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_rows(rows: list[dict]) -> str:
    # The exact serialization used by phase1_run._digest_rows.
    return hashlib.sha256(json.dumps(
        rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()


def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return f"NON_FINITE:{value}"
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(finite_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_hashes() -> tuple[dict, dict]:
    """Only CPU file I/O; inspect every canonical file even if one read fails."""
    hashes, errors = {}, {}
    for name in CANONICAL_FILES:
        try:
            hashes[name] = file_sha256(CORE_PATH / name)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return hashes, errors


def frozen_source_hashes() -> tuple[dict, dict]:
    hashes, errors = {}, {}
    files = sorted([
        *LAB_ROOT.glob("src/**/*.py"), *LAB_ROOT.glob("scripts/*.py"),
        *LAB_ROOT.glob("configs/*.json"),
    ])
    for path in files:
        key = path.relative_to(REPO_ROOT).as_posix()
        try:
            hashes[key] = file_sha256(path)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
    return hashes, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Fresh directory below this audit directory")
    args = parser.parse_args()
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output_dir or AUDIT_ROOT / f"a11_first16_diagnostic_{suffix}_{os.getpid()}").resolve()
    if output == AUDIT_ROOT or not output.is_relative_to(AUDIT_ROOT):
        parser.error("--output-dir must be a new child directory of this audit directory")
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "diagnostic.json"
    events_path = output / "events.jsonl"
    report: dict[str, Any] = {
        "schema_version": 1, "status": "STARTING", "started_utc": utc_now(),
        "process_id": os.getpid(), "python": sys.executable,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "output_directory": str(output), "canonical_core": str(CORE_PATH),
        "diagnostic_only": True, "arm": "B", "core": "a11",
        "max_microbatches": MAX_MICROBATCHES, "completed_microbatches": 0,
        "optimizer_constructed": False, "optimizer_updates": 0,
        "optimizer_steps_attempted": 0, "scaler_updates": 0,
        "target_evaluations": 0, "dev_evaluations": 0,
        "preservation_evaluations": 0, "checkpoint_writes": 0,
        "gradient_accumulation": GRAD_ACCUM, "loss_scale": SCALE,
        "seed": SEED, "max_length": MAX_LENGTH, "train_packs": TRAIN_PACKS,
        "autocast_dtype": "float16", "adapter_dtype": "float32",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_LAUNCH_BLOCKING", "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF", "PYTHONUTF8", "PYTHONUNBUFFERED",
        )},
        "events": [], "metric_errors": [], "zeroed_after_microbatches": [],
        "scope_note": "No optimizer exists; second accumulation group retains initial parameters.",
    }
    torch_module = None
    model = None
    start = time.perf_counter()

    def event(phase: str, state: str, example: dict | None = None, **detail: Any) -> None:
        record = {"utc": utc_now(), "phase": phase, "state": state, **detail}
        if example is not None:
            record.update(
                example_id=example["example_id"], input_token_count=len(example["input_ids"]),
                prompt_token_count=example["n_prompt_tokens"],
                answer_token_count=example["n_answer_tokens"],
            )
        report["last_event"] = record
        report["events"].append(record)
        payload = json.dumps(finite_json(record), ensure_ascii=False, allow_nan=False)
        with events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json(receipt_path, report)
        print(payload, flush=True)

    def metric(label: str, function: Any) -> Any:
        try:
            return finite_json(function())
        except Exception as exc:
            report["metric_errors"].append({
                "utc": utc_now(), "metric": label,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            return None

    write_json(receipt_path, report)
    try:
        event("canonical_hash_before", "START")
        before, before_errors = canonical_hashes()
        sources_before, source_errors = frozen_source_hashes()
        report.update(
            checkpoint_sha256_before=before, checkpoint_hash_errors_before=before_errors,
            source_sha256_before=sources_before, source_hash_errors_before=source_errors,
        )
        event("canonical_hash_before", "COMPLETE")
        if before_errors or len(before) != 10 or source_errors:
            raise RuntimeError("Canonical/source preflight file hashing failed")
        arm_a_manifest = json.loads((ARM_A_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
        arm_a_dataset = json.loads((ARM_A_ROOT / "dataset.json").read_text(encoding="utf-8"))
        if arm_a_manifest["status"] != "COMPLETE" or arm_a_manifest["config"]["core"] != "a11":
            raise RuntimeError("A completed bounded A1.1 Arm A receipt is required")
        if before != arm_a_manifest["checkpoint_sha256_before"] or before != arm_a_manifest["checkpoint_sha256_after"]:
            raise RuntimeError("Canonical files differ from completed Arm A")
        if sources_before != arm_a_manifest["source_sha256"]:
            raise RuntimeError("Frozen source/config files differ from completed Arm A")
        report["arm_a_hash_receipts_match"] = True
        if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
            raise RuntimeError("Root caller must supply CUDA_LAUNCH_BLOCKING=1 before Python starts")
        for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            if os.environ.get(variable) != ALLOCATOR_POLICY:
                raise RuntimeError(f"Root caller must supply the recorded allocator policy in {variable}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
            raise RuntimeError("Diagnostic requires default GPU visibility or CUDA_VISIBLE_DEVICES=0")

        event("imports", "START")
        sys.path.insert(0, str(LAB_ROOT / "src"))
        import torch
        from transformers import AutoTokenizer
        from slcmi.adapters import (
            DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_LAYERS, DEFAULT_RANK,
            build_adapter, trained_layer_indices,
        )
        from slcmi.evaluation import load_model
        from slcmi.provenance import CORES
        from slcmi.training import _batch, _training_environment, base_dtype, gate_disabled
        from slcmi.training_data import (
            CURRICULUM_SEED, LCR_SFT, build_dataset, load_packs, sample_curriculum,
        )
        torch_module = torch
        event("imports", "COMPLETE")
        core = CORES["a11"]
        if core["path"].resolve() != CORE_PATH.resolve() or core["dtype"] != "float16":
            raise RuntimeError("Frozen slcmi core selection differs from the diagnostic contract")
        if (DEFAULT_LAYERS, DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT) != ((17, 18, 19), 4, 8, 0.0):
            raise RuntimeError("Frozen adapter defaults differ from the diagnostic contract")
        if file_sha256(LCR_SFT) != arm_a_manifest["config"]["data_sha256"]["curriculum"]:
            raise RuntimeError("Curriculum file differs from completed Arm A")

        event("curriculum", "START")
        packs = load_packs(LCR_SFT)
        train_raw = sample_curriculum(packs, "train", TRAIN_PACKS)
        tokenizer = AutoTokenizer.from_pretrained(core["path"], trust_remote_code=True, local_files_only=True)
        train_set, train_stats = build_dataset(tokenizer, train_raw, MAX_LENGTH)
        train_fingerprint = digest_rows(train_set)
        if train_stats["n_dropped_too_long"] or len(train_set) != 2400:
            raise RuntimeError("The exact 600-pack bounded curriculum must retain all 2400 examples")
        if train_fingerprint != arm_a_dataset["fingerprint"]["train"]:
            raise RuntimeError("Ordered tokenized training fingerprint differs from completed Arm A")
        first16 = train_set[:MAX_MICROBATCHES]
        report.update(
            curriculum_seed=CURRICULUM_SEED, train_stats=train_stats,
            train_fingerprint=train_fingerprint, arm_a_train_fingerprint_match=True,
            first16_example_ids=[row["example_id"] for row in first16],
            first16_input_tokens=[len(row["input_ids"]) for row in first16],
        )
        write_json(output / "first16_inputs.json", [
            {key: row[key] for key in (
                "example_id", "scenario_id", "input_ids", "labels",
                "n_prompt_tokens", "n_answer_tokens",
            )} for row in first16
        ])
        del tokenizer, packs, train_raw, train_set
        gc.collect()
        event("curriculum", "COMPLETE")

        event("model_load", "START")
        tokenizer, model = load_model(core["path"], "cuda", dtype=core["dtype"], gate=True)
        torch.cuda.synchronize()
        event("model_load", "COMPLETE")
        event("adapter_build", "START")
        model, adapter_info = build_adapter(
            model, "lora", DEFAULT_LAYERS, DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT, SEED,
        )
        torch.cuda.synchronize()
        report["adapter"] = adapter_info
        if trained_layer_indices(model) != {17, 18, 19}:
            raise RuntimeError("Trainable layers differ from the fixed late-attention adapter")
        named_params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not named_params or any(not ({"lora_A", "lora_B"} & set(name.split("."))) for name, _ in named_params):
            raise RuntimeError("Only actual LoRA A/B parameters may be trainable")
        event("adapter_build", "COMPLETE")

        # Match the start of slcmi.training.train, omitting optimizer construction.
        torch.manual_seed(SEED)
        if base_dtype(model) != torch.float16:
            raise RuntimeError("A1.1 frozen base is not float16")
        for _, parameter in named_params:
            parameter.data = parameter.data.float()
        scaler = torch.amp.GradScaler("cuda", init_scale=SCALE, enabled=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        report["status"] = "RUNNING"
        with _training_environment(model, True) as checkpointing, gate_disabled(model) as original_bias:
            if not checkpointing or not original_bias or model.config.abstain_logit_bias != 0.0:
                raise RuntimeError("Mandatory checkpointing or gate-off training state is absent")
            report.update(gradient_checkpointing_active=True, gate_disabled_for_training=True)
            for index, example in enumerate(first16, start=1):
                event("input_transfer", "START", example, microbatch=index)
                batch = _batch(example, "cuda")
                torch.cuda.synchronize()
                event("input_transfer", "COMPLETE", example, microbatch=index)
                event("forward", "START", example, microbatch=index)
                with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = model(**batch)
                torch.cuda.synchronize()
                event("forward", "COMPLETE", example, microbatch=index)
                event("backward", "START", example, microbatch=index)
                loss = out.loss / GRAD_ACCUM
                scaler.scale(loss).backward()
                torch.cuda.synchronize()
                report["completed_microbatches"] = index
                observed_loss = metric(f"microbatch_{index}_loss", lambda: float(out.loss.detach()))
                event("backward", "COMPLETE", example, microbatch=index, unscaled_loss=observed_loss)
                del out, loss, batch
                if index % GRAD_ACCUM == 0:
                    event("zero_grad", "START", example, microbatch=index)
                    count = sum(parameter.grad is not None for _, parameter in named_params)
                    if count == 0:
                        raise RuntimeError("No gradient reached a LoRA parameter")
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    report["zeroed_after_microbatches"].append(index)
                    event("zero_grad", "COMPLETE", example, microbatch=index, parameters_with_gradient=count)
            report["loss_scale_after"] = metric("loss_scale_after", scaler.get_scale)
        report["gate_restored_after_training"] = model.config.abstain_logit_bias == original_bias
        if report["completed_microbatches"] != MAX_MICROBATCHES or report["zeroed_after_microbatches"] != [8, 16]:
            raise RuntimeError("Diagnostic microbatch/gradient-clear bounds were not fulfilled")
        report["status"] = "COMPLETE"
        event("diagnostic", "COMPLETE")
    except BaseException as exc:
        report["status"] = "FAILED"
        report["failure"] = {
            "type": type(exc).__name__, "message": str(exc),
            "last_event": report.get("last_event"), "traceback": traceback.format_exc(),
        }
        (output / "traceback.txt").write_text(report["failure"]["traceback"], encoding="utf-8")
        write_json(receipt_path, report)
        print(report["failure"]["traceback"], file=sys.stderr, flush=True)
    finally:
        # Persist failure evidence and CPU-only integrity receipts BEFORE optional
        # CUDA metrics. No Run, empty_cache, GPU copies, model saves, or combine.
        after, after_errors = canonical_hashes()
        sources_after, source_errors_after = frozen_source_hashes()
        report.update(
            checkpoint_sha256_after=after, checkpoint_hash_errors_after=after_errors,
            source_sha256_after=sources_after, source_hash_errors_after=source_errors_after,
            canonical_checkpoint_unchanged=(
                not after_errors and len(after) == 10
                and after == report.get("checkpoint_sha256_before")
            ),
            frozen_sources_unchanged=(
                not source_errors_after and sources_after == report.get("source_sha256_before")
            ),
            ended_utc=utc_now(), wall_seconds=round(time.perf_counter() - start, 6),
        )
        if not report["canonical_checkpoint_unchanged"] or not report["frozen_sources_unchanged"]:
            report["status"] = "FAILED"
            report["integrity_failure"] = "Canonical or frozen-source before/after hash checks failed"
        write_json(receipt_path, report)
        if torch_module is not None and torch_module.cuda.is_initialized():
            report["cuda_metrics"] = {
                "allocated_bytes": metric("allocated_bytes", torch_module.cuda.memory_allocated),
                "reserved_bytes": metric("reserved_bytes", torch_module.cuda.memory_reserved),
                "peak_allocated_bytes": metric("peak_allocated_bytes", torch_module.cuda.max_memory_allocated),
                "peak_reserved_bytes": metric("peak_reserved_bytes", torch_module.cuda.max_memory_reserved),
                "free_total_bytes": metric("free_total_bytes", lambda: list(torch_module.cuda.mem_get_info())),
            }
        write_json(receipt_path, report)
        print(json.dumps({
            "status": report["status"], "receipt": str(receipt_path),
            "completed_microbatches": report["completed_microbatches"],
            "optimizer_updates": 0,
            "canonical_checkpoint_unchanged": report["canonical_checkpoint_unchanged"],
        }), flush=True)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
