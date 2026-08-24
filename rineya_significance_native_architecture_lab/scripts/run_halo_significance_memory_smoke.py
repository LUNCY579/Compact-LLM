#!/usr/bin/env python3
"""Run the bounded Experiment-D Halo CPU runtime smoke.

The run has two gates only: exact M=0 hard-eviction identity and finite M=4
closed-loop attention.  It does not decode or score an answer and is not a
quality comparison among memory variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    build_manifest,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.runtime import (  # noqa: E402
    SignificanceCache,
    SignificanceMemory,
    SignificancePolicy,
    chunked_prefill,
    chunked_prefill_with_memory,
)


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _resolve(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _dtype(name: str):
    import torch

    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def _prompt_only(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("context"), str) or not isinstance(row.get("question"), str):
                raise RuntimeError("prompt row requires string context and question")
            # Deliberately return only inference-time fields.  Needle profiles,
            # answers, annotations, and all other benchmark metadata stay out
            # of the model/memory runtime.
            return {
                "id": str(row.get("id", "halo-memory-smoke-0")),
                "context": row["context"],
                "question": row["question"],
            }
    raise RuntimeError("benchmark has no prompt row")


def _check_geometry(model_config: Any, expected: dict[str, Any]) -> None:
    checks = {
        "num_hidden_layers": "expected_layers",
        "hidden_size": "expected_hidden_size",
        "num_attention_heads": "expected_query_heads",
        "num_key_value_heads": "expected_kv_heads",
        "head_dim": "expected_head_dim",
        "vocab_size": "expected_vocab_size",
    }
    for field, expectation in checks.items():
        if int(getattr(model_config, field)) != int(expected[expectation]):
            raise RuntimeError(f"Halo geometry mismatch for {field}")
    if str(expected["expected_architecture"]) not in (model_config.architectures or []):
        raise RuntimeError("Halo architecture mismatch")


def _cache_kv_bytes(cache: SignificanceCache) -> int:
    return int(
        sum(
            tensor.numel() * tensor.element_size()
            for layer in cache.layers
            for tensor in (layer.keys, layer.values)
            if tensor is not None
        )
    )


def _cache_identity(left: SignificanceCache, right: SignificanceCache) -> dict[str, Any]:
    mismatches: list[str] = []
    if left.true_length != right.true_length:
        mismatches.append("true_length")
    if left.stored_length() != right.stored_length():
        mismatches.append("stored_length")
    for layer_index, (left_layer, right_layer) in enumerate(
        zip(left.layers, right.layers, strict=True)
    ):
        for name in ("keys", "values"):
            if not __import__("torch").equal(
                getattr(left_layer, name),
                getattr(right_layer, name),
            ):
                mismatches.append(f"layer_{layer_index}:{name}")
        for name in ("scores", "counts", "positions", "entry_ids"):
            if not __import__("torch").equal(
                getattr(left_layer.state, name),
                getattr(right_layer.state, name),
            ):
                mismatches.append(f"layer_{layer_index}:{name}")
        if left_layer.evictions != right_layer.evictions:
            mismatches.append(f"layer_{layer_index}:evictions")
    return {"exact": not mismatches, "mismatches": mismatches}


def _slot_diagnostics(memory: SignificanceMemory) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for layer_index in range(memory.num_layers):
        state = memory.retrieve_slots(layer_index)
        valid = state.valid
        layers.append(
            {
                "layer_index": layer_index,
                "physical_slots_per_batch_head": memory.num_slots,
                "valid_slots": int(valid.sum().item()),
                "valid_slots_per_batch_head": state.valid.sum(dim=-1).tolist(),
                "source_count_sum": int(state.source_counts.sum().item()),
                "source_count_min_valid": int(state.source_counts[valid].min().item()),
                "source_count_max_valid": int(state.source_counts[valid].max().item()),
                "source_position_min": int(state.source_positions[valid].min().item()),
                "source_position_max": int(state.source_positions[valid].max().item()),
                "created_at": sorted({int(value) for value in state.created_at[valid].tolist()}),
                "last_updated_at": sorted(
                    {int(value) for value in state.last_updated_at[valid].tolist()}
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "variant": memory.variant,
        "slot_assignment": "semantic_entry_id_modulo_slot_count",
        "key_semantics": "weighted aggregate of already post-RoPE evicted keys",
        "value_semantics": "weighted aggregate of evicted values",
        "source_position_semantics": "maximum represented absolute position; diagnostic only",
        "visibility_semantics": "current slot contents become visible at last_updated_at",
        "invalid_slot_semantics": "physically present and fully attention-masked",
        "state_sha256": memory.state_sha256(),
        "byte_ledger": asdict(memory.byte_ledger()),
        "layers": layers,
    }


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    import psutil
    import torch
    import transformers
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.expanduser().resolve()
    config = _load(config_path)
    paths = dict(config["paths"])
    runtime = dict(config["runtime"])
    model_expected = dict(config["model"])
    cache_config = dict(config["cache"])
    memory_config = dict(config["memory_slots"])
    base = REPO_ROOT if paths["base"] == "repository_root" else LAB_ROOT
    model_path = _resolve(str(paths["model"]), base=base)
    tokenizer_path = _resolve(str(paths["tokenizer"]), base=base)
    benchmark_path = _resolve(str(paths["benchmark"]), base=base)
    output = (
        output_override.expanduser().resolve()
        if output_override is not None
        else _resolve(str(paths["output_root"]), base=base)
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started = time.perf_counter()
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]
    seed = int(runtime["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dtype = _dtype(str(runtime["dtype"]))

    hashes = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(tokenizer_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
        "chat_template": None,
    }
    expected_hashes = dict(model_expected["expected_hashes"])
    mismatches = {
        name: {"expected": str(expected), "observed": hashes[name]}
        for name, expected in expected_hashes.items()
        if str(expected).lower() != str(hashes[name]).lower()
    }
    if mismatches:
        raise RuntimeError(f"checkpoint hash mismatch: {mismatches}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    _check_geometry(model.config, model_expected)
    rss_samples.append(process.memory_info().rss)

    prompt = _prompt_only(benchmark_path)
    input_ids = tokenizer(
        prompt["context"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][:, : int(runtime["max_input_tokens"])].contiguous()
    query_ids = tokenizer(
        prompt["question"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][:, : int(config["query"]["max_tokens"])].contiguous()
    chunk_size = int(cache_config["compression_chunk_tokens"])
    if int(input_ids.shape[-1]) != 2 * chunk_size:
        raise RuntimeError("memory smoke requires exactly two full chunks")
    if int(query_ids.shape[-1]) <= 0:
        raise RuntimeError("query tokenization is empty")
    prompt_sha = hashlib.sha256(
        input_ids.numpy().tobytes() + b"\x00QUERY\x00" + query_ids.numpy().tobytes()
    ).hexdigest()
    policy = SignificancePolicy(
        budget=int(cache_config["retained_history_budget"]),
        sinks=int(cache_config["sink_tokens"]),
        recent=int(cache_config["recent_tokens"]),
        count_exponent=float(config["significance"]["exponent"]),
        require_heavy=True,
    )

    def new_memory(slots: int) -> SignificanceMemory:
        return SignificanceMemory(
            num_layers=int(model.config.num_hidden_layers),
            batch_size=int(input_ids.shape[0]),
            num_kv_heads=int(model.config.num_key_value_heads),
            num_slots=slots,
            head_dim=int(model.config.head_dim),
            variant=str(memory_config["aggregation"]),
            device="cpu",
            kv_dtype=dtype,
            recency_tau=float(memory_config["recency_tau"]),
            random_seed=int(memory_config["random_seed"]),
        )

    hard_started = time.perf_counter()
    hard_cache = SignificanceCache(config=model.config)
    hard = chunked_prefill(
        model,
        input_ids,
        hard_cache,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=query_ids,
        verify_query_rollback=True,
        logits_to_keep=1,
    )
    hard_seconds = time.perf_counter() - hard_started
    rss_samples.append(process.memory_info().rss)

    zero_started = time.perf_counter()
    zero_cache = SignificanceCache(config=model.config)
    zero_memory = new_memory(0)
    zero = chunked_prefill_with_memory(
        model,
        input_ids,
        zero_cache,
        zero_memory,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=query_ids,
        verify_query_rollback=True,
        logits_to_keep=1,
    )
    zero_seconds = time.perf_counter() - zero_started
    rss_samples.append(process.memory_info().rss)
    hard_logits = hard.output.logits.float()
    zero_logits = zero.output.logits.float()
    logit_delta = (hard_logits - zero_logits).abs()
    cache_identity = _cache_identity(hard_cache, zero_cache)
    zero_identity = {
        "exact_logits": bool(torch.equal(hard_logits, zero_logits)),
        "max_abs_logit_delta": float(logit_delta.max().item()),
        "mean_abs_logit_delta": float(logit_delta.mean().item()),
        "exact_cache_state": bool(cache_identity["exact"]),
        "cache_mismatches": cache_identity["mismatches"],
        "hard_compression_reports": [asdict(report) for report in hard.compression_reports],
        "m0_compression_reports": [asdict(report) for report in zero.compression_reports],
        "m0_memory_write_boundaries": len(zero.memory_write_reports),
        "m0_memory_bytes": zero_memory.byte_ledger().total_bytes,
        "hard_wall_seconds": hard_seconds,
        "m0_wall_seconds": zero_seconds,
    }
    zero_pass = (
        zero_identity["exact_logits"]
        and zero_identity["exact_cache_state"]
        and zero_identity["m0_memory_write_boundaries"] == 0
        and zero_identity["m0_memory_bytes"] == 0
    )

    memory_started = time.perf_counter()
    memory_cache = SignificanceCache(config=model.config)
    memory = new_memory(int(memory_config["count"]))
    memory_result = chunked_prefill_with_memory(
        model,
        input_ids,
        memory_cache,
        memory,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=query_ids,
        verify_query_rollback=True,
        logits_to_keep=1,
    )
    memory_seconds = time.perf_counter() - memory_started
    rss_samples.append(process.memory_info().rss)
    memory_logits = memory_result.output.logits.float()
    memory.assert_valid()
    memory_cache.assert_aligned()
    ledger = memory.byte_ledger()
    expected_capacity = (
        int(model.config.num_hidden_layers)
        * int(input_ids.shape[0])
        * int(model.config.num_key_value_heads)
        * int(memory_config["count"])
    )
    expected_stored = policy.budget + chunk_size
    memory_write_count = sum(len(boundary) for boundary in memory_result.memory_write_reports)
    memory_runtime = {
        "variant": memory.variant,
        "slot_count_per_layer_kv_head": memory.num_slots,
        "finite_logits": bool(torch.isfinite(memory_logits).all().item()),
        "true_length": memory_cache.true_length,
        "ordinary_stored_length": memory_cache.stored_length(),
        "expected_ordinary_stored_length": expected_stored,
        "compression_reports": [asdict(report) for report in memory_result.compression_reports],
        "memory_write_reports": [
            [asdict(report) for report in boundary]
            for boundary in memory_result.memory_write_reports
        ],
        "memory_layer_writes": memory_write_count,
        "memory_state_sha256": memory.state_sha256(),
        "byte_ledger": asdict(ledger),
        "attention_accounting": asdict(memory_result.attention_accounting),
        "scoring_events_per_layer": memory_cache.scoring_events(),
        "wall_seconds": memory_seconds,
    }
    memory_pass = (
        memory_runtime["finite_logits"]
        and memory_cache.true_length == int(input_ids.shape[-1])
        and memory_cache.stored_length() == expected_stored
        and len(memory_result.compression_reports) == 1
        and memory_result.compression_reports[0].evictions > 0
        and memory_write_count == int(model.config.num_hidden_layers)
        and ledger.slot_capacity == expected_capacity
        and ledger.valid_slots == expected_capacity
        and memory_result.attention_accounting.calls_with_valid_slots
        == int(model.config.num_hidden_layers)
    )
    status = "PASS" if zero_pass and memory_pass else "FAIL"

    slot_diagnostics = _slot_diagnostics(memory)
    atomic_write_json(output / "slot_diagnostics.json", slot_diagnostics)
    diagnostics = {
        "schema_version": "1.0",
        "status": status,
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "oracle_access": {
            "answer_key_loaded": False,
            "answer_generation_performed": False,
            "memory_write_inputs": [
                "post_rope_keys",
                "values",
                "scores_S",
                "counts_C",
                "absolute_positions",
                "semantic_entry_ids",
                "causal_boundary_position",
            ],
            "evidence_annotations_entered_memory_writes": False,
        },
        "m0_identity": zero_identity,
        "m4_runtime": memory_runtime,
    }
    atomic_write_json(output / "runtime_diagnostics.json", diagnostics)

    run_id = output.name
    base_case = {
        "schema_version": "1.0",
        "run_id": run_id,
        "task_family": "R0",
        "generator_seed": seed,
        "prompt_sha256": prompt_sha,
        "context_tokens": int(input_ids.shape[-1]),
        "chunk_size": chunk_size,
        "prediction": None,
        "correct": None,
        "full_cache_correct": None,
        "compression_conditioned_eligible": None,
        "evidence": None,
        "trace_ids": [],
        "error": None,
    }
    case_rows = [
        {
            **base_case,
            "case_id": f"{prompt['id']}__M0",
            "condition": "SIGNIFICANCE_MEMORY_M0_IDENTITY",
            "status": "OK" if zero_pass else "INVALID",
            "efficiency": {
                "kv_only_bytes": _cache_kv_bytes(zero_cache),
                "total_persistent_bytes": _cache_kv_bytes(zero_cache)
                + zero_cache.matrix_bytes()
                + zero_cache.metadata_bytes(),
                "wall_seconds": zero_seconds,
                "peak_rss_bytes": max(rss_samples),
                "peak_vram_bytes": 0,
            },
            "failure_labels": [] if zero_pass else ["IMPLEMENTATION_ERROR"],
        },
        {
            **base_case,
            "case_id": f"{prompt['id']}__M4",
            "condition": "SIGNIFICANCE_MEMORY_M4_NORMALIZED_RUNTIME",
            "status": "OK" if memory_pass else "INVALID",
            "efficiency": {
                "kv_only_bytes": _cache_kv_bytes(memory_cache),
                "memory_slot_bytes": ledger.total_bytes,
                "total_persistent_bytes": _cache_kv_bytes(memory_cache)
                + memory_cache.matrix_bytes()
                + memory_cache.metadata_bytes()
                + ledger.total_bytes,
                "wall_seconds": memory_seconds,
                "peak_rss_bytes": max(rss_samples),
                "peak_vram_bytes": 0,
            },
            "failure_labels": [] if memory_pass else ["IMPLEMENTATION_ERROR"],
        },
    ]
    (output / "cases.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in case_rows),
        encoding="utf-8",
    )

    summary = {
        "schema_version": "1.0",
        "status": status,
        "evidence_class": "RUNTIME_COMPATIBILITY_ONLY",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "checkpoint_loaded": True,
        "model_inference_performed": True,
        "answer_generation_performed": False,
        "quality_measured": False,
        "cases": 2,
        "m0_identity_pass": bool(zero_pass),
        "m4_finite_runtime_pass": bool(memory_pass),
        "memory_slots_per_layer_kv_head": memory.num_slots,
        "memory_total_persistent_bytes": ledger.total_bytes,
        "claims_supported": config["claims"]["supports"],
        "claims_not_supported": config["claims"]["does_not_support"],
    }
    atomic_write_json(output / "summary.json", summary)
    resolved = output / "resolved_config.yaml"
    resolved.write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )

    manifest = build_manifest(
        repo=REPO_ROOT,
        config=config,
        run_id=run_id,
        execution_mode="CLOSED_LOOP_MODEL_EXECUTION",
        model={
            "status": "VERIFIED",
            "id": model_expected["core_id"],
            "path": _repo_path(model_path),
            "architecture": model_expected["expected_architecture"],
            "hashes": hashes,
        },
        benchmark={"name": "one truncated prompt-only Rineya Long Context Demo row"},
        architecture_variant="Experiment D fixed significance memory slots",
        seed=seed,
        compute_tier="B",
        status="COMPLETE" if status == "PASS" else "INVALID_OR_CONFOUNDED",
        evidence_class=(
            "NEW_CHATBUBBLE_EVIDENCE" if status == "PASS" else "INVALID_OR_CONFOUNDED"
        ),
    )
    manifest["created_at"] = started_at
    manifest["completed_at"] = utc_now()
    manifest["config"] = {
        "resolved_path": "resolved_config.yaml",
        "sha256": sha256_file(resolved),
        "source_path": _repo_path(config_path),
    }
    manifest["runtime"].update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "precision": str(dtype).removeprefix("torch."),
            "peak_sampled_rss_bytes": max(rss_samples),
        }
    )
    manifest["budget"] = {
        "ordinary_resident_entries": memory_cache.stored_length(),
        "memory_slots_per_layer_kv_head": memory.num_slots,
        "memory_slot_capacity": ledger.slot_capacity,
        "kv_only_bytes": _cache_kv_bytes(memory_cache),
        "significance_matrix_bytes": memory_cache.matrix_bytes(),
        "alignment_metadata_bytes": memory_cache.metadata_bytes(),
        "memory_slot_bytes": ledger.total_bytes,
        "total_persistent_bytes": _cache_kv_bytes(memory_cache)
        + memory_cache.matrix_bytes()
        + memory_cache.metadata_bytes()
        + ledger.total_bytes,
        "matched": False,
    }
    manifest["artifacts"] = {
        "summary": "summary.json",
        "cases": "cases.jsonl",
        "diagnostics": "runtime_diagnostics.json",
        "slot_diagnostics": "slot_diagnostics.json",
        "resolved_config": "resolved_config.yaml",
    }
    manifest["notes"] = [
        *(f"NOT SUPPORTED: {claim}" for claim in config["claims"]["does_not_support"]),
        "Memory K aggregates are already post-RoPE and are not assigned a synthetic RoPE coordinate.",
        "M=0 delegates to the existing rectangular hard-eviction implementation.",
        "RSS is sampled at phase boundaries rather than continuously measured.",
    ]
    atomic_write_json(output / "manifest.json", manifest)
    elapsed = time.perf_counter() - started
    print(json.dumps({"output": str(output), "status": status, "wall_seconds": elapsed}, indent=2))
    return {"summary": summary, "manifest": manifest, "diagnostics": diagnostics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_significance_memory_smoke.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    outcome = run(args.config, args.output)
    return 0 if outcome["summary"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
