#!/usr/bin/env python3
"""Run one honest Halo closed-loop heterogeneous-budget prototype smoke."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    git_snapshot,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.runtime import (  # noqa: E402
    HeterogeneousSignificanceCache,
    SignificanceCache,
    SignificancePolicy,
    chunked_prefill,
    heterogeneous_chunked_prefill,
    heterogeneous_decode_append,
)


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _resolve(value: str, *, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _first_prompt(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("context"), str) or not isinstance(row.get("question"), str):
                raise RuntimeError("prompt row requires string context and question")
            return {
                "id": str(row.get("id", "halo-heterogeneous-smoke-0")),
                "context": row["context"],
                "question": row["question"],
            }
    raise RuntimeError("benchmark has no prompt row")


def _dtype(name: str):
    import torch

    values = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _uniform_cache_identity(rectangular: SignificanceCache, padded: HeterogeneousSignificanceCache) -> dict[str, Any]:
    mismatches: list[str] = []
    for layer_idx, (left, right) in enumerate(zip(rectangular.layers, padded.layers, strict=True)):
        if right.validity is None or not bool(right.validity.all().item()):
            mismatches.append(f"layer_{layer_idx}:validity")
        for name in ("keys", "values"):
            if not __import__("torch").equal(getattr(left, name), getattr(right, name)):
                mismatches.append(f"layer_{layer_idx}:{name}")
        for name in ("scores", "counts", "positions", "entry_ids"):
            if not __import__("torch").equal(getattr(left.state, name), getattr(right.state, name)):
                mismatches.append(f"layer_{layer_idx}:{name}")
    return {"exact": not mismatches, "mismatches": mismatches}


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.resolve()
    config = _load(config_path)
    paths = dict(config["paths"])
    runtime = dict(config["runtime"])
    cache_config = dict(config["cache"])
    allocation_config = dict(config["allocation"])
    model_config = dict(config["model"])
    model_path = _resolve(str(paths["model"]), base=REPO_ROOT)
    tokenizer_path = _resolve(str(paths["tokenizer"]), base=REPO_ROOT)
    benchmark_path = _resolve(str(paths["benchmark"]), base=REPO_ROOT)
    output = (
        output_override.expanduser().resolve()
        if output_override is not None
        else _resolve(str(paths["output_root"]), base=REPO_ROOT)
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    hashes = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(tokenizer_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    expected_hashes = dict(model_config["expected_hashes"])
    mismatches = {
        name: {"expected": expected, "observed": hashes[name]}
        for name, expected in expected_hashes.items()
        if str(expected).lower() != hashes[name].lower()
    }
    if mismatches:
        raise RuntimeError(f"checkpoint hash mismatch: {mismatches}")

    started_at = utc_now()
    started = time.perf_counter()
    process = psutil.Process()
    rss = [process.memory_info().rss]
    seed = int(runtime["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dtype = _dtype(str(runtime["dtype"]))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    rss.append(process.memory_info().rss)
    if int(model.config.num_hidden_layers) != int(model_config["expected_layers"]):
        raise RuntimeError("unexpected Halo layer count")
    if int(model.config.num_key_value_heads) != int(model_config["expected_kv_heads"]):
        raise RuntimeError("unexpected Halo KV-head count")

    prompt = _first_prompt(benchmark_path)
    input_ids = tokenizer(
        prompt["context"], add_special_tokens=False, return_tensors="pt"
    )["input_ids"][:, : int(runtime["max_input_tokens"])].contiguous()
    query_ids = tokenizer(
        prompt["question"], add_special_tokens=False, return_tensors="pt"
    )["input_ids"][:, : int(runtime["query_tokens"])].contiguous()
    chunk_size = int(runtime["chunk_tokens"])
    if int(input_ids.shape[-1]) != 2 * chunk_size:
        raise RuntimeError("smoke requires exactly two equal chunks")
    if int(query_ids.shape[-1]) <= 0:
        raise RuntimeError("query tokenization was empty")
    token_hash = hashlib.sha256(
        input_ids.numpy().tobytes() + b"\x00QUERY\x00" + query_ids.numpy().tobytes()
    ).hexdigest()

    average_budget = int(cache_config["average_budget_per_layer_head"])
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_key_value_heads)
    global_budget = int(cache_config["fixed_global_entry_budget"])
    if global_budget != layers * heads * average_budget:
        raise RuntimeError("declared global budget does not equal layers * KV heads * average")
    policy = SignificancePolicy(
        budget=average_budget,
        sinks=int(cache_config["sink_tokens"]),
        recent=int(cache_config["recent_tokens"]),
        count_exponent=float(cache_config["count_exponent"]),
        require_heavy=True,
    )

    uniform_started = time.perf_counter()
    rectangular = SignificanceCache(config=model.config)
    rectangular_result = chunked_prefill(
        model,
        input_ids,
        rectangular,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=query_ids,
        observation_window=int(runtime["observation_window"]),
        verify_query_rollback=True,
        logits_to_keep=1,
    )
    padded_uniform = HeterogeneousSignificanceCache(config=model.config)
    padded_uniform_result = heterogeneous_chunked_prefill(
        model,
        input_ids,
        padded_uniform,
        chunk_size=chunk_size,
        policy=policy,
        global_budget=global_budget,
        allocation_strategy="uniform",
        minimum_per_unit=average_budget,
        maximum_per_unit=average_budget,
        query_ids=query_ids,
        observation_window=int(runtime["observation_window"]),
        verify_query_rollback=True,
        logits_to_keep=1,
    )
    uniform_cache = _uniform_cache_identity(rectangular, padded_uniform)
    rectangular_logits = rectangular_result.output.logits.float()
    padded_logits = padded_uniform_result.output.logits.float()
    logit_delta = (rectangular_logits - padded_logits).abs()
    uniform_identity = {
        "exact_logits": bool(torch.equal(rectangular_logits, padded_logits)),
        "max_abs_logit_delta": float(logit_delta.max().item()),
        "exact_cache_state": bool(uniform_cache["exact"]),
        "cache_mismatches": uniform_cache["mismatches"],
        "allocations": padded_uniform_result.compression_reports[0].allocations,
        "wall_seconds": time.perf_counter() - uniform_started,
    }
    rss.append(process.memory_info().rss)
    del rectangular_result, padded_uniform_result, rectangular, padded_uniform, logit_delta
    gc.collect()

    adaptive_started = time.perf_counter()
    adaptive_cache = HeterogeneousSignificanceCache(config=model.config)
    adaptive_result = heterogeneous_chunked_prefill(
        model,
        input_ids,
        adaptive_cache,
        chunk_size=chunk_size,
        policy=policy,
        global_budget=global_budget,
        allocation_strategy=str(allocation_config["strategy"]),
        minimum_per_unit=int(cache_config["minimum_per_layer_head"]),
        maximum_per_unit=int(cache_config["maximum_per_layer_head"]),
        allocation_exponent=float(cache_config["count_exponent"]),
        allocation_seed=int(allocation_config["seed"]),
        query_ids=query_ids,
        observation_window=int(runtime["observation_window"]),
        verify_query_rollback=True,
        logits_to_keep=1,
    )
    adaptive_cache.assert_aligned()
    decision = adaptive_result.allocation_decisions[0]
    if decision is None:
        raise RuntimeError("adaptive smoke did not return a live allocation decision")
    allocations = decision.budget.allocations
    report = adaptive_result.compression_reports[0]
    allocation_values = sorted({int(value) for value in allocations.ravel()})
    heterogeneous = len(allocation_values) > 1
    before_decode_true = adaptive_cache.true_length
    before_decode_logical = adaptive_cache.logical_entries_per_sequence()
    next_id = adaptive_result.output.logits[:, -1].argmax(dim=-1, keepdim=True)
    decoded = heterogeneous_decode_append(
        model,
        next_id,
        adaptive_cache,
        recording_window=1,
        logits_to_keep=1,
    )
    ledger = adaptive_cache.byte_accounting()
    adaptive_runtime = {
        "allocation_strategy": decision.budget.strategy,
        "allocation_values": allocation_values,
        "allocation_min": int(allocations.min()),
        "allocation_max": int(allocations.max()),
        "allocation_sum": int(allocations.sum()),
        "observed_heterogeneity": heterogeneous,
        "compression_report": report.__dict__,
        "layer_physical_lengths_after_decode": adaptive_cache.layer_physical_lengths(),
        "logical_entries_before_decode": before_decode_logical,
        "logical_entries_after_decode": adaptive_cache.logical_entries_per_sequence(),
        "true_length_before_decode": before_decode_true,
        "true_length_after_decode": adaptive_cache.true_length,
        "decode_logits_finite": bool(torch.isfinite(decoded.logits).all().item()),
        "scoring_events_per_layer": adaptive_cache.scoring_events(),
        "byte_accounting": ledger.__dict__,
        "wall_seconds": time.perf_counter() - adaptive_started,
    }
    rss.append(process.memory_info().rss)

    uniform_pass = (
        uniform_identity["exact_logits"] and uniform_identity["exact_cache_state"]
    )
    adaptive_pass = (
        report.after_logical_entries == global_budget
        and int(allocations.sum()) == global_budget
        and heterogeneous
        and before_decode_true == int(input_ids.shape[-1])
        and adaptive_cache.true_length == int(input_ids.shape[-1]) + 1
        and adaptive_cache.logical_entries_per_sequence() == before_decode_logical + layers * heads
        and adaptive_runtime["decode_logits_finite"]
    )
    status = "PASS" if uniform_pass and adaptive_pass else "FAIL"
    diagnostics = {
        "uniform_identity": uniform_identity,
        "adaptive_runtime": adaptive_runtime,
        "status": status,
    }
    atomic_write_json(output / "runtime_diagnostics.json", diagnostics)
    summary = {
        "schema_version": "1.0",
        "status": status,
        "evidence_class": "RUNTIME_COMPATIBILITY_ONLY",
        "cases": 1,
        "uniform_identity_pass": uniform_pass,
        "heterogeneous_runtime_pass": adaptive_pass,
        "checkpoint_loaded": True,
        "model_inference_performed": True,
        "answer_generation_performed": False,
        "quality_measured": False,
        "physical_padding_disclosed": True,
        "claims_supported": config["claims"]["supports"],
        "claims_not_supported": config["claims"]["does_not_support"],
    }
    atomic_write_json(output / "summary.json", summary)
    case = {
        "schema_version": "1.0",
        "run_id": output.name,
        "case_id": prompt["id"],
        "condition": "HALO_HETEROGENEOUS_BUDGET_RUNTIME_SMOKE",
        "task_family": "R0",
        "generator_seed": seed,
        "prompt_sha256": token_hash,
        "status": "OK" if status == "PASS" else "INVALID",
        "context_tokens": int(input_ids.shape[-1]),
        "chunk_size": chunk_size,
        "query_tokens": int(query_ids.shape[-1]),
        "prediction": None,
        "correct": None,
        "failure_labels": [] if status == "PASS" else ["IMPLEMENTATION_ERROR"],
    }
    (output / "cases.jsonl").write_text(
        json.dumps(case, sort_keys=True) + "\n", encoding="utf-8"
    )
    resolved = output / "resolved_config.yaml"
    resolved.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repository = git_snapshot(REPO_ROOT)
    # ``git_snapshot`` includes a diagnostic checkout path. Run manifests are
    # portable artifacts, so never persist that host-specific absolute path.
    repository.pop("repo_root", None)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE" if status == "PASS" else "INVALID_OR_CONFOUNDED",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE" if status == "PASS" else "INVALID_OR_CONFOUNDED",
        "created_at": started_at,
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": repository,
            "source": {
                "remote": "https://github.com/LUNCY579/RineyaBeta.git",
                "implementation": {
                    "ref": "refs/heads/main",
                    "commit": "9e00be74098878290dc2b0d54cbffee4652c6088",
                    "path": "scripts/kv/significance.py",
                    "blob": "30ee29724dce52de5a288f6421123fb718240541",
                },
                "paper": {
                    "ref": "refs/heads/r2-healing-harness",
                    "commit": "1664f54de0bbdfde45e311b75bbf3038dbfffe42",
                    "path": "paper/significance_matrix/main.tex",
                    "blob": "b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3",
                },
            },
        },
        "experiment": {
            "name": config["experiment"]["name"],
            "architecture_variant": "Experiment B closed-loop heterogeneous logical budget",
            "compute_tier": "B",
            "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
            "seed": seed,
        },
        "config": {
            "resolved_path": "resolved_config.yaml",
            "sha256": sha256_file(resolved),
            "source_path": config_path.relative_to(REPO_ROOT).as_posix(),
        },
        "model": {
            "status": "VERIFIED",
            "id": model_config["core_id"],
            "path": model_path.relative_to(REPO_ROOT).as_posix(),
            "architecture": "Qwen3ForCausalLM",
            "hashes": hashes,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "precision": str(dtype).removeprefix("torch."),
            "peak_sampled_rss_bytes": max(rss),
        },
        "budget": {
            "logical_global_entries_at_boundary": global_budget,
            "matched": True,
            "physical_padded_storage": True,
            "post_decode_byte_accounting": ledger.__dict__,
        },
        "artifacts": {
            "summary": "summary.json",
            "cases": "cases.jsonl",
            "diagnostics": "runtime_diagnostics.json",
            "resolved_config": "resolved_config.yaml",
        },
        "notes": [
            f"NOT SUPPORTED: {claim}" for claim in summary["claims_not_supported"]
        ],
    }
    atomic_write_json(output / "manifest.json", manifest)
    result = {
        "output": str(output),
        "status": status,
        "wall_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_heterogeneous_budget_smoke.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args.config, args.output)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
