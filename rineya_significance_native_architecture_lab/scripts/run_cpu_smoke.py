#!/usr/bin/env python3
"""Run the bounded, real-R2 CPU fidelity smoke test.

This is intentionally not a benchmark.  It proves that the disabled chunked
path preserves full-cache logits and that one query-conditioned Cornerstone-1
cache can score, roll back, compress, and retain aligned state on the local R2
checkpoint.  It does not decode an answer or make a quality claim.
"""

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
    SignificanceCache,
    SignificancePolicy,
    chunked_prefill,
)


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _resolve_path(value: str | None, *, base: Path) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _dtype(name: str):
    import torch

    values = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported smoke dtype {name!r}; expected float32 or bfloat16") from exc


def _repo_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _cache_bytes(cache: SignificanceCache) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return int(total)


def _read_prompt(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = {"answer", "answers", "supporting_facts", "evidence"}.intersection(row)
            if forbidden:
                raise RuntimeError(f"runtime prompt row exposes oracle fields: {sorted(forbidden)}")
            if not isinstance(row.get("context"), str) or not isinstance(row.get("question"), str):
                raise RuntimeError("prompt-only row must contain string context and question")
            return {"id": str(row.get("id", "cpu-smoke-0")), "context": row["context"], "question": row["question"]}
    raise RuntimeError(f"no prompt rows found in {path}")


def _check_geometry(config, expected: dict[str, Any]) -> None:
    checks = {
        "architectures": expected.get("expected_architecture"),
        "num_hidden_layers": expected.get("expected_layers"),
        "hidden_size": expected.get("expected_hidden_size"),
        "num_attention_heads": expected.get("expected_query_heads"),
        "num_key_value_heads": expected.get("expected_kv_heads"),
        "head_dim": expected.get("expected_head_dim"),
        "vocab_size": expected.get("expected_vocab_size"),
    }
    for name, wanted in checks.items():
        if wanted is None:
            continue
        got = getattr(config, name, None)
        if name == "architectures":
            if wanted not in (got or []):
                raise RuntimeError(f"model architecture mismatch: expected {wanted}, got {got}")
        elif int(got) != int(wanted):
            raise RuntimeError(f"model geometry mismatch for {name}: expected {wanted}, got {got}")


def run(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = Path(config_path).expanduser().resolve()
    config = _load_document(config_path)
    paths = dict(config.get("paths", {}))
    runtime = dict(config.get("runtime", {}))
    cache_config = dict(config.get("cache", {}))
    significance = dict(config.get("significance", {}))
    model_expectation = dict(config.get("model", {}))
    query_config = dict(config.get("query", {}))

    base_name = str(paths.get("base", "package_root"))
    if base_name == "package_root":
        declared_base = LAB_ROOT
    elif base_name == "repository_root":
        declared_base = REPO_ROOT
    else:
        raise RuntimeError("paths.base must be 'package_root' or 'repository_root'")
    model_path = _resolve_path(paths.get("model"), base=declared_base)
    tokenizer_path = _resolve_path(paths.get("tokenizer") or paths.get("model"), base=declared_base)
    benchmark_path = _resolve_path(paths.get("benchmark"), base=declared_base)
    if model_path is None or tokenizer_path is None or not model_path.is_dir():
        raise RuntimeError("CPU smoke requires --model-path or paths.model naming a local checkpoint")
    if benchmark_path is None or not benchmark_path.is_file():
        raise RuntimeError(f"prompt-only smoke input is missing: {benchmark_path}")
    output = output_root or Path(str(paths.get("output_root", "results/runs/cpu_smoke")))
    if not output.is_absolute():
        output = declared_base / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty CPU smoke directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started = time.perf_counter()
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]
    dtype = _dtype(str(runtime.get("dtype", "float32")))
    seed = int(runtime.get("seed", 17))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    weight_path = model_path / "model.safetensors"
    tokenizer_json = tokenizer_path / "tokenizer.json"
    config_json = model_path / "config.json"
    hashes = {
        "weights": sha256_file(weight_path),
        "tokenizer": sha256_file(tokenizer_json),
        "config": sha256_file(config_json),
        "chat_template": None,
    }
    expected_hashes = dict(model_expectation.get("expected_hashes", {}))
    hash_mismatches = {
        name: {"expected": str(expected).lower(), "observed": str(hashes.get(name)).lower()}
        for name, expected in expected_hashes.items()
        if expected is not None and str(hashes.get(name)).lower() != str(expected).lower()
    }
    if hash_mismatches:
        raise RuntimeError(f"checkpoint hash mismatch: {hash_mismatches}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    _check_geometry(model.config, model_expectation)
    rss_samples.append(process.memory_info().rss)

    prompt = _read_prompt(benchmark_path)
    max_input = int(runtime.get("max_input_tokens", 192))
    encoded = tokenizer(prompt["context"], add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = encoded[:, :max_input].contiguous()
    if input_ids.shape[-1] < 3:
        raise RuntimeError("CPU smoke context tokenized to fewer than three tokens")
    query_ids = tokenizer(prompt["question"], add_special_tokens=False, return_tensors="pt")["input_ids"]
    query_limit = int(query_config.get("max_tokens", 32))
    query_ids = query_ids[:, : min(query_limit, query_ids.shape[-1])].contiguous()
    context_token_sha = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()
    query_token_sha = hashlib.sha256(query_ids.numpy().tobytes()).hexdigest()
    combined_token_sha = hashlib.sha256(
        input_ids.numpy().tobytes() + b"\x00QUERY\x00" + query_ids.numpy().tobytes()
    ).hexdigest()
    chunk_size = int(cache_config.get("compression_chunk_tokens", 64))

    identity_started = time.perf_counter()
    with torch.inference_mode():
        reference = model(input_ids, use_cache=False, logits_to_keep=0).logits.float()
    identity_cache = SignificanceCache(config=model.config)
    identity = chunked_prefill(
        model,
        input_ids,
        identity_cache,
        chunk_size=chunk_size,
        policy=None,
        collect_logits=True,
        logits_to_keep=0,
    )
    if identity.logits is None or identity.logits.shape != reference.shape:
        raise RuntimeError("disabled chunked path returned the wrong logit shape")
    delta = (identity.logits - reference).abs()
    max_delta = float(delta.max().item())
    mean_delta = float(delta.mean().item())
    argmax_agreement = float((identity.logits.argmax(-1) == reference.argmax(-1)).float().mean().item())
    tolerance = float(runtime.get("identity_tolerance", 1e-3))
    identity_pass = max_delta < tolerance and argmax_agreement == 1.0
    identity_seconds = time.perf_counter() - identity_started
    rss_samples.append(process.memory_info().rss)
    del identity, identity_cache, reference, delta
    gc.collect()

    pooling = dict(significance.get("score_pooling", {}))
    policy = SignificancePolicy(
        budget=int(cache_config.get("retained_history_budget", cache_config.get("total_budget_slots", 48))),
        sinks=int(cache_config.get("sink_tokens", 4)),
        recent=int(cache_config.get("recent_tokens", 16)),
        count_exponent=float(significance.get("exponent", 1.0)),
        pool_width=int(pooling.get("width", 1)) if pooling.get("enabled", False) else 1,
        require_heavy=True,
    )
    compressed_started = time.perf_counter()
    sig_cache = SignificanceCache(config=model.config)
    compressed = chunked_prefill(
        model,
        input_ids,
        sig_cache,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=query_ids,
        observation_window=int(query_ids.shape[-1]),
        verify_query_rollback=True,
        collect_logits=False,
        logits_to_keep=1,
    )
    sig_cache.assert_aligned()
    compressed_seconds = time.perf_counter() - compressed_started
    rss_samples.append(process.memory_info().rss)
    scoring_events = sig_cache.scoring_events()
    effective_evictions = sum(report.evictions for report in compressed.compression_reports)
    max_resident = policy.budget + chunk_size
    runtime_pass = (
        sig_cache.true_length == int(input_ids.shape[-1])
        and sig_cache.stored_length() <= max_resident
        and effective_evictions > 0
        and len(set(scoring_events)) == 1
        and scoring_events[0] == max(0, (input_ids.shape[-1] + chunk_size - 1) // chunk_size - 1)
    )
    final_token_id = int(compressed.output.logits[0, -1].argmax().item())
    diagnostics = {
        "identity": {
            "status": "PASS" if identity_pass else "FAIL",
            "max_abs_logit_delta": max_delta,
            "mean_abs_logit_delta": mean_delta,
            "argmax_agreement": argmax_agreement,
            "tolerance": tolerance,
            "wall_seconds": identity_seconds,
        },
        "compressed_runtime": {
            "status": "PASS" if runtime_pass else "FAIL",
            "true_length": sig_cache.true_length,
            "stored_length": sig_cache.stored_length(),
            "budget": policy.budget,
            "chunk_size": chunk_size,
            "compression_reports": [report.__dict__ for report in compressed.compression_reports],
            "effective_evictions": effective_evictions,
            "scoring_events_per_layer": scoring_events,
            "query_tokens": int(query_ids.shape[-1]),
            "context_token_sha256": context_token_sha,
            "query_token_sha256": query_token_sha,
            "query_probe_boundaries": max(0, (input_ids.shape[-1] + chunk_size - 1) // chunk_size - 1),
            "kv_bytes": _cache_bytes(sig_cache),
            "significance_matrix_bytes": sig_cache.matrix_bytes(),
            "alignment_metadata_bytes": sig_cache.metadata_bytes(),
            "wall_seconds": compressed_seconds,
            "final_context_logit_argmax_token_id": final_token_id,
        },
    }
    atomic_write_json(output / "runtime_diagnostics.json", diagnostics)

    run_id = output.name
    case = {
        "schema_version": "1.0",
        "run_id": run_id,
        "case_id": prompt["id"],
        "scenario_id": None,
        "condition": "B3_runtime_fidelity_smoke",
        "task_family": "R0",
        "generator_seed": seed,
        "prompt_sha256": combined_token_sha,
        "context_token_sha256": context_token_sha,
        "query_token_sha256": query_token_sha,
        "query_tokens": int(query_ids.shape[-1]),
        "context_tokens": int(input_ids.shape[-1]),
        "evidence_depth": None,
        "chunk_size": chunk_size,
        "status": "OK" if identity_pass and runtime_pass else "INVALID",
        "prediction": None,
        "correct": None,
        "full_cache_correct": None,
        "compression_conditioned_eligible": None,
        "evidence": None,
        "efficiency": {
            "kv_only_bytes": _cache_bytes(sig_cache),
            "total_persistent_bytes": _cache_bytes(sig_cache) + sig_cache.matrix_bytes() + sig_cache.metadata_bytes(),
            "wall_seconds": compressed_seconds,
            "peak_rss_bytes": max(rss_samples),
            "peak_vram_bytes": 0,
        },
        "failure_labels": [] if identity_pass and runtime_pass else ["IMPLEMENTATION_ERROR"],
        "trace_ids": [],
        "error": None,
    }
    (output / "cases.jsonl").write_text(json.dumps(case, sort_keys=True) + "\n", encoding="utf-8")

    overall_pass = identity_pass and runtime_pass
    summary = {
        "schema_version": "1.0",
        "status": "PASS" if overall_pass else "FAIL",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "checkpoint_loaded": True,
        "model_inference_performed": True,
        "answer_generation_performed": False,
        "cases": 1,
        "identity": diagnostics["identity"],
        "compressed_runtime": diagnostics["compressed_runtime"],
        "claims_supported": [
            "the disabled chunked path preserves this published-core checkpoint's logits within the declared dtype-specific tolerance",
            "query-conditioned S/C state survives exact rollback and aligned compression on one bounded CPU smoke case",
        ],
        "claims_not_supported": [
            "Rineya published-core answer accuracy or long-context reasoning improves",
            "the smoke latency represents GPU or deployment latency",
            "attention priors, memory slots, adaptive budgets, or controllers improve quality",
        ],
    }
    atomic_write_json(output / "summary.json", summary)
    resolved_path = output / "resolved_config.yaml"
    resolved_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo = git_snapshot(REPO_ROOT)
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "COMPLETE" if overall_pass else "INVALID_OR_CONFOUNDED",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE" if overall_pass else "INVALID_OR_CONFOUNDED",
        "created_at": started_at,
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": {"remote": repo["remote"], "commit": repo["commit"], "dirty": repo["dirty"]},
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
            "name": str(config.get("experiment", {}).get("name", "published_core_cornerstone1_cpu_smoke")),
            "compute_tier": "B",
            "architecture_variant": "B0 identity gate plus B3 normalized query-conditioned cache",
            "benchmark": "one truncated prompt-only Rineya Long Context Demo row",
            "seed": seed,
            "query_conditioned": True,
            "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        },
        "config": {
            "resolved_path": "resolved_config.yaml",
            "sha256": sha256_file(resolved_path),
            "source_path": _repo_path(config_path),
        },
        "model": {
            "status": "VERIFIED" if expected_hashes else "HASH_RECORDED",
            "id": str(model_expectation.get("core_id") or model_path.name),
            "path": _repo_path(model_path),
            "architecture": "Qwen3ForCausalLM",
            "hashes": hashes,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "precision": str(dtype).removeprefix("torch."),
            "hardware": {"platform": platform.platform(), "logical_cpu_count": psutil.cpu_count()},
        },
        "budget": {
            "resident_entries": sig_cache.stored_length(),
            "kv_only_bytes": _cache_bytes(sig_cache),
            "total_persistent_bytes": _cache_bytes(sig_cache) + sig_cache.matrix_bytes() + sig_cache.metadata_bytes(),
            "matched": False,
        },
        "artifacts": {"summary": "summary.json", "cases": "cases.jsonl", "diagnostics": "runtime_diagnostics.json", "resolved_config": "resolved_config.yaml"},
        "notes": summary["claims_not_supported"] + ["RSS is the maximum of sampled checkpoints, not a continuous peak monitor."],
    }
    atomic_write_json(output / "manifest.json", manifest)
    result = {"output": str(output), "manifest": manifest, "summary": summary, "wall_seconds": time.perf_counter() - started}
    print(json.dumps({"output": str(output), "status": summary["status"], "wall_seconds": result["wall_seconds"]}, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "cpu_smoke.yaml")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = _load_document(args.config.resolve())
    if args.model_path is not None:
        config.setdefault("paths", {})["model"] = str(args.model_path.resolve())
        config["paths"]["tokenizer"] = str(args.model_path.resolve())
        with __import__("tempfile").TemporaryDirectory(prefix="rineya-cpu-smoke-") as temporary:
            resolved = Path(temporary) / "config.json"
            resolved.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            outcome = run(resolved, args.output)
    else:
        outcome = run(args.config, args.output)
    return 0 if outcome["summary"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
