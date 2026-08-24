#!/usr/bin/env python3
"""Run the bounded Halo Experiment-C CPU compatibility smoke."""

from __future__ import annotations

import argparse
import copy
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

from rineya_significance_lab.config import load_config  # noqa: E402
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    git_snapshot,
    sha256_file,
    utc_now,
)


def _resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_prompt(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context = row.get("context") or row.get("prompt") or row.get("text")
            if isinstance(context, str) and context.strip():
                return str(row.get("id", "halo-attention-prior-smoke")), context
    raise RuntimeError(f"no prompt text found in {path}")


def _dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def _check_geometry(model_config, expected: dict[str, Any]) -> None:
    checks = {
        "num_hidden_layers": "expected_layers",
        "hidden_size": "expected_hidden_size",
        "num_attention_heads": "expected_query_heads",
        "num_key_value_heads": "expected_kv_heads",
        "head_dim": "expected_head_dim",
        "vocab_size": "expected_vocab_size",
    }
    for attribute, expected_key in checks.items():
        wanted = expected.get(expected_key)
        if wanted is not None and int(getattr(model_config, attribute)) != int(wanted):
            raise RuntimeError(
                f"Halo geometry mismatch for {attribute}: "
                f"expected {wanted}, got {getattr(model_config, attribute)}"
            )
    architecture = str(expected.get("expected_architecture", ""))
    if architecture and architecture not in (getattr(model_config, "architectures", None) or []):
        raise RuntimeError(
            f"Halo architecture mismatch: expected {architecture}, got {model_config.architectures}"
        )


def _cache_bytes(cache) -> int:
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return int(total)


def _cache_digest(cache, *, include_significance: bool) -> str:
    import torch

    digest = hashlib.sha256()
    for layer_idx, layer in enumerate(cache.layers):
        if not layer.is_initialized or layer.state is None:
            raise RuntimeError(f"layer {layer_idx} is missing completed cache state")
        tensors = [
            ("keys", layer.keys),
            ("values", layer.values),
            ("positions", layer.state.positions),
            ("entry_ids", layer.state.entry_ids),
        ]
        if include_significance:
            tensors.extend(
                [
                    ("scores", layer.state.scores),
                    ("counts", layer.state.counts),
                ]
            )
        for name, tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{layer_idx}:{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _run_condition(
    model,
    input_ids,
    *,
    condition_id: str,
    lambda_value: float | None,
    prior_config: dict[str, Any],
    cache_config: dict[str, Any],
    runtime_config: dict[str, Any],
    process,
    record_significance: bool = True,
) -> tuple[object, dict[str, Any]]:
    import torch

    from rineya_significance_lab.runtime import (
        AttentionPriorAccessDeclaration,
        AttentionPriorTrace,
        SignificanceAttentionPrior,
        SignificanceCache,
        SignificancePolicy,
        chunked_prefill,
    )

    access_config = dict(prior_config["access"])
    prior = None
    trace = None
    if lambda_value is not None:
        prior = SignificanceAttentionPrior(
            mode=str(prior_config["mode"]),
            global_lambda=float(lambda_value),
            layer_lambdas=None,
            clip=float(prior_config["clip"]),
            epsilon=float(prior_config["epsilon"]),
            max_abs_lambda=float(prior_config["max_abs_lambda"]),
            shuffle_seed=int(prior_config["shuffle_seed"]),
            access=AttentionPriorAccessDeclaration(query_access=str(access_config["query"])),
        )
        trace = AttentionPriorTrace()
    cache = SignificanceCache(config=model.config)
    policy = (
        SignificancePolicy(
            budget=int(cache_config["retained_history_budget"]),
            sinks=int(cache_config["sink_tokens"]),
            recent=int(cache_config["recent_tokens"]),
            count_exponent=1.0,
        )
        if record_significance
        else None
    )
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter()
    output = chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=int(runtime_config["chunk_size"]),
        policy=policy,
        query_ids=None,
        observation_window=int(runtime_config["observation_window"]),
        collect_logits=True,
        logits_to_keep=0,
        attention_prior=prior,
        prior_trace=trace,
    )
    wall_seconds = time.perf_counter() - started
    cache.assert_aligned()
    if output.logits is None:
        raise RuntimeError(f"{condition_id} returned no logits")
    logits = output.logits.detach().cpu()
    finite = bool(torch.isfinite(logits).all().item())
    trace_payload = trace.to_dict() if trace is not None else None
    applications = trace.applications if trace is not None else []
    max_abs_bias = max((row.bias_abs_max for row in applications), default=0.0)
    max_current_key_bias = max(
        (row.current_key_bias_abs_max for row in applications), default=0.0
    )
    max_invalid_key_bias = max(
        (row.invalid_key_bias_abs_max for row in applications), default=0.0
    )
    nonzero_applications = sum(row.nonzero_bias_applied for row in applications)
    zero_bypasses = sum(row.zero_lambda_bypass for row in applications)
    evictions = sum(report.evictions for report in output.compression_reports)
    kv_bytes = _cache_bytes(cache)
    matrix_bytes = cache.matrix_bytes()
    metadata_bytes = cache.metadata_bytes()
    diagnostics = {
        "condition": condition_id,
        "lambda": lambda_value,
        "finite_logits": finite,
        "wall_seconds": wall_seconds,
        "true_length": cache.true_length,
        "stored_length": cache.stored_length(),
        "scoring_events_per_layer": cache.scoring_events(),
        "compression_reports": [report.__dict__ for report in output.compression_reports],
        "evictions": evictions,
        "kv_structure_sha256": _cache_digest(cache, include_significance=False),
        "full_state_sha256": _cache_digest(cache, include_significance=True),
        "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        "logit_shape": list(logits.shape),
        "final_argmax_token_id": int(logits[0, -1].argmax().item()),
        "max_abs_bias": max_abs_bias,
        "max_current_key_abs_bias": max_current_key_bias,
        "max_invalid_key_abs_bias": max_invalid_key_bias,
        "nonzero_bias_applications": nonzero_applications,
        "zero_lambda_bypasses": zero_bypasses,
        "attention_prior_trace": trace_payload,
        "kv_only_bytes": kv_bytes,
        "significance_matrix_bytes": matrix_bytes,
        "alignment_metadata_bytes": metadata_bytes,
        "total_persistent_bytes": kv_bytes + matrix_bytes + metadata_bytes,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": int(process.memory_info().rss),
    }
    del cache, output
    gc.collect()
    return logits, diagnostics


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    paths = dict(config["paths"])
    runtime_config = dict(config["runtime"])
    model_config = dict(config["model"])
    cache_config = dict(config["cache"])
    prior_config = dict(config["attention_prior"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    model_path = _resolve_path(paths["model"], base=base)
    tokenizer_path = _resolve_path(paths["tokenizer"], base=base)
    benchmark_path = _resolve_path(paths["benchmark"], base=base)
    output = output_override or _resolve_path(paths["output_root"], base=base)
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    total_started = time.perf_counter()
    seed = int(runtime_config["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dtype = _dtype(str(runtime_config["dtype"]))
    process = psutil.Process()

    weight_path = model_path / "model.safetensors"
    tokenizer_json = tokenizer_path / "tokenizer.json"
    config_json = model_path / "config.json"
    hashes: dict[str, str | None] = {
        "weights": sha256_file(weight_path),
        "tokenizer": sha256_file(tokenizer_json),
        "config": sha256_file(config_json),
        "chat_template": None,
    }
    mismatches = {
        name: {"expected": expected, "observed": hashes.get(name)}
        for name, expected in dict(model_config["expected_hashes"]).items()
        if str(expected).lower() != str(hashes.get(name)).lower()
    }
    if mismatches:
        raise RuntimeError(f"Halo checkpoint hash mismatch: {mismatches}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    _check_geometry(model.config, model_config)
    prompt_id, prompt = _read_prompt(benchmark_path)
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = input_ids[:, : int(runtime_config["max_input_tokens"])].contiguous()
    chunk_size = int(runtime_config["chunk_size"])
    if input_ids.shape[-1] < chunk_size * 2:
        raise RuntimeError("attention-prior smoke needs at least two complete chunks")
    context_token_sha256 = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()

    baseline_logits, baseline = _run_condition(
        model,
        input_ids,
        condition_id="baseline_no_prior",
        lambda_value=None,
        prior_config=prior_config,
        cache_config=cache_config,
        runtime_config=runtime_config,
        process=process,
        record_significance=False,
    )
    zero_logits, zero = _run_condition(
        model,
        input_ids,
        condition_id="C0_zero_bias",
        lambda_value=0.0,
        prior_config=prior_config,
        cache_config=cache_config,
        runtime_config=runtime_config,
        process=process,
    )
    nonzero_lambda = float(prior_config["lambda"])
    nonzero_logits, nonzero = _run_condition(
        model,
        input_ids,
        condition_id="C1_tiny_global_lambda",
        lambda_value=nonzero_lambda,
        prior_config=prior_config,
        cache_config=cache_config,
        runtime_config=runtime_config,
        process=process,
    )

    identity_delta = (zero_logits - baseline_logits).abs()
    zero_identity = {
        "bit_exact_logits": bool(torch.equal(zero_logits, baseline_logits)),
        "bit_exact_kv_structure": zero["kv_structure_sha256"] == baseline["kv_structure_sha256"],
        "max_abs_logit_delta": float(identity_delta.max().item()),
        "mean_abs_logit_delta": float(identity_delta.mean().item()),
        "argmax_agreement": float(
            (zero_logits.argmax(-1) == baseline_logits.argmax(-1)).float().mean().item()
        ),
    }
    zero_pass = (
        zero_identity["bit_exact_logits"]
        and zero_identity["bit_exact_kv_structure"]
        and zero["finite_logits"]
        and zero["zero_lambda_bypasses"] > 0
        and zero["nonzero_bias_applications"] == 0
        and zero["evictions"] == 0
    )
    nonzero_delta = (nonzero_logits - baseline_logits).abs()
    nonzero_bound = abs(nonzero_lambda) * float(prior_config["clip"])
    nonzero_pass = (
        nonzero["finite_logits"]
        and nonzero["nonzero_bias_applications"] > 0
        and float(nonzero["max_abs_bias"]) <= nonzero_bound + 1.0e-6
        and float(nonzero["max_current_key_abs_bias"]) == 0.0
        and float(nonzero["max_invalid_key_abs_bias"]) == 0.0
        and nonzero["evictions"] == 0
        and nonzero["true_length"] == int(input_ids.shape[-1])
    )
    diagnostics = {
        "scope": "runtime_and_finite_output_compatibility_only",
        "answer_generation_performed": False,
        "quality_scoring_performed": False,
        "access_declaration": prior_config["access"],
        "baseline": baseline,
        "C0_zero_bias": zero,
        "C0_identity": zero_identity,
        "C1_tiny_global_lambda": nonzero,
        "C1_vs_baseline_descriptive_only": {
            "max_abs_logit_delta": float(nonzero_delta.max().item()),
            "mean_abs_logit_delta": float(nonzero_delta.mean().item()),
            "argmax_agreement": float(
                (nonzero_logits.argmax(-1) == baseline_logits.argmax(-1)).float().mean().item()
            ),
            "not_a_quality_metric": True,
        },
    }
    atomic_write_json(output / "attention_prior_diagnostics.json", diagnostics)

    run_id = output.name
    cases = []
    for condition, passed, condition_diagnostics in (
        ("C0_zero_bias", zero_pass, zero),
        ("C1_tiny_global_lambda", nonzero_pass, nonzero),
    ):
        cases.append(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "case_id": f"{prompt_id}__{condition}",
                "scenario_id": prompt_id,
                "condition": condition,
                "task_family": "R0",
                "generator_seed": seed,
                "prompt_sha256": context_token_sha256,
                "context_tokens": int(input_ids.shape[-1]),
                "evidence_depth": None,
                "chunk_size": chunk_size,
                "status": "OK" if passed else "INVALID",
                "prediction": None,
                "correct": None,
                "full_cache_correct": None,
                "compression_conditioned_eligible": None,
                "evidence": None,
                "efficiency": {
                    "kv_only_bytes": condition_diagnostics["kv_only_bytes"],
                    "total_persistent_bytes": condition_diagnostics["total_persistent_bytes"],
                    "wall_seconds": condition_diagnostics["wall_seconds"],
                    "peak_rss_bytes": max(
                        condition_diagnostics["rss_before_bytes"],
                        condition_diagnostics["rss_after_bytes"],
                    ),
                    "peak_vram_bytes": 0,
                },
                "failure_labels": [] if passed else ["IMPLEMENTATION_ERROR"],
                "trace_ids": [],
                "error": None,
            }
        )
    (output / "cases.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )

    overall_pass = bool(zero_pass and nonzero_pass)
    summary = {
        "schema_version": "1.0",
        "status": "PASS" if overall_pass else "FAIL",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE" if overall_pass else "INVALID_OR_CONFOUNDED",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "architecture_experiment": "C_SOFT_SIGNIFICANCE_AWARE_ATTENTION",
        "model": "Rineya Halo published Qwen3-0.6B core",
        "cases": 2,
        "quality_evaluation": False,
        "answer_generation": False,
        "C0_zero_bias": {"status": "PASS" if zero_pass else "FAIL", **zero_identity},
        "C1_tiny_global_lambda": {
            "status": "PASS" if nonzero_pass else "FAIL",
            "lambda": nonzero_lambda,
            "clip": float(prior_config["clip"]),
            "max_abs_bias_observed": nonzero["max_abs_bias"],
            "max_current_key_abs_bias": nonzero["max_current_key_abs_bias"],
            "max_invalid_key_abs_bias": nonzero["max_invalid_key_abs_bias"],
            "nonzero_bias_applications": nonzero["nonzero_bias_applications"],
            "finite_logits": nonzero["finite_logits"],
        },
        "claims_supported": [
            "lambda=0 logits and KV structure are bit-exact to the schedule-matched plain unrecorded chunked B0 path",
            "one lambda=0.01 normalized historical-significance prior completed with finite Halo logits",
            "the prior stayed within its declared clipped logit bound and did not bias current-forward keys",
        ],
        "claims_not_supported": list(config["claims"]["prohibited"]),
    }
    atomic_write_json(output / "summary.json", summary)
    portable_config = copy.deepcopy(config)
    portable_config.get("_meta", {}).pop("source_config", None)
    resolved_path = output / "resolved_config.yaml"
    resolved_path.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repo = git_snapshot(REPO_ROOT)
    completed_at = utc_now()
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "COMPLETE" if overall_pass else "INVALID_OR_CONFOUNDED",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE" if overall_pass else "INVALID_OR_CONFOUNDED",
        "created_at": started_at,
        "completed_at": completed_at,
        "repositories": {
            "chatbubble": {
                "remote": repo["remote"],
                "commit": repo["commit"],
                "dirty": repo["dirty"],
            },
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
            "name": str(config["experiment"]["name"]),
            "compute_tier": "B",
            "architecture_variant": "C0 zero prior and C1 clipped normalized S/C prior",
            "benchmark": "one 24-token prompt-only runtime smoke; no answer decode",
            "seed": seed,
            "query_conditioned": False,
            "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        },
        "config": {
            "resolved_path": "resolved_config.yaml",
            "sha256": sha256_file(resolved_path),
            "source_path": _repo_path(config_path),
        },
        "model": {
            "status": "VERIFIED",
            "id": str(model_config["core_id"]),
            "path": _repo_path(model_path),
            "architecture": str(model_config["expected_architecture"]),
            "hashes": hashes,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "precision": str(dtype).removeprefix("torch."),
            "hardware": {
                "platform": platform.platform(),
                "logical_cpu_count": psutil.cpu_count(),
            },
        },
        "budget": {
            "resident_entries": int(nonzero["stored_length"]),
            "kv_only_bytes": int(nonzero["kv_only_bytes"]),
            "total_persistent_bytes": int(nonzero["total_persistent_bytes"]),
            "matched": True,
        },
        "attention_prior": {
            "mode": "normalized",
            "transform": "clipped_z_score",
            "lambda": nonzero_lambda,
            "clip": float(prior_config["clip"]),
            "access": prior_config["access"],
            "only_previously_observed_cached_keys": True,
        },
        "artifacts": {
            "summary": "summary.json",
            "cases": "cases.jsonl",
            "diagnostics": "attention_prior_diagnostics.json",
            "resolved_config": "resolved_config.yaml",
        },
        "notes": [f"NOT SUPPORTED: {claim}" for claim in summary["claims_not_supported"]]
        + [
            "Compatibility smoke only; no answer was generated or scored.",
            "The baseline, C0, and C1 conditions used identical full-cache storage and zero evictions.",
        ],
    }
    atomic_write_json(output / "manifest.json", manifest)
    result = {
        "output": str(output),
        "status": summary["status"],
        "wall_seconds": time.perf_counter() - total_started,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_attention_prior_smoke.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    outcome = run(args.config, args.output)
    return 0 if outcome["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
