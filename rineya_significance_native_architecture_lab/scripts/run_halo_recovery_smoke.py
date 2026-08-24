#!/usr/bin/env python3
"""Run a two-case live Halo smoke for Architecture-E accept/recover paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.recovery import (  # noqa: E402
    assert_pre_answer_feature_rows,
    extract_cornerstone_risk_features,
)
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    build_manifest,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.recovery import (  # noqa: E402
    RecoveryAction,
    RecoverySignal,
    ThresholdRecoveryController,
    execute_recovery_decision,
)
from rineya_significance_lab.runtime import (  # noqa: E402
    CountControlSpec,
    RuntimeCountController,
    SignificanceCache,
    SignificancePolicy,
    chunked_prefill,
)

_FORBIDDEN_PROMPT_FIELDS = frozenset(
    {
        "accepted_answers",
        "answer",
        "correct",
        "evidence",
        "gold",
        "label",
        "prediction",
        "reference_answer",
        "target",
    }
)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery config must be a mapping")
    return value


def _resolve_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    raw = dict(config["paths"])
    base = REPO_ROOT if raw.get("base") == "repository_root" else LAB_ROOT
    return {
        name: (base / str(value)).resolve()
        for name, value in raw.items()
        if name != "base"
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _token_hash(ids: Any) -> str:
    value = ids.detach().to("cpu", dtype=__import__("torch").int64).contiguous()
    canonical = json.dumps(
        tuple(int(token) for token in value.flatten().tolist()), separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_kv_bytes(cache: SignificanceCache) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if tensor is not None:
                total += int(tensor.numel() * tensor.element_size())
    return total


def _decode(
    model: Any,
    tokenizer: Any,
    cache: SignificanceCache,
    first_logits: Any,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [int(eos)]))
    logits = first_logits
    generated: list[int] = []
    decode_appends = 0
    stop_reason = "max_new_tokens"
    started = time.perf_counter()
    for index in range(max_new_tokens):
        token = int(logits[0, -1].float().argmax().item())
        generated.append(token)
        if token in eos_ids:
            stop_reason = "eos_token"
            break
        if index == max_new_tokens - 1:
            break
        step = chunked_prefill(
            model,
            torch.tensor([[token]], dtype=torch.long),
            cache,
            chunk_size=1,
            policy=None,
            collect_logits=False,
            logits_to_keep=1,
        )
        logits = step.output.logits
        decode_appends += 1
    return {
        "prediction": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "token_ids": generated,
        "generated_tokens": len(generated),
        "decode_cache_appends": decode_appends,
        "stop_reason": stop_reason,
        "decode_wall_seconds": time.perf_counter() - started,
    }


def _assert_contiguous_positions(cache: SignificanceCache, expected_length: int) -> None:
    import torch

    expected = torch.arange(expected_length, dtype=torch.long).view(1, 1, -1)
    for layer_index, layer in enumerate(cache.layers):
        if layer.state is None:
            raise RuntimeError(f"fresh full-cache layer {layer_index} has no position state")
        observed = layer.state.positions.detach().to(device="cpu", dtype=torch.long)
        if observed.shape[-1] != expected_length:
            raise RuntimeError(
                f"fresh full-cache layer {layer_index} length {observed.shape[-1]} "
                f"!= {expected_length}"
            )
        if not torch.equal(observed, expected.expand_as(observed)):
            raise RuntimeError(f"position leakage/non-contiguity in full-cache layer {layer_index}")


def _live_full_cache_recompute(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    *,
    chunk_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Recompute from immutable input tokens in a new cache starting at position zero."""

    cache = SignificanceCache(config=model.config)
    if cache.true_length != 0 or cache.stored_length() != 0:
        raise RuntimeError("new recovery cache did not start empty at absolute position zero")
    started = time.perf_counter()
    prefill_started = time.perf_counter()
    prefill = chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=chunk_size,
        policy=None,
        collect_logits=False,
        logits_to_keep=1,
    )
    prefill_wall = time.perf_counter() - prefill_started
    input_tokens = int(input_ids.shape[-1])
    if cache.true_length != input_tokens or cache.stored_length() != input_tokens:
        raise RuntimeError("full-cache recomputation did not restore every source token")
    _assert_contiguous_positions(cache, input_tokens)
    context_kv_bytes = _cache_kv_bytes(cache)
    decoded = _decode(
        model,
        tokenizer,
        cache,
        prefill.output.logits,
        max_new_tokens=max_new_tokens,
    )
    expected_final = input_tokens + int(decoded["decode_cache_appends"])
    if cache.true_length != expected_final or cache.stored_length() != expected_final:
        raise RuntimeError("full-cache decode position accounting drifted")
    _assert_contiguous_positions(cache, expected_final)
    return {
        **decoded,
        "path": RecoveryAction.RECOMPUTE_FULL_CACHE.value,
        "fresh_cache_zero_state_verified": True,
        "compressed_cache_reused": False,
        "source_context_fully_restored": True,
        "recompute_input_position_start": 0,
        "recompute_input_position_end_inclusive": input_tokens - 1,
        "recompute_positions_contiguous": True,
        "resident_context_entries_after_prefill": input_tokens,
        "resident_entries_after_decode": expected_final,
        "context_kv_bytes_after_prefill": context_kv_bytes,
        "kv_bytes_after_decode": _cache_kv_bytes(cache),
        "prefill_wall_seconds": prefill_wall,
        "recompute_wall_seconds": time.perf_counter() - started,
    }


def _features_from_live_boundaries(
    config: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray[Any, Any]],
    summaries: Sequence[Any],
    case_id: str,
    cluster_id: str,
    family: str,
):
    if len(summaries) < 2:
        raise RuntimeError("live recovery signal requires at least two compression boundaries")
    final = summaries[-1]
    risk = dict(config["risk_signal"])
    return extract_cornerstone_risk_features(
        arrays,
        case_id=case_id,
        cluster_id=cluster_id,
        task_family=family,
        boundary_index=int(final.boundary_index),
        layer_count=int(final.layer_count),
        top_mass_fraction=float(risk["top_mass_fraction"]),
        weights={name: float(value) for name, value in dict(risk["weights"]).items()},
    )


def _run_one(
    config: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    prompt: Mapping[str, Any],
    saved_feature: Mapping[str, Any],
    controller: ThresholdRecoveryController,
    source_prediction: str,
    full_cache_prediction: str,
) -> dict[str, Any]:
    import psutil

    exposed = _FORBIDDEN_PROMPT_FIELDS.intersection(prompt)
    if exposed:
        raise RuntimeError(f"prompt-only recovery input exposes oracle fields: {sorted(exposed)}")
    case_id = str(prompt["case_id"])
    cluster_id = str(prompt["cluster_id"])
    family = str(prompt["family"])
    input_ids = tokenizer(
        str(prompt["prompt"]), add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    query_ids = tokenizer(
        str(prompt["question"]), add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    if _token_hash(input_ids) != str(prompt["token_ids_sha256"]):
        raise RuntimeError(f"{case_id}: tokenizer no longer reproduces frozen prompt tokens")
    if int(input_ids.shape[-1]) != int(prompt["token_count"]):
        raise RuntimeError(f"{case_id}: prompt token count drift")

    cache_config = dict(config["cache"])
    runtime = dict(config["runtime"])
    count_controller = RuntimeCountController(
        CountControlSpec("true_realized_count", "true", int(runtime["seed"])),
        case_id=case_id,
        chunk_size=int(cache_config["compression_chunk_tokens"]),
    )
    policy = SignificancePolicy(
        budget=int(cache_config["compressed_retained_history_budget"]),
        sinks=int(cache_config["sink_tokens"]),
        recent=int(cache_config["recent_tokens"]),
        count_exponent=float(cache_config["significance_exponent"]),
        pool_width=int(cache_config["score_pooling_width"]),
        require_heavy=True,
    )
    compressed_cache = SignificanceCache(config=model.config)
    process = psutil.Process()
    rss_samples = [int(process.memory_info().rss)]
    compressed_started = time.perf_counter()
    prefill = chunked_prefill(
        model,
        input_ids,
        compressed_cache,
        chunk_size=int(cache_config["compression_chunk_tokens"]),
        policy=policy,
        query_ids=query_ids,
        observation_window=int(query_ids.shape[-1]),
        verify_query_rollback=bool(runtime["verify_query_rollback"]),
        collect_logits=False,
        logits_to_keep=1,
        compression_callback=count_controller.compress,
    )
    compressed_prefill_wall = time.perf_counter() - compressed_started
    compressed_cache.assert_aligned()
    rss_samples.append(int(process.memory_info().rss))
    live_features = _features_from_live_boundaries(
        config,
        arrays=count_controller.diagnostic_arrays(),
        summaries=count_controller.summaries,
        case_id=case_id,
        cluster_id=cluster_id,
        family=family,
    )
    saved_risk = float(saved_feature["risk_score"])
    if not math.isclose(live_features.risk_score, saved_risk, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(
            f"{case_id}: live pre-answer risk {live_features.risk_score} "
            f"!= saved trace risk {saved_risk}"
        )
    signal = RecoverySignal(
        case_id=case_id,
        cluster_id=cluster_id,
        risk_score=live_features.risk_score,
        components=live_features.risk_components,
    )
    # This timestamp/decision occurs before _decode or _live_full_cache_recompute.
    decision_wall_offset = time.perf_counter() - compressed_started
    decision = controller.decide(signal)
    compressed_prefill_bytes = (
        _cache_kv_bytes(compressed_cache)
        + compressed_cache.matrix_bytes()
        + compressed_cache.metadata_bytes()
    )

    if decision.action is RecoveryAction.ACCEPT:
        decoded = _decode(
            model,
            tokenizer,
            compressed_cache,
            prefill.output.logits,
            max_new_tokens=int(runtime["max_new_tokens"]),
        )
        accepted_payload = {
            **decoded,
            "path": RecoveryAction.ACCEPT.value,
            "recompute_wall_seconds": 0.0,
            "context_kv_bytes_after_prefill": 0,
            "kv_bytes_after_decode": _cache_kv_bytes(compressed_cache),
        }

        def forbidden_recompute(_: RecoveryAction) -> dict[str, Any]:
            raise RuntimeError("accept path invoked recomputation")

        execution = execute_recovery_decision(
            decision, accepted_payload, recompute=forbidden_recompute
        )
        literal_accept_identity = execution.output is accepted_payload
        reference_prediction = source_prediction
    else:
        unused_accept_payload = {"path": "unused_compressed_decode"}

        def recompute(action: RecoveryAction) -> dict[str, Any]:
            if action is not RecoveryAction.RECOMPUTE_FULL_CACHE:
                raise RuntimeError(f"unsupported live recovery action: {action}")
            return _live_full_cache_recompute(
                model,
                tokenizer,
                input_ids,
                chunk_size=int(cache_config["compression_chunk_tokens"]),
                max_new_tokens=int(runtime["max_new_tokens"]),
            )

        execution = execute_recovery_decision(
            decision, unused_accept_payload, recompute=recompute
        )
        literal_accept_identity = False
        reference_prediction = full_cache_prediction
    rss_samples.append(int(process.memory_info().rss))
    payload = execution.output
    prediction = str(payload["prediction"])
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "cluster_id": cluster_id,
        "task_family": family,
        "decision_phase": "AFTER_COMPRESSED_PREFILL_BEFORE_ANY_ANSWER_TOKEN_DECODE",
        "risk_score": live_features.risk_score,
        "risk_components": live_features.risk_components,
        "saved_trace_risk_score": saved_risk,
        "live_matches_saved_preanswer_risk": True,
        "controller_threshold": controller.threshold,
        "controller_id": controller.controller_id,
        "action": decision.action.value,
        "recomputation_calls": execution.recomputation_calls,
        "literal_accept_payload_identity": literal_accept_identity,
        "prediction": prediction,
        "prediction_sha256": _text_hash(prediction),
        "matches_prior_same_path_prediction": prediction == reference_prediction,
        "prior_same_path_prediction_sha256": _text_hash(reference_prediction),
        "generated_tokens": payload.get("generated_tokens"),
        "stop_reason": payload.get("stop_reason"),
        "compressed_prefill_wall_seconds": compressed_prefill_wall,
        "controller_decision_wall_offset_seconds": decision_wall_offset,
        "recompute_wall_seconds": float(payload.get("recompute_wall_seconds", 0.0)),
        "compressed_prefill_persistent_bytes": compressed_prefill_bytes,
        "recompute_context_kv_bytes": int(payload.get("context_kv_bytes_after_prefill", 0)),
        "path_kv_bytes_after_decode": int(payload.get("kv_bytes_after_decode", 0)),
        "recompute_input_token_bytes": (
            int(input_ids.shape[-1]) * int(dict(config["accounting"])["input_token_dtype_bytes"])
            if execution.recomputation_calls
            else 0
        ),
        "sampled_peak_rss_bytes": max(rss_samples),
        "fresh_cache_zero_state_verified": payload.get("fresh_cache_zero_state_verified"),
        "compressed_cache_reused": payload.get("compressed_cache_reused"),
        "source_context_fully_restored": payload.get("source_context_fully_restored"),
        "recompute_input_position_start": payload.get("recompute_input_position_start"),
        "recompute_input_position_end_inclusive": payload.get(
            "recompute_input_position_end_inclusive"
        ),
        "recompute_positions_contiguous": payload.get("recompute_positions_contiguous"),
        "resident_context_entries_after_recompute_prefill": payload.get(
            "resident_context_entries_after_prefill"
        ),
        "resident_entries_after_recompute_decode": payload.get("resident_entries_after_decode"),
        "answer_key_accessed": False,
        "quality_scored": False,
        "claim_scope": "MECHANISM_IDENTITY_AND_RECOVERY_ACTION_ONLY",
    }


def run(config_path: Path) -> dict[str, Any]:
    import torch
    import transformers
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.resolve()
    config = _load_config(config_path)
    # Reuse the screen's strict validation without importing it as a package.
    import importlib.util

    screen_path = LAB_ROOT / "scripts" / "run_halo_recovery_screen.py"
    spec = importlib.util.spec_from_file_location("halo_recovery_screen_validation", screen_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load recovery-screen validation")
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    screen.validate_config(config)

    paths = _resolve_paths(config)
    output = paths["smoke_output"]
    if output.exists():
        if (output / "manifest.json").exists():
            raise RuntimeError(f"refusing to overwrite completed recovery smoke: {output}")
        allowed_partial = {"resolved_config.yaml", "paths.jsonl", "summary.json"}
        unexpected = {path.name for path in output.iterdir()}.difference(allowed_partial)
        if unexpected:
            raise RuntimeError(f"unexpected files in partial recovery smoke: {sorted(unexpected)}")
    else:
        output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    features = _jsonl(paths["screen_output"] / "features.pre_answer.jsonl")
    assert_pre_answer_feature_rows(features)
    ordered = sorted(features, key=lambda row: (float(row["risk_score"]), str(row["case_id"])))
    accept_feature = ordered[0]
    recover_feature = ordered[-1]
    low = float(accept_feature["risk_score"])
    high = float(recover_feature["risk_score"])
    if not low < high:
        raise RuntimeError("saved pre-answer risks cannot produce both smoke actions")
    threshold = (low + high) / 2.0
    controller = ThresholdRecoveryController(threshold=threshold)
    if controller.decide(
        RecoverySignal(
            str(accept_feature["case_id"]),
            str(accept_feature["cluster_id"]),
            low,
            dict(accept_feature["risk_components"]),
        )
    ).action is not RecoveryAction.ACCEPT:
        raise RuntimeError("smoke threshold did not produce the required accept path")
    if controller.decide(
        RecoverySignal(
            str(recover_feature["case_id"]),
            str(recover_feature["cluster_id"]),
            high,
            dict(recover_feature["risk_components"]),
        )
    ).action is not RecoveryAction.RECOMPUTE_FULL_CACHE:
        raise RuntimeError("smoke threshold did not produce the required recovery path")

    selected_ids = {str(accept_feature["case_id"]), str(recover_feature["case_id"])}
    prompts = {
        str(row["case_id"]): row
        for row in _jsonl(paths["prompt_only_inference"])
        if str(row["case_id"]) in selected_ids
    }
    if set(prompts) != selected_ids:
        raise RuntimeError("prompt-only input is missing a selected smoke case")
    source_predictions = {
        str(row["case_id"]): str(row["prediction"])
        for row in _jsonl(paths["source_run"] / "predictions.unscored.jsonl")
        if str(row.get("condition")) == "true_realized_count"
        and str(row["case_id"]) in selected_ids
    }
    full_predictions = {
        str(row["case_id"]): str(row["prediction"])
        for row in _jsonl(paths["full_cache_run"] / "predictions.unscored.jsonl")
        if str(row["case_id"]) in selected_ids
    }
    if set(source_predictions) != selected_ids or set(full_predictions) != selected_ids:
        raise RuntimeError("prior unscored path references are incomplete")

    model_config = dict(config["model"])
    model_path = paths["model"]
    expected_hashes = dict(model_config["expected_hashes"])
    observed_hashes = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(model_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"Halo checkpoint hash mismatch: {observed_hashes}")
    torch.manual_seed(int(dict(config["runtime"])["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    if int(model.config.num_hidden_layers) != int(model_config["expected_layers"]):
        raise RuntimeError("Halo layer geometry drift")
    if int(model.config.num_key_value_heads) != int(model_config["expected_kv_heads"]):
        raise RuntimeError("Halo KV-head geometry drift")

    started_at = utc_now()
    rows = []
    for feature in (accept_feature, recover_feature):
        case_id = str(feature["case_id"])
        rows.append(
            _run_one(
                config,
                model=model,
                tokenizer=tokenizer,
                prompt=prompts[case_id],
                saved_feature=feature,
                controller=controller,
                source_prediction=source_predictions[case_id],
                full_cache_prediction=full_predictions[case_id],
            )
        )
    del model
    _write_jsonl(output / "paths.jsonl", rows)

    actions = {str(row["action"]): row for row in rows}
    accept = actions[RecoveryAction.ACCEPT.value]
    recovered = actions[RecoveryAction.RECOMPUTE_FULL_CACHE.value]
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "CLOSED_LOOP_RECOVERY_MECHANISM_SMOKE",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "architecture_experiment": "E_SIGNIFICANCE_CONTROLLED_RECOMPUTATION",
        "cases": 2,
        "controller_threshold": threshold,
        "threshold_selection": (
            "midpoint of saved minimum/maximum pre-answer risks; outcomes were not consulted"
        ),
        "decision_before_answer_decode": all(
            row["decision_phase"]
            == "AFTER_COMPRESSED_PREFILL_BEFORE_ANY_ANSWER_TOKEN_DECODE"
            for row in rows
        ),
        "accept_path": {
            "case_id": accept["case_id"],
            "recomputation_calls": accept["recomputation_calls"],
            "literal_payload_identity": accept["literal_accept_payload_identity"],
            "matches_prior_compressed_prediction": accept[
                "matches_prior_same_path_prediction"
            ],
        },
        "recovery_path": {
            "case_id": recovered["case_id"],
            "recomputation_calls": recovered["recomputation_calls"],
            "fresh_cache_zero_state_verified": recovered[
                "fresh_cache_zero_state_verified"
            ],
            "compressed_cache_reused": recovered["compressed_cache_reused"],
            "source_context_fully_restored": recovered["source_context_fully_restored"],
            "positions_contiguous_from_zero": recovered["recompute_positions_contiguous"],
            "matches_prior_full_cache_prediction": recovered[
                "matches_prior_same_path_prediction"
            ],
            "recompute_wall_seconds": recovered["recompute_wall_seconds"],
            "recompute_context_kv_bytes": recovered["recompute_context_kv_bytes"],
            "recompute_input_token_bytes": recovered["recompute_input_token_bytes"],
        },
        "answer_key_accessed": False,
        "quality_scored": False,
        "claim": (
            "This proves path identity and one fresh full-cache recovery action only; two "
            "risk-selected cases cannot establish controller quality or latency benefit."
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    manifest = build_manifest(
        repo=REPO_ROOT,
        config=config,
        run_id=output.name,
        execution_mode="CLOSED_LOOP_MODEL_EXECUTION",
        model={
            "status": "VERIFIED",
            "id": model_config["core_id"],
            "path": str(config["paths"]["model"]),
            "architecture": model_config["expected_architecture"],
            "hashes": observed_hashes,
        },
        benchmark={"name": "two risk-extreme B0-selected Halo clusters"},
        architecture_variant="E live deterministic accept/full-cache recovery gate",
        seed=int(dict(config["runtime"])["seed"]),
        compute_tier="B",
        status="COMPLETE",
        resolved_config_path="configs/halo_recovery_controller.yaml",
    )
    manifest["created_at"] = started_at
    manifest["completed_at"] = utc_now()
    manifest["input_provenance"] = {
        "screen_manifest_sha256": sha256_file(paths["screen_output"] / "manifest.json"),
        "pre_answer_features_sha256": sha256_file(
            paths["screen_output"] / "features.pre_answer.jsonl"
        ),
        "source_unscored_predictions_sha256": sha256_file(
            paths["source_run"] / "predictions.unscored.jsonl"
        ),
        "full_cache_unscored_predictions_sha256": sha256_file(
            paths["full_cache_run"] / "predictions.unscored.jsonl"
        ),
        "prompt_only_inference_sha256": sha256_file(paths["prompt_only_inference"]),
    }
    manifest["artifacts"] = {
        "resolved_config": "resolved_config.yaml",
        "paths": "paths.jsonl",
        "summary": "summary.json",
    }
    manifest["runtime"]["torch"] = torch.__version__
    manifest["runtime"]["transformers"] = transformers.__version__
    manifest["notes"] = [
        "Both decisions consumed live on-policy Cornerstone state before answer decoding.",
        "The accept branch returned the same payload object and made zero recovery calls.",
        "The recovery branch discarded compressed state and recomputed all source tokens in a fresh cache from absolute position zero.",
        "No answer/evidence file was opened and no quality score was computed.",
        "This is a two-case mechanism smoke, not controller validation or a latency benchmark.",
    ]
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_recovery_controller.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
