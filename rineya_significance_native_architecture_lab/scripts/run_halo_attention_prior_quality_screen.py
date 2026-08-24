#!/usr/bin/env python3
"""Run the restartable 60-cell exploratory Halo attention-prior quality screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import (  # noqa: E402
    FrozenEvidenceAnnotation,
    score_posthoc_evidence_retention,
)
from rineya_significance_lab.evaluation import (  # noqa: E402
    ATTENTION_PRIOR_QUALITY_CONDITION_IDS,
    clustered_paired_analysis,
    compressed_retention_failure_labels,
    expected_cell_order,
    recover_prediction_prefix,
    retained_set_diagnostics,
    score_evidence_retention_endpoint,
    specificity_decision,
    validate_attention_prior_conditions,
)
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    git_snapshot,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.runtime import (  # noqa: E402
    AttentionPriorAccessDeclaration,
    AttentionPriorTrace,
    SignificanceAttentionPrior,
    SignificanceCache,
    SignificancePolicy,
    attention_prior_decode_append,
    chunked_prefill,
)

FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "accepted_answers",
        "answer",
        "correct",
        "evidence",
        "evidence_annotation",
        "gold",
        "label",
        "reference_answer",
        "supporting_facts",
        "target",
    }
)


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner helper: {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def _save_positions(path: Path, resident: Sequence[np.ndarray]) -> str:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **{f"layer_{index:03d}": value for index, value in enumerate(resident)},
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _load_positions(path: Path, *, layers: int, heads: int) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        names = sorted(arrays.files)
        if names != [f"layer_{index:03d}" for index in range(layers)]:
            raise RuntimeError(f"resident-position layer grid changed: {path}")
        result = [np.asarray(arrays[name], dtype=np.int64) for name in names]
    if any(value.ndim != 2 or value.shape[0] != heads for value in result):
        raise RuntimeError(f"resident-position head grid changed: {path}")
    return result


def _tensor_hash_update(digest: Any, tensor: Any, *, name: str) -> None:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _cache_state_sha256(cache: SignificanceCache) -> str:
    digest = hashlib.sha256()
    for layer_idx, layer in enumerate(cache.layers):
        if not layer.is_initialized or layer.state is None:
            raise RuntimeError(f"cache layer {layer_idx} is incomplete")
        for name, tensor in (
            ("keys", layer.keys),
            ("values", layer.values),
            ("scores", layer.state.scores),
            ("counts", layer.state.counts),
            ("positions", layer.state.positions),
            ("entry_ids", layer.state.entry_ids),
        ):
            _tensor_hash_update(digest, tensor, name=f"{layer_idx}:{name}")
    return digest.hexdigest()


def _cache_bytes(cache: SignificanceCache) -> int:
    return int(
        sum(
            tensor.numel() * tensor.element_size()
            for layer in cache.layers
            if layer.is_initialized
            for tensor in (layer.keys, layer.values)
        )
    )


def _resident_arrays(cache: SignificanceCache) -> list[np.ndarray]:
    return [
        layer.state.positions[0].detach().cpu().numpy().copy()
        for layer in cache.layers
        if layer.state is not None
    ]


def _resident_grid(cache: SignificanceCache) -> list[list[list[int]]]:
    return [
        [[int(value) for value in head] for head in layer.state.positions[0].tolist()]
        for layer in cache.layers
        if layer.state is not None
    ]


class BoundaryCollector:
    def __init__(self, *, input_tokens: int) -> None:
        self.input_tokens = int(input_tokens)
        self.rows: list[dict[str, object]] = []
        self._previous: list[list[list[int]]] | None = None

    def __call__(
        self,
        cache: SignificanceCache,
        policy: SignificancePolicy,
        chunk_index: int,
    ):
        report = cache.compress(policy)
        current = _resident_grid(cache)
        metrics = retained_set_diagnostics(
            current,
            input_tokens=self.input_tokens,
            previous_positions=self._previous,
        )
        self.rows.append(
            {
                "boundary_index": int(chunk_index),
                "true_length": cache.true_length,
                "stored_length": cache.stored_length(),
                "compression": asdict(report),
                **metrics,
            }
        )
        self._previous = current
        return report


def summarize_prior_trace(trace: AttentionPriorTrace) -> dict[str, object]:
    applications = trace.applications
    if not applications:
        raise RuntimeError("attention-prior trace is empty")

    def average(name: str, rows: Sequence[Any]) -> float:
        return float(mean(float(getattr(row, name)) for row in rows)) if rows else 0.0

    by_role: dict[str, dict[str, object]] = {}
    for role in sorted({row.forward_role for row in applications}):
        rows = [row for row in applications if row.forward_role == role]
        entropy_rows = [row for row in rows if row.attention_entropy_normalized_mean is not None]
        by_role[role] = {
            "applications": len(rows),
            "nonzero_bias_applications": sum(row.nonzero_bias_applied for row in rows),
            "mean_attention_entropy_normalized": average(
                "attention_entropy_normalized_mean", entropy_rows
            ),
            "mean_attention_entropy_nats": average("attention_entropy_nats_mean", entropy_rows),
            "mean_source_entropy_normalized": average("source_entropy_normalized", rows),
            "mean_prior_entropy_normalized": average("prior_entropy_normalized", rows),
            "maximum_abs_bias": max(row.bias_abs_max for row in rows),
            "maximum_current_key_abs_bias": max(row.current_key_bias_abs_max for row in rows),
            "maximum_invalid_key_abs_bias": max(row.invalid_key_bias_abs_max for row in rows),
            "mean_old_minus_recent_bias": average("old_minus_recent_bias", rows),
            "mean_clipped_fraction": average("clipped_fraction", rows),
        }

    temporal: list[dict[str, object]] = []
    grouped: defaultdict[tuple[str, int, int], list[Any]] = defaultdict(list)
    for row in applications:
        grouped[(row.forward_role, row.key_length, row.scoring_events_before_forward)].append(row)
    for (role, key_length, scoring_events), rows in sorted(grouped.items()):
        entropy_rows = [row for row in rows if row.attention_entropy_normalized_mean is not None]
        temporal.append(
            {
                "forward_role": role,
                "key_length": key_length,
                "scoring_events_before_forward": scoring_events,
                "layers": len(rows),
                "mean_attention_entropy_normalized": average(
                    "attention_entropy_normalized_mean", entropy_rows
                ),
                "mean_source_entropy_normalized": average("source_entropy_normalized", rows),
                "maximum_abs_bias": max(row.bias_abs_max for row in rows),
                "mean_old_minus_recent_bias": average("old_minus_recent_bias", rows),
            }
        )
    entropy_rows = [
        row for row in applications if row.attention_entropy_normalized_mean is not None
    ]
    return {
        "applications": len(applications),
        "nonzero_bias_applications": sum(row.nonzero_bias_applied for row in applications),
        "mean_attention_entropy_normalized": average(
            "attention_entropy_normalized_mean", entropy_rows
        ),
        "mean_source_entropy_normalized": average("source_entropy_normalized", applications),
        "mean_prior_entropy_normalized": average("prior_entropy_normalized", applications),
        "maximum_abs_bias": max(row.bias_abs_max for row in applications),
        "maximum_current_key_abs_bias": max(
            row.current_key_bias_abs_max for row in applications
        ),
        "maximum_invalid_key_abs_bias": max(
            row.invalid_key_bias_abs_max for row in applications
        ),
        "mean_old_minus_recent_bias": average("old_minus_recent_bias", applications),
        "mean_clipped_fraction": average("clipped_fraction", applications),
        "by_forward_role": by_role,
        "temporal_by_role_and_key_length": temporal,
        "by_layer": trace.to_dict()["layers"],
    }


def summarize_boundaries(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("bounded B3 execution produced no compression boundaries")
    numeric = (
        "mean_cross_head_jaccard_distance",
        "mean_unique_position_fraction_across_heads",
        "early_position_fraction",
        "middle_position_fraction",
        "late_position_fraction",
        "mean_normalized_absolute_position",
    )
    churn = [
        float(row["mean_boundary_churn_jaccard_distance"])
        for row in rows
        if row["mean_boundary_churn_jaccard_distance"] is not None
    ]
    return {
        "boundaries": len(rows),
        **{f"mean_{name}": mean(float(row[name]) for row in rows) for name in numeric},
        "mean_boundary_churn_jaccard_distance": mean(churn) if churn else None,
        "rows": [dict(row) for row in rows],
    }


def _prior_for(condition: Any, config: Mapping[str, Any]) -> SignificanceAttentionPrior:
    prior = dict(config["attention_prior"])
    return SignificanceAttentionPrior(
        mode=condition.source,
        global_lambda=condition.lambda_value,
        clip=float(prior["clip"]),
        epsilon=float(prior["epsilon"]),
        max_abs_lambda=float(prior["max_abs_lambda"]),
        shuffle_seed=int(prior["shuffle_seed"]),
        access=AttentionPriorAccessDeclaration(query_access="conditioned"),
    )


def execute_cell(
    model: Any,
    prepared: Any,
    *,
    condition: Any,
    config: Mapping[str, Any],
    eos_token_ids: set[int],
    prior_override: SignificanceAttentionPrior | object | None = ...,
) -> dict[str, Any]:
    import psutil
    import torch

    cache_config = dict(config["cache"])
    runtime = dict(config["runtime"])
    policy = SignificancePolicy(
        budget=int(cache_config["retained_history_budget"]),
        sinks=int(cache_config["sink_tokens"]),
        recent=int(cache_config["recent_tokens"]),
        count_exponent=1.0,
        pool_width=1,
        require_heavy=True,
    )
    prior = _prior_for(condition, config) if prior_override is ... else prior_override
    trace = AttentionPriorTrace() if prior is not None else None
    cache = SignificanceCache(config=model.config)
    collector = BoundaryCollector(input_tokens=int(prepared.input_ids.shape[-1]))
    process = psutil.Process()
    rss = [int(process.memory_info().rss)]
    trajectory = hashlib.sha256()

    prefill_started = time.perf_counter()
    prefill = chunked_prefill(
        model,
        prepared.input_ids,
        cache,
        chunk_size=int(cache_config["compression_chunk_tokens"]),
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(runtime["verify_query_rollback"]),
        collect_logits=False,
        logits_to_keep=1,
        compression_callback=collector,
        attention_prior=prior,
        prior_trace=trace,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    cache.assert_aligned()
    resident_after_prefill = cache.stored_length()
    logits = prefill.output.logits
    _tensor_hash_update(trajectory, logits.float(), name="prefill_final_logits")
    rss.append(int(process.memory_info().rss))

    decode_started = time.perf_counter()
    generated: list[int] = []
    decode_appends = 0
    stop_reason = "max_new_tokens"
    for index in range(int(runtime["max_new_tokens"])):
        token_id = int(logits[0, -1].float().argmax(dim=-1).item())
        generated.append(token_id)
        if token_id in eos_token_ids:
            stop_reason = "eos_token"
            break
        if index == int(runtime["max_new_tokens"]) - 1:
            break
        token = torch.tensor([[token_id]], dtype=torch.long)
        if prior is None:
            step = chunked_prefill(
                model,
                token,
                cache,
                chunk_size=1,
                policy=None,
                collect_logits=False,
                logits_to_keep=1,
            )
            output = step.output
        else:
            output = attention_prior_decode_append(
                model,
                cache,
                token,
                attention_prior=prior,
                prior_trace=trace,
                logits_to_keep=1,
            )
        logits = output.logits
        _tensor_hash_update(trajectory, logits.float(), name=f"decode_{decode_appends:03d}")
        decode_appends += 1
    decode_seconds = time.perf_counter() - decode_started
    cache.assert_aligned()
    rss.append(int(process.memory_info().rss))

    expected_boundaries = (int(prepared.input_ids.shape[-1]) - 1) // int(
        cache_config["compression_chunk_tokens"]
    )
    if len(collector.rows) != expected_boundaries:
        raise RuntimeError("compression-boundary count changed")
    if cache.scoring_events() != [expected_boundaries] * len(cache.layers):
        raise RuntimeError("query-probe scoring-event count changed")
    trace_summary = summarize_prior_trace(trace) if trace is not None else None
    if trace_summary is not None:
        if float(trace_summary["maximum_current_key_abs_bias"]) != 0.0:
            raise RuntimeError("attention prior reached a current-forward key")
        if float(trace_summary["maximum_invalid_key_abs_bias"]) != 0.0:
            raise RuntimeError("attention prior reached a masked key")
    kv_bytes = _cache_bytes(cache)
    matrix_bytes = cache.matrix_bytes()
    metadata_bytes = cache.metadata_bytes()
    result = {
        "condition": condition.condition_id,
        "source": condition.source,
        "lambda": condition.lambda_value,
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "decode_cache_appends": decode_appends,
        "stop_reason": stop_reason,
        "trajectory_logits_sha256": trajectory.hexdigest(),
        "cache_state_sha256": _cache_state_sha256(cache),
        "resident_after_prefill": resident_after_prefill,
        "resident_after_decode": cache.stored_length(),
        "query_access": {
            "mode": "conditioned",
            "query_token_sha256": prepared.query_token_sha256,
            "query_tokens": int(prepared.query_ids.shape[-1]),
            "probe_schedule": "every_nonfinal_chunk_boundary",
            "answer_access_during_inference": False,
        },
        "attention_feedback": trace_summary,
        "retained_set": summarize_boundaries(collector.rows),
        "efficiency": {
            "kv_only_bytes": kv_bytes,
            "significance_matrix_bytes": matrix_bytes,
            "alignment_metadata_bytes": metadata_bytes,
            "total_persistent_bytes": kv_bytes + matrix_bytes + metadata_bytes,
            "prefill_wall_seconds": prefill_seconds,
            "decode_wall_seconds": decode_seconds,
            "wall_seconds": prefill_seconds + decode_seconds,
            "sampled_peak_rss_bytes": max(rss),
            "peak_vram_bytes": 0,
        },
        "_resident_positions": _resident_arrays(cache),
    }
    del cache, prefill
    gc.collect()
    return result


def identity_gate(
    model: Any,
    prepared: Any,
    *,
    c0_condition: Any,
    config: Mapping[str, Any],
    eos_token_ids: set[int],
) -> dict[str, object]:
    baseline = execute_cell(
        model,
        prepared,
        condition=c0_condition,
        config=config,
        eos_token_ids=eos_token_ids,
        prior_override=None,
    )
    zero = execute_cell(
        model,
        prepared,
        condition=c0_condition,
        config=config,
        eos_token_ids=eos_token_ids,
    )
    baseline_positions = baseline.pop("_resident_positions")
    zero_positions = zero.pop("_resident_positions")
    checks = {
        "generated_token_ids_exact": baseline["generated_token_ids"] == zero["generated_token_ids"],
        "trajectory_logits_exact": baseline["trajectory_logits_sha256"]
        == zero["trajectory_logits_sha256"],
        "cache_state_exact": baseline["cache_state_sha256"] == zero["cache_state_sha256"],
        "resident_positions_exact": all(
            np.array_equal(left, right)
            for left, right in zip(baseline_positions, zero_positions, strict=True)
        ),
        "retained_set_diagnostics_exact": baseline["retained_set"] == zero["retained_set"],
        "zero_prior_nonzero_applications": int(
            dict(zero["attention_feedback"])["nonzero_bias_applications"]
        )
        == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "case_id": prepared.case_id,
        "checks": checks,
        "baseline": {
            "generated_token_ids": baseline["generated_token_ids"],
            "trajectory_logits_sha256": baseline["trajectory_logits_sha256"],
            "cache_state_sha256": baseline["cache_state_sha256"],
        },
        "lambda_zero": {
            "generated_token_ids": zero["generated_token_ids"],
            "trajectory_logits_sha256": zero["trajectory_logits_sha256"],
            "cache_state_sha256": zero["cache_state_sha256"],
        },
        "answer_key_accessed": False,
    }


def validate_quality_config(config: Mapping[str, Any]) -> tuple[Any, ...]:
    if config["experiment"]["classification"] != "EXPLORATORY_NOT_REGISTERED":
        raise ValueError("attention-prior quality screen must remain exploratory")
    if int(config["protocol"]["total_cells"]) != 60:
        raise ValueError("quality screen must contain exactly 60 cells")
    if bool(config["protocol"]["new_conditions_registered"]):
        raise ValueError("attention-prior conditions were not registered")
    frozen_lambda = float(config["frozen_lambda_source"]["lambda"])
    return validate_attention_prior_conditions(config["conditions"], frozen_lambda=frozen_lambda)


def matching_audit(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, object]:
    case_ids = [str(value) for value in config["selection"]["selected_case_ids"]]
    planned = expected_cell_order(case_ids)
    recovered = recover_prediction_prefix(rows, planned)
    failures: list[str] = []
    for case_id in case_ids:
        case_rows = [row for row in recovered if row["case_id"] == case_id]
        query_hashes = {str(row["query_access"]["query_token_sha256"]) for row in case_rows}
        query_lengths = {int(row["query_access"]["query_tokens"]) for row in case_rows}
        resident_prefill = {int(row["resident_after_prefill"]) for row in case_rows}
        answer_access = {bool(row["query_access"]["answer_access_during_inference"]) for row in case_rows}
        if len(query_hashes) != 1:
            failures.append(f"{case_id}:query_hash")
        if len(query_lengths) != 1:
            failures.append(f"{case_id}:query_length")
        if len(resident_prefill) != 1:
            failures.append(f"{case_id}:resident_after_prefill")
        if answer_access != {False}:
            failures.append(f"{case_id}:answer_access")
    checks = {
        "exact_12_by_5_order": len(recovered) == 60,
        "same_model_for_all_cells": True,
        "same_seed_for_all_cells": True,
        "same_B128_C128_storage_contract": True,
        "same_exponent_one_cache_policy": True,
        "same_query_tokens_and_probe_schedule_within_case": not any(
            "query_" in failure for failure in failures
        ),
        "same_resident_prefill_size_within_case": not any(
            "resident_after_prefill" in failure for failure in failures
        ),
        "same_manual_greedy_decoder_and_max_tokens": True,
        "no_answer_access_during_inference": not any(
            "answer_access" in failure for failure in failures
        ),
    }
    return {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) and not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "cells": len(recovered),
        "clusters": len(case_ids),
        "model_id": config["model"]["core_id"],
        "seed": int(config["runtime"]["seed"]),
        "cache": {
            "retained_history_budget": int(config["cache"]["retained_history_budget"]),
            "compression_chunk_tokens": int(config["cache"]["compression_chunk_tokens"]),
            "sink_tokens": int(config["cache"]["sink_tokens"]),
            "recent_tokens": int(config["cache"]["recent_tokens"]),
        },
        "decoder": {
            "type": config["runtime"]["decoding"],
            "max_new_tokens": int(config["runtime"]["max_new_tokens"]),
        },
    }


def intervention_activity(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    by_case: defaultdict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["condition"])] = row
    if not by_case or any(
        set(condition_rows) != set(ATTENTION_PRIOR_QUALITY_CONDITION_IDS)
        for condition_rows in by_case.values()
    ):
        raise RuntimeError("intervention activity requires a complete five-condition grid")
    reference = "C1_NORMALIZED_SC"
    comparisons = {}
    for condition in ATTENTION_PRIOR_QUALITY_CONDITION_IDS:
        if condition == reference:
            continue
        comparisons[f"{condition}_vs_{reference}"] = {
            "same_prediction_count": sum(
                rows_by_condition[condition]["prediction"]
                == rows_by_condition[reference]["prediction"]
                for rows_by_condition in by_case.values()
            ),
            "same_logits_trajectory_count": sum(
                rows_by_condition[condition]["trajectory_logits_sha256"]
                == rows_by_condition[reference]["trajectory_logits_sha256"]
                for rows_by_condition in by_case.values()
            ),
            "same_final_cache_state_count": sum(
                rows_by_condition[condition]["cache_state_sha256"]
                == rows_by_condition[reference]["cache_state_sha256"]
                for rows_by_condition in by_case.values()
            ),
            "clusters": len(by_case),
        }
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "reference_condition": reference,
        "comparisons": comparisons,
        "interpretation": (
            "trajectory/cache differences establish that nonzero interventions were active; "
            "they are not quality improvements"
        ),
    }
def _prepared_cases(
    tokenizer: Any,
    inference: Mapping[str, Mapping[str, Any]],
    case_ids: Sequence[str],
    *,
    capability: Any,
    mini: Any,
    query_limit: int,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for case_id in case_ids:
        row = inference[case_id]
        forbidden = FORBIDDEN_INFERENCE_FIELDS.intersection(row)
        if forbidden:
            raise RuntimeError(f"inference row exposes answer fields: {sorted(forbidden)}")
        input_ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        query_ids = tokenizer(
            str(row["question"]), add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        if capability._token_hash(input_ids) != str(row["token_ids_sha256"]):
            raise RuntimeError(f"frozen prompt token hash mismatch: {case_id}")
        if int(query_ids.shape[-1]) > query_limit:
            raise RuntimeError(f"query exceeds frozen token limit: {case_id}")
        prepared[case_id] = mini.PreparedCase(
            case_id=case_id,
            scenario_id=str(row["cluster_id"]),
            metadata={"family": row["family"]},
            input_ids=input_ids,
            context_ids=input_ids,
            query_ids=query_ids,
            prompt_text_sha256=str(row["prompt_sha256"]),
            input_token_sha256=str(row["token_ids_sha256"]),
            context_token_sha256=str(row["token_ids_sha256"]),
            query_token_sha256=mini._token_sha256(query_ids),
        )
    return prepared


def _contrast(
    rows_by_condition: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    left: str,
    right: str,
    metric: str,
    config: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, object]:
    case_ids = list(config["selection"]["selected_case_ids"])

    def value(row: Mapping[str, Any]) -> float:
        if metric == "answer_correct":
            return float(bool(row["correct"]))
        if metric == "primary_endpoint":
            return float(row["evidence"]["layer_fact_exact_completion_rate"])
        if metric == "attention_entropy":
            return float(row["attention_feedback"]["mean_attention_entropy_normalized"])
        if metric == "retained_diversity":
            return float(row["retained_set"]["mean_mean_cross_head_jaccard_distance"])
        if metric == "retained_churn":
            raw = row["retained_set"]["mean_boundary_churn_jaccard_distance"]
            return 0.0 if raw is None else float(raw)
        raise KeyError(metric)

    result = clustered_paired_analysis(
        case_ids,
        [value(rows_by_condition[left][case_id]) for case_id in case_ids],
        [value(rows_by_condition[right][case_id]) for case_id in case_ids],
        minimum_clusters=int(config["analysis"]["minimum_independent_clusters"]),
        confidence=float(config["analysis"]["confidence"]),
        bootstrap_samples=int(config["analysis"]["bootstrap_samples"]),
        randomization_samples=int(config["analysis"]["randomization_samples"]),
        seed=int(config["runtime"]["seed"]) + seed_offset,
    )
    return asdict(result)


def analyze_scored(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    condition_ids = list(ATTENTION_PRIOR_QUALITY_CONDITION_IDS)
    case_ids = [str(value) for value in config["selection"]["selected_case_ids"]]
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {
        condition: {
            str(row["case_id"]): row for row in rows if row["condition"] == condition
        }
        for condition in condition_ids
    }
    if any(set(indexed[condition]) != set(case_ids) for condition in condition_ids):
        raise RuntimeError("scored rows do not form the frozen 12 x 5 grid")
    comparisons: dict[str, dict[str, object]] = {}
    controls = [
        "C0_B3_NO_PRIOR",
        "C5_RAW_S",
        "C4_SHUFFLED_SC",
        "C6_AGE_ONLY",
    ]
    offset = 0
    for control in controls:
        for metric in (
            "primary_endpoint",
            "answer_correct",
            "attention_entropy",
            "retained_diversity",
            "retained_churn",
        ):
            comparisons[f"C1_NORMALIZED_SC_minus_{control}__{metric}"] = _contrast(
                indexed,
                left=control,
                right="C1_NORMALIZED_SC",
                metric=metric,
                config=config,
                seed_offset=offset,
            )
            offset += 1
    specificity_inputs = {
        control: comparisons[f"C1_NORMALIZED_SC_minus_{control}__primary_endpoint"]
        for control in ("C5_RAW_S", "C4_SHUFFLED_SC", "C6_AGE_ONLY")
    }
    specificity = specificity_decision(specificity_inputs)
    by_condition: dict[str, dict[str, object]] = {}
    by_family: dict[str, dict[str, dict[str, object]]] = {}
    for condition in condition_ids:
        condition_rows = [indexed[condition][case_id] for case_id in case_ids]
        by_condition[condition] = {
            "clusters": len(condition_rows),
            "correct": sum(bool(row["correct"]) for row in condition_rows),
            "mean_primary_endpoint": mean(
                float(row["evidence"]["layer_fact_exact_completion_rate"])
                for row in condition_rows
            ),
            "strict_evidence_successes": sum(
                bool(row["evidence"]["all_required_survived"]) for row in condition_rows
            ),
            "mean_attention_entropy_normalized": mean(
                float(row["attention_feedback"]["mean_attention_entropy_normalized"])
                for row in condition_rows
            ),
            "mean_source_entropy_normalized": mean(
                float(row["attention_feedback"]["mean_source_entropy_normalized"])
                for row in condition_rows
            ),
            "maximum_abs_bias": max(
                float(row["attention_feedback"]["maximum_abs_bias"])
                for row in condition_rows
            ),
            "mean_retained_diversity": mean(
                float(row["retained_set"]["mean_mean_cross_head_jaccard_distance"])
                for row in condition_rows
            ),
            "mean_retained_churn": mean(
                float(row["retained_set"]["mean_boundary_churn_jaccard_distance"] or 0.0)
                for row in condition_rows
            ),
            "mean_early_capture": mean(
                float(row["retained_set"]["mean_early_position_fraction"])
                for row in condition_rows
            ),
            "mean_late_capture": mean(
                float(row["retained_set"]["mean_late_position_fraction"])
                for row in condition_rows
            ),
        }
        by_family[condition] = {}
        for family in config["selection"]["families"]:
            family_rows = [row for row in condition_rows if row["task_family"] == family]
            by_family[condition][str(family)] = {
                "clusters": len(family_rows),
                "correct": sum(bool(row["correct"]) for row in family_rows),
                "mean_primary_endpoint": mean(
                    float(row["evidence"]["layer_fact_exact_completion_rate"])
                    for row in family_rows
                ),
            }
    answer_pairs = {}
    for control in controls:
        left = indexed[control]
        right = indexed["C1_NORMALIZED_SC"]
        answer_pairs[f"C1_NORMALIZED_SC_minus_{control}"] = {
            "wins": sum(not bool(left[key]["correct"]) and bool(right[key]["correct"]) for key in case_ids),
            "losses": sum(bool(left[key]["correct"]) and not bool(right[key]["correct"]) for key in case_ids),
            "ties": sum(bool(left[key]["correct"]) == bool(right[key]["correct"]) for key in case_ids),
        }
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_NOT_REGISTERED",
        "independence_unit": "whole_cluster_id",
        "clusters": len(case_ids),
        "cells": len(rows),
        "primary_endpoint": "layer_fact_exact_completion_rate",
        "by_condition": by_condition,
        "by_family": by_family,
        "paired_answer_transitions": answer_pairs,
        "whole_cluster_paired_comparisons": comparisons,
        "specificity_decision": specificity,
        "unavailable_diagnostics": {
            "false_positive_persistence": (
                "not scored because the frozen evaluation view names distractor/stale segments "
                "but does not provide exact tokenizer-position spans for them"
            )
        },
        "feedback_safety_generalization": "PROHIBITED",
    }


def run(
    config_path: Path,
    *,
    resume: bool = False,
    max_new_cells: int | None = None,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.resolve(strict=True)
    config = _load(config_path)
    conditions = validate_quality_config(config)
    config_sha = sha256_file(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    resolved = {
        name: _resolve(base, value)
        for name, value in paths.items()
        if name != "base"
    }
    output = resolved["output_root"]
    state_path = output / "run_state.json"
    partial_path = output / "predictions.partial.jsonl"
    positions_dir = output / "resident_positions"
    diagnostics_dir = output / "diagnostics"
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError(f"refusing non-empty output without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    positions_dir.mkdir(exist_ok=True)
    diagnostics_dir.mkdir(exist_ok=True)

    selected_ids = [str(value) for value in config["selection"]["selected_case_ids"]]
    planned = expected_cell_order(selected_ids)
    if len(planned) != int(config["protocol"]["total_cells"]):
        raise RuntimeError("frozen cell plan is not 12 x 5")
    if resume:
        if not state_path.is_file() or not partial_path.is_file():
            raise RuntimeError("resume requires run_state.json and predictions.partial.jsonl")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha:
            raise RuntimeError("resume config hash changed")
        if state.get("status") == "COMPLETE":
            raise RuntimeError("quality screen is already complete")
        unscored = recover_prediction_prefix(_jsonl(partial_path), planned)
    else:
        unscored: list[dict[str, object]] = []
        state = {
            "schema_version": "1.0",
            "run_id": output.name,
            "status": "RUNNING",
            "config_sha256": config_sha,
            "created_at": utc_now(),
            "completed_cells": 0,
            "total_cells": len(planned),
            "answer_key_accessed": False,
        }
        atomic_write_json(state_path, state)
        _write_jsonl(partial_path, unscored)
    for row in unscored:
        for name in ("resident_positions", "diagnostics"):
            artifact = output / str(row[name])
            if not artifact.is_file() or sha256_file(artifact) != str(row[f"{name}_sha256"]):
                raise RuntimeError(f"resume artifact is missing or changed: {artifact}")

    frozen = dict(config["frozen_case_source"])
    for key, sha_key in (
        ("registered_config", "registered_config_sha256"),
        ("registration_artifact", "registration_artifact_sha256"),
    ):
        path = _resolve(base, frozen[key])
        if sha256_file(path) != str(frozen[sha_key]):
            raise RuntimeError(f"frozen {key} changed")
    registered = _load(_resolve(base, frozen["registered_config"]))
    if selected_ids != [str(value) for value in registered["selection"]["selected_case_ids"]]:
        raise RuntimeError("selected IDs differ from the registered source set")
    lambda_source = dict(config["frozen_lambda_source"])
    for key, sha_key in (("config", "config_sha256"), ("smoke_summary", "smoke_summary_sha256")):
        path = _resolve(base, lambda_source[key])
        if sha256_file(path) != str(lambda_source[sha_key]):
            raise RuntimeError(f"frozen lambda {key} changed")
    smoke_summary = json.loads(_resolve(base, lambda_source["smoke_summary"]).read_text(encoding="utf-8"))
    if float(smoke_summary["C1_tiny_global_lambda"]["lambda"]) != float(lambda_source["lambda"]):
        raise RuntimeError("quality lambda differs from the compatibility smoke")

    source_hashes = dict(frozen["source_hashes"])
    if sha256_file(resolved["inference"]) != str(source_hashes["inference"]):
        raise RuntimeError("frozen inference view changed")
    if sha256_file(resolved["capability_cases"]) != str(source_hashes["capability_cases"]):
        raise RuntimeError("frozen capability cases changed")
    capability_rows = {str(row["case_id"]): row for row in _jsonl(resolved["capability_cases"])}
    for case_id in selected_ids:
        row = capability_rows.get(case_id)
        if row is None or not bool(row.get("correct")):
            raise RuntimeError(f"selected case is not B0-correct: {case_id}")

    inference_rows = {
        str(row["case_id"]): row
        for row in _jsonl(resolved["inference"])
        if str(row["case_id"]) in set(selected_ids)
    }
    if set(inference_rows) != set(selected_ids):
        raise RuntimeError("frozen inference view is missing selected IDs")
    model_path = resolved["model"]
    observed_hashes = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(model_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    if observed_hashes != dict(config["model"]["expected_hashes"]):
        raise RuntimeError("Halo checkpoint hash mismatch")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name, expected in (
        ("num_hidden_layers", config["model"]["expected_layers"]),
        ("hidden_size", config["model"]["expected_hidden_size"]),
        ("num_attention_heads", config["model"]["expected_query_heads"]),
        ("num_key_value_heads", config["model"]["expected_kv_heads"]),
        ("head_dim", config["model"]["expected_head_dim"]),
        ("vocab_size", config["model"]["expected_vocab_size"]),
    ):
        if int(getattr(model.config, name)) != int(expected):
            raise RuntimeError(f"Halo geometry mismatch for {name}")
    mini = _module("attention_quality_mini", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module(
        "attention_quality_capability", LAB_ROOT / "scripts" / "run_halo_capability.py"
    )
    prepared = _prepared_cases(
        tokenizer,
        inference_rows,
        selected_ids,
        capability=capability,
        mini=mini,
        query_limit=int(config["query"]["max_tokens"]),
    )
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))

    identity_path = output / "c0_identity_gate.json"
    if not identity_path.is_file():
        identity = identity_gate(
            model,
            prepared[selected_ids[0]],
            c0_condition=conditions[0],
            config=config,
            eos_token_ids=eos_ids,
        )
        atomic_write_json(identity_path, identity)
    else:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "PASS" or identity.get("answer_key_accessed") is not False:
        raise RuntimeError(f"C0 identity gate failed before answer access: {identity}")

    newly_completed = 0
    for case_id, condition_id in planned[len(unscored) :]:
        condition = next(row for row in conditions if row.condition_id == condition_id)
        torch.manual_seed(int(config["runtime"]["seed"]))
        result = execute_cell(
            model,
            prepared[case_id],
            condition=condition,
            config=config,
            eos_token_ids=eos_ids,
        )
        resident = result.pop("_resident_positions")
        position_path = positions_dir / f"{case_id}__{condition_id}.npz"
        position_sha = _save_positions(position_path, resident)
        diagnostic_path = diagnostics_dir / f"{case_id}__{condition_id}.json"
        diagnostic_payload = {
            "case_id": case_id,
            "condition": condition_id,
            "attention_feedback": result["attention_feedback"],
            "retained_set": result["retained_set"],
            "query_access": result["query_access"],
        }
        atomic_write_json(diagnostic_path, diagnostic_payload)
        prediction = tokenizer.decode(
            result.pop("generated_token_ids"), skip_special_tokens=True
        ).strip()
        row = {
            "case_id": case_id,
            "cluster_id": inference_rows[case_id]["cluster_id"],
            "family": inference_rows[case_id]["family"],
            "condition": condition_id,
            "prediction": prediction,
            "resident_positions": position_path.relative_to(output).as_posix(),
            "resident_positions_sha256": position_sha,
            "diagnostics": diagnostic_path.relative_to(output).as_posix(),
            "diagnostics_sha256": sha256_file(diagnostic_path),
            **result,
        }
        unscored.append(row)
        _write_jsonl(partial_path, unscored)
        state.update(
            {
                "status": "RUNNING",
                "completed_cells": len(unscored),
                "last_completed_cell": [case_id, condition_id],
                "answer_key_accessed": False,
                "updated_at": utc_now(),
            }
        )
        atomic_write_json(state_path, state)
        print(
            json.dumps(
                {
                    "completed_cells": len(unscored),
                    "total_cells": len(planned),
                    "last_cell": [case_id, condition_id],
                }
            ),
            flush=True,
        )
        newly_completed += 1
        if max_new_cells is not None and newly_completed >= max_new_cells:
            state["status"] = "INTERRUPTED"
            atomic_write_json(state_path, state)
            return {
                "status": "INCOMPLETE",
                "completed_cells": len(unscored),
                "total_cells": len(planned),
                "answer_key_accessed": False,
                "output": str(output),
            }

    unscored = recover_prediction_prefix(unscored, planned)
    if len(unscored) != len(planned):
        raise RuntimeError("inference did not complete all 60 cells")
    forbidden_outcomes = {"accepted_answers", "answer", "correct", "evidence"}
    if any(forbidden_outcomes.intersection(row) for row in unscored):
        raise RuntimeError("unscored checkpoint contains answer-derived fields")
    predictions_path = output / "predictions.unscored.jsonl"
    _write_jsonl(predictions_path, unscored)
    matching = matching_audit(unscored, config)
    if matching["status"] != "PASS":
        raise RuntimeError(f"matched execution audit failed: {matching}")
    atomic_write_json(output / "matching_audit.json", matching)
    activity = intervention_activity(unscored)
    atomic_write_json(output / "intervention_activity.json", activity)
    state.update(
        {
            "status": "INFERENCE_COMPLETE",
            "completed_cells": len(unscored),
            "predictions_sha256_before_answer_access": sha256_file(predictions_path),
            "answer_key_accessed": False,
            "inference_completed_at": utc_now(),
        }
    )
    atomic_write_json(state_path, state)

    # First evaluation-file access occurs only after all 60 durable predictions.
    if sha256_file(resolved["evaluation"]) != str(source_hashes["evaluation"]):
        raise RuntimeError("frozen evaluation view changed")
    evaluation = {str(row["case_id"]): row for row in _jsonl(resolved["evaluation"])}
    scored: list[dict[str, Any]] = []
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_key_value_heads)
    for row in unscored:
        key = evaluation[str(row["case_id"])]
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
        position_path = output / str(row["resident_positions"])
        if sha256_file(position_path) != str(row["resident_positions_sha256"]):
            raise RuntimeError("resident-position artifact changed before outcome scoring")
        resident = _load_positions(position_path, layers=layers, heads=heads)
        retention = score_posthoc_evidence_retention(annotation, resident)
        evidence = retention.to_dict()
        endpoint = score_evidence_retention_endpoint(retention).to_dict()
        correct = capability._task_correct(
            str(row["family"]),
            str(row["prediction"]),
            [str(value) for value in key["accepted_answers"]],
        )
        scored.append(
            {
                "schema_version": "1.0",
                "run_id": output.name,
                "case_id": row["case_id"],
                "scenario_id": row["cluster_id"],
                "condition": row["condition"],
                "task_family": row["family"],
                "generator_seed": None,
                "prompt_sha256": annotation.prompt_sha256,
                "context_tokens": annotation.token_count,
                "evidence_depth": 0.5,
                "chunk_size": int(config["cache"]["compression_chunk_tokens"]),
                "status": "OK",
                "prediction": row["prediction"],
                "correct": correct,
                "full_cache_correct": True,
                "compression_conditioned_eligible": True,
                "evidence": {
                    "required_count": annotation.required_token_count,
                    "surviving_count_final": round(
                        evidence["minimum_per_layer_required_token_availability_fraction"]
                        * annotation.required_token_count
                    ),
                    "all_required_survived": evidence[
                        "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer"
                    ],
                    **evidence,
                    **endpoint,
                },
                "attention_feedback": row["attention_feedback"],
                "retained_set": row["retained_set"],
                "query_access": row["query_access"],
                "efficiency": row["efficiency"],
                "generated_tokens": row["generated_tokens"],
                "decode_cache_appends": row["decode_cache_appends"],
                "stop_reason": row["stop_reason"],
                "failure_labels": compressed_retention_failure_labels(
                    correct=correct,
                    evidence=evidence,
                ),
                "failure_label_scope": "descriptive_final_retained_state_not_causal_attribution",
                "trace_ids": [row["resident_positions"], row["diagnostics"]],
                "error": None,
                "accepted_answers": key["accepted_answers"],
            }
        )
    cases_path = output / "cases.jsonl"
    _write_jsonl(cases_path, scored)
    statistics = analyze_scored(scored, config)
    atomic_write_json(output / "statistics.json", statistics)
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_NOT_REGISTERED",
        "selected_frozen_clusters": len(selected_ids),
        "cells": len(scored),
        "conditions": list(ATTENTION_PRIOR_QUALITY_CONDITION_IDS),
        "c0_identity_gate": identity,
        "by_condition": statistics["by_condition"],
        "by_family": statistics["by_family"],
        "paired_answer_transitions": statistics["paired_answer_transitions"],
        "specificity_decision": statistics["specificity_decision"],
        "matching_audit": matching,
        "intervention_activity": activity,
        "claims_supported": [
            "five closed-loop attention-prior conditions were measured on the exact frozen 12-cluster Halo set",
            "all conditions used B128/C128, exponent-one cache selection, identical query access, and manual greedy decoding",
        ],
        "claims_not_supported": list(config["claims"]["prohibited"]),
    }
    atomic_write_json(output / "summary.json", summary)
    resolved_config = output / "resolved_config.yaml"
    resolved_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repository = git_snapshot(REPO_ROOT)
    repository.pop("repo_root", None)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": state["created_at"],
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
            "compute_tier": "B",
            "architecture_variant": "Experiment C normalized/raw/shuffled/age attention priors",
            "benchmark": "halo_endpoint_validation_v2_frozen_12_clusters",
            "seed": int(config["runtime"]["seed"]),
            "query_conditioned": True,
            "classification": "EXPLORATORY_NOT_REGISTERED",
        },
        "config": {
            "resolved_path": "resolved_config.yaml",
            "sha256": sha256_file(resolved_config),
            "source_path": config_path.relative_to(REPO_ROOT).as_posix(),
        },
        "model": {
            "status": "VERIFIED",
            "id": config["model"]["core_id"],
            "path": model_path.relative_to(REPO_ROOT).as_posix(),
            "architecture": "Qwen3ForCausalLM",
            "hashes": {**observed_hashes, "chat_template": None},
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "precision": "bfloat16",
            "hardware": {"platform": platform.platform(), "logical_cpu_count": os.cpu_count()},
        },
        "budget": {
            "retained_history_entries_per_layer_head": int(
                config["cache"]["retained_history_budget"]
            ),
            "chunk_tokens": int(config["cache"]["compression_chunk_tokens"]),
            "matched": True,
        },
        "artifacts": {
            "identity_gate": "c0_identity_gate.json",
            "run_state": "run_state.json",
            "matching_audit": "matching_audit.json",
            "intervention_activity": "intervention_activity.json",
            "partial_predictions": "predictions.partial.jsonl",
            "predictions": "predictions.unscored.jsonl",
            "cases": "cases.jsonl",
            "statistics": "statistics.json",
            "summary": "summary.json",
            "resident_positions": "resident_positions/",
            "diagnostics": "diagnostics/",
            "resolved_config": "resolved_config.yaml",
        },
        "notes": [f"NOT SUPPORTED: {claim}" for claim in summary["claims_not_supported"]]
        + ["The new attention-prior conditions and contrasts were not registered."],
    }
    atomic_write_json(output / "manifest.json", manifest)
    state.update(
        {
            "status": "COMPLETE",
            "answer_key_accessed": True,
            "completed_at": manifest["completed_at"],
            "cases_sha256": sha256_file(cases_path),
        }
    )
    atomic_write_json(state_path, state)
    return summary


def refresh_completed_audits(config_path: Path) -> dict[str, Any]:
    """Recompute analysis/matching metadata from an already completed sealed grid."""

    config_path = config_path.resolve(strict=True)
    config = _load(config_path)
    validate_quality_config(config)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    output = _resolve(base, paths["output_root"])
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE" or state.get("answer_key_accessed") is not True:
        raise RuntimeError("audit refresh requires a completed quality screen")
    unscored = _jsonl(output / "predictions.unscored.jsonl")
    matching = matching_audit(unscored, config)
    if matching["status"] != "PASS":
        raise RuntimeError(f"matched execution audit failed: {matching}")
    atomic_write_json(output / "matching_audit.json", matching)
    activity = intervention_activity(unscored)
    atomic_write_json(output / "intervention_activity.json", activity)
    cases = _jsonl(output / "cases.jsonl")
    statistics = analyze_scored(cases, config)
    atomic_write_json(output / "statistics.json", statistics)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["by_condition"] = statistics["by_condition"]
    summary["by_family"] = statistics["by_family"]
    summary["paired_answer_transitions"] = statistics["paired_answer_transitions"]
    summary["specificity_decision"] = statistics["specificity_decision"]
    summary["matching_audit"] = matching
    summary["intervention_activity"] = activity
    atomic_write_json(summary_path, summary)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["matching_audit"] = "matching_audit.json"
    manifest["artifacts"]["intervention_activity"] = "intervention_activity.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "output": str(output),
        "matching_audit": matching,
        "intervention_activity": activity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_attention_prior_quality_screen_v1.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--refresh-completed-audits", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        parser.error("--max-new-cells must be positive")
    if args.refresh_completed_audits and (args.resume or args.max_new_cells is not None):
        parser.error("--refresh-completed-audits cannot be combined with execution options")
    result = (
        refresh_completed_audits(args.config)
        if args.refresh_completed_audits
        else run(args.config, resume=args.resume, max_new_cells=args.max_new_cells)
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"PASS", "INCOMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
