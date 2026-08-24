#!/usr/bin/env python3
"""Run the bounded exploratory Halo Experiment-B heterogeneous quality screen."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
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
    EVIDENCE_RETENTION_ENDPOINT_VERSION,
    clustered_paired_analysis,
    compressed_retention_failure_labels,
    score_evidence_retention_endpoint,
)
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

CONDITION_IDS = ("B3_UNIFORM", "B3_ENTROPY", "B3_SHUFFLED")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("quality-screen config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _policy(config: dict[str, Any]) -> SignificancePolicy:
    cache = dict(config["cache"])
    return SignificancePolicy(
        budget=int(cache["retained_history_budget_per_layer_head"]),
        sinks=int(cache["sink_tokens"]),
        recent=int(cache["recent_tokens"]),
        count_exponent=float(cache["count_exponent"]),
        pool_width=1,
        require_heavy=True,
    )


def _query_access(prepared: Any, *, input_tokens: int, chunk_size: int) -> dict[str, Any]:
    boundaries = max(0, math.ceil(input_tokens / chunk_size) - 1)
    query_tokens = int(prepared.query_ids.shape[-1])
    return {
        "mode": "conditioned",
        "probe_schedule": "every_nonfinal_chunk_boundary",
        "prompt_input_tokens": input_tokens,
        "compression_boundaries": boundaries,
        "query_tokens_per_probe": query_tokens,
        "query_probe_token_evaluations": boundaries * query_tokens,
        "query_token_sha256": prepared.query_token_sha256,
    }


def rectangular_byte_ledger(cache: SignificanceCache) -> dict[str, int]:
    """Return the same logical/physical fields as the padded cache ledger."""

    cache.assert_aligned()
    kv_bytes = 0
    logical_entries = 0
    for layer in cache.layers:
        if layer.keys is None or layer.values is None or layer.state is None:
            continue
        kv_bytes += (
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
        )
        logical_entries += layer.state.scores.numel()
    matrix_bytes = cache.matrix_bytes()
    metadata_bytes = cache.metadata_bytes()
    total = kv_bytes + matrix_bytes + metadata_bytes
    return {
        "batch_size": 1,
        "logical_entries_per_sequence": logical_entries,
        "logical_kv_bytes": kv_bytes,
        "physical_kv_bytes": kv_bytes,
        "logical_matrix_bytes": matrix_bytes,
        "physical_matrix_bytes": matrix_bytes,
        "logical_metadata_bytes": metadata_bytes,
        "physical_metadata_bytes": metadata_bytes,
        "validity_mask_bytes": 0,
        "physical_total_bytes": total,
        "logical_payload_bytes": total,
        "representation_overhead_bytes": 0,
    }


def rectangular_resident_positions(cache: SignificanceCache) -> list[list[list[int]]]:
    cache.assert_aligned()
    result: list[list[list[int]]] = []
    for layer in cache.layers:
        if layer.state is None:
            raise RuntimeError("rectangular resident-position export found an empty layer")
        result.append(
            [
                [int(value) for value in layer.state.positions[0, head].detach().cpu().tolist()]
                for head in range(layer.state.positions.shape[1])
            ]
        )
    validate_resident_positions(result)
    return result


def heterogeneous_resident_positions(
    cache: HeterogeneousSignificanceCache,
) -> list[list[list[int]]]:
    """Export only valid semantic positions; padded ``-1`` slots never escape."""

    cache.assert_aligned()
    result: list[list[list[int]]] = []
    for layer in cache.layers:
        if layer.state is None or layer.validity is None:
            raise RuntimeError("heterogeneous resident-position export found an empty layer")
        heads: list[list[int]] = []
        for head in range(layer.state.positions.shape[1]):
            valid = layer.validity[0, head]
            values = layer.state.positions[0, head, valid].detach().cpu().tolist()
            heads.append([int(value) for value in values])
        result.append(heads)
    validate_resident_positions(result)
    return result


def validate_resident_positions(resident: list[list[list[int]]]) -> None:
    if not resident or not resident[0]:
        raise ValueError("resident positions require at least one layer and head")
    head_count = len(resident[0])
    for layer_idx, heads in enumerate(resident):
        if len(heads) != head_count:
            raise ValueError("resident-position layers disagree on KV-head count")
        for head_idx, values in enumerate(heads):
            if any(value < 0 for value in values):
                raise ValueError(
                    f"padding escaped resident positions at layer/head {(layer_idx, head_idx)}"
                )
            if values != sorted(values) or len(values) != len(set(values)):
                raise ValueError(
                    f"resident positions are not unique chronological values at {(layer_idx, head_idx)}"
                )


def save_resident_positions(path: Path, resident: list[list[list[int]]]) -> str:
    """Persist truly variable head rows without a padding sentinel."""

    validate_resident_positions(resident)
    arrays: dict[str, np.ndarray[Any, np.dtype[np.int64]]] = {}
    for layer_idx, heads in enumerate(resident):
        for head_idx, values in enumerate(heads):
            arrays[f"layer_{layer_idx:03d}_head_{head_idx:03d}"] = np.asarray(
                values, dtype=np.int64
            )
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def load_resident_positions(
    path: Path,
    *,
    layer_count: int,
    head_count: int,
) -> list[list[list[int]]]:
    with np.load(path, allow_pickle=False) as arrays:
        expected = {
            f"layer_{layer_idx:03d}_head_{head_idx:03d}"
            for layer_idx in range(layer_count)
            for head_idx in range(head_count)
        }
        if set(arrays.files) != expected:
            raise RuntimeError("resident-position NPZ key grid is incomplete or contains extras")
        resident = [
            [
                arrays[f"layer_{layer_idx:03d}_head_{head_idx:03d}"].astype(
                    np.int64, copy=False
                ).tolist()
                for head_idx in range(head_count)
            ]
            for layer_idx in range(layer_count)
        ]
    validate_resident_positions(resident)
    return resident


def _allocation_summary(
    allocations: Any,
    *,
    boundary_index: int,
    global_budget: int,
) -> dict[str, Any]:
    values = np.asarray(allocations, dtype=np.int64)
    if values.ndim != 2 or int(values.sum()) != global_budget:
        raise RuntimeError("allocation matrix violates its fixed global budget")
    unique = sorted({int(value) for value in values.ravel()})
    layer_totals = values.sum(axis=1)
    return {
        "boundary_index": boundary_index,
        "global_budget": global_budget,
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "unique_values": unique,
        "unique_count": len(unique),
        "heterogeneous": len(unique) > 1,
        "layer_total_minimum": int(layer_totals.min()),
        "layer_total_maximum": int(layer_totals.max()),
    }


def _manual_heterogeneous_decode(
    model: Any,
    cache: HeterogeneousSignificanceCache,
    first_logits: Any,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> dict[str, Any]:
    import torch

    generated: list[int] = []
    decode_cache_appends = 0
    logits = first_logits
    stop_reason = "max_new_tokens"
    for index in range(max_new_tokens):
        token_id = int(logits[0, -1].float().argmax(dim=-1).item())
        generated.append(token_id)
        if token_id in eos_token_ids:
            stop_reason = "eos_token"
            break
        if index == max_new_tokens - 1:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=logits.device)
        step = heterogeneous_decode_append(
            model,
            token,
            cache,
            recording_window=None,
            logits_to_keep=1,
        )
        logits = step.output.logits
        decode_cache_appends += 1
    return {
        "token_ids": generated,
        "generated_tokens": len(generated),
        "decode_cache_appends": decode_cache_appends,
        "stop_reason": stop_reason,
    }


def uniform_identity_gate(
    model: Any,
    prepared: Any,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Require bitwise uniform padded/rectangular identity before screen cells."""

    import torch

    policy = _policy(config)
    cache_config = dict(config["cache"])
    runtime = dict(config["runtime"])
    chunk_size = int(cache_config["compression_chunk_tokens"])
    global_budget = int(cache_config["fixed_global_entry_budget"])
    rectangular = SignificanceCache(config=model.config)
    left = chunked_prefill(
        model,
        prepared.input_ids,
        rectangular,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(runtime["verify_query_rollback"]),
        logits_to_keep=1,
    )
    padded = HeterogeneousSignificanceCache(config=model.config)
    right = heterogeneous_chunked_prefill(
        model,
        prepared.input_ids,
        padded,
        chunk_size=chunk_size,
        policy=policy,
        global_budget=global_budget,
        allocation_strategy="uniform",
        minimum_per_unit=policy.budget,
        maximum_per_unit=policy.budget,
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(runtime["verify_query_rollback"]),
        logits_to_keep=1,
    )
    mismatches: list[str] = []
    for layer_idx, (rect_layer, pad_layer) in enumerate(
        zip(rectangular.layers, padded.layers, strict=True)
    ):
        if pad_layer.validity is None or not bool(pad_layer.validity.all().item()):
            mismatches.append(f"layer_{layer_idx}:validity")
        for name in ("keys", "values"):
            if not torch.equal(getattr(rect_layer, name), getattr(pad_layer, name)):
                mismatches.append(f"layer_{layer_idx}:{name}")
        if rect_layer.state is None or pad_layer.state is None:
            mismatches.append(f"layer_{layer_idx}:state_missing")
            continue
        for name in ("scores", "counts", "positions", "entry_ids"):
            if not torch.equal(getattr(rect_layer.state, name), getattr(pad_layer.state, name)):
                mismatches.append(f"layer_{layer_idx}:{name}")
        for name in ("true_length", "next_entry_id", "evictions"):
            if getattr(rect_layer, name) != getattr(pad_layer, name):
                mismatches.append(f"layer_{layer_idx}:{name}")
        if rect_layer.state.scoring_events != pad_layer.state.scoring_events:
            mismatches.append(f"layer_{layer_idx}:scoring_events")
    left_logits = left.output.logits.float()
    right_logits = right.output.logits.float()
    delta = (left_logits - right_logits).abs()
    exact_logits = bool(torch.equal(left_logits, right_logits))
    expected_boundaries = max(
        0, math.ceil(int(prepared.input_ids.shape[-1]) / chunk_size) - 1
    )
    uniform_reports = len(right.compression_reports) == expected_boundaries and all(
        report.after_logical_entries == global_budget
        and {value for row in report.allocations for value in row} == {policy.budget}
        for report in right.compression_reports
    )
    result = {
        "status": "PASS" if exact_logits and not mismatches and uniform_reports else "FAIL",
        "case_id": prepared.case_id,
        "performed_before_outcome_access": True,
        "exact_logits": exact_logits,
        "max_abs_logit_delta": float(delta.max().item()),
        "exact_cache_state": not mismatches,
        "cache_mismatches": mismatches,
        "uniform_allocation_reports_exact": uniform_reports,
        "compression_boundaries": expected_boundaries,
        "global_budget": global_budget,
    }
    del left, right, rectangular, padded, left_logits, right_logits, delta
    gc.collect()
    if result["status"] != "PASS":
        raise RuntimeError(f"same-process uniform identity gate failed: {result}")
    return result


def execute_rectangular(
    model: Any,
    prepared: Any,
    *,
    config: dict[str, Any],
    eos_token_ids: set[int],
    mini: Any,
) -> tuple[dict[str, Any], list[list[list[int]]]]:
    import psutil

    policy = _policy(config)
    cache_config = dict(config["cache"])
    runtime = dict(config["runtime"])
    chunk_size = int(cache_config["compression_chunk_tokens"])
    global_budget = int(cache_config["fixed_global_entry_budget"])
    input_tokens = int(prepared.input_ids.shape[-1])
    cache = SignificanceCache(config=model.config)
    process = psutil.Process()
    rss = [process.memory_info().rss]
    started = time.perf_counter()
    prefill = chunked_prefill(
        model,
        prepared.input_ids,
        cache,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(runtime["verify_query_rollback"]),
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - started
    cache.assert_aligned()
    prefill_ledger = rectangular_byte_ledger(cache)
    rss.append(process.memory_info().rss)
    reports = []
    kv_heads = int(model.config.num_key_value_heads)
    for boundary_idx, report in enumerate(prefill.compression_reports):
        after_logical = report.after_entries * kv_heads
        if after_logical != global_budget:
            raise RuntimeError("rectangular B3 failed fixed logical global budget")
        reports.append(
            {
                "boundary_index": boundary_idx,
                "global_budget": global_budget,
                "before_logical_entries": report.before_entries * kv_heads,
                "after_logical_entries": after_logical,
                "evicted_logical_entries": report.evictions * kv_heads,
                "allocation_minimum": policy.budget,
                "allocation_maximum": policy.budget,
                "heterogeneous": False,
            }
        )
    decode_started = time.perf_counter()
    decoded = mini._manual_greedy_decode(
        model,
        cache,
        prefill.output.logits,
        max_new_tokens=int(runtime["max_new_tokens"]),
        eos_token_ids=eos_token_ids,
    )
    decode_seconds = time.perf_counter() - decode_started
    cache.assert_aligned()
    post_ledger = rectangular_byte_ledger(cache)
    rss.append(process.memory_info().rss)
    boundaries = max(0, math.ceil(input_tokens / chunk_size) - 1)
    if cache.scoring_events() != [boundaries] * len(cache.layers):
        raise RuntimeError("rectangular query-probe accounting failed")
    resident = rectangular_resident_positions(cache)
    return (
        {
            "condition": "B3_UNIFORM",
            "storage": "rectangular",
            "allocation_strategy": "uniform",
            "generated_token_ids": decoded["token_ids"],
            "generated_tokens": decoded["generated_tokens"],
            "decode_cache_appends": decoded["decode_cache_appends"],
            "stop_reason": decoded["stop_reason"],
            "manual_greedy": True,
            "query_access": _query_access(
                prepared, input_tokens=input_tokens, chunk_size=chunk_size
            ),
            "allocation": {
                "fixed_global_budget": global_budget,
                "boundary_summaries": reports,
                "heterogeneous_boundary_count": 0,
            },
            "efficiency": {
                "prefill": prefill_ledger,
                "post_decode": post_ledger,
                "kv_only_bytes": post_ledger["physical_kv_bytes"],
                "total_persistent_bytes": post_ledger["physical_total_bytes"],
                "architecture_total_persistent_bytes": post_ledger["physical_total_bytes"],
                "logical_payload_bytes": post_ledger["logical_payload_bytes"],
                "physical_total_bytes": post_ledger["physical_total_bytes"],
                "representation_overhead_bytes": 0,
                "sampled_peak_rss_bytes": max(rss),
                "peak_rss_bytes": max(rss),
                "peak_vram_bytes": 0,
                "wall_seconds": prefill_seconds + decode_seconds,
            },
            "timing": {
                "prefill_wall_seconds": prefill_seconds,
                "decode_wall_seconds": decode_seconds,
            },
        },
        resident,
    )


def execute_heterogeneous(
    model: Any,
    prepared: Any,
    *,
    condition_id: str,
    strategy: str,
    config: dict[str, Any],
    eos_token_ids: set[int],
) -> tuple[dict[str, Any], list[list[list[int]]]]:
    import psutil

    policy = _policy(config)
    cache_config = dict(config["cache"])
    runtime = dict(config["runtime"])
    chunk_size = int(cache_config["compression_chunk_tokens"])
    global_budget = int(cache_config["fixed_global_entry_budget"])
    input_tokens = int(prepared.input_ids.shape[-1])
    cache = HeterogeneousSignificanceCache(config=model.config)
    process = psutil.Process()
    rss = [process.memory_info().rss]
    started = time.perf_counter()
    prefill = heterogeneous_chunked_prefill(
        model,
        prepared.input_ids,
        cache,
        chunk_size=chunk_size,
        policy=policy,
        global_budget=global_budget,
        allocation_strategy=strategy,
        minimum_per_unit=int(cache_config["minimum_per_layer_head"]),
        maximum_per_unit=int(cache_config["maximum_per_layer_head"]),
        allocation_exponent=float(cache_config["count_exponent"]),
        allocation_seed=int(runtime["seed"]),
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(runtime["verify_query_rollback"]),
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - started
    cache.assert_aligned()
    prefill_ledger = asdict(cache.byte_accounting())
    rss.append(process.memory_info().rss)
    boundary_summaries = []
    for boundary_idx, (decision, report) in enumerate(
        zip(prefill.allocation_decisions, prefill.compression_reports, strict=True)
    ):
        if decision is None or report.after_logical_entries != global_budget:
            raise RuntimeError("heterogeneous boundary lacks an exact live budget decision")
        summary = _allocation_summary(
            decision.budget.allocations,
            boundary_index=boundary_idx,
            global_budget=global_budget,
        )
        summary.update(
            {
                "before_logical_entries": report.before_logical_entries,
                "after_logical_entries": report.after_logical_entries,
                "before_physical_slots": report.before_physical_slots,
                "after_physical_slots": report.after_physical_slots,
                "evicted_logical_entries": report.evicted_logical_entries,
            }
        )
        boundary_summaries.append(summary)
    heterogeneous_boundaries = sum(bool(row["heterogeneous"]) for row in boundary_summaries)
    if len(boundary_summaries) > 1 and heterogeneous_boundaries == 0:
        raise RuntimeError(f"{condition_id} never produced a heterogeneous live allocation")
    decode_started = time.perf_counter()
    decoded = _manual_heterogeneous_decode(
        model,
        cache,
        prefill.output.logits,
        max_new_tokens=int(runtime["max_new_tokens"]),
        eos_token_ids=eos_token_ids,
    )
    decode_seconds = time.perf_counter() - decode_started
    cache.assert_aligned()
    post_ledger = asdict(cache.byte_accounting())
    rss.append(process.memory_info().rss)
    boundaries = max(0, math.ceil(input_tokens / chunk_size) - 1)
    if cache.scoring_events() != [boundaries] * len(cache.layers):
        raise RuntimeError("heterogeneous query-probe accounting failed")
    resident = heterogeneous_resident_positions(cache)
    return (
        {
            "condition": condition_id,
            "storage": "padded_heterogeneous",
            "allocation_strategy": strategy,
            "generated_token_ids": decoded["token_ids"],
            "generated_tokens": decoded["generated_tokens"],
            "decode_cache_appends": decoded["decode_cache_appends"],
            "stop_reason": decoded["stop_reason"],
            "manual_greedy": True,
            "query_access": _query_access(
                prepared, input_tokens=input_tokens, chunk_size=chunk_size
            ),
            "allocation": {
                "fixed_global_budget": global_budget,
                "boundary_summaries": boundary_summaries,
                "heterogeneous_boundary_count": heterogeneous_boundaries,
            },
            "efficiency": {
                "prefill": prefill_ledger,
                "post_decode": post_ledger,
                "kv_only_bytes": post_ledger["physical_kv_bytes"],
                "total_persistent_bytes": post_ledger["physical_total_bytes"],
                "architecture_total_persistent_bytes": post_ledger["physical_total_bytes"],
                "logical_payload_bytes": post_ledger["logical_payload_bytes"],
                "physical_total_bytes": post_ledger["physical_total_bytes"],
                "representation_overhead_bytes": post_ledger[
                    "representation_overhead_bytes"
                ],
                "sampled_peak_rss_bytes": max(rss),
                "peak_rss_bytes": max(rss),
                "peak_vram_bytes": 0,
                "wall_seconds": prefill_seconds + decode_seconds,
            },
            "timing": {
                "prefill_wall_seconds": prefill_seconds,
                "decode_wall_seconds": decode_seconds,
            },
        },
        resident,
    )


def validate_query_access(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    failures: list[str] = []
    for case_id, case_rows in sorted(by_case.items()):
        if {str(row["condition"]) for row in case_rows} != set(CONDITION_IDS):
            failures.append(f"{case_id}:condition_set")
            continue
        reference = case_rows[0]["query_access"]
        if any(row["query_access"] != reference for row in case_rows[1:]):
            failures.append(f"{case_id}:query_access")
    return {
        "status": "PASS" if not failures else "FAIL",
        "case_count": len(by_case),
        "conditions": list(CONDITION_IDS),
        "query_access_equal_within_case": not failures,
        "failures": failures,
    }


def paired_statistics(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    analysis = dict(config["analysis"])
    indexed = {
        (str(row["case_id"]), str(row["condition"])): row
        for row in rows
    }
    case_ids = sorted({str(row["case_id"]) for row in rows})
    cluster_ids = [str(indexed[(case_id, "B3_UNIFORM")]["scenario_id"]) for case_id in case_ids]
    pairs = (
        ("B3_UNIFORM", "B3_ENTROPY"),
        ("B3_UNIFORM", "B3_SHUFFLED"),
        ("B3_SHUFFLED", "B3_ENTROPY"),
    )
    comparisons: dict[str, Any] = {}
    seed = int(config["runtime"]["seed"])
    for pair_idx, (baseline, candidate) in enumerate(pairs):
        endpoint = clustered_paired_analysis(
            cluster_ids,
            [
                float(indexed[(case_id, baseline)]["evidence"]["layer_fact_exact_completion_rate"])
                for case_id in case_ids
            ],
            [
                float(indexed[(case_id, candidate)]["evidence"]["layer_fact_exact_completion_rate"])
                for case_id in case_ids
            ],
            minimum_clusters=int(analysis["minimum_independent_clusters"]),
            confidence=float(analysis["confidence"]),
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            randomization_samples=int(analysis["randomization_samples"]),
            seed=seed + pair_idx * 2,
        )
        answer = clustered_paired_analysis(
            cluster_ids,
            [float(bool(indexed[(case_id, baseline)]["correct"])) for case_id in case_ids],
            [float(bool(indexed[(case_id, candidate)]["correct"])) for case_id in case_ids],
            minimum_clusters=int(analysis["minimum_independent_clusters"]),
            confidence=float(analysis["confidence"]),
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            randomization_samples=int(analysis["randomization_samples"]),
            seed=seed + pair_idx * 2 + 1,
        )
        name = f"{candidate}_minus_{baseline}"
        comparisons[f"{name}__endpoint"] = asdict(endpoint)
        comparisons[f"{name}__answer_correct"] = asdict(answer)
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_NOT_REGISTERED",
        "independence_unit": "cluster_id",
        "cluster_count": len(set(cluster_ids)),
        "primary_endpoint": "layer_fact_exact_completion_rate",
        "comparisons": comparisons,
        "interpretation_limit": (
            "The case IDs and endpoint were frozen elsewhere, but these three allocation "
            "conditions and contrasts were not registered. All intervals and p-values are exploratory."
        ),
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = [float(row["evidence"]["layer_fact_exact_completion_rate"]) for row in rows]
    logical = [int(row["efficiency"]["logical_payload_bytes"]) for row in rows]
    physical = [int(row["efficiency"]["physical_total_bytes"]) for row in rows]
    overhead = [int(row["efficiency"]["representation_overhead_bytes"]) for row in rows]
    heterogeneous_boundaries = [
        int(row["allocation"]["heterogeneous_boundary_count"]) for row in rows
    ]
    return {
        "clusters": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "mean_layer_fact_exact_completion_rate": float(np.mean(endpoints)),
        "strict_evidence_survival": sum(
            bool(row["evidence"]["all_required_survived"]) for row in rows
        ),
        "mean_logical_payload_bytes_after_decode": float(np.mean(logical)),
        "mean_physical_total_bytes_after_decode": float(np.mean(physical)),
        "mean_representation_overhead_bytes": float(np.mean(overhead)),
        "total_heterogeneous_boundaries": sum(heterogeneous_boundaries),
    }


def run(config_path: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.resolve(strict=True)
    config = _load(config_path)
    if str(config["experiment"]["classification"]) != "EXPLORATORY_NOT_REGISTERED":
        raise RuntimeError("quality screen must remain exploratory and unregistered")
    if [str(row["id"]) for row in config["conditions"]] != list(CONDITION_IDS):
        raise RuntimeError("quality-screen condition order changed")
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    resolved = {name: _resolve(base, value) for name, value in paths.items() if name != "base"}
    output = resolved["output_root"]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty quality-screen output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    positions_dir = output / "resident_positions"
    positions_dir.mkdir()

    frozen = dict(config["frozen_case_source"])
    registered_config_path = _resolve(base, frozen["registered_config"])
    if sha256_file(registered_config_path) != str(frozen["registered_config_sha256"]):
        raise RuntimeError("registered source config hash changed")
    registered_config = _load(registered_config_path)
    selected_ids_ordered = [str(value) for value in config["selection"]["selected_case_ids"]]
    if selected_ids_ordered != [
        str(value) for value in registered_config["selection"]["selected_case_ids"]
    ]:
        raise RuntimeError("quality-screen selected IDs differ from the frozen registered set")
    if len(selected_ids_ordered) != 12 or len(set(selected_ids_ordered)) != 12:
        raise RuntimeError("quality screen requires exactly 12 unique case IDs")
    source_hashes = dict(frozen["source_hashes"])
    # Verify prompt/capability inputs now. The evaluation view is not even
    # hashed until all 36 predictions and resident states are durable, keeping
    # answer-key access temporally separated from inference.
    observed_preoutcome_sources = {
        "capability_cases": sha256_file(resolved["capability_cases"]),
        "inference": sha256_file(resolved["inference"]),
    }
    expected_preoutcome_sources = {
        name: source_hashes[name] for name in observed_preoutcome_sources
    }
    if observed_preoutcome_sources != expected_preoutcome_sources:
        raise RuntimeError("frozen prompt/capability source views changed")
    if str(config["endpoint"]["version"]) != EVIDENCE_RETENTION_ENDPOINT_VERSION:
        raise RuntimeError("frozen endpoint version changed")
    if str(config["endpoint"]["definition_sha256"]) != str(
        registered_config["endpoint"]["definition_sha256"]
    ):
        raise RuntimeError("frozen endpoint definition hash changed")

    capability_rows = {str(row["case_id"]): row for row in _jsonl(resolved["capability_cases"])}
    families = {str(value) for value in config["selection"]["families"]}
    for case_id in selected_ids_ordered:
        capability_row = capability_rows.get(case_id)
        if capability_row is None or not bool(capability_row.get("correct")):
            raise RuntimeError(f"selected case is not frozen B0-correct: {case_id}")
        if str(capability_row.get("task_family")) not in families:
            raise RuntimeError(f"selected case has an undeclared family: {case_id}")
    inference = {
        str(row["case_id"]): row
        for row in _jsonl(resolved["inference"])
        if str(row["case_id"]) in set(selected_ids_ordered)
    }
    if set(inference) != set(selected_ids_ordered):
        raise RuntimeError("frozen inference view is missing selected cases")

    model_path = resolved["model"]
    expected_hashes = dict(config["model"]["expected_hashes"])
    observed_hashes = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(model_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError("Halo checkpoint hash mismatch")
    started_at = utc_now()
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
        ("num_attention_heads", config["model"]["expected_query_heads"]),
        ("num_key_value_heads", config["model"]["expected_kv_heads"]),
        ("head_dim", config["model"]["expected_head_dim"]),
    ):
        if int(getattr(model.config, name)) != int(expected):
            raise RuntimeError(f"Halo geometry mismatch for {name}")
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_key_value_heads)
    global_budget = int(config["cache"]["fixed_global_entry_budget"])
    unit_budget = int(config["cache"]["retained_history_budget_per_layer_head"])
    if global_budget != layers * heads * unit_budget:
        raise RuntimeError("fixed global budget does not equal layers * KV heads * B")

    mini = _module("heterogeneous_screen_mini", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module(
        "heterogeneous_screen_capability", LAB_ROOT / "scripts" / "run_halo_capability.py"
    )
    prepared_cases: dict[str, Any] = {}
    for case_id in selected_ids_ordered:
        row = inference[case_id]
        input_ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        query_ids = tokenizer(
            str(row["question"]), add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        if capability._token_hash(input_ids) != str(row["token_ids_sha256"]):
            raise RuntimeError(f"frozen prompt token hash mismatch: {case_id}")
        if int(query_ids.shape[-1]) > int(config["query"]["max_tokens"]):
            raise RuntimeError(f"query exceeds frozen limit: {case_id}")
        prepared_cases[case_id] = mini.PreparedCase(
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

    identity = uniform_identity_gate(
        model,
        prepared_cases[selected_ids_ordered[0]],
        config=config,
    )
    atomic_write_json(output / "uniform_identity_gate.json", identity)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))

    unscored: list[dict[str, Any]] = []
    for case_id in selected_ids_ordered:
        prepared = prepared_cases[case_id]
        inference_row = inference[case_id]
        for condition_id in CONDITION_IDS:
            if condition_id == "B3_UNIFORM":
                result, resident = execute_rectangular(
                    model,
                    prepared,
                    config=config,
                    eos_token_ids=eos_ids,
                    mini=mini,
                )
            else:
                strategy = (
                    "entropy_proportional"
                    if condition_id == "B3_ENTROPY"
                    else "shuffled_statistic"
                )
                result, resident = execute_heterogeneous(
                    model,
                    prepared,
                    condition_id=condition_id,
                    strategy=strategy,
                    config=config,
                    eos_token_ids=eos_ids,
                )
            position_name = f"{case_id}__{condition_id}.npz"
            position_path = positions_dir / position_name
            position_sha = save_resident_positions(position_path, resident)
            prediction = tokenizer.decode(
                result.pop("generated_token_ids"), skip_special_tokens=True
            ).strip()
            unscored.append(
                {
                    "case_id": case_id,
                    "cluster_id": inference_row["cluster_id"],
                    "family": inference_row["family"],
                    "condition": condition_id,
                    "prediction": prediction,
                    "resident_positions": f"resident_positions/{position_name}",
                    "resident_positions_sha256": position_sha,
                    **result,
                }
            )
    matching = validate_query_access(unscored)
    if matching["status"] != "PASS":
        raise RuntimeError(f"matched query-access audit failed: {matching}")
    atomic_write_json(output / "matching_audit.json", matching)
    predictions_path = output / "predictions.unscored.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in unscored),
        encoding="utf-8",
    )
    if len(unscored) != int(config["protocol"]["total_cells"]):
        raise RuntimeError("quality screen did not durably produce exactly 36 cells")

    # Outcome access starts only after every prediction and exact resident state
    # is durable. No registered endpoint output is consulted.
    if sha256_file(resolved["evaluation"]) != str(source_hashes["evaluation"]):
        raise RuntimeError("frozen evaluation source view changed")
    evaluation = {str(row["case_id"]): row for row in _jsonl(resolved["evaluation"])}
    scored: list[dict[str, Any]] = []
    for row in unscored:
        key = evaluation[str(row["case_id"])]
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
        position_path = output / str(row["resident_positions"])
        if sha256_file(position_path) != str(row["resident_positions_sha256"]):
            raise RuntimeError("resident-position artifact changed before outcome scoring")
        resident = load_resident_positions(
            position_path,
            layer_count=layers,
            head_count=heads,
        )
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
                "efficiency": row["efficiency"],
                "query_access": row["query_access"],
                "allocation": row["allocation"],
                "storage": row["storage"],
                "allocation_strategy": row["allocation_strategy"],
                "generated_tokens": row["generated_tokens"],
                "decode_cache_appends": row["decode_cache_appends"],
                "stop_reason": row["stop_reason"],
                "failure_labels": compressed_retention_failure_labels(
                    correct=correct,
                    evidence=evidence,
                ),
                "failure_label_scope": "descriptive_final_retained_state_not_causal_attribution",
                "trace_ids": [row["resident_positions"]],
                "error": None,
                "accepted_answers": key["accepted_answers"],
            }
        )
    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    statistics = paired_statistics(scored, config=config)
    atomic_write_json(output / "statistics.json", statistics)
    by_condition = {
        condition: _condition_summary(
            [row for row in scored if row["condition"] == condition]
        )
        for condition in CONDITION_IDS
    }
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_NOT_REGISTERED",
        "selected_frozen_clusters": len(selected_ids_ordered),
        "cells": len(scored),
        "conditions": list(CONDITION_IDS),
        "uniform_identity_gate": identity,
        "matching_audit": matching,
        "by_condition": by_condition,
        "cluster_aware_paired_contrasts": statistics["comparisons"],
        "claims_supported": [
            "closed-loop rectangular, entropy, and shuffled allocation outcomes were measured on these 12 frozen Halo cases",
            "every compression boundary conserved the exact same logical global entry budget",
        ],
        "claims_not_supported": [
            "registered or confirmatory evidence for the new allocation conditions",
            "population-level answer or retention improvement",
            "memory saving from head raggedness because the implementation uses physical padding",
            "latency improvement or deployment efficiency",
        ],
    }
    atomic_write_json(output / "summary.json", summary)
    resolved_config = output / "resolved_config.yaml"
    resolved_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    repository = git_snapshot(REPO_ROOT)
    repository.pop("repo_root", None)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
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
            "compute_tier": "B",
            "architecture_variant": "Experiment B uniform versus live entropy/shuffled budgets",
            "benchmark": "halo_endpoint_validation_v2_frozen_case_ids",
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
            "transformers": __import__("transformers").__version__,
            "device": "cpu",
            "precision": "bfloat16",
            "hardware": None,
        },
        "budget": {
            "logical_global_entries_at_each_boundary": global_budget,
            "matched": True,
            "physical_padded_storage_for_heterogeneous_conditions": True,
            "memory_saving_claim": False,
        },
        "artifacts": {
            "uniform_identity_gate": "uniform_identity_gate.json",
            "matching_audit": "matching_audit.json",
            "predictions": "predictions.unscored.jsonl",
            "cases": "cases.jsonl",
            "statistics": "statistics.json",
            "summary": "summary.json",
            "resident_positions": "resident_positions/",
            "resolved_config": "resolved_config.yaml",
        },
        "notes": summary["claims_not_supported"],
    }
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_heterogeneous_budget_quality_screen.yaml",
    )
    args = parser.parse_args(argv)
    result = run(args.config)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
