#!/usr/bin/env python3
"""Run the 72-cell exploratory Halo significance-memory quality screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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
    SignificanceCache,
    SignificanceMemory,
    SignificancePolicy,
    chunked_prefill,
    chunked_prefill_with_memory,
    match_memory_total_budget,
    memory_decode_append,
    rectangular_position_bytes,
)

CONDITION_IDS = (
    "M0_HARD",
    "M4_MEAN",
    "M4_RECENCY",
    "M4_RAW_S",
    "M4_NORMALIZED_S_OVER_C",
    "M4_RANDOM",
)


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    num_slots: int
    aggregation: str
    ordinary_budget: int


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
        raise ValueError("memory quality-screen config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def conditions_from_config(config: dict[str, Any]) -> tuple[ConditionSpec, ...]:
    conditions = tuple(
        ConditionSpec(
            condition_id=str(row["id"]),
            num_slots=int(row["slots"]),
            aggregation=str(row["aggregation"]),
            ordinary_budget=int(row["ordinary_budget"]),
        )
        for row in config["conditions"]
    )
    if tuple(row.condition_id for row in conditions) != CONDITION_IDS:
        raise RuntimeError("memory-screen condition set or order changed")
    if conditions[0].num_slots != 0 or any(row.num_slots != 4 for row in conditions[1:]):
        raise RuntimeError("memory-screen slot counts must be M0 followed by five M4 controls")
    expected_variants = {
        "mean",
        "recency",
        "raw_significance",
        "normalized_significance",
        "random",
    }
    if {row.aggregation for row in conditions[1:]} != expected_variants:
        raise RuntimeError("memory-screen aggregation controls are incomplete")
    return conditions


def solve_total_budget_contract(
    config: dict[str, Any],
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: Any,
) -> dict[str, Any]:
    conditions = conditions_from_config(config)
    unit_bytes = rectangular_position_bytes(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_dtype=kv_dtype,
    )
    probe = SignificanceMemory(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        num_slots=4,
        head_dim=head_dim,
        variant="mean",
        device="cpu",
        kv_dtype=kv_dtype,
        recency_tau=float(config["memory_slots"]["recency_tau"]),
        random_seed=int(config["memory_slots"]["random_seed"]),
    )
    memory_bytes = probe.byte_ledger().total_bytes
    baseline_budget = int(config["cache"]["baseline_retained_history_budget"])
    match = match_memory_total_budget(
        baseline_ordinary_budget=baseline_budget,
        ordinary_bytes_per_budget_unit=unit_bytes,
        memory_bytes=memory_bytes,
        minimum_ordinary_budget=(
            int(config["cache"]["sink_tokens"]) + int(config["cache"]["recent_tokens"]) + 1
        ),
    )
    declared = dict(config["total_persistent_budget"])
    observed = {
        "ordinary_bytes_per_budget_unit": unit_bytes,
        "m0_boundary_total_bytes": match.baseline_total_bytes,
        "m4_memory_bytes": memory_bytes,
        "m4_boundary_total_bytes": match.candidate_total_bytes,
        "unavoidable_discrete_slack_bytes": match.discrete_slack_bytes,
        "memory_ordinary_budget_reduction": match.ordinary_budget_reduction,
    }
    if any(int(declared[name]) != value for name, value in observed.items()):
        raise RuntimeError(
            f"declared total-byte contract disagrees with runtime geometry: {observed}"
        )
    if conditions[0].ordinary_budget != match.baseline_ordinary_budget:
        raise RuntimeError("M0 condition budget differs from total-byte contract")
    if any(row.ordinary_budget != match.candidate_ordinary_budget for row in conditions[1:]):
        raise RuntimeError("an M4 condition budget differs from total-byte contract")
    return {
        **asdict(match),
        "ordinary_budget_reduction": match.ordinary_budget_reduction,
        "matched_total_persistent_bytes": True,
        "match_basis": declared["match_basis"],
        "included_state": list(declared["includes"]),
    }


def _new_memory(
    model: Any,
    condition: ConditionSpec,
    *,
    config: dict[str, Any],
) -> SignificanceMemory:
    return SignificanceMemory(
        num_layers=int(model.config.num_hidden_layers),
        batch_size=1,
        num_kv_heads=int(model.config.num_key_value_heads),
        num_slots=condition.num_slots,
        head_dim=int(model.config.head_dim),
        variant=condition.aggregation,
        device="cpu",
        kv_dtype=next(model.parameters()).dtype,
        recency_tau=float(config["memory_slots"]["recency_tau"]),
        random_seed=int(config["memory_slots"]["random_seed"]),
    )


def _policy(condition: ConditionSpec, *, config: dict[str, Any]) -> SignificancePolicy:
    cache = dict(config["cache"])
    return SignificancePolicy(
        budget=condition.ordinary_budget,
        sinks=int(cache["sink_tokens"]),
        recent=int(cache["recent_tokens"]),
        count_exponent=float(cache["selection_count_exponent"]),
        pool_width=1,
        require_heavy=True,
    )


def _ordinary_ledger(cache: SignificanceCache) -> dict[str, int]:
    cache.assert_aligned()
    kv_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for tensor in (layer.keys, layer.values)
        if tensor is not None
    )
    matrix_bytes = cache.matrix_bytes()
    metadata_bytes = cache.metadata_bytes()
    return {
        "kv_bytes": int(kv_bytes),
        "significance_matrix_bytes": matrix_bytes,
        "alignment_metadata_bytes": metadata_bytes,
        "ordinary_total_bytes": int(kv_bytes) + matrix_bytes + metadata_bytes,
    }


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


def _cache_identity(left: SignificanceCache, right: SignificanceCache) -> list[str]:
    import torch

    mismatches: list[str] = []
    if left.true_length != right.true_length:
        mismatches.append("true_length")
    if left.stored_length() != right.stored_length():
        mismatches.append("stored_length")
    for layer_idx, (left_layer, right_layer) in enumerate(
        zip(left.layers, right.layers, strict=True)
    ):
        for name in ("keys", "values"):
            if not torch.equal(getattr(left_layer, name), getattr(right_layer, name)):
                mismatches.append(f"layer_{layer_idx}:{name}")
        if left_layer.state is None or right_layer.state is None:
            mismatches.append(f"layer_{layer_idx}:state")
            continue
        for name in ("scores", "counts", "positions", "entry_ids"):
            if not torch.equal(getattr(left_layer.state, name), getattr(right_layer.state, name)):
                mismatches.append(f"layer_{layer_idx}:{name}")
        for name in ("true_length", "next_entry_id", "evictions"):
            if getattr(left_layer, name) != getattr(right_layer, name):
                mismatches.append(f"layer_{layer_idx}:{name}")
        if left_layer.state.scoring_events != right_layer.state.scoring_events:
            mismatches.append(f"layer_{layer_idx}:scoring_events")
    return mismatches


def save_resident_positions(path: Path, cache: SignificanceCache) -> str:
    arrays: dict[str, np.ndarray[Any, np.dtype[np.int64]]] = {}
    for layer_idx, layer in enumerate(cache.layers):
        if layer.state is None:
            raise RuntimeError("resident-position export found an empty layer")
        for head_idx in range(layer.state.positions.shape[1]):
            values = layer.state.positions[0, head_idx].detach().cpu().numpy().astype(np.int64)
            if values.tolist() != sorted(values.tolist()) or len(values) != len(
                set(values.tolist())
            ):
                raise RuntimeError("resident positions are not unique chronological values")
            arrays[f"layer_{layer_idx:03d}_head_{head_idx:03d}"] = values
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
            raise RuntimeError("resident-position artifact has an incomplete layer/head grid")
        result = [
            [
                arrays[f"layer_{layer_idx:03d}_head_{head_idx:03d}"]
                .astype(np.int64, copy=False)
                .tolist()
                for head_idx in range(head_count)
            ]
            for layer_idx in range(layer_count)
        ]
    return result


def save_memory_slots(path: Path, memory: SignificanceMemory) -> str:
    """Persist exact slot tensor bits plus every validity/position field."""

    import torch

    states = [memory.retrieve_slots(index) for index in range(memory.num_layers)]
    arrays = {
        "keys_bfloat16_bits": torch.stack([state.keys for state in states])
        .detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy(),
        "values_bfloat16_bits": torch.stack([state.values for state in states])
        .detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy(),
        "valid": torch.stack([state.valid for state in states]).detach().cpu().numpy(),
        "source_positions": torch.stack([state.source_positions for state in states])
        .detach()
        .cpu()
        .numpy(),
        "created_at": torch.stack([state.created_at for state in states]).detach().cpu().numpy(),
        "last_updated_at": torch.stack([state.last_updated_at for state in states])
        .detach()
        .cpu()
        .numpy(),
        "source_counts": torch.stack([state.source_counts for state in states])
        .detach()
        .cpu()
        .numpy(),
        "weight_mass": torch.stack([state.weight_mass for state in states]).detach().cpu().numpy(),
    }
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def validate_memory_slot_artifact(
    path: Path,
    *,
    layers: int,
    heads: int,
    slots: int,
    head_dim: int,
) -> None:
    with np.load(path, allow_pickle=False) as arrays:
        expected = {
            "keys_bfloat16_bits",
            "values_bfloat16_bits",
            "valid",
            "source_positions",
            "created_at",
            "last_updated_at",
            "source_counts",
            "weight_mass",
        }
        if set(arrays.files) != expected:
            raise RuntimeError("memory-slot artifact fields changed")
        kv_shape = (layers, 1, heads, slots, head_dim)
        metadata_shape = kv_shape[:-1]
        if arrays["keys_bfloat16_bits"].shape != kv_shape:
            raise RuntimeError("memory-slot K shape changed")
        if arrays["values_bfloat16_bits"].shape != kv_shape:
            raise RuntimeError("memory-slot V shape changed")
        if any(
            arrays[name].shape != metadata_shape
            for name in expected - {"keys_bfloat16_bits", "values_bfloat16_bits"}
        ):
            raise RuntimeError("memory-slot metadata shape changed")


def fixed_accounting_decode(
    model: Any,
    cache: SignificanceCache,
    memory: SignificanceMemory | None,
    first_logits: Any,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> dict[str, Any]:
    """Decode a prediction while always reserving/appending the same fixed tail.

    Prediction tokens stop at the first EOS.  Cache appends continue through
    ``max_new_tokens - 1`` with that EOS token, solely so every condition has
    identical realized persistent-state length.  Post-EOS tokens never enter
    the prediction or correctness scorer.
    """

    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prediction_ids: list[int] = []
    trajectory_ids: list[int] = []
    logits = first_logits
    stopped = False
    eos_fill: int | None = None
    padding_appends = 0
    memory_attention_calls = 0
    memory_attention_calls_with_valid_slots = 0
    for index in range(max_new_tokens):
        argmax_id = int(logits[0, -1].float().argmax(dim=-1).item())
        trajectory_ids.append(argmax_id)
        if not stopped:
            prediction_ids.append(argmax_id)
            if argmax_id in eos_token_ids:
                stopped = True
                eos_fill = argmax_id
        if index == max_new_tokens - 1:
            break
        append_id = int(eos_fill) if stopped and eos_fill is not None else argmax_id
        if stopped and eos_fill is not None:
            padding_appends += 1
        token = torch.tensor([[append_id]], dtype=torch.long, device=logits.device)
        if memory is None:
            step = chunked_prefill(
                model,
                token,
                cache,
                chunk_size=1,
                policy=None,
                collect_logits=False,
                logits_to_keep=1,
            )
            logits = step.output.logits
        else:
            step = memory_decode_append(
                model,
                token,
                cache,
                memory,
                logits_to_keep=1,
            )
            logits = step.output.logits
            memory_attention_calls += step.attention_accounting.attention_calls
            memory_attention_calls_with_valid_slots += (
                step.attention_accounting.calls_with_valid_slots
            )
    return {
        "prediction_token_ids": prediction_ids,
        "trajectory_token_ids": trajectory_ids,
        "generated_tokens": len(prediction_ids),
        "fixed_decode_steps": max_new_tokens,
        "decode_cache_appends": max_new_tokens - 1,
        "post_eos_accounting_appends": padding_appends,
        "stop_reason": "eos_token" if stopped else "max_new_tokens",
        "memory_attention_calls": memory_attention_calls,
        "memory_attention_calls_with_valid_slots": memory_attention_calls_with_valid_slots,
        "final_logits": logits,
    }


def m0_identity_gate(
    model: Any,
    prepared: Any,
    *,
    config: dict[str, Any],
    eos_token_ids: set[int],
) -> dict[str, Any]:
    """Require bitwise hard-eviction/M0 identity before any outcome access."""

    import torch

    condition = conditions_from_config(config)[0]
    policy = _policy(condition, config=config)
    chunk_size = int(config["cache"]["compression_chunk_tokens"])
    max_new = int(config["runtime"]["max_new_tokens"])
    hard_cache = SignificanceCache(config=model.config)
    hard_prefill = chunked_prefill(
        model,
        prepared.input_ids,
        hard_cache,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(config["runtime"]["verify_query_rollback"]),
        logits_to_keep=1,
    )
    hard_decode = fixed_accounting_decode(
        model,
        hard_cache,
        None,
        hard_prefill.output.logits,
        max_new_tokens=max_new,
        eos_token_ids=eos_token_ids,
    )

    m0_cache = SignificanceCache(config=model.config)
    m0_memory = _new_memory(model, condition, config=config)
    m0_prefill = chunked_prefill_with_memory(
        model,
        prepared.input_ids,
        m0_cache,
        m0_memory,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=int(prepared.query_ids.shape[-1]),
        verify_query_rollback=bool(config["runtime"]["verify_query_rollback"]),
        logits_to_keep=1,
    )
    m0_decode = fixed_accounting_decode(
        model,
        m0_cache,
        m0_memory,
        m0_prefill.output.logits,
        max_new_tokens=max_new,
        eos_token_ids=eos_token_ids,
    )
    mismatches = _cache_identity(hard_cache, m0_cache)
    hard_logits = hard_decode.pop("final_logits").float()
    m0_logits = m0_decode.pop("final_logits").float()
    delta = (hard_logits - m0_logits).abs()
    checks = {
        "exact_final_logits": bool(torch.equal(hard_logits, m0_logits)),
        "max_abs_final_logit_delta": float(delta.max().item()),
        "exact_prediction_tokens": hard_decode["prediction_token_ids"]
        == m0_decode["prediction_token_ids"],
        "exact_fixed_decode_trajectory": hard_decode["trajectory_token_ids"]
        == m0_decode["trajectory_token_ids"],
        "exact_cache_state": not mismatches,
        "cache_mismatches": mismatches,
        "m0_memory_bytes": m0_memory.byte_ledger().total_bytes,
        "m0_memory_write_boundaries": len(m0_prefill.memory_write_reports),
    }
    passed = (
        checks["exact_final_logits"]
        and checks["exact_prediction_tokens"]
        and checks["exact_fixed_decode_trajectory"]
        and checks["exact_cache_state"]
        and checks["m0_memory_bytes"] == 0
        and checks["m0_memory_write_boundaries"] == 0
    )
    result = {
        "schema_version": "1.0",
        "status": "PASS" if passed else "FAIL",
        "case_id": prepared.case_id,
        "performed_before_outcome_access": True,
        "fixed_decode_steps": max_new,
        **checks,
    }
    del hard_cache, m0_cache, m0_memory, hard_prefill, m0_prefill, hard_logits, m0_logits, delta
    gc.collect()
    if not passed:
        raise RuntimeError(f"same-process M0 identity gate failed: {result}")
    return result


def execute_condition(
    model: Any,
    prepared: Any,
    condition: ConditionSpec,
    *,
    config: dict[str, Any],
    budget_contract: dict[str, Any],
    eos_token_ids: set[int],
) -> tuple[dict[str, Any], SignificanceCache, SignificanceMemory]:
    import psutil

    policy = _policy(condition, config=config)
    memory = _new_memory(model, condition, config=config)
    cache = SignificanceCache(config=model.config)
    input_tokens = int(prepared.input_ids.shape[-1])
    query_tokens = int(prepared.query_ids.shape[-1])
    chunk_size = int(config["cache"]["compression_chunk_tokens"])
    max_new = int(config["runtime"]["max_new_tokens"])
    boundaries = max(0, math.ceil(input_tokens / chunk_size) - 1)
    final_chunk = input_tokens - boundaries * chunk_size
    process = psutil.Process()
    rss = [process.memory_info().rss]

    prefill_started = time.perf_counter()
    prefill = chunked_prefill_with_memory(
        model,
        prepared.input_ids,
        cache,
        memory,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=query_tokens,
        verify_query_rollback=bool(config["runtime"]["verify_query_rollback"]),
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    cache.assert_aligned()
    memory.assert_valid()
    rss.append(process.memory_info().rss)

    decode_started = time.perf_counter()
    decoded = fixed_accounting_decode(
        model,
        cache,
        memory,
        prefill.output.logits,
        max_new_tokens=max_new,
        eos_token_ids=eos_token_ids,
    )
    decode_seconds = time.perf_counter() - decode_started
    decoded.pop("final_logits")
    cache.assert_aligned()
    memory.assert_valid()
    rss.append(process.memory_info().rss)

    expected_stored = policy.budget + final_chunk + max_new - 1
    expected_true = input_tokens + max_new - 1
    if cache.stored_length() != expected_stored or cache.true_length != expected_true:
        raise RuntimeError(
            f"fixed decode/cache accounting failed for {condition.condition_id}: "
            f"stored={cache.stored_length()}/{expected_stored}, true={cache.true_length}/{expected_true}"
        )
    if cache.scoring_events() != [boundaries] * len(cache.layers):
        raise RuntimeError("query-probe event count differs from the declared schedule")
    if len(prefill.compression_reports) != boundaries:
        raise RuntimeError("compression-report count differs from the declared schedule")
    if condition.num_slots and len(prefill.memory_write_reports) != boundaries:
        raise RuntimeError("memory-write boundary count differs from compression schedule")
    if not condition.num_slots and prefill.memory_write_reports:
        raise RuntimeError("M0 unexpectedly emitted memory writes")

    ordinary = _ordinary_ledger(cache)
    memory_ledger = memory.byte_ledger()
    unit_bytes = int(budget_contract["ordinary_bytes_per_budget_unit"])
    baseline_budget = int(budget_contract["baseline_ordinary_budget"])
    boundary_total = policy.budget * unit_bytes + memory_ledger.total_bytes
    baseline_boundary = baseline_budget * unit_bytes
    realized_total = ordinary["ordinary_total_bytes"] + memory_ledger.total_bytes
    baseline_realized_capacity = (baseline_budget + final_chunk + max_new - 1) * unit_bytes
    slack = baseline_realized_capacity - realized_total
    expected_slack = 0 if condition.num_slots == 0 else int(budget_contract["discrete_slack_bytes"])
    if boundary_total > baseline_boundary or slack != expected_slack:
        raise RuntimeError(
            f"total persistent-byte match failed for {condition.condition_id}: "
            f"boundary={boundary_total}/{baseline_boundary}, slack={slack}/{expected_slack}"
        )
    if ordinary["ordinary_total_bytes"] != cache.stored_length() * unit_bytes:
        raise RuntimeError("ordinary persistent byte ledger disagrees with resident length")

    reports = []
    layer_count = len(cache.layers)
    for report in prefill.compression_reports:
        reports.append(
            {
                "compressed_layers": report.layers,
                "before_entries_per_layer": report.before_entries // layer_count,
                "after_entries_per_layer": report.after_entries // layer_count,
                "evictions_per_layer": report.evictions // layer_count,
            }
        )
    result = {
        "condition": condition.condition_id,
        "memory_variant": condition.aggregation,
        "memory_slots_per_layer_kv_head": condition.num_slots,
        "ordinary_retained_history_budget": policy.budget,
        "selection_count_exponent": policy.count_exponent,
        "generated_token_ids": decoded.pop("prediction_token_ids"),
        "decode_trajectory_sha256": _canonical_sha256(decoded.pop("trajectory_token_ids")),
        **decoded,
        "query_access": _query_access(
            prepared,
            input_tokens=input_tokens,
            chunk_size=chunk_size,
        ),
        "accounting": {
            "input_tokens": input_tokens,
            "chunk_size": chunk_size,
            "compression_boundaries": boundaries,
            "final_active_chunk_entries": final_chunk,
            "resident_entries_after_decode": cache.stored_length(),
            "true_length_after_decode": cache.true_length,
            "scoring_events_per_layer": cache.scoring_events(),
            "compression_reports": reports,
            "memory_write_boundaries": len(prefill.memory_write_reports),
            "memory_layer_writes": sum(len(value) for value in prefill.memory_write_reports),
            "prefill_memory_attention": asdict(prefill.attention_accounting),
        },
        "budget": {
            "match_basis": budget_contract["match_basis"],
            "ordinary_bytes_per_budget_unit": unit_bytes,
            "m0_boundary_total_bytes": baseline_boundary,
            "condition_boundary_total_bytes": boundary_total,
            "condition_boundary_slack_bytes": baseline_boundary - boundary_total,
            "m0_fixed_decode_capacity_bytes": baseline_realized_capacity,
            "condition_realized_total_persistent_bytes": realized_total,
            "condition_realized_slack_bytes": slack,
            "matched_total_persistent_bytes": True,
        },
        "efficiency": {
            **ordinary,
            "memory_slot_bytes": memory_ledger.total_bytes,
            "memory_kv_bytes": memory_ledger.kv_bytes,
            "memory_metadata_bytes": memory_ledger.metadata_bytes,
            "architecture_total_persistent_bytes": realized_total,
            "total_persistent_bytes": realized_total,
            "sampled_peak_rss_bytes": max(rss),
            "peak_rss_bytes": max(rss),
            "peak_vram_bytes": 0,
            "wall_seconds_for_audit_only": prefill_seconds + decode_seconds,
        },
        "memory_state_sha256": memory.state_sha256(),
        "timing": {
            "prefill_wall_seconds_for_audit_only": prefill_seconds,
            "decode_wall_seconds_for_audit_only": decode_seconds,
            "latency_claim_allowed": False,
        },
    }
    return result, cache, memory


def validate_matching(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    failures: list[str] = []
    for case_id, case_rows in sorted(by_case.items()):
        if [str(row["condition"]) for row in case_rows] != list(CONDITION_IDS):
            failures.append(f"{case_id}:condition_order")
            continue
        reference_access = case_rows[0]["query_access"]
        if any(row["query_access"] != reference_access for row in case_rows[1:]):
            failures.append(f"{case_id}:query_access")
        if len({int(row["decode_cache_appends"]) for row in case_rows}) != 1:
            failures.append(f"{case_id}:fixed_decode_appends")
        baseline = case_rows[0]
        baseline_total = int(baseline["budget"]["condition_realized_total_persistent_bytes"])
        for row in case_rows:
            if not bool(row["budget"]["matched_total_persistent_bytes"]):
                failures.append(f"{case_id}:{row['condition']}:budget_flag")
            if int(row["budget"]["condition_realized_total_persistent_bytes"]) > baseline_total:
                failures.append(f"{case_id}:{row['condition']}:total_bytes")
    return {
        "schema_version": "1.0",
        "status": "PASS" if not failures else "FAIL",
        "case_count": len(by_case),
        "conditions": list(CONDITION_IDS),
        "query_access_equal_within_case": not any("query_access" in value for value in failures),
        "fixed_decode_appends_equal_within_case": not any(
            "fixed_decode_appends" in value for value in failures
        ),
        "all_condition_total_bytes_at_or_below_m0": not any(
            "total_bytes" in value or "budget_flag" in value for value in failures
        ),
        "failures": failures,
    }


def paired_statistics(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    indexed = {(str(row["case_id"]), str(row["condition"])): row for row in rows}
    case_ids = [str(value) for value in config["selection"]["selected_case_ids"]]
    cluster_ids = [str(indexed[(case_id, CONDITION_IDS[0])]["scenario_id"]) for case_id in case_ids]
    analysis = dict(config["analysis"])
    comparisons: dict[str, Any] = {}
    comparison_index = 0
    for left_index, left in enumerate(CONDITION_IDS):
        for right in CONDITION_IDS[left_index + 1 :]:
            endpoint = clustered_paired_analysis(
                cluster_ids,
                [
                    float(indexed[(case_id, left)]["evidence"]["layer_fact_exact_completion_rate"])
                    for case_id in case_ids
                ],
                [
                    float(indexed[(case_id, right)]["evidence"]["layer_fact_exact_completion_rate"])
                    for case_id in case_ids
                ],
                minimum_clusters=int(analysis["minimum_independent_clusters"]),
                confidence=float(analysis["confidence"]),
                bootstrap_samples=int(analysis["bootstrap_samples"]),
                randomization_samples=int(analysis["randomization_samples"]),
                seed=int(config["runtime"]["seed"]) + comparison_index * 2,
            )
            answer = clustered_paired_analysis(
                cluster_ids,
                [float(bool(indexed[(case_id, left)]["correct"])) for case_id in case_ids],
                [float(bool(indexed[(case_id, right)]["correct"])) for case_id in case_ids],
                minimum_clusters=int(analysis["minimum_independent_clusters"]),
                confidence=float(analysis["confidence"]),
                bootstrap_samples=int(analysis["bootstrap_samples"]),
                randomization_samples=int(analysis["randomization_samples"]),
                seed=int(config["runtime"]["seed"]) + comparison_index * 2 + 1,
            )
            name = f"{right}_minus_{left}"
            comparisons[f"{name}__endpoint"] = asdict(endpoint)
            comparisons[f"{name}__answer_correct"] = asdict(answer)
            comparison_index += 1
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_LOCAL_NOT_REGISTERED",
        "independence_unit": "whole_cluster_id",
        "cluster_count": len(set(cluster_ids)),
        "condition_pair_count": comparison_index,
        "primary_endpoint": "layer_fact_exact_completion_rate",
        "multiplicity_adjustment": "none_exploratory",
        "comparisons": comparisons,
        "interpretation_limit": (
            "Case IDs and endpoint v1 were frozen elsewhere, but memory conditions, byte-matched "
            "budgets, and all contrasts were not registered; intervals and p-values are exploratory."
        ),
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = [float(row["evidence"]["layer_fact_exact_completion_rate"]) for row in rows]
    totals = [int(row["efficiency"]["architecture_total_persistent_bytes"]) for row in rows]
    slacks = [int(row["budget"]["condition_realized_slack_bytes"]) for row in rows]
    return {
        "clusters": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "mean_layer_fact_exact_completion_rate": float(np.mean(endpoints)),
        "strict_evidence_survival": sum(
            bool(row["evidence"]["all_required_survived"]) for row in rows
        ),
        "ordinary_retained_history_budget": int(rows[0]["ordinary_retained_history_budget"]),
        "memory_slots_per_layer_kv_head": int(rows[0]["memory_slots_per_layer_kv_head"]),
        "persistent_bytes_minimum": min(totals),
        "persistent_bytes_maximum": max(totals),
        "discrete_slack_bytes": sorted(set(slacks)),
    }


def _checkpoint_result(
    path: Path,
    *,
    fingerprint: str,
    output: Path,
    layers: int,
    heads: int,
    slots: int,
    head_dim: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if str(checkpoint.get("plan_fingerprint")) != fingerprint:
        raise RuntimeError(f"cell checkpoint plan changed: {path.name}")
    result = checkpoint.get("result")
    if not isinstance(result, dict) or _canonical_sha256(result) != checkpoint.get("result_sha256"):
        raise RuntimeError(f"cell checkpoint result hash changed: {path.name}")
    resident_path = output / str(result["resident_positions"])
    slots_path = output / str(result["memory_slots"])
    if sha256_file(resident_path) != str(result["resident_positions_sha256"]):
        raise RuntimeError(f"cell resident-position artifact changed: {path.name}")
    if sha256_file(slots_path) != str(result["memory_slots_sha256"]):
        raise RuntimeError(f"cell memory-slot artifact changed: {path.name}")
    validate_memory_slot_artifact(
        slots_path,
        layers=layers,
        heads=heads,
        slots=slots,
        head_dim=head_dim,
    )
    return result


def run(
    config_path: Path,
    *,
    max_new_cells: int | None = None,
) -> dict[str, Any]:
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = config_path.resolve(strict=True)
    config = _load(config_path)
    if str(config["experiment"]["classification"]) != "EXPLORATORY_NOT_REGISTERED":
        raise RuntimeError("memory quality screen must remain exploratory and unregistered")
    if bool(config["protocol"].get("successive_halving_used")):
        raise RuntimeError("canonical v1 is the full 72-cell grid, not successive halving")
    conditions = conditions_from_config(config)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    resolved = {name: _resolve(base, value) for name, value in paths.items() if name != "base"}
    output = resolved["output_root"]
    if (output / "manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite completed memory quality screen: {output}")
    output.mkdir(parents=True, exist_ok=True)
    positions_dir = output / "resident_positions"
    slots_dir = output / "memory_slots"
    checkpoints_dir = output / "cell_checkpoints"
    positions_dir.mkdir(exist_ok=True)
    slots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    frozen = dict(config["frozen_case_source"])
    registered_config_path = _resolve(base, frozen["registered_config"])
    if sha256_file(registered_config_path) != str(frozen["registered_config_sha256"]):
        raise RuntimeError("registered source config hash changed")
    registered = _load(registered_config_path)
    selected_ids = [str(value) for value in config["selection"]["selected_case_ids"]]
    if selected_ids != [str(value) for value in registered["selection"]["selected_case_ids"]]:
        raise RuntimeError("memory screen case IDs differ from the frozen registered set")
    if len(selected_ids) != 12 or len(set(selected_ids)) != 12:
        raise RuntimeError("memory screen requires exactly 12 unique frozen cases")
    source_hashes = dict(frozen["source_hashes"])
    observed_preoutcome = {
        "capability_cases": sha256_file(resolved["capability_cases"]),
        "inference": sha256_file(resolved["inference"]),
    }
    if observed_preoutcome != {name: source_hashes[name] for name in observed_preoutcome}:
        raise RuntimeError("frozen prompt/capability source views changed")
    if str(config["endpoint"]["version"]) != EVIDENCE_RETENTION_ENDPOINT_VERSION:
        raise RuntimeError("frozen endpoint version changed")
    if str(config["endpoint"]["definition_sha256"]) != str(
        registered["endpoint"]["definition_sha256"]
    ):
        raise RuntimeError("frozen endpoint definition hash changed")

    capability_rows = {str(row["case_id"]): row for row in _jsonl(resolved["capability_cases"])}
    families = {str(value) for value in config["selection"]["families"]}
    for case_id in selected_ids:
        capability_row = capability_rows.get(case_id)
        if capability_row is None or not bool(capability_row.get("correct")):
            raise RuntimeError(f"selected case is not frozen B0-correct: {case_id}")
        if str(capability_row.get("task_family")) not in families:
            raise RuntimeError(f"selected case has an undeclared family: {case_id}")
    inference = {
        str(row["case_id"]): row
        for row in _jsonl(resolved["inference"])
        if str(row["case_id"]) in set(selected_ids)
    }
    if set(inference) != set(selected_ids):
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
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        .to("cpu")
        .eval()
    )
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
    head_dim = int(model.config.head_dim)
    budget_contract = solve_total_budget_contract(
        config,
        num_layers=layers,
        batch_size=1,
        num_kv_heads=heads,
        head_dim=head_dim,
        kv_dtype=torch.bfloat16,
    )
    atomic_write_json(output / "total_byte_budget.json", budget_contract)

    mini = _module("memory_quality_mini", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module(
        "memory_quality_capability", LAB_ROOT / "scripts" / "run_halo_capability.py"
    )
    prepared_cases: dict[str, Any] = {}
    for case_id in selected_ids:
        row = inference[case_id]
        input_ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        query_ids = tokenizer(str(row["question"]), add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
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
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))

    identity = m0_identity_gate(
        model,
        prepared_cases[selected_ids[0]],
        config=config,
        eos_token_ids=eos_ids,
    )
    atomic_write_json(output / "m0_identity_gate.json", identity)

    config_fingerprint = sha256_file(config_path)
    expected_cells = [(case_id, condition) for case_id in selected_ids for condition in conditions]
    unscored: list[dict[str, Any]] = []
    new_cells = 0
    for case_id, condition in expected_cells:
        prepared = prepared_cases[case_id]
        inference_row = inference[case_id]
        plan = {
            "config_sha256": config_fingerprint,
            "case_id": case_id,
            "cluster_id": inference_row["cluster_id"],
            "condition": asdict(condition),
            "input_token_sha256": prepared.input_token_sha256,
            "query_token_sha256": prepared.query_token_sha256,
            "budget_contract_sha256": _canonical_sha256(budget_contract),
        }
        fingerprint = _canonical_sha256(plan)
        stem = f"{case_id}__{condition.condition_id}"
        checkpoint_path = checkpoints_dir / f"{stem}.json"
        existing = _checkpoint_result(
            checkpoint_path,
            fingerprint=fingerprint,
            output=output,
            layers=layers,
            heads=heads,
            slots=condition.num_slots,
            head_dim=head_dim,
        )
        if existing is not None:
            unscored.append(existing)
            continue
        if max_new_cells is not None and new_cells >= max_new_cells:
            continue
        result, cache, memory = execute_condition(
            model,
            prepared,
            condition,
            config=config,
            budget_contract=budget_contract,
            eos_token_ids=eos_ids,
        )
        position_name = f"{stem}.npz"
        position_path = positions_dir / position_name
        position_sha = save_resident_positions(position_path, cache)
        slots_name = f"{stem}.npz"
        slots_path = slots_dir / slots_name
        slots_sha = save_memory_slots(slots_path, memory)
        validate_memory_slot_artifact(
            slots_path,
            layers=layers,
            heads=heads,
            slots=condition.num_slots,
            head_dim=head_dim,
        )
        prediction = tokenizer.decode(
            result.pop("generated_token_ids"),
            skip_special_tokens=True,
        ).strip()
        row = {
            "case_id": case_id,
            "cluster_id": inference_row["cluster_id"],
            "family": inference_row["family"],
            "prediction": prediction,
            "resident_positions": f"resident_positions/{position_name}",
            "resident_positions_sha256": position_sha,
            "memory_slots": f"memory_slots/{slots_name}",
            "memory_slots_sha256": slots_sha,
            **result,
        }
        checkpoint = {
            "schema_version": "1.0",
            "plan_fingerprint": fingerprint,
            "result_sha256": _canonical_sha256(row),
            "result": row,
        }
        atomic_write_json(checkpoint_path, checkpoint)
        unscored.append(row)
        new_cells += 1
        del cache, memory
        gc.collect()

    completed_keys = {(str(row["case_id"]), str(row["condition"])) for row in unscored}
    complete = len(completed_keys) == len(expected_cells)
    run_state = {
        "schema_version": "1.0",
        "status": "INFERENCE_COMPLETE_OUTCOMES_SEALED" if complete else "RUNNING",
        "completed_cells": len(completed_keys),
        "expected_cells": len(expected_cells),
        "new_cells_this_invocation": new_cells,
        "identity_gate": identity["status"],
        "outcome_accessed": False,
    }
    atomic_write_json(output / "run_state.json", run_state)
    if not complete:
        return run_state

    ordered = {(str(row["case_id"]), str(row["condition"])): row for row in unscored}
    unscored = [ordered[(case_id, condition.condition_id)] for case_id, condition in expected_cells]
    matching = validate_matching(unscored)
    if matching["status"] != "PASS":
        raise RuntimeError(f"matching audit failed: {matching}")
    atomic_write_json(output / "matching_audit.json", matching)
    predictions_path = output / "predictions.unscored.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in unscored),
        encoding="utf-8",
    )
    if len(unscored) != int(config["protocol"]["total_cells"]):
        raise RuntimeError("memory screen did not durably produce exactly 72 cells")

    # This is the first evaluation-view access. Every prediction, resident
    # position tensor, and exact slot-state tensor is already durable and hashed.
    if sha256_file(resolved["evaluation"]) != str(source_hashes["evaluation"]):
        raise RuntimeError("frozen evaluation source view changed")
    evaluation = {str(row["case_id"]): row for row in _jsonl(resolved["evaluation"])}
    scored: list[dict[str, Any]] = []
    for row in unscored:
        key = evaluation[str(row["case_id"])]
        position_path = output / str(row["resident_positions"])
        slots_path = output / str(row["memory_slots"])
        if sha256_file(position_path) != str(row["resident_positions_sha256"]):
            raise RuntimeError("resident-position artifact changed before outcome scoring")
        if sha256_file(slots_path) != str(row["memory_slots_sha256"]):
            raise RuntimeError("memory-slot artifact changed before outcome scoring")
        validate_memory_slot_artifact(
            slots_path,
            layers=layers,
            heads=heads,
            slots=int(row["memory_slots_per_layer_kv_head"]),
            head_dim=head_dim,
        )
        resident = load_resident_positions(
            position_path,
            layer_count=layers,
            head_count=heads,
        )
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
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
                "ordinary_retained_history_budget": row["ordinary_retained_history_budget"],
                "memory_slots_per_layer_kv_head": row["memory_slots_per_layer_kv_head"],
                "memory_variant": row["memory_variant"],
                "query_access": row["query_access"],
                "budget": row["budget"],
                "accounting": row["accounting"],
                "efficiency": row["efficiency"],
                "generated_tokens": row["generated_tokens"],
                "fixed_decode_steps": row["fixed_decode_steps"],
                "decode_cache_appends": row["decode_cache_appends"],
                "post_eos_accounting_appends": row["post_eos_accounting_appends"],
                "stop_reason": row["stop_reason"],
                "failure_labels": compressed_retention_failure_labels(
                    correct=correct,
                    evidence=evidence,
                ),
                "failure_label_scope": "descriptive_final_retained_state_not_causal_attribution",
                "trace_ids": [row["resident_positions"], row["memory_slots"]],
                "resident_positions_sha256": row["resident_positions_sha256"],
                "memory_slots_sha256": row["memory_slots_sha256"],
                "memory_state_sha256": row["memory_state_sha256"],
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
        condition: _condition_summary([row for row in scored if row["condition"] == condition])
        for condition in CONDITION_IDS
    }
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_LOCAL_NOT_REGISTERED",
        "selected_frozen_clusters": len(selected_ids),
        "cells": len(scored),
        "conditions": list(CONDITION_IDS),
        "successive_halving_used": False,
        "m0_identity_gate": identity,
        "matching_audit": matching,
        "total_byte_budget": budget_contract,
        "by_condition": by_condition,
        "whole_cluster_paired_contrasts": statistics["comparisons"],
        "claims_supported": [
            "all six closed-loop conditions were measured on the same 12 frozen Halo clusters",
            "every M4 arm stayed below the M0 total persistent-state capacity with disclosed integer slack",
            "exact ordinary resident positions and exact memory-slot tensor bits were durable before outcome access",
        ],
        "claims_not_supported": [
            "registered, confirmatory, or population-level evidence for memory slots",
            "a quality improvement attributable to normalized significance without multiplicity-aware confirmation",
            "memory or latency saving relative to a deployment implementation",
            "latency comparison; fixed post-EOS accounting appends intentionally equalize persistent state",
        ],
    }
    atomic_write_json(output / "summary.json", summary)
    resolved_config = output / "resolved_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
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
            "architecture_variant": "Experiment D fixed significance memory quality screen",
            "benchmark": "halo_endpoint_validation_v2_frozen_case_ids",
            "seed": int(config["runtime"]["seed"]),
            "query_conditioned": True,
            "classification": "EXPLORATORY_LOCAL_NOT_REGISTERED",
            "registration_reuse_scope": "case_ids_and_endpoint_only_new_conditions_not_registered",
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
            **budget_contract,
            "matched": True,
            "fixed_decode_steps": int(config["runtime"]["max_new_tokens"]),
            "latency_claim": False,
        },
        "artifacts": {
            "m0_identity_gate": "m0_identity_gate.json",
            "total_byte_budget": "total_byte_budget.json",
            "matching_audit": "matching_audit.json",
            "predictions": "predictions.unscored.jsonl",
            "cases": "cases.jsonl",
            "statistics": "statistics.json",
            "summary": "summary.json",
            "resident_positions": "resident_positions/",
            "memory_slots": "memory_slots/",
            "cell_checkpoints": "cell_checkpoints/",
            "run_state": "run_state.json",
            "resolved_config": "resolved_config.yaml",
        },
        "notes": summary["claims_not_supported"],
    }
    atomic_write_json(output / "manifest.json", manifest)
    run_state.update(
        {
            "status": "COMPLETE",
            "outcome_accessed": True,
            "evaluation_sha256": source_hashes["evaluation"],
            "completed_at": manifest["completed_at"],
        }
    )
    atomic_write_json(output / "run_state.json", run_state)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_significance_memory_quality_screen.yaml",
    )
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args(argv)
    result = run(args.config, max_new_cells=args.max_new_cells)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"PASS", "RUNNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
