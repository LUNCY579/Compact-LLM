#!/usr/bin/env python3
"""Run restartable Halo B2/B3 compression-round causal controls.

Inference checkpoints are written after every model cell.  The answer/evidence
view is not opened until every planned cell is durable, so interrupted runs can
resume without leaking labels into unfinished inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
from rineya_significance_lab.evaluation.evidence_endpoints import (  # noqa: E402
    score_evidence_retention_endpoint,
)
from rineya_significance_lab.evaluation.failure_taxonomy import (  # noqa: E402
    compressed_retention_failure_labels,
)
from rineya_significance_lab.evaluation.schedule_interaction import (  # noqa: E402
    clustered_method_round_interaction,
    validate_matched_schedule_rows,
)
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    git_snapshot,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.runtime.cache import SignificanceCache  # noqa: E402
from rineya_significance_lab.runtime.scheduled_prefill import (  # noqa: E402
    FixedTerminalSchedule,
    build_fixed_terminal_schedule,
    fixed_terminal_prefill,
)

METHODS = ("B2", "B3")
CELL_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _module(name: str, path: Path):
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
        raise ValueError("schedule-control config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_positions(path: Path, positions: Sequence[np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            **{f"layer_{index:03d}": value for index, value in enumerate(positions)},
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _resolve_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    return {
        key: (base / str(value)).resolve()
        for key, value in paths.items()
        if key != "base"
    }


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject model drift and any hidden asymmetry in the causal design."""

    conditions = config.get("conditions")
    if not isinstance(conditions, list) or [row.get("id") for row in conditions] != list(METHODS):
        raise ValueError("conditions must be ordered exactly as B2 then B3")
    expected = {"B2": 0.0, "B3": 1.0}
    for row in conditions:
        method = str(row.get("id"))
        if row.get("policy") != "significance" or float(row.get("exponent", -1)) != expected[method]:
            raise ValueError(f"{method} policy/exponent is not the registered raw/normalized contrast")
        if not bool(row.get("query_conditioned", False)):
            raise ValueError(f"{method} must receive the matched query probes")

    runtime = dict(config.get("runtime", {}))
    required_runtime = {
        "device": "cpu",
        "dtype": "bfloat16",
        "decoding": "manual_greedy",
        "attention_implementation": "sdpa",
    }
    for key, expected_value in required_runtime.items():
        if str(runtime.get(key)).lower() != expected_value:
            raise ValueError(f"runtime.{key} must be {expected_value!r}")
    if not bool(runtime.get("deterministic")) or int(runtime.get("batch_size", 0)) != 1:
        raise ValueError("schedule control requires deterministic batch-size-one execution")
    if int(runtime.get("max_new_tokens", 0)) <= 0:
        raise ValueError("runtime.max_new_tokens must be positive")

    model = dict(config.get("model", {}))
    core_id = str(model.get("core_id", ""))
    allowed = [str(value) for value in model.get("allowed_core_ids", [])]
    if allowed != [core_id] or ".halo." not in core_id.lower():
        raise ValueError("only the declared published Halo core is permitted")
    if model.get("expected_source_base") != "Qwen3-0.6B" or bool(model.get("trust_remote_code", True)):
        raise ValueError("Halo must remain a trust-remote-code-disabled Qwen3-0.6B core")

    cache = dict(config.get("cache", {}))
    budget = int(cache.get("retained_history_budget", 0))
    sinks = int(cache.get("sink_tokens", -1))
    recent = int(cache.get("recent_tokens", -1))
    if budget <= 0 or sinks < 0 or recent < 0 or sinks + recent >= budget:
        raise ValueError("cache budget must leave at least one heavy-selection slot")
    schedule = dict(config.get("schedule", {}))
    chunks = [int(value) for value in schedule.get("prefix_chunk_tokens", [])]
    terminal = int(schedule.get("terminal_active_tokens", 0))
    if terminal <= 0 or len(chunks) < 2 or chunks != sorted(set(chunks)):
        raise ValueError("schedule needs a positive terminal suffix and sorted unique chunk sizes")
    if int(schedule.get("minimum_distinct_round_counts", 0)) < 2:
        raise ValueError("schedule.minimum_distinct_round_counts must be at least two")
    query = dict(config.get("query", {}))
    if query.get("mode") != "conditioned":
        raise ValueError("query.mode must be conditioned")
    if query.get("probe_schedule") != "every_prefix_chunk_boundary_fixed_terminal_suffix":
        raise ValueError("query probe schedule must match fixed-terminal execution")
    pooling = dict(dict(config.get("significance", {})).get("score_pooling", {}))
    if bool(pooling.get("enabled")) or int(pooling.get("width", 0)) != 1:
        raise ValueError("score pooling must remain disabled")
    checkpointing = dict(config.get("checkpointing", {}))
    if not bool(checkpointing.get("resume")) or not bool(checkpointing.get("flush_every_cell")):
        raise ValueError("restartability requires resume and flush_every_cell")


def _selected_inference_rows(
    config: Mapping[str, Any],
    inference_rows: Sequence[Mapping[str, Any]],
    capability_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selection = dict(config["selection"])
    families = {str(value) for value in selection["families"]}
    if not families or "R2" in families:
        raise ValueError("active schedule control must use eligible R1/R3/R4 families, not R2")
    maximum = int(selection["maximum_eligible_clusters_per_family"])
    minimum = int(selection["minimum_total_eligible_clusters"])
    eligible: defaultdict[str, list[str]] = defaultdict(list)
    for row in capability_rows:
        family = str(row.get("task_family"))
        if row.get("correct") and family in families:
            eligible[family].append(str(row["case_id"]))
    selected_ids = {
        case_id
        for family in sorted(families)
        for case_id in sorted(eligible[family])[:maximum]
    }
    selected = [dict(row) for row in inference_rows if str(row["case_id"]) in selected_ids]
    selected.sort(key=lambda row: str(row["case_id"]))
    if len(selected) < minimum or len({str(row["cluster_id"]) for row in selected}) != len(selected):
        raise RuntimeError("schedule control lacks the configured independent B0-eligible clusters")
    if {str(row["case_id"]) for row in selected} != selected_ids:
        raise RuntimeError("selected capability cases are missing from the inference view")
    return selected


def build_cell_plans(
    config: Mapping[str, Any],
    inference_rows: Sequence[Mapping[str, Any]],
    capability_rows: Sequence[Mapping[str, Any]],
    *,
    model_identity_sha256: str,
) -> list[dict[str, Any]]:
    """Create the complete deterministic cell matrix before model loading."""

    validate_config(config)
    selected = _selected_inference_rows(config, inference_rows, capability_rows)
    cache = dict(config["cache"])
    schedule_config = dict(config["schedule"])
    runtime = dict(config["runtime"])
    conditions = {str(row["id"]): dict(row) for row in config["conditions"]}
    terminal = int(schedule_config["terminal_active_tokens"])
    chunks = [int(value) for value in schedule_config["prefix_chunk_tokens"]]
    plans: list[dict[str, Any]] = []
    for row in selected:
        input_tokens = int(row["token_count"])
        if input_tokens - terminal < int(cache["retained_history_budget"]):
            raise ValueError(f"case {row['case_id']} is too short to saturate the history budget")
        schedules = [
            build_fixed_terminal_schedule(
                schedule_id=f"c{chunk:04d}",
                input_tokens=input_tokens,
                prefix_chunk_tokens=chunk,
                terminal_active_tokens=terminal,
                retained_history_budget=int(cache["retained_history_budget"]),
            )
            for chunk in chunks
        ]
        distinct_rounds = {schedule.compression_rounds for schedule in schedules}
        if len(distinct_rounds) < int(schedule_config["minimum_distinct_round_counts"]):
            raise ValueError(f"case {row['case_id']} has too few distinct realized round counts")
        held_constant = {
            "prompt_sha256": str(row["prompt_sha256"]),
            "input_token_sha256": str(row["token_ids_sha256"]),
            "query_text_sha256": hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest(),
            "absolute_position_ids_sha256": _canonical_sha256(list(range(input_tokens))),
            "model_identity_sha256": model_identity_sha256,
            "retained_history_budget": int(cache["retained_history_budget"]),
            "sink_tokens": int(cache["sink_tokens"]),
            "recent_tokens": int(cache["recent_tokens"]),
            "terminal_active_tokens": terminal,
            "decoding": str(runtime["decoding"]),
            "device": str(runtime["device"]),
            "dtype": str(runtime["dtype"]),
            "attention_implementation": str(runtime["attention_implementation"]),
            "max_new_tokens": int(runtime["max_new_tokens"]),
            "seed": int(runtime["seed"]),
        }
        for schedule in schedules:
            for method in METHODS:
                condition = conditions[method]
                cell_id = f"{row['case_id']}__{schedule.schedule_id}__{method}"
                if not CELL_ID.fullmatch(cell_id):
                    raise ValueError(f"unsafe cell identifier: {cell_id!r}")
                plan = {
                    "cell_id": cell_id,
                    "case_id": str(row["case_id"]),
                    "cluster_id": str(row["cluster_id"]),
                    "family": str(row["family"]),
                    "method": method,
                    "count_exponent": float(condition["exponent"]),
                    "schedule_id": schedule.schedule_id,
                    "schedule": schedule.to_dict(),
                    "held_constant": held_constant,
                }
                plan["plan_fingerprint"] = _canonical_sha256(plan)
                plans.append(plan)
    return plans


def run_checkpointed_cells(
    plans: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    max_new_cells: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Resume verified cell checkpoints and atomically add at most ``max_new_cells``."""

    if max_new_cells is not None and max_new_cells < 0:
        raise ValueError("max_new_cells must be non-negative or null")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    newly_executed = 0
    for plan in plans:
        cell_id = str(plan["cell_id"])
        path = checkpoint_dir / f"{cell_id}.json"
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint.get("cell_id") != cell_id:
                raise RuntimeError(f"checkpoint cell ID mismatch: {path}")
            if checkpoint.get("plan_fingerprint") != plan["plan_fingerprint"]:
                raise RuntimeError(f"checkpoint plan drift detected: {path}")
            result = checkpoint.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"checkpoint result is invalid: {path}")
            if checkpoint.get("result_sha256") != _canonical_sha256(result):
                raise RuntimeError(f"checkpoint result hash mismatch: {path}")
            rows.append(result)
            continue
        if max_new_cells is not None and newly_executed >= max_new_cells:
            continue
        result = dict(executor(plan))
        if result.get("cell_id") != cell_id:
            raise RuntimeError(f"executor returned the wrong cell ID for {cell_id}")
        atomic_write_json(
            path,
            {
                "schema_version": "1.0",
                "cell_id": cell_id,
                "plan_fingerprint": plan["plan_fingerprint"],
                "completed_at": utc_now(),
                "result_sha256": _canonical_sha256(result),
                "result": result,
            },
        )
        rows.append(result)
        newly_executed += 1
    rows.sort(key=lambda row: str(row["cell_id"]))
    return rows, newly_executed


def _schedule_from_plan(plan: Mapping[str, Any]) -> FixedTerminalSchedule:
    row = dict(plan["schedule"])
    return FixedTerminalSchedule(
        schedule_id=str(row["schedule_id"]),
        input_tokens=int(row["input_tokens"]),
        prefix_tokens=int(row["prefix_tokens"]),
        prefix_chunk_tokens=int(row["prefix_chunk_tokens"]),
        terminal_active_tokens=int(row["terminal_active_tokens"]),
        chunk_sizes=tuple(int(value) for value in row["chunk_sizes"]),
        boundary_positions=tuple(int(value) for value in row["boundary_positions"]),
        compression_rounds=int(row["compression_rounds"]),
        retained_history_budget=int(row["retained_history_budget"]),
        final_resident_entries=int(row["final_resident_entries"]),
    )


def _cache_bytes(cache: SignificanceCache) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return int(total)


def _execute_model_cell(
    plan: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    input_ids_by_case: Mapping[str, Any],
    query_ids_by_case: Mapping[str, Any],
    config: Mapping[str, Any],
    mini: Any,
    positions_dir: Path,
) -> dict[str, Any]:
    import psutil
    import torch

    seed = int(dict(config["runtime"])["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    case_id = str(plan["case_id"])
    input_ids = input_ids_by_case[case_id]
    query_ids = query_ids_by_case[case_id]
    schedule = _schedule_from_plan(plan)
    condition = mini.Condition(
        str(plan["method"]), "significance", float(plan["count_exponent"]), True
    )
    cache_config = dict(config["cache"])
    policy = mini._policy_for(condition, cache_config, full_cache_budget=schedule.input_tokens)
    cache = SignificanceCache(config=model.config)
    process = psutil.Process()
    rss = [process.memory_info().rss]
    started = time.perf_counter()
    prefill_started = time.perf_counter()
    prefill = fixed_terminal_prefill(
        model,
        input_ids,
        cache,
        schedule=schedule,
        policy=policy,
        query_ids=query_ids,
        verify_query_rollback=bool(dict(config["runtime"])["verify_query_rollback"]),
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    rss.append(process.memory_info().rss)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    decode_started = time.perf_counter()
    decoded = mini._manual_greedy_decode(
        model,
        cache,
        prefill.output.logits,
        max_new_tokens=int(dict(config["runtime"])["max_new_tokens"]),
        eos_token_ids=eos_ids,
    )
    decode_seconds = time.perf_counter() - decode_started
    cache.assert_aligned()
    rss.append(process.memory_info().rss)
    resident_after_decode = cache.stored_length()
    expected_after_decode = schedule.final_resident_entries + int(decoded["decode_cache_appends"])
    if resident_after_decode != expected_after_decode:
        raise RuntimeError("decode cache accounting differs from the fixed schedule")
    reports: list[dict[str, int]] = []
    layers = len(cache.layers)
    for report in prefill.compression_reports:
        if report.before_entries % layers or report.after_entries % layers or report.evictions % layers:
            raise RuntimeError("compression report does not divide evenly across model layers")
        reports.append(
            {
                "compressed_layers": report.layers,
                "before_entries_per_layer": report.before_entries // layers,
                "after_entries_per_layer": report.after_entries // layers,
                "evictions_per_layer": report.evictions // layers,
            }
        )
    positions = [
        layer.state.positions[0].detach().cpu().numpy()
        for layer in cache.layers
        if layer.state is not None
    ]
    positions_relative = Path(f"{plan['cell_id']}.npz")
    positions_path = positions_dir / positions_relative
    positions_sha256 = _atomic_save_positions(positions_path, positions)
    kv_bytes = _cache_bytes(cache)
    matrix_bytes = cache.matrix_bytes()
    metadata_bytes = cache.metadata_bytes()
    wall_seconds = time.perf_counter() - started
    return {
        **{
            key: value
            for key, value in plan.items()
            if key not in {"plan_fingerprint", "held_constant"}
        },
        "held_constant": {
            **dict(plan["held_constant"]),
            "query_token_sha256": mini._token_sha256(query_ids),
        },
        "plan_fingerprint": plan["plan_fingerprint"],
        "prediction": tokenizer.decode(decoded["token_ids"], skip_special_tokens=True).strip(),
        "generated_tokens": decoded["generated_tokens"],
        "stop_reason": decoded["stop_reason"],
        "resident_positions": f"resident_positions/{positions_relative.as_posix()}",
        "resident_positions_sha256": positions_sha256,
        "query_access": {
            "query_tokens_per_probe": int(query_ids.shape[-1]),
            "query_probes": schedule.compression_rounds,
            "query_probe_token_evaluations": prefill.query_probe_token_evaluations,
            "scoring_events_per_layer": list(prefill.scoring_events_per_layer),
        },
        "accounting": {
            "boundary_accounting": list(prefill.boundary_accounting),
            "compression_reports": reports,
            "resident_entries_after_prefill": schedule.final_resident_entries,
            "decode_cache_appends": int(decoded["decode_cache_appends"]),
            "resident_entries_after_decode": resident_after_decode,
            "max_transient_resident_entries": prefill.max_transient_resident_entries,
        },
        "efficiency": {
            "kv_only_bytes": kv_bytes,
            "significance_matrix_bytes": matrix_bytes,
            "alignment_metadata_bytes": metadata_bytes,
            "architecture_total_persistent_bytes": kv_bytes + matrix_bytes + metadata_bytes,
            "total_persistent_bytes": kv_bytes + matrix_bytes + metadata_bytes,
            "sampled_peak_rss_bytes": max(rss),
            "peak_rss_bytes": max(rss),
            "peak_vram_bytes": 0,
            "prefill_wall_seconds": prefill_seconds,
            "decode_wall_seconds": decode_seconds,
            "wall_seconds": wall_seconds,
        },
    }


def _score_rows(
    unscored: Sequence[Mapping[str, Any]],
    evaluation_path: Path,
    output: Path,
    capability: Any,
) -> list[dict[str, Any]]:
    evaluation = {str(row["case_id"]): row for row in _jsonl(evaluation_path)}
    scored: list[dict[str, Any]] = []
    for raw in unscored:
        row = dict(raw)
        key = evaluation[str(row["case_id"])]
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
        positions_path = output / str(row["resident_positions"])
        if sha256_file(positions_path) != row["resident_positions_sha256"]:
            raise RuntimeError(f"resident-position checkpoint hash mismatch: {positions_path}")
        with np.load(positions_path, allow_pickle=False) as arrays:
            resident = [arrays[name].tolist() for name in sorted(arrays.files)]
        retention = score_posthoc_evidence_retention(annotation, resident)
        evidence = retention.to_dict()
        frozen_endpoint = score_evidence_retention_endpoint(retention).to_dict()
        accepted = [str(value) for value in key["accepted_answers"]]
        correct = capability._task_correct(str(row["family"]), str(row["prediction"]), accepted)
        held_constant = {
            **dict(row["held_constant"]),
            "answer_key_sha256": _canonical_sha256(accepted),
            "evidence_annotation_sha256": _canonical_sha256(key["evidence_annotation"]),
        }
        scored.append(
            {
                "schema_version": "1.0",
                "run_id": output.name,
                "case_id": row["case_id"],
                "scenario_id": row["cluster_id"],
                "cluster_id": row["cluster_id"],
                "condition": f"{row['method']}@{row['schedule_id']}",
                "method": row["method"],
                "count_exponent": row["count_exponent"],
                "schedule_id": row["schedule_id"],
                "schedule": row["schedule"],
                "held_constant": held_constant,
                "query_access": row["query_access"],
                "task_family": row["family"],
                "generator_seed": None,
                "prompt_sha256": annotation.prompt_sha256,
                "context_tokens": annotation.token_count,
                "evidence_depth": 0.5,
                "chunk_size": row["schedule"]["prefix_chunk_tokens"],
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
                    **frozen_endpoint,
                    "fraction_layers_with_complete_required_token_union": evidence[
                        "fraction_of_evaluated_layers_with_all_required_fact_tokens_available"
                    ],
                },
                "efficiency": row["efficiency"],
                "failure_labels": compressed_retention_failure_labels(
                    correct=correct,
                    evidence=evidence,
                ),
                "failure_label_scope": "descriptive_final_retained_state_not_causal_attribution",
                "trace_ids": [row["resident_positions"]],
                "error": None,
                "accepted_answers": accepted,
                "accounting": row["accounting"],
            }
        )
    scored.sort(key=lambda row: (str(row["case_id"]), str(row["schedule_id"]), str(row["method"])))
    return scored


def _write_run_state(
    output: Path,
    *,
    config_sha256: str,
    plan_sha256: str,
    planned_cells: int,
    completed_cells: int,
    status: str,
    answer_view_opened: bool = False,
) -> None:
    state_path = output / "run_state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    atomic_write_json(
        state_path,
        {
            "schema_version": "1.0",
            "status": status,
            "created_at": previous.get("created_at", utc_now()),
            "config_sha256": config_sha256,
            "plan_sha256": plan_sha256,
            "planned_cells": planned_cells,
            "completed_cells": completed_cells,
            "updated_at": utc_now(),
            "answer_view_opened": answer_view_opened,
        },
    )


def run(
    config_path: Path,
    *,
    max_new_cells: int | None = None,
    rescore: bool = False,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = _load(config_path)
    validate_config(config)
    paths = _resolve_paths(config)
    output = paths["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_file(config_path)
    existing_state = output / "run_state.json"
    if existing_state.exists():
        state = json.loads(existing_state.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha256:
            raise RuntimeError("refusing to resume schedule control after config drift")
        if state.get("status") == "COMPLETE" and not rescore:
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))

    expected = dict(dict(config["model"])["expected_hashes"])
    observed = {
        "weights": sha256_file(paths["model"] / "model.safetensors"),
        "tokenizer": sha256_file(paths["model"] / "tokenizer.json"),
        "config": sha256_file(paths["model"] / "config.json"),
    }
    if observed != expected:
        raise RuntimeError(f"Halo checkpoint hash mismatch: {observed}")
    model_identity_sha256 = _canonical_sha256(observed)
    inference_rows = _jsonl(paths["inference"])
    capability_rows = _jsonl(paths["capability_cases"])
    plans = build_cell_plans(
        config,
        inference_rows,
        capability_rows,
        model_identity_sha256=model_identity_sha256,
    )
    plan_sha256 = _canonical_sha256(plans)
    if existing_state.exists():
        state = json.loads(existing_state.read_text(encoding="utf-8"))
        if state.get("plan_sha256") != plan_sha256:
            raise RuntimeError("refusing to resume schedule control after cell-plan drift")
    else:
        _write_run_state(
            output,
            config_sha256=config_sha256,
            plan_sha256=plan_sha256,
            planned_cells=len(plans),
            completed_cells=0,
            status="IN_PROGRESS",
        )
        atomic_write_json(output / "cell_plan.json", {"plan_sha256": plan_sha256, "cells": plans})

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    completed_before = sum((checkpoint_dir / f"{plan['cell_id']}.json").exists() for plan in plans)
    effective_limit = max_new_cells
    configured_limit = dict(config["checkpointing"]).get("max_new_cells_per_invocation")
    if effective_limit is None and configured_limit is not None:
        effective_limit = int(configured_limit)

    mini = _module("halo_schedule_mini_helpers", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module(
        "halo_schedule_capability_helpers",
        LAB_ROOT / "scripts" / "run_halo_capability.py",
    )
    tokenizer = None
    model = None
    selected = _selected_inference_rows(config, inference_rows, capability_rows)
    input_ids_by_case: dict[str, Any] = {}
    query_ids_by_case: dict[str, Any] = {}

    def executor(plan: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal tokenizer, model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(paths["model"], trust_remote_code=False)
            for row in selected:
                case_id = str(row["case_id"])
                observed_prompt_sha256 = hashlib.sha256(
                    str(row["prompt"]).encode("utf-8")
                ).hexdigest()
                if observed_prompt_sha256 != row["prompt_sha256"]:
                    raise RuntimeError(f"prompt text hash mismatch for {case_id}")
                input_ids = tokenizer(
                    str(row["prompt"]), add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                query_ids = tokenizer(
                    str(row["question"]), add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                if capability._token_hash(input_ids) != row["token_ids_sha256"]:
                    raise RuntimeError(f"token hash mismatch for {case_id}")
                if int(query_ids.shape[-1]) > int(dict(config["query"])["max_tokens"]):
                    raise RuntimeError(f"query token limit exceeded for {case_id}")
                input_ids_by_case[case_id] = input_ids
                query_ids_by_case[case_id] = query_ids
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                paths["model"],
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            ).to("cpu").eval()
        return _execute_model_cell(
            plan,
            model=model,
            tokenizer=tokenizer,
            input_ids_by_case=input_ids_by_case,
            query_ids_by_case=query_ids_by_case,
            config=config,
            mini=mini,
            positions_dir=output / "resident_positions",
        )

    unscored, newly_executed = run_checkpointed_cells(
        plans,
        checkpoint_dir,
        executor,
        max_new_cells=effective_limit,
    )
    del model
    gc.collect()
    complete = len(unscored) == len(plans)
    if not complete:
        _write_run_state(
            output,
            config_sha256=config_sha256,
            plan_sha256=plan_sha256,
            planned_cells=len(plans),
            completed_cells=len(unscored),
            status="IN_PROGRESS",
        )
        return {
            "schema_version": "1.0",
            "status": "IN_PROGRESS",
            "planned_cells": len(plans),
            "completed_cells": len(unscored),
            "newly_executed_cells": newly_executed,
            "answer_view_opened": False,
            "resume_command": f"python scripts/{Path(__file__).name} --config {config_path}",
        }

    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)
    _write_run_state(
        output,
        config_sha256=config_sha256,
        plan_sha256=plan_sha256,
        planned_cells=len(plans),
        completed_cells=len(unscored),
        status="SCORING",
        answer_view_opened=True,
    )
    scored = _score_rows(unscored, paths["evaluation"], output, capability)
    matching = validate_matched_schedule_rows(scored)
    analysis_config = dict(config["analysis"])
    analyses = {
        name: clustered_method_round_interaction(
            scored,
            outcome_path=str(path),
            minimum_clusters=int(analysis_config["minimum_independent_clusters"]),
            confidence=float(analysis_config["confidence"]),
            bootstrap_samples=int(analysis_config["bootstrap_samples"]),
            randomization_samples=int(analysis_config["randomization_samples"]),
            seed=int(dict(config["runtime"])["seed"]) + index,
        )
        for index, (name, path) in enumerate(dict(analysis_config["outcomes"]).items())
    }
    _atomic_write_jsonl(output / "cases.jsonl", scored)
    atomic_write_json(output / "statistics.json", {"matching_audit": matching, "outcomes": analyses})
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_HELD_CONSTANT_COMPRESSION_ROUND_CONTROL",
        "independent_clusters": matching["independent_clusters"],
        "planned_cells": len(plans),
        "completed_cells": len(scored),
        "newly_executed_cells": newly_executed,
        "resumed_cells": completed_before,
        "methods": list(METHODS),
        "round_counts_per_case": matching["round_counts_per_case"],
        "matching_audit": matching,
        "interaction_analyses": analyses,
        "claim_limit": "Exploratory B0-selected calibration control; not confirmatory evidence.",
    }
    atomic_write_json(output / "summary.json", summary)
    repo = git_snapshot(REPO_ROOT)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": json.loads(existing_state.read_text(encoding="utf-8")).get("created_at")
        if existing_state.exists()
        else utc_now(),
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": {"remote": repo["remote"], "commit": repo["commit"], "dirty": repo["dirty"]},
            "source": {
                "remote": "https://github.com/LUNCY579/RineyaBeta.git",
                "implementation": {"ref": "refs/heads/main", "commit": "9e00be74098878290dc2b0d54cbffee4652c6088", "path": "scripts/kv/significance.py", "blob": "30ee29724dce52de5a288f6421123fb718240541"},
                "paper": {"ref": "refs/heads/r2-healing-harness", "commit": "1664f54de0bbdfde45e311b75bbf3038dbfffe42", "path": "paper/significance_matrix/main.tex", "blob": "b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3"},
            },
        },
        "experiment": {"name": dict(config["experiment"])["name"], "compute_tier": "B", "architecture_variant": "B2/B3 held-constant fixed-terminal schedule control", "benchmark": "halo_calibration_v3_B0_eligible", "seed": dict(config["runtime"])["seed"], "query_conditioned": True},
        "config": {"resolved_path": f"configs/{config_path.name}", "sha256": config_sha256},
        "model": {"status": "VERIFIED", "id": dict(config["model"])["core_id"], "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze", "architecture": "Qwen3ForCausalLM", "hashes": {**observed, "chat_template": None}},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu", "precision": "bfloat16", "hardware": None},
        "budget": {"retained_history_entries": dict(config["cache"])["retained_history_budget"], "fixed_terminal_active_entries": dict(config["schedule"])["terminal_active_tokens"]},
        "artifacts": {"cell_plan": "cell_plan.json", "checkpoints": "checkpoints/", "predictions": "predictions.unscored.jsonl", "cases": "cases.jsonl", "statistics": "statistics.json", "summary": "summary.json", "resident_positions": "resident_positions/"},
        "notes": ["Exploratory B0-eligible schedule control; inference was restartable by immutable cell fingerprint.", "The answer/evidence view was opened only after all inference checkpoints were durable."],
    }
    atomic_write_json(output / "manifest.json", manifest)
    _write_run_state(
        output,
        config_sha256=config_sha256,
        plan_sha256=plan_sha256,
        planned_cells=len(plans),
        completed_cells=len(scored),
        status="COMPLETE",
        answer_view_opened=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=LAB_ROOT / "configs" / "halo_schedule_control.yaml"
    )
    parser.add_argument(
        "--max-new-cells",
        type=int,
        default=None,
        help="Checkpoint at most this many previously unfinished cells, then exit cleanly.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Rebuild scored artifacts from complete immutable checkpoints without model inference.",
    )
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        max_new_cells=args.max_new_cells,
        rescore=args.rescore,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
