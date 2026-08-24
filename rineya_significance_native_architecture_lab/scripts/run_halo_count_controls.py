#!/usr/bin/env python3
"""Run closed-loop Halo exposure-count causal controls under one frozen schedule."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
from rineya_significance_lab.evaluation.boundary_evidence import (  # noqa: E402
    score_observed_boundary_evidence,
)
from rineya_significance_lab.evaluation.evidence_endpoints import (  # noqa: E402
    score_evidence_retention_endpoint,
)
from rineya_significance_lab.evaluation.failure_taxonomy import (  # noqa: E402
    compressed_retention_failure_labels,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file, utc_now  # noqa: E402
from rineya_significance_lab.runtime import (  # noqa: E402
    CLEAN_COUNT_CONTROL_NAMES_V2,
    COUNT_CONTROL_NAMES,
    CountControlSpec,
    RuntimeCountController,
    boundary_resident_positions,
)

_FORBIDDEN_INFERENCE_FIELDS = frozenset(
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
        raise ValueError("count-control config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_inference_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "case_id",
        "cluster_id",
        "family",
        "prompt",
        "prompt_sha256",
        "question",
        "token_count",
        "token_ids_sha256",
    }
    seen: set[str] = set()
    for row in rows:
        missing = required.difference(row)
        forbidden = _FORBIDDEN_INFERENCE_FIELDS.intersection(row)
        if missing or forbidden:
            raise RuntimeError(
                f"prompt-only inference row {row.get('case_id')!r} is invalid: "
                f"missing={sorted(missing)}, forbidden={sorted(forbidden)}"
            )
        case_id = str(row["case_id"])
        if case_id in seen:
            raise RuntimeError(f"duplicate prompt-only inference case: {case_id}")
        seen.add(case_id)


def _resolve_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    return {
        key: (base / str(value)).resolve()
        for key, value in paths.items()
        if key != "base" and value not in (None, "", "null")
    }


def validate_count_control_config(config: Mapping[str, Any]) -> tuple[CountControlSpec, ...]:
    """Reject any condition-specific query, budget, schedule, or ranking drift."""

    protocol = config.get("protocol", {})
    if protocol.get("execution_mode") != "closed_loop_model_execution":
        raise ValueError("count causal controls require closed_loop_model_execution")
    contract = str(protocol.get("count_control_contract", "global_v1"))
    if contract not in {"global_v1", "eligible_heavy_v2"}:
        raise ValueError(f"unknown count-control contract: {contract!r}")
    runtime = config.get("runtime", {})
    expected_runtime = {
        "device": "cpu",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "batch_size": 1,
        "deterministic": True,
        "decoding": "manual_greedy",
    }
    mismatched_runtime = {
        key: {"observed": runtime.get(key), "expected": expected}
        for key, expected in expected_runtime.items()
        if runtime.get(key) != expected
    }
    if mismatched_runtime:
        raise ValueError(f"count-control runtime drift: {mismatched_runtime}")
    if int(runtime.get("max_new_tokens", 0)) <= 0:
        raise ValueError("runtime.max_new_tokens must be positive")

    model = config.get("model", {})
    core_id = str(model.get("core_id", ""))
    if tuple(model.get("allowed_core_ids", ())) != (core_id,):
        raise ValueError("model.allowed_core_ids must contain exactly model.core_id")
    if ".halo." not in core_id.lower() or model.get("expected_source_base") != "Qwen3-0.6B":
        raise ValueError("only the published Qwen3-0.6B Halo core is permitted")
    if bool(model.get("trust_remote_code", True)):
        raise ValueError("trust_remote_code must remain false")

    cache = config.get("cache", {})
    budget = int(cache.get("retained_history_budget", 0))
    chunk = int(cache.get("compression_chunk_tokens", 0))
    sinks = int(cache.get("sink_tokens", -1))
    recent = int(cache.get("recent_tokens", -1))
    if budget <= 0 or chunk <= 0 or sinks < 0 or recent < 0 or sinks + recent >= budget:
        raise ValueError("cache settings must be positive and leave at least one heavy slot")
    if int(cache.get("total_budget_slots", -1)) != budget:
        raise ValueError("total_budget_slots must equal retained_history_budget")
    if not bool(cache.get("matched_across_all_count_controls", False)):
        raise ValueError("cache budget must be declared matched across all controls")

    significance = config.get("significance", {})
    if not bool(significance.get("enabled", False)) or float(significance.get("exponent", -1)) != 1.0:
        raise ValueError("all count controls require enabled significance with exponent 1.0")
    if not bool(significance.get("preserve_true_counts_across_all_controls", False)):
        raise ValueError("the experiment must preserve true persistent counts")
    pooling = significance.get("score_pooling", {})
    if bool(pooling.get("enabled", False)) or int(pooling.get("width", 0)) != 1:
        raise ValueError("score pooling must remain disabled")
    query = config.get("query", {})
    if query.get("mode") != "conditioned":
        raise ValueError("all conditions must use query-conditioned scoring")
    if query.get("probe_schedule") != "every_nonfinal_chunk_boundary":
        raise ValueError("query probes must occur at every non-final chunk boundary")
    if query.get("access_fairness") != "identical_across_all_count_controls":
        raise ValueError("query-access fairness must be explicit")

    raw_conditions = config.get("conditions")
    if not isinstance(raw_conditions, list):
        raise ValueError("conditions must be a list")
    forbidden_overrides = {
        "budget",
        "chunk_size",
        "exponent",
        "policy",
        "query_conditioned",
        "query_mode",
        "probe_schedule",
        "sink_tokens",
        "recent_tokens",
    }
    specs: list[CountControlSpec] = []
    for row in raw_conditions:
        if not isinstance(row, dict):
            raise ValueError("every count-control condition must be a mapping")
        overlap = forbidden_overrides.intersection(row)
        if overlap:
            raise ValueError(f"condition-specific matched-contract override: {sorted(overlap)}")
        specs.append(CountControlSpec.from_mapping(row, seed=int(runtime["seed"])))
    transforms = tuple(spec.transform for spec in specs)
    expected_transforms = (
        CLEAN_COUNT_CONTROL_NAMES_V2
        if contract == "eligible_heavy_v2"
        else COUNT_CONTROL_NAMES
    )
    if transforms != expected_transforms:
        raise ValueError(
            "conditions must cover the frozen count controls for "
            f"{contract} in order: {expected_transforms}"
        )
    if len({spec.condition_id for spec in specs}) != len(specs):
        raise ValueError("count-control condition IDs must be unique")
    families = tuple(config.get("selection", {}).get("families", ()))
    if not families or "R2" in families:
        raise ValueError("active count controls require non-R2 B0-eligible benchmark families")
    if contract == "eligible_heavy_v2":
        intervention = config.get("count_intervention", {})
        required_intervention = {
            "version": 2,
            "scope": "eligible_heavy_ranking_pool_only",
            "protected_slots_immutable": True,
            "eligible_multiset_preserved_per_layer_head": True,
            "semantic_derangement_required": True,
            "prior_global_controls_status": "ELIGIBLE_POOL_CONFOUNDED",
        }
        drift = {
            key: {"observed": intervention.get(key), "expected": expected}
            for key, expected in required_intervention.items()
            if intervention.get(key) != expected
        }
        if drift:
            raise ValueError(f"eligible-heavy v2 intervention contract drift: {drift}")
        if (sinks, recent) != (4, 32):
            raise ValueError("eligible-heavy v2 freezes exactly four sink and 32 recent entries")
    return tuple(specs)


def select_case_ids(config: Mapping[str, Any], capability_rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    selection = config["selection"]
    families = {str(value) for value in selection["families"]}
    maximum = int(selection["maximum_eligible_clusters_per_family"])
    if maximum <= 0:
        raise ValueError("maximum_eligible_clusters_per_family must be positive")
    eligible: defaultdict[str, list[str]] = defaultdict(list)
    for row in capability_rows:
        family = str(row.get("task_family"))
        if bool(row.get("correct")) and family in families:
            eligible[family].append(str(row["case_id"]))
    selected = tuple(
        case_id
        for family in sorted(families)
        for case_id in sorted(set(eligible[family]))[:maximum]
    )
    if len(selected) < 8:
        raise RuntimeError("count-control run requires at least eight B0-eligible clusters")
    return selected


_MATCHED_ACCOUNTING_FIELDS = (
    "input_tokens",
    "chunk_size",
    "chunk_count",
    "chunk_sizes",
    "compression_boundaries",
    "boundary_accounting",
    "retained_history_budget",
    "history_entries_after_last_boundary",
    "final_active_chunk_entries",
    "query_tokens_per_probe",
    "query_probe_token_evaluations",
    "max_transient_resident_entries",
    "resident_entries_after_prefill_expected",
    "resident_entries_after_prefill",
    "scoring_events_per_layer",
    "compression_reports",
    "effective_evictions_per_layer",
)


def assert_matched_execution_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_ids: Sequence[str],
) -> None:
    """Prove prompt schedule, query access, budget, and scoring events are paired."""

    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    expected_ids = tuple(condition_ids)
    for case_id, case_rows in grouped.items():
        observed_ids = tuple(str(row["condition"]) for row in case_rows)
        if observed_ids != expected_ids:
            raise RuntimeError(
                f"{case_id}: conditions are missing or reordered: {observed_ids} != {expected_ids}"
            )
        reference = case_rows[0]
        for row in case_rows[1:]:
            if row.get("prompt_sha256") != reference.get("prompt_sha256"):
                raise RuntimeError(f"{case_id}: prompt changed across count controls")
            if row.get("query_token_sha256") != reference.get("query_token_sha256"):
                raise RuntimeError(f"{case_id}: query tokens changed across count controls")
            left = reference["accounting"]
            right = row["accounting"]
            for field in _MATCHED_ACCOUNTING_FIELDS:
                if left[field] != right[field]:
                    raise RuntimeError(
                        f"{case_id}: matched execution field {field!r} differs between "
                        f"{reference['condition']} and {row['condition']}"
                    )


def recover_aligned_checkpoint_prefix(
    expected_order: Sequence[tuple[str, str]],
    predictions: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover the common atomic prefix of the two inference checkpoint files."""

    prediction_order = [
        (str(row.get("case_id")), str(row.get("condition"))) for row in predictions
    ]
    boundary_order = [
        (str(row.get("case_id")), str(row.get("condition"))) for row in boundaries
    ]
    expected = list(expected_order)
    if prediction_order != expected[: len(prediction_order)]:
        raise RuntimeError("partial prediction checkpoint is not an exact frozen-order prefix")
    if boundary_order != expected[: len(boundary_order)]:
        raise RuntimeError("partial boundary checkpoint is not an exact frozen-order prefix")
    common = min(len(prediction_order), len(boundary_order))
    if boundary_order[:common] != prediction_order[:common]:
        raise RuntimeError("partial boundary index is not aligned with partial predictions")
    return (
        [dict(row) for row in predictions[:common]],
        [dict(row) for row in boundaries[:common]],
    )


def dry_run_plan(config_path: Path) -> dict[str, Any]:
    """Validate all frozen inputs without loading or executing Halo."""

    config = _load(config_path.resolve())
    specs = validate_count_control_config(config)
    paths = _resolve_paths(config)
    selected = select_case_ids(config, _jsonl(paths["capability_cases"]))
    inference_rows = _jsonl(paths["inference"])
    _validate_inference_rows(inference_rows)
    inference_ids = {str(row["case_id"]) for row in inference_rows}
    missing = sorted(set(selected).difference(inference_ids))
    if missing:
        raise RuntimeError(f"selected cases missing from prompt-only inference view: {missing}")
    return {
        "status": "DRY_RUN_VALID",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "model_loaded": False,
        "model_id": config["model"]["core_id"],
        "conditions": [spec.condition_id for spec in specs],
        "count_transforms": [spec.transform for spec in specs],
        "count_control_contract": config["protocol"].get(
            "count_control_contract", "global_v1"
        ),
        "selected_case_ids": list(selected),
        "selected_clusters": len(selected),
        "planned_model_executions": len(selected) * len(specs),
        "query_access": config["query"]["access_fairness"],
        "retained_history_budget": config["cache"]["retained_history_budget"],
        "compression_chunk_tokens": config["cache"]["compression_chunk_tokens"],
        "protected_sink_tokens": config["cache"]["sink_tokens"],
        "protected_recent_tokens": config["cache"]["recent_tokens"],
        "claim_limit": "Configuration/input validation only; no model or outcome evidence.",
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mini = _module("halo_count_control_eval_helpers", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module("halo_count_control_capability_helpers", LAB_ROOT / "scripts" / "run_halo_capability.py")
    started_at = utc_now()
    config_path = config_path.resolve()
    config = _load(config_path)
    specs = validate_count_control_config(config)
    control_contract = str(config["protocol"].get("count_control_contract", "global_v1"))
    paths = _resolve_paths(config)
    output = paths["output_root"]
    selected = select_case_ids(config, _jsonl(paths["capability_cases"]))
    selected_set = set(selected)
    inference_rows = _jsonl(paths["inference"])
    _validate_inference_rows(inference_rows)
    inference = {
        str(row["case_id"]): row
        for row in inference_rows
        if str(row["case_id"]) in selected_set
    }
    if set(inference) != selected_set:
        raise RuntimeError("prompt-only inference view is missing selected cases")

    fingerprint = _config_fingerprint(config)
    state_path = output / "run_state.json"
    partial_predictions_path = output / "predictions.partial.jsonl"
    partial_boundaries_path = output / "boundary_diagnostics.partial.jsonl"
    if output.exists():
        if not resume:
            raise RuntimeError(f"refusing to overwrite count-control output: {output}")
        if not state_path.is_file():
            raise RuntimeError("resume requires a durable run_state.json")
        run_state = json.loads(state_path.read_text(encoding="utf-8"))
        if run_state.get("config_sha256") != fingerprint:
            raise RuntimeError("resume config does not match the original frozen run")
        if run_state.get("status") == "COMPLETE":
            raise RuntimeError("count-control run is already complete")
        if not partial_predictions_path.is_file() or not partial_boundaries_path.is_file():
            raise RuntimeError("resume requires both partial inference checkpoints")
        unscored = _jsonl(partial_predictions_path)
        boundary_index_rows = _jsonl(partial_boundaries_path)
    else:
        output.mkdir(parents=True)
        (output / "resident_positions").mkdir()
        (output / "boundary_diagnostics").mkdir()
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
        unscored = []
        boundary_index_rows = []
        _write_jsonl(partial_predictions_path, unscored)
        _write_jsonl(partial_boundaries_path, boundary_index_rows)
        atomic_write_json(
            state_path,
            {
                "schema_version": "1.0",
                "status": "INFERENCE_IN_PROGRESS",
                "config_sha256": fingerprint,
                "completed_executions": 0,
                "planned_executions": len(selected) * len(specs),
            },
        )

    expected_order = [
        (case_id, spec.condition_id) for case_id in selected for spec in specs
    ]
    original_checkpoint_lengths = (len(unscored), len(boundary_index_rows))
    unscored, boundary_index_rows = recover_aligned_checkpoint_prefix(
        expected_order,
        unscored,
        boundary_index_rows,
    )
    # The two files are independently atomic. A process termination between
    # their replacements may leave one exactly one row ahead; recover the
    # common durable prefix instead of making a long CPU run unrestartable.
    if original_checkpoint_lengths != (len(unscored), len(boundary_index_rows)):
        _write_jsonl(partial_predictions_path, unscored)
        _write_jsonl(partial_boundaries_path, boundary_index_rows)
    completed_order = [
        (str(row["case_id"]), str(row["condition"])) for row in unscored
    ]
    for row in unscored:
        for key in ("resident_positions", "boundary_diagnostics"):
            if not (output / str(row[key])).is_file():
                raise RuntimeError(f"partial checkpoint references missing artifact: {row[key]}")
    completed = set(completed_order)

    core = json.loads((paths["model"] / "CORE.json").read_text(encoding="utf-8"))
    mini._validate_halo_core(core, config["model"])
    checkpoint = mini._checkpoint_hashes(paths["model"], config["model"])
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(paths["model"], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        paths["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    mini._check_geometry(model.config, config["model"])
    eos = tokenizer.eos_token_id
    eos_ids = {int(value) for value in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    for case_id in selected:
        row = inference[case_id]
        input_ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        query_ids = tokenizer(str(row["question"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        if int(query_ids.shape[-1]) > int(config["query"]["max_tokens"]):
            raise RuntimeError(f"{case_id}: query exceeds frozen query.max_tokens")
        frozen_input_hash = capability._token_hash(input_ids)
        if frozen_input_hash != str(row["token_ids_sha256"]):
            raise RuntimeError(f"{case_id}: Halo tokenizer no longer reproduces frozen prompt tokens")
        if int(input_ids.shape[-1]) != int(row["token_count"]):
            raise RuntimeError(f"{case_id}: token count changed; truncation or tokenizer drift detected")
        prepared = mini.PreparedCase(
            case_id=case_id,
            scenario_id=str(row["cluster_id"]),
            metadata={"family": row["family"]},
            input_ids=input_ids,
            context_ids=input_ids,
            query_ids=query_ids,
            prompt_text_sha256=str(row["prompt_sha256"]),
            input_token_sha256=mini._token_sha256(input_ids),
            context_token_sha256=mini._token_sha256(input_ids),
            query_token_sha256=mini._token_sha256(query_ids),
        )
        for spec in specs:
            if (case_id, spec.condition_id) in completed:
                continue
            controller = RuntimeCountController(
                spec,
                case_id=case_id,
                chunk_size=int(config["cache"]["compression_chunk_tokens"]),
                protected_sinks=(
                    int(config["cache"]["sink_tokens"])
                    if control_contract == "eligible_heavy_v2"
                    else None
                ),
                protected_recent=(
                    int(config["cache"]["recent_tokens"])
                    if control_contract == "eligible_heavy_v2"
                    else None
                ),
            )
            condition = mini.Condition(spec.condition_id, "significance", 1.0, True)
            result = mini.execute_condition(
                model,
                prepared,
                condition,
                cache_config=config["cache"],
                runtime_config={
                    "max_new_tokens": config["runtime"]["max_new_tokens"],
                    "verify_query_rollback": config["runtime"]["verify_query_rollback"],
                },
                eos_token_ids=eos_ids,
                return_resident_positions=True,
                compression_callback=controller.compress,
            )
            resident = result.pop("_resident_positions")
            stem = f"{case_id}__{spec.condition_id}"
            resident_path = output / "resident_positions" / f"{stem}.npz"
            diagnostic_path = output / "boundary_diagnostics" / f"{stem}.npz"
            _write_npz_atomic(
                resident_path,
                {f"layer_{index:03d}": value for index, value in enumerate(resident)},
            )
            _write_npz_atomic(diagnostic_path, controller.diagnostic_arrays())
            summaries = [summary.to_dict() for summary in controller.summaries]
            expected_boundaries = int(result["accounting"]["compression_boundaries"])
            if len(summaries) != expected_boundaries:
                raise RuntimeError(
                    f"{stem}: diagnostic boundary count {len(summaries)} != {expected_boundaries}"
                )
            boundary_index_rows.append(
                {
                    "case_id": case_id,
                    "condition": spec.condition_id,
                    "count_transform": spec.transform,
                    "diagnostic_npz": diagnostic_path.relative_to(output).as_posix(),
                    "boundaries": summaries,
                }
            )
            prediction = tokenizer.decode(
                result.pop("generated_token_ids"), skip_special_tokens=True
            ).strip()
            unscored.append(
                {
                    "case_id": case_id,
                    "cluster_id": row["cluster_id"],
                    "family": row["family"],
                    "condition": spec.condition_id,
                    "count_transform": spec.transform,
                    "prediction": prediction,
                    "prompt_sha256": prepared.prompt_text_sha256,
                    "query_token_sha256": prepared.query_token_sha256,
                    "resident_positions": resident_path.relative_to(output).as_posix(),
                    "boundary_diagnostics": diagnostic_path.relative_to(output).as_posix(),
                    "boundary_summaries": summaries,
                    **result,
                }
            )
            _write_jsonl(partial_predictions_path, unscored)
            _write_jsonl(partial_boundaries_path, boundary_index_rows)
            atomic_write_json(
                state_path,
                {
                    "schema_version": "1.0",
                    "status": "INFERENCE_IN_PROGRESS",
                    "config_sha256": fingerprint,
                    "completed_executions": len(unscored),
                    "planned_executions": len(expected_order),
                    "last_completed": {
                        "case_id": case_id,
                        "condition": spec.condition_id,
                    },
                },
            )
    if len(unscored) != len(expected_order):
        raise RuntimeError("inference loop ended without every frozen case/condition")
    assert_matched_execution_contract(
        unscored,
        condition_ids=[spec.condition_id for spec in specs],
    )
    _write_jsonl(output / "predictions.unscored.jsonl", unscored)
    _write_jsonl(output / "boundary_diagnostics.jsonl", boundary_index_rows)
    atomic_write_json(
        state_path,
        {
            "schema_version": "1.0",
            "status": "INFERENCE_COMPLETE_EVALUATION_PENDING",
            "config_sha256": fingerprint,
            "completed_executions": len(unscored),
            "planned_executions": len(expected_order),
        },
    )

    # Open the answer/evidence view only after every unscored outcome is durable.
    evaluation = {str(row["case_id"]): row for row in _jsonl(paths["evaluation"])}
    scored: list[dict[str, Any]] = []
    boundary_evidence_rows: list[dict[str, Any]] = []
    for row in unscored:
        key = evaluation[str(row["case_id"])]
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
        with np.load(output / row["resident_positions"], allow_pickle=False) as arrays:
            resident = [arrays[name].tolist() for name in sorted(arrays.files)]
        retention = score_posthoc_evidence_retention(annotation, resident)
        evidence = retention.to_dict()
        frozen_endpoint = score_evidence_retention_endpoint(retention).to_dict()
        with np.load(output / row["boundary_diagnostics"], allow_pickle=False) as arrays:
            diagnostic_arrays = {name: arrays[name].copy() for name in arrays.files}
        for summary in row["boundary_summaries"]:
            boundary_index = int(summary["boundary_index"])
            boundary_resident = boundary_resident_positions(
                diagnostic_arrays,
                boundary_index=boundary_index,
                layer_count=int(summary["layer_count"]),
            )
            boundary_score = score_observed_boundary_evidence(
                annotation,
                boundary_resident,
                true_length=int(summary["true_length"]),
            )
            boundary_evidence_rows.append(
                {
                    "schema_version": "1.0",
                    "case_id": row["case_id"],
                    "cluster_id": row["cluster_id"],
                    "condition": row["condition"],
                    "count_transform": row["count_transform"],
                    **summary,
                    "boundary_evidence": boundary_score.to_dict(),
                }
            )
        correct = capability._task_correct(
            str(row["family"]),
            str(row["prediction"]),
            [str(value) for value in key["accepted_answers"]],
        )
        strict_survival = bool(
            evidence[
                "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer"
            ]
        )
        failure_labels = compressed_retention_failure_labels(
            correct=correct,
            evidence=evidence,
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
                "chunk_size": config["cache"]["compression_chunk_tokens"],
                "status": "OK",
                "prediction": row["prediction"],
                "correct": correct,
                "full_cache_correct": True,
                "compression_conditioned_eligible": True,
                "count_transform": row["count_transform"],
                "evidence": {
                    "required_count": annotation.required_token_count,
                    "surviving_count_final": round(
                        evidence["minimum_per_layer_required_token_availability_fraction"]
                        * annotation.required_token_count
                    ),
                    "all_required_survived": strict_survival,
                    **evidence,
                    **frozen_endpoint,
                },
                "efficiency": row["efficiency"],
                "failure_labels": failure_labels,
                "failure_label_scope": "descriptive_final_retained_state_not_causal_attribution",
                "trace_ids": [row["resident_positions"], row["boundary_diagnostics"]],
                "error": None,
                "accepted_answers": key["accepted_answers"],
            }
        )
    _write_jsonl(output / "cases.jsonl", scored)
    _write_jsonl(output / "boundary_evidence.jsonl", boundary_evidence_rows)

    by_condition: dict[str, dict[str, Any]] = {}
    for spec in specs:
        rows = [row for row in scored if row["condition"] == spec.condition_id]
        by_condition[spec.condition_id] = {
            "count_transform": spec.transform,
            "clusters": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "strict_evidence_survival": sum(
                bool(row["evidence"]["all_required_survived"]) for row in rows
            ),
            "mean_layer_fact_exact_completion_rate": sum(
                float(row["evidence"]["layer_fact_exact_completion_rate"])
                for row in rows
            )
            / len(rows),
            "mean_minimum_per_layer_required_token_availability_fraction": sum(
                float(row["evidence"]["minimum_per_layer_required_token_availability_fraction"])
                for row in rows
            )
            / len(rows),
        }
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_CLOSED_LOOP_COUNT_CAUSAL_CONTROLS",
        "count_control_contract": control_contract,
        "selected_B0_eligible_clusters": len(selected),
        "families": sorted({str(row["family"]) for row in unscored}),
        "conditions": [spec.condition_id for spec in specs],
        "matched_query_budget_schedule_contract": "PASS",
        "eligible_pool_invariants": (
            {
                "protected_denominators_immutable": "PASS",
                "eligible_count_multiset_preserved_per_layer_head": "PASS",
                "semantic_fixed_points": 0,
            }
            if control_contract == "eligible_heavy_v2"
            else None
        ),
        "by_condition": by_condition,
        "boundary_evidence_rows": len(boundary_evidence_rows),
        "inferential_statistics": None,
        "claim_limit": "Exploratory B0-selected calibration controls; not confirmatory evidence.",
    }
    atomic_write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": started_at,
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": {
                "remote": "https://github.com/LUNCY579/ChatBubble.git",
                "commit": "03777703dd8587f1248d9a72115eafb9356d50b8",
                "dirty": True,
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
            "name": config["experiment"]["name"],
            "compute_tier": "B",
            "architecture_variant": "Halo closed-loop exposure count causal controls",
            "benchmark": "halo_calibration_v3_B0_eligible",
            "seed": config["runtime"]["seed"],
            "query_conditioned": True,
            "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
            "count_control_contract": control_contract,
        },
        "config": {
            "resolved_path": "resolved_config.yaml",
            "sha256": sha256_file(output / "resolved_config.yaml"),
        },
        "model": {
            "status": "VERIFIED",
            "id": config["model"]["core_id"],
            "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze",
            "architecture": "Qwen3ForCausalLM",
            "hashes": dict(config["model"]["expected_hashes"]),
            "checkpoint_integrity": checkpoint,
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
            "retained_history_entries_per_layer_head": config["cache"]["retained_history_budget"],
            "matched_across_conditions": True,
        },
        "artifacts": {
            "predictions": "predictions.unscored.jsonl",
            "cases": "cases.jsonl",
            "summary": "summary.json",
            "resident_positions": "resident_positions/",
            "boundary_diagnostics": "boundary_diagnostics/",
            "boundary_index": "boundary_diagnostics.jsonl",
            "boundary_evidence": "boundary_evidence.jsonl",
            "restart_state": "run_state.json",
            "partial_predictions": "predictions.partial.jsonl",
            "partial_boundary_index": "boundary_diagnostics.partial.jsonl",
        },
        "notes": [
            "True realized C remains persistent in every condition; only the boundary ranking denominator is perturbed.",
            "All conditions receive identical prompt/query tokens, query probes, cache budget, chunk schedule, and greedy decoding procedure.",
            "No Rineya/R2 model, Aegis model, or benchmark R2 family is active.",
            *(
                [
                    "V2 shuffle/rotation transforms only the eligible heavy-ranking pool after excluding exactly four sink and 32 recent entries.",
                    "The prior v1 global shuffled/wrong-token controls remain ELIGIBLE_POOL_CONFOUNDED and are not reinterpreted.",
                ]
                if control_contract == "eligible_heavy_v2"
                else []
            ),
        ],
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(
        state_path,
        {
            "schema_version": "1.0",
            "status": "COMPLETE",
            "config_sha256": fingerprint,
            "completed_executions": len(unscored),
            "planned_executions": len(expected_order),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_count_causal_controls.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume are mutually exclusive")
    result = dry_run_plan(args.config) if args.dry_run else run(args.config, resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
