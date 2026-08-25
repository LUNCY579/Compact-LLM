#!/usr/bin/env python3
"""Run the bounded Halo-only Cornerstone-1 published-core evaluation.

The run is deliberately exploratory and descriptive.  It evaluates exactly
three prompt-only long-context cases under an instrumented full-cache control
(B0), raw accumulated significance (B2), and observation-normalized
significance (B3).  Every condition uses the same prompt tokens, chunk
boundaries, query tokens, temporary query probes, and manual greedy decoder.

The answer-key file is not opened until every generation has completed and the
unscored predictions have been durably written.  The script refuses to reuse an
existing output directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import (  # noqa: E402
    build_manifest,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.runtime import (  # noqa: E402
    SignificanceCache,
    SignificancePolicy,
    chunked_prefill,
)

CASE_IDS = ("RLC-001-B", "RLC-010-B", "RLC-019-B")
CONDITION_IDS = ("B0", "B2", "B3")
FORBIDDEN_PROMPT_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "accepted_answers",
        "evidence",
        "gold",
        "label",
        "reference_answer",
        "supporting_facts",
        "target",
    }
)


@dataclass(frozen=True)
class Condition:
    """One frozen cache condition in the bounded comparison."""

    condition_id: str
    policy: str
    exponent: float | None
    query_conditioned: bool

    @property
    def is_full_cache(self) -> bool:
        return self.policy == "full_cache"


@dataclass(frozen=True)
class PreparedCase:
    """Prompt-only tensors prepared once and shared by every condition."""

    case_id: str
    scenario_id: str | None
    metadata: dict[str, Any]
    input_ids: Any
    context_ids: Any
    query_ids: Any
    prompt_text_sha256: str
    input_token_sha256: str
    context_token_sha256: str
    query_token_sha256: str


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _resolve_path(value: Any, *, base: Path) -> Path:
    if value in (None, "", "null"):
        raise ValueError("required path is null or empty")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _declared_base(config: Mapping[str, Any]) -> Path:
    base_name = str(config.get("paths", {}).get("base", "package_root"))
    if base_name == "package_root":
        return LAB_ROOT
    if base_name == "repository_root":
        return REPO_ROOT
    raise ValueError("paths.base must be 'package_root' or 'repository_root'")


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _conditions_from_config(config: Mapping[str, Any]) -> tuple[Condition, ...]:
    raw = config.get("conditions")
    if not isinstance(raw, list):
        raise ValueError("conditions must be a list")
    conditions: list[Condition] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every condition must be a mapping")
        exponent = item.get("exponent")
        conditions.append(
            Condition(
                condition_id=str(item.get("id")),
                policy=str(item.get("policy")),
                exponent=None if exponent is None else float(exponent),
                query_conditioned=bool(item.get("query_conditioned", False)),
            )
        )
    return tuple(conditions)


def validate_config(config: Mapping[str, Any]) -> tuple[Condition, ...]:
    """Reject scope drift and comparison asymmetries before model loading."""

    benchmark = config.get("benchmark", {})
    case_ids = tuple(str(value) for value in benchmark.get("case_ids", []))
    if case_ids != CASE_IDS:
        raise ValueError(f"benchmark.case_ids must be exactly {list(CASE_IDS)}, got {list(case_ids)}")

    conditions = _conditions_from_config(config)
    if tuple(condition.condition_id for condition in conditions) != CONDITION_IDS:
        raise ValueError(f"conditions must be ordered exactly as {list(CONDITION_IDS)}")
    expected = {
        "B0": ("full_cache", None),
        "B2": ("significance", 0.0),
        "B3": ("significance", 1.0),
    }
    for condition in conditions:
        policy, exponent = expected[condition.condition_id]
        if condition.policy != policy or condition.exponent != exponent:
            raise ValueError(
                f"{condition.condition_id} must use policy={policy!r}, exponent={exponent!r}"
            )
        if not condition.query_conditioned:
            raise ValueError(
                f"{condition.condition_id} must receive the common query probe for access fairness"
            )

    runtime = config.get("runtime", {})
    if str(runtime.get("device", "")).lower() != "cpu":
        raise ValueError("the Halo mini evaluation is CPU-only")
    if str(runtime.get("dtype", "")).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("the Halo mini evaluation requires bfloat16")
    if str(runtime.get("decoding", "")) != "manual_greedy":
        raise ValueError("runtime.decoding must be manual_greedy")
    if str(runtime.get("attention_implementation", "")) != "sdpa":
        raise ValueError("runtime.attention_implementation must be sdpa")
    if not bool(runtime.get("deterministic", False)):
        raise ValueError("runtime.deterministic must remain true")
    if int(runtime.get("batch_size", 0)) != 1:
        raise ValueError("runtime.batch_size must be 1")
    if int(runtime.get("max_new_tokens", 0)) <= 0:
        raise ValueError("runtime.max_new_tokens must be positive")

    model = config.get("model", {})
    allowed = tuple(str(value) for value in model.get("allowed_core_ids", []))
    if allowed != (str(model.get("core_id")),):
        raise ValueError("model.allowed_core_ids must contain only model.core_id")
    if ".halo." not in allowed[0].lower():
        raise ValueError("the only permitted published core is Halo")
    if str(model.get("expected_source_base")) != "Qwen3-0.6B":
        raise ValueError("Halo must declare the Qwen3-0.6B source base")
    if bool(model.get("trust_remote_code", True)):
        raise ValueError("trust_remote_code must remain false")

    cache = config.get("cache", {})
    chunk_size = int(cache.get("compression_chunk_tokens", 0))
    history_budget = int(cache.get("retained_history_budget", 0))
    sinks = int(cache.get("sink_tokens", -1))
    recent = int(cache.get("recent_tokens", -1))
    if chunk_size <= 0 or history_budget <= 0:
        raise ValueError("chunk size and retained-history budget must be positive")
    if sinks < 0 or recent < 0 or sinks + recent >= history_budget:
        raise ValueError("the retained-history budget must leave at least one heavy slot")
    pooling = config.get("significance", {}).get("score_pooling", {})
    if bool(pooling.get("enabled", False)) or int(pooling.get("width", 1)) != 1:
        raise ValueError("score pooling must remain disabled for Cornerstone-1 isolation")
    query = config.get("query", {})
    if str(query.get("mode")) != "conditioned":
        raise ValueError("query.mode must be conditioned")
    if str(query.get("rendering")) != "raw_question_no_special_tokens":
        raise ValueError("query.rendering must be raw_question_no_special_tokens")
    if str(query.get("probe_schedule")) != "every_nonfinal_chunk_boundary":
        raise ValueError("query.probe_schedule must be every_nonfinal_chunk_boundary")
    return conditions


def _validate_halo_core(core: Mapping[str, Any], model_config: Mapping[str, Any]) -> None:
    """Prevent accidental evaluation of R2, Aegis, or another family core."""

    expected_id = str(model_config["core_id"])
    checks = {
        "core_id": (str(core.get("core_id")), expected_id),
        "codename": (str(core.get("codename")), "Halo"),
        "directory_name": (str(core.get("directory_name")), "Rineya-Halo-1.2.0-devfreeze"),
        "source_base": (str(core.get("source_base")), "Qwen3-0.6B"),
    }
    mismatches = {name: {"observed": got, "expected": wanted} for name, (got, wanted) in checks.items() if got != wanted}
    if mismatches:
        raise RuntimeError(f"refusing non-Halo or unexpected published core: {mismatches}")


def _checkpoint_hashes(model_path: Path, model_config: Mapping[str, Any]) -> dict[str, Any]:
    declared = model_config.get("checkpoint_files")
    expected = model_config.get("expected_hashes")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("model.checkpoint_files must be a non-empty mapping")
    if not isinstance(expected, dict) or set(expected) != set(declared):
        raise ValueError("model.expected_hashes must cover every declared checkpoint file exactly")
    root = model_path.resolve()
    files: dict[str, dict[str, str]] = {}
    for logical_name, relative_value in declared.items():
        relative = Path(str(relative_value))
        if relative.is_absolute():
            raise ValueError(f"checkpoint file must be relative: {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"checkpoint file escapes model directory: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"declared checkpoint file is missing: {path}")
        observed = sha256_file(path)
        wanted = str(expected[logical_name]).lower()
        if observed != wanted:
            raise RuntimeError(
                f"checkpoint hash mismatch for {logical_name}: expected {wanted}, observed {observed}"
            )
        files[str(logical_name)] = {"path": relative.as_posix(), "sha256": observed}
    aggregate_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256",
        "files": files,
        "aggregate_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
    }


def _check_geometry(config: Any, expected: Mapping[str, Any]) -> None:
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
        elif got is None or int(got) != int(wanted):
            raise RuntimeError(f"model geometry mismatch for {name}: expected {wanted}, got {got}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _assert_prompt_only(row: Mapping[str, Any]) -> None:
    forbidden = FORBIDDEN_PROMPT_FIELDS.intersection(row)
    if forbidden:
        raise RuntimeError(f"prompt row {row.get('id')} exposes oracle fields: {sorted(forbidden)}")
    if not isinstance(row.get("context"), str) or not isinstance(row.get("question"), str):
        raise RuntimeError(f"prompt row {row.get('id')} needs string context and question fields")
    messages = row.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise RuntimeError(f"prompt row {row.get('id')} messages must be a list")
        bad_roles = [
            message.get("role")
            for message in messages
            if not isinstance(message, dict) or message.get("role") not in {"system", "user"}
        ]
        if bad_roles:
            raise RuntimeError(f"prompt row {row.get('id')} contains non-prompt message roles: {bad_roles}")


def _read_prompt_rows(path: Path, requested_ids: Sequence[str] = CASE_IDS) -> list[dict[str, Any]]:
    requested = tuple(requested_ids)
    if requested != CASE_IDS:
        raise ValueError(f"this bounded evaluator accepts exactly {list(CASE_IDS)}")
    selected: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in _load_jsonl(path):
        row_id = str(row.get("id", ""))
        if row_id not in requested:
            continue
        _assert_prompt_only(row)
        if row_id in selected:
            duplicates.add(row_id)
        selected[row_id] = row
    if duplicates:
        raise RuntimeError(f"duplicate selected prompt IDs: {sorted(duplicates)}")
    missing = [row_id for row_id in requested if row_id not in selected]
    if missing:
        raise RuntimeError(f"missing selected prompt IDs: {missing}")
    return [selected[row_id] for row_id in requested]


def _render_prompt(tokenizer: Any, row: Mapping[str, Any], prompt_config: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if str(prompt_config.get("rendering")) != "chat_template":
        raise ValueError("prompt.rendering must be chat_template")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"row {row.get('id')} has no chat messages")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("the verified Halo tokenizer has no chat template")
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(prompt_config.get("enable_thinking", False)),
        )
    )


def _token_sha256(input_ids: Any) -> str:
    import torch

    tensor = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(shape + b"\x00" + tensor.numpy().tobytes()).hexdigest()


def _prepare_case(
    tokenizer: Any,
    row: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    query_config: Mapping[str, Any],
) -> PreparedCase:
    rendered = _render_prompt(tokenizer, row, prompt_config)
    input_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    context_ids = tokenizer(str(row["context"]), add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    query_ids = tokenizer(str(row["question"]), add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[-1] <= 0:
        raise RuntimeError(f"row {row.get('id')} produced invalid prompt token shape")
    if query_ids.shape[-1] <= 0:
        raise RuntimeError(f"row {row.get('id')} produced an empty query")
    query_limit = int(query_config.get("max_tokens", 0))
    if query_limit <= 0 or query_ids.shape[-1] > query_limit:
        raise RuntimeError(
            f"row {row.get('id')} has {query_ids.shape[-1]} query tokens, limit is {query_limit}; "
            "the evaluator refuses to truncate query access"
        )
    metadata = {
        name: row.get(name)
        for name in (
            "domain",
            "task_type",
            "difficulty",
            "needle_profile",
            "length_bucket",
        )
    }
    return PreparedCase(
        case_id=str(row["id"]),
        scenario_id=str(row["scenario_id"]) if row.get("scenario_id") is not None else None,
        metadata=metadata,
        input_ids=input_ids.contiguous(),
        context_ids=context_ids.contiguous(),
        query_ids=query_ids.contiguous(),
        prompt_text_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        input_token_sha256=_token_sha256(input_ids),
        context_token_sha256=_token_sha256(context_ids),
        query_token_sha256=_token_sha256(query_ids),
    )


def _cache_bytes(cache: SignificanceCache) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return int(total)


def _resident_schedule(
    *,
    input_tokens: int,
    chunk_size: int,
    retained_history_budget: int,
    query_tokens: int,
    decode_cache_appends: int,
) -> dict[str, Any]:
    chunk_sizes = [min(chunk_size, input_tokens - start) for start in range(0, input_tokens, chunk_size)]
    resident = 0
    peak_with_probe = 0
    boundaries: list[dict[str, int]] = []
    for index, size in enumerate(chunk_sizes):
        resident += size
        peak_with_probe = max(peak_with_probe, resident)
        if index < len(chunk_sizes) - 1:
            before = resident
            peak_with_probe = max(peak_with_probe, resident + query_tokens)
            resident = min(resident, retained_history_budget)
            boundaries.append(
                {
                    "boundary_index": index,
                    "history_plus_chunk_before_compression": before,
                    "temporary_query_entries": query_tokens,
                    "resident_after_compression": resident,
                }
            )
    after_prefill = resident
    return {
        "input_tokens": input_tokens,
        "chunk_size": chunk_size,
        "chunk_count": len(chunk_sizes),
        "chunk_sizes": chunk_sizes,
        "compression_boundaries": len(boundaries),
        "boundary_accounting": boundaries,
        "retained_history_budget": retained_history_budget,
        "history_entries_after_last_boundary": boundaries[-1]["resident_after_compression"] if boundaries else 0,
        "final_active_chunk_entries": chunk_sizes[-1],
        "query_tokens_per_probe": query_tokens,
        "query_probe_token_evaluations": query_tokens * len(boundaries),
        "max_transient_resident_entries": peak_with_probe,
        "resident_entries_after_prefill_expected": after_prefill,
        "decode_cache_appends": decode_cache_appends,
        "resident_entries_after_decode_expected": after_prefill + decode_cache_appends,
    }


def _manual_greedy_decode(
    model: Any,
    cache: SignificanceCache,
    first_logits: Any,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> dict[str, Any]:
    """Decode with explicit argmax and one-token forwards; never call generate()."""

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
        token = __import__("torch").tensor([[token_id]], dtype=__import__("torch").long)
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
        decode_cache_appends += 1
    return {
        "token_ids": generated,
        "generated_tokens": len(generated),
        "decode_cache_appends": decode_cache_appends,
        "stop_reason": stop_reason,
    }


def _policy_for(
    condition: Condition,
    cache_config: Mapping[str, Any],
    *,
    full_cache_budget: int,
) -> SignificancePolicy:
    if condition.is_full_cache:
        return SignificancePolicy(
            budget=full_cache_budget,
            sinks=0,
            recent=0,
            count_exponent=0.0,
            pool_width=1,
            require_heavy=False,
        )
    # `score_pool_width` stays absent from every Cornerstone-1 config, so this
    # defaults to 1 and the isolated Cornerstone-1 path is unchanged. Screens
    # that deliberately study contiguous-span preservation (a Cornerstone-2
    # factor) may set it, and must report it as a separate factor.
    return SignificancePolicy(
        budget=int(cache_config["retained_history_budget"]),
        sinks=int(cache_config["sink_tokens"]),
        recent=int(cache_config["recent_tokens"]),
        count_exponent=float(condition.exponent),
        pool_width=int(cache_config.get("score_pool_width", 1)),
        require_heavy=True,
        ranking_mode="position" if condition.policy == "position" else "significance",
    )


def execute_condition(
    model: Any,
    prepared: PreparedCase,
    condition: Condition,
    *,
    cache_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    eos_token_ids: set[int],
    return_resident_positions: bool = False,
    compression_callback: Callable[[SignificanceCache, SignificancePolicy, int], Any] | None = None,
) -> dict[str, Any]:
    """Execute one closed-loop condition using the shared runtime adapter."""

    import psutil

    input_tokens = int(prepared.input_ids.shape[-1])
    query_tokens = int(prepared.query_ids.shape[-1])
    max_new_tokens = int(runtime_config["max_new_tokens"])
    chunk_size = int(cache_config["compression_chunk_tokens"])
    full_budget = input_tokens + max_new_tokens
    policy = _policy_for(condition, cache_config, full_cache_budget=full_budget)
    cache = SignificanceCache(config=model.config)
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]

    prefill_started = time.perf_counter()
    prefill = chunked_prefill(
        model,
        prepared.input_ids,
        cache,
        chunk_size=chunk_size,
        policy=policy,
        query_ids=prepared.query_ids,
        observation_window=query_tokens,
        verify_query_rollback=bool(runtime_config.get("verify_query_rollback", True)),
        collect_logits=False,
        logits_to_keep=1,
        compression_callback=compression_callback,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    cache.assert_aligned()
    resident_after_prefill = cache.stored_length()
    rss_samples.append(process.memory_info().rss)

    decode_started = time.perf_counter()
    decoded = _manual_greedy_decode(
        model,
        cache,
        prefill.output.logits,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    decode_seconds = time.perf_counter() - decode_started
    cache.assert_aligned()
    resident_after_decode = cache.stored_length()
    rss_samples.append(process.memory_info().rss)

    schedule = _resident_schedule(
        input_tokens=input_tokens,
        chunk_size=chunk_size,
        retained_history_budget=policy.budget,
        query_tokens=query_tokens,
        decode_cache_appends=int(decoded["decode_cache_appends"]),
    )
    schedule["retained_history_budget_role"] = (
        "no_eviction_full_cache_capacity" if condition.is_full_cache else "bounded_history"
    )
    if resident_after_prefill != schedule["resident_entries_after_prefill_expected"]:
        raise RuntimeError(
            f"resident prefill accounting mismatch: runtime={resident_after_prefill}, schedule={schedule}"
        )
    if resident_after_decode != schedule["resident_entries_after_decode_expected"]:
        raise RuntimeError(
            f"resident decode accounting mismatch: runtime={resident_after_decode}, schedule={schedule}"
        )
    expected_events = int(schedule["compression_boundaries"])
    scoring_events = cache.scoring_events()
    if scoring_events != [expected_events] * len(cache.layers):
        raise RuntimeError(
            f"query probe accounting mismatch: expected {expected_events} events/layer, got {scoring_events}"
        )

    layer_count = len(cache.layers)
    reports = []
    for report in prefill.compression_reports:
        if report.before_entries % layer_count or report.after_entries % layer_count or report.evictions % layer_count:
            raise RuntimeError("compression report does not divide evenly across model layers")
        reports.append(
            {
                "compressed_layers": report.layers,
                "before_entries_per_layer": report.before_entries // layer_count,
                "after_entries_per_layer": report.after_entries // layer_count,
                "evictions_per_layer": report.evictions // layer_count,
            }
        )
    effective_evictions = sum(report["evictions_per_layer"] for report in reports)
    if condition.is_full_cache and effective_evictions:
        raise RuntimeError("B0 full-cache control evicted entries")
    if not condition.is_full_cache and input_tokens > policy.budget and effective_evictions <= 0:
        raise RuntimeError(f"{condition.condition_id} did not perform an effective eviction")

    kv_bytes = _cache_bytes(cache)
    matrix_bytes = cache.matrix_bytes()
    metadata_bytes = cache.metadata_bytes()
    total_seconds = prefill_seconds + decode_seconds
    result = {
        "condition": condition.condition_id,
        "policy": condition.policy,
        "significance_exponent": condition.exponent,
        "query_conditioned": condition.query_conditioned,
        "query_probe_role": "instrumentation_only" if condition.is_full_cache else "ranking_signal",
        "generated_token_ids": decoded["token_ids"],
        "generated_tokens": decoded["generated_tokens"],
        "stop_reason": decoded["stop_reason"],
        "manual_greedy": True,
        "accounting": {
            **schedule,
            "resident_entries_after_prefill": resident_after_prefill,
            "resident_entries_after_decode": resident_after_decode,
            "scoring_events_per_layer": scoring_events,
            "compression_reports": reports,
            "effective_evictions_per_layer": effective_evictions,
        },
        "efficiency": {
            "kv_only_bytes": kv_bytes,
            "significance_matrix_bytes": matrix_bytes,
            "alignment_metadata_bytes": metadata_bytes,
            "runtime_total_persistent_bytes": kv_bytes + matrix_bytes + metadata_bytes,
            "architecture_total_persistent_bytes": (
                kv_bytes if condition.is_full_cache else kv_bytes + matrix_bytes + metadata_bytes
            ),
            "total_persistent_bytes": (
                kv_bytes if condition.is_full_cache else kv_bytes + matrix_bytes + metadata_bytes
            ),
            "sampled_peak_rss_bytes": max(rss_samples),
            "peak_rss_bytes": max(rss_samples),
            "peak_vram_bytes": 0,
            "wall_seconds": total_seconds,
        },
        "timing": {
            "prefill_wall_seconds": prefill_seconds,
            "decode_wall_seconds": decode_seconds,
            "inference_wall_seconds": total_seconds,
        },
    }
    if return_resident_positions:
        result["_resident_positions"] = [
            layer.state.positions[0].detach().cpu().numpy()
            for layer in cache.layers
            if layer.state is not None
        ]
    return result


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_unscored_predictions(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_pairs = [(case_id, condition_id) for case_id in CASE_IDS for condition_id in CONDITION_IDS]
    observed_pairs = [(str(row.get("case_id")), str(row.get("condition"))) for row in rows]
    if observed_pairs != expected_pairs:
        raise RuntimeError(f"unscored prediction order/coverage mismatch: {observed_pairs}")
    forbidden = FORBIDDEN_PROMPT_FIELDS | {
        "correct",
        "full_cache_correct",
        "compression_conditioned_eligible",
    }
    for row in rows:
        exposed = forbidden.intersection(row)
        if exposed:
            raise RuntimeError(f"unscored prediction contains answer-derived fields: {sorted(exposed)}")


def _read_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in _load_jsonl(path):
        row_id = str(row.get("id", ""))
        if row_id not in CASE_IDS:
            continue
        if row_id in selected:
            duplicates.add(row_id)
        if not isinstance(row.get("answer"), str):
            raise RuntimeError(f"answer-key row {row_id} has no string answer")
        accepted = row.get("accepted_answers", [row["answer"]])
        if not isinstance(accepted, list) or not accepted or not all(isinstance(v, str) for v in accepted):
            raise RuntimeError(f"answer-key row {row_id} has invalid accepted_answers")
        selected[row_id] = row
    if duplicates:
        raise RuntimeError(f"duplicate selected answer-key IDs: {sorted(duplicates)}")
    missing = [case_id for case_id in CASE_IDS if case_id not in selected]
    if missing:
        raise RuntimeError(f"missing selected answer-key IDs: {missing}")
    return selected


def blind_inference_then_load_key(
    inference: Callable[[], list[dict[str, Any]]],
    *,
    unscored_path: Path,
    answer_key_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Enforce the inference/key phase boundary in one testable operation."""

    predictions = inference()
    _assert_unscored_predictions(predictions)
    _atomic_write_jsonl(unscored_path, predictions)
    # Deliberately first answer-key access in the run: after all inference and
    # after a durable, answer-blind prediction artifact exists.
    answer_key = _read_answer_key(answer_key_path)
    return predictions, answer_key


def _normalize_answer(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", value)
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    answer_key: Mapping[str, Mapping[str, Any]],
    *,
    run_id: str,
    seed: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Join answer-derived fields only after blind inference is complete."""

    _assert_unscored_predictions(predictions)
    correctness: dict[tuple[str, str], bool] = {}
    for row in predictions:
        case_id = str(row["case_id"])
        accepted = {_normalize_answer(value) for value in answer_key[case_id]["accepted_answers"]}
        correctness[(case_id, str(row["condition"]))] = _normalize_answer(
            str(row["prediction"])
        ) in accepted

    scored: list[dict[str, Any]] = []
    for source in predictions:
        row = dict(source)
        case_id = str(row["case_id"])
        condition = str(row["condition"])
        correct = correctness[(case_id, condition)]
        full_correct = correctness[(case_id, "B0")]
        eligible = full_correct if condition != "B0" else None
        failure_labels: list[str] = []
        if not full_correct:
            failure_labels.append("FULL_CACHE_CAPABILITY_FAILURE")
        elif not correct:
            failure_labels.append("UNKNOWN")
        row.update(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "scenario_id": answer_key[case_id].get("scenario_id"),
                "task_family": "R0",
                "generator_seed": seed,
                "prompt_sha256": row["input_token_sha256"],
                "chunk_size": chunk_size,
                "status": "OK",
                "correct": correct,
                "full_cache_correct": full_correct,
                "compression_conditioned_eligible": eligible,
                "full_cache_conditioned_correct": correct if eligible else None,
                "reference_answer": answer_key[case_id]["answer"],
                "accepted_answers": answer_key[case_id]["accepted_answers"],
                "failure_labels": failure_labels,
                "trace_ids": [],
                "error": None,
            }
        )
        scored.append(row)
    return scored


def _descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in CONDITION_IDS:
        selected = [row for row in rows if row["condition"] == condition]
        correct = sum(bool(row["correct"]) for row in selected)
        condition_summary: dict[str, Any] = {
            "cases": len(selected),
            "correct_cases": correct,
            "exact_accuracy": correct / len(selected) if selected else None,
            "total_inference_wall_seconds": sum(
                float(row["timing"]["inference_wall_seconds"]) for row in selected
            ),
        }
        if condition != "B0":
            eligible = [row for row in selected if row["compression_conditioned_eligible"]]
            eligible_correct = sum(bool(row["correct"]) for row in eligible)
            condition_summary["full_cache_conditioned_correctness"] = {
                "eligible_cases": len(eligible),
                "correct_cases": eligible_correct,
                "accuracy": eligible_correct / len(eligible) if eligible else None,
            }
        else:
            condition_summary["full_cache_conditioned_correctness"] = {
                "applicable": False,
                "reason": "B0 is the full-cache capability reference",
            }
        by_condition[condition] = condition_summary
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "BOUNDED_EXPLORATORY",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only (Qwen3-0.6B source base)",
        "case_ids": list(CASE_IDS),
        "conditions": list(CONDITION_IDS),
        "answer_key_opened_only_after_all_inference": True,
        "manual_greedy_decode": True,
        "descriptive_results": by_condition,
        "inferential_statistics": None,
        "interpretation_limits": [
            "Exactly three preselected exploratory cases were evaluated.",
            "No confidence interval, hypothesis test, or population-level claim is reported.",
            "B0 is an instrumented full-cache behavioral control; its temporary query probes are rolled back and do not select cache entries.",
            "CPU timing is local-run accounting, not a deployment or GPU latency claim.",
        ],
    }


def _schema_validate(instance: Mapping[str, Any], schema_path: Path) -> None:
    import jsonschema

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


def _manifest_document(
    *,
    config: Mapping[str, Any],
    run_id: str,
    resolved_config_path: Path,
    model_path: Path,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    benchmark_hashes: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, str],
    completed_at: str,
) -> dict[str, Any]:
    standard_hash_names = {
        "weights": "weights",
        "tokenizer": "tokenizer",
        "config": "config",
        "chat_template": "chat_template",
    }
    files = checkpoint["files"]
    standard_hashes = {
        target: files[source]["sha256"] if source in files else None
        for target, source in standard_hash_names.items()
    }
    manifest = build_manifest(
        repo=REPO_ROOT,
        config=config,
        run_id=run_id,
        execution_mode="closed_loop_model_execution",
        architecture_variant="B0 full cache / B2 e=0 / B3 e=1",
        seed=int(config["runtime"]["seed"]),
        compute_tier="B",
        status="COMPLETE",
        evidence_class="NEW_CHATBUBBLE_EVIDENCE",
        resolved_config_path=artifacts["resolved_config"],
        benchmark={"name": str(config["benchmark"]["name"])},
        model={
            "status": "VERIFIED",
            "id": str(core["core_id"]),
            "path": _repo_path(model_path),
            "architecture": str(config["model"]["expected_architecture"]),
            "hashes": standard_hashes,
            "checkpoint_files": checkpoint["files"],
            "checkpoint_aggregate_sha256": checkpoint["aggregate_sha256"],
        },
    )
    manifest["completed_at"] = completed_at
    manifest["config"] = {
        "resolved_path": artifacts["resolved_config"],
        "sha256": sha256_file(resolved_config_path),
        "source_path": str(config["_source_config_path"]).replace("\\", "/"),
    }
    manifest["runtime"].update(
        {
            "device": "cpu",
            "precision": "bfloat16",
            "manual_greedy": True,
            "attention_implementation": "sdpa",
            "platform": platform.platform(),
        }
    )
    by_condition: dict[str, Any] = {}
    for condition in CONDITION_IDS:
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "resident_entries_after_prefill": [
                row["accounting"]["resident_entries_after_prefill"] for row in selected
            ],
            "resident_entries_after_decode": [
                row["accounting"]["resident_entries_after_decode"] for row in selected
            ],
            "kv_only_bytes_after_decode": [row["efficiency"]["kv_only_bytes"] for row in selected],
            "total_persistent_bytes_after_decode": [
                row["efficiency"]["architecture_total_persistent_bytes"] for row in selected
            ],
        }
    manifest["budget"] = {
        "resident_entries": None,
        "kv_only_bytes": None,
        "total_persistent_bytes": None,
        "matched": False,
        "comparison_note": "B2 and B3 share the same resident-entry budget; B0 is full cache.",
        "by_condition": by_condition,
    }
    manifest["benchmark"] = {
        "case_ids": list(CASE_IDS),
        "prompt_sha256": benchmark_hashes["prompts"],
        "answer_key_sha256_after_inference": benchmark_hashes["answer_key"],
        "answer_key_join_order": "opened_after_unscored_predictions_were_written",
    }
    manifest["artifacts"] = dict(artifacts)
    manifest["notes"] = [
        "Bounded exploratory run: exactly three cases and no inferential statistics.",
        "All conditions use identical prompt tokens, chunking, query tokens, query-probe boundaries, and manual greedy decoding.",
        "B0 query probes are instrumentation-only because no entries are evicted.",
        "No R2, Aegis, Halo Uncensored, or other checkpoint is in scope.",
        "No broad model-family or architecture-quality claim is supported by this run.",
    ]
    return manifest


def _create_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite existing evaluation directory: {path}") from exc


def _write_resolved_config(path: Path, config: Mapping[str, Any]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config_path = config_path.expanduser().resolve()
    config = _load_mapping(config_path)
    conditions = validate_config(config)
    config["_source_config_path"] = _repo_path(config_path)
    base = _declared_base(config)
    paths = config["paths"]
    model_path = _resolve_path(paths["model"], base=base)
    tokenizer_path = _resolve_path(paths.get("tokenizer", paths["model"]), base=base)
    prompt_path = _resolve_path(paths["prompts"], base=base)
    # Resolve only: the answer-key file is intentionally neither opened nor
    # hashed until blind inference has completed.
    answer_key_path = _resolve_path(paths["answer_key"], base=base)
    output = output_override.expanduser().resolve() if output_override else _resolve_path(paths["output_root"], base=base)
    _create_output_dir(output)

    outputs = config["outputs"]
    artifact_paths = {name: output / str(filename) for name, filename in outputs.items()}
    required_artifacts = {
        "resolved_config",
        "manifest",
        "cases",
        "summary",
        "unscored_predictions",
    }
    if set(artifact_paths) != required_artifacts:
        raise ValueError(f"outputs must declare exactly {sorted(required_artifacts)}")
    _write_resolved_config(artifact_paths["resolved_config"], config)

    core_path = model_path / "CORE.json"
    if tokenizer_path != model_path:
        raise RuntimeError("Halo mini evaluation requires the tokenizer packaged with the exact core")
    if not core_path.is_file():
        raise FileNotFoundError(f"published core metadata is missing: {core_path}")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    _validate_halo_core(core, config["model"])
    checkpoint = _checkpoint_hashes(model_path, config["model"])

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    _check_geometry(model.config, config["model"])

    prompt_rows = _read_prompt_rows(prompt_path)
    prepared_cases = [
        _prepare_case(tokenizer, row, config["prompt"], config["query"]) for row in prompt_rows
    ]
    max_position_embeddings = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    max_new_tokens = int(config["runtime"]["max_new_tokens"])
    for prepared in prepared_cases:
        if max_position_embeddings and prepared.input_ids.shape[-1] + max_new_tokens > max_position_embeddings:
            raise RuntimeError(
                f"{prepared.case_id} exceeds model context limit; this evaluator refuses truncation"
            )

    eos = tokenizer.eos_token_id
    eos_token_ids = {int(value) for value in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())
    run_id = output.name

    def infer_all() -> list[dict[str, Any]]:
        unscored: list[dict[str, Any]] = []
        for prepared in prepared_cases:
            invariant = (
                prepared.input_token_sha256,
                prepared.context_token_sha256,
                prepared.query_token_sha256,
            )
            for condition in conditions:
                result = execute_condition(
                    model,
                    prepared,
                    condition,
                    cache_config=config["cache"],
                    runtime_config=config["runtime"],
                    eos_token_ids=eos_token_ids,
                )
                if invariant != (
                    prepared.input_token_sha256,
                    prepared.context_token_sha256,
                    prepared.query_token_sha256,
                ):
                    raise RuntimeError("shared token tensors changed between conditions")
                prediction = tokenizer.decode(result.pop("generated_token_ids"), skip_special_tokens=True).strip()
                unscored.append(
                    {
                        "case_id": prepared.case_id,
                        "scenario_id": prepared.scenario_id,
                        "condition": condition.condition_id,
                        "prediction": prediction,
                        "prompt_text_sha256": prepared.prompt_text_sha256,
                        "input_token_sha256": prepared.input_token_sha256,
                        "context_token_sha256": prepared.context_token_sha256,
                        "query_token_sha256": prepared.query_token_sha256,
                        "input_tokens": int(prepared.input_ids.shape[-1]),
                        "context_tokens": int(prepared.context_ids.shape[-1]),
                        "query_tokens": int(prepared.query_ids.shape[-1]),
                        "benchmark_metadata": prepared.metadata,
                        **result,
                    }
                )
                gc.collect()
        return unscored

    unscored, answer_key = blind_inference_then_load_key(
        infer_all,
        unscored_path=artifact_paths["unscored_predictions"],
        answer_key_path=answer_key_path,
    )
    scored = score_predictions(
        unscored,
        answer_key,
        run_id=run_id,
        seed=seed,
        chunk_size=int(config["cache"]["compression_chunk_tokens"]),
    )
    case_schema = LAB_ROOT / "schemas" / "case.schema.json"
    for row in scored:
        _schema_validate(row, case_schema)
    _atomic_write_jsonl(artifact_paths["cases"], scored)

    summary = _descriptive_summary(scored)
    _atomic_write_json(artifact_paths["summary"], summary)
    artifacts = {name: path.name for name, path in artifact_paths.items()}
    benchmark_hashes = {
        "prompts": sha256_file(prompt_path),
        # This is intentionally computed only after the post-inference key load.
        "answer_key": sha256_file(answer_key_path),
    }
    manifest = _manifest_document(
        config=config,
        run_id=run_id,
        resolved_config_path=artifact_paths["resolved_config"],
        model_path=model_path,
        core=core,
        checkpoint=checkpoint,
        benchmark_hashes=benchmark_hashes,
        rows=scored,
        artifacts=artifacts,
        completed_at=utc_now(),
    )
    manifest["created_at"] = started_at
    _schema_validate(manifest, LAB_ROOT / "schemas" / "manifest.schema.json")
    _atomic_write_json(artifact_paths["manifest"], manifest)
    result = {"output": str(output), "manifest": manifest, "summary": summary}
    print(json.dumps({"output": str(output), "status": summary["status"]}, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_mini_eval.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh output directory override; existing paths are always rejected.",
    )
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
