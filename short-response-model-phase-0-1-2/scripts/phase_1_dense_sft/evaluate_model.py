from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from rapidfuzz.fuzz import ratio as fuzzy_ratio
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.response_control import (
    RESPONSE_CONTROL_MODES,
    build_stopping_criteria,
    finalize_response,
    infer_stop_reason,
)
from scripts.common.experiment_utils import (
    approved_content_only_migration,
    atomic_write_text,
    read_json,
    sha256,
    write_json,
    write_json_best_effort,
    write_jsonl,
)
from scripts.common.incremental_results import DurableJsonlStore
from scripts.common.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
    aggregate_quality_rows,
    build_corpus_profile,
    complete_utterance as quality_complete_utterance,
    evaluate_response,
    load_quality_config,
    normalize_text as quality_normalize_text,
    token_f1 as quality_token_f1,
    word_tokens as quality_word_tokens,
)

# PEFT is optional for base-model evaluation. When an adapter path is
# supplied later, the same evaluator can attach the LoRA/QLoRA adapter.
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]

# Shared default for the untouched baseline and adapted comparisons.
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
# Resolve the pilot benchmark relative to the project root so the command
# works regardless of the caller's current directory.
DEFAULT_BENCHMARK = (
    ROOT
    / "data"
    / "benchmark"
    / "shortchatbench_pilot.jsonl"
)
# This stable policy tells every evaluated model what short-response
# behavior is expected. It is not itself a learned adapter.
DEFAULT_SYSTEM_PROMPT = (
    "Reply like a casual text message. Use only a few words or one short sentence. "
    "Be accurate, use the conversation context, ask one brief clarification when "
    "information is missing, and keep safety boundaries concise."
)


# Configure model/adapter selection, benchmark location, decoding, and
# output paths from the command line.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a base model or PEFT adapter.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "baseline_evaluation")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--response-control",
        choices=RESPONSE_CONTROL_MODES,
        default="eos_only",
        help=(
            "eos_only measures learned behavior without an external sentence boundary; "
            "one_sentence adds the production runtime guarantee."
        ),
    )
    p.add_argument(
        "--min-new-tokens-before-boundary",
        type=int,
        default=1,
        help="Do not allow punctuation-based stopping before this many generated tokens.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--quality-config",
        type=Path,
        default=ROOT / "configs" / "quality_evaluation.json",
        help="Versioned observable-defect and explicit response-contract policy.",
    )
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of using CPU when CUDA is unavailable.",
    )
    p.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Override checkpoint dtype at load time (bfloat16 fits 28L Qwen3 on 8GB).",
    )
    p.add_argument("--resume", action="store_true", help="Reuse durable completed example rows by stable benchmark ID.")
    p.add_argument("--run-root", type=Path, default=None, help="Optional Phase 2B run root for LIVE_PROGRESS files.")
    p.add_argument("--progress-context", type=Path, default=None, help="JSON context merged into Phase 2B live progress.")
    p.add_argument("--candidate-status", type=Path, default=None, help="Candidate status file updated during evaluation.")
    p.add_argument("--progress-update-interval", type=int, default=5, help="Refresh live Markdown every N completed examples; JSON updates every example.")
    p.add_argument("--disable-live-markdown", action="store_true", help="Disable LIVE_PROGRESS.md updates while retaining JSON progress.")
    return p.parse_args()


# Parse benchmark records and report the exact line for malformed JSON.
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def normalize_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize pilot and SFT/sealed rows into one evaluator contract.

    SFT-style rows end with the reference assistant answer.  That answer must
    never be included in the generation prompt; doing so would leak the sealed
    target.  The reference is retained only for post-generation diagnostics.
    """

    normalized = dict(record)
    messages = [dict(message) for message in record.get("messages", [])]
    reference = record.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    metadata = record.get("metadata") or {}
    normalized["messages"] = messages
    normalized["reference"] = reference
    normalized["category"] = str(
        record.get("category") or metadata.get("reply_class") or "uncategorized"
    )
    if not isinstance(record.get("expected"), dict):
        normalized["expected"] = {
            "min_tokens": int(metadata.get("target_token_min", 1)),
            "max_tokens": int(metadata.get("target_token_max", 32)),
        }
    return normalized


def reference_token_f1(reference: str | None, reply: str) -> float | None:
    """Backward-compatible lexical diagnostic supplied by the v2 evaluator."""

    return quality_token_f1(reference, reply)


# Backward-compatible wrappers used by a few phase scripts and tests.
def normalize(text: str) -> str:
    return quality_normalize_text(text)


def words(text: str) -> list[str]:
    return quality_word_tokens(text)


def contains(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def complete_utterance(reply: str) -> bool:
    return quality_complete_utterance(reply)


# Measure surface similarity between the user message and model reply.
# High similarity can indicate parroting rather than a genuine response.
def echo_score(user: str, reply: str) -> float:
    if not user.strip() or not reply.strip():
        return 0.0
    return float(fuzzy_ratio(normalize(user), normalize(reply)))


# Backward-compatible entry point. New callers should use evaluate_response()
# directly so quality and semantic fields are produced in one pass.
def score_case(record: dict[str, Any], reply: str, token_count: int) -> dict[str, Any]:
    return evaluate_response(record, reply, token_count, visible_model_tokens=token_count)


def semantic_audit(record: dict[str, Any], reply: str) -> dict[str, Any]:
    evaluated = evaluate_response(
        record,
        reply,
        max(1, len(quality_word_tokens(reply))),
        visible_model_tokens=max(1, len(quality_word_tokens(reply))),
    )
    return {
        "critical_failures": evaluated["critical_failures"],
        "major_issues": evaluated["major_issues"],
        "semantic_warnings": evaluated["semantic_warnings"],
        "semantic_severity": evaluated["semantic_severity"],
        "semantic_pass": evaluated["semantic_pass"],
    }


# Use a nearest-rank percentile, which is simple and deterministic for
# the small pilot benchmark.
def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return float(ordered[index])


# Synchronize asynchronous GPU work before and after timing.
def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Render a generation prompt across Qwen and generic chat templates."""

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# Generate one reply for every benchmark record, score it, and write raw
# results plus aggregate JSON and Markdown reports. Results are appended and
# fsynced per example so interruption never leaves completed evidence only in
# memory.
def build_evaluation_summary(outputs: list[dict[str, Any]], args: argparse.Namespace, quality_config: Any) -> dict[str, Any]:
    token_counts = [float(r["visible_generated_tokens"]) for r in outputs]
    actual_token_counts = [float(r["generated_tokens"]) for r in outputs]
    stop_reasons = Counter(str(r["stop_reason"]) for r in outputs)
    times = [float(r["elapsed_seconds"]) for r in outputs]
    repeats = Counter(normalize(r["reply"]) for r in outputs)
    quality_summary = aggregate_quality_rows(outputs)
    return {
        "case_count": len(outputs),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "quality_policy_name": quality_config.raw.get("policy_name"),
        "quality_config": str(args.quality_config),
        "quality_config_sha256": hashlib.sha256(args.quality_config.read_bytes()).hexdigest(),
        "response_control_mode": args.response_control,
        "min_new_tokens_before_boundary": args.min_new_tokens_before_boundary,
        "max_new_tokens": args.max_new_tokens,
        "median_reply_tokens": statistics.median(token_counts),
        "p90_reply_tokens": percentile(token_counts, 0.90),
        "median_actual_generated_tokens": statistics.median(actual_token_counts),
        "p90_actual_generated_tokens": percentile(actual_token_counts, 0.90),
        "one_sentence_violation_rate": statistics.mean(1.0 if r["one_sentence_violation"] else 0.0 for r in outputs),
        "response_control_trim_rate": statistics.mean(1.0 if r["response_control_trimmed"] else 0.0 for r in outputs),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "mean_generation_seconds": statistics.mean(times),
        "p90_generation_seconds": percentile(times, 0.90),
        "peak_vram_gb": max(float(r["peak_vram_gb"]) for r in outputs),
        "duplicate_reply_count": sum(c - 1 for c in repeats.values() if c > 1),
        "duplicate_reply_rate": sum(c - 1 for c in repeats.values() if c > 1) / max(1, len(outputs)),
        **quality_summary,
    }
    
def escape_markdown_cell(value: Any) -> str:
    """Make arbitrary text safe for a single Markdown table cell."""

    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def format_optional_number(value: Any, format_spec: str = ".3f", *, fallback: str = "N/A") -> str:
    """Format numeric evidence without crashing when a metric is unavailable."""

    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return format(number, format_spec)


def render_evaluation_report(args: argparse.Namespace, summary: dict[str, Any], outputs: list[dict[str, Any]]) -> str:
    category_quality = summary.get("category_quality_scores") or {}
    category_quality_pass = summary.get("category_quality_pass_rates") or {}
    category_intelligibility = summary.get("category_intelligibility_pass_rates") or {}
    category_observable = summary.get(
        "category_observable_quality_pass_rates",
        summary.get("category_semantic_pass_rates") or {},
    ) or {}
    category_alignment = summary.get("category_reference_alignment_scores") or {}
    category_complete = summary.get("category_complete_utterance_rates") or {}
    category_hard = summary.get("category_hard_failure_case_counts") or {}
    category_major = summary.get("category_major_case_counts") or {}
    cat_rows = "\n".join(
        f"| {escape_markdown_cell(category)} | {format_optional_number(score)} | "
        f"{format_optional_number(category_quality_pass.get(category))} | "
        f"{format_optional_number(category_intelligibility.get(category))} | "
        f"{format_optional_number(category_observable.get(category))} | "
        f"{format_optional_number(category_alignment.get(category))} | "
        f"{format_optional_number(category_complete.get(category))} | "
        f"{category_hard.get(category, 0)} | {category_major.get(category, 0)} |"
        for category, score in category_quality.items()
    ) or "| None | N/A | N/A | N/A | N/A | N/A | N/A | 0 | 0 |"
    worst = sorted(
        outputs,
        key=lambda row: (
            float(row.get("quality_score") or 0.0),
            bool(row.get("semantic_pass", row.get("quality_pass", False))),
        ),
    )[:15]
    worst_rows = "\n".join(
        f"| {escape_markdown_cell(row.get('id', 'unknown'))} | "
        f"{escape_markdown_cell(row.get('category', 'uncategorized'))} | "
        f"{format_optional_number(row.get('quality_score'))} | "
        f"{format_optional_number(row.get('legacy_heuristic_score'))} | "
        f"{escape_markdown_cell(row.get('reply', ''))} |"
        for row in worst
    ) or "| None | - | N/A | N/A | - |"
    flagged = [
        row
        for row in outputs
        if row.get("semantic_severity", "none") != "none" or row.get("review_diagnostics")
    ]
    defect_rows = "\n".join(
        f"| {escape_markdown_cell(row.get('id', 'unknown'))} | "
        f"{escape_markdown_cell(row.get('semantic_severity', 'none'))} | "
        f"{escape_markdown_cell(', '.join(map(str, row.get('critical_failures') or [])) or '-')} | "
        f"{escape_markdown_cell(', '.join(map(str, row.get('major_issues') or [])) or '-')} | "
        f"{escape_markdown_cell(', '.join(map(str, row.get('semantic_warnings') or row.get('review_diagnostics') or [])) or '-')} |"
        for row in flagged
    ) or "| None | none | - | - | - |"
    return f"""# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `{args.model_id}`
- Adapter: `{args.adapter_path or 'None (base model)'}`
- Benchmark: `{args.benchmark}`
- Cases: {summary['case_count']}
- Deterministic decoding: {not args.do_sample}
- Response control: `{args.response_control}`
- Quality evaluation version: `{summary['quality_evaluation_version']}`
- Quality config: `{args.quality_config}`

## Aggregate Quality

| Metric | Value |
|---|---:|
| Mean quality score | {summary['mean_quality_score']:.3f} |
| Legacy three-check mean | {summary['mean_legacy_heuristic_score']:.3f} |
| Quality pass rate | {summary['quality_pass_rate']:.3f} |
| Intelligibility pass rate | {summary['intelligibility_pass_rate']:.3f} |
| Hard-failure cases | {summary['hard_failure_case_count']} |
| Zero-quality cases | {summary['zero_quality_case_count']} |
| Minimum case quality | {summary['minimum_quality_score']:.3f} |
| P10 case quality | {summary['p10_quality_score']:.3f} |
| Worst-decile mean quality | {summary['worst_decile_mean_quality_score']:.3f} |
| Observable defect-free rate | {summary.get('observable_quality_pass_rate', summary.get('semantic_pass_rate', 0.0)):.3f} |
| Critical failures | {summary['critical_failure_count']} |
| Major issues | {summary['major_issue_count']} |
| Complete-utterance rate | {summary['complete_utterance_rate']:.3f} |
| Mean reference alignment | {format_optional_number(summary.get('mean_reference_alignment_score'))} |
| Mean lexical reference token-F1 | {format_optional_number(summary.get('mean_reference_token_f1'))} |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | {summary['median_reply_tokens']} |
| P90 visible reply tokens | {summary['p90_reply_tokens']} |
| Median actual generated tokens | {summary['median_actual_generated_tokens']} |
| P90 actual generated tokens | {summary['p90_actual_generated_tokens']} |
| One-sentence violation rate | {summary['one_sentence_violation_rate']:.3f} |
| Runtime trim rate | {summary['response_control_trim_rate']:.3f} |
| Stop reasons | `{summary['stop_reason_counts']}` |
| Mean generation time (s) | {summary['mean_generation_seconds']:.3f} |
| P90 generation time (s) | {summary['p90_generation_seconds']:.3f} |
| Peak allocated VRAM (GB) | {summary['peak_vram_gb']:.3f} |
| Duplicate replies beyond first | {summary['duplicate_reply_count']} |
| Duplicate reply rate | {summary['duplicate_reply_rate']:.3f} |

## Mean Quality Dimensions

`{summary['mean_quality_dimensions']}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{cat_rows}

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
{worst_rows}

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
{defect_rows}

### Failure type counts

- Critical: `{summary['critical_failure_type_counts']}`
- Major: `{summary['major_issue_type_counts']}`
- Warnings: `{summary['warning_type_counts']}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
"""


def _canonical_manifest_identifier(value: Any) -> str | None:
    """Normalize local paths while preserving Hub/external model identifiers."""

    if value is None:
        return None
    text = str(value)
    candidate = Path(text)
    resolved = candidate if candidate.is_absolute() else ROOT / candidate
    if resolved.exists():
        try:
            return str(resolved.resolve()).casefold()
        except OSError:
            return str(resolved.absolute()).casefold()
    return text


def evaluation_manifest(args: argparse.Namespace, record_ids: list[str]) -> dict[str, Any]:
    ids_digest = hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest()
    return {
        "schema_version": 2,
        "model_id": str(args.model_id),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256(args.benchmark),
        "quality_config_sha256": sha256(args.quality_config),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "evaluator_script_sha256": sha256(Path(__file__)),
        "quality_evaluation_script_sha256": sha256(ROOT / "scripts/common/quality_evaluation.py"),
        "requested_limit": args.limit,
        "case_count": len(record_ids),
        "record_ids_sha256": ids_digest,
        "max_new_tokens": int(args.max_new_tokens),
        "response_control": args.response_control,
        "min_new_tokens_before_boundary": int(args.min_new_tokens_before_boundary),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature) if args.do_sample else None,
        "top_p": float(args.top_p) if args.do_sample else None,
        "system_prompt": args.system_prompt,
    }


def evaluation_manifest_mismatches(
    existing: dict[str, Any], requested: dict[str, Any]
) -> list[str]:
    """Compare manifests while accepting path spelling and legacy schema fields."""

    path_fields = {"model_id", "adapter_path", "benchmark"}
    optional_legacy_fields = {
        "requested_limit",
        "case_count",
        "record_ids_sha256",
        "temperature",
        "top_p",
        "evaluator_script_sha256",
        "quality_evaluation_script_sha256",
    }
    mismatches: list[str] = []
    for key, requested_value in requested.items():
        if key == "schema_version":
            continue
        if key not in existing:
            if key in optional_legacy_fields:
                continue
            mismatches.append(f"{key}: missing from existing manifest")
            continue
        existing_value = existing.get(key)
        if key in path_fields:
            equal = _canonical_manifest_identifier(existing_value) == _canonical_manifest_identifier(
                requested_value
            )
        else:
            equal = existing_value == requested_value
        if not equal:
            mismatches.append(
                f"{key}: existing={existing_value!r}, requested={requested_value!r}"
            )
    return mismatches


def rescore_migrated_rows(
    *,
    records: list[dict[str, Any]],
    store: DurableJsonlStore,
    quality_config: Any,
    corpus_profile: dict[str, Any],
    migration: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Rescore durable replies after an approved contract-only data migration."""

    records_by_id = {str(record["id"]): record for record in records}
    rescored: list[dict[str, Any]] = []
    for stable_id, row in store.rows_by_id.items():
        record = records_by_id.get(stable_id)
        if record is None:
            raise RuntimeError(f"Contract migration cannot map existing result ID {stable_id!r}.")
        if row.get("messages") != record.get("messages"):
            raise RuntimeError(
                f"Contract migration rejected: generation messages changed for {stable_id!r}."
            )
        if row.get("reference") != record.get("reference"):
            raise RuntimeError(
                f"Contract migration rejected: assistant target/reference changed for {stable_id!r}."
            )
        visible_tokens = row.get("visible_generated_tokens")
        try:
            visible_tokens = int(visible_tokens)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Contract migration cannot rescore {stable_id!r}: visible token count is missing."
            ) from exc
        reply = str(row.get("reply", ""))
        quality = evaluate_response(
            record,
            reply,
            visible_tokens,
            visible_model_tokens=visible_tokens,
            config=quality_config,
            corpus_profile=corpus_profile,
        )
        updated = dict(row)
        updated.update(
            category=record.get("category", "uncategorized"),
            messages=record.get("messages") or [],
            reference=record.get("reference"),
            reference_token_f1=reference_token_f1(record.get("reference"), reply),
        )
        updated.update(quality)
        rescored.append(updated)

    ordered_ids = [str(record["id"]) for record in records]
    rescored_by_id = {str(row["id"]): row for row in rescored}
    ordered = [rescored_by_id[stable_id] for stable_id in ordered_ids if stable_id in rescored_by_id]
    write_jsonl(store.path, ordered)
    store.rows_by_id = rescored_by_id
    write_json(
        output_dir / "QUALITY_CONTRACT_MIGRATION.json",
        {
            "schema_version": 1,
            "migration_type": "approved_contract_only_dataset_migration",
            "previous_benchmark_sha256": migration["previous_sha256"],
            "current_benchmark_sha256": migration["current_sha256"],
            "dataset_manifest": migration["manifest_path"],
            "stable_ids_messages_targets_unchanged": True,
            "rescored_existing_case_count": len(ordered),
            "generation_was_not_repeated": True,
            "migrated_at_unix": time.time(),
        },
    )
    print(f"[quality contract migration] rescored {len(ordered)} durable replies under current contracts")
    return ordered


def update_incremental_progress(
    args: argparse.Namespace,
    outputs: list[dict[str, Any]],
    *,
    total: int,
    status: str,
    force_markdown: bool = False,
    error: str | None = None,
) -> None:
    from scripts.phase_2_layer_selection.incremental_reporting import (
        provisional_metrics,
        transition_candidate_status,
        write_live_progress,
    )

    metrics = provisional_metrics(outputs)
    candidate_state = "evaluation_complete" if len(outputs) == total and status == "complete" else (
        "interrupted" if status == "interrupted" else "failed" if status == "failed" else "evaluating"
    )
    progress = {
        "schema_version": 1,
        "status": status,
        "examples_completed": len(outputs),
        "examples_total": total,
        "metrics_are_provisional": len(outputs) < total,
        "critical_count": metrics["critical_count"],
        "major_count": metrics["major_count"],
        "warning_count": metrics["warning_count"],
        "mean_quality_score": metrics["mean_quality_score"],
        "recent_defects": metrics["recent_defects"],
        "last_error": error,
        "last_update_time": time.time(),
    }
    write_json_best_effort(
        args.output_dir / "evaluation_progress.json",
        progress,
        label="evaluation progress",
    )
    if args.candidate_status is not None:
        try:
            transition_candidate_status(
                args.candidate_status,
                candidate_state,
                examples_completed=len(outputs),
                examples_total=total,
                critical_defects=metrics["critical_count"],
                major_defect_count=metrics["major_count"],
                warning_count=metrics["warning_count"],
                quality_metrics={"mean_quality_score": metrics["mean_quality_score"]},
                interruption=error if status == "interrupted" else None,
                failure=error if status == "failed" else None,
            )
        except OSError as exc:
            print(
                f"[warning] unable to update candidate status {args.candidate_status}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    if args.run_root is not None:
        context = (
            read_json(args.progress_context)
            if args.progress_context and args.progress_context.exists()
            else {}
        )
        markdown_due = (
            force_markdown
            or len(outputs) == total
            or len(outputs) % max(1, args.progress_update_interval) == 0
        )

        # Merge context and current evaluation data before expanding keyword
        # arguments. Current evaluator state intentionally overrides stale
        # values from progress_context.json.
        live_updates = dict(context)
        live_updates.update(
            examples_completed=len(outputs),
            examples_total=total,
            current_critical_count=metrics["critical_count"],
            current_major_count=metrics["major_count"],
            current_warning_count=metrics["warning_count"],
            provisional_mean_quality=metrics["mean_quality_score"],
            recent_defects=metrics["recent_defects"],
            metrics_are_provisional=len(outputs) < total,
            candidate_status=(
                "evaluation_complete"
                if len(outputs) == total
                else candidate_state
            ),
            most_recent_error=error,
        )

        try:
            write_live_progress(
                args.run_root,
                markdown=(not args.disable_live_markdown) and markdown_due,
                **live_updates,
            )
        except OSError as exc:
            print(
                f"[warning] unable to update live progress under {args.run_root}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def finalize_evaluation(args: argparse.Namespace, outputs: list[dict[str, Any]], quality_config: Any) -> None:
    summary = build_evaluation_summary(outputs, args, quality_config)
    write_json(args.output_dir / "summary.json", summary)
    report = render_evaluation_report(args, summary, outputs)
    report_path = args.output_dir / "MODEL_EVALUATION.md"
    atomic_write_text(report_path, report)
    atomic_write_text(args.output_dir / "BASELINE_EVALUATION.md", report)
    print(f"\nResults: {args.output_dir / 'results.jsonl'}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Report: {report_path}")


def main() -> None:
    args = parse_args()
    if args.progress_update_interval <= 0:
        raise ValueError("--progress-update-interval must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_config = load_quality_config(args.quality_config)
    records = [normalize_benchmark_record(row) for row in read_jsonl(args.benchmark)]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("The selected benchmark slice contains no evaluation records.")
    record_ids = [str(record["id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Benchmark contains duplicate stable example IDs.")
    corpus_profile = build_corpus_profile(records)

    manifest_path = args.output_dir / "evaluation_manifest.json"
    expected_manifest = evaluation_manifest(args, record_ids)
    contract_migration: dict[str, Any] | None = None
    if args.resume and manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        mismatches = evaluation_manifest_mismatches(existing_manifest, expected_manifest)
        if mismatches:
            migration = approved_content_only_migration(
                args.benchmark,
                previous_sha256=existing_manifest.get("benchmark_sha256"),
                current_sha256=expected_manifest.get("benchmark_sha256"),
            )
            remaining = [item for item in mismatches if not item.startswith("benchmark_sha256:")]
            if migration is not None and not remaining:
                contract_migration = migration
                print(
                    "[approved benchmark migration] generation inputs are unchanged; "
                    "durable replies will be rescored under the new explicit contracts"
                )
            else:
                detail = "\n  - ".join(mismatches)
                raise RuntimeError(
                    "Evaluation artifacts are incompatible with the requested model, benchmark, or policy.\n  - "
                    + detail
                )
    elif args.resume and (args.output_dir / "results.jsonl").exists():
        raise RuntimeError(
            "Existing generation rows have no evaluation manifest and cannot be reused safely. "
            "Preserve them for manual review, but start a new output directory or use a versioned "
            "migration utility that can prove the model, adapter, benchmark, and policy provenance."
        )

    store = DurableJsonlStore.open(args.output_dir / "results.jsonl", resume=args.resume)
    unknown_ids = sorted(set(store.rows_by_id) - set(record_ids))
    if unknown_ids:
        raise RuntimeError(f"Existing results contain IDs not present in this benchmark: {unknown_ids[:8]}")
    for warning in store.recovery_warnings:
        print(f"[jsonl recovery] {warning}")
    if contract_migration is not None:
        rescore_migrated_rows(
            records=records,
            store=store,
            quality_config=quality_config,
            corpus_profile=corpus_profile,
            migration=contract_migration,
            output_dir=args.output_dir,
        )
    write_json(manifest_path, expected_manifest)
    outputs = store.rows_in_order(record_ids)
    update_incremental_progress(args, outputs, total=len(records), status="running", force_markdown=True)

    if len(outputs) == len(records):
        outputs = store.finalize_order(record_ids)
        finalize_evaluation(args, outputs, quality_config)
        update_incremental_progress(args, outputs, total=len(records), status="complete", force_markdown=True)
        return

    try:
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable and --require-cuda was requested.")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is unavailable; evaluation will run on CPU and VRAM metrics will be 0.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype_map[args.torch_dtype],
            device_map="auto",
        )
        if args.adapter_path is not None:
            if PeftModel is None:
                raise RuntimeError("Install peft before loading an adapter.")
            model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()

        for index, record in enumerate(records, 1):
            if store.contains(str(record["id"])):
                print(f"[{index:03d}/{len(records):03d}] {record['id']}: [resume reuse]")
                continue
            messages = record["messages"]
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": args.system_prompt}, *messages]
            rendered = apply_chat_template(tokenizer, messages)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sync_cuda()
            started = time.perf_counter()
            stopping_criteria, boundary_criterion = build_stopping_criteria(
                tokenizer, int(inputs.input_ids.shape[1]), args.response_control, args.min_new_tokens_before_boundary
            )
            kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if stopping_criteria is not None:
                kwargs["stopping_criteria"] = stopping_criteria
            if args.do_sample:
                kwargs.update(temperature=args.temperature, top_p=args.top_p)
            with torch.inference_mode():
                generated = model.generate(**inputs, **kwargs)
            sync_cuda()
            elapsed = time.perf_counter() - started
            new_tokens = generated[0, inputs.input_ids.shape[1] :]
            raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            finalized = finalize_response(raw_reply, args.response_control)
            reply = finalized.text
            visible_token_count = len(tokenizer(reply, add_special_tokens=False)["input_ids"])
            stop_reason = infer_stop_reason(
                new_tokens.tolist(), eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens, criterion=boundary_criterion,
            )
            quality = evaluate_response(
                record, reply, visible_token_count, visible_model_tokens=visible_token_count,
                config=quality_config, corpus_profile=corpus_profile,
            )
            row = {
                "id": record["id"], "category": record["category"], "messages": record["messages"],
                "reference": record.get("reference"),
                "reference_token_f1": reference_token_f1(record.get("reference"), reply),
                "reply": reply, "raw_reply": finalized.raw_text,
                "response_control_mode": args.response_control,
                "response_control_trimmed": finalized.trimmed,
                "stop_reason": stop_reason, "sentence_count": finalized.sentence_count,
                "one_sentence_violation": finalized.sentence_count > 1,
                "prompt_tokens": int(inputs.input_ids.shape[1]),
                "generated_tokens": int(new_tokens.shape[0]),
                "visible_generated_tokens": visible_token_count,
                "elapsed_seconds": elapsed,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                **quality,
            }
            store.append(row)
            outputs = store.rows_in_order(record_ids)
            update_incremental_progress(args, outputs, total=len(records), status="running")
            flag_text = ""
            if row["critical_failures"]:
                flag_text = f" CRITICAL={row['critical_failures']}"
            elif row["major_issues"]:
                flag_text = f" MAJOR={row['major_issues']}"
            elif row["semantic_warnings"]:
                flag_text = f" WARN={row['semantic_warnings']}"
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']}: {reply!r} "
                f"quality={row['quality_score']:.3f} legacy={row['legacy_heuristic_score']:.3f}{flag_text}"
            )

        outputs = store.finalize_order(record_ids)
        finalize_evaluation(args, outputs, quality_config)
        update_incremental_progress(args, outputs, total=len(records), status="complete", force_markdown=True)
    except KeyboardInterrupt:
        outputs = store.rows_in_order(record_ids)
        update_incremental_progress(
            args, outputs, total=len(records), status="interrupted", force_markdown=True,
            error="KeyboardInterrupt",
        )
        raise
    except Exception as exc:
        outputs = store.rows_in_order(record_ids)
        update_incremental_progress(
            args, outputs, total=len(records), status="failed", force_markdown=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
