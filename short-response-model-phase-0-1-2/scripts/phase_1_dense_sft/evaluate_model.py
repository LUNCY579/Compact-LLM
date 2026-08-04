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
from scripts.common.experiment_utils import atomic_write_text, read_json, sha256, write_json
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

def render_evaluation_report(args: argparse.Namespace, summary: dict[str, Any], outputs: list[dict[str, Any]]) -> str:
    cat_rows = "\n".join(
        f"| {category} | {score:.3f} | "
        f"{summary['category_quality_pass_rates'][category]:.3f} | "
        f"{summary['category_intelligibility_pass_rates'][category]:.3f} | "
        f"{summary.get('category_observable_quality_pass_rates', summary.get('category_semantic_pass_rates', {}))[category]:.3f} | "
        f"{summary['category_reference_alignment_scores'][category]:.3f} | "
        f"{summary['category_complete_utterance_rates'][category]:.3f} | "
        f"{summary['category_hard_failure_case_counts'][category]} | "
        f"{summary['category_major_case_counts'][category]} |"
        for category, score in summary["category_quality_scores"].items()
    )
    worst = sorted(outputs, key=lambda r: (r.get("quality_score", 0.0), r.get("semantic_pass", False)))[:15]
    worst_rows = "\n".join(
        f"| {row['id']} | {row['category']} | {row['quality_score']:.3f} | "
        f"{row['legacy_heuristic_score']:.3f} | {escape_markdown_cell(row['reply'])} |"
        for row in worst
    )
    flagged = [r for r in outputs if r["semantic_severity"] != "none" or r.get("review_diagnostics")]
    defect_rows = "\n".join(
        f"| {row['id']} | {row['semantic_severity']} | "
        f"{', '.join(row['critical_failures']) or '-'} | "
        f"{', '.join(row['major_issues']) or '-'} | "
        f"{', '.join(row['semantic_warnings']) or '-'} |"
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
| Mean reference alignment | {summary['mean_reference_alignment_score'] if summary['mean_reference_alignment_score'] is not None else 'N/A'} |
| Mean lexical reference token-F1 | {summary['mean_reference_token_f1'] if summary['mean_reference_token_f1'] is not None else 'N/A'} |

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


def evaluation_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": str(args.model_id),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256(args.benchmark),
        "quality_config_sha256": sha256(args.quality_config),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "max_new_tokens": int(args.max_new_tokens),
        "response_control": args.response_control,
        "min_new_tokens_before_boundary": int(args.min_new_tokens_before_boundary),
        "do_sample": bool(args.do_sample),
        "system_prompt": args.system_prompt,
    }


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
    write_json(args.output_dir / "evaluation_progress.json", progress)
    if args.candidate_status is not None:
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

        write_live_progress(
            args.run_root,
            markdown=(not args.disable_live_markdown) and markdown_due,
            **live_updates,
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
    record_ids = [str(record["id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Benchmark contains duplicate stable example IDs.")
    corpus_profile = build_corpus_profile(records)

    manifest_path = args.output_dir / "evaluation_manifest.json"
    expected_manifest = evaluation_manifest(args)
    if args.resume and manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        if existing_manifest != expected_manifest:
            raise RuntimeError("Evaluation artifacts are incompatible with the requested model, benchmark, or policy.")
    elif args.resume and (args.output_dir / "results.jsonl").exists():
        print("[resume warning] Legacy results have no evaluation manifest; stable IDs will be validated.")
    write_json(manifest_path, expected_manifest)

    store = DurableJsonlStore.open(args.output_dir / "results.jsonl", resume=args.resume)
    unknown_ids = sorted(set(store.rows_by_id) - set(record_ids))
    if unknown_ids:
        raise RuntimeError(f"Existing results contain IDs not present in this benchmark: {unknown_ids[:8]}")
    for warning in store.recovery_warnings:
        print(f"[jsonl recovery] {warning}")
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
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
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
