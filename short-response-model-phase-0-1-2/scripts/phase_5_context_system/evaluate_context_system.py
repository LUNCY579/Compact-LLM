# Purpose: Measures retrieval quality, token reduction, traces, and compact/full response quality-v2.
from __future__ import annotations

"""Evaluate retrieval, budgeting, contradiction filtering, and model context use.

Generation quality uses the shared evaluator-v2 harness. Compact and full
contexts are scored independently with identical criteria, and tail/hard-failure
metrics are retained so a good average cannot hide a collapsed reply.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_system import Budget, assemble, memory_text, retrieve_with_trace, tokens
from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl
from scripts.common.quality_evaluation import (
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)
from scripts.common.response_control import (
    RESPONSE_CONTROL_MODES,
    build_stopping_criteria,
    finalize_response,
    infer_stop_reason,
)

DEFAULT_POLICY = (
    "Reply naturally in a few words or one short sentence. Use supplied memory only when relevant. "
    "Never use stale or contradicted facts."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 5 context-system evaluation.")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/context/context_eval.jsonl")
    parser.add_argument("--model", default=None, help="Optional selected model for compact-vs-full generation.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/phase_5_context")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--total-budget", type=int, default=512)
    parser.add_argument("--memory-budget", type=int, default=160)
    parser.add_argument("--recent-budget", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--response-control", choices=RESPONSE_CONTROL_MODES, default="one_sentence")
    parser.add_argument("--min-new-tokens-before-boundary", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--quality-config", type=Path, default=ROOT / "configs/quality_evaluation.json")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def quality_record(row: dict[str, Any]) -> dict[str, Any]:
    expected = dict(row.get("expected") or {})
    expected.setdefault("min_tokens", 1)
    expected.setdefault("max_tokens", 32)
    return {
        "id": row.get("id"),
        "category": row.get("category", "context_consistency"),
        "messages": [{"role": "user", "content": str(row["query"])}],
        "expected": expected,
        "metadata": {"target_token_min": expected["min_tokens"], "target_token_max": expected["max_tokens"]},
    }


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, response_control: str, min_new_tokens_before_boundary: int) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopping_criteria, boundary_criterion = build_stopping_criteria(
        tokenizer, int(encoded["input_ids"].shape[1]), response_control, min_new_tokens_before_boundary
    )
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if stopping_criteria is not None:
        kwargs["stopping_criteria"] = stopping_criteria
    output = model.generate(**encoded, **kwargs)
    continuation = output[0, encoded["input_ids"].shape[1] :]
    raw_reply = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    finalized = finalize_response(raw_reply, response_control)
    return {
        "reply": finalized.text,
        "raw_reply": finalized.raw_text,
        "trimmed": finalized.trimmed,
        "sentence_count": finalized.sentence_count,
        "generated_tokens": int(continuation.shape[0]),
        "visible_generated_tokens": len(tokenizer(finalized.text, add_special_tokens=False)["input_ids"]),
        "stop_reason": infer_stop_reason(
            continuation.tolist(), eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens, criterion=boundary_criterion,
        ),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.cases)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    if not rows:
        raise SystemExit(f"No Phase 5 evaluation cases in {args.cases}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    normalized_records = [quality_record(row) for row in rows]
    quality_config = load_quality_config(args.quality_config)
    corpus_profile = build_corpus_profile(normalized_records)

    tokenizer = model = None
    if args.model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
        model.eval()

    budget = Budget(args.total_budget, args.memory_budget, args.recent_budget)
    traces: list[dict[str, Any]] = []
    recall_values: list[float] = []
    precision_values: list[float] = []
    forbidden_selected = 0
    selected_token_counts: list[int] = []
    all_memory_token_counts: list[int] = []
    compact_prompt_tokens: list[int] = []
    full_prompt_tokens: list[int] = []
    assembly_ms: list[float] = []
    compact_quality_rows: list[dict[str, Any]] = []
    full_quality_rows: list[dict[str, Any]] = []
    compact_stop_reasons: list[str] = []
    full_stop_reasons: list[str] = []

    for case_index, (row, eval_record) in enumerate(zip(rows, normalized_records)):
        memories = row.get("memories", [])
        started = time.perf_counter()
        selected, retrieval_trace = retrieve_with_trace(row["query"], memories, args.top_k)
        count_fn = (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])) if tokenizer else None
        prompt = assemble(row.get("policy", DEFAULT_POLICY), selected, row.get("recent_turns", []), row["query"], budget, count_fn)
        assembly_ms.append((time.perf_counter() - started) * 1000)

        selected_ids = {str(item.get("id")) for item in selected}
        required = {str(item) for item in row.get("required_memory_ids", [])}
        forbidden = {str(item) for item in row.get("forbidden_memory_ids", [])}
        true_positive = len(selected_ids & required)
        recall_values.append(1.0 if not required else true_positive / len(required))
        precision_values.append((1.0 if not selected_ids else 0.0) if not required else true_positive / max(1, len(selected_ids)))
        forbidden_selected += len(selected_ids & forbidden)

        selected_tokens = sum(len(tokens(memory_text(item))) for item in selected)
        all_tokens = sum(len(tokens(memory_text(item))) for item in memories)
        selected_token_counts.append(selected_tokens)
        all_memory_token_counts.append(all_tokens)
        compact_count = count_fn(prompt) if count_fn else len(tokens(prompt))
        compact_prompt_tokens.append(compact_count)

        valid_full = [item for item in memories if not item.get("stale") and not item.get("contradicted")]
        full_prompt = assemble(
            row.get("policy", DEFAULT_POLICY), valid_full, row.get("recent_turns", []), row["query"],
            Budget(total=max(args.total_budget * 8, 4096), memory=max(args.memory_budget * 8, 2048), recent=max(args.recent_budget * 2, 512)),
            count_fn,
        )
        full_count = count_fn(full_prompt) if count_fn else len(tokens(full_prompt))
        full_prompt_tokens.append(full_count)

        trace_row: dict[str, Any] = {
            "id": row.get("id", case_index), "query": row["query"],
            "required_memory_ids": sorted(required), "forbidden_memory_ids": sorted(forbidden),
            "selected_memory_ids": sorted(selected_ids), "retrieval_trace": retrieval_trace,
            "compact_prompt": prompt, "full_prompt": full_prompt,
            "compact_prompt_tokens": compact_count, "full_prompt_tokens": full_count,
        }
        if model is not None and tokenizer is not None:
            compact_generation = generate(model, tokenizer, prompt, args.max_new_tokens, args.response_control, args.min_new_tokens_before_boundary)
            full_generation = generate(model, tokenizer, full_prompt, args.max_new_tokens, args.response_control, args.min_new_tokens_before_boundary)
            compact_eval = evaluate_response(
                eval_record, compact_generation["reply"], int(compact_generation["visible_generated_tokens"]),
                visible_model_tokens=int(compact_generation["visible_generated_tokens"]), config=quality_config,
                corpus_profile=corpus_profile,
            )
            full_eval = evaluate_response(
                eval_record, full_generation["reply"], int(full_generation["visible_generated_tokens"]),
                visible_model_tokens=int(full_generation["visible_generated_tokens"]), config=quality_config,
                corpus_profile=corpus_profile,
            )
            compact_quality_rows.append({"id": trace_row["id"], "category": eval_record["category"], **compact_eval})
            full_quality_rows.append({"id": trace_row["id"], "category": eval_record["category"], **full_eval})
            compact_stop_reasons.append(str(compact_generation["stop_reason"]))
            full_stop_reasons.append(str(full_generation["stop_reason"]))
            trace_row.update(
                compact_reply=compact_generation["reply"], full_reply=full_generation["reply"],
                compact_generation=compact_generation, full_generation=full_generation,
                compact_quality=compact_eval, full_quality=full_eval,
                compact_reply_score=compact_eval["quality_score"], full_reply_score=full_eval["quality_score"],
            )
        traces.append(trace_row)

    total_all = sum(all_memory_token_counts)
    total_selected = sum(selected_token_counts)
    compact_summary = aggregate_quality_rows(compact_quality_rows) if compact_quality_rows else None
    full_summary = aggregate_quality_rows(full_quality_rows) if full_quality_rows else None
    summary: dict[str, Any] = {
        "quality_evaluation_version": "2.0" if compact_summary else None,
        "cases": len(rows), "model": args.model,
        "response_control_mode": args.response_control,
        "min_new_tokens_before_boundary": args.min_new_tokens_before_boundary,
        "compact_stop_reason_counts": {reason: compact_stop_reasons.count(reason) for reason in sorted(set(compact_stop_reasons))},
        "full_stop_reason_counts": {reason: full_stop_reasons.count(reason) for reason in sorted(set(full_stop_reasons))},
        "relevant_memory_recall": statistics.fmean(recall_values),
        "relevant_memory_precision": statistics.fmean(precision_values),
        "forbidden_memory_selection_count": forbidden_selected,
        "avg_selected_memory_tokens": statistics.fmean(selected_token_counts),
        "avg_all_memory_tokens": statistics.fmean(all_memory_token_counts),
        "memory_token_reduction_fraction": 1.0 - total_selected / max(1, total_all),
        "avg_compact_prompt_tokens": statistics.fmean(compact_prompt_tokens),
        "avg_full_prompt_tokens": statistics.fmean(full_prompt_tokens),
        "prompt_token_reduction_fraction": 1.0 - sum(compact_prompt_tokens) / max(1, sum(full_prompt_tokens)),
        "median_assembly_ms": statistics.median(assembly_ms),
        "p90_assembly_ms": percentile(assembly_ms, 0.90),
        "compact_quality": compact_summary,
        "full_quality": full_summary,
        # Historical flat fields retained for downstream compatibility.
        "compact_generation_quality": compact_summary.get("mean_quality_score") if compact_summary else None,
        "full_generation_quality": full_summary.get("mean_quality_score") if full_summary else None,
        "compact_minus_full_quality": (
            float(compact_summary["mean_quality_score"]) - float(full_summary["mean_quality_score"])
            if compact_summary and full_summary else None
        ),
    }
    write_json(args.out_dir / "evaluation.json", summary)
    write_jsonl(args.out_dir / "retrieval_traces.jsonl", traces)
    print(args.out_dir / "evaluation.json")


if __name__ == "__main__":
    main()
