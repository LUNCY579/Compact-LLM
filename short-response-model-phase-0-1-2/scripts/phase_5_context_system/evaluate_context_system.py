# Purpose: Measures retrieval quality, token reduction, traces, and optional compact/full model quality.
from __future__ import annotations

"""Evaluate retrieval, budgeting, contradiction filtering, and model context use."""

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
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def score_reply(reply: str, expected: dict[str, Any]) -> float:
    value = reply.casefold()
    checks: list[bool] = []
    include_any = [str(item).casefold() for item in expected.get("must_include_any", [])]
    if include_any:
        checks.append(any(item in value for item in include_any))
    include_all = [str(item).casefold() for item in expected.get("must_include_all", [])]
    checks.extend(item in value for item in include_all)
    forbidden = [str(item).casefold() for item in expected.get("must_not_include_any", [])]
    if forbidden:
        checks.append(not any(item in value for item in forbidden))
    if expected.get("should_ask_question"):
        checks.append("?" in reply)
    return statistics.fmean(checks) if checks else 1.0


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    response_control: str,
    min_new_tokens_before_boundary: int,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopping_criteria, boundary_criterion = build_stopping_criteria(
        tokenizer,
        int(encoded["input_ids"].shape[1]),
        response_control,
        min_new_tokens_before_boundary,
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
        "visible_generated_tokens": len(
            tokenizer(finalized.text, add_special_tokens=False)["input_ids"]
        ),
        "stop_reason": infer_stop_reason(
            continuation.tolist(),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            criterion=boundary_criterion,
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
    compact_quality: list[float] = []
    full_quality: list[float] = []
    compact_stop_reasons: list[str] = []
    full_stop_reasons: list[str] = []

    for case_index, row in enumerate(rows):
        memories = row.get("memories", [])
        started = time.perf_counter()
        selected, retrieval_trace = retrieve_with_trace(row["query"], memories, args.top_k)
        prompt = assemble(
            row.get("policy", DEFAULT_POLICY),
            selected,
            row.get("recent_turns", []),
            row["query"],
            budget,
            (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])) if tokenizer else None,
        )
        assembly_ms.append((time.perf_counter() - started) * 1000)

        selected_ids = {str(item.get("id")) for item in selected}
        required = {str(item) for item in row.get("required_memory_ids", [])}
        forbidden = {str(item) for item in row.get("forbidden_memory_ids", [])}
        true_positive = len(selected_ids & required)
        recall_values.append(1.0 if not required else true_positive / len(required))
        if not required:
            # Cases driven by recent turns or clarification do not have a
            # positive memory target; selecting no memory is fully precise.
            precision_values.append(1.0 if not selected_ids else 0.0)
        else:
            precision_values.append(true_positive / max(1, len(selected_ids)))
        forbidden_selected += len(selected_ids & forbidden)

        selected_tokens = sum(len(tokens(memory_text(item))) for item in selected)
        all_tokens = sum(len(tokens(memory_text(item))) for item in memories)
        selected_token_counts.append(selected_tokens)
        all_memory_token_counts.append(all_tokens)
        compact_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) if tokenizer else len(tokens(prompt))
        compact_prompt_tokens.append(compact_count)

        valid_full = [item for item in memories if not item.get("stale") and not item.get("contradicted")]
        full_prompt = assemble(
            row.get("policy", DEFAULT_POLICY),
            valid_full,
            row.get("recent_turns", []),
            row["query"],
            Budget(total=max(args.total_budget * 8, 4096), memory=max(args.memory_budget * 8, 2048), recent=max(args.recent_budget * 2, 512)),
            (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])) if tokenizer else None,
        )
        full_count = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"]) if tokenizer else len(tokens(full_prompt))
        full_prompt_tokens.append(full_count)

        trace_row: dict[str, Any] = {
            "id": row.get("id", case_index),
            "query": row["query"],
            "required_memory_ids": sorted(required),
            "forbidden_memory_ids": sorted(forbidden),
            "selected_memory_ids": sorted(selected_ids),
            "retrieval_trace": retrieval_trace,
            "compact_prompt": prompt,
            "full_prompt": full_prompt,
            "compact_prompt_tokens": compact_count,
            "full_prompt_tokens": full_count,
        }
        if model is not None and tokenizer is not None:
            compact_generation = generate(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.response_control,
                args.min_new_tokens_before_boundary,
            )
            full_generation = generate(
                model,
                tokenizer,
                full_prompt,
                args.max_new_tokens,
                args.response_control,
                args.min_new_tokens_before_boundary,
            )
            compact_reply = compact_generation["reply"]
            full_reply = full_generation["reply"]
            compact_score = score_reply(compact_reply, row.get("expected", {}))
            full_score = score_reply(full_reply, row.get("expected", {}))
            compact_stop_reasons.append(str(compact_generation["stop_reason"]))
            full_stop_reasons.append(str(full_generation["stop_reason"]))
            compact_quality.append(compact_score)
            full_quality.append(full_score)
            trace_row.update(
                compact_reply=compact_reply,
                full_reply=full_reply,
                compact_generation=compact_generation,
                full_generation=full_generation,
                compact_reply_score=compact_score,
                full_reply_score=full_score,
            )
        traces.append(trace_row)

    total_all = sum(all_memory_token_counts)
    total_selected = sum(selected_token_counts)
    summary = {
        "cases": len(rows),
        "model": args.model,
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
        "compact_generation_quality": statistics.fmean(compact_quality) if compact_quality else None,
        "full_generation_quality": statistics.fmean(full_quality) if full_quality else None,
        "compact_minus_full_quality": (
            statistics.fmean(compact_quality) - statistics.fmean(full_quality)
            if compact_quality and full_quality
            else None
        ),
    }
    write_json(args.out_dir / "evaluation.json", summary)
    write_jsonl(args.out_dir / "retrieval_traces.jsonl", traces)
    print(args.out_dir / "evaluation.json")


if __name__ == "__main__":
    main()
