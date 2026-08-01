# Purpose: Measures dense versus gathered sparse decode, VRAM, quality, active ratios, and fallback rates.
from __future__ import annotations

"""Benchmark dense versus routed sparse decode on identical prompts.

The script validates model/router provenance, measures real wall-clock latency
and peak allocated VRAM, scores dense and sparse replies with the same Phase 1
quality harness, and records router active/fallback rates.  A slower gathered
PyTorch reference is a valid negative result; the phase can still recommend a
dense production runtime.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256_tree, write_json, write_jsonl
from scripts.common.model_introspection import find_decoder_stack, first_parameter_device, get_gated_mlp
from scripts.common.sparse_mlp import RoutedSparseGatedMLP, TinyBlockRouter
from scripts.common.quality_evaluation import aggregate_quality_rows, build_corpus_profile, evaluate_response, load_quality_config
from scripts.common.quality_gate import assert_valid_reference, compare_quality_summaries
from scripts.common.response_control import (
    RESPONSE_CONTROL_MODES,
    finalize_response,
    first_sentence_boundary,
    validate_response_control_mode,
)
from scripts.phase_1_dense_sft.evaluate_model import normalize_benchmark_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure dense and routed-reference runtime.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--routers", type=Path, default=None)
    parser.add_argument("--dense-only", action="store_true", help="Measure the dense deployment baseline when Phase 6 found no viable sparse ratio.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_7_router/runtime_comparison.json")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--response-control", choices=RESPONSE_CONTROL_MODES, default="one_sentence")
    parser.add_argument("--min-new-tokens-before-boundary", type=int, default=1)
    parser.add_argument("--warmup-cases", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--force-confidence-threshold", type=float, default=None)
    parser.add_argument("--quality-config", type=Path, default=ROOT / "configs/quality_evaluation.json")
    parser.add_argument("--allow-provenance-mismatch", action="store_true")
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def apply_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def prompt_from_row(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        return apply_template(tokenizer, messages)
    prompt = row.get("prompt") or row.get("text")
    if not isinstance(prompt, str):
        raise ValueError("Benchmark rows need messages, prompt, or text.")
    return prompt


def greedy_timed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    response_control: str,
    min_new_tokens_before_boundary: int,
) -> dict[str, Any]:
    response_control = validate_response_control_mode(response_control)
    device = first_parameter_device(model)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    synchronize()
    prefill_ms = (time.perf_counter() - started) * 1000
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    past = output.past_key_values
    decode_times: list[float] = []
    stop_reason = "unknown"

    while True:
        if tokenizer.eos_token_id is not None and generated[-1] == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        if response_control == "one_sentence" and len(generated) >= max(1, min_new_tokens_before_boundary):
            current_text = tokenizer.decode(generated, skip_special_tokens=True)
            decision = first_sentence_boundary(current_text)
            if decision.found:
                stop_reason = decision.reason or "validated_sentence_boundary"
                break
        if len(generated) >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        synchronize()
        decode_times.append((time.perf_counter() - started) * 1000)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        past = output.past_key_values

    raw_reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
    finalized = finalize_response(raw_reply, response_control)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_times,
        "generated_ids": generated,
        "generated_tokens": len(generated),
        "visible_generated_tokens": len(
            tokenizer(finalized.text, add_special_tokens=False)["input_ids"]
        ),
        "raw_reply": finalized.raw_text,
        "reply": finalized.text,
        "response_control_trimmed": finalized.trimmed,
        "sentence_count": finalized.sentence_count,
        "one_sentence_violation": finalized.sentence_count > 1,
        "stop_reason": stop_reason,
        "peak_vram_gb": peak_vram,
    }


def summarize_runtime(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decode = [value for run in runs for value in run["decode_ms"]]
    return {
        "median_prefill_ms": statistics.median(run["prefill_ms"] for run in runs),
        "median_decode_ms_per_token": statistics.median(decode) if decode else None,
        "median_total_decode_ms": statistics.median(sum(run["decode_ms"]) for run in runs),
        "median_generated_tokens": statistics.median(run["generated_tokens"] for run in runs),
        "median_visible_generated_tokens": statistics.median(run["visible_generated_tokens"] for run in runs),
        "one_sentence_violation_rate": statistics.fmean(
            1.0 if run["one_sentence_violation"] else 0.0 for run in runs
        ),
        "response_control_trim_rate": statistics.fmean(
            1.0 if run["response_control_trimmed"] else 0.0 for run in runs
        ),
        "stop_reason_counts": {
            reason: sum(1 for run in runs if run["stop_reason"] == reason)
            for reason in sorted({run["stop_reason"] for run in runs})
        },
        "peak_vram_gb": max(run["peak_vram_gb"] for run in runs),
    }


def score_outputs(
    records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    quality_config: Any,
    corpus_profile: dict[str, Any],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for index, (record, run) in enumerate(zip(records, runs)):
        assessment = evaluate_response(
            record, run["reply"], int(run["visible_generated_tokens"]),
            visible_model_tokens=int(run["visible_generated_tokens"]),
            config=quality_config, corpus_profile=corpus_profile,
        )
        scored.append({"id": record.get("id", index), "category": record.get("category"), **assessment})
    return aggregate_quality_rows(scored)


def main() -> None:
    args = parse_args()
    if "sealed" in args.benchmark.parts:
        raise SystemExit("Phase 7 development benchmarking may not use sealed data.")
    if not args.dense_only and args.routers is None:
        raise SystemExit("--routers is required unless --dense-only is selected.")

    model_fingerprint = sha256_tree(Path(args.model))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()

    records = [normalize_benchmark_record(row) for row in read_jsonl(args.benchmark)[: args.max_cases]]
    quality_config = load_quality_config(args.quality_config)
    corpus_profile = build_corpus_profile(records)
    prompts = [prompt_from_row(tokenizer, row) for row in records]
    if not prompts:
        raise ValueError("Benchmark produced no prompts.")

    for prompt in prompts[: args.warmup_cases]:
        greedy_timed(
            model, tokenizer, prompt, min(4, args.max_new_tokens),
            args.response_control, args.min_new_tokens_before_boundary
        )

    dense_runs: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        for prompt in prompts:
            dense_runs.append(greedy_timed(
                model, tokenizer, prompt, args.max_new_tokens,
                args.response_control, args.min_new_tokens_before_boundary
            ))
    dense_summary = summarize_runtime(dense_runs)
    dense_quality = score_outputs(records, dense_runs[: len(records)], quality_config=quality_config, corpus_profile=corpus_profile)
    assert_valid_reference(dense_quality, label="Phase 7 dense reference")

    # A negative Phase 6 result is a valid mature-product outcome. Phase 7
    # still records the real dense baseline so Phase 8 has measured runtime
    # evidence, but it does not fabricate routers or sparse metrics.
    if args.dense_only:
        raw_path = args.out.with_name(args.out.stem + "_raw.jsonl")
        write_jsonl(
            raw_path,
            [
                {
                    "run_index": index,
                    "id": records[index % len(records)].get("id", index % len(records)),
                    "category": records[index % len(records)].get("category"),
                    "dense_reply": run["reply"],
                    "dense_raw_reply": run["raw_reply"],
                    "dense_stop_reason": run["stop_reason"],
                    "dense_prefill_ms": run["prefill_ms"],
                    "dense_decode_ms": run["decode_ms"],
                }
                for index, run in enumerate(dense_runs)
            ],
        )
        write_json(
            args.out,
            {
                "model": args.model,
                "model_fingerprint": model_fingerprint,
                "benchmark": str(args.benchmark),
                "cases": len(prompts),
                "repeats": args.repeats,
                "response_control_mode": args.response_control,
                "min_new_tokens_before_boundary": args.min_new_tokens_before_boundary,
                "sparse_attempted": False,
                "dense": dense_summary,
                "dense_quality": dense_quality,
                "sparse_reference": None,
                "sparse_quality": None,
                "quality_score_loss": None,
                "decode_speedup": None,
                "peak_vram_reduction_fraction": None,
                "exact_generation_match_rate": None,
                "layer_router_stats": {},
                "passes_response_control_target": bool(
                    args.response_control == "one_sentence"
                    and dense_summary["one_sentence_violation_rate"] == 0.0
                ),
                "quality_evaluation_version": "2.0",
                "passes_quality_target": True,
                "passes_runtime_target": False,
                "raw_results": str(raw_path),
                "warning": "Phase 6 found no viable sparse ratio; Phase 7 measured and selected the dense runtime baseline.",
            },
        )
        print(args.out)
        return

    assert args.routers is not None
    router_data = torch.load(args.routers, map_location="cpu", weights_only=False)
    expected_fingerprint = router_data.get("source_model_fingerprint")
    if expected_fingerprint and expected_fingerprint != model_fingerprint and not args.allow_provenance_mismatch:
        raise SystemExit(
            "Router/model provenance mismatch. Rebuild Phase 6 traces and Phase 7 routers for this exact model, "
            "or use --allow-provenance-mismatch only for a labeled control."
        )

    wrappers: list[RoutedSparseGatedMLP] = []
    keep_ratio = float(args.keep_ratio if args.keep_ratio is not None else router_data["target_ratio"])
    for layer_index, layer in enumerate(find_decoder_stack(model).layers):
        layer_key = str(layer_index)
        checkpoint = router_data["routers"].get(layer_key)
        if checkpoint is None:
            raise KeyError(f"Router checkpoint has no layer {layer_key}.")
        router = TinyBlockRouter(
            int(checkpoint["hidden_size"]),
            int(checkpoint["router_width"]),
            int(checkpoint["num_blocks"]),
        )
        router.load_state_dict(checkpoint["state_dict"])
        dense_mlp = get_gated_mlp(layer)
        router.to(device=dense_mlp.gate_proj.weight.device, dtype=dense_mlp.gate_proj.weight.dtype)
        threshold = (
            float(args.force_confidence_threshold)
            if args.force_confidence_threshold is not None
            else float(checkpoint["confidence_threshold"])
        )
        wrapper = RoutedSparseGatedMLP(
            dense_mlp,
            router,
            block_size=int(router_data["block_size"]),
            keep_ratio=keep_ratio,
            confidence_threshold=threshold,
            dense_prefill=True,
        )
        layer.mlp = wrapper
        wrappers.append(wrapper)

    for prompt in prompts[: args.warmup_cases]:
        greedy_timed(
            model, tokenizer, prompt, min(4, args.max_new_tokens),
            args.response_control, args.min_new_tokens_before_boundary
        )
    for wrapper in wrappers:
        wrapper.stats = type(wrapper.stats)()

    sparse_runs: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        for prompt in prompts:
            sparse_runs.append(greedy_timed(
                model, tokenizer, prompt, args.max_new_tokens,
                args.response_control, args.min_new_tokens_before_boundary
            ))

    sparse_summary = summarize_runtime(sparse_runs)
    dense_decode = dense_summary["median_decode_ms_per_token"]
    sparse_decode = sparse_summary["median_decode_ms_per_token"]
    speedup = dense_decode / sparse_decode if dense_decode and sparse_decode else None
    dense_vram = dense_summary["peak_vram_gb"]
    sparse_vram = sparse_summary["peak_vram_gb"]
    vram_reduction = 1 - sparse_vram / dense_vram if dense_vram and sparse_vram else None

    paired = min(len(dense_runs), len(sparse_runs))
    exact = [dense_runs[index]["generated_ids"] == sparse_runs[index]["generated_ids"] for index in range(paired)]
    sparse_quality = score_outputs(records, sparse_runs[: len(records)], quality_config=quality_config, corpus_profile=corpus_profile)

    raw_rows: list[dict[str, Any]] = []
    for index in range(paired):
        case_index = index % len(records)
        dense_case_quality = evaluate_response(
            records[case_index], dense_runs[index]["reply"], int(dense_runs[index]["visible_generated_tokens"]),
            visible_model_tokens=int(dense_runs[index]["visible_generated_tokens"]),
            config=quality_config, corpus_profile=corpus_profile,
        )
        sparse_case_quality = evaluate_response(
            records[case_index], sparse_runs[index]["reply"], int(sparse_runs[index]["visible_generated_tokens"]),
            visible_model_tokens=int(sparse_runs[index]["visible_generated_tokens"]),
            config=quality_config, corpus_profile=corpus_profile,
        )
        raw_rows.append(
            {
                "run_index": index,
                "id": records[case_index].get("id", case_index),
                "category": records[case_index].get("category"),
                "dense_reply": dense_runs[index]["reply"],
                "sparse_reply": sparse_runs[index]["reply"],
                "dense_raw_reply": dense_runs[index]["raw_reply"],
                "sparse_raw_reply": sparse_runs[index]["raw_reply"],
                "dense_stop_reason": dense_runs[index]["stop_reason"],
                "sparse_stop_reason": sparse_runs[index]["stop_reason"],
                "dense_quality": dense_case_quality,
                "sparse_quality": sparse_case_quality,
                "quality_score_delta": sparse_case_quality["quality_score"] - dense_case_quality["quality_score"],
                "exact_token_match": exact[index],
                "dense_prefill_ms": dense_runs[index]["prefill_ms"],
                "sparse_prefill_ms": sparse_runs[index]["prefill_ms"],
                "dense_decode_ms": dense_runs[index]["decode_ms"],
                "sparse_decode_ms": sparse_runs[index]["decode_ms"],
            }
        )
    raw_path = args.out.with_name(args.out.stem + "_raw.jsonl")
    write_jsonl(raw_path, raw_rows)

    layer_stats = {str(index): wrapper.stats.as_dict() for index, wrapper in enumerate(wrappers)}
    quality_comparison = compare_quality_summaries(
        dense_quality, sparse_quality,
        max_mean_quality_loss=0.01, max_quality_index_loss=0.015,
        max_worst_decile_loss=0.03, max_category_quality_loss_allowed=0.04,
        max_semantic_pass_loss=0.01, max_reference_alignment_loss=0.05,
        min_quality_pass_rate=0.98, min_intelligibility_pass_rate=1.0,
        min_semantic_pass_rate=0.98, min_complete_utterance_rate=0.99,
        max_hard_failure_cases=0, max_critical_failures=0, max_major_issue_cases=0,
    )
    quality_loss = quality_comparison["mean_quality_loss"]
    payload = {
        "model": args.model,
        "model_fingerprint": model_fingerprint,
        "routers": str(args.routers),
        "router_source_model_fingerprint": expected_fingerprint,
        "benchmark": str(args.benchmark),
        "cases": len(prompts),
        "repeats": args.repeats,
        "response_control_mode": args.response_control,
        "min_new_tokens_before_boundary": args.min_new_tokens_before_boundary,
        "sparse_attempted": True,
        "keep_ratio": keep_ratio,
        "dense": dense_summary,
        "sparse_reference": sparse_summary,
        "dense_quality": dense_quality,
        "sparse_quality": sparse_quality,
        "quality_evaluation_version": "2.0",
        "quality_score_loss": quality_loss,
        "quality_comparison": quality_comparison,
        "decode_speedup": speedup,
        "peak_vram_reduction_fraction": vram_reduction,
        "exact_generation_match_rate": statistics.fmean(exact) if exact else None,
        "layer_router_stats": layer_stats,
        "passes_response_control_target": bool(
            args.response_control == "one_sentence"
            and dense_summary["one_sentence_violation_rate"] == 0.0
            and sparse_summary["one_sentence_violation_rate"] == 0.0
        ),
        "passes_quality_target": bool(
            quality_comparison["eligible"]
            and sparse_summary["one_sentence_violation_rate"] == 0.0
        ),
        "passes_runtime_target": bool(
            (speedup is not None and speedup >= 1.20)
            or (vram_reduction is not None and vram_reduction >= 0.20)
        ),
        "raw_results": str(raw_path),
        "warning": (
            "This is the PyTorch gather reference, not a packed custom kernel. A slower result is valid and should "
            "lead to dense deployment unless a later optimized kernel proves value."
        ),
    }
    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
