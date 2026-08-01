# Purpose: Runs contribution frontiers and actual dense-compute masked generation comparisons.
from __future__ import annotations

"""Measure oracle contribution retention and real masked-generation quality.

All MLP projections are still executed before masking, so the script answers
only the quality question.  Runtime claims remain forbidden until Phase 7.
"""

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl
from scripts.common.model_introspection import find_decoder_stack, first_parameter_device, get_gated_mlp
from scripts.common.sparse_mlp import OracleMaskedGatedMLP
from scripts.common.quality_evaluation import aggregate_quality_rows, build_corpus_profile, evaluate_response, load_quality_config
from scripts.common.quality_gate import assert_valid_reference, compare_quality_summaries
from scripts.common.response_control import (
    RESPONSE_CONTROL_MODES,
    build_stopping_criteria,
    finalize_response,
    infer_stop_reason,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 dense-compute sparse simulations.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.8, 0.65, 0.5, 0.35])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_6_sparse/frontier.json")
    parser.add_argument("--model", default=None, help="Optional model for masked generation comparisons.")
    parser.add_argument("--benchmark", type=Path, default=None, help="Development benchmark only; sealed data is forbidden.")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--response-control", choices=RESPONSE_CONTROL_MODES, default="one_sentence")
    parser.add_argument("--min-new-tokens-before-boundary", type=int, default=1)
    return parser.parse_args()


def apply_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def prompt_from_row(tokenizer: Any, row: dict[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return apply_template(tokenizer, messages)
    text = row.get("prompt") or row.get("text")
    if not isinstance(text, str):
        raise ValueError("Benchmark row has no prompt/messages.")
    return text


def generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    response_control: str,
    min_new_tokens_before_boundary: int,
) -> dict[str, Any]:
    device = first_parameter_device(model)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
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
    with torch.inference_mode():
        output = model.generate(**encoded, **kwargs)
    continuation = output[0, encoded["input_ids"].shape[1] :].tolist()
    raw_reply = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    finalized = finalize_response(raw_reply, response_control)
    return {
        "reply": finalized.text,
        "raw_reply": finalized.raw_text,
        "ids": continuation,
        "visible_token_count": len(tokenizer(finalized.text, add_special_tokens=False)["input_ids"]),
        "stop_reason": infer_stop_reason(
            continuation,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            criterion=boundary_criterion,
        ),
        "trimmed": finalized.trimmed,
        "sentence_count": finalized.sentence_count,
    }


def common_prefix_fraction(left: list[int], right: list[int]) -> float:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched / max(1, max(len(left), len(right)))


def aggregate_scored(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper around the shared evaluator-v2 aggregator."""
    return aggregate_quality_rows(rows)


def main() -> None:
    args = parse_args()
    if args.benchmark and "sealed" in args.benchmark.parts:
        raise SystemExit("Phase 6 may not use sealed evaluation data.")
    trace_data = torch.load(args.traces, map_location="cpu", weights_only=False)
    frontier: list[dict[str, Any]] = []
    for ratio in args.ratios:
        retained_values: list[float] = []
        for layer in trace_data["layers"].values():
            scores = layer["block_scores"].float()
            blocks = scores.shape[-1]
            keep = max(1, min(blocks, int(round(blocks * ratio))))
            retained = torch.topk(scores, k=keep, dim=-1).values.sum(dim=-1) / scores.sum(dim=-1).clamp_min(1e-12)
            retained_values.extend(retained.tolist())
        ordered = sorted(retained_values)
        frontier.append(
            {
                "active_ratio": ratio,
                "mean_oracle_contribution_retained": statistics.fmean(retained_values),
                "p10_oracle_contribution_retained": ordered[max(0, int(0.10 * len(ordered)) - 1)],
                "quality_evaluation_required": True,
                "speed_claim_allowed": False,
            }
        )

    payload: dict[str, Any] = {
        "model": trace_data.get("model"),
        "model_fingerprint": trace_data.get("model_fingerprint"),
        "frontier": frontier,
        "warning": "Oracle masking still uses dense computation; no speed or VRAM claim is allowed.",
    }

    if args.model and args.benchmark:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.phase_1_dense_sft.evaluate_model import normalize_benchmark_record

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
        model.eval()
        rows = [normalize_benchmark_record(row) for row in read_jsonl(args.benchmark)[: args.max_cases]]
        quality_config = load_quality_config()
        corpus_profile = build_corpus_profile(rows)
        prompts = [prompt_from_row(tokenizer, row) for row in rows]
        dense_outputs = [
            generation(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.response_control,
                args.min_new_tokens_before_boundary,
            )
            for prompt in prompts
        ]

        dense_scored: list[dict[str, Any]] = []
        for index, (row, generated) in enumerate(zip(rows, dense_outputs)):
            assessed = evaluate_response(
                row, generated["reply"], int(generated["visible_token_count"]),
                visible_model_tokens=int(generated["visible_token_count"]),
                config=quality_config, corpus_profile=corpus_profile,
            )
            dense_scored.append({"id": row.get("id", index), "category": row.get("category"), **assessed})
        dense_metrics = aggregate_scored(dense_scored)
        assert_valid_reference(dense_metrics, label="Phase 6 dense reference")
        payload["dense_quality"] = dense_metrics
        payload["response_control_mode"] = args.response_control
        payload["min_new_tokens_before_boundary"] = args.min_new_tokens_before_boundary

        wrappers: list[OracleMaskedGatedMLP] = []
        for layer in find_decoder_stack(model).layers:
            original = get_gated_mlp(layer)
            wrapper = OracleMaskedGatedMLP(original, int(trace_data["block_size"]), 1.0)
            layer.mlp = wrapper
            wrappers.append(wrapper)

        comparisons: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        for ratio in args.ratios:
            for wrapper in wrappers:
                wrapper.keep_ratio = ratio
                wrapper.stats = type(wrapper.stats)()
            sparse_outputs = [
                generation(
                    model, tokenizer, prompt, args.max_new_tokens,
                    args.response_control, args.min_new_tokens_before_boundary
                )
                for prompt in prompts
            ]
            exact = [dense["ids"] == sparse["ids"] for dense, sparse in zip(dense_outputs, sparse_outputs)]
            prefixes = [
                common_prefix_fraction(dense["ids"], sparse["ids"])
                for dense, sparse in zip(dense_outputs, sparse_outputs)
            ]
            sparse_scored: list[dict[str, Any]] = []
            for index, (row, generated) in enumerate(zip(rows, sparse_outputs)):
                assessed = evaluate_response(
                    row, generated["reply"], int(generated["visible_token_count"]),
                    visible_model_tokens=int(generated["visible_token_count"]),
                    config=quality_config, corpus_profile=corpus_profile,
                )
                sparse_scored.append({"id": row.get("id", index), "category": row.get("category"), **assessed})
            quality = aggregate_scored(sparse_scored)
            comparison = compare_quality_summaries(
                dense_metrics, quality,
                max_mean_quality_loss=0.01, max_quality_index_loss=0.015,
                max_worst_decile_loss=0.03, max_category_quality_loss_allowed=0.04,
                max_semantic_pass_loss=0.01, max_reference_alignment_loss=0.05,
                min_quality_pass_rate=0.98, min_intelligibility_pass_rate=1.0,
                min_semantic_pass_rate=0.98, min_complete_utterance_rate=0.99,
                max_hard_failure_cases=0, max_critical_failures=0, max_major_issue_cases=0,
            )
            comparisons.append(
                {
                    "active_ratio": ratio,
                    "exact_generation_match_rate": statistics.fmean(exact),
                    "mean_token_common_prefix_fraction": statistics.fmean(prefixes),
                    "mean_reported_active_ratio": statistics.fmean(
                        wrapper.stats.as_dict()["mean_active_block_ratio"] for wrapper in wrappers
                    ),
                    **quality,
                    "quality_comparison": comparison,
                    "quality_eligible": comparison["eligible"],
                    "mean_quality_loss_from_dense": comparison["mean_quality_loss"],
                    "mean_score_loss_from_dense": comparison["mean_quality_loss"],
                    "quality_index_loss_from_dense": comparison["quality_index_loss"],
                    "worst_decile_quality_loss_from_dense": comparison["worst_decile_quality_loss"],
                    "semantic_pass_loss_from_dense": comparison["semantic_pass_loss"],
                    "speed_claim_allowed": False,
                }
            )
            for index, (dense_generated, sparse_generated, scored) in enumerate(
                zip(dense_outputs, sparse_outputs, sparse_scored)
            ):
                raw_rows.append(
                    {
                        "id": rows[index].get("id", index),
                        "category": rows[index].get("category"),
                        "active_ratio": ratio,
                        "prompt": prompts[index],
                        "dense_reply": dense_generated["reply"],
                        "masked_reply": sparse_generated["reply"],
                        "dense_stop_reason": dense_generated["stop_reason"],
                        "masked_stop_reason": sparse_generated["stop_reason"],
                        "masked_quality_score": scored["quality_score"],
                        "masked_heuristic_score": scored["quality_score"],
                        "quality_pass": scored["quality_pass"],
                        "quality_hard_fail": scored["quality_hard_fail"],
                        "intelligibility_diagnostics": scored["intelligibility_diagnostics"],
                        "critical_failures": scored["critical_failures"],
                        "major_issues": scored["major_issues"],
                    }
                )
        payload["masked_generation_comparison"] = comparisons
        raw_path = args.out.with_name(args.out.stem + "_generations.jsonl")
        write_jsonl(raw_path, raw_rows)
        payload["raw_generations"] = str(raw_path)

    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
