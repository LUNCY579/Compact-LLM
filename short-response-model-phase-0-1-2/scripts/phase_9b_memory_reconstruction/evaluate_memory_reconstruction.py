from __future__ import annotations

"""Compare memory-off, raw replay, prompted, trained, and direct Phase 9B strategies."""

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
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

from scripts.common.experiment_utils import read_jsonl, sha256, sha256_tree, write_json
from scripts.common.quality_evaluation import (
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
    token_f1,
)
from scripts.phase_9_student_distillation.evaluate_rag import generate_answer
from scripts.phase_9_student_distillation.phase9_common import ABSTENTION_TEXT, grounding_check, is_abstention, tree_bytes
from scripts.phase_9_student_distillation.phase9_metrics import functional_answer_correct, is_safe_conflict_response
from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    PIPELINE_REVISION,
    build_answer_messages,
    build_direct_memory_messages,
    build_reconstruction_messages,
    build_raw_replay_messages,
    decision_accuracy,
    parse_reconstruction_output,
    safe_mean,
    source_id_precision,
    source_id_recall,
    validate_reconstruction_record,
    write_jsonl,
)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]

BASELINE_STRATEGIES = ("memory_off", "raw_replay", "prompted_reconstruction")
CANDIDATE_STRATEGIES = ("trained_reconstruction", "direct_candidate")
ALL_STRATEGIES = (*BASELINE_STRATEGIES, *CANDIDATE_STRATEGIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 9B memory reconstruction.")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--baseline-adapter", type=Path, default=None)
    parser.add_argument("--candidate-model", default=None)
    parser.add_argument("--candidate-adapter", type=Path, default=None)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--reconstruction-max-new-tokens", type=int, default=128)
    parser.add_argument("--answer-max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _strategy_list(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(strategies) - set(ALL_STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown Phase 9B strategies: {unknown}")
    if not strategies:
        raise ValueError("At least one Phase 9B strategy is required")
    return strategies


def _load_model(model_path: str, adapter_path: Path | None) -> tuple[Any, Any]:
    if adapter_path is not None and PeftModel is None:
        raise RuntimeError("PEFT is required when an adapter is supplied")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _unload(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _expected_decision_class(record: dict[str, Any]) -> str:
    decision = record["approved_reconstruction"]["decision"]
    if decision == "insufficient":
        return "abstain"
    if decision == "conflict":
        return "conflict"
    return "answer"


def _evaluation_row(record: dict[str, Any], prompt_messages: list[dict[str, str]]) -> dict[str, Any]:
    decision = _expected_decision_class(record)
    phase9_decision = "answer" if decision == "answer" else decision
    # Phase 9B references are canonical examples, not mandatory surface forms.
    # Keep strict behavioral contracts only where the behavior itself is the
    # target (abstain/conflict); factual/procedural answers are judged by
    # grounding + semantic/reference anchors in _functional_success.
    requirements: list[dict[str, Any]] = []
    if decision == "abstain":
        requirements.append(
            {
                "id": "phase9b_required_abstention",
                "label": "missing_required_abstention",
                "severity": "major",
                "weight": 1.0,
                "description": "The reply must abstain because current state/memory is insufficient.",
                "failure_evidence": "The model answered despite insufficient memory/state.",
                "match": {"any_phrases": [ABSTENTION_TEXT]},
            }
        )
    elif decision == "conflict":
        requirements.append(
            {
                "id": "phase9b_safe_conflict",
                "label": "unsafe_conflict_resolution",
                "severity": "major",
                "weight": 1.0,
                "description": "The reply must not choose one side of unresolved equal-authority conflict.",
                "failure_evidence": "The model resolved an unresolved conflict without support.",
                "match": {
                    "any_phrases": [
                        ABSTENTION_TEXT,
                        "conflict",
                        "disagree",
                        "can't give one",
                        "cannot give one",
                    ]
                },
            }
        )
    return {
        "id": record["id"],
        "topic_id": record["topic_id"],
        "question": record["question"],
        "mode": record["mode"],
        "decision": phase9_decision,
        "messages": prompt_messages,
        "response_contract": {
            "schema_version": "4.5",
            "open_ended": False,
            "requirements": requirements,
            "prohibitions": [],
            "reference_is_example": True,
        },
        "category": f"phase9b_{record['approved_reconstruction']['decision']}",
    }


def _grounding_passages(record: dict[str, Any]) -> list[dict[str, Any]]:
    target = record["approved_reconstruction"]
    state_text = json.dumps(record["current_state"], ensure_ascii=False, sort_keys=True)
    support = " ".join([*target["applicable_facts"], target["guidance"], state_text]).strip()
    return [{"record_id": "reconstructed_support", "text": support or record["approved_answer"]}]


def _functional_success(record: dict[str, Any], answer: str, quality: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    expected = _expected_decision_class(record)
    if expected == "abstain":
        success = is_abstention(answer)
        return success, {"reason": "correct_abstention" if success else "failed_to_abstain"}
    if expected == "conflict":
        success = is_safe_conflict_response(answer)
        explicit = any(token in answer.casefold() for token in ("conflict", "disagree", "can't give one", "cannot give one"))
        return success, {
            "reason": "safe_conflict_handling" if success else "conflict_missed",
            "explicit_conflict_recognition": explicit,
        }
    grounding = grounding_check(answer, _grounding_passages(record))
    row = _evaluation_row(record, [])
    correct, details = functional_answer_correct(
        row,
        answer,
        reference=record["approved_answer"],
        abstained=is_abstention(answer),
        grounding_supported=grounding.supported,
    )
    if not correct and token_f1(record["approved_answer"], answer) >= 0.5 and grounding.supported:
        # The Phase 9 fallback already uses 0.5. This explicit fallback covers
        # records whose decision metadata was adapted into Phase 9B.
        correct = True
        details = {**details, "reason": "reference_anchor_passed"}
    return correct, {**details, "quality_score": quality.get("quality_score")}


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_sequence_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    return generate_answer(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
        response_control="eos_only",
    )


def _run_strategy(
    *,
    strategy: str,
    record: dict[str, Any],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    quality_config: dict[str, Any],
    corpus_profile: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    reconstruction_packet: dict[str, Any] | None = None
    reconstruction_generation: dict[str, Any] | None = None
    if strategy == "memory_off":
        messages = build_direct_memory_messages(
            question=record["question"],
            current_state=record["current_state"],
            memories=[],
            mode=record["mode"],
        )
    elif strategy == "raw_replay":
        messages = build_raw_replay_messages(
            question=record["question"],
            memories=record["memories"],
            mode=record["mode"],
        )
    elif strategy == "direct_candidate":
        messages = build_direct_memory_messages(
            question=record["question"],
            current_state=record["current_state"],
            memories=record["memories"],
            mode=record["mode"],
        )
    else:
        reconstruction_messages = build_reconstruction_messages(
            question=record["question"],
            current_state=record["current_state"],
            memories=record["memories"],
            mode=record["mode"],
        )
        reconstruction_generation = _generate(
            model=model,
            tokenizer=tokenizer,
            messages=reconstruction_messages,
            max_sequence_length=args.max_sequence_length,
            max_new_tokens=args.reconstruction_max_new_tokens,
        )
        parsed = parse_reconstruction_output(str(reconstruction_generation["answer"]))
        reconstruction_packet = {
            "payload": parsed.payload,
            "valid": parsed.valid,
            "error": parsed.error,
            "decision_correct": decision_accuracy(record["approved_reconstruction"], parsed.payload),
            "source_id_recall": source_id_recall(record["approved_reconstruction"], parsed.payload),
            "source_id_precision": source_id_precision(record["approved_reconstruction"], parsed.payload),
        }
        messages = build_answer_messages(
            question=record["question"],
            reconstruction=parsed.payload,
            mode=record["mode"],
        )

    answer_generation = _generate(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_sequence_length=args.max_sequence_length,
        max_new_tokens=args.answer_max_new_tokens,
    )
    answer = str(answer_generation["answer"])
    quality_record = _evaluation_row(record, messages)
    quality = evaluate_response(
        quality_record,
        answer,
        max(1, int(answer_generation.get("visible_tokens") or 1)),
        visible_model_tokens=max(1, int(answer_generation.get("visible_tokens") or 1)),
        config=quality_config,
        corpus_profile=corpus_profile,
    )
    functional_success, functional = _functional_success(record, answer, quality)
    reconstruction_seconds = float((reconstruction_generation or {}).get("generation_seconds") or 0.0)
    answer_seconds = float(answer_generation.get("generation_seconds") or 0.0)
    reconstruction_prompt_tokens = int((reconstruction_generation or {}).get("prompt_tokens") or 0)
    answer_prompt_tokens = int(answer_generation.get("prompt_tokens") or 0)
    reconstruction_generated_tokens = int((reconstruction_generation or {}).get("generated_tokens") or 0)
    answer_generated_tokens = int(answer_generation.get("generated_tokens") or 0)
    return {
        "evaluation_id": f"{strategy}:{record['id']}",
        "strategy": strategy,
        "id": record["id"],
        "topic_id": record["topic_id"],
        "question": record["question"],
        "scenario_type": (record.get("metadata") or {}).get("scenario_type"),
        "difficulty": (record.get("metadata") or {}).get("difficulty"),
        "counterfactual_pair_id": (record.get("metadata") or {}).get("counterfactual_pair_id"),
        "counterfactual_role": (record.get("metadata") or {}).get("counterfactual_role"),
        "negative_transfer_expected": bool((record.get("metadata") or {}).get("negative_transfer_expected")),
        "expected_reconstruction": record["approved_reconstruction"],
        "approved_answer": record["approved_answer"],
        "answer": answer,
        "functional_success": functional_success,
        "functional_diagnostics": functional,
        "quality": quality,
        "reconstruction": reconstruction_packet,
        "timing": {
            "reconstruction_seconds": reconstruction_seconds,
            "answer_seconds": answer_seconds,
            "total_generation_seconds": reconstruction_seconds + answer_seconds,
            "wall_seconds": time.perf_counter() - started,
        },
        "tokens": {
            "reconstruction_prompt": reconstruction_prompt_tokens,
            "answer_prompt": answer_prompt_tokens,
            "total_prompt": reconstruction_prompt_tokens + answer_prompt_tokens,
            "reconstruction_generated": reconstruction_generated_tokens,
            "answer_generated": answer_generated_tokens,
            "total_generated": reconstruction_generated_tokens + answer_generated_tokens,
            "answer_visible": int(answer_generation.get("visible_tokens") or 0),
        },
        "stop_reasons": {
            "reconstruction": (reconstruction_generation or {}).get("stop_reason"),
            "answer": answer_generation.get("stop_reason"),
        },
    }


def _rate_packet(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    success = sum(1 for row in rows if row.get("functional_success"))
    return {
        "case_count": count,
        "success_count": success,
        "success_rate": success / count if count else None,
    }


def _counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_id = str(row.get("counterfactual_pair_id") or "").strip()
        if pair_id:
            by_pair[pair_id].append(row)
    complete = 0
    both_functional = 0
    decision_sensitive = 0
    answer_sensitive = 0
    for pair_rows in by_pair.values():
        roles = {str(row.get("counterfactual_role") or ""): row for row in pair_rows}
        if "s_plus" not in roles or "s_minus" not in roles:
            continue
        complete += 1
        plus, minus = roles["s_plus"], roles["s_minus"]
        if plus.get("functional_success") and minus.get("functional_success"):
            both_functional += 1
        plus_recon = (plus.get("reconstruction") or {}).get("payload") or {}
        minus_recon = (minus.get("reconstruction") or {}).get("payload") or {}
        if plus_recon and minus_recon and plus_recon.get("decision") != minus_recon.get("decision"):
            decision_sensitive += 1
        if str(plus.get("answer") or "").strip().casefold() != str(minus.get("answer") or "").strip().casefold():
            answer_sensitive += 1
    return {
        "complete_pairs": complete,
        "both_functional_count": both_functional,
        "both_functional_rate": both_functional / complete if complete else None,
        "decision_sensitivity_count": decision_sensitive,
        "decision_sensitivity_rate": decision_sensitive / complete if complete else None,
        "answer_sensitivity_count": answer_sensitive,
        "answer_sensitivity_rate": answer_sensitive / complete if complete else None,
    }


def _strategy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality_summary = aggregate_quality_rows([row["quality"] for row in rows])
    reconstruction_rows = [row for row in rows if row.get("reconstruction") is not None]
    decision_correct = [row for row in reconstruction_rows if row["reconstruction"]["decision_correct"]]
    parse_valid = [row for row in reconstruction_rows if row["reconstruction"]["valid"]]
    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_decision[str(row["expected_reconstruction"]["decision"])].append(row)
        by_scenario[str(row.get("scenario_type") or "unknown")].append(row)
    conflict_rows = by_decision.get("conflict", [])
    explicit_conflicts = sum(
        1
        for row in conflict_rows
        if bool((row.get("functional_diagnostics") or {}).get("explicit_conflict_recognition"))
    )
    negative_expected = [row for row in rows if row.get("negative_transfer_expected")]
    return {
        **quality_summary,
        "case_count": len(rows),
        "functional_success_count": sum(1 for row in rows if row["functional_success"]),
        "functional_success_rate": sum(1 for row in rows if row["functional_success"]) / len(rows) if rows else None,
        "reconstruction_case_count": len(reconstruction_rows),
        "reconstruction_parse_valid_rate": len(parse_valid) / len(reconstruction_rows) if reconstruction_rows else None,
        "reconstruction_decision_accuracy": len(decision_correct) / len(reconstruction_rows) if reconstruction_rows else None,
        "source_id_recall": safe_mean(
            row["reconstruction"].get("source_id_recall") for row in reconstruction_rows
        ),
        "source_id_precision": safe_mean(
            row["reconstruction"].get("source_id_precision") for row in reconstruction_rows
        ),
        "mean_total_generation_seconds": statistics.fmean(
            float(row["timing"]["total_generation_seconds"]) for row in rows
        ) if rows else None,
        "median_total_generation_seconds": statistics.median(
            float(row["timing"]["total_generation_seconds"]) for row in rows
        ) if rows else None,
        "mean_total_prompt_tokens": statistics.fmean(float(row["tokens"]["total_prompt"]) for row in rows) if rows else None,
        "mean_total_generated_tokens": statistics.fmean(float(row["tokens"]["total_generated"]) for row in rows) if rows else None,
        "mean_answer_visible_tokens": statistics.fmean(float(row["tokens"]["answer_visible"]) for row in rows) if rows else None,
        "decision_counts": dict(Counter(row["expected_reconstruction"]["decision"] for row in rows)),
        "decision_performance": {key: _rate_packet(value) for key, value in sorted(by_decision.items())},
        "scenario_performance": {key: _rate_packet(value) for key, value in sorted(by_scenario.items())},
        "safe_conflict_handling": _rate_packet(conflict_rows),
        "explicit_conflict_recognition_count": explicit_conflicts,
        "explicit_conflict_recognition_rate": explicit_conflicts / len(conflict_rows) if conflict_rows else None,
        "negative_transfer_expected_cases": len(negative_expected),
        "negative_transfer_expected_success_rate": (
            sum(1 for row in negative_expected if row.get("functional_success")) / len(negative_expected)
            if negative_expected else None
        ),
        "counterfactual": _counterfactual_summary(rows),
    }


def _pairwise(rows: list[dict[str, Any]], strategies: list[str]) -> dict[str, Any]:
    by_strategy_id = {(row["strategy"], row["id"]): row for row in rows}
    ids = sorted({row["id"] for row in rows})
    memory_off = "memory_off" if "memory_off" in strategies else None
    raw = "raw_replay" if "raw_replay" in strategies else None
    packet: dict[str, Any] = {}
    for strategy in strategies:
        if strategy == memory_off:
            continue
        comparable = [
            record_id for record_id in ids
            if (strategy, record_id) in by_strategy_id
            and (memory_off, record_id) in by_strategy_id if memory_off is not None
        ]
        if comparable:
            negative = sum(
                1
                for record_id in comparable
                if by_strategy_id[(memory_off, record_id)]["functional_success"]
                and not by_strategy_id[(strategy, record_id)]["functional_success"]
            )
            packet[strategy] = {
                "memory_off_comparable_cases": len(comparable),
                "negative_transfer_count": negative,
                "negative_transfer_rate": negative / len(comparable),
            }
        if raw and strategy != raw:
            raw_comparable = [
                record_id for record_id in ids
                if (strategy, record_id) in by_strategy_id and (raw, record_id) in by_strategy_id
            ]
            if raw_comparable:
                rescues = sum(
                    1 for record_id in raw_comparable
                    if not by_strategy_id[(raw, record_id)]["functional_success"]
                    and by_strategy_id[(strategy, record_id)]["functional_success"]
                )
                harms = sum(
                    1 for record_id in raw_comparable
                    if by_strategy_id[(raw, record_id)]["functional_success"]
                    and not by_strategy_id[(strategy, record_id)]["functional_success"]
                )
                packet.setdefault(strategy, {}).update(
                    {
                        "raw_comparable_cases": len(raw_comparable),
                        "raw_failure_rescue_count": rescues,
                        "raw_failure_rescue_rate": rescues / len(raw_comparable),
                        "raw_success_harm_count": harms,
                        "raw_success_harm_rate": harms / len(raw_comparable),
                    }
                )
    return packet


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 9B memory-reconstruction evaluation",
        "",
        f"- Benchmark: `{summary['benchmark']}`",
        f"- Research ready: `{summary['research_ready']}`",
        f"- Smoke test: `{summary['smoke_test']}`",
        "",
        "## Strategies",
        "",
    ]
    for name, packet in summary["strategies"].items():
        pair = (summary.get("pairwise") or {}).get(name) or {}
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Functional success: `{packet.get('functional_success_rate')}`",
                f"- Mean quality: `{packet.get('mean_quality_score')}`",
                f"- Reconstruction decision accuracy: `{packet.get('reconstruction_decision_accuracy')}`",
                f"- Parse-valid rate: `{packet.get('reconstruction_parse_valid_rate')}`",
                f"- Negative-transfer rate vs memory-off: `{pair.get('negative_transfer_rate')}`",
                f"- Raw failure rescue rate: `{pair.get('raw_failure_rescue_rate')}`",
                f"- Raw success harm rate: `{pair.get('raw_success_harm_rate')}`",
                f"- Mean generation seconds: `{packet.get('mean_total_generation_seconds')}`",
                f"- Mean prompt/generated tokens: `{packet.get('mean_total_prompt_tokens')}` / `{packet.get('mean_total_generated_tokens')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    strategies = _strategy_list(args.strategies)
    if any(name in strategies for name in CANDIDATE_STRATEGIES) and not args.candidate_model:
        args.candidate_model = args.baseline_model
    benchmark = [validate_reconstruction_record(row) for row in read_jsonl(args.benchmark)]
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    if not benchmark:
        raise ValueError("Phase 9B benchmark is empty")
    quality_config = load_quality_config()
    corpus_profile = build_corpus_profile(
        [
            {
                "id": row["id"],
                "question": row["question"],
                "category": f"phase9b_{row['approved_reconstruction']['decision']}",
                "messages": [],
            }
            for row in benchmark
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    peak_vram_by_role: dict[str, int | None] = {}

    baseline_requested = [name for name in strategies if name in BASELINE_STRATEGIES]
    if baseline_requested:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        baseline_model, baseline_tokenizer = _load_model(args.baseline_model, args.baseline_adapter)
        for strategy in baseline_requested:
            for record in benchmark:
                all_results.append(
                    _run_strategy(
                        strategy=strategy,
                        record=record,
                        model=baseline_model,
                        tokenizer=baseline_tokenizer,
                        args=args,
                        quality_config=quality_config,
                        corpus_profile=corpus_profile,
                    )
                )
        peak_vram_by_role["baseline"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        _unload(baseline_model)

    candidate_requested = [name for name in strategies if name in CANDIDATE_STRATEGIES]
    if candidate_requested:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        candidate_model, candidate_tokenizer = _load_model(str(args.candidate_model), args.candidate_adapter)
        for strategy in candidate_requested:
            for record in benchmark:
                all_results.append(
                    _run_strategy(
                        strategy=strategy,
                        record=record,
                        model=candidate_model,
                        tokenizer=candidate_tokenizer,
                        args=args,
                        quality_config=quality_config,
                        corpus_profile=corpus_profile,
                    )
                )
        peak_vram_by_role["candidate"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        _unload(candidate_model)

    write_jsonl(args.output_dir / "results.jsonl", all_results)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_results:
        by_strategy[row["strategy"]].append(row)
    strategy_summaries = {name: _strategy_summary(by_strategy[name]) for name in strategies}
    pairwise = _pairwise(all_results, strategies)
    decision_classes = {row["approved_reconstruction"]["decision"] for row in benchmark}
    research_ready = not args.smoke_test and len(benchmark) >= 4 and len(decision_classes) >= 3
    summary = {
        "schema_version": "1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "baseline_model": args.baseline_model,
        "baseline_model_tree_sha256": sha256_tree(Path(args.baseline_model)) if Path(args.baseline_model).exists() else None,
        "baseline_adapter": str(args.baseline_adapter) if args.baseline_adapter else None,
        "candidate_model": str(args.candidate_model) if args.candidate_model else None,
        "candidate_model_tree_sha256": sha256_tree(Path(str(args.candidate_model))) if args.candidate_model and Path(str(args.candidate_model)).exists() else None,
        "candidate_adapter": str(args.candidate_adapter) if args.candidate_adapter else None,
        "candidate_adapter_tree_sha256": sha256_tree(args.candidate_adapter) if args.candidate_adapter else None,
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256(args.benchmark),
        "benchmark_rows": len(benchmark),
        "decision_classes": sorted(decision_classes),
        "strategies": strategy_summaries,
        "pairwise": pairwise,
        "peak_allocated_vram_bytes": peak_vram_by_role,
        "model_weight_bytes": {
            "baseline": tree_bytes(Path(args.baseline_model)) if Path(args.baseline_model).exists() else None,
            "candidate": tree_bytes(Path(str(args.candidate_model))) if args.candidate_model and Path(str(args.candidate_model)).exists() else None,
            "candidate_adapter": tree_bytes(args.candidate_adapter),
        },
        "smoke_test": bool(args.smoke_test),
        "research_ready": research_ready,
        "evaluation_valid": all(
            packet.get("case_count") == len(benchmark) for packet in strategy_summaries.values()
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
