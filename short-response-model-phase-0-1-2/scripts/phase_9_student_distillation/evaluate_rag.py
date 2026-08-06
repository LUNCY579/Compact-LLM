from __future__ import annotations

"""Evaluate closed-book, oracle-RAG, real-RAG, and adversarial-RAG behavior."""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import (
    append_jsonl_durable,
    read_json,
    read_jsonl,
    read_jsonl_recover,
    sha256,
    write_json,
)
from scripts.common.quality_evaluation import (
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
    token_f1,
)
from scripts.common.response_control import build_stopping_criteria, finalize_response, infer_stop_reason
from scripts.phase_9_student_distillation.phase9_common import (
    ABSTENTION_TEXT,
    CONFLICT_TEXT,
    SQLiteFTSRetriever,
    build_grounded_messages,
    evidence_gate,
    grounding_check,
    is_abstention,
    passage_id,
    process_rss_bytes,
    route_mode,
    sanitize_passages,
    tree_bytes,
)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]

CONDITIONS = ("closed_book", "oracle_rag", "real_rag", "adversarial_rag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 9 student or teacher under RAG conditions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knowledge-db", type=Path, default=None)
    parser.add_argument("--gate-calibration", type=Path, default=None)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--reranked-top-k", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="eos_only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-uncalibrated-real-rag", action="store_true")
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def normalize_condition_list(value: str) -> list[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown evaluation conditions: {unknown}")
    if not conditions:
        raise ValueError("At least one evaluation condition is required")
    return conditions


def passage_list(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(row.get(key) or []):
        if isinstance(item, str):
            result.append({"record_id": f"{row.get('id')}_{key}_{index}", "text": item, "normalized_score": 1.0})
        elif isinstance(item, dict):
            value = dict(item)
            value.setdefault("record_id", passage_id(value))
            value.setdefault("normalized_score", 1.0)
            result.append(value)
        else:
            raise ValueError(f"{row.get('id')} contains invalid {key} passage")
    return result


def prepare_evidence(
    *,
    row: dict[str, Any],
    condition: str,
    retriever: SQLiteFTSRetriever | None,
    retrieval_top_k: int,
    reranked_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]:
    started = time.perf_counter()
    rejected: list[dict[str, Any]] = []
    if condition == "closed_book":
        evidence: list[dict[str, Any]] = []
        procedures: list[dict[str, Any]] = []
    elif condition == "oracle_rag":
        evidence, rejected_evidence = sanitize_passages(passage_list(row, "evidence"))
        procedures, rejected_procedures = sanitize_passages(passage_list(row, "procedures"))
        rejected = [*rejected_evidence, *rejected_procedures]
    elif condition == "adversarial_rag":
        source = row.get("adversarial_evidence") if row.get("adversarial_evidence") is not None else row.get("evidence")
        adversarial_row = {**row, "evidence": source or []}
        evidence, rejected_evidence = sanitize_passages(passage_list(adversarial_row, "evidence"))
        procedures, rejected_procedures = sanitize_passages(passage_list(row, "procedures"))
        rejected = [*rejected_evidence, *rejected_procedures]
    else:
        if retriever is None:
            raise RuntimeError("real_rag requires --knowledge-db")
        raw_evidence = retriever.search(str(row.get("question") or ""), limit=retrieval_top_k, kind="knowledge")
        raw_procedures = retriever.search(str(row.get("question") or ""), limit=max(3, reranked_top_k), kind="procedure")
        evidence, rejected_evidence = sanitize_passages(raw_evidence)
        procedures, rejected_procedures = sanitize_passages(raw_procedures)
        evidence = evidence[:reranked_top_k]
        procedures = procedures[: min(2, reranked_top_k)]
        rejected = [*rejected_evidence, *rejected_procedures]
    return evidence, procedures, rejected, time.perf_counter() - started


def generate_answer(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_sequence_length: int,
    max_new_tokens: int,
    response_control: str,
) -> dict[str, Any]:
    prompt = apply_chat_template(tokenizer, messages)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    stopping, criterion = build_stopping_criteria(tokenizer, prompt_length, response_control)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[0, prompt_length:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    finalized = finalize_response(raw, response_control)
    return {
        "answer": finalized.text,
        "raw_answer": finalized.raw_text,
        "generated_tokens": int(new_ids.numel()),
        "visible_tokens": len(tokenizer(finalized.text, add_special_tokens=False)["input_ids"]),
        "generation_seconds": elapsed,
        "stop_reason": infer_stop_reason(
            new_ids.tolist(),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            criterion=criterion,
        ),
        "response_control_trimmed": finalized.trimmed,
        "prompt_tokens": prompt_length,
    }


def answer_expected(row: dict[str, Any]) -> bool:
    return str(row.get("decision") or "answer") == "answer"


def conflict_expected(row: dict[str, Any]) -> bool:
    return str(row.get("decision") or "") == "conflict"


def reference_answer(row: dict[str, Any]) -> str | None:
    if row.get("reference") is not None:
        return str(row["reference"])
    messages = row.get("messages") or []
    if messages and messages[-1].get("role") == "assistant":
        return str(messages[-1].get("content") or "")
    return None


def row_quality_record(row: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "messages": messages,
        "reference": reference_answer(row),
        "category": row.get("category") or row.get("decision") or row.get("mode") or "uncategorized",
        "response_contract": row.get("response_contract") or {},
    }


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    quality = aggregate_quality_rows([row["quality"] for row in rows])
    supported = [row for row in rows if row["answer_expected"]]
    unsupported = [row for row in rows if not row["answer_expected"] and not row["conflict_expected"]]
    conflicts = [row for row in rows if row["conflict_expected"]]
    retrieval_cases = [row for row in rows if row.get("retrieval_applicable")]
    retrieval_hits = [row for row in retrieval_cases if row.get("support_retrieved")]
    generation_times = [float(row["generation_seconds"]) for row in rows]
    retrieval_times = [float(row["retrieval_seconds"]) for row in rows]
    visible_tokens = [int(row["visible_tokens"]) for row in rows]
    correct_supported = [row for row in supported if row.get("supported_answer_correct")]
    correct_abstentions = [row for row in unsupported if row.get("abstained")]
    unnecessary = [row for row in supported if row.get("abstained")]
    unsupported_confident = [row for row in unsupported if not row.get("abstained")]
    faithful = [row for row in rows if row.get("grounding_supported")]
    conflict_detected = [row for row in conflicts if row.get("conflict_response")]
    return {
        **quality,
        "supported_case_count": len(supported),
        "unsupported_case_count": len(unsupported),
        "conflict_case_count": len(conflicts),
        "supported_answer_correctness": len(correct_supported) / max(1, len(supported)),
        "evidence_faithfulness": len(faithful) / max(1, len(rows)),
        "correct_abstention_rate": len(correct_abstentions) / max(1, len(unsupported)),
        "unnecessary_abstention_rate": len(unnecessary) / max(1, len(supported)),
        "unsupported_confident_answer_rate": len(unsupported_confident) / max(1, len(unsupported)),
        "conflict_detection_rate": len(conflict_detected) / max(1, len(conflicts)),
        "retrieval_recall_at_k": len(retrieval_hits) / max(1, len(retrieval_cases)),
        "median_completion_tokens": statistics.median(visible_tokens),
        "p90_completion_tokens": sorted(visible_tokens)[min(len(visible_tokens) - 1, int(0.9 * len(visible_tokens)))],
        "mean_generation_seconds": statistics.fmean(generation_times),
        "mean_retrieval_seconds": statistics.fmean(retrieval_times),
        "stop_reason_counts": dict(Counter(str(row.get("stop_reason")) for row in rows)),
        "external_gate_reason_counts": dict(Counter(str(row.get("gate_reason")) for row in rows)),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 9 RAG evaluation",
        "",
        f"- Model: `{summary['model']}`",
        f"- Adapter: `{summary.get('adapter')}`",
        f"- Benchmark: `{summary['benchmark']}`",
        f"- Model/checkpoint bytes: `{summary['model_weight_bytes']}`",
        f"- Peak allocated VRAM bytes: `{summary.get('peak_allocated_vram_bytes')}`",
        f"- Process RSS bytes: `{summary.get('process_rss_bytes')}`",
        "",
        "## Conditions",
        "",
    ]
    for condition, packet in summary["conditions"].items():
        lines.extend(
            [
                f"### {condition}",
                "",
                f"- Cases: `{packet.get('case_count')}`",
                f"- Mean quality: `{packet.get('mean_quality_score')}`",
                f"- Supported correctness: `{packet.get('supported_answer_correctness')}`",
                f"- Evidence faithfulness: `{packet.get('evidence_faithfulness')}`",
                f"- Correct abstention: `{packet.get('correct_abstention_rate')}`",
                f"- Unsupported confident answer rate: `{packet.get('unsupported_confident_answer_rate')}`",
                f"- Retrieval recall@k: `{packet.get('retrieval_recall_at_k')}`",
                f"- Median/P90 completion tokens: `{packet.get('median_completion_tokens')}` / `{packet.get('p90_completion_tokens')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    conditions = normalize_condition_list(args.conditions)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    if "real_rag" in conditions and args.knowledge_db is None:
        raise ValueError("real_rag requires --knowledge-db")
    if args.adapter is not None and PeftModel is None:
        raise RuntimeError("PEFT is required when --adapter is supplied")

    benchmark = read_jsonl(args.benchmark)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    quality_config = load_quality_config()
    corpus_profile = build_corpus_profile(benchmark)
    minimum_score: float | None = None
    if args.gate_calibration is not None:
        calibration = read_json(args.gate_calibration)
        minimum_score = float(calibration["minimum_retrieval_score"])
    elif "real_rag" in conditions and not args.allow_uncalibrated_real_rag:
        raise ValueError(
            "real_rag requires --gate-calibration. Use --allow-uncalibrated-real-rag only for diagnostics; "
            "uncalibrated scores cannot support a release decision."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        existing, _ = read_jsonl_recover(results_path, id_field="evaluation_id")
        completed = {str(row["evaluation_id"]): row for row in existing}
    elif results_path.exists():
        raise FileExistsError(f"Evaluation results already exist; use --resume: {results_path}")

    retriever = SQLiteFTSRetriever(args.knowledge_db, read_only=True) if args.knowledge_db else None
    try:
        for condition in conditions:
            for row in benchmark:
                evaluation_id = f"{condition}:{row.get('id')}"
                if evaluation_id in completed:
                    continue
                question = str(row.get("question") or "")
                mode = route_mode(question, row.get("mode"))
                evidence, procedures, rejected, retrieval_seconds = prepare_evidence(
                    row=row,
                    condition=condition,
                    retriever=retriever,
                    retrieval_top_k=args.retrieval_top_k,
                    reranked_top_k=args.reranked_top_k,
                )
                gate_reason = "not_applicable"
                external_answer: str | None = None
                if mode != "social_roleplay":
                    if condition == "closed_book":
                        external_answer = ABSTENTION_TEXT
                        gate_reason = "closed_book_no_evidence"
                    elif condition == "real_rag":
                        gate = evidence_gate(
                            question,
                            evidence,
                            minimum_score=minimum_score,
                            require_calibrated_threshold=not args.allow_uncalibrated_real_rag,
                        )
                        gate_reason = gate.reason
                        if gate.conflict_detected:
                            external_answer = CONFLICT_TEXT
                        elif not gate.should_answer:
                            external_answer = ABSTENTION_TEXT
                    else:
                        gate_reason = "oracle_or_adversarial_evidence_supplied"

                messages = build_grounded_messages(
                    question=question,
                    evidence=evidence,
                    procedures=procedures,
                    mode=mode,
                    system_prompt=row.get("system_prompt"),
                )
                if external_answer is None:
                    generated = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        max_sequence_length=args.max_sequence_length,
                        max_new_tokens=args.max_new_tokens,
                        response_control=args.response_control,
                    )
                else:
                    generated = {
                        "answer": external_answer,
                        "raw_answer": external_answer,
                        "generated_tokens": 0,
                        "visible_tokens": len(tokenizer(external_answer, add_special_tokens=False)["input_ids"]),
                        "generation_seconds": 0.0,
                        "stop_reason": "external_gate",
                        "response_control_trimmed": False,
                        "prompt_tokens": 0,
                    }
                answer = generated["answer"]
                grounding = grounding_check(answer, evidence)
                reference = reference_answer(row)
                answer_f1 = token_f1(reference, answer) if reference else None
                quality = evaluate_response(
                    row_quality_record(row, messages),
                    answer,
                    max(1, generated["visible_tokens"]),
                    visible_model_tokens=max(1, generated["visible_tokens"]),
                    config=quality_config,
                    corpus_profile=corpus_profile,
                )
                expected_support = set(str(value) for value in (row.get("support_document_ids") or []))
                retrieved_ids = {passage_id(item) for item in evidence}
                expected_answer = answer_expected(row)
                expected_conflict = conflict_expected(row)
                abstained = is_abstention(answer)
                conflict_response = "conflict" in answer.casefold() or "disagree" in answer.casefold()
                supported_correct = bool(expected_answer and not abstained and quality["quality_score"] >= 0.5)
                if answer_f1 is not None:
                    supported_correct = supported_correct and answer_f1 > 0.0
                result = {
                    "evaluation_id": evaluation_id,
                    "id": row.get("id"),
                    "condition": condition,
                    "question": question,
                    "mode": mode,
                    "decision": row.get("decision"),
                    "answer": answer,
                    "reference": reference,
                    "answer_token_f1": answer_f1,
                    "answer_expected": expected_answer,
                    "conflict_expected": expected_conflict,
                    "supported_answer_correct": supported_correct,
                    "abstained": abstained,
                    "conflict_response": conflict_response,
                    "evidence": evidence,
                    "procedures": procedures,
                    "rejected_passages": rejected,
                    "retrieved_document_ids": sorted(retrieved_ids),
                    "support_document_ids": sorted(expected_support),
                    "retrieval_applicable": bool(expected_support) and condition == "real_rag",
                    "support_retrieved": bool(expected_support & retrieved_ids),
                    "gate_reason": gate_reason,
                    "grounding_supported": grounding.supported,
                    "grounding": {
                        "hard_failures": list(grounding.hard_failures),
                        "diagnostics": list(grounding.diagnostics),
                        "unsupported_numbers": list(grounding.unsupported_numbers),
                        "lexical_support_fraction": grounding.lexical_support_fraction,
                    },
                    "quality": quality,
                    "retrieval_seconds": retrieval_seconds,
                    **generated,
                }
                append_jsonl_durable(results_path, result)
                completed[evaluation_id] = result
                print(f"Evaluated {evaluation_id}")
    finally:
        if retriever is not None:
            retriever.close()

    ordered = [completed[f"{condition}:{row.get('id')}"] for condition in conditions for row in benchmark]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_condition[row["condition"]].append(row)
    condition_packets = {condition: condition_summary(by_condition[condition]) for condition in conditions}
    primary_rows = [
        row
        for row in ordered
        if row["condition"] in ({"oracle_rag", "real_rag"} & set(conditions) or set(conditions))
    ]
    overall_quality = aggregate_quality_rows([row["quality"] for row in primary_rows])
    model_weight_bytes = tree_bytes(Path(args.model)) if Path(args.model).exists() else 0
    if args.adapter is not None:
        model_weight_bytes += tree_bytes(args.adapter)
    summary = {
        **overall_quality,
        "phase": "9_student_distillation",
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256(args.benchmark),
        "conditions": condition_packets,
        "primary_quality_conditions": sorted({row["condition"] for row in primary_rows}),
        "model_weight_bytes": model_weight_bytes,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "process_rss_bytes": process_rss_bytes(),
        "knowledge_db": str(args.knowledge_db) if args.knowledge_db else None,
        "gate_calibration": str(args.gate_calibration) if args.gate_calibration else None,
        "response_control": args.response_control,
        "max_new_tokens": args.max_new_tokens,
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
