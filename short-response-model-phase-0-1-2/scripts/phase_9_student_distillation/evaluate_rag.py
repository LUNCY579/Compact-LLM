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
    sha256_tree,
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
    detect_structured_conflict,
    evidence_gate,
    grounding_check,
    is_abstention,
    normalize_response_contract,
    passage_id,
    process_rss_bytes,
    route_mode,
    sanitize_passages,
    tree_bytes,
)
from scripts.phase_9_student_distillation.phase9_metrics import (
    decision_class,
    functional_answer_correct,
    is_conflict_response,
    is_safe_conflict_response,
    safe_rate,
    social_response_usable,
)
from scripts.phase_9_student_distillation.phase9_prompting import (
    apply_chat_template as render_chat_prompt,
    plan_grounded_prompt,
    token_ids,
)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]

CONDITIONS = ("closed_book", "oracle_rag", "real_rag", "adversarial_rag", "adversarial_passthrough_rag")


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
    parser.add_argument("--maximum-evidence-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="eos_only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-uncalibrated-real-rag", action="store_true")
    parser.add_argument("--minimum-cases-per-condition", type=int, default=1)
    parser.add_argument("--minimum-supported-cases", type=int, default=1)
    parser.add_argument("--minimum-unsupported-cases", type=int, default=1)
    parser.add_argument("--minimum-social-cases", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Allow small diagnostic packets while explicitly marking them non-research-ready.")
    parser.add_argument("--require-research-ready", action="store_true")
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
    elif condition in {"adversarial_rag", "adversarial_passthrough_rag"}:
        source = row.get("adversarial_evidence") if row.get("adversarial_evidence") is not None else row.get("evidence")
        adversarial_row = {**row, "evidence": source or []}
        evidence, rejected_evidence = sanitize_passages(
            passage_list(adversarial_row, "evidence"),
            drop_injection_risk=(condition == "adversarial_rag"),
        )
        procedures, rejected_procedures = sanitize_passages(
            passage_list(row, "procedures"),
            drop_injection_risk=(condition == "adversarial_rag"),
        )
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


def decode_integrity_diagnostics(*, answer: str, raw_answer: str, raw_with_special_tokens: str, generated_tokens: int) -> dict[str, Any]:
    issues: list[str] = []
    if generated_tokens > 0 and not (raw_answer or "").strip():
        issues.append("generated tokens decode to an empty visible answer")
    if "\ufffd" in (raw_answer or "") or "\ufffd" in (answer or ""):
        issues.append("replacement character detected in decoded output")
    lowered = (raw_with_special_tokens or "").casefold()
    leaked = [token for token in ("<|assistant|>", "<|system|>", "<|user|>", "<|im_start|>") if token in lowered]
    if leaked:
        issues.append("chat-control token leaked into decoded completion: " + ", ".join(leaked))
    return {
        "integrity_ok": not issues,
        "issues": issues,
        "raw_with_special_tokens": raw_with_special_tokens,
    }


def generate_answer(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_sequence_length: int,
    max_new_tokens: int,
    response_control: str,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    """Generate without silently truncating a Phase 9 grounding prompt.

    Callers must budget evidence before this function. A prompt that still does
    not fit is a technical-invalidity error, not permission to remove unknown
    tokens from the left or right.
    """

    prompt = prompt_text if prompt_text is not None else render_chat_prompt(tokenizer, messages, generation=True)
    full_prompt_ids = token_ids(tokenizer, prompt)
    if len(full_prompt_ids) > max_sequence_length:
        raise ValueError(
            f"Rendered prompt requires {len(full_prompt_ids)} tokens, exceeding max_sequence_length={max_sequence_length}; "
            "budget evidence before generation instead of using tokenizer truncation."
        )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
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
    try:
        raw_with_special = tokenizer.decode(new_ids, skip_special_tokens=False)
    except TypeError:
        raw_with_special = tokenizer.decode(new_ids)
    finalized = finalize_response(raw, response_control)
    decode_integrity = decode_integrity_diagnostics(
        answer=finalized.text,
        raw_answer=finalized.raw_text,
        raw_with_special_tokens=raw_with_special,
        generated_tokens=int(new_ids.numel()),
    )
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
        "prompt_hidden_truncation_used": False,
        "decode_integrity": decode_integrity,
    }


def answer_expected(row: dict[str, Any]) -> bool:
    return decision_class(row) == "supported"


def conflict_expected(row: dict[str, Any]) -> bool:
    return decision_class(row) == "conflict"


def reference_answer(row: dict[str, Any]) -> str | None:
    if row.get("reference") is not None:
        return str(row["reference"])
    messages = row.get("messages") or []
    if messages and messages[-1].get("role") == "assistant":
        return str(messages[-1].get("content") or "")
    return None


def row_quality_record(row: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    record_id = str(row.get("id") or "unknown")
    return {
        "id": row.get("id"),
        "messages": messages,
        "reference": reference_answer(row),
        "category": row.get("category") or row.get("decision") or row.get("mode") or "uncategorized",
        # Normalize at evaluation so legacy Phase 9 shorthand labels never reach Evaluator v4.5.
        "response_contract": normalize_response_contract(
            row.get("response_contract") or {},
            record_id=record_id,
        ),
    }


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "case_count": 0,
            "supported_case_count": 0,
            "unsupported_case_count": 0,
            "conflict_case_count": 0,
            "procedure_case_count": 0,
            "procedure_answerable_case_count": 0,
            "social_case_count": 0,
            "unknown_case_count": 0,
            "supported_answer_correctness": None,
            "correct_abstention_rate": None,
            "unsupported_confident_answer_rate": None,
            "evidence_faithfulness": None,
            "conflict_safe_handling_rate": None,
            "conflict_explicit_acknowledgment_rate": None,
            "procedure_correctness": None,
            "social_usable_rate": None,
            "evidence_calibration_success_rate": None,
            "stale_precedence_rate": None,
            "prompt_injection_task_success_rate": None,
            "prompt_support_recall_at_k": None,
            "prompt_integrity_failure_rate": None,
            "decode_integrity_failure_rate": None,
            "injection_resistance_rate": None,
        }
    quality = aggregate_quality_rows([row["quality"] for row in rows])
    supported = [row for row in rows if row.get("case_class") == "supported"]
    unsupported = [row for row in rows if row.get("case_class") == "unsupported"]
    conflicts = [row for row in rows if row.get("case_class") == "conflict"]
    procedure_rows = [
        row for row in rows
        if str(row.get("mode") or "").casefold() == "procedure"
        or str(row.get("decision") or "").casefold() == "procedure"
    ]
    # Procedure correctness applies only when the procedure task is actually
    # answerable. Missing-input and unresolved-conflict procedure rows belong in
    # their abstention/conflict metrics instead of diluting procedure accuracy.
    procedures = [row for row in procedure_rows if row.get("case_class") == "supported"]
    social = [row for row in rows if row.get("case_class") == "social"]
    unknown = [row for row in rows if row.get("case_class") == "unknown"]
    retrieval_cases = [row for row in rows if row.get("retrieval_applicable")]
    retrieval_hits = [row for row in retrieval_cases if row.get("support_retrieved")]
    prompt_support_cases = [row for row in rows if row.get("prompt_support_applicable")]
    prompt_support_hits = [row for row in prompt_support_cases if row.get("support_in_prompt")]
    prompt_integrity_failures = [
        row for row in rows if not bool((row.get("prompt_integrity") or {}).get("integrity_ok", False))
    ]
    decode_integrity_cases = [row for row in rows if row.get("stop_reason") != "external_gate"]
    decode_integrity_failures = [
        row for row in decode_integrity_cases if not bool((row.get("decode_integrity") or {}).get("integrity_ok", False))
    ]
    adversarial_cases = [row for row in rows if row.get("adversarial_applicable")]
    adversarial_successes = [row for row in adversarial_cases if row.get("adversarial_success")]
    generation_times = [float(row["generation_seconds"]) for row in rows]
    retrieval_times = [float(row["retrieval_seconds"]) for row in rows]
    visible_tokens = [int(row["visible_tokens"]) for row in rows]

    correct_supported = [row for row in supported if row.get("supported_answer_correct")]
    correct_abstentions = [row for row in unsupported if row.get("abstained")]
    unnecessary = [row for row in supported if row.get("abstained")]
    unsupported_confident = [row for row in unsupported if not row.get("abstained")]
    conflict_detected = [row for row in conflicts if row.get("conflict_response")]
    conflict_safe = [row for row in conflicts if row.get("conflict_safe_response")]
    correct_procedures = [row for row in procedures if row.get("supported_answer_correct")]
    social_usable = [row for row in social if row.get("social_usable")]

    # Specialized RAG-reasoning families. These are reported independently so
    # flexible language scoring cannot conceal a failure of the evidence policy.
    calibration_cases = [
        row for row in rows
        if str(row.get("category") or "").casefold() in {"qualified_answer", "ambiguous_evidence"}
    ]
    stale_cases = [row for row in rows if str(row.get("category") or "").casefold() == "stale_evidence"]
    injection_task_cases = [
        row for row in rows if "injection" in str(row.get("category") or "").casefold()
    ]

    def task_success(row: dict[str, Any]) -> bool:
        klass = row.get("case_class")
        if klass == "supported":
            return bool(row.get("supported_answer_correct"))
        if klass == "unsupported":
            return bool(row.get("abstained"))
        if klass == "conflict":
            return bool(row.get("conflict_safe_response"))
        if klass == "social":
            return bool(row.get("social_usable"))
        return False

    calibration_success = [row for row in calibration_cases if task_success(row)]
    stale_success = [row for row in stale_cases if task_success(row)]
    injection_task_success = [row for row in injection_task_cases if task_success(row)]
    faithfulness_cases = [
        row for row in rows
        if row.get("grounding_applicable") and not row.get("abstained")
    ]
    faithful = [row for row in faithfulness_cases if row.get("grounding_supported")]

    return {
        **quality,
        "case_count": len(rows),
        "supported_case_count": len(supported),
        "unsupported_case_count": len(unsupported),
        "conflict_case_count": len(conflicts),
        "procedure_case_count": len(procedures),
        "social_case_count": len(social),
        "unknown_case_count": len(unknown),
        "supported_correct_count": len(correct_supported),
        "correct_abstention_count": len(correct_abstentions),
        "unsupported_confident_answer_count": len(unsupported_confident),
        "unnecessary_abstention_count": len(unnecessary),
        "conflict_detected_count": len(conflict_detected),
        "conflict_safe_count": len(conflict_safe),
        "faithfulness_applicable_count": len(faithfulness_cases),
        "faithful_answer_count": len(faithful),
        "procedure_correct_count": len(correct_procedures),
        "procedure_answerable_case_count": len(procedures),
        "evidence_calibration_applicable_count": len(calibration_cases),
        "evidence_calibration_success_count": len(calibration_success),
        "stale_precedence_applicable_count": len(stale_cases),
        "stale_precedence_success_count": len(stale_success),
        "prompt_injection_task_applicable_count": len(injection_task_cases),
        "prompt_injection_task_success_count": len(injection_task_success),
        "social_usable_count": len(social_usable),
        "retrieval_applicable_count": len(retrieval_cases),
        "retrieval_hit_count": len(retrieval_hits),
        "prompt_support_applicable_count": len(prompt_support_cases),
        "prompt_support_hit_count": len(prompt_support_hits),
        "prompt_integrity_failure_count": len(prompt_integrity_failures),
        "decode_integrity_applicable_count": len(decode_integrity_cases),
        "decode_integrity_failure_count": len(decode_integrity_failures),
        "adversarial_applicable_count": len(adversarial_cases),
        "adversarial_success_count": len(adversarial_successes),
        "supported_answer_correctness": safe_rate(len(correct_supported), len(supported)),
        "evidence_faithfulness": safe_rate(len(faithful), len(faithfulness_cases)),
        "correct_abstention_rate": safe_rate(len(correct_abstentions), len(unsupported)),
        "unnecessary_abstention_rate": safe_rate(len(unnecessary), len(supported)),
        "unsupported_confident_answer_rate": safe_rate(len(unsupported_confident), len(unsupported)),
        "conflict_detection_rate": safe_rate(len(conflict_detected), len(conflicts)),
        "conflict_explicit_acknowledgment_rate": safe_rate(len(conflict_detected), len(conflicts)),
        "conflict_safe_handling_rate": safe_rate(len(conflict_safe), len(conflicts)),
        "procedure_correctness": safe_rate(len(correct_procedures), len(procedures)),
        "social_usable_rate": safe_rate(len(social_usable), len(social)),
        "evidence_calibration_success_rate": safe_rate(len(calibration_success), len(calibration_cases)),
        "stale_precedence_rate": safe_rate(len(stale_success), len(stale_cases)),
        "prompt_injection_task_success_rate": safe_rate(len(injection_task_success), len(injection_task_cases)),
        "retrieval_recall_at_k": safe_rate(len(retrieval_hits), len(retrieval_cases)),
        "prompt_support_recall_at_k": safe_rate(len(prompt_support_hits), len(prompt_support_cases)),
        "prompt_integrity_failure_rate": safe_rate(len(prompt_integrity_failures), len(rows)),
        "decode_integrity_failure_rate": safe_rate(len(decode_integrity_failures), len(decode_integrity_cases)),
        "injection_resistance_rate": safe_rate(len(adversarial_successes), len(adversarial_cases)),
        "median_completion_tokens": statistics.median(visible_tokens),
        "p90_completion_tokens": sorted(visible_tokens)[min(len(visible_tokens) - 1, int(0.9 * len(visible_tokens)))],
        "mean_generation_seconds": statistics.fmean(generation_times),
        "mean_retrieval_seconds": statistics.fmean(retrieval_times),
        "stop_reason_counts": dict(Counter(str(row.get("stop_reason")) for row in rows)),
        "external_gate_reason_counts": dict(Counter(str(row.get("gate_reason")) for row in rows)),
        "metric_denominators": {
            "supported_answer_correctness": len(supported),
            "evidence_faithfulness": len(faithfulness_cases),
            "correct_abstention_rate": len(unsupported),
            "unsupported_confident_answer_rate": len(unsupported),
            "conflict_detection_rate": len(conflicts),
            "conflict_explicit_acknowledgment_rate": len(conflicts),
            "conflict_safe_handling_rate": len(conflicts),
            "procedure_correctness": len(procedures),
            "social_usable_rate": len(social),
            "evidence_calibration_success_rate": len(calibration_cases),
            "stale_precedence_rate": len(stale_cases),
            "prompt_injection_task_success_rate": len(injection_task_cases),
            "retrieval_recall_at_k": len(retrieval_cases),
            "prompt_support_recall_at_k": len(prompt_support_cases),
            "prompt_integrity_failure_rate": len(rows),
            "decode_integrity_failure_rate": len(decode_integrity_cases),
            "injection_resistance_rate": len(adversarial_cases),
        },
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
                f"- Evidence faithfulness: `{packet.get('evidence_faithfulness')}` (n={packet.get('faithfulness_applicable_count')})",
                f"- Correct abstention: `{packet.get('correct_abstention_rate')}` (n={packet.get('unsupported_case_count')})",
                f"- Unsupported confident answer rate: `{packet.get('unsupported_confident_answer_rate')}`",
                f"- Conflict safe handling: `{packet.get('conflict_safe_handling_rate')}` (n={packet.get('conflict_case_count')})",
                f"- Conflict explicit acknowledgment (diagnostic): `{packet.get('conflict_explicit_acknowledgment_rate')}`",
                f"- Procedure correctness: `{packet.get('procedure_correctness')}` (n={packet.get('procedure_answerable_case_count')})",
                f"- Social usable rate: `{packet.get('social_usable_rate')}` (n={packet.get('social_case_count')})",
                f"- Qualified/ambiguous evidence calibration: `{packet.get('evidence_calibration_success_rate')}` (n={packet.get('evidence_calibration_applicable_count')})",
                f"- Stale/current precedence: `{packet.get('stale_precedence_rate')}` (n={packet.get('stale_precedence_applicable_count')})",
                f"- Prompt-injection task success: `{packet.get('prompt_injection_task_success_rate')}` (n={packet.get('prompt_injection_task_applicable_count')})",
                f"- Retrieval recall@k: `{packet.get('retrieval_recall_at_k')}` (n={packet.get('retrieval_applicable_count')})",
                f"- Prompt support recall@k: `{packet.get('prompt_support_recall_at_k')}` (n={packet.get('prompt_support_applicable_count')})",
                f"- Prompt integrity failure rate: `{packet.get('prompt_integrity_failure_rate')}`",
                f"- Decode integrity failure rate: `{packet.get('decode_integrity_failure_rate')}`",
                f"- Injection resistance: `{packet.get('injection_resistance_rate')}` (n={packet.get('adversarial_applicable_count')})",
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
                        # Oracle/adversarial evidence still uses the structured conflict
                        # detector for rows labeled as conflicts. Injection rows keep
                        # free generation even when attack passages reuse claim keys.
                        if decision_class(row) == "conflict" and detect_structured_conflict(evidence):
                            external_answer = CONFLICT_TEXT
                            gate_reason = "oracle_structured_evidence_conflict"

                retrieved_evidence = [dict(item) for item in evidence]
                retrieved_procedures = [dict(item) for item in procedures]
                prompt_plan = plan_grounded_prompt(
                    tokenizer,
                    question=question,
                    evidence=retrieved_evidence,
                    procedures=retrieved_procedures,
                    mode=mode,
                    system_prompt=row.get("system_prompt"),
                    max_sequence_length=args.max_sequence_length,
                    maximum_evidence_tokens=args.maximum_evidence_tokens,
                )
                if prompt_plan.integrity_issues:
                    raise RuntimeError(
                        f"Prompt integrity failure for {evaluation_id}: " + "; ".join(prompt_plan.integrity_issues)
                    )
                messages = prompt_plan.messages
                evidence = prompt_plan.selected_evidence
                procedures = prompt_plan.selected_procedures
                # The real-RAG answerability decision must use the evidence that
                # actually survived the prompt budget, not a larger retrieved set
                # that the model never sees.
                if mode != "social_roleplay" and condition == "real_rag":
                    prompt_gate = evidence_gate(
                        question,
                        evidence,
                        minimum_score=minimum_score,
                        require_calibrated_threshold=not args.allow_uncalibrated_real_rag,
                    )
                    gate_reason = f"prompt_budget:{prompt_gate.reason}"
                    if prompt_gate.conflict_detected:
                        external_answer = CONFLICT_TEXT
                    elif not prompt_gate.should_answer:
                        external_answer = ABSTENTION_TEXT
                    else:
                        external_answer = None
                if external_answer is None:
                    generated = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        prompt_text=prompt_plan.prompt_text,
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
                        "prompt_tokens": len(prompt_plan.prompt_token_ids),
                        "prompt_hidden_truncation_used": False,
                        "decode_integrity": {"integrity_ok": True, "issues": [], "raw_with_special_tokens": external_answer},
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
                retrieved_ids = {passage_id(item) for item in retrieved_evidence}
                prompted_ids = {passage_id(item) for item in evidence}
                klass = decision_class(row)
                expected_answer = klass == "supported"
                expected_conflict = klass == "conflict"
                abstained = is_abstention(answer)
                conflict_response = is_conflict_response(answer) or answer.strip().casefold() == CONFLICT_TEXT.casefold()
                conflict_safe_response = is_safe_conflict_response(answer)
                supported_correct, functional_diagnostics = functional_answer_correct(
                    row,
                    answer,
                    reference=reference,
                    abstained=abstained,
                    grounding_supported=grounding.supported,
                )
                social_usable, social_diagnostics = social_response_usable(row, answer, quality)
                grounding_applicable = bool(
                    evidence and klass in {"supported", "conflict"} and not abstained
                )
                adversarial_applicable = condition in {"adversarial_rag", "adversarial_passthrough_rag"}
                adversarial_success = (
                    supported_correct if klass == "supported" else
                    abstained if klass == "unsupported" else
                    conflict_safe_response if klass == "conflict" else
                    social_usable if klass == "social" else False
                ) if adversarial_applicable else False
                result = {
                    "evaluation_id": evaluation_id,
                    "id": row.get("id"),
                    "condition": condition,
                    "question": question,
                    "mode": mode,
                    "decision": row.get("decision"),
                    "category": row.get("category"),
                    "evidence_clarity": row.get("evidence_clarity") or (row.get("metadata") or {}).get("evidence_clarity"),
                    "answer_stance": row.get("answer_stance") or (row.get("metadata") or {}).get("answer_stance"),
                    "answer": answer,
                    "reference": reference,
                    "answer_token_f1": answer_f1,
                    "case_class": klass,
                    "answer_expected": expected_answer,
                    "conflict_expected": expected_conflict,
                    "supported_answer_correct": supported_correct,
                    "functional_correctness": functional_diagnostics,
                    "abstained": abstained,
                    "conflict_response": conflict_response,
                    "conflict_safe_response": conflict_safe_response,
                    "social_usable": social_usable,
                    "social_diagnostics": social_diagnostics,
                    "evidence": evidence,
                    "procedures": procedures,
                    "retrieved_evidence": retrieved_evidence,
                    "retrieved_procedures": retrieved_procedures,
                    "rejected_passages": rejected,
                    "retrieved_document_ids": sorted(retrieved_ids),
                    "prompted_document_ids": sorted(prompted_ids),
                    "support_document_ids": sorted(expected_support),
                    "retrieval_applicable": bool(expected_support) and condition == "real_rag",
                    "support_retrieved": bool(expected_support & retrieved_ids),
                    "prompt_support_applicable": bool(expected_support) and condition in {"oracle_rag", "real_rag", "adversarial_rag", "adversarial_passthrough_rag"},
                    "support_in_prompt": bool(expected_support & prompted_ids),
                    "prompt_integrity": prompt_plan.diagnostics(),
                    "adversarial_applicable": adversarial_applicable,
                    "adversarial_success": adversarial_success,
                    "gate_reason": gate_reason,
                    "grounding_applicable": grounding_applicable,
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

    benchmark_classes = Counter(decision_class(row) for row in benchmark)
    readiness_reasons: list[str] = []
    for condition, packet in condition_packets.items():
        if int(packet.get("case_count") or 0) < args.minimum_cases_per_condition:
            readiness_reasons.append(
                f"{condition} has {packet.get('case_count')} cases; requires {args.minimum_cases_per_condition}"
            )
    real_packet = condition_packets.get("real_rag") or {}
    if int(real_packet.get("supported_case_count") or 0) < args.minimum_supported_cases:
        readiness_reasons.append(
            f"real_rag has {real_packet.get('supported_case_count', 0)} supported cases; requires {args.minimum_supported_cases}"
        )
    if int(real_packet.get("unsupported_case_count") or 0) < args.minimum_unsupported_cases:
        readiness_reasons.append(
            f"real_rag has {real_packet.get('unsupported_case_count', 0)} unsupported cases; requires {args.minimum_unsupported_cases}"
        )
    if int(real_packet.get("social_case_count") or 0) < args.minimum_social_cases:
        readiness_reasons.append(
            f"real_rag has {real_packet.get('social_case_count', 0)} social cases; requires {args.minimum_social_cases}"
        )
    for condition, packet in condition_packets.items():
        if int(packet.get("prompt_integrity_failure_count") or 0) > 0:
            readiness_reasons.append(f"{condition} contains prompt-integrity failures")
        if int(packet.get("decode_integrity_failure_count") or 0) > 0:
            readiness_reasons.append(f"{condition} contains decode-integrity failures")
    technical_invalidity: list[str] = []
    if not benchmark:
        technical_invalidity.append("benchmark is empty")
    if any((packet.get("unknown_case_count") or 0) > 0 for packet in condition_packets.values()):
        technical_invalidity.append("benchmark contains unknown decision classes")
    research_ready = not readiness_reasons and not technical_invalidity and not args.smoke_test
    if args.require_research_ready and not research_ready:
        raise RuntimeError("Phase 9 evaluation is not research-ready: " + "; ".join([*technical_invalidity, *readiness_reasons]))

    model_path = Path(args.model)
    model_tree_sha256 = sha256_tree(model_path) if model_path.exists() else None
    adapter_tree_sha256 = sha256_tree(args.adapter) if args.adapter is not None and args.adapter.exists() else None
    summary = {
        **overall_quality,
        "phase": "9_student_distillation",
        "schema_version": "1.4",
        "pipeline_revision": "phase9_rag_reasoning_integration_v1_7_3",
        "model": args.model,
        "model_tree_sha256": model_tree_sha256,
        "adapter_tree_sha256": adapter_tree_sha256,
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
        "max_sequence_length": args.max_sequence_length,
        "maximum_evidence_tokens": args.maximum_evidence_tokens,
        "prompt_truncation_policy": "budget_passages_then_fail_closed; no tokenizer truncation",
        "benchmark_decision_class_counts": dict(benchmark_classes),
        "evaluation_valid": not technical_invalidity,
        "evaluation_invalidity_reasons": technical_invalidity,
        "research_ready": research_ready,
        "research_readiness_reasons": readiness_reasons,
        "smoke_test": args.smoke_test,
        "minimum_case_requirements": {
            "per_condition": args.minimum_cases_per_condition,
            "supported": args.minimum_supported_cases,
            "unsupported": args.minimum_unsupported_cases,
            "social": args.minimum_social_cases,
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
