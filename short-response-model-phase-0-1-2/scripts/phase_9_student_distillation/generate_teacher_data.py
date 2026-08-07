from __future__ import annotations

"""Generate auditable teacher candidates for Phase 9 RAG distillation.

The output is deliberately a review packet, not automatically approved training
material. A human must set ``review_status`` to ``accepted`` and select or write
an approved answer before ``build_pilot_dataset.py`` will consume the row.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import (
    append_jsonl_durable,
    environment,
    read_jsonl,
    read_jsonl_recover,
    sha256,
    write_json,
)
from scripts.common.quality_evaluation import build_corpus_profile, evaluate_response, load_quality_config
from scripts.common.response_control import build_stopping_criteria, finalize_response, infer_stop_reason
from scripts.phase_9_student_distillation.phase9_common import (
    ABSTENTION_TEXT,
    SQLiteFTSRetriever,
    build_grounded_messages,
    grounding_check,
    passage_id,
    route_mode,
    sanitize_passages,
    validate_prompt_record,
)
from scripts.phase_9_student_distillation.phase9_prompting import plan_grounded_prompt, token_ids

TEACHER_PIPELINE_REVISION = "phase9_rag_reasoning_integration_v1_7_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 9 teacher answers and a human review packet.")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--knowledge-db", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, default=None)
    parser.add_argument("--retrieval-mode", choices=("oracle", "real"), default="oracle")
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--reranked-top-k", type=int, default=3)
    parser.add_argument("--samples-per-question", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--maximum-evidence-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="eos_only")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def oracle_passages(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = row.get(key) or []
    output: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            output.append({"record_id": f"{row['id']}_{key}_{index}", "text": value, "normalized_score": 1.0})
        elif isinstance(value, dict):
            item = dict(value)
            item.setdefault("record_id", passage_id(item))
            item.setdefault("normalized_score", 1.0)
            output.append(item)
        else:
            raise ValueError(f"{row['id']} contains a non-string/non-object {key} passage")
    return output


def retrieve_passages(
    row: dict[str, Any],
    retriever: SQLiteFTSRetriever,
    *,
    top_k: int,
    final_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    factual = retriever.search(row["question"], limit=top_k, kind="knowledge")
    procedures = retriever.search(row["question"], limit=max(3, final_k), kind="procedure")
    factual, rejected = sanitize_passages(factual)
    procedures, rejected_procedures = sanitize_passages(procedures)
    return factual[:final_k], procedures[: min(2, final_k)], [*rejected, *rejected_procedures]


def generate_candidate(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    response_control: str,
    sample_index: int,
    temperature: float,
    top_p: float,
    max_sequence_length: int,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    prompt = prompt_text if prompt_text is not None else apply_chat_template(tokenizer, messages)
    full_ids = token_ids(tokenizer, prompt)
    if len(full_ids) > max_sequence_length:
        raise ValueError(
            f"Teacher prompt requires {len(full_ids)} tokens, exceeding max_sequence_length={max_sequence_length}; "
            "evidence must be budgeted before generation."
        )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    stopping, criterion = build_stopping_criteria(tokenizer, prompt_length, response_control)
    do_sample = sample_index > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": stopping,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[0, prompt_length:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    finalized = finalize_response(raw, response_control)
    stop_reason = infer_stop_reason(
        new_ids.tolist(),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        criterion=criterion,
    )
    return {
        "answer": finalized.text,
        "raw_answer": finalized.raw_text,
        "visible_token_count": len(tokenizer(finalized.text, add_special_tokens=False)["input_ids"]),
        "generated_token_count": int(new_ids.numel()),
        "stop_reason": stop_reason,
        "response_control_trimmed": finalized.trimmed,
        "generation_seconds": elapsed,
        "sampled": do_sample,
    }


def review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 9 teacher-data human review packet",
        "",
        "Set each accepted JSONL row to `review_status: accepted` and either set "
        "`selected_candidate_index` or write `approved_answer`. Reject unsupported, verbose, "
        "unsafe, role-breaking, or factually incomplete candidates.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- Mode: `{row['mode']}`",
                f"- Topic: `{row['topic_id']}`",
                f"- Question: {row['question']}",
                f"- Decision target: `{row.get('decision', 'unspecified')}`",
                "- Evidence IDs: " + ", ".join(row.get("retrieved_document_ids") or ["(none)"]),
                "",
            ]
        )
        for index, candidate in enumerate(row.get("candidates") or []):
            lines.extend(
                [
                    f"### Candidate {index}",
                    "",
                    candidate.get("answer") or "*(empty)*",
                    "",
                    f"- Grounding: `{candidate['grounding']['supported']}`",
                    f"- Quality: `{candidate['quality'].get('quality_score')}`",
                    f"- Source: `{candidate.get('candidate_source', 'unknown')}`",
                    f"- Stop: `{candidate['stop_reason']}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.samples_per_question <= 0:
        raise ValueError("--samples-per-question must be positive")
    if args.retrieval_mode == "real" and args.knowledge_db is None:
        raise ValueError("--knowledge-db is required for real retrieval")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    prompts = [validate_prompt_record(row) for row in read_jsonl(args.prompts)]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        existing, _ = read_jsonl_recover(args.output, id_field="id", repair_truncated_tail=True)
        prompts_digest = sha256(args.prompts)
        incompatible: list[str] = []
        for prior in existing:
            provenance = prior.get("teacher_provenance") or {}
            prompt_integrity = prior.get("prompt_integrity") or {}
            compatible = (
                str(prior.get("schema_version") or "") == "1.4"
                and str(provenance.get("pipeline_revision") or "") == TEACHER_PIPELINE_REVISION
                and str(provenance.get("model") or "") == str(args.teacher)
                and str(provenance.get("prompts_sha256") or "") == prompts_digest
                and str(provenance.get("retrieval_mode") or "") == str(args.retrieval_mode)
                and int(provenance.get("max_sequence_length") or -1) == int(args.max_sequence_length)
                and int(provenance.get("maximum_evidence_tokens") or -1) == int(args.maximum_evidence_tokens)
                and bool(prompt_integrity.get("integrity_ok"))
                and not bool(prompt_integrity.get("hidden_truncation_used"))
            )
            if compatible:
                completed[str(prior["id"])] = prior
            else:
                incompatible.append(str(prior.get("id") or "<missing-id>"))
        if incompatible:
            raise RuntimeError(
                "Existing teacher candidates are incompatible with the Phase 9 v1.7.2 grounded-recovery "
                "contract. Use a fresh OutputRoot or archive/remove the old candidate JSONL before "
                f"Bootstrap. Incompatible IDs: {incompatible[:10]}"
            )
    elif args.output.exists():
        raise FileExistsError(f"Output exists; use --resume or choose another path: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    quality_config = load_quality_config()
    corpus_profile = build_corpus_profile(prompts)
    retriever = SQLiteFTSRetriever(args.knowledge_db, read_only=True) if args.knowledge_db else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for row in prompts:
            if row["id"] in completed:
                continue
            mode = route_mode(row["question"], row["mode"])
            rejected: list[dict[str, Any]] = []
            if mode == "social_roleplay":
                evidence: list[dict[str, Any]] = []
                procedures: list[dict[str, Any]] = []
            elif args.retrieval_mode == "oracle":
                evidence, rejected_evidence = sanitize_passages(oracle_passages(row, "evidence"))
                procedures, rejected_procedures = sanitize_passages(oracle_passages(row, "procedures"))
                rejected = [*rejected_evidence, *rejected_procedures]
            else:
                assert retriever is not None
                evidence, procedures, rejected = retrieve_passages(
                    row,
                    retriever,
                    top_k=args.retrieval_top_k,
                    final_k=args.reranked_top_k,
                )
            prompt_plan = plan_grounded_prompt(
                tokenizer,
                question=row["question"],
                evidence=evidence,
                procedures=procedures,
                mode=mode,
                system_prompt=row.get("system_prompt"),
                max_sequence_length=args.max_sequence_length,
                maximum_evidence_tokens=args.maximum_evidence_tokens,
            )
            if prompt_plan.integrity_issues:
                raise RuntimeError(
                    f"Prompt integrity failure for {row['id']}: " + "; ".join(prompt_plan.integrity_issues)
                )
            expected_support = {str(value) for value in (row.get("support_document_ids") or [])}
            prompted_support = {passage_id(item) for item in prompt_plan.selected_evidence}
            if expected_support and not (expected_support & prompted_support):
                raise RuntimeError(
                    f"Required support was removed by the prompt budget for {row['id']}: "
                    f"expected={sorted(expected_support)}, prompted={sorted(prompted_support)}"
                )
            messages = prompt_plan.messages
            evidence = prompt_plan.selected_evidence
            procedures = prompt_plan.selected_procedures
            candidates: list[dict[str, Any]] = []
            for sample_index in range(args.samples_per_question):
                candidate = generate_candidate(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    response_control=args.response_control,
                    sample_index=sample_index,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_sequence_length=args.max_sequence_length,
                    prompt_text=prompt_plan.prompt_text,
                )
                grounding = grounding_check(candidate["answer"], evidence)
                quality_record = {
                    "id": row["id"],
                    "messages": messages,
                    "reference": row.get("answer") or row.get("reference"),
                    "category": row.get("category") or row.get("decision") or mode,
                    "response_contract": row.get("response_contract") or {},
                }
                quality = evaluate_response(
                    quality_record,
                    candidate["answer"],
                    max(1, candidate["visible_token_count"]),
                    visible_model_tokens=max(1, candidate["visible_token_count"]),
                    config=quality_config,
                    corpus_profile=corpus_profile,
                )
                candidate["grounding"] = {
                    "supported": grounding.supported,
                    "hard_failures": list(grounding.hard_failures),
                    "diagnostics": list(grounding.diagnostics),
                    "unsupported_numbers": list(grounding.unsupported_numbers),
                    "lexical_support_fraction": grounding.lexical_support_fraction,
                }
                candidate["quality"] = quality
                candidate["candidate_source"] = "teacher_generation"
                candidates.append(candidate)

            authoritative_answer = str(row.get("answer") or row.get("reference") or "").strip()
            if authoritative_answer and all(
                authoritative_answer.casefold() != str(item.get("answer") or "").strip().casefold()
                for item in candidates
            ):
                grounding = grounding_check(authoritative_answer, evidence)
                quality_record = {
                    "id": row["id"],
                    "messages": messages,
                    "reference": authoritative_answer,
                    "category": row.get("category") or row.get("decision") or mode,
                    "response_contract": row.get("response_contract") or {},
                }
                quality = evaluate_response(
                    quality_record,
                    authoritative_answer,
                    max(1, len(tokenizer(authoritative_answer, add_special_tokens=False)["input_ids"])),
                    visible_model_tokens=max(1, len(tokenizer(authoritative_answer, add_special_tokens=False)["input_ids"])),
                    config=quality_config,
                    corpus_profile=corpus_profile,
                )
                candidates.insert(
                    0,
                    {
                        "answer": authoritative_answer,
                        "raw_answer": authoritative_answer,
                        "visible_token_count": len(tokenizer(authoritative_answer, add_special_tokens=False)["input_ids"]),
                        "generated_token_count": 0,
                        "stop_reason": "authoritative_target",
                        "response_control_trimmed": False,
                        "generation_seconds": 0.0,
                        "sampled": False,
                        "candidate_source": "authoritative_prompt_target",
                        "grounding": {
                            "supported": grounding.supported,
                            "hard_failures": list(grounding.hard_failures),
                            "diagnostics": list(grounding.diagnostics),
                            "unsupported_numbers": list(grounding.unsupported_numbers),
                            "lexical_support_fraction": grounding.lexical_support_fraction,
                        },
                        "quality": quality,
                    },
                )

            output_row = {
                "schema_version": "1.4",
                "id": row["id"],
                "topic_id": row["topic_id"],
                "question": row["question"],
                "mode": mode,
                "decision": row.get("decision"),
                "category": row.get("category"),
                "scenario_flags": list(row.get("scenario_flags") or []),
                "messages": messages,
                "evidence": evidence,
                "procedures": procedures,
                "retrieved_document_ids": [passage_id(item) for item in evidence],
                "retrieved_procedure_ids": [passage_id(item) for item in procedures],
                "support_document_ids": row.get("support_document_ids") or [],
                "rejected_passages": rejected,
                "prompt_integrity": prompt_plan.diagnostics(),
                "response_contract": row.get("response_contract") or {},
                "authoritative_answer": authoritative_answer or None,
                "candidates": candidates,
                "review_status": "pending_human_review",
                "selected_candidate_index": None,
                "approved_answer": None,
                "teacher_provenance": {
                    "pipeline_revision": TEACHER_PIPELINE_REVISION,
                    "model": args.teacher,
                    "prompts_path": str(args.prompts),
                    "prompts_sha256": sha256(args.prompts),
                    "retrieval_mode": args.retrieval_mode,
                    "knowledge_db": str(args.knowledge_db) if args.knowledge_db else None,
                    "response_control": args.response_control,
                    "max_new_tokens": args.max_new_tokens,
                    "max_sequence_length": args.max_sequence_length,
                    "maximum_evidence_tokens": args.maximum_evidence_tokens,
                    "prompt_truncation_policy": "budget passages then fail closed",
                    "environment": environment(),
                },
            }
            append_jsonl_durable(args.output, output_row)
            completed[row["id"]] = output_row
            print(f"Generated {row['id']} ({len(completed)}/{len(prompts)})")
    finally:
        if retriever is not None:
            retriever.close()

    ordered = [completed[row["id"]] for row in prompts if row["id"] in completed]
    review_path = args.review_packet or args.output.with_suffix(".review.md")
    review_path.write_text(review_markdown(ordered), encoding="utf-8")
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "schema_version": "1.4",
            "pipeline_revision": TEACHER_PIPELINE_REVISION,
            "maximum_evidence_tokens": args.maximum_evidence_tokens,
            "max_sequence_length": args.max_sequence_length,
            "record_count": len(ordered),
            "pending_human_review": sum(row.get("review_status") != "accepted" for row in ordered),
            "teacher": args.teacher,
            "prompts": str(args.prompts),
            "prompts_sha256": sha256(args.prompts),
            "output": str(args.output),
            "review_packet": str(review_path),
        },
    )
    print(f"Teacher candidates: {args.output}")
    print(f"Review packet: {review_path}")
    print("No candidate is training-approved until human review_status is accepted.")


if __name__ == "__main__":
    main()
