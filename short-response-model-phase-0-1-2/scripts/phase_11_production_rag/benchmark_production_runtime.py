from __future__ import annotations

"""Benchmark the complete Phase 11 retrieval → gate → generation → grounding runtime."""

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
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
from scripts.common.response_control import finalize_response
from scripts.phase_9_student_distillation.evaluate_rag import (
    answer_expected,
    apply_chat_template,
    conflict_expected,
    reference_answer,
    row_quality_record,
)
from scripts.phase_9_student_distillation.phase9_metrics import (
    decision_class,
    functional_answer_correct,
    is_safe_conflict_response,
    safe_rate,
    social_response_usable,
)
from scripts.phase_9_student_distillation.phase9_common import (
    ABSTENTION_TEXT,
    CONFLICT_TEXT,
    SQLiteFTSRetriever,
    build_grounded_messages,
    is_abstention,
    process_rss_bytes,
    sanitize_passages,
    select_retrieved_passages,
    tree_bytes,
)
from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    build_answer_messages,
    build_reconstruction_messages,
    memory_from_phase9_passage,
    parse_reconstruction_output,
)
from scripts.phase_11_production_rag.phase11_common import (
    answerability_gate,
    passage_id,
    production_grounding_check,
    route_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Phase 11 production RAG runtime.")
    parser.add_argument("--model", required=True, help="Transformers directory/model ID or GGUF path.")
    parser.add_argument("--runtime-backend", choices=("transformers", "llama_cpp"), default="transformers")
    parser.add_argument("--tokenizer-source", default=None, help="Required for GGUF/llama.cpp prompt formatting and token counts.")
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument("--llama-cli-arg", action="append", default=[])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--knowledge-db", type=Path, required=True)
    parser.add_argument("--answerability-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--selected-top-k", type=int, default=3)
    parser.add_argument("--retrieval-selection-policy", choices=("legacy", "query_aware_v21"), default="query_aware_v21")
    parser.add_argument(
        "--maximum-evidence-tokens",
        type=int,
        default=512,
        help="Maximum tokenizer tokens retained across trusted procedures and factual evidence.",
    )
    parser.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="one_sentence")
    parser.add_argument(
        "--memory-reconstruction-mode",
        choices=("off", "prompted", "trained"),
        default="off",
        help="Optional Phase 9B source-state reconstruction pass before final generation.",
    )
    parser.add_argument("--reconstruction-max-new-tokens", type=int, default=128)
    parser.add_argument("--high-risk-policy", choices=("abstain", "fallback"), default="abstain")
    parser.add_argument("--complex-policy", choices=("clarify", "abstain", "fallback"), default="clarify")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def process_memory_packet() -> dict[str, int | None]:
    """Return current and peak process memory without adding a runtime dependency."""
    current = process_rss_bytes()
    peak: int | None = None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                current = int(counters.WorkingSetSize)
                peak = int(counters.PeakWorkingSetSize)
        except Exception:
            peak = None
    else:
        try:
            import resource

            raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # Linux reports KiB; macOS reports bytes.
            peak = raw if sys.platform == "darwin" else raw * 1024
        except Exception:
            peak = None
    return {
        "process_rss_bytes": current,
        "peak_process_rss_bytes": peak,
        "total_process_memory_bytes": peak or current,
    }


def enforce_evidence_budget(
    tokenizer: Any,
    evidence: list[dict[str, Any]],
    procedures: list[dict[str, Any]],
    maximum_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep an auditable, priority-ordered subset inside the production context budget."""
    if maximum_tokens <= 0:
        return [], [], [*evidence, *procedures]
    candidates: list[tuple[str, dict[str, Any]]] = []
    if evidence:
        candidates.append(("evidence", evidence[0]))
    if procedures:
        candidates.append(("procedure", procedures[0]))
    candidates.extend(("evidence", item) for item in evidence[1:])
    candidates.extend(("procedure", item) for item in procedures[1:])
    kept_evidence: list[dict[str, Any]] = []
    kept_procedures: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used = 0
    for kind, item in candidates:
        rendered = f"[{passage_id(item)}]\n{item.get('text', '')}"
        cost = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if used + cost > maximum_tokens:
            rejected.append({**item, "rejection_reason": "maximum_evidence_tokens"})
            continue
        used += cost
        if kind == "evidence":
            kept_evidence.append(item)
        else:
            kept_procedures.append(item)
    return kept_evidence, kept_procedures, rejected


def child_process_rss_bytes(pid: int) -> int | None:
    """Best-effort resident memory for an external runtime process."""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
            )
            if not handle:
                return None
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                )
                return int(counters.WorkingSetSize) if ok else None
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return None
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        try:
            for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except Exception:
            return None
    return None


def load_transformers_model(path: str, *, load_in_4bit: bool = False, load_in_8bit: bool = False) -> Any:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type="nf4" if load_in_4bit else "fp4",
            bnb_4bit_use_double_quant=bool(load_in_4bit),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model


def measure_prefill(model: Any, tokenizer: Any, messages: list[dict[str, Any]], max_length: int) -> tuple[float, int]:
    prompt = apply_chat_template(tokenizer, messages)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        model(**encoded, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started, int(encoded["input_ids"].shape[1])


def generate_transformers(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_sequence_length: int,
    max_new_tokens: int,
    response_control: str,
) -> dict[str, Any]:
    from scripts.phase_9_student_distillation.evaluate_rag import generate_answer

    prefill_seconds, prompt_tokens = measure_prefill(model, tokenizer, messages, max_sequence_length)
    generated = generate_answer(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
        response_control=response_control,
    )
    generated["prefill_probe_seconds"] = prefill_seconds
    generated["prompt_tokens"] = prompt_tokens
    generated["runtime_backend"] = "transformers"
    return generated


def generate_llama_cpp(
    llama_cli: Path,
    model_path: Path,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    response_control: str,
    extra_args: list[str],
) -> dict[str, Any]:
    prompt = apply_chat_template(tokenizer, messages)
    command = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(max_new_tokens),
        "--temp",
        "0",
        "--no-display-prompt",
        "--simple-io",
        *extra_args,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    child_peak_rss = 0
    while process.poll() is None:
        current_child_rss = child_process_rss_bytes(process.pid)
        if current_child_rss is not None:
            child_peak_rss = max(child_peak_rss, current_child_rss)
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"llama.cpp generation failed ({process.returncode}): {stderr[-2000:]}"
        )
    raw = stdout.strip()
    finalized = finalize_response(raw, response_control)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    visible_tokens = len(tokenizer(finalized.text, add_special_tokens=False)["input_ids"])
    return {
        "answer": finalized.text,
        "raw_answer": finalized.raw_text,
        "generated_tokens": visible_tokens,
        "visible_tokens": visible_tokens,
        "generation_seconds": elapsed,
        "prefill_probe_seconds": None,
        "prompt_tokens": prompt_tokens,
        "stop_reason": "llama_cpp_process_completed",
        "response_control_trimmed": finalized.trimmed,
        "runtime_backend": "llama_cpp",
        "external_runtime_peak_rss_bytes": child_peak_rss or None,
    }


def main() -> None:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of --load-in-4bit/--load-in-8bit")
    if args.runtime_backend == "llama_cpp":
        if args.llama_cli is None or not args.llama_cli.exists():
            raise ValueError("llama_cpp runtime requires --llama-cli")
        if not Path(args.model).exists():
            raise FileNotFoundError(args.model)
        if not args.tokenizer_source:
            raise ValueError("llama_cpp runtime requires --tokenizer-source")

    benchmark = read_jsonl(args.benchmark)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    if not benchmark:
        raise ValueError("Production benchmark is empty")
    calibration = read_json(args.answerability_calibration)
    quality_config = load_quality_config()
    corpus_profile = build_corpus_profile(benchmark)
    tokenizer_source = args.tokenizer_source or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = None
    fallback_model = None
    fallback_tokenizer = None
    if args.runtime_backend == "transformers":
        model = load_transformers_model(
            args.model,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    completed_rows: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        existing, _ = read_jsonl_recover(results_path, id_field="id")
        completed_rows = {str(row["id"]): row for row in existing}
    elif results_path.exists():
        raise FileExistsError(f"Results exist; use --resume: {results_path}")

    with SQLiteFTSRetriever(args.knowledge_db, read_only=True) as retriever:
        for row in benchmark:
            record_id = str(row.get("id"))
            if record_id in completed_rows:
                continue
            started_total = time.perf_counter()
            question = str(row.get("question") or "")
            route = route_request(question, row.get("mode"))
            retrieval_started = time.perf_counter()
            evidence: list[dict[str, Any]] = []
            procedures: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            if route.mode in {"grounded_factual", "procedure"}:
                raw_evidence = retriever.search(question, limit=args.retrieval_top_k, kind="knowledge")
                evidence, rejected_evidence = sanitize_passages(raw_evidence)
                evidence = select_retrieved_passages(question, evidence, selected_top_k=args.selected_top_k, kind="knowledge", policy=args.retrieval_selection_policy)
                rejected.extend(rejected_evidence)
                if route.mode == "procedure":
                    raw_procedures = retriever.search(question, limit=max(3, args.selected_top_k), kind="procedure")
                    procedures, rejected_procedures = sanitize_passages(raw_procedures)
                    procedures = select_retrieved_passages(question, procedures, selected_top_k=min(2, args.selected_top_k), kind="procedure", policy=args.retrieval_selection_policy)
                    rejected.extend(rejected_procedures)
                evidence, procedures, budget_rejected = enforce_evidence_budget(
                    tokenizer, evidence, procedures, args.maximum_evidence_tokens
                )
                rejected.extend(budget_rejected)
            retrieval_seconds = time.perf_counter() - retrieval_started

            external_answer: str | None = None
            gate_reason = "not_applicable"
            used_fallback = False
            if route.mode in {"grounded_factual", "procedure"}:
                gate_passages = [*procedures, *evidence] if route.mode == "procedure" else evidence
                gate = answerability_gate(question, gate_passages, calibration)
                gate_reason = gate.reason
                if gate.conflict_detected:
                    external_answer = CONFLICT_TEXT
                elif not gate.should_answer:
                    external_answer = ABSTENTION_TEXT
            elif route.mode == "high_risk":
                if args.high_risk_policy == "fallback" and args.fallback_model:
                    used_fallback = True
                    gate_reason = "high_risk_fallback"
                else:
                    external_answer = "Please use a qualified professional or emergency service for that."
                    gate_reason = "high_risk_external_boundary"
            elif route.mode == "complex":
                if args.complex_policy == "fallback" and args.fallback_model:
                    used_fallback = True
                    gate_reason = "complex_fallback"
                elif args.complex_policy == "clarify":
                    external_answer = "That needs a longer answer. Which part should I handle first?"
                    gate_reason = "complex_clarification"
                else:
                    external_answer = ABSTENTION_TEXT
                    gate_reason = "complex_abstention"

            messages = build_grounded_messages(
                question=question,
                evidence=evidence,
                procedures=procedures,
                mode=route.mode if route.mode in {"grounded_factual", "procedure", "social_roleplay"} else "social_roleplay",
                system_prompt=row.get("system_prompt"),
            )

            reconstruction_packet: dict[str, Any] | None = None
            reconstruction_generated: dict[str, Any] | None = None
            reconstruction_memories: list[dict[str, Any]] = []
            if procedures:
                reconstruction_memories.extend(
                    memory_from_phase9_passage(item, default_type="procedure") for item in procedures
                )
            if evidence:
                reconstruction_memories.extend(
                    memory_from_phase9_passage(item, default_type="factual") for item in evidence
                )

            # The answerability gate remains authoritative. Reconstruction is only
            # allowed to transform evidence that survived retrieval/filtering/gating;
            # it cannot turn an unsupported request into a confident answer.
            if (
                external_answer is None
                and not used_fallback
                and args.memory_reconstruction_mode != "off"
                and reconstruction_memories
            ):
                current_state = row.get("current_state")
                if not isinstance(current_state, dict):
                    current_state = {}
                current_state = {"route": route.mode, **current_state}
                reconstruction_messages = build_reconstruction_messages(
                    question=question,
                    current_state=current_state,
                    memories=reconstruction_memories,
                    mode=(route.mode if route.mode in {"grounded_factual", "procedure", "social_roleplay"} else None),
                )
                if args.runtime_backend == "transformers":
                    assert model is not None
                    reconstruction_generated = generate_transformers(
                        model,
                        tokenizer,
                        reconstruction_messages,
                        max_sequence_length=args.max_sequence_length,
                        max_new_tokens=args.reconstruction_max_new_tokens,
                        response_control="eos_only",
                    )
                else:
                    assert args.llama_cli is not None
                    reconstruction_generated = generate_llama_cpp(
                        args.llama_cli,
                        Path(args.model),
                        tokenizer,
                        reconstruction_messages,
                        max_new_tokens=args.reconstruction_max_new_tokens,
                        response_control="eos_only",
                        extra_args=args.llama_cli_arg,
                    )
                parsed_reconstruction = parse_reconstruction_output(
                    str(reconstruction_generated.get("answer") or "")
                )
                reconstruction_packet = {
                    "mode": args.memory_reconstruction_mode,
                    "valid": parsed_reconstruction.valid,
                    "error": parsed_reconstruction.error,
                    "payload": parsed_reconstruction.payload,
                    "memory_ids": [item["memory_id"] for item in reconstruction_memories],
                }
                messages = build_answer_messages(
                    question=question,
                    reconstruction=parsed_reconstruction.payload,
                    mode=(route.mode if route.mode in {"grounded_factual", "procedure", "social_roleplay"} else None),
                    system_prompt=row.get("system_prompt"),
                )

            answer_tokenizer = tokenizer
            generation_tokenizer_source = tokenizer_source
            if external_answer is None:
                selected_model = model
                selected_tokenizer = tokenizer
                if used_fallback:
                    if args.runtime_backend != "transformers":
                        raise RuntimeError("Fallback model is supported only by the Transformers runtime")
                    if fallback_model is None:
                        fallback_model = load_transformers_model(args.fallback_model)
                        fallback_tokenizer = AutoTokenizer.from_pretrained(
                            args.fallback_model, trust_remote_code=True
                        )
                        fallback_tokenizer.pad_token = fallback_tokenizer.pad_token or fallback_tokenizer.eos_token
                    selected_model = fallback_model
                    assert fallback_tokenizer is not None
                    selected_tokenizer = fallback_tokenizer
                    answer_tokenizer = fallback_tokenizer
                    generation_tokenizer_source = args.fallback_model
                if args.runtime_backend == "transformers":
                    assert selected_model is not None
                    generated = generate_transformers(
                        selected_model,
                        selected_tokenizer,
                        messages,
                        max_sequence_length=args.max_sequence_length,
                        max_new_tokens=args.max_new_tokens,
                        response_control=args.response_control,
                    )
                else:
                    assert args.llama_cli is not None
                    generated = generate_llama_cpp(
                        args.llama_cli,
                        Path(args.model),
                        tokenizer,
                        messages,
                        max_new_tokens=args.max_new_tokens,
                        response_control=args.response_control,
                        extra_args=args.llama_cli_arg,
                    )
            else:
                visible_tokens = len(answer_tokenizer(external_answer, add_special_tokens=False)["input_ids"])
                generated = {
                    "answer": external_answer,
                    "raw_answer": external_answer,
                    "generated_tokens": 0,
                    "visible_tokens": visible_tokens,
                    "generation_seconds": 0.0,
                    "prefill_probe_seconds": 0.0,
                    "prompt_tokens": 0,
                    "stop_reason": "external_policy",
                    "response_control_trimmed": False,
                    "runtime_backend": args.runtime_backend,
                    "external_runtime_peak_rss_bytes": None,
                }

            answer = generated["answer"]
            grounding_passages = [*procedures, *evidence] if route.mode == "procedure" else evidence
            post_grounding = production_grounding_check(answer, grounding_passages)
            post_grounding_action = "returned"
            if route.mode in {"grounded_factual", "procedure"} and not is_abstention(answer) and not post_grounding.supported:
                answer = ABSTENTION_TEXT
                generated["answer"] = answer
                generated["visible_tokens"] = len(answer_tokenizer(answer, add_special_tokens=False)["input_ids"])
                post_grounding_action = "replaced_with_abstention"

            quality = evaluate_response(
                row_quality_record(row, messages),
                answer,
                max(1, int(generated["visible_tokens"])),
                visible_model_tokens=max(1, int(generated["visible_tokens"])),
                config=quality_config,
                corpus_profile=corpus_profile,
            )
            reference = reference_answer(row)
            answer_f1 = token_f1(reference, answer) if reference else None
            klass = decision_class(row)
            expected_answer = answer_expected(row)
            expected_conflict = conflict_expected(row)
            abstained = is_abstention(answer)
            supported_correct, functional_diagnostics = functional_answer_correct(
                row,
                answer,
                reference=reference,
                abstained=abstained,
                grounding_supported=(post_grounding.supported or abstained),
            )
            social_usable, social_diagnostics = social_response_usable(row, answer, quality)
            conflict_safe_response = is_safe_conflict_response(answer)
            expected_support = set(str(value) for value in (row.get("support_document_ids") or []))
            retrieved_ids = {passage_id(item) for item in [*procedures, *evidence]}
            result = {
                "id": record_id,
                "question": question,
                "route": route.mode,
                "mode": row.get("mode") or route.mode,
                "route_reason": route.reason,
                "decision": row.get("decision"),
                "category": row.get("category"),
                "case_class": klass,
                "answer": answer,
                "reference": reference,
                "answer_token_f1": answer_f1,
                "answer_expected": expected_answer,
                "conflict_expected": expected_conflict,
                "supported_answer_correct": supported_correct,
                "functional_correctness": functional_diagnostics,
                "abstained": abstained,
                "conflict_response": "conflict" in answer.casefold() or "disagree" in answer.casefold(),
                "conflict_safe_response": conflict_safe_response,
                "social_usable": social_usable,
                "social_diagnostics": social_diagnostics,
                "evidence": evidence,
                "procedures": procedures,
                "rejected_passages": rejected,
                "retrieved_document_ids": sorted(retrieved_ids),
                "support_document_ids": sorted(expected_support),
                "retrieval_applicable": bool(expected_support),
                "support_retrieved": bool(expected_support & retrieved_ids),
                "gate_reason": gate_reason,
                "used_fallback": used_fallback,
                "generation_tokenizer_source": generation_tokenizer_source,
                "grounding_supported": post_grounding.supported or is_abstention(answer),
                "grounding": {
                    "hard_failures": list(post_grounding.hard_failures),
                    "diagnostics": list(post_grounding.diagnostics),
                    "lexical_support_fraction": post_grounding.base.lexical_support_fraction,
                    "action": post_grounding_action,
                },
                "reconstruction": reconstruction_packet,
                "reconstruction_seconds": float((reconstruction_generated or {}).get("generation_seconds") or 0.0),
                "reconstruction_prompt_tokens": int((reconstruction_generated or {}).get("prompt_tokens") or 0),
                "reconstruction_generated_tokens": int((reconstruction_generated or {}).get("generated_tokens") or 0),
                "quality": quality,
                "retrieval_seconds": retrieval_seconds,
                "total_seconds": time.perf_counter() - started_total,
                **generated,
            }
            append_jsonl_durable(results_path, result)
            completed_rows[record_id] = result
            print(f"Benchmarked {record_id}: route={route.mode}, gate={gate_reason}, action={post_grounding_action}")

    ordered = [completed_rows[str(row.get("id"))] for row in benchmark]
    quality_summary = aggregate_quality_rows([row["quality"] for row in ordered])
    supported = [row for row in ordered if row["case_class"] == "supported"]
    unsupported = [row for row in ordered if row["case_class"] == "unsupported"]
    conflicts = [row for row in ordered if row["case_class"] == "conflict"]
    social = [row for row in ordered if row["case_class"] == "social"]
    procedures = [
        row for row in supported
        if str(row.get("mode") or "").casefold() == "procedure"
    ]
    calibration_cases = [
        row for row in ordered
        if str(row.get("category") or "").casefold() in {"qualified_answer", "ambiguous_evidence"}
    ]
    stale_cases = [row for row in ordered if str(row.get("category") or "").casefold() == "stale_evidence"]
    injection_cases = [row for row in ordered if "injection" in str(row.get("category") or "").casefold()]
    def task_success(item: dict[str, Any]) -> bool:
        if item["case_class"] == "supported": return bool(item["supported_answer_correct"])
        if item["case_class"] == "unsupported": return bool(item["abstained"])
        if item["case_class"] == "conflict": return bool(item["conflict_safe_response"])
        if item["case_class"] == "social": return bool(item["social_usable"])
        return False
    memory_packet = process_memory_packet()
    external_runtime_peak = max(
        (int(row.get("external_runtime_peak_rss_bytes") or 0) for row in ordered),
        default=0,
    )
    if external_runtime_peak > 0:
        memory_packet["external_runtime_peak_rss_bytes"] = external_runtime_peak
        memory_packet["total_process_memory_bytes"] = int(
            memory_packet.get("total_process_memory_bytes") or 0
        ) + external_runtime_peak
    else:
        memory_packet["external_runtime_peak_rss_bytes"] = None
    prefill_values = [
        float(row["prefill_probe_seconds"])
        for row in ordered
        if row["prefill_probe_seconds"] is not None
    ]
    summary = {
        **quality_summary,
        "phase": "11_production_rag",
        "model": args.model,
        "runtime_backend": args.runtime_backend,
        "tokenizer_source": tokenizer_source,
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256(args.benchmark),
        "case_count": len(ordered),
        "conditions": {
            "real_rag": {
                **quality_summary,
                "case_count": len(ordered),
                "supported_case_count": len(supported),
                "unsupported_case_count": len(unsupported),
                "conflict_case_count": len(conflicts),
                "social_case_count": len(social),
                "procedure_answerable_case_count": len(procedures),
                "supported_answer_correctness": safe_rate(sum(bool(row["supported_answer_correct"]) for row in supported), len(supported)),
                "evidence_faithfulness": safe_rate(
                    sum(bool(row["grounding_supported"]) for row in supported if not row["abstained"]),
                    sum(not bool(row["abstained"]) for row in supported),
                ),
                "correct_abstention_rate": safe_rate(sum(bool(row["abstained"]) for row in unsupported), len(unsupported)),
                "unnecessary_abstention_rate": safe_rate(sum(bool(row["abstained"]) for row in supported), len(supported)),
                "unsupported_confident_answer_rate": safe_rate(sum(not bool(row["abstained"]) for row in unsupported), len(unsupported)),
                "conflict_detection_rate": safe_rate(sum(bool(row["conflict_response"]) for row in conflicts), len(conflicts)),
                "conflict_safe_handling_rate": safe_rate(sum(bool(row["conflict_safe_response"]) for row in conflicts), len(conflicts)),
                "procedure_correctness": safe_rate(sum(bool(row["supported_answer_correct"]) for row in procedures), len(procedures)),
                "social_usable_rate": safe_rate(sum(bool(row["social_usable"]) for row in social), len(social)),
                "evidence_calibration_success_rate": safe_rate(sum(task_success(row) for row in calibration_cases), len(calibration_cases)),
                "stale_precedence_rate": safe_rate(sum(task_success(row) for row in stale_cases), len(stale_cases)),
                "prompt_injection_task_success_rate": safe_rate(sum(task_success(row) for row in injection_cases), len(injection_cases)),
                "retrieval_recall_at_k": safe_rate(
                    sum(bool(row["support_retrieved"]) for row in ordered if row["retrieval_applicable"]),
                    sum(bool(row["retrieval_applicable"]) for row in ordered),
                ),
                "metric_denominators": {
                    "supported_answer_correctness": len(supported),
                    "correct_abstention_rate": len(unsupported),
                    "conflict_safe_handling_rate": len(conflicts),
                    "procedure_correctness": len(procedures),
                    "social_usable_rate": len(social),
                    "evidence_calibration_success_rate": len(calibration_cases),
                    "stale_precedence_rate": len(stale_cases),
                    "prompt_injection_task_success_rate": len(injection_cases),
                },
                "median_completion_tokens": statistics.median([int(row["visible_tokens"]) for row in ordered]),
                "p90_completion_tokens": percentile([float(row["visible_tokens"]) for row in ordered], 0.9),
            }
        },
        "runtime": {
            "median_total_seconds": statistics.median([float(row["total_seconds"]) for row in ordered]),
            "median_reconstruction_seconds": statistics.median([float(row.get("reconstruction_seconds") or 0.0) for row in ordered]),
            "p90_reconstruction_seconds": percentile([float(row.get("reconstruction_seconds") or 0.0) for row in ordered], 0.9),
            "reconstruction_valid_rate": safe_rate(
                sum(bool((row.get("reconstruction") or {}).get("valid")) for row in ordered if row.get("reconstruction") is not None),
                sum(row.get("reconstruction") is not None for row in ordered),
            ),
            "p90_total_seconds": percentile([float(row["total_seconds"]) for row in ordered], 0.9),
            "median_retrieval_seconds": statistics.median([float(row["retrieval_seconds"]) for row in ordered]),
            "p90_retrieval_seconds": percentile([float(row["retrieval_seconds"]) for row in ordered], 0.9),
            "median_generation_seconds": statistics.median([float(row["generation_seconds"]) for row in ordered]),
            "p90_generation_seconds": percentile([float(row["generation_seconds"]) for row in ordered], 0.9),
            "median_prefill_probe_seconds": safe_median(prefill_values),
            "p90_prefill_probe_seconds": percentile(prefill_values, 0.9),
            "route_counts": dict(Counter(str(row["route"]) for row in ordered)),
            "gate_reason_counts": dict(Counter(str(row["gate_reason"]) for row in ordered)),
            "post_grounding_action_counts": dict(Counter(str(row["grounding"]["action"]) for row in ordered)),
            "fallback_count": sum(bool(row["used_fallback"]) for row in ordered),
        },
        "model_weight_bytes": (
            tree_bytes(Path(args.model)) if Path(args.model).is_dir() else Path(args.model).stat().st_size if Path(args.model).exists() else 0
        ),
        "knowledge_index_bytes": args.knowledge_db.stat().st_size,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        **memory_packet,
        "response_control": args.response_control,
        "memory_reconstruction_mode": args.memory_reconstruction_mode,
        "reconstruction_max_new_tokens": args.reconstruction_max_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "maximum_evidence_tokens": args.maximum_evidence_tokens,
        "answerability_calibration": str(args.answerability_calibration),
        "measurement_note": (
            "prefill_probe_seconds is a separate diagnostic forward performed before end-to-end generation; it intentionally duplicates prompt work. "
            "total_seconds measures the actual retrieval/gate/generation/grounding request path."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    report = [
        "# Phase 11 production RAG runtime benchmark",
        "",
        f"- Model: `{args.model}`",
        f"- Runtime backend: `{args.runtime_backend}`",
        f"- Cases: `{len(ordered)}`",
        f"- Model bytes: `{summary['model_weight_bytes']}`",
        f"- Index bytes: `{summary['knowledge_index_bytes']}`",
        f"- Process RSS bytes: `{summary['process_rss_bytes']}`",
        f"- Peak process RSS bytes: `{summary['peak_process_rss_bytes']}`",
        f"- Total process memory bytes: `{summary['total_process_memory_bytes']}`",
        f"- External runtime peak RSS bytes: `{summary['external_runtime_peak_rss_bytes']}`",
        f"- Peak allocated VRAM bytes: `{summary['peak_allocated_vram_bytes']}`",
        f"- Median/P90 total seconds: `{summary['runtime']['median_total_seconds']}` / `{summary['runtime']['p90_total_seconds']}`",
        f"- Supported correctness: `{summary['conditions']['real_rag']['supported_answer_correctness']}`",
        f"- Evidence faithfulness: `{summary['conditions']['real_rag']['evidence_faithfulness']}`",
        f"- Correct abstention: `{summary['conditions']['real_rag']['correct_abstention_rate']}`",
        f"- Unsupported confident answer rate: `{summary['conditions']['real_rag']['unsupported_confident_answer_rate']}`",
        f"- Conflict safe handling: `{summary['conditions']['real_rag']['conflict_safe_handling_rate']}`",
        f"- Procedure correctness: `{summary['conditions']['real_rag']['procedure_correctness']}`",
        f"- Evidence calibration: `{summary['conditions']['real_rag']['evidence_calibration_success_rate']}`",
        f"- Stale/current precedence: `{summary['conditions']['real_rag']['stale_precedence_rate']}`",
        "",
        summary["measurement_note"],
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
