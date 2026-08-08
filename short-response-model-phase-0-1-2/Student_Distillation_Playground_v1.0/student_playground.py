from __future__ import annotations

"""Exploratory playground for Phase 9/9B student models.

This tool is intentionally outside the official phase directories. It reuses the
repository's official prompt builders and generation path, but its results are
presentation/research diagnostics only and MUST NOT be used as official phase
gate evidence unless the cases are separately promoted into a governed dev/test
benchmark with leakage controls.
"""

import argparse
import gc
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABSTENTION_TEXT = "I don't know based on the available information."


def _is_abstention(answer: str) -> bool:
    return _normalize(answer) == _normalize(ABSTENTION_TEXT)


def _is_safe_conflict_response(answer: str) -> bool:
    low = _normalize(answer)
    return _is_abstention(answer) or any(token in low for token in ("conflict", "disagree", "cannot determine", "can't determine", "cannot give one", "can't give one"))


try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - handled at runtime
    PeftModel = None  # type: ignore[assignment]

PLAYGROUND_VERSION = "1.0"
PACKAGE_VERSION = "1.7.6"
SUPPORTED_TASK_TYPES = {"logic", "evidence", "memory"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("id") or "").strip()
            if not case_id:
                raise ValueError(f"{path}:{line_no}: missing id")
            if case_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {case_id}")
            seen.add(case_id)
            task_type = str(row.get("task_type") or "").strip()
            if task_type not in SUPPORTED_TASK_TYPES:
                raise ValueError(f"{path}:{line_no}: unsupported task_type={task_type!r}")
            if not str(row.get("question") or "").strip():
                raise ValueError(f"{path}:{line_no}: missing question")
            if not isinstance(row.get("expected"), dict):
                raise ValueError(f"{path}:{line_no}: expected must be an object")
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _normalize(value: str) -> str:
    value = value.casefold().strip()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9.+:/_-]+", " ", value)
    return " ".join(value.split())


def _phrase_present(answer: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(answer)


def score_final_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    expected = case["expected"]
    decision = str(expected.get("decision") or "answer").casefold()
    required = [str(x) for x in expected.get("required_phrases") or []]
    forbidden = [str(x) for x in expected.get("forbidden_phrases") or []]
    aliases = [str(x) for x in expected.get("aliases") or []]
    canonical = str(expected.get("answer") or "")

    reasons: list[str] = []
    if decision == "abstain":
        correct = _is_abstention(answer)
        if not correct:
            reasons.append("expected_abstention")
    elif decision == "conflict":
        correct = _is_safe_conflict_response(answer)
        if not correct:
            reasons.append("expected_safe_conflict_handling")
    else:
        required_ok = all(_phrase_present(answer, phrase) for phrase in required)
        forbidden_ok = not any(_phrase_present(answer, phrase) for phrase in forbidden)
        surface_candidates = [canonical, *aliases]
        surface_candidates = [x for x in surface_candidates if x.strip()]
        surface_ok = True
        if surface_candidates and not required:
            normalized_answer = _normalize(answer)
            surface_ok = any(
                normalized_answer == _normalize(candidate)
                or _normalize(candidate) in normalized_answer
                for candidate in surface_candidates
            )
        correct = required_ok and forbidden_ok and surface_ok
        if not required_ok:
            reasons.append("missing_required_phrase")
        if not forbidden_ok:
            reasons.append("contains_forbidden_phrase")
        if not surface_ok:
            reasons.append("answer_mismatch")

    return {
        "correct": bool(correct),
        "decision_expected": decision,
        "required_phrases": required,
        "forbidden_phrases": forbidden,
        "reasons": reasons,
    }


def _logic_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    context = str(case.get("context") or "").strip()
    user = str(case["question"])
    if context:
        user = f"<CONTEXT>\n{context}\n</CONTEXT>\n\n<QUESTION>\n{user}\n</QUESTION>"
    system = (
        "Solve the small logic problem accurately. Return only the concise final answer, "
        "not a chain-of-thought explanation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _evidence_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    evidence = case.get("evidence") or []
    procedures = case.get("procedures") or []
    from scripts.phase_9_student_distillation.phase9_common import build_grounded_messages
    return build_grounded_messages(
        question=str(case["question"]),
        evidence=evidence,
        procedures=procedures,
        mode=str(case.get("mode") or "grounded_factual"),
    )


def _memory_direct_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    from scripts.phase_9b_memory_reconstruction.phase9b_common import build_direct_memory_messages
    return build_direct_memory_messages(
        question=str(case["question"]),
        current_state=dict(case.get("current_state") or {}),
        memories=list(case.get("memories") or []),
        mode=str(case.get("mode") or "grounded_factual"),
    )


def _load_model(model_path: str, adapter_path: Path | None, require_cuda: bool) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    if adapter_path is not None and PeftModel is None:
        raise RuntimeError("PEFT is required when --adapter is supplied")
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


def _parameter_count(model: Any) -> int:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return 0


def _generation_metrics(generation: dict[str, Any]) -> dict[str, Any]:
    seconds = float(generation.get("generation_seconds") or 0.0)
    tokens = int(generation.get("generated_tokens") or 0)
    return {
        "generation_seconds": seconds,
        "generated_tokens": tokens,
        "visible_tokens": int(generation.get("visible_tokens") or 0),
        "prompt_tokens": int(generation.get("prompt_tokens") or 0),
        "tokens_per_second": (tokens / seconds) if seconds > 0 else None,
        "stop_reason": generation.get("stop_reason"),
        "decode_integrity": generation.get("decode_integrity"),
    }


def _run_case(
    *,
    case: dict[str, Any],
    model: Any,
    tokenizer: Any,
    max_sequence_length: int,
    max_new_tokens: int,
    reconstruction_max_new_tokens: int,
    memory_mode: str,
) -> dict[str, Any]:
    from scripts.phase_9_student_distillation.evaluate_rag import generate_answer
    from scripts.phase_9_student_distillation.phase9_common import grounding_check
    from scripts.phase_9b_memory_reconstruction.phase9b_common import (
        build_answer_messages,
        build_reconstruction_messages,
        decision_accuracy,
        parse_reconstruction_output,
        source_id_precision,
        source_id_recall,
    )

    task_type = case["task_type"]
    reconstruction: dict[str, Any] | None = None
    reconstruction_generation: dict[str, Any] | None = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_started = time.perf_counter()
    if task_type == "logic":
        messages = _logic_messages(case)
        generation = generate_answer(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            response_control="eos_only",
        )
    elif task_type == "evidence":
        messages = _evidence_messages(case)
        generation = generate_answer(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            response_control="eos_only",
        )
    elif task_type == "memory" and memory_mode == "direct":
        messages = _memory_direct_messages(case)
        generation = generate_answer(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            response_control="eos_only",
        )
    elif task_type == "memory":
        reconstruction_messages = build_reconstruction_messages(
            question=str(case["question"]),
            current_state=dict(case.get("current_state") or {}),
            memories=list(case.get("memories") or []),
            mode=str(case.get("mode") or "grounded_factual"),
        )
        reconstruction_generation = generate_answer(
            model=model,
            tokenizer=tokenizer,
            messages=reconstruction_messages,
            max_sequence_length=max_sequence_length,
            max_new_tokens=reconstruction_max_new_tokens,
            response_control="eos_only",
        )
        parsed = parse_reconstruction_output(str(reconstruction_generation["answer"]))
        reconstruction = {
            "payload": parsed.payload,
            "valid": parsed.valid,
            "error": parsed.error,
            "raw_text": parsed.raw_text,
        }
        answer_messages = build_answer_messages(
            question=str(case["question"]),
            reconstruction=parsed.payload,
            mode=str(case.get("mode") or "grounded_factual"),
        )
        generation = generate_answer(
            model=model,
            tokenizer=tokenizer,
            messages=answer_messages,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            response_control="eos_only",
        )
    else:  # pragma: no cover
        raise AssertionError(task_type)

    total_seconds = time.perf_counter() - total_started
    answer = str(generation["answer"])
    score = score_final_answer(case, answer)

    diagnostics: dict[str, Any] = {}
    if task_type == "evidence":
        passages = [*list(case.get("evidence") or []), *list(case.get("procedures") or [])]
        grounding = grounding_check(answer, passages)
        diagnostics["grounding_supported"] = bool(grounding.supported)
        diagnostics["grounding_reason"] = grounding.reason
    elif task_type == "memory" and reconstruction is not None:
        expected_recon = (case.get("expected") or {}).get("reconstruction") or {}
        predicted_recon = reconstruction["payload"]
        diagnostics["reconstruction_valid_json"] = bool(reconstruction["valid"])
        diagnostics["reconstruction_decision_correct"] = decision_accuracy(expected_recon, predicted_recon)
        diagnostics["source_id_precision"] = source_id_precision(expected_recon, predicted_recon)
        diagnostics["source_id_recall"] = source_id_recall(expected_recon, predicted_recon)
        diagnostics["predicted_reconstruction"] = predicted_recon
        diagnostics["reconstruction_error"] = reconstruction["error"]

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = int(torch.cuda.max_memory_allocated())

    result = {
        "id": case["id"],
        "task_type": task_type,
        "difficulty": case.get("difficulty", "unknown"),
        "question": case["question"],
        "answer": answer,
        "expected": case["expected"],
        "correct": score["correct"],
        "score": score,
        "diagnostics": diagnostics,
        "timing": {**_generation_metrics(generation), "total_case_seconds": total_seconds},
        "peak_vram_bytes": peak_vram,
    }
    if reconstruction_generation is not None:
        result["reconstruction_timing"] = _generation_metrics(reconstruction_generation)
        result["reconstruction_raw"] = reconstruction_generation.get("answer")
    return result


def _compact_expected(case: dict[str, Any]) -> str:
    expected = case["expected"]
    decision = str(expected.get("decision") or "answer")
    if decision == "abstain":
        return ABSTENTION_TEXT
    if decision == "conflict":
        return "safe conflict/abstention"
    return str(expected.get("answer") or ", ".join(expected.get("required_phrases") or []))


def _render_context(case: dict[str, Any]) -> str:
    task_type = case["task_type"]
    if task_type == "logic":
        return str(case.get("context") or "(no extra context)")
    if task_type == "evidence":
        chunks = []
        for item in [*list(case.get("procedures") or []), *list(case.get("evidence") or [])]:
            chunks.append(str(item.get("text") or item))
        return "\n".join(chunks) or "(no evidence)"
    state = json.dumps(case.get("current_state") or {}, ensure_ascii=False, sort_keys=True)
    memories = "\n".join(f"- {m.get('text')}" for m in case.get("memories") or [])
    return f"Current state: {state}\nMemories:\n{memories or '(none)'}"


def _print_video_case(index: int, total: int, case: dict[str, Any], result: dict[str, Any], show_context: bool) -> None:
    width = 78
    status = "PASS" if result["correct"] else "MISS"
    print("\n" + "=" * width)
    print(f"STUDENT PLAYGROUND  |  Case {index}/{total}  |  {case['task_type'].upper()}  |  {status}")
    print("=" * width)
    if show_context:
        print("CONTEXT / EVIDENCE")
        print(_render_context(case))
        print("-" * width)
    print("QUESTION")
    print(case["question"])
    print("-" * width)
    print("STUDENT")
    print(result["answer"] or "(empty)")
    print("-" * width)
    print(f"EXPECTED: {_compact_expected(case)}")
    t = result["timing"]
    tps = t.get("tokens_per_second")
    tps_text = f"{tps:.1f} tok/s" if isinstance(tps, (int, float)) else "n/a tok/s"
    vram = result.get("peak_vram_bytes")
    vram_text = f"{vram / (1024**3):.2f} GiB" if isinstance(vram, int) else "n/a"
    print(f"LATENCY: {t['total_case_seconds']:.3f}s | TOKENS: {t['generated_tokens']} | {tps_text} | PEAK VRAM: {vram_text}")
    if case["task_type"] == "memory" and "reconstruction_decision_correct" in result["diagnostics"]:
        recon = result["diagnostics"]
        decision = (recon.get("predicted_reconstruction") or {}).get("decision")
        print(f"MEMORY DECISION: {decision} | correct={recon['reconstruction_decision_correct']}")


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(rows: Iterable[dict[str, Any]]) -> float | None:
        rows = list(rows)
        return (sum(bool(r["correct"]) for r in rows) / len(rows)) if rows else None

    by_type: dict[str, Any] = {}
    for task_type in sorted({r["task_type"] for r in results}):
        rows = [r for r in results if r["task_type"] == task_type]
        by_type[task_type] = {"cases": len(rows), "accuracy": rate(rows)}
    by_difficulty: dict[str, Any] = {}
    for difficulty in sorted({str(r.get("difficulty")) for r in results}):
        rows = [r for r in results if str(r.get("difficulty")) == difficulty]
        by_difficulty[difficulty] = {"cases": len(rows), "accuracy": rate(rows)}

    latencies = [float(r["timing"]["total_case_seconds"]) for r in results]
    gen_latencies = [float(r["timing"]["generation_seconds"]) for r in results]
    token_rates = [float(r["timing"]["tokens_per_second"]) for r in results if r["timing"].get("tokens_per_second") is not None]
    peak_vrams = [int(r["peak_vram_bytes"]) for r in results if isinstance(r.get("peak_vram_bytes"), int)]

    memory_two_pass = [r for r in results if "reconstruction_decision_correct" in r["diagnostics"]]
    evidence = [r for r in results if r["task_type"] == "evidence"]
    abstain_rows = [r for r in results if str(r["expected"].get("decision")) == "abstain"]
    conflict_rows = [r for r in results if str(r["expected"].get("decision")) == "conflict"]

    return {
        "schema_version": "1.0",
        "playground_version": PLAYGROUND_VERSION,
        "package_version": PACKAGE_VERSION,
        "official_gate_evidence": False,
        "case_count": len(results),
        "correct_cases": sum(bool(r["correct"]) for r in results),
        "accuracy": rate(results),
        "by_task_type": by_type,
        "by_difficulty": by_difficulty,
        "abstention_accuracy": rate(abstain_rows),
        "conflict_accuracy": rate(conflict_rows),
        "evidence_grounding_supported_rate": (
            sum(bool(r["diagnostics"].get("grounding_supported")) for r in evidence) / len(evidence)
            if evidence else None
        ),
        "memory_reconstruction_decision_accuracy": (
            sum(bool(r["diagnostics"].get("reconstruction_decision_correct")) for r in memory_two_pass) / len(memory_two_pass)
            if memory_two_pass else None
        ),
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p90": _percentile(latencies, 0.90),
            "generation_mean": statistics.mean(gen_latencies) if gen_latencies else None,
        },
        "throughput_tokens_per_second": {
            "mean": statistics.mean(token_rates) if token_rates else None,
            "median": statistics.median(token_rates) if token_rates else None,
        },
        "peak_vram_bytes_max": max(peak_vrams) if peak_vrams else None,
    }


def _write_report(path: Path, summary: dict[str, Any], results: list[dict[str, Any]], dataset: Path, model: str, adapter: Path | None, memory_mode: str) -> None:
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{100 * float(value):.1f}%"

    lines = [
        "# Student Distillation Playground Report",
        "",
        f"- Package: v{PACKAGE_VERSION}",
        f"- Playground: v{PLAYGROUND_VERSION}",
        f"- Dataset: `{dataset}`",
        f"- Model: `{model}`",
        f"- Adapter: `{adapter}`" if adapter else "- Adapter: none",
        f"- Memory mode: `{memory_mode}`",
        "- Status: exploratory only; not official phase-gate evidence",
        "",
        "## Summary",
        "",
        f"- Accuracy: **{pct(summary['accuracy'])}** ({summary['correct_cases']}/{summary['case_count']})",
        f"- Abstention accuracy: {pct(summary['abstention_accuracy'])}",
        f"- Conflict handling accuracy: {pct(summary['conflict_accuracy'])}",
        f"- Evidence grounding diagnostic: {pct(summary['evidence_grounding_supported_rate'])}",
        f"- Memory reconstruction decision accuracy: {pct(summary['memory_reconstruction_decision_accuracy'])}",
        f"- Median total case latency: {summary['latency_seconds']['median']:.3f}s" if summary['latency_seconds']['median'] is not None else "- Median total case latency: n/a",
        f"- P90 total case latency: {summary['latency_seconds']['p90']:.3f}s" if summary['latency_seconds']['p90'] is not None else "- P90 total case latency: n/a",
        "",
        "## By task type",
        "",
    ]
    for name, item in summary["by_task_type"].items():
        lines.append(f"- {name}: {pct(item['accuracy'])} ({item['cases']} cases)")
    lines.extend(["", "## Cases", ""])
    for row in results:
        status = "PASS" if row["correct"] else "MISS"
        lines.append(f"### {row['id']} — {status}")
        lines.append("")
        lines.append(f"- Type: {row['task_type']} / {row.get('difficulty')}")
        lines.append(f"- Question: {row['question']}")
        lines.append(f"- Student: {row['answer']}")
        lines.append(f"- Expected: {_compact_expected({'expected': row['expected']})}")
        lines.append(f"- Total latency: {row['timing']['total_case_seconds']:.3f}s")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run exploratory logic/evidence/memory tests against a distilled student.")
    p.add_argument("--model", required=True, help="Base/full Hugging Face model path")
    p.add_argument("--adapter", type=Path, default=None, help="Optional PEFT adapter path")
    p.add_argument("--dataset", type=Path, required=True, help="Playground JSONL dataset")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-type", choices=("all", "logic", "evidence", "memory"), default="all")
    p.add_argument("--case-id", action="append", default=[], help="Run only a case id; repeatable")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--memory-mode", choices=("direct", "two_pass"), default="direct")
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--reconstruction-max-new-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1, help="Unscored warmup generations using the first selected case")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--video-mode", action="store_true", help="Print a clean case-by-case presentation view")
    p.add_argument("--show-context", action="store_true", help="Show evidence/state in video mode")
    p.add_argument("--pause-seconds", type=float, default=0.0, help="Optional pause after each displayed case")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _read_jsonl(dataset)
    if args.task_type != "all":
        cases = [row for row in cases if row["task_type"] == args.task_type]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [row for row in cases if row["id"] in wanted]
        missing = wanted - {row["id"] for row in cases}
        if missing:
            raise ValueError(f"Unknown case ids: {sorted(missing)}")
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        raise ValueError("No cases selected")

    model, tokenizer = _load_model(args.model, args.adapter, args.require_cuda)
    model_params = _parameter_count(model)

    # Warmup is intentionally unscored so first-kernel initialization does not
    # dominate the presentation latency distribution.
    for _ in range(max(0, args.warmup)):
        _run_case(
            case=cases[0], model=model, tokenizer=tokenizer,
            max_sequence_length=args.max_sequence_length,
            max_new_tokens=args.max_new_tokens,
            reconstruction_max_new_tokens=args.reconstruction_max_new_tokens,
            memory_mode=args.memory_mode,
        )

    results: list[dict[str, Any]] = []
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            result = _run_case(
                case=case, model=model, tokenizer=tokenizer,
                max_sequence_length=args.max_sequence_length,
                max_new_tokens=args.max_new_tokens,
                reconstruction_max_new_tokens=args.reconstruction_max_new_tokens,
                memory_mode=args.memory_mode,
            )
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if args.video_mode:
                _print_video_case(index, len(cases), case, result, args.show_context)
                if args.pause_seconds > 0:
                    time.sleep(args.pause_seconds)

    summary = _summarize(results)
    summary["run"] = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(dataset),
        "dataset_sha256": _sha256_file(dataset),
        "memory_mode": args.memory_mode,
        "model_parameters_loaded": model_params,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(output_dir / "REPORT.md", summary, results, dataset, args.model, args.adapter, args.memory_mode)

    print("\nPlayground complete")
    print(f"Accuracy: {summary['correct_cases']}/{summary['case_count']} ({100 * summary['accuracy']:.1f}%)")
    print(f"Results: {results_path}")
    print(f"Report:  {output_dir / 'REPORT.md'}")
    print("Exploratory only: this run does not alter or satisfy any official phase gate.")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
