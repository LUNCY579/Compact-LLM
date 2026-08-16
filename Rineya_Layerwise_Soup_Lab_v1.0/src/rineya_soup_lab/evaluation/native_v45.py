from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from ..errors import EvaluationError
from ..utils import append_jsonl, read_json, sha256_file, utc_timestamp, write_json
from ..vendor.official_v45.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
    normalize_text,
    token_f1,
)
from ..vendor.official_v45.response_control import (
    build_stopping_criteria,
    finalize_response,
    infer_stop_reason,
)
from .dataset import load_dataset


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return float(ordered[index])


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _render_chat(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _extract_final_answer(
    tokenizer: Any,
    new_tokens: torch.Tensor,
    *,
    extract_final_after_thinking: bool,
) -> tuple[str, str, int]:
    raw_with_special = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    ordinary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not extract_final_after_thinking:
        return ordinary, raw_with_special, 0
    markers = ("</think>", "<|end_think|>", "[/THINK]")
    final = ordinary
    found = None
    for marker in markers:
        if marker in raw_with_special:
            found = marker
            final = raw_with_special.rsplit(marker, 1)[1]
            break
    if found is not None:
        # Remove remaining known special tokens without deleting ordinary text.
        for token in getattr(tokenizer, "all_special_tokens", []):
            final = final.replace(str(token), "")
        final = final.strip()
    visible_ids = tokenizer(final, add_special_tokens=False)["input_ids"] if final else []
    reasoning_tokens = max(0, int(new_tokens.numel()) - len(visible_ids))
    return final, raw_with_special, reasoning_tokens


def _token_probability_metrics(
    scores: tuple[torch.Tensor, ...] | list[torch.Tensor],
    generated_ids: list[int],
    eos_token_id: int | list[int] | None,
) -> dict[str, Any]:
    if not scores:
        return {}
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id or [])
    entropies: list[float] = []
    selected_logprobs: list[float] = []
    eos_probs: list[float] = []
    margins: list[float] = []
    for step, logits_batch in enumerate(scores):
        logits = logits_batch[0].detach().float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropies.append(float(-(probs * log_probs).sum().item()))
        if step < len(generated_ids):
            selected_logprobs.append(float(log_probs[int(generated_ids[step])].item()))
        if eos_ids:
            eos_probs.append(float(probs[eos_ids].sum().item()))
        top2 = torch.topk(probs, k=min(2, probs.numel())).values
        margins.append(float(top2[0].item() - (top2[1].item() if top2.numel() > 1 else 0.0)))
    return {
        "mean_token_entropy": statistics.mean(entropies),
        "mean_selected_token_logprob": statistics.mean(selected_logprobs) if selected_logprobs else None,
        "mean_eos_probability": statistics.mean(eos_probs) if eos_probs else None,
        "eos_probability_at_last_step": eos_probs[-1] if eos_probs else None,
        "mean_top1_probability_margin": statistics.mean(margins),
    }


def _evaluation_manifest(
    model_path: Path,
    tokenizer_path: Path,
    dataset_path: Path,
    quality_config_path: Path,
    protocol_name: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "created_at": utc_timestamp(),
        "model_path": str(model_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "quality_config_path": str(quality_config_path.resolve()),
        "quality_config_sha256": sha256_file(quality_config_path),
        "protocol_name": protocol_name,
        "protocol": dict(protocol),
    }


def _manifest_compatible(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    fields = (
        "quality_evaluation_version",
        "model_path",
        "tokenizer_path",
        "dataset_sha256",
        "quality_config_sha256",
        "protocol_name",
        "protocol",
    )
    return all(existing.get(field) == expected.get(field) for field in fields)


def evaluate_model_bundled_v45(
    *,
    model_path: Path,
    tokenizer_path: Path,
    dataset_path: Path,
    output_dir: Path,
    quality_config_path: Path,
    protocol_name: str,
    protocol: Mapping[str, Any],
    trust_remote_code: bool = False,
    resume: bool = True,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate replies and score them with the bundled official v4.5 rules."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EvaluationError("Install transformers before running model evaluation.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    expected_manifest = _evaluation_manifest(
        model_path, tokenizer_path, dataset_path, quality_config_path, protocol_name, protocol
    )
    results_path = output_dir / "results.jsonl"
    if resume and manifest_path.exists() and results_path.exists():
        existing = read_json(manifest_path)
        if not _manifest_compatible(existing, expected_manifest):
            raise EvaluationError(
                f"Existing evaluation evidence under {output_dir} does not match this model/dataset/protocol."
            )
        existing_rows = load_existing_rows(results_path)
    elif results_path.exists():
        raise EvaluationError(
            f"Existing results without a reusable matching manifest: {output_dir}. Use a new run directory."
        )
    else:
        existing_rows = {}
        write_json(manifest_path, expected_manifest)

    quality_config = load_quality_config(quality_config_path)
    records = load_dataset(dataset_path, limit=limit)
    corpus_profile = build_corpus_profile(records)
    record_ids = [str(row["id"]) for row in records]
    unknown = sorted(set(existing_rows) - set(record_ids))
    if unknown:
        raise EvaluationError(f"Existing result IDs are absent from the dataset: {unknown[:10]}")
    if len(existing_rows) == len(records):
        outputs = [existing_rows[value] for value in record_ids]
        summary = build_summary(outputs, protocol_name, protocol, quality_config_path)
        finalize_evidence(output_dir, outputs, summary)
        return summary, outputs

    if protocol.get("require_cuda") and not torch.cuda.is_available():
        raise EvaluationError("CUDA is unavailable but this protocol requires CUDA.")
    seed = int(protocol.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        eos_value = tokenizer.eos_token_id
        if isinstance(eos_value, list):
            pad_token_id = eos_value[0] if eos_value else None
        else:
            pad_token_id = eos_value
    if pad_token_id is None:
        raise EvaluationError("Tokenizer defines neither pad_token_id nor eos_token_id.")

    outputs_by_id = dict(existing_rows)
    try:
        for index, record in enumerate(records, 1):
            stable_id = str(record["id"])
            if stable_id in outputs_by_id:
                continue
            messages = [dict(item) for item in record["messages"]]
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": str(protocol.get("system_prompt", ""))}, *messages]
            rendered = _render_chat(tokenizer, messages, bool(protocol.get("enable_thinking", False)))
            inputs = tokenizer(rendered, return_tensors="pt").to(input_device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _sync_cuda()
            start = time.perf_counter()
            stopping_criteria, boundary_criterion = build_stopping_criteria(
                tokenizer,
                int(inputs.input_ids.shape[1]),
                str(protocol.get("response_control", "eos_only")),
                int(protocol.get("min_new_tokens_before_boundary", 1)),
            )
            capture = bool(protocol.get("capture_token_probabilities", False))
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(protocol.get("max_new_tokens", 32)),
                "do_sample": bool(protocol.get("do_sample", False)),
                "pad_token_id": pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": capture,
                "output_scores": capture,
            }
            if stopping_criteria is not None:
                generation_kwargs["stopping_criteria"] = stopping_criteria
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = float(protocol.get("temperature", 0.6))
                generation_kwargs["top_p"] = float(protocol.get("top_p", 0.9))
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            _sync_cuda()
            elapsed = time.perf_counter() - start
            if capture:
                sequences = generated.sequences
                scores = generated.scores
            else:
                sequences = generated
                scores = ()
            new_tokens = sequences[0, inputs.input_ids.shape[1] :]
            extracted, raw_with_special, reasoning_tokens = _extract_final_answer(
                tokenizer,
                new_tokens,
                extract_final_after_thinking=bool(protocol.get("extract_final_after_thinking", False)),
            )
            finalized = finalize_response(extracted, str(protocol.get("response_control", "eos_only")))
            reply = finalized.text
            visible_token_count = len(tokenizer(reply, add_special_tokens=False)["input_ids"])
            stop_reason = infer_stop_reason(
                new_tokens.tolist(),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=int(protocol.get("max_new_tokens", 32)),
                criterion=boundary_criterion,
            )
            quality = evaluate_response(
                record,
                reply,
                visible_token_count,
                visible_model_tokens=visible_token_count,
                config=quality_config,
                corpus_profile=corpus_profile,
            )
            probability_metrics = _token_probability_metrics(
                scores, new_tokens.tolist(), tokenizer.eos_token_id
            ) if capture else {}
            row = {
                "id": stable_id,
                "category": record["category"],
                "messages": record["messages"],
                "reference": record.get("reference"),
                "reference_token_f1": token_f1(record.get("reference"), reply),
                "reply": reply,
                "raw_reply": finalized.raw_text,
                "raw_with_special_tokens": raw_with_special,
                "thinking_enabled": bool(protocol.get("enable_thinking", False)),
                "reasoning_tokens_estimate": reasoning_tokens,
                "response_control_mode": protocol.get("response_control", "eos_only"),
                "response_control_trimmed": finalized.trimmed,
                "stop_reason": stop_reason,
                "sentence_count": finalized.sentence_count,
                "one_sentence_violation": finalized.sentence_count > 1,
                "prompt_tokens": int(inputs.input_ids.shape[1]),
                "generated_tokens": int(new_tokens.shape[0]),
                "visible_generated_tokens": visible_token_count,
                "elapsed_seconds": elapsed,
                "peak_vram_gb": (
                    torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                ),
                **probability_metrics,
                **quality,
            }
            outputs_by_id[stable_id] = row
            append_jsonl(results_path, row)
            print(
                f"[{index:03d}/{len(records):03d}] {stable_id}: {reply!r} "
                f"quality={float(row['quality_score']):.3f}",
                flush=True,
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outputs = [outputs_by_id[value] for value in record_ids]
    summary = build_summary(outputs, protocol_name, protocol, quality_config_path)
    finalize_evidence(output_dir, outputs, summary)
    return summary, outputs


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    import json

    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                remaining = handle.read()
                if remaining.strip():
                    raise EvaluationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                break
            stable_id = str(row["id"])
            if stable_id in rows:
                raise EvaluationError(f"Duplicate result ID {stable_id} in {path}")
            rows[stable_id] = row
    return rows


def build_summary(
    outputs: list[dict[str, Any]],
    protocol_name: str,
    protocol: Mapping[str, Any],
    quality_config_path: Path,
) -> dict[str, Any]:
    if not outputs:
        raise EvaluationError("Cannot summarize an empty evaluation.")
    token_counts = [float(row["visible_generated_tokens"]) for row in outputs]
    generated_counts = [float(row["generated_tokens"]) for row in outputs]
    times = [float(row["elapsed_seconds"]) for row in outputs]
    repeats = Counter(normalize_text(str(row["reply"])) for row in outputs)
    stop_reasons = Counter(str(row["stop_reason"]) for row in outputs)
    summary = {
        "case_count": len(outputs),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "quality_config": str(quality_config_path),
        "quality_config_sha256": sha256_file(quality_config_path),
        "protocol_name": protocol_name,
        "response_control_mode": protocol.get("response_control"),
        "enable_thinking": bool(protocol.get("enable_thinking", False)),
        "max_new_tokens": int(protocol.get("max_new_tokens", 32)),
        "median_reply_tokens": statistics.median(token_counts),
        "p90_reply_tokens": _percentile(token_counts, 0.90),
        "median_actual_generated_tokens": statistics.median(generated_counts),
        "p90_actual_generated_tokens": _percentile(generated_counts, 0.90),
        "mean_reasoning_tokens_estimate": statistics.mean(
            float(row.get("reasoning_tokens_estimate", 0.0)) for row in outputs
        ),
        "one_sentence_violation_rate": statistics.mean(
            1.0 if row.get("one_sentence_violation") else 0.0 for row in outputs
        ),
        "response_control_trim_rate": statistics.mean(
            1.0 if row.get("response_control_trimmed") else 0.0 for row in outputs
        ),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "mean_generation_seconds": statistics.mean(times),
        "p90_generation_seconds": _percentile(times, 0.90),
        "peak_vram_gb": max(float(row.get("peak_vram_gb", 0.0)) for row in outputs),
        "duplicate_reply_count": sum(count - 1 for count in repeats.values() if count > 1),
        "duplicate_reply_rate": sum(count - 1 for count in repeats.values() if count > 1) / len(outputs),
        **aggregate_quality_rows(outputs),
    }
    for key in (
        "mean_token_entropy",
        "mean_selected_token_logprob",
        "mean_eos_probability",
        "eos_probability_at_last_step",
        "mean_top1_probability_margin",
    ):
        values = [float(row[key]) for row in outputs if row.get(key) is not None]
        if values:
            summary[key] = statistics.mean(values)
    return summary


def finalize_evidence(
    output_dir: Path,
    outputs: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    # Progress rows are already appended to results.jsonl. Rewriting that file
    # on Windows is what antivirus/indexers lock; do not replace it here.
    write_json(output_dir / "summary.json", dict(summary))
    lines = [
        "# BUNDLED QUALITY EVALUATOR v4.5 RESULT",
        "",
        f"Cases: **{summary['case_count']}**",
        f"Protocol: `{summary.get('protocol_name')}`",
        f"Thinking enabled: **{summary.get('enable_thinking')}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "mean_quality_score",
        "worst_decile_mean_quality_score",
        "complete_utterance_rate",
        "observable_quality_pass_rate",
        "median_reply_tokens",
        "p90_reply_tokens",
        "duplicate_reply_rate",
        "mean_generation_seconds",
        "peak_vram_gb",
    ):
        if key in summary:
            lines.append(f"| {key} | {summary[key]} |")
    lines.extend(
        [
            "",
            "The evaluator reports observable defects and explicit contract fulfillment under v4.5. "
            "Candidate selection is performed separately by the development objective and paired uncertainty analysis.",
            "",
        ]
    )
    (output_dir / "MODEL_EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
