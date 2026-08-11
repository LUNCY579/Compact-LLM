from __future__ import annotations

"""Evaluate one v2.3.1 checkpoint on train/DEV/OOD and continuous logits.

This is deliberately a non-sealed evaluator.  It loads the greedy anchor once,
optionally attaches one LoRA checkpoint, applies an in-memory capability dose,
and evaluates all three probes in a single process.
"""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import (
    apply_chat_template,
    exact_correct,
    is_abstain,
    rate,
    read_jsonl,
    write_json,
    write_jsonl,
)


def load_model(base_model: str, adapter: Path | None, dose: float) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if adapter is not None:
        if PeftModel is None:
            raise RuntimeError("peft is required for checkpoint evaluation")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        # Scaling LoRA-B is exactly equivalent to scaling the LoRA delta while
        # keeping rank/alpha intact. This mirrors the repository composition rule
        # but avoids materializing dozens of temporary dense checkpoints.
        dose = float(dose)
        for module in model.modules():
            lora_b = getattr(module, "lora_B", None)
            if lora_b is None:
                continue
            try:
                values = lora_b.values()
            except AttributeError:
                continue
            for layer in values:
                if hasattr(layer, "weight"):
                    layer.weight.data.mul_(dose)
    model.eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, messages: list[dict[str, Any]], max_seq: int, max_new: int) -> tuple[str, int, float]:
    prompt = apply_chat_template(tokenizer, messages, generation=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if int(encoded["input_ids"].shape[1]) > max_seq:
        raise RuntimeError(f"prompt length {encoded['input_ids'].shape[1]} exceeds {max_seq}")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), int(generated.numel()), elapsed


def target_positions(tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    prompt_text = apply_chat_template(tokenizer, messages[:-1], generation=True)
    full_text = apply_chat_template(tokenizer, messages, generation=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        common = 0
        for left, right in zip(prompt_ids, full_ids):
            if left != right:
                break
            common += 1
        # Chat templates occasionally differ by one boundary token between
        # generation and closed conversation forms. Require a strong prefix,
        # otherwise the metric would measure the wrong positions.
        if common < max(1, len(prompt_ids) - 2):
            raise ValueError("chat-template prompt/full tokenization is not sufficiently prefix aligned")
        prompt_len = common
    else:
        prompt_len = len(prompt_ids)
    if prompt_len >= len(full_ids):
        raise ValueError("assistant completion has no target tokens")
    return full_ids, list(range(prompt_len, len(full_ids)))


def continuous_metrics(model: Any, tokenizer: Any, row: dict[str, Any], max_seq: int) -> dict[str, float | int | None]:
    messages = [dict(message) for message in (row.get("messages") or [])]
    full_ids, positions = target_positions(tokenizer, messages)
    if len(full_ids) > max_seq:
        raise RuntimeError(f"full target length {len(full_ids)} exceeds {max_seq}")
    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0]
    nlls: list[float] = []
    margins: list[float] = []
    ranks: list[int] = []
    probs: list[float] = []
    entropies: list[float] = []
    for target_position in positions:
        if target_position <= 0:
            continue
        token_id = int(full_ids[target_position])
        vector = logits[target_position - 1].float()
        log_probs = F.log_softmax(vector, dim=-1)
        probs_vec = log_probs.exp()
        target_log_prob = float(log_probs[token_id].item())
        nlls.append(-target_log_prob)
        target_logit = float(vector[token_id].item())
        masked = vector.clone()
        masked[token_id] = -float("inf")
        best_wrong = float(masked.max().item())
        margins.append(target_logit - best_wrong)
        rank = int((vector > vector[token_id]).sum().item()) + 1
        ranks.append(rank)
        probs.append(float(probs_vec[token_id].item()))
        entropy = float(-(probs_vec * log_probs).sum().item())
        entropies.append(entropy)
    if not nlls:
        return {}
    return {
        "target_token_nll_mean": statistics.mean(nlls),
        "target_token_margin_mean": statistics.mean(margins),
        "target_token_rank_mean": statistics.mean(ranks),
        "target_token_gold_probability_mean": statistics.mean(probs),
        "first_target_nll": nlls[0],
        "first_target_margin": margins[0],
        "first_target_rank": ranks[0],
        "first_target_gold_probability": probs[0],
        "first_target_entropy": entropies[0],
        "completion_target_count": len(nlls),
    }


def evaluate_split(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    label: str,
    max_seq: int,
    max_new: int,
    logit_limit: int,
    output_root: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    answerable_logit_seen = 0
    for index, row in enumerate(rows, 1):
        messages = [dict(message) for message in (row.get("messages") or [])]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{row.get('id')}: target missing")
        answer, tokens, seconds = generate(model, tokenizer, messages[:-1], max_seq, max_new)
        decision = str(row.get("decision") or "answer")
        abstained = is_abstain(answer)
        correct = exact_correct(answer, row)
        record: dict[str, Any] = {
            "id": str(row.get("id") or index),
            "domain": row.get("domain"),
            "reasoning_family": row.get("reasoning_family"),
            "surface_format": row.get("surface_format"),
            "difficulty": row.get("difficulty"),
            "decision": decision,
            "answer": answer,
            "gold": row.get("approved_answer") or row.get("answer"),
            "correct": correct,
            "abstained": abstained,
            "false_abstention": decision == "answer" and abstained,
            "answered_when_answerable": decision == "answer" and not abstained,
            "generated_tokens": tokens,
            "generation_seconds": seconds,
        }
        if decision == "answer" and answerable_logit_seen < logit_limit:
            try:
                record.update(continuous_metrics(model, tokenizer, row, max_seq))
                answerable_logit_seen += 1
            except (ValueError, RuntimeError) as exc:
                record["continuous_metric_error"] = str(exc)
        results.append(record)
        output_rows.append(record)
    write_jsonl(output_root / f"{label}_results.jsonl", output_rows)

    answerable = [row for row in results if row["decision"] == "answer"]
    unanswerable = [row for row in results if row["decision"] != "answer"]
    hard_answerable = [row for row in answerable if row.get("difficulty") == "hard"]
    families: dict[str, list[bool]] = defaultdict(list)
    for row in answerable:
        families[str(row.get("reasoning_family") or "")].append(bool(row["correct"]))
    continuous_rows = [row for row in answerable if row.get("target_token_nll_mean") is not None]

    def mean_key(key: str) -> float | None:
        values = [float(row[key]) for row in continuous_rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        return statistics.mean(values) if values else None

    return {
        "case_count": len(results),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "overall_correctness": rate(row["correct"] for row in results),
        "answerable_correctness": rate(row["correct"] for row in answerable),
        "hard_answerable_correctness": rate(row["correct"] for row in hard_answerable),
        "abstention_correctness": rate(row["correct"] for row in unanswerable),
        "false_abstention_rate": rate(row["false_abstention"] for row in answerable),
        "answer_coverage": rate(row["answered_when_answerable"] for row in answerable),
        "macro_reasoning_family_correctness": (
            statistics.mean(sum(values) / len(values) for values in families.values() if values) if families else None
        ),
        "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in results) if results else None,
        "mean_generation_seconds": statistics.mean(row["generation_seconds"] for row in results) if results else None,
        "continuous_case_count": len(continuous_rows),
        "target_token_nll_mean": mean_key("target_token_nll_mean"),
        "target_token_margin_mean": mean_key("target_token_margin_mean"),
        "target_token_rank_mean": mean_key("target_token_rank_mean"),
        "target_token_gold_probability_mean": mean_key("target_token_gold_probability_mean"),
        "first_target_nll_mean": mean_key("first_target_nll"),
        "first_target_margin_mean": mean_key("first_target_margin"),
        "first_target_rank_mean": mean_key("first_target_rank"),
        "first_target_gold_probability_mean": mean_key("first_target_gold_probability"),
        "first_target_entropy_mean": mean_key("first_target_entropy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--train-probe", type=Path, required=True)
    parser.add_argument("--dev-probe", type=Path, required=True)
    parser.add_argument("--ood-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--logit-limit", type=int, default=96)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    if args.dose < 0.0:
        raise ValueError("dose must be nonnegative")
    for path in (args.train_probe, args.dev_probe, args.ood_probe):
        if "sealed" in str(path).casefold():
            raise ValueError("v2.3.1 checkpoint evaluator refuses sealed benchmarks")

    model, tokenizer = load_model(args.base_model, args.adapter, args.dose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for label, path in (("train", args.train_probe), ("dev", args.dev_probe), ("ood", args.ood_probe)):
        rows = read_jsonl(path)
        if args.limit_per_split:
            rows = rows[: args.limit_per_split]
        splits[label] = evaluate_split(
            model,
            tokenizer,
            rows,
            label=label,
            max_seq=args.max_sequence_length,
            max_new=args.max_new_tokens,
            logit_limit=args.logit_limit,
            output_root=args.output_dir,
        )
    summary = {
        "schema_version": "2.3.1",
        "base_model": args.base_model,
        "adapter": str(args.adapter) if args.adapter else None,
        "dose": args.dose,
        "step": args.step,
        "sealed_data_used": False,
        "splits": splits,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
