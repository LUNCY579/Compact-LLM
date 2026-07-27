from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from rapidfuzz.fuzz import ratio as fuzzy_ratio
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "benchmark"
    / "shortchatbench_pilot.jsonl"
)
DEFAULT_SYSTEM_PROMPT = (
    "Reply like a casual text message. Use only a few words or one short sentence. "
    "Be accurate, use the conversation context, ask one brief clarification when "
    "information is missing, and keep safety boundaries concise."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a base model or PEFT adapter.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline_evaluation"))
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9+._-]*", normalize(text))


def contains(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def complete_utterance(reply: str) -> bool:
    value = reply.strip()
    if not value:
        return False
    if len(words(value)) <= 4:
        return True
    if value.endswith((".", "!", "?", "…", "😊", "😭", "🙂", "👍")):
        return True
    return not normalize(value).endswith(
        (" and", " or", " but", " because", " which", " that", " to", " with", " for")
    )


def echo_score(user: str, reply: str) -> float:
    if not user.strip() or not reply.strip():
        return 0.0
    return float(fuzzy_ratio(normalize(user), normalize(reply)))


def score_case(record: dict[str, Any], reply: str, token_count: int) -> dict[str, Any]:
    exp = record.get("expected", {})
    checks: dict[str, bool] = {
        "nonempty": bool(reply.strip()),
        "complete_utterance": complete_utterance(reply),
        "token_range": int(exp.get("min_tokens", 1)) <= token_count <= int(exp.get("max_tokens", 32)),
    }
    for phrase in exp.get("must_include_all", []):
        checks[f"include:{phrase}"] = contains(reply, phrase)
    include_any = exp.get("must_include_any", [])
    if include_any:
        checks["include_any"] = any(contains(reply, p) for p in include_any)
    forbidden = exp.get("must_not_include_any", [])
    if forbidden:
        checks["exclude_forbidden"] = not any(contains(reply, p) for p in forbidden)
    if exp.get("should_ask_question"):
        checks["asks_question"] = "?" in reply
    if exp.get("should_not_ask_question"):
        checks["does_not_ask_question"] = "?" not in reply
    if exp.get("should_not_echo_user"):
        user = record["messages"][-1]["content"]
        checks["does_not_echo_user"] = echo_score(user, reply) < 78.0

    passed = sum(checks.values())
    total = len(checks)
    return {
        "checks": checks,
        "passed_checks": passed,
        "total_checks": total,
        "heuristic_score": passed / total if total else 1.0,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return float(ordered[index])


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix PyTorch/GPU setup first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.benchmark)
    if args.limit is not None:
        records = records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    if args.adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("Install peft before loading an adapter.")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        messages = record["messages"]
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": args.system_prompt}, *messages]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        torch.cuda.reset_peak_memory_stats()
        sync_cuda()
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            kwargs.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        sync_cuda()
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, inputs.input_ids.shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        scored = score_case(record, reply, int(new_tokens.shape[0]))
        row = {
            "id": record["id"],
            "category": record["category"],
            "messages": record["messages"],
            "reply": reply,
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(new_tokens.shape[0]),
            "elapsed_seconds": elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            **scored,
        }
        outputs.append(row)
        print(f"[{index:03d}/{len(records):03d}] {record['id']}: {reply!r} score={row['heuristic_score']:.3f}")

    result_path = args.output_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    token_counts = [float(r["generated_tokens"]) for r in outputs]
    times = [float(r["elapsed_seconds"]) for r in outputs]
    scores = [float(r["heuristic_score"]) for r in outputs]
    category_scores: dict[str, list[float]] = defaultdict(list)
    for row in outputs:
        category_scores[row["category"]].append(float(row["heuristic_score"]))
    repeats = Counter(normalize(r["reply"]) for r in outputs)
    summary = {
        "case_count": len(outputs),
        "median_reply_tokens": statistics.median(token_counts),
        "p90_reply_tokens": percentile(token_counts, 0.90),
        "mean_heuristic_score": statistics.mean(scores),
        "complete_utterance_rate": statistics.mean(
            1.0 if r["checks"].get("complete_utterance") else 0.0 for r in outputs
        ),
        "mean_generation_seconds": statistics.mean(times),
        "p90_generation_seconds": percentile(times, 0.90),
        "peak_vram_gb": max(float(r["peak_vram_gb"]) for r in outputs),
        "duplicate_reply_count": sum(c - 1 for c in repeats.values() if c > 1),
        "category_scores": {
            k: statistics.mean(v) for k, v in sorted(category_scores.items())
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cat_rows = "\n".join(f"| {k} | {v:.3f} |" for k, v in summary["category_scores"].items())
    worst = sorted(outputs, key=lambda r: r["heuristic_score"])[:10]
    worst_rows = "\n".join(
        f"| {r['id']} | {r['category']} | {r['heuristic_score']:.3f} | {r['reply'].replace('|', '\\|')} |"
        for r in worst
    )
    report = f"""# BASELINE EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter on ShortChatBench Pilot.
The automated score is a pilot harness, not a replacement for human review.

## Inputs

- Model: `{args.model_id}`
- Adapter: `{args.adapter_path or 'None (base model)'}`
- Benchmark: `{args.benchmark}`
- Cases: {summary['case_count']}
- Deterministic decoding: {not args.do_sample}

## Aggregate Results

| Metric | Value |
|---|---:|
| Median generated tokens | {summary['median_reply_tokens']} |
| P90 generated tokens | {summary['p90_reply_tokens']} |
| Mean heuristic score | {summary['mean_heuristic_score']:.3f} |
| Complete-utterance rate | {summary['complete_utterance_rate']:.3f} |
| Mean generation time (s) | {summary['mean_generation_seconds']:.3f} |
| P90 generation time (s) | {summary['p90_generation_seconds']:.3f} |
| Peak allocated VRAM (GB) | {summary['peak_vram_gb']:.3f} |
| Duplicate replies beyond first occurrence | {summary['duplicate_reply_count']} |

## Category Scores

| Category | Mean heuristic score |
|---|---:|
{cat_rows}

## Lowest-Scoring Cases

| ID | Category | Score | Reply |
|---|---|---:|---|
{worst_rows}

## Gate Status

`HUMAN REVIEW REQUIRED`

Review correctness, naturalness, context use, and safety before using this as a
formal project baseline.
"""
    (args.output_dir / "BASELINE_EVALUATION.md").write_text(report, encoding="utf-8")
    print(f"\nResults: {result_path}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Report: {args.output_dir / 'BASELINE_EVALUATION.md'}")


if __name__ == "__main__":
    main()
