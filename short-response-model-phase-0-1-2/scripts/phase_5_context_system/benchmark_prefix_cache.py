# Purpose: Measures exact stable-prefix KV reuse against full prefill and checks numerical consistency.
from __future__ import annotations

"""Measure exact stable-prefix KV reuse against repeated full prefill."""

import argparse
import copy
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

from scripts.common.experiment_utils import read_jsonl, sha256_tree, write_json

DEFAULT_PREFIX = (
    "System: Reply naturally in a few words or one short sentence. Use relevant memory, "
    "ignore stale facts, and keep safety boundaries concise.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark stable-prefix KV reuse.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/context/context_eval.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_5_context/prefix_cache.json")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def immutable_cache(cache: Any) -> Any:
    """Prefer legacy tuple caches because decode calls do not mutate them."""

    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()
    device = model.device

    prefix_ids = tokenizer(args.prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        prefix_output = model(input_ids=prefix_ids, use_cache=True)
    cached_prefix = immutable_cache(prefix_output.past_key_values)

    full_times: list[float] = []
    cached_times: list[float] = []
    max_logit_difference = 0.0
    cases = read_jsonl(args.cases)[: args.max_cases]
    if not cases:
        raise SystemExit(f"No prefix-cache cases in {args.cases}")

    for _ in range(args.repeats):
        for row in cases:
            dynamic_text = f"User: {row['query']}\nAssistant:"
            dynamic_ids = tokenizer(dynamic_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            full_ids = torch.cat([prefix_ids, dynamic_ids], dim=1)
            full_mask = torch.ones_like(full_ids)

            sync()
            started = time.perf_counter()
            with torch.inference_mode():
                full_output = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
            sync()
            full_times.append((time.perf_counter() - started) * 1000)

            # Some cache classes are mutable, so copy outside the timed region.
            cache_for_call = copy.deepcopy(cached_prefix)
            sync()
            started = time.perf_counter()
            with torch.inference_mode():
                cached_output = model(
                    input_ids=dynamic_ids,
                    attention_mask=full_mask,
                    past_key_values=cache_for_call,
                    use_cache=True,
                )
            sync()
            cached_times.append((time.perf_counter() - started) * 1000)
            difference = (full_output.logits[:, -1, :].float() - cached_output.logits[:, -1, :].float()).abs().max()
            max_logit_difference = max(max_logit_difference, float(difference.item()))

    median_full = statistics.median(full_times)
    median_cached = statistics.median(cached_times)
    payload = {
        "model": args.model,
        "model_fingerprint": sha256_tree(Path(args.model)),
        "cases": len(cases),
        "repeats": args.repeats,
        "prefix_tokens": int(prefix_ids.shape[1]),
        "median_full_prefill_ms": median_full,
        "median_cached_dynamic_ms": median_cached,
        "steady_state_speedup": median_full / median_cached if median_cached > 0 else None,
        "max_last_logit_absolute_difference": max_logit_difference,
        "numerically_consistent": max_logit_difference <= 1e-3,
        "note": "Steady-state speedup excludes the one-time prefix cache construction cost.",
    }
    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
