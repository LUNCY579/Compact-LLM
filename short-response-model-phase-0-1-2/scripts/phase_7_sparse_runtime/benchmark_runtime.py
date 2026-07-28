from __future__ import annotations

"""Benchmark dense versus routed sparse single-token decode on identical prompts.

The routed path is a PyTorch reference using gathered MLP rows/columns. It is
expected to establish correctness and expose overhead; it is not a substitute
for the later Triton/custom kernel prototype.
"""

import argparse
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

from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl
from scripts.common.model_introspection import find_decoder_stack, first_parameter_device, get_gated_mlp
from scripts.common.sparse_mlp import RoutedSparseGatedMLP, TinyBlockRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure actual dense and routed-reference runtime.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--routers", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_7_router/runtime_comparison.json")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup-cases", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=None, help="Override router checkpoint target ratio.")
    parser.add_argument("--force-confidence-threshold", type=float, default=None)
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prompt_from_row(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    prompt = row.get("prompt") or row.get("text")
    if not isinstance(prompt, str):
        raise ValueError("Benchmark rows need messages, prompt, or text.")
    return prompt


def greedy_timed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = first_parameter_device(model)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    past = output.past_key_values
    decode_times: list[float] = []

    for _ in range(max_new_tokens - 1):
        if tokenizer.eos_token_id is not None and generated[-1] == tokenizer.eos_token_id:
            break
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
        synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        synchronize()
        decode_times.append((time.perf_counter() - start) * 1000)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        past = output.past_key_values

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_times,
        "generated_ids": generated,
        "reply": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "peak_vram_gb": peak_vram,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decode = [value for run in runs for value in run["decode_ms"]]
    return {
        "median_prefill_ms": statistics.median(run["prefill_ms"] for run in runs),
        "median_decode_ms_per_token": statistics.median(decode) if decode else None,
        "median_total_decode_ms": statistics.median(sum(run["decode_ms"]) for run in runs),
        "median_generated_tokens": statistics.median(len(run["generated_ids"]) for run in runs),
        "peak_vram_gb": max((run["peak_vram_gb"] or 0.0) for run in runs) if runs else None,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()
    router_data = torch.load(args.routers, map_location="cpu", weights_only=False)
    rows = read_jsonl(args.benchmark)[: args.max_cases]
    prompts = [prompt_from_row(tokenizer, row) for row in rows]
    if not prompts:
        raise ValueError("Benchmark produced no prompts.")

    # Warm the model/runtime before either measured mode.
    for prompt in prompts[: args.warmup_cases]:
        greedy_timed(model, tokenizer, prompt, min(4, args.max_new_tokens))

    dense_runs: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        for prompt in prompts:
            dense_runs.append(greedy_timed(model, tokenizer, prompt, args.max_new_tokens))

    wrappers: list[RoutedSparseGatedMLP] = []
    keep_ratio = float(args.keep_ratio if args.keep_ratio is not None else router_data["target_ratio"])
    for layer_index, layer in enumerate(find_decoder_stack(model).layers):
        layer_key = str(layer_index)
        checkpoint = router_data["routers"].get(layer_key)
        if checkpoint is None:
            raise KeyError(f"Router checkpoint has no layer {layer_key}.")
        router = TinyBlockRouter(
            int(checkpoint["hidden_size"]),
            int(checkpoint["router_width"]),
            int(checkpoint["num_blocks"]),
        )
        router.load_state_dict(checkpoint["state_dict"])
        dense_mlp = get_gated_mlp(layer)
        router.to(device=dense_mlp.gate_proj.weight.device, dtype=dense_mlp.gate_proj.weight.dtype)
        threshold = (
            float(args.force_confidence_threshold)
            if args.force_confidence_threshold is not None
            else float(checkpoint["confidence_threshold"])
        )
        wrapper = RoutedSparseGatedMLP(
            dense_mlp,
            router,
            block_size=int(router_data["block_size"]),
            keep_ratio=keep_ratio,
            confidence_threshold=threshold,
            dense_prefill=True,
        )
        layer.mlp = wrapper
        wrappers.append(wrapper)

    for prompt in prompts[: args.warmup_cases]:
        greedy_timed(model, tokenizer, prompt, min(4, args.max_new_tokens))
    # Warmup calls are excluded from reported active-ratio/fallback telemetry.
    for wrapper in wrappers:
        wrapper.stats = type(wrapper.stats)()
    sparse_runs: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        for prompt in prompts:
            sparse_runs.append(greedy_timed(model, tokenizer, prompt, args.max_new_tokens))

    dense_summary = summarize(dense_runs)
    sparse_summary = summarize(sparse_runs)
    dense_decode = dense_summary["median_decode_ms_per_token"]
    sparse_decode = sparse_summary["median_decode_ms_per_token"]
    speedup = dense_decode / sparse_decode if dense_decode and sparse_decode else None
    dense_vram = dense_summary["peak_vram_gb"]
    sparse_vram = sparse_summary["peak_vram_gb"]
    vram_reduction = 1 - sparse_vram / dense_vram if dense_vram and sparse_vram else None

    paired = min(len(dense_runs), len(sparse_runs))
    exact = [dense_runs[index]["generated_ids"] == sparse_runs[index]["generated_ids"] for index in range(paired)]
    raw_rows = []
    for index in range(paired):
        raw_rows.append(
            {
                "run_index": index,
                "dense_reply": dense_runs[index]["reply"],
                "sparse_reply": sparse_runs[index]["reply"],
                "exact_token_match": exact[index],
                "dense_prefill_ms": dense_runs[index]["prefill_ms"],
                "sparse_prefill_ms": sparse_runs[index]["prefill_ms"],
                "dense_decode_ms": dense_runs[index]["decode_ms"],
                "sparse_decode_ms": sparse_runs[index]["decode_ms"],
            }
        )
    raw_path = args.out.with_name(args.out.stem + "_raw.jsonl")
    write_jsonl(raw_path, raw_rows)

    layer_stats = {str(index): wrapper.stats.as_dict() for index, wrapper in enumerate(wrappers)}
    payload = {
        "model": args.model,
        "routers": str(args.routers),
        "benchmark": str(args.benchmark),
        "cases": len(prompts),
        "repeats": args.repeats,
        "keep_ratio": keep_ratio,
        "dense": dense_summary,
        "sparse_reference": sparse_summary,
        "decode_speedup": speedup,
        "peak_vram_reduction_fraction": vram_reduction,
        "exact_generation_match_rate": statistics.fmean(exact) if exact else None,
        "layer_router_stats": layer_stats,
        "passes_runtime_target": bool(
            (speedup is not None and speedup >= 1.20)
            or (vram_reduction is not None and vram_reduction >= 0.20)
        ),
        "raw_results": str(raw_path),
        "warning": (
            "This is the PyTorch gather reference, not the final custom kernel. A slower result is a valid negative result. "
            "Quality gates remain separate from speed/memory gates."
        ),
    }
    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
