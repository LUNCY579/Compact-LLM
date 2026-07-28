from __future__ import annotations

"""Measure oracle contribution retention and optional masked-model divergence."""

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json, write_jsonl
from scripts.common.model_introspection import find_decoder_stack, first_parameter_device, get_gated_mlp
from scripts.common.sparse_mlp import OracleMaskedGatedMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 dense-compute sparse simulations.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.8, 0.65, 0.5, 0.35])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_6_sparse/frontier.json")
    parser.add_argument("--model", default=None, help="Optional model for real masked generation comparisons.")
    parser.add_argument("--benchmark", type=Path, default=None, help="Development benchmark only; do not use sealed evaluation here.")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def prompt_from_row(tokenizer: Any, row: dict[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    text = row.get("prompt") or row.get("text")
    if not isinstance(text, str):
        raise ValueError("Benchmark row has no prompt/messages.")
    return text


def generation(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, list[int]]:
    device = first_parameter_device(model)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = output[0, encoded["input_ids"].shape[1] :].tolist()
    return tokenizer.decode(continuation, skip_special_tokens=True).strip(), continuation


def common_prefix_fraction(left: list[int], right: list[int]) -> float:
    limit = min(len(left), len(right))
    matched = 0
    for index in range(limit):
        if left[index] != right[index]:
            break
        matched += 1
    return matched / max(1, max(len(left), len(right)))


def main() -> None:
    args = parse_args()
    trace_data = torch.load(args.traces, map_location="cpu", weights_only=False)
    frontier = []
    for ratio in args.ratios:
        retained_values: list[float] = []
        for layer in trace_data["layers"].values():
            scores = layer["block_scores"].float()
            blocks = scores.shape[-1]
            keep = max(1, min(blocks, int(round(blocks * ratio))))
            retained = torch.topk(scores, k=keep, dim=-1).values.sum(dim=-1) / scores.sum(dim=-1).clamp_min(1e-12)
            retained_values.extend(retained.tolist())
        frontier.append(
            {
                "active_ratio": ratio,
                "mean_oracle_contribution_retained": statistics.fmean(retained_values),
                "p10_oracle_contribution_retained": sorted(retained_values)[max(0, int(0.10 * len(retained_values)) - 1)],
                "quality_evaluation_required": True,
                "speed_claim_allowed": False,
            }
        )

    payload: dict[str, Any] = {
        "frontier": frontier,
        "warning": "Oracle masking and contribution analysis still use dense computation; no speed claim is allowed.",
    }

    if args.model and args.benchmark:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
        model.eval()
        rows = read_jsonl(args.benchmark)[: args.max_cases]
        prompts = [prompt_from_row(tokenizer, row) for row in rows]
        dense_outputs = [generation(model, tokenizer, prompt, args.max_new_tokens) for prompt in prompts]

        wrappers: list[OracleMaskedGatedMLP] = []
        for layer in find_decoder_stack(model).layers:
            original = get_gated_mlp(layer)
            wrapper = OracleMaskedGatedMLP(original, int(trace_data["block_size"]), 1.0)
            layer.mlp = wrapper
            wrappers.append(wrapper)

        comparisons = []
        raw_rows = []
        for ratio in args.ratios:
            for wrapper in wrappers:
                wrapper.keep_ratio = ratio
                wrapper.stats = type(wrapper.stats)()
            sparse_outputs = [generation(model, tokenizer, prompt, args.max_new_tokens) for prompt in prompts]
            exact = [dense[1] == sparse[1] for dense, sparse in zip(dense_outputs, sparse_outputs)]
            prefixes = [common_prefix_fraction(dense[1], sparse[1]) for dense, sparse in zip(dense_outputs, sparse_outputs)]
            comparisons.append(
                {
                    "active_ratio": ratio,
                    "exact_generation_match_rate": statistics.fmean(exact),
                    "mean_token_common_prefix_fraction": statistics.fmean(prefixes),
                    "mean_reported_active_ratio": statistics.fmean(
                        wrapper.stats.as_dict()["mean_active_block_ratio"] for wrapper in wrappers
                    ),
                    "speed_claim_allowed": False,
                }
            )
            for index, ((dense_text, _), (sparse_text, _)) in enumerate(zip(dense_outputs, sparse_outputs)):
                raw_rows.append(
                    {
                        "case_index": index,
                        "active_ratio": ratio,
                        "prompt": prompts[index],
                        "dense_reply": dense_text,
                        "masked_reply": sparse_text,
                    }
                )
        payload["masked_generation_comparison"] = comparisons
        raw_path = args.out.with_name(args.out.stem + "_generations.jsonl")
        write_jsonl(raw_path, raw_rows)
        payload["raw_generations"] = str(raw_path)

    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
