"""v3 E5 Gate 0 — validate the streaming forward against transformers.

Amendment v3.1 makes this a gate: no donor measurement may be reported from an
unvalidated hand-written forward pass. Qwen3 uses grouped-query attention,
per-head `q_norm`/`k_norm` RMSNorm, and RoPE with `rope_theta` 1e6; any of those
can be silently wrong and still produce plausible-looking activations.

Qwen3-0.6B fits in RAM, so it can be run both ways and compared exactly.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json  # noqa: E402
from rineya_cross_scale.streaming_forward import (  # noqa: E402
    StreamingWeights,
    streaming_forward,
    streaming_forward_many,
)

TOLERANCE = 1e-3
CAPTURE_LAYERS = (0, 7, 14, 21, 27)


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E5 Gate 0 streaming-forward validation.")
    parser.add_argument("--model", type=Path,
                        default=Path("C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
                                     "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"))
    parser.add_argument("--dataset", type=Path,
                        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e5_cross_scale_atlas")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    started = time.perf_counter()

    rows = [json.loads(line) for line in args.dataset.read_text("utf-8").splitlines() if line.strip()]
    # Take a short and a long prompt so the causal mask and RoPE are exercised.
    rows.sort(key=lambda r: len(r["prompt"]))
    picked = [rows[0], rows[len(rows) // 3], rows[2 * len(rows) // 3], rows[-1]][: args.cases]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    reference = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.float32, trust_remote_code=False
    )
    reference.eval()

    comparisons = []
    with torch.inference_mode():
        for row in picked:
            text = render_prompt(tokenizer, row["prompt"])
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            output = reference(**encoded, output_hidden_states=True, use_cache=False)
            expected = {"embedding": output.hidden_states[0][0, -1].clone()}
            for layer in CAPTURE_LAYERS:
                expected[f"layer_{layer}"] = output.hidden_states[layer + 1][0, -1].clone()
            expected_logits = output.logits[0, -1].clone()
            del output
            comparisons.append({
                "case_id": row["case_id"],
                "tokens": int(encoded["input_ids"].shape[1]),
                "input_ids": encoded["input_ids"][0].clone(),
                "expected": expected,
                "expected_logits": expected_logits,
            })
    del reference, tokenizer
    gc.collect()

    def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(actual - expected)
                     / max(float(torch.linalg.vector_norm(expected)), 1e-30))

    results = []
    with StreamingWeights(args.model) as weights:
        for case in comparisons:
            produced = streaming_forward(
                weights, case["input_ids"], capture_layers=CAPTURE_LAYERS, want_logits=True
            )
            errors = {
                slot: relative_error(produced["captured"][slot], value)
                for slot, value in case["expected"].items()
            }
            errors["logits"] = relative_error(produced["logits"], case["expected_logits"])
            top1_match = bool(
                produced["logits"].argmax() == case["expected_logits"].argmax())
            results.append({
                "case_id": case["case_id"],
                "tokens": case["tokens"],
                "relative_errors": errors,
                "max_relative_error": max(errors.values()),
                "top1_token_matches": top1_match,
            })
            print(f"  {case['case_id'][:44]:<46} tokens={case['tokens']:<5} "
                  f"max_err={max(errors.values()):.3e} top1={'ok' if top1_match else 'MISMATCH'}",
                  flush=True)
            del produced
            gc.collect()

    # The capture path uses the loop-inverted batched forward, so validate that
    # too: a per-prompt pass that is correct says nothing about the batched one.
    batched_errors = []
    with StreamingWeights(args.model) as weights:
        produced = streaming_forward_many(
            weights, [c["input_ids"] for c in comparisons],
            capture_layers=CAPTURE_LAYERS, want_logits=True,
        )
    for case, got in zip(comparisons, produced):
        errors = {slot: relative_error(got["captured"][slot], value)
                  for slot, value in case["expected"].items()}
        errors["logits"] = relative_error(got["logits"], case["expected_logits"])
        batched_errors.append({
            "case_id": case["case_id"],
            "relative_errors": errors,
            "max_relative_error": max(errors.values()),
            "top1_token_matches": bool(got["logits"].argmax() == case["expected_logits"].argmax()),
        })
        print(f"  [batched] {case['case_id'][:36]:<38} max_err={max(errors.values()):.3e}", flush=True)
    del produced
    gc.collect()

    worst_single = max(r["max_relative_error"] for r in results)
    worst_batched = max(r["max_relative_error"] for r in batched_errors)
    worst = max(worst_single, worst_batched)
    passed = (worst < TOLERANCE
              and all(r["top1_token_matches"] for r in results)
              and all(r["top1_token_matches"] for r in batched_errors))

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e5_gate0_streaming_forward_validation",
        "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
        "reference_implementation": "transformers Qwen3ForCausalLM, float32, CPU",
        "model": str(args.model),
        "tolerance": TOLERANCE,
        "capture_layers": list(CAPTURE_LAYERS),
        "cases": results,
        "batched_cases": batched_errors,
        "worst_relative_error_per_prompt": worst_single,
        "worst_relative_error_batched": worst_batched,
        "worst_relative_error": worst,
        "gate": "PASS" if passed else "FAIL",
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 1},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "gate0_streaming_forward_validation.json", payload)
    print(f"\nworst relative error {worst:.3e} against tolerance {TOLERANCE:.0e} -> {payload['gate']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
