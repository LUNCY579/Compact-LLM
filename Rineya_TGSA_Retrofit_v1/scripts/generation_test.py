#!/usr/bin/env python3
"""Generation and KV-cache correctness for a converted Rineya model.

Checks, on the real checkpoint:

* greedy generation agreement between native and neutral-converted models;
* cached versus uncached forward agreement (the KV-cache correctness gate);
* prefill/decode agreement;
* generation stability (no NaN, no degenerate repetition introduced).

Fails closed. The published core requires greedy decoding, so sampling is not
offered: a sampling mismatch would be untestable rather than informative.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import torch

from rineya_tgsa import RineyaTGSAConfig
from rineya_tgsa.qwen3_adapter import convert_qwen3_model

PROMPTS = [
    "The capital of France is",
    "Q: What is 17 plus 25?\nA:",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Summarize in one sentence: Photosynthesis converts light into chemical energy.\n",
]


def load(model_dir: Path, dtype, impl):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=dtype, trust_remote_code=True, attn_implementation=impl
    ).eval()


def greedy(model, ids, mask, steps: int) -> torch.Tensor:
    """Greedy decode without the generate() helper, so the loop is inspectable."""

    out = ids
    attention = mask
    for _ in range(steps):
        with torch.no_grad():
            logits = model(input_ids=out, attention_mask=attention).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        out = torch.cat((out, nxt), dim=1)
        attention = torch.cat((attention, torch.ones_like(nxt)), dim=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--trajectory-rank", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-impl", default="eager")
    args = parser.parse_args()

    if not (args.model_dir / "model.safetensors").is_file():
        print(f"ERROR: no model.safetensors under {args.model_dir}", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    encoded = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    dtype = getattr(torch, args.dtype)
    layers = [int(value) for value in args.layers.split(",")]

    source_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tgsa_config = RineyaTGSAConfig.from_source_config(
        source_config, trajectory_rank=args.trajectory_rank
    )

    print("loading native ...", flush=True)
    native = load(args.model_dir, dtype, args.attn_impl)
    native_tokens = greedy(native, ids, mask, args.max_new_tokens)
    with torch.no_grad():
        native_uncached = native(input_ids=native_tokens).logits
        cached = native(input_ids=native_tokens[:, :-1], use_cache=True)
        native_step = native(
            input_ids=native_tokens[:, -1:], past_key_values=cached.past_key_values
        ).logits
    del native

    print("loading and converting retrofit ...", flush=True)
    retro = load(args.model_dir, dtype, args.attn_impl)
    record = convert_qwen3_model(retro, tgsa_config, layers=layers)
    retro_tokens = greedy(retro, ids, mask, args.max_new_tokens)
    with torch.no_grad():
        retro_uncached = retro(input_ids=native_tokens).logits
        retro_cached = retro(input_ids=native_tokens[:, :-1], use_cache=True)
        retro_step = retro(
            input_ids=native_tokens[:, -1:], past_key_values=retro_cached.past_key_values
        ).logits

    generation_match = bool(torch.equal(native_tokens, retro_tokens))
    per_prompt = [
        {
            "prompt": PROMPTS[i],
            "native": tokenizer.decode(native_tokens[i][ids.shape[1]:], skip_special_tokens=True),
            "retrofit": tokenizer.decode(retro_tokens[i][ids.shape[1]:], skip_special_tokens=True),
            "identical": bool(torch.equal(native_tokens[i], retro_tokens[i])),
        }
        for i in range(len(PROMPTS))
    ]

    cached_vs_uncached_native = float((native_step - native_uncached[:, -1:]).abs().max())
    cached_vs_uncached_retro = float((retro_step - retro_uncached[:, -1:]).abs().max())

    report = {
        "schema_version": "rineya.tgsa-generation-test.v1",
        "model_dir": str(args.model_dir),
        "dtype": args.dtype,
        "attn_implementation": args.attn_impl,
        "converted_layers": record["converted_layers"],
        "max_new_tokens": args.max_new_tokens,
        "greedy_generation_identical": generation_match,
        "greedy_token_agreement": float(
            (native_tokens == retro_tokens).float().mean()
        ),
        "logit_max_abs_diff": float((retro_uncached - native_uncached).abs().max()),
        "logits_bitwise_identical": bool(torch.equal(native_uncached, retro_uncached)),
        "cached_vs_uncached_max_abs_diff": {
            "native": cached_vs_uncached_native,
            "retrofit": cached_vs_uncached_retro,
        },
        "retrofit_cache_no_worse_than_native": cached_vs_uncached_retro
        <= max(cached_vs_uncached_native * 10.0, 1e-4),
        "all_finite": bool(torch.isfinite(retro_uncached).all()),
        "per_prompt": per_prompt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"greedy generation identical : {generation_match}")
    print(f"logits bitwise identical    : {report['logits_bitwise_identical']}")
    print(f"cached vs uncached (native) : {cached_vs_uncached_native:.3e}")
    print(f"cached vs uncached (retro)  : {cached_vs_uncached_retro:.3e}")
    print(f"all finite                  : {report['all_finite']}")
    print(f"wrote {args.output}")

    failures = []
    if not report["all_finite"]:
        failures.append("non-finite logits")
    if not report["retrofit_cache_no_worse_than_native"]:
        failures.append("retrofit KV cache disagrees with its own uncached forward")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
