#!/usr/bin/env python3
"""Real-checkpoint TGSA + MTP conversion and exact-decoding smoke test.

This script does not train or claim quality. It validates the conversion path on
an actual local Rineya/Qwen3 checkpoint, measures main-logit drift in the exact
deployment dtype/attention implementation, and proves verified MTP decoding
cannot change greedy output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rineya_tgsa.config import RineyaTGSAConfig
from rineya_tgsa.mtp_adapter import (
    CombinedRetrofitConfig,
    adapter_only_state_dict,
    convert_and_attach_tgsa_mtp,
    state_sha256,
)
from tgsa.phase2_decode import assert_exact_greedy_equivalence


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")
    return mapping[value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--prompt", default="Explain why careful verification matters.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-offset", type=int, default=4)
    parser.add_argument("--trajectory-rank", type=int, default=32)
    parser.add_argument("--mtp-rank", type=int, default=64)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn-implementation", default="eager", choices=("eager", "sdpa"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:
        raise SystemExit("transformers with Qwen3 support is required") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        native_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits.detach().cpu()

    source_config = model.config.to_dict()
    tgsa_config = RineyaTGSAConfig.from_source_config(
        source_config,
        trajectory_rank=args.trajectory_rank,
        condition_q=True,
        condition_k=False,
        condition_v=False,
    )
    combined_config = CombinedRetrofitConfig(
        max_offset=args.max_offset,
        mtp_rank=args.mtp_rank,
    )
    wrapper, manifest = convert_and_attach_tgsa_mtp(
        model,
        tgsa_config=tgsa_config,
        layers=args.layers,
        combined_config=combined_config,
    )
    wrapper.to(args.device)
    wrapper.eval()
    with torch.no_grad():
        combined_logits = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits.detach().cpu()
    difference = (native_logits.float() - combined_logits.float()).abs()

    decode = assert_exact_greedy_equivalence(
        wrapper,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        min_confidence=combined_config.mtp_min_proposal_confidence,
    )
    adapter_state = adapter_only_state_dict(wrapper)
    report = {
        "schema_version": "rineya.tgsa-mtp-real-smoke.v1",
        "model_dir": str(args.model_dir),
        "layers": args.layers,
        "dtype": str(args.dtype),
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "prompt_token_count": int(input_ids.shape[1]),
        "conversion_manifest": manifest,
        "main_logit_drift": {
            "bitwise_identical": bool(torch.equal(native_logits, combined_logits)),
            "max_absolute": float(difference.max().item()),
            "mean_absolute": float(difference.mean().item()),
            "native_rms": float(native_logits.float().square().mean().sqrt().item()),
        },
        "verified_decode": decode,
        "adapter_tensor_count": len(adapter_state),
        "adapter_state_sha256": state_sha256(adapter_state),
        "claim_scope": (
            "real-checkpoint conversion, drift, and exact-decoding validation only; no "
            "quality improvement or wall-clock acceleration is claimed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["main_logit_drift"], indent=2))
    print(f"verified greedy equivalence: {decode['exact']}")


if __name__ == "__main__":
    main()
