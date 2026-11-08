#!/usr/bin/env python3
"""Single-example trajectory inspector, layer by layer and head by head.

Emits summaries by default; full tensors only with --dump-tensors, so a routine
trace does not produce gigabytes nobody reads.

For every converted layer and every sequential head it reports the carrier
summary, velocity, encoded control, correction gate, attention entropy, and the
conditioning magnitude actually injected. Head 0 is labelled as the bootstrap
head rather than silently reported as having zero trajectory.

Fails closed on a missing checkpoint or an unconverted layer request.
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
from rineya_tgsa.qwen3_adapter import Qwen3TGSAAttention, convert_qwen3_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layers", default="0,14,27")
    parser.add_argument("--trace-layers", default=None,
                        help="subset of converted layers to trace; default all")
    parser.add_argument("--trajectory-rank", type=int, default=32)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--top-tokens", type=int, default=5)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--dump-tensors", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not (args.model_dir / "model.safetensors").is_file():
        print(f"ERROR: no model.safetensors under {args.model_dir}", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tgsa_config = RineyaTGSAConfig.from_source_config(
        source_config, trajectory_rank=args.trajectory_rank
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).eval()

    converted = [int(value) for value in args.layers.split(",")]
    record = convert_qwen3_model(model, tgsa_config, layers=converted)

    traced = (
        [int(value) for value in args.trace_layers.split(",")]
        if args.trace_layers
        else record["converted_layers"]
    )
    missing = [index for index in traced if index not in record["converted_layers"]]
    if missing:
        print(f"ERROR: layers {missing} were not converted; nothing to trace", file=sys.stderr)
        return 1

    for index in traced:
        model.model.layers[index].self_attn.set_telemetry("summary")

    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output = model(input_ids=ids)
    logits = output.logits[0, -1]
    top = logits.topk(args.top_tokens)

    layers_report = []
    for index in traced:
        module = model.model.layers[index].self_attn
        telemetry = module.last_telemetry
        if telemetry is None:
            continue
        heads = []
        for head in range(module.num_heads):
            heads.append(
                {
                    "head": head,
                    "role": "bootstrap" if head == 0 else "trajectory",
                    "carrier_rms": telemetry.carrier_rms[head],
                    "velocity_rms": telemetry.velocity_rms[head],
                    "control_rms": telemetry.control_rms[head],
                    "conditioning_rms": telemetry.conditioning_rms[head],
                    "attention_entropy": telemetry.attention_entropy[head],
                }
            )
        layers_report.append(
            {
                "layer": index,
                "effective_conditioned_heads": module.effective_conditioned_heads,
                "conditioning_is_neutral": module.conditioning_is_neutral(),
                "heads": heads,
            }
        )

    trace = {
        "schema_version": "rineya.tgsa-trace.v1",
        "model_dir": str(args.model_dir),
        "dtype": args.dtype,
        "attn_implementation": args.attn_impl,
        "prompt": args.prompt,
        "token_count": int(ids.shape[1]),
        "converted_layers": record["converted_layers"],
        "traced_layers": traced,
        "head_zero_semantics": (
            "head 0 is the bootstrap head: it has no previous carrier, so its velocity "
            "and control are identically zero and its conditioning weights receive no "
            "gradient. Heads 1..H-1 carry valid trajectory finite differences."
        ),
        "top_next_tokens": [
            {"token_id": int(t), "token": tokenizer.decode([int(t)]), "logit": float(v)}
            for v, t in zip(top.values, top.indices)
        ],
        "layers": layers_report,
    }

    if args.gradients:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        groups: dict[str, float] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            root = "trajectory" if any(
                key in name
                for key in ("carrier_init", "trajectory_encoders", "correction_", "conditioning")
            ) else "pretrained"
            groups[root] = groups.get(root, 0.0) + float(parameter.grad.detach().float().norm()) ** 2
        trace["gradient_health"] = {
            "loss": float(out.loss),
            "all_finite": all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            ),
            "group_norms": {k: v**0.5 for k, v in sorted(groups.items())},
        }

    payload = json.dumps(trace, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
