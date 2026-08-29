#!/usr/bin/env python3
"""Real-checkpoint neutral-conversion damage report for Rineya-Supa-A1.

Loads the actual published checkpoint, converts selected layers with neutral
(zero) conditioning, and compares against the unmodified model on development
prompts. Includes deliberately perturbed positive controls so the harness's
sensitivity is demonstrated rather than asserted.

Fails closed on a missing checkpoint or a conversion that is not neutral.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import torch
from torch import nn
from rineya_tgsa import RineyaTGSAConfig
from rineya_tgsa.qwen3_adapter import Qwen3TGSAAttention, convert_qwen3_model


PROMPTS = [
    "The capital of France is",
    "Q: What is 17 plus 25?\nA:",
    "Once upon a time, in a small village by the sea,",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "The three branches of the United States government are",
    "Translate to French: The book is on the table.\n",
    "Summarize: Photosynthesis is the process by which plants convert light energy into chemical energy.\n",
    "A: I don't know the answer to that question because",
]


def load(model_dir: Path, dtype, attn_impl: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).eval()


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a = native logits, b = retrofit logits."""
    af, bf = a.float(), b.float()
    diff = (bf - af).abs()
    p = torch.log_softmax(af, -1)
    q = torch.log_softmax(bf, -1)
    kl = (p.exp() * (p - q)).sum(-1)
    top1 = (af.argmax(-1) == bf.argmax(-1)).float().mean()
    native_best = af.argmax(-1, keepdim=True)
    top5 = (bf.topk(5, -1).indices == native_best).any(-1).float().mean()
    nll_a = torch.nn.functional.cross_entropy(af[:, :-1].flatten(0, 1), af[:, 1:].argmax(-1).flatten())
    nll_b = torch.nn.functional.cross_entropy(bf[:, :-1].flatten(0, 1), af[:, 1:].argmax(-1).flatten())
    return {
        "logit_abs_diff_max": float(diff.max()), "logit_abs_diff_mean": float(diff.mean()),
        "kl_max": float(kl.max()), "kl_mean": float(kl.mean()), "kl_median": float(kl.median()),
        "top1_agreement": float(top1), "top5_agreement": float(top5),
        "nll_native": float(nll_a), "nll_retrofit": float(nll_b),
        "nll_delta": float(nll_b - nll_a),
        "bitwise_identical": bool(torch.equal(a, b)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", default="0,7,14,21,27")
    p.add_argument("--trajectory-rank", type=int, default=32)
    p.add_argument("--max-length", type=int, default=48)
    p.add_argument("--dtypes", default="float32")
    p.add_argument("--attn-impl", default="eager")
    args = p.parse_args()

    if not (args.model_dir / "model.safetensors").is_file():
        print(f"ERROR: no model.safetensors under {args.model_dir}", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    enc = tok(PROMPTS, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_length)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    layers = [int(x) for x in args.layers.split(",")]

    results = []
    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        print(f"[{dtype_name}] loading native...", flush=True)
        native = load(args.model_dir, dtype, args.attn_impl)
        with torch.no_grad():
            native_logits = native(input_ids=ids, attention_mask=mask).logits
        source_cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
        tgsa_cfg = RineyaTGSAConfig.from_source_config(source_cfg, trajectory_rank=args.trajectory_rank)

        print(f"[{dtype_name}] converting layers {layers}...", flush=True)
        retro = load(args.model_dir, dtype, args.attn_impl)
        record = convert_qwen3_model(retro, tgsa_cfg, layers=layers)
        with torch.no_grad():
            retro_logits = retro(input_ids=ids, attention_mask=mask).logits
        neutral = metrics(native_logits, retro_logits)
        neutral.update({"dtype": dtype_name, "attn_implementation": args.attn_impl,
                        "condition": "neutral", "converted_layers": layers,
                        "conversion_record": record})
        results.append(neutral)
        print(f"  NEUTRAL  kl_max={neutral['kl_max']:.3e} top1={neutral['top1_agreement']:.4f} "
              f"|dlogit|max={neutral['logit_abs_diff_max']:.3e} bitwise={neutral['bitwise_identical']}",
              flush=True)

        # Positive controls: the harness must react to a real change.
        for std in (1e-3, 1e-1):
            perturbed = load(args.model_dir, dtype, args.attn_impl)
            convert_qwen3_model(perturbed, tgsa_cfg, layers=layers)
            with torch.no_grad():
                for index in layers:
                    module = perturbed.model.layers[index].self_attn
                    for target in module.conditioning:
                        for head in range(1, module.num_heads):
                            nn.init.normal_(module.conditioning[target][head].weight, std=std)
            with torch.no_grad():
                bad_logits = perturbed(input_ids=ids, attention_mask=mask).logits
            control = metrics(native_logits, bad_logits)
            control.update({"dtype": dtype_name, "attn_implementation": args.attn_impl,
                            "condition": f"positive_control_std={std}", "converted_layers": layers})
            results.append(control)
            print(f"  CONTROL std={std:<6g} kl_max={control['kl_max']:.3e} "
                  f"top1={control['top1_agreement']:.4f} "
                  f"|dlogit|max={control['logit_abs_diff_max']:.3e}", flush=True)
            del perturbed
        del native, retro

    payload = {"schema_version": "rineya.tgsa-real-conversion-damage.v1",
               "model_dir": str(args.model_dir), "prompts": len(PROMPTS),
               "sequence_length": int(ids.shape[1]), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
