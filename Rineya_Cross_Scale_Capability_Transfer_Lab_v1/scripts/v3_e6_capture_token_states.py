"""v3 E6 (capture) — per-token residual states for the translator fit.

E5 measured similarity on one representation per prompt. E6 must *fit* a map
into a 1024-dimensional target space, and 59 prompts cannot do that. Amendment
v3.1 recorded the hazard in advance and requires all token positions plus a
prompt-disjoint holdout.

Writes one array per model of shape [total_tokens, depths, width] together with
the prompt index of every token, so the holdout can be taken by prompt.

Donors stream layer-by-layer. Targets are small enough to run through
transformers, and are captured the same way for symmetry.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json  # noqa: E402
from rineya_cross_scale.streaming_forward import (  # noqa: E402
    StreamingWeights,
    streaming_forward_many,
)

RELATIVE_DEPTHS = (0.00, 0.25, 0.50, 0.75, 1.00)
TOKENIZER = Path("C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
                 "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")


def capture_layers_for(num_layers: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(f * (num_layers - 1))) for f in RELATIVE_DEPTHS}))


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="v3 E6 per-token state capture.")
    parser.add_argument("--models", nargs="+",
                        default=["qwen4b", "qwen8b", "qwen06", "supa"])
    parser.add_argument("--destination", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_token_states")
    parser.add_argument("--dataset", type=Path,
                        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    rows = [json.loads(line) for line in args.dataset.read_text("utf-8").splitlines() if line.strip()]
    encoded = [tokenizer(render_prompt(tokenizer, r["prompt"]), return_tensors="pt",
                         truncation=True, max_length=args.max_context_tokens)["input_ids"][0]
               for r in rows]
    prompt_index = np.concatenate([np.full(int(ids.shape[0]), i, dtype=np.int32)
                                   for i, ids in enumerate(encoded)])

    sources = {
        "qwen4b": ("stream", json.loads(
            (LAB_ROOT / "cross_scale_evidence/02_architecture/qwen4b.json").read_text("utf-8"))["resolved_path"]),
        "qwen8b": ("stream", json.loads(
            (LAB_ROOT / "cross_scale_evidence/02_architecture/qwen8b.json").read_text("utf-8"))["resolved_path"]),
        "qwen06": ("transformers", str(TOKENIZER)),
        "supa": ("transformers",
                 "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                 "Rineya-Supa-A1.1-1.1.0-devfreeze"),
    }

    args.destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.models:
        out = args.destination / f"{name}.npz"
        manifest = args.destination / f"{name}.manifest.json"
        if out.exists() and manifest.exists():
            print(f"[skip] {name}")
            records.append(json.loads(manifest.read_text("utf-8")))
            continue
        mode, path = sources[name]
        started = time.perf_counter()

        if mode == "stream":
            with StreamingWeights(Path(path)) as weights:
                layers = capture_layers_for(int(weights.config["num_hidden_layers"]))
                width = int(weights.config["hidden_size"])
                print(f"[run ] {name} streaming, width {width}, layers {layers}", flush=True)
                with torch.inference_mode():
                    produced = streaming_forward_many(
                        weights, encoded, capture_layers=layers,
                        want_logits=False, pooling="all",
                        progress=lambda d, t: (
                            print(f"  [{name}] layer {d}/{t} "
                                  f"({time.perf_counter() - started:.0f}s)", flush=True)
                            if d % 6 == 0 else None),
                    )
            slots = [f"layer_{layer}" for layer in layers]
            blocks = [np.concatenate([p["captured"][s].numpy() for p in produced], axis=0)
                      for s in slots]
            del produced
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), dtype=torch.float32, trust_remote_code=False)
            model.eval()
            layers = capture_layers_for(int(model.config.num_hidden_layers))
            width = int(model.config.hidden_size)
            slots = [f"layer_{layer}" for layer in layers]
            print(f"[run ] {name} transformers, width {width}, layers {layers}", flush=True)
            per_slot: list[list[np.ndarray]] = [[] for _ in slots]
            with torch.inference_mode():
                for ids in encoded:
                    output = model(input_ids=ids.unsqueeze(0),
                                   output_hidden_states=True, use_cache=False,
                                   logits_to_keep=1)
                    for index, layer in enumerate(layers):
                        per_slot[index].append(
                            output.hidden_states[layer + 1][0].float().numpy())
                    del output
            blocks = [np.concatenate(chunks, axis=0) for chunks in per_slot]
            del model, per_slot
        gc.collect()

        stacked = np.stack(blocks, axis=1).astype(np.float32)
        del blocks
        np.savez(out, states=stacked, prompt_index=prompt_index)
        record = {
            "model": name, "mode": mode, "path": str(path),
            "hidden_size": width, "capture_layers": list(layers),
            "capture_slots": slots, "relative_depths": list(RELATIVE_DEPTHS),
            "tokens": int(stacked.shape[0]), "prompts": len(rows),
            "resources": {"wall_seconds": time.perf_counter() - started,
                          "models_resident": 0 if mode == "stream" else 1},
        }
        atomic_write_json(manifest, record)
        records.append(record)
        print(f"[done] {name} {stacked.shape} {record['resources']['wall_seconds']:.0f}s", flush=True)
        del stacked
        gc.collect()

    atomic_write_json(args.destination / "token_state_index.json", {
        "schema_version": "1.0",
        "experiment": "v3_e6_token_state_capture",
        "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
        "split": "calibration", "prompts_only": True,
        "total_tokens": int(prompt_index.shape[0]),
        "models": records,
    })
    print(f"captured {len(records)} models, {prompt_index.shape[0]} token positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
