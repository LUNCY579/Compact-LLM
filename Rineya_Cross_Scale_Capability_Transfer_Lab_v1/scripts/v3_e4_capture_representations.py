"""v3 E4 (capture) — sparse representation capture for the functional atlas.

Runs one 0.6B-shaped checkpoint at a time over the frozen calibration prompts and
writes a compact per-model summary to disk: final-prompt-token hidden states at
matched relative depths, and the final-token logit vector. The model is released
before the next one is loaded, so peak memory is one model, never two.

Prompts only. No calibration answer, scorer, or evaluation bank is read.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, sha256_file  # noqa: E402
from rineya_cross_scale.checkpoint import checkpoint_fingerprint  # noqa: E402

# Frozen by amendment v3: relative depths 0.00/0.25/0.50/0.75/1.00 of a 28-layer
# stack, plus the embedding output. Index 0 of `hidden_states` is the embedding.
CAPTURE_LAYERS = (0, 7, 14, 21, 27)
SEED = 17


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Identical to the frozen evaluation runner's prompt rendering."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def peak_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def capture_model(
    *,
    name: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    destination: Path,
    threads: int,
    max_context_tokens: int,
    pooling: str = "last",
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    # Deliberately trust_remote_code=False: the atlas measures weight-space
    # behaviour, so Supa's non-weight abstention logit offset must not apply.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float32, trust_remote_code=False
    )
    model.eval()
    load_seconds = time.perf_counter() - started

    hidden = np.zeros((len(rows), 1 + len(CAPTURE_LAYERS), model.config.hidden_size), dtype=np.float32)
    logits = np.zeros((len(rows), model.config.vocab_size), dtype=np.float16)
    token_counts = np.zeros(len(rows), dtype=np.int32)

    inference_started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            text = render_prompt(tokenizer, row["prompt"])
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_context_tokens
            )
            token_counts[index] = int(encoded["input_ids"].shape[1])
            output = model(
                **encoded, output_hidden_states=True, use_cache=False, logits_to_keep=1
            )
            states = output.hidden_states

            def reduce(tensor: torch.Tensor) -> np.ndarray:
                # "last" is the final prompt token. "mean" pools every prompt
                # position, which avoids the final-token slot being dominated by
                # the chat template's trailing token.
                view = tensor[0].float()
                return (view.mean(dim=0) if pooling == "mean" else view[-1]).numpy()

            hidden[index, 0] = reduce(states[0])
            for slot, layer in enumerate(CAPTURE_LAYERS, start=1):
                hidden[index, slot] = reduce(states[layer + 1])
            logits[index] = output.logits[0, -1].float().numpy().astype(np.float16)
            del output, states, encoded
            if index % 10 == 9:
                gc.collect()
    inference_seconds = time.perf_counter() - inference_started

    fingerprint = checkpoint_fingerprint(model_path, hash_weights=True)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / f"{name}.npz",
        hidden=hidden,
        logits=logits,
        token_counts=token_counts,
    )
    record = {
        "model": name,
        "model_path": str(model_path),
        "model_digest": fingerprint["digest"],
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "capture_layers": list(CAPTURE_LAYERS),
        "pooling": pooling,
        "capture_slots": ["embedding"] + [f"layer_{layer}" for layer in CAPTURE_LAYERS],
        "rows": len(rows),
        "total_prompt_tokens": int(token_counts.sum()),
        "resources": {
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "wall_seconds": time.perf_counter() - started,
            "rss_bytes_after": peak_rss_bytes(),
            "threads": threads,
            "device": "cpu",
            "dtype": "float32",
        },
    }
    atomic_write_json(destination / f"{name}.manifest.json", record)

    del model, tokenizer, hidden, logits
    gc.collect()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E4 representation capture.")
    parser.add_argument("--models", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "protocol" / "atlas_models.json")
    parser.add_argument("--dataset", type=Path,
                        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl")
    parser.add_argument("--destination", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_activations")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    args = parser.parse_args()

    dataset_sha = sha256_file(args.dataset)
    rows = [json.loads(line) for line in args.dataset.read_text("utf-8").splitlines() if line.strip()]
    # Prompts only: drop every scoring field before the model ever sees a row.
    rows = [{"case_id": r["case_id"], "family": r["family"], "prompt": r["prompt"]} for r in rows]

    registry = json.loads(args.models.read_text("utf-8"))
    records = []
    for entry in registry["models"]:
        if args.only and entry["name"] != args.only:
            continue
        manifest_path = args.destination / f"{entry['name']}.manifest.json"
        if manifest_path.exists() and (args.destination / f"{entry['name']}.npz").exists():
            print(f"[skip] {entry['name']} already captured")
            records.append(json.loads(manifest_path.read_text("utf-8")))
            continue
        print(f"[run ] {entry['name']} <- {entry['path']}", flush=True)
        record = capture_model(
            name=entry["name"],
            model_path=Path(entry["path"]),
            rows=rows,
            destination=args.destination,
            threads=args.threads,
            max_context_tokens=args.max_context_tokens,
            pooling=args.pooling,
        )
        print(f"[done] {entry['name']} {record['resources']['wall_seconds']:.1f}s", flush=True)
        records.append(record)

    atomic_write_json(
        args.destination / "capture_index.json",
        {
            "schema_version": "1.0",
            "experiment": "v3_e4_representation_capture",
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_sha,
            "split": "calibration",
            "prompts_only": True,
            "seed": SEED,
            "capture_layers": list(CAPTURE_LAYERS),
            "pooling": args.pooling,
            "models": records,
        },
    )
    print(f"captured {len(records)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
