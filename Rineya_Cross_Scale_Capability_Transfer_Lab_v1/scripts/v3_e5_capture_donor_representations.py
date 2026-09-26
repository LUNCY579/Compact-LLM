"""v3 E5 (capture) — donor-side representations without loading a donor.

Runs Qwen3-4B and Qwen3-8B over the frozen calibration prompts using the
streaming forward, which never holds a model in memory. Captures the residual
state at the donor's matched *relative* depths and the final-token logits over
the shared vocabulary.

Gated by `v3_e5_validate_streaming_forward.py`: this script refuses to run
unless that gate recorded PASS, and it verifies each donor's weight-shard
SHA-256 against the values recorded in `02_architecture/` before reading them.

Resumable per (model, pooling): a completed capture is skipped on re-run. The
loop is inverted (layer outermost) so the weights are read exactly once.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from rineya_cross_scale.streaming_forward import (  # noqa: E402
    StreamingWeights,
    streaming_forward_many,
)

SEED = 17
# Matched relative depths: fractions 0.00/0.25/0.50/0.75/1.00 of the stack.
RELATIVE_DEPTHS = (0.00, 0.25, 0.50, 0.75, 1.00)


def capture_layers_for(num_layers: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(f * (num_layers - 1))) for f in RELATIVE_DEPTHS}))


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


def verify_provenance(architecture: Path, name: str) -> dict[str, Any]:
    record = json.loads((architecture / f"{name}.json").read_text("utf-8"))
    snapshot = Path(record["resolved_path"])
    files = record["weights"] if isinstance(record["weights"], list) else record["weights"].get("files", [])
    checked = []
    for entry in files:
        path = snapshot / entry["name"]
        if not path.is_file():
            raise SystemExit(f"{name}: missing weight shard {entry['name']}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise SystemExit(f"{name}: SHA-256 mismatch on {entry['name']} — provenance failure")
        checked.append({"name": entry["name"], "sha256": entry["sha256"]})
    return {"snapshot": str(snapshot), "revision": record["revision"], "shards": checked}


def peak_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E5 donor representation capture.")
    parser.add_argument("--donors", nargs="+", default=["qwen4b", "qwen8b"])
    parser.add_argument("--architecture", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "02_architecture")
    parser.add_argument("--dataset", type=Path,
                        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl")
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
                                     "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"))
    parser.add_argument("--destination", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_donor_activations")
    parser.add_argument("--gate", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e5_cross_scale_atlas" / "gate0_streaming_forward_validation.json")
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text("utf-8"))
    if gate.get("gate") != "PASS":
        raise SystemExit(
            f"Gate 0 is {gate.get('gate')}; amendment v3.1 forbids donor measurement "
            "from an unvalidated streaming forward.")

    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    dataset_sha = sha256_file(args.dataset)
    rows = [json.loads(line) for line in args.dataset.read_text("utf-8").splitlines() if line.strip()]
    rows = [{"case_id": r["case_id"], "family": r["family"], "prompt": r["prompt"]} for r in rows]

    args.destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.donors:
        tag = f"{name}_{args.pooling}"
        final_path = args.destination / f"{tag}.npz"
        manifest_path = args.destination / f"{tag}.manifest.json"
        if final_path.exists() and manifest_path.exists():
            print(f"[skip] {tag} already captured")
            records.append(json.loads(manifest_path.read_text("utf-8")))
            continue

        provenance = verify_provenance(args.architecture, name)
        print(f"[prov] {name} {len(provenance['shards'])} shards verified against 02_architecture",
              flush=True)

        started = time.perf_counter()
        with StreamingWeights(Path(provenance["snapshot"])) as weights:
            layers = capture_layers_for(int(weights.config["num_hidden_layers"]))
            width = int(weights.config["hidden_size"])
            vocab = int(weights.config["vocab_size"])
            slots = ["embedding"] + [f"layer_{layer}" for layer in layers]
            print(f"[plan] {name}: width {width}, layers {weights.config['num_hidden_layers']}, "
                  f"capture {layers}", flush=True)

            hidden = np.zeros((len(rows), len(slots), width), dtype=np.float32)
            logits = np.zeros((len(rows), vocab), dtype=np.float16)
            tokens = np.zeros(len(rows), dtype=np.int32)

            encoded = []
            for row in rows:
                text = render_prompt(tokenizer, row["prompt"])
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=args.max_context_tokens)["input_ids"][0]
                encoded.append(ids)
            tokens[:] = [int(ids.shape[0]) for ids in encoded]
            print(f"[plan] {sum(tokens)} prompt tokens; one pass over the weights", flush=True)

            def report(done: int, total: int) -> None:
                if done % 4 == 0 or done == total:
                    print(f"  [{tag}] layer {done}/{total} "
                          f"({time.perf_counter() - started:.0f}s, "
                          f"rss {(peak_rss_bytes() or 0) / 2**30:.2f} GiB)", flush=True)

            with torch.inference_mode():
                produced = streaming_forward_many(
                    weights, encoded, capture_layers=layers,
                    want_logits=True, pooling=args.pooling, progress=report,
                )
            for index, got in enumerate(produced):
                for slot_index, slot in enumerate(slots):
                    hidden[index, slot_index] = got["captured"][slot].numpy()
                logits[index] = got["logits"].numpy()[:vocab].astype(np.float16)
            del produced, encoded
            gc.collect()

        np.savez_compressed(final_path, hidden=hidden, logits=logits, token_counts=tokens)
        record = {
            "model": name,
            "role": "donor",
            "pooling": args.pooling,
            "provenance": provenance,
            "hidden_size": width,
            "vocab_size": vocab,
            "capture_layers": list(layers),
            "capture_slots": slots,
            "relative_depths": list(RELATIVE_DEPTHS),
            "rows": len(rows),
            "total_prompt_tokens": int(tokens.sum()),
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "rss_bytes_after": peak_rss_bytes(),
                "models_resident": 0,
                "threads": args.threads,
                "device": "cpu",
                "arithmetic_dtype": "float32",
                "method": "streaming layer-by-layer forward over memory-mapped safetensors",
            },
        }
        atomic_write_json(manifest_path, record)
        records.append(record)
        print(f"[done] {tag} {record['resources']['wall_seconds']:.0f}s", flush=True)
        del hidden, logits, tokens
        gc.collect()

    atomic_write_json(
        args.destination / f"donor_capture_index_{args.pooling}.json",
        {
            "schema_version": "1.0",
            "experiment": "v3_e5_donor_representation_capture",
            "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
            "gate0": gate["gate"],
            "gate0_worst_relative_error": gate["worst_relative_error"],
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_sha,
            "split": "calibration",
            "prompts_only": True,
            "seed": SEED,
            "pooling": args.pooling,
            "models": records,
        },
    )
    print(f"captured {len(records)} donors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
