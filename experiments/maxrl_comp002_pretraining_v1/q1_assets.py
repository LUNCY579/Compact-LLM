"""Bind one candidate R/S anchor's local asset bytes without a native model load.

This read-only step is earlier than tokenizer execution, model loading, source
competence or a training grant. It writes a create-only receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import sys

ASSETS = ("model.safetensors", "config.json", "generation_config.json",
          "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
EXPECTED_CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                   "num_hidden_layers": 28, "vocab_size": 151936,
                   "tie_word_embeddings": True}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key in {path.name}: {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def header(path: Path) -> dict:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("short safetensors preamble")
        n = struct.unpack("<Q", length_bytes)[0]
        if n <= 0 or n > (1 << 24):
            raise ValueError("safetensors header size out of range")
        info = json.loads(stream.read(n))
    info.pop("__metadata__", None)
    ranges = sorted(tuple(v["data_offsets"]) for v in info.values())
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] + 8 + n != path.stat().st_size:
        raise ValueError("safetensors extent mismatch")
    if any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("safetensors gaps/overlaps")
    dtypes = sorted({v["dtype"] for v in info.values()})
    if len(info) != 310 or dtypes != ["BF16"]:
        raise ValueError(f"unexpected model tensor inventory: {len(info)} {dtypes}")
    return {"tensors": len(info), "dtypes": dtypes, "header_bytes": n,
            "data_bytes": ranges[-1][1], "first_keys": sorted(info)[:5]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    if args.out.exists():
        raise SystemExit("receipt already exists (create-only)")
    if not model.is_dir():
        raise SystemExit("model path is not a directory")
    if any(model.glob("*.index.json")) or any(model.glob("adapter*")):
        raise SystemExit("sharded/adapted asset needs a separate binding")
    assets = {}
    for name in ASSETS:
        path = model / name
        if not path.is_file():
            raise SystemExit(f"missing required asset: {name}")
        assets[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    config = read_json(model / "config.json")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise SystemExit(f"unexpected config field {key}: {config.get(key)!r}")
    if config.get("auto_map") is not None:
        raise SystemExit("remote/custom code mapping present")
    generation = read_json(model / "generation_config.json")
    tokenizer = read_json(model / "tokenizer_config.json")
    template = tokenizer.get("chat_template")
    if not isinstance(template, str) or not template:
        raise SystemExit("missing string chat template")
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    receipt = {"experiment": "COMP-RS-001", "stage": "Q1_ASSET_CANDIDATE",
               "candidate_anchor": "Rineya-Supa-1.7.0-devfreeze",
               "model_path": str(model), "assets": assets,
               "model_header": header(model / "model.safetensors"),
               "architecture": {key: config[key] for key in EXPECTED_CONFIG},
               "config_dtype": config.get("torch_dtype"),
               "config_max_position_embeddings": config.get("max_position_embeddings"),
               "config_bos_eos": {"bos": config.get("bos_token_id"), "eos": config.get("eos_token_id")},
               "generation_token_ids": {key: generation.get(key) for key in
                                        ("bos_token_id", "eos_token_id", "pad_token_id")},
               "tokenizer_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
               "tokenizer_template_chars": len(template),
               "runtime_candidate": {"python_executable": sys.executable,
                                     "python": platform.python_version(), "platform": platform.platform(),
                                     "packages": packages},
               "native_model_loads": 0, "native_tokenizer_loads": 0,
               "native_qualified": False, "training_authorized": False,
               "note": "File/header binding only. Candidate selection, loader, tokenization, scope, mechanics and competence remain open."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"stage": receipt["stage"], "model_weight_sha256": assets["model.safetensors"]["sha256"],
                      "tensor_count": receipt["model_header"]["tensors"],
                      "native_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
