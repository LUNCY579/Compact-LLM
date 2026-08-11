from __future__ import annotations

"""Merge one LoRA adapter onto a base checkpoint entirely on CPU."""

import argparse
import json
import shutil
import uuid
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)

    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tok.save_pretrained(staging)
        manifest = {
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "device": "cpu",
            "formula": "W = W_base + Delta_adapter",
        }
        (staging / "materialize_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.replace(args.output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
