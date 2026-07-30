from __future__ import annotations

"""Materialize a base checkpoint plus optional PEFT adapter as one full model.

Structural deletion must operate on real model weights rather than a dangling
base-model ID plus adapter. This utility merges the adapter in higher precision,
saves a reloadable checkpoint, and records provenance without changing the
source artifacts.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import sha256_tree, write_json
from scripts.common.model_introspection import count_parameters, find_decoder_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a full checkpoint from a model and optional PEFT adapter.")
    parser.add_argument("--model-id", required=True, help="Base/local model ID or path.")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-label", default="materialized_model")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None  # type: ignore[assignment]

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    adapter_fingerprint = None
    if args.adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("Install peft before materializing an adapter.")
        if not args.adapter_path.exists():
            raise FileNotFoundError(args.adapter_path)
        adapter_fingerprint = sha256_tree(args.adapter_path)
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
        # safe_merge catches non-finite or obviously invalid adapter deltas.
        model = model.merge_and_unload(safe_merge=True)

    model.eval()
    stack = find_decoder_stack(model)
    parameters = count_parameters(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    if not args.skip_reload_check:
        reloaded = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if len(find_decoder_stack(reloaded).layers) != len(stack.layers):
            raise RuntimeError("Reloaded materialized model has a different layer count.")

    write_json(
        args.output_dir / "materialization_manifest.json",
        {
            "source_label": args.source_label,
            "model_id": args.model_id,
            "model_fingerprint": sha256_tree(Path(args.model_id)),
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "adapter_fingerprint": adapter_fingerprint,
            "output_model": str(args.output_dir),
            "output_fingerprint": sha256_tree(args.output_dir),
            "num_hidden_layers": len(stack.layers),
            "retained_original_layer_ids": list(range(len(stack.layers))),
            "parameters": parameters,
            "reload_check_performed": not args.skip_reload_check,
            "note": "The source model and adapter remain unchanged and available for rollback.",
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
