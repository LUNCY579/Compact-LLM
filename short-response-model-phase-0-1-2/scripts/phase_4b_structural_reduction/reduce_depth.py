# Purpose: Creates physical depth-reduction candidates and verifies saved checkpoint reloadability.
from __future__ import annotations

"""Create physically shallower decoder checkpoints for Phase 4B.

The implementation delegates the mutation and cache reindexing rules to
``scripts.common.depth_reduction`` so Phase 2B and Phase 4B cannot drift.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.depth_reduction import choose_layers, reduce_model_depth
from scripts.common.experiment_utils import write_json
from scripts.common.model_introspection import find_decoder_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove decoder layers and save a smaller Phase 4B candidate.")
    parser.add_argument("--model", required=True, help="Accepted Phase 4 checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("drop-last", "keep-first", "uniform", "uniform-drop", "explicit"), default="drop-last")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--keep-layers", default="")
    parser.add_argument("--smoke-text", default="Say hello briefly.")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", low_cpu_mem_usage=True)
    model.eval()

    stack = find_decoder_stack(model)
    keep = choose_layers(len(stack.layers), args.mode, args.count, args.keep_layers)
    reduction = reduce_model_depth(model, retained_current_layer_ids=keep)

    encoded = tokenizer(args.smoke_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Depth-reduced smoke forward produced non-finite logits.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    reload_logits_shape = None
    if not args.skip_reload_check:
        reloaded = AutoModelForCausalLM.from_pretrained(args.output_dir, torch_dtype="auto", low_cpu_mem_usage=True)
        reloaded.eval()
        with torch.no_grad():
            reload_output = reloaded(**encoded)
        if not torch.isfinite(reload_output.logits).all():
            raise RuntimeError("Reloaded depth-reduced checkpoint produced non-finite logits.")
        reload_logits_shape = list(reload_output.logits.shape)

    write_json(args.output_dir / "depth_reduction_manifest.json", {
        "source_model": args.model,
        "mode": args.mode,
        "requested_count": args.count,
        "original_layer_count": reduction.original_layer_count,
        "retained_original_layer_ids": reduction.retained_original_layer_ids,
        "dropped_original_layer_ids": reduction.dropped_original_layer_ids,
        "new_layer_count": len(reduction.retained_original_layer_ids),
        "parameters_before": reduction.parameters_before,
        "parameters_after": reduction.parameters_after,
        "parameter_reduction_fraction": reduction.parameter_reduction_fraction,
        "reload_check_performed": not args.skip_reload_check,
        "reload_smoke_logits_shape": reload_logits_shape,
        "warning": "Layer removal changes downstream representations. Evaluate before optional recovery tuning.",
    })
    print(args.output_dir)


if __name__ == "__main__":
    main()
