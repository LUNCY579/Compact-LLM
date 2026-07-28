from __future__ import annotations

"""Create physically shallower decoder checkpoints for Phase 4B.

The script removes complete decoder blocks, updates layer-related configuration,
performs a smoke forward, saves the smaller checkpoint, and reloads it. It does
not claim that fewer layers improve speed or preserve quality; those claims are
made only after the common candidate evaluator runs.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import write_json
from scripts.common.model_introspection import (
    count_parameters,
    find_decoder_stack,
    replace_decoder_layers,
    update_config_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove decoder layers and save a smaller Phase 4B candidate."
    )
    parser.add_argument("--model", required=True, help="Accepted Phase 4 checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("drop-last", "keep-first", "uniform", "uniform-drop", "explicit"),
        default="drop-last",
        help=(
            "drop-last removes a suffix; uniform keeps a requested count; "
            "uniform-drop removes a requested count while preserving coverage."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help=(
            "Layers to drop for drop-last/uniform-drop, or layers to keep for "
            "keep-first/uniform."
        ),
    )
    parser.add_argument(
        "--keep-layers",
        default="",
        help="Comma-separated original layer IDs for explicit mode.",
    )
    parser.add_argument("--smoke-text", default="Say hello briefly.")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def choose_layers(total: int, mode: str, count: int, explicit: str) -> list[int]:
    """Return original layer indices retained by one depth-reduction policy."""

    if total < 1:
        raise ValueError("Model has no decoder layers.")

    if mode == "drop-last":
        keep_count = total - count
        if keep_count < 1:
            raise ValueError("Cannot drop every layer.")
        return list(range(keep_count))

    if mode == "keep-first":
        if not 1 <= count <= total:
            raise ValueError("keep-first count is out of range.")
        return list(range(count))

    if mode == "uniform-drop":
        # Convert a drop count into the existing uniform keep-count policy. This
        # gives drop-last candidates an equal-layer-count structural control.
        keep_count = total - count
        if keep_count < 1:
            raise ValueError("Cannot drop every layer.")
        return choose_layers(total, "uniform", keep_count, explicit)

    if mode == "uniform":
        if not 1 <= count <= total:
            raise ValueError("uniform keep count is out of range.")
        if count == 1:
            return [total // 2]
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        # Rounding should preserve cardinality for this construction, but fail
        # rather than silently saving an unexpected layer count.
        unique = sorted(set(indices))
        if len(unique) != count:
            raise RuntimeError("Uniform layer selection produced duplicate indices.")
        return unique

    indices = sorted({int(part.strip()) for part in explicit.split(",") if part.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= total:
        raise ValueError("Explicit layer list is empty or out of range.")
    return indices


def main() -> None:
    # Import large optional dependencies only when the executable is run. This
    # keeps unit tests for choose_layers lightweight.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.eval()

    stack = find_decoder_stack(model)
    original_layer_count = len(stack.layers)
    keep = choose_layers(original_layer_count, args.mode, args.count, args.keep_layers)
    before = count_parameters(model)
    new_stack = replace_decoder_layers(model, [stack.layers[index] for index in keep])

    # Cache implementations often store an original layer index on attention
    # modules. Reindex after removal so future KV caches use contiguous IDs.
    for new_index, layer in enumerate(new_stack.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_index
        attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = new_index
    update_config_field(model.config, "num_hidden_layers", len(new_stack.layers))

    # Catch shape/configuration failures before writing a checkpoint.
    encoded = tokenizer(args.smoke_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Depth-reduced smoke forward produced non-finite logits.")

    after = count_parameters(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Reloading is essential: an in-memory model may work even when the saved
    # configuration cannot reconstruct the changed tensor topology.
    reload_logits_shape = None
    if not args.skip_reload_check:
        reloaded = AutoModelForCausalLM.from_pretrained(
            args.output_dir, torch_dtype="auto", low_cpu_mem_usage=True
        )
        reloaded.eval()
        with torch.no_grad():
            reload_output = reloaded(**encoded)
        if not torch.isfinite(reload_output.logits).all():
            raise RuntimeError(
                "Reloaded depth-reduced checkpoint produced non-finite logits."
            )
        reload_logits_shape = list(reload_output.logits.shape)

    write_json(
        args.output_dir / "depth_reduction_manifest.json",
        {
            "source_model": args.model,
            "mode": args.mode,
            "requested_count": args.count,
            "original_layer_count": original_layer_count,
            "retained_original_layer_ids": keep,
            "new_layer_count": len(new_stack.layers),
            "parameters_before": before,
            "parameters_after": after,
            "parameter_reduction_fraction": 1 - after["total"] / before["total"],
            "reload_check_performed": not args.skip_reload_check,
            "reload_smoke_logits_shape": reload_logits_shape,
            "warning": (
                "Layer removal changes every downstream representation. Evaluate "
                "the untouched reduced checkpoint before optional recovery tuning."
            ),
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
