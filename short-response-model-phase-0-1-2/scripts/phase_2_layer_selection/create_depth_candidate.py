from __future__ import annotations

"""Create one physically shallower Phase 2B candidate.

The caller supplies a *current* layer index. The script preserves the original
layer ID through progressive deletions, saves a reloadable checkpoint, and never
modifies the source model directory.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.depth_reduction import reduce_model_depth
from scripts.common.experiment_utils import read_json, sha256_tree, write_json
from scripts.common.model_introspection import find_decoder_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop one current decoder layer for Phase 2B.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--drop-current-layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-text", default="Reply briefly: hello")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def lineage_for_model(model_path: Path, current_count: int) -> list[int]:
    """Resolve current-position to original-layer mapping from prior manifests."""

    candidates = (
        model_path / "depth_reduction_manifest.json",
        model_path / "materialization_manifest.json",
        model_path / "phase_2b_stage_manifest.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path)
        ids = payload.get("retained_original_layer_ids")
        if isinstance(ids, list) and len(ids) == current_count:
            return [int(value) for value in ids]
    return list(range(current_count))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    stack = find_decoder_stack(model)
    total = len(stack.layers)
    drop = int(args.drop_current_layer)
    if not 0 <= drop < total:
        raise ValueError(f"drop-current-layer {drop} is outside [0, {total - 1}].")
    if total <= 2:
        raise ValueError("Refusing to reduce a model below two decoder layers.")

    current_original_ids = lineage_for_model(Path(args.model), total)
    keep = [index for index in range(total) if index != drop]
    reduction = reduce_model_depth(
        model,
        retained_current_layer_ids=keep,
        current_original_layer_ids=current_original_ids,
    )

    encoded = tokenizer(args.smoke_text, return_tensors="pt")
    with torch.inference_mode():
        output = model(**encoded)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Reduced model produced non-finite logits.")

    # A failed save can leave an incomplete checkpoint directory. The parent
    # model is immutable, so clean only the candidate output before rewriting.
    source_path = Path(args.model)
    if source_path.exists() and source_path.resolve() == args.output_dir.resolve():
        raise ValueError("Refusing to overwrite the source model in place.")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    if not args.skip_reload_check:
        reloaded = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if len(find_decoder_stack(reloaded).layers) != total - 1:
            raise RuntimeError("Reloaded candidate does not preserve the reduced layer count.")

    dropped_original_id = current_original_ids[drop]
    write_json(
        args.output_dir / "depth_reduction_manifest.json",
        {
            "source_model": args.model,
            "source_fingerprint": sha256_tree(Path(args.model)),
            "drop_current_layer_id": drop,
            "dropped_original_layer_ids": [dropped_original_id],
            "retained_original_layer_ids": reduction.retained_original_layer_ids,
            "original_current_layer_count": total,
            "new_layer_count": total - 1,
            "parameters_before": reduction.parameters_before,
            "parameters_after": reduction.parameters_after,
            "parameter_reduction_fraction_from_parent": reduction.parameter_reduction_fraction,
            "reload_check_performed": not args.skip_reload_check,
            "comment": (
                "This is a reversible branch. The source checkpoint is unchanged; "
                "only an explicitly selected frontier stage should be used downstream."
            ),
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
