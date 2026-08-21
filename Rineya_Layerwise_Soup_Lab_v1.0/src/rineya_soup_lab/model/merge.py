from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from ..errors import CompatibilityError
from ..utils import (
    environment_manifest,
    require_non_overlapping_path,
    sha256_file,
    utc_timestamp,
    write_json,
)
from .checkpoint import (
    CheckpointReader,
    StreamingSafetensorsWriter,
    checkpoint_fingerprint,
    checkpoint_fingerprint_digest,
    is_weight_filename,
    parse_size_gb,
)
from .compatibility import TIED_EMBED_KEY, TIED_LM_HEAD_KEY, can_alias_tied_lm_head
from .layer_map import MergeRecipe, ParameterClassifier


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _output_dtype(anchor: torch.Tensor, setting: str) -> torch.dtype:
    if setting == "preserve_anchor":
        return anchor.dtype
    if setting not in DTYPE_MAP:
        raise ValueError(f"Unsupported output dtype: {setting}")
    return DTYPE_MAP[setting]


def _copy_checkpoint_metadata(anchor_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    excluded_files = {
        "merge_recipe.json",
        "merge_manifest.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "scaler.pt",
        "rng_state.pth",
    }
    excluded_directories = {"artifacts", "runs", "logs", "wandb"}
    for item in anchor_path.iterdir():
        if item.is_dir():
            lowered = item.name.lower()
            if (
                item.name.startswith(".")
                or lowered in excluded_directories
                or lowered.startswith("checkpoint-")
            ):
                continue
            shutil.copytree(item, destination / item.name, dirs_exist_ok=True)
            continue
        if is_weight_filename(item.name) or item.name in excluded_files:
            continue
        # Avoid silently copying multi-gigabyte training artifacts with unknown names.
        if item.stat().st_size > 256 * 1024 * 1024:
            continue
        shutil.copy2(item, destination / item.name)



def _read_maybe_tied(reader: CheckpointReader, key: str, *, alias_tied_lm_head: bool):
    if key in reader.keys:
        return reader.get_tensor(key)
    if alias_tied_lm_head and key == TIED_LM_HEAD_KEY and TIED_EMBED_KEY in reader.keys:
        return reader.get_tensor(TIED_EMBED_KEY)
    raise CompatibilityError(f"Missing tensor {key}")


def merge_checkpoints(
    anchor_path: Path,
    greedy_path: Path,
    destination: Path,
    recipe: MergeRecipe,
    *,
    layer_key_regex: str,
    accumulation_dtype: str = "float32",
    output_dtype: str = "preserve_anchor",
    max_shard_size_gb: float = 1.5,
    require_non_floating_equal: bool = True,
    copy_non_floating_from: str = "anchor",
    overwrite: bool = False,
    source_fingerprints: Mapping[str, Any] | None = None,
    allow_tied_lm_head_alias: bool = False,
) -> dict[str, Any]:
    if recipe.kind == "endpoint":
        raise ValueError("Endpoint recipes do not need materialization.")
    if accumulation_dtype != "float32":
        raise ValueError("Version 1.0 requires float32 accumulation for numerical safety.")
    anchor_path = anchor_path.expanduser().resolve()
    greedy_path = greedy_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    sources = (anchor_path, greedy_path)
    require_non_overlapping_path(
        destination, sources, label="merge destination (overlaps source checkpoint)"
    )
    temporary = destination.with_name(f".{destination.name}.building")
    require_non_overlapping_path(
        temporary, sources, label="merge temporary directory (overlaps source checkpoint)"
    )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if temporary.exists():
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink()
    temporary.mkdir(parents=True)
    classifier = ParameterClassifier(layer_key_regex)
    writer = StreamingSafetensorsWriter(temporary, parse_size_gb(max_shard_size_gb))
    tensor_manifest: list[dict[str, Any]] = []
    try:
        _copy_checkpoint_metadata(anchor_path, temporary)
        with CheckpointReader(anchor_path) as anchor, CheckpointReader(greedy_path) as greedy:
            alias_tied_lm_head = bool(allow_tied_lm_head_alias) and can_alias_tied_lm_head(
                set(anchor.keys),
                set(greedy.keys),
                anchor.config(),
                greedy.config(),
            )
            if set(anchor.keys) != set(greedy.keys) and not alias_tied_lm_head:
                raise CompatibilityError("Checkpoint tensor keys changed after preflight.")
            for index, key in enumerate(anchor.keys, 1):
                a = _read_maybe_tied(anchor, key, alias_tied_lm_head=alias_tied_lm_head)
                g = _read_maybe_tied(greedy, key, alias_tied_lm_head=alias_tied_lm_head)
                if tuple(a.shape) != tuple(g.shape):
                    raise CompatibilityError(f"Shape mismatch for {key}: {tuple(a.shape)} vs {tuple(g.shape)}")
                location = classifier.classify(key)
                alpha = recipe.alpha_for(location)
                if a.is_floating_point() and g.is_floating_point():
                    merged = a.to(torch.float32).mul_(1.0 - alpha).add_(g.to(torch.float32), alpha=alpha)
                    merged = merged.to(_output_dtype(a, output_dtype))
                else:
                    if require_non_floating_equal and not a.equal(g):
                        raise CompatibilityError(f"Non-floating tensor differs: {key}")
                    merged = a if copy_non_floating_from == "anchor" else g
                writer.add(key, merged)
                tensor_manifest.append(
                    {
                        "key": key,
                        "group": location.group,
                        "layer_index": location.layer_index,
                        "greedy_alpha": alpha,
                        "shape": list(a.shape),
                        "anchor_dtype": str(a.dtype),
                        "greedy_dtype": str(g.dtype),
                        "output_dtype": str(merged.dtype),
                    }
                )
                del a, g, merged
                if index % 50 == 0:
                    gc.collect()
        output_index = writer.finalize()
        recipe_payload = recipe.to_dict()
        write_json(temporary / "merge_recipe.json", recipe_payload)
        output_checkpoint_fingerprint = checkpoint_fingerprint(temporary)
        output_checkpoint_fingerprint["path"] = str(destination)
        output_checkpoint_digest = checkpoint_fingerprint_digest(output_checkpoint_fingerprint)
        manifest = {
            "schema_version": "1.0",
            "created_at": utc_timestamp(),
            "candidate_id": recipe.id,
            "recipe": recipe_payload,
            "source_fingerprints": dict(source_fingerprints or {}),
            "merge_precision": {
                "accumulation_dtype": accumulation_dtype,
                "output_dtype": output_dtype,
            },
            "output_index": output_index,
            "output_checkpoint_fingerprint": output_checkpoint_fingerprint,
            "output_checkpoint_digest": output_checkpoint_digest,
            "tensor_count": len(tensor_manifest),
            "tensor_groups": _group_counts(tensor_manifest),
            "environment": environment_manifest(),
        }
        write_json(temporary / "merge_manifest.json", manifest)
        write_json(temporary / "tensor_coefficients.json", tensor_manifest)
        temporary.replace(destination)
        manifest["output_path"] = str(destination)
        manifest["merge_manifest_sha256"] = sha256_file(destination / "merge_manifest.json")
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        group = str(row["group"])
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))
