from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

from consensus_lab.io import read_json, sha256_file
from consensus_lab.paths import EXCLUDED_DIRNAMES, LabPaths


WEIGHT_NAMES = {"model.safetensors", "pytorch_model.bin"}
V22_RUN = Path("artifacts/phase_9_capability_composition_v22_lr1e4_s1200")
V23_RUN = Path("artifacts/phase_9_residual_capability_v23")


V22_INTERPRETATION: dict[str, tuple[list[str], list[str]]] = {
    "frozen_anchor": (
        ["Strong Real-RAG procedure preservation", "High evidence faithfulness", "Immutable v2.2 baseline"],
        ["Zero v2.2 multi-domain DEV correctness", "Near-zero v2.1.2 direct reasoning"],
    ),
    "greedy_soup": (
        ["Selected under both official DEV policies", "Perfect DEV abstention", "Improved supported answering"],
        ["Gain is mostly abstention behavior", "Zero v2.3 answerable reasoning", "Dense merge is not an adapter expert"],
    ),
    "track_a_broad": (
        ["Highest v2.2 multi-domain DEV score", "Broad reasoning diversity"],
        ["Catastrophic procedure and grounding regressions"],
    ),
    "track_b_anchored": (
        ["Broad reasoning gain with strong procedure preservation", "KL-anchored to frozen student"],
        ["Still loses faithfulness and supported-answer quality"],
    ),
    "specialist_general": (["Strongest Track C direct-reasoning specialist"], ["Severe protected-capability regressions"]),
    "specialist_grounding": (["Preserves original procedure", "Grounding-specialized training"], ["Weak direct reasoning"]),
    "specialist_procedure": (["Only v2.2 candidate with nonzero procedural challenge score"], ["Weak direct reasoning and original procedure"]),
    "specialist_abstention": (["Perfect DEV abstention", "Accepted at half-dose in D_greedy"], ["Over-abstention risk"]),
    "specialist_social": (["Best social/preservation behavior among specialists"], ["Essentially zero direct reasoning"]),
}


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _model_weight_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name in WEIGHT_NAMES or path.name.startswith("model-") and path.suffix == ".safetensors":
            files.append(path)
        elif path.name.startswith("pytorch_model-") and path.suffix == ".bin":
            files.append(path)
    return sorted(files)


def _safetensor_statistics(files: list[Path]) -> tuple[int | None, dict[str, int]]:
    if not files or any(path.suffix != ".safetensors" for path in files):
        return None, {}
    parameters = 0
    dtypes: Counter[str] = Counter()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                count = math.prod(tensor.get_shape())
                parameters += count
                dtypes[str(tensor.get_dtype())] += count
    return parameters, dict(sorted(dtypes.items()))


def _tokenizer_fingerprint(directory: Path) -> str | None:
    tokenizer = directory / "tokenizer.json"
    return sha256_file(tokenizer) if tokenizer.is_file() else None


def _candidate_status(relative: str) -> str:
    lowered = relative.casefold()
    if "checkpoint-" in lowered or "dynamics_snapshots" in lowered or "\\trials\\" in lowered:
        return "intermediate_or_trial"
    if lowered.endswith("initialization") or "reference\\full_model" in lowered:
        return "initialization_or_reference"
    return "final_or_named_candidate"


def discover_models(paths: LabPaths) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for config_path in sorted(paths.official_root.rglob("config.json")):
        directory = config_path.parent
        weights = _model_weight_files(directory)
        if not weights:
            continue
        config = read_json(config_path)
        parameter_count, dtype_counts = _safetensor_statistics(weights)
        relative = _relative(directory, paths.official_root)
        name = directory.name
        if relative.endswith(str(V22_RUN / "frozen_anchor").replace("/", "\\")):
            name = "frozen_anchor"
        elif relative.endswith(str(V22_RUN / "track_d_greedy_soup/final_model").replace("/", "\\")):
            name = "greedy_soup"
        strengths, weaknesses = V22_INTERPRETATION.get(name, ([], []))
        models.append(
            {
                "kind": "dense_model",
                "name": name,
                "path": str(directory.resolve()),
                "relative_path": relative,
                "candidate_status": _candidate_status(relative),
                "architecture": (config.get("architectures") or [None])[0],
                "model_type": config.get("model_type"),
                "tokenizer": config.get("tokenizer_class") or "local tokenizer.json",
                "tokenizer_sha256": _tokenizer_fingerprint(directory),
                "vocabulary_size": config.get("vocab_size"),
                "hidden_size": config.get("hidden_size"),
                "intermediate_size": config.get("intermediate_size"),
                "layer_count": config.get("num_hidden_layers"),
                "parameter_count": parameter_count,
                "dtype_parameter_counts": dtype_counts,
                "weight_bytes": sum(path.stat().st_size for path in weights),
                "weight_files": [_relative(path, paths.official_root) for path in weights],
                "merged": True,
                "adapter_type": None,
                "source_phase": relative.split("\\", 2)[1] if relative.startswith("artifacts\\") else "unknown",
                "known_strengths": strengths,
                "known_weaknesses": weaknesses,
                "safe_for_experiment": True,
                "direct_vote_compatibility_key": {
                    "model_type": config.get("model_type"),
                    "vocab_size": config.get("vocab_size"),
                    "hidden_size": config.get("hidden_size"),
                    "layers": config.get("num_hidden_layers"),
                    "tokenizer_sha256": _tokenizer_fingerprint(directory),
                },
            }
        )
    return models


def _v22_candidate_metrics(paths: LabPaths) -> dict[str, dict[str, Any]]:
    selection_path = paths.official_root / V22_RUN / "capability_composition_v22_selection.json"
    if not selection_path.is_file():
        return {}
    return {row["name"]: row for row in read_json(selection_path).get("candidates", [])}


def _adapter_name(relative: str) -> str:
    mapping = {
        "track_a_broad_reasoning": "track_a_broad",
        "track_b_anchored_reasoning": "track_b_anchored",
        "track_c_general": "specialist_general",
        "track_c_grounding": "specialist_grounding",
        "track_c_procedure": "specialist_procedure",
        "track_c_abstention": "specialist_abstention",
        "track_c_social": "specialist_social",
    }
    for key, value in mapping.items():
        if key in relative.replace("\\", "/"):
            return value
    parent = Path(relative).parent.name
    return parent or Path(relative).name


def discover_adapters(paths: LabPaths) -> list[dict[str, Any]]:
    metrics = _v22_candidate_metrics(paths)
    adapters: list[dict[str, Any]] = []
    for config_path in sorted(paths.official_root.rglob("adapter_config.json")):
        directory = config_path.parent
        config = read_json(config_path)
        relative = _relative(directory, paths.official_root)
        declared_base = str(config.get("base_model_name_or_path") or "")
        resolved_base = paths.resolve_declared_source(declared_base) if declared_base else None
        tokenizer_sha = _tokenizer_fingerprint(directory)
        model_file = directory / "adapter_model.safetensors"
        parameter_count, dtypes = _safetensor_statistics([model_file] if model_file.is_file() else [])
        name = _adapter_name(relative)
        strengths, weaknesses = V22_INTERPRETATION.get(name, ([], []))
        base_available = bool(resolved_base and resolved_base.exists())
        direct_group = None
        if base_available:
            direct_group = f"{resolved_base}|{tokenizer_sha or 'unknown-tokenizer'}"
        manifest = directory.parent / "training_manifest.json"
        manifest_data = read_json(manifest) if manifest.is_file() else {}
        candidate_metrics = metrics.get(name)
        adapters.append(
            {
                "kind": "adapter",
                "name": name,
                "path": str(directory.resolve()),
                "relative_path": relative,
                "candidate_status": _candidate_status(relative),
                "model_architecture": "Qwen3ForCausalLM" if tokenizer_sha else None,
                "tokenizer": "local tokenizer.json" if tokenizer_sha else "inherited from base",
                "tokenizer_sha256": tokenizer_sha,
                "vocabulary_size": 151936 if tokenizer_sha else None,
                "hidden_size": None,
                "layer_count": None,
                "parameter_count": parameter_count,
                "dtype_parameter_counts": dtypes,
                "merged": False,
                "adapter_type": config.get("peft_type"),
                "adapter_rank": config.get("r"),
                "adapter_alpha": config.get("lora_alpha"),
                "adapter_method": "DoRA" if config.get("use_dora") else "LoRA",
                "adapter_target_modules": sorted(config.get("target_modules") or []),
                "adapter_layer_indices": manifest_data.get("lora_layer_indices"),
                "base_checkpoint_provenance": declared_base,
                "resolved_base_checkpoint": str(resolved_base) if resolved_base else None,
                "training_stage": manifest_data.get("pipeline_revision") or relative,
                "source_phase": relative.split("\\", 2)[1] if relative.startswith("artifacts\\") else "unknown",
                "relevant_metrics": candidate_metrics,
                "known_strengths": strengths,
                "known_weaknesses": weaknesses,
                "compatible_with_direct_voting": base_available and bool(tokenizer_sha),
                "compatibility_limitation": None if base_available else "Declared base checkpoint is unavailable",
                "direct_vote_group": direct_group,
                "safe_for_experiment": base_available and model_file.is_file(),
                "artifact_bytes": sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
                "artifact_sha256_from_manifest": manifest_data.get("artifact_tree_sha256"),
                "optimizer_steps": manifest_data.get("optimizer_steps"),
                "train_file": manifest_data.get("train_file"),
                "train_sha256": manifest_data.get("train_sha256"),
            }
        )
    return adapters


def compatibility_groups(adapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for adapter in adapters:
        if adapter.get("direct_vote_group"):
            groups[str(adapter["direct_vote_group"])].append(adapter)
    packets: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        packets.append(
            {
                "key": key,
                "size": len(members),
                "base": members[0]["resolved_base_checkpoint"],
                "tokenizer_sha256": members[0]["tokenizer_sha256"],
                "members": [member["relative_path"] for member in members],
            }
        )
    return packets


def build_inventory(paths: LabPaths) -> dict[str, Any]:
    paths.validate_layout()
    models = discover_models(paths)
    adapters = discover_adapters(paths)
    return {
        "schema_version": "1.0",
        "scope": "official_repository_only",
        "official_root": str(paths.official_root),
        "excluded_labs": sorted(str(paths.workspace_root / name) for name in EXCLUDED_DIRNAMES),
        "dense_model_count": len(models),
        "adapter_count": len(adapters),
        "dense_models": models,
        "adapters": adapters,
        "direct_vote_compatibility_groups": compatibility_groups(adapters),
    }
