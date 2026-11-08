"""Bounded historical-soup transfer support for the frozen A1.1/Baya trial."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
SOUP_SRC = REPO / "Rineya_Layerwise_Soup_Lab_v1.0" / "src"

TRIAL_SCHEMA_VERSION = "rineya.historical-transfer-trial.v1"
ARTIFACT_SCHEMA_VERSION = "rineya.historical-soup-artifact.v1"
PREFLIGHT_SCHEMA_VERSION = "rineya.historical-soup-preflight.v1"

ARCHITECTURE_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "tie_word_embeddings",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "sliding_window",
    "use_sliding_window",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != TRIAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported trial schema: {value.get('schema_version')!r}")
    if value.get("registration_status") != "FROZEN_BEFORE_OUTCOME":
        raise ValueError("trial registration is not frozen before outcome")
    return value


def _rope_theta(config: Mapping[str, Any]) -> float | None:
    if config.get("rope_theta") is not None:
        return float(config["rope_theta"])
    parameters = config.get("rope_parameters")
    if isinstance(parameters, Mapping) and parameters.get("rope_theta") is not None:
        return float(parameters["rope_theta"])
    return None


def _layer_types(config: Mapping[str, Any]) -> list[str]:
    count = int(config.get("num_hidden_layers") or 0)
    value = config.get("layer_types")
    return ["full_attention"] * count if value is None else [str(item) for item in value]


def _hash_check(directory: Path, filename: str, expected: str) -> dict[str, Any]:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual} != registered {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def preflight(
    registration_path: Path,
    parent_dir: Path,
    ingredient_dir: Path,
) -> dict[str, Any]:
    registration_path = registration_path.resolve(strict=True)
    parent_dir = parent_dir.resolve(strict=True)
    ingredient_dir = ingredient_dir.resolve(strict=True)
    registration = load_registration(registration_path)
    parent_spec = registration["parent"]
    ingredient_spec = registration["ingredient"]

    parent_files = {
        "weights": _hash_check(parent_dir, "model.safetensors", parent_spec["weight_sha256"]),
        "config": _hash_check(parent_dir, "config.json", parent_spec["config_sha256"]),
        "tokenizer": _hash_check(parent_dir, "tokenizer.json", parent_spec["tokenizer_sha256"]),
    }
    ingredient_files = {
        "weights": _hash_check(
            ingredient_dir, "model.safetensors", ingredient_spec["weight_sha256"]
        ),
        "config": _hash_check(ingredient_dir, "config.json", ingredient_spec["config_sha256"]),
        "tokenizer": _hash_check(
            ingredient_dir, "tokenizer.json", ingredient_spec["tokenizer_sha256"]
        ),
    }
    if parent_files["weights"]["bytes"] != int(parent_spec["weight_bytes"]):
        raise ValueError("parent weight byte count differs from registration")
    if ingredient_files["weights"]["bytes"] != int(ingredient_spec["weight_bytes"]):
        raise ValueError("ingredient weight byte count differs from registration")

    parent_config = json.loads((parent_dir / "config.json").read_text(encoding="utf-8"))
    ingredient_config = json.loads((ingredient_dir / "config.json").read_text(encoding="utf-8"))
    config_mismatches = {
        field: {"parent": parent_config.get(field), "ingredient": ingredient_config.get(field)}
        for field in ARCHITECTURE_FIELDS
        if parent_config.get(field) != ingredient_config.get(field)
    }
    if _rope_theta(parent_config) != _rope_theta(ingredient_config):
        config_mismatches["effective_rope_theta"] = {
            "parent": _rope_theta(parent_config),
            "ingredient": _rope_theta(ingredient_config),
        }
    if _layer_types(parent_config) != _layer_types(ingredient_config):
        config_mismatches["effective_layer_types"] = {
            "parent": _layer_types(parent_config),
            "ingredient": _layer_types(ingredient_config),
        }
    if config_mismatches:
        raise ValueError(f"architecture config mismatches: {sorted(config_mismatches)}")

    with safe_open(parent_dir / "model.safetensors", framework="pt", device="cpu") as parent:
        with safe_open(
            ingredient_dir / "model.safetensors", framework="pt", device="cpu"
        ) as ingredient:
            parent_keys = set(parent.keys())
            ingredient_keys = set(ingredient.keys())
            missing_in_parent = sorted(ingredient_keys - parent_keys)
            missing_in_ingredient = sorted(parent_keys - ingredient_keys)
            shape_mismatches = [
                key
                for key in sorted(parent_keys & ingredient_keys)
                if parent.get_slice(key).get_shape() != ingredient.get_slice(key).get_shape()
            ]
    if missing_in_parent or missing_in_ingredient or shape_mismatches:
        raise ValueError(
            "tensor incompatibility: "
            f"missing_in_parent={len(missing_in_parent)}, "
            f"missing_in_ingredient={len(missing_in_ingredient)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )
    expected_count = int(registration["compatibility_freeze"]["tensor_count_each"])
    if len(parent_keys) != expected_count:
        raise ValueError(f"tensor count {len(parent_keys)} != registered {expected_count}")

    required_assets = registration["candidate"]["required_runtime_assets"]
    missing_assets = [name for name in required_assets if not (parent_dir / name).is_file()]
    if missing_assets:
        raise FileNotFoundError(f"parent lacks required runtime assets: {missing_assets}")

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "verified_at_utc": utc_now(),
        "trial_id": registration["trial_id"],
        "registration": {
            "path": str(registration_path),
            "sha256": sha256_file(registration_path),
            "registration_base_commit": registration["registration_base_commit"],
        },
        "parent": parent_files,
        "ingredient": ingredient_files,
        "compatibility": {
            "passed": True,
            "tensor_count": len(parent_keys),
            "tensor_key_sets_identical": True,
            "tensor_shapes_identical": True,
            "architecture_fields_identical": True,
            "effective_rope_theta": _rope_theta(parent_config),
            "tokenizer_sha256_identical": (
                parent_spec["tokenizer_sha256"] == ingredient_spec["tokenizer_sha256"]
            ),
            "parent_architecture": parent_config.get("architectures"),
            "ingredient_architecture": ingredient_config.get("architectures"),
            "intentional_runtime_identity": parent_spec["required_class"],
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_inherited_identity(destination: Path, forbidden: list[str]) -> None:
    for name in forbidden:
        target = destination / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _candidate_card(registration: Mapping[str, Any]) -> str:
    return (
        f"# {registration['candidate']['name']}\n\n"
        "Status: **bounded transfer candidate — not a release**.\n\n"
        "This artifact is the single frozen 50/50 Supa-A1.1/Baya dense soup from "
        f"trial `{registration['trial_id']}`. It retains the parent A1.1 tokenizer, "
        "custom class, and S4 abstention offset. See `TRIAL_ARTIFACT.json` and "
        "`merge_manifest.json` for provenance. Promotion is forbidden until the "
        "separate ten-gate production ledger passes.\n"
    )


def build_candidate(
    registration_path: Path,
    parent_dir: Path,
    ingredient_dir: Path,
    destination: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    registration_path = registration_path.resolve(strict=True)
    destination = destination.resolve()
    evidence_dir = evidence_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"candidate destination already exists: {destination}")
    report = preflight(registration_path, parent_dir, ingredient_dir)
    _write_json(evidence_dir / "preflight.json", report)
    registration = load_registration(registration_path)

    if str(SOUP_SRC) not in sys.path:
        sys.path.insert(0, str(SOUP_SRC))
    from rineya_soup_lab.model.layer_map import MergeRecipe
    from rineya_soup_lab.model.merge import merge_checkpoints

    coefficient = float(registration["recipe"]["ingredient_coefficient"])
    recipe = MergeRecipe(
        global_alpha=coefficient,
        label=registration["candidate"]["name"],
        metadata={
            "experiment": registration["trial_id"],
            "registration_sha256": report["registration"]["sha256"],
        },
    )
    merge_manifest = merge_checkpoints(
        Path(parent_dir),
        Path(ingredient_dir),
        destination,
        recipe,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        accumulation_dtype="float32",
        output_dtype="float16",
        max_shard_size_gb=0.25,
        require_non_floating_equal=True,
        copy_non_floating_from="anchor",
        overwrite=False,
        source_fingerprints={
            "parent": report["parent"],
            "ingredient": report["ingredient"],
        },
        allow_tied_lm_head_alias=False,
    )
    _remove_inherited_identity(
        destination, list(registration["candidate"]["forbidden_inherited_identity_assets"])
    )
    (destination / "README.md").write_text(
        _candidate_card(registration), encoding="utf-8", newline="\n"
    )

    verification = verify_candidate(registration_path, destination)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "trial_id": registration["trial_id"],
        "candidate_name": registration["candidate"]["name"],
        "registration": report["registration"],
        "source_preflight": report,
        "recipe": registration["recipe"],
        "merge_manifest_sha256": sha256_file(destination / "merge_manifest.json"),
        "output_checkpoint_digest": merge_manifest["output_checkpoint_digest"],
        "verification": verification,
    }
    _write_json(destination / "TRIAL_ARTIFACT.json", artifact)
    artifact["trial_artifact_sha256"] = sha256_file(destination / "TRIAL_ARTIFACT.json")
    _write_json(evidence_dir / "build_result.json", artifact)
    return artifact


def verify_candidate(registration_path: Path, destination: Path) -> dict[str, Any]:
    registration = load_registration(registration_path.resolve(strict=True))
    destination = destination.resolve(strict=True)
    forbidden_present = [
        name
        for name in registration["candidate"]["forbidden_inherited_identity_assets"]
        if (destination / name).exists()
    ]
    if forbidden_present:
        raise ValueError(f"candidate inherited forbidden parent identity: {forbidden_present}")
    missing_assets = [
        name
        for name in registration["candidate"]["required_runtime_assets"]
        if not (destination / name).is_file()
    ]
    if missing_assets:
        raise FileNotFoundError(f"candidate lacks runtime assets: {missing_assets}")

    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    parent = registration["parent"]
    if config.get("architectures") != [parent["required_class"]]:
        raise ValueError("candidate architecture does not preserve the parent custom class")
    if int(config.get("abstain_token_id")) != int(parent["abstain_token_id"]):
        raise ValueError("candidate abstain_token_id drifted")
    if float(config.get("abstain_logit_bias")) != float(parent["abstain_logit_bias"]):
        raise ValueError("candidate abstain_logit_bias drifted")

    weight_files = sorted(destination.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("candidate contains no safetensors weights")
    weight_records = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in weight_files
    ]
    index_path = destination / "model.safetensors.index.json"
    return {
        "passed": True,
        "weight_files": weight_records,
        "weight_bytes": sum(item["bytes"] for item in weight_records),
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        "config_sha256": sha256_file(destination / "config.json"),
        "tokenizer_sha256": sha256_file(destination / "tokenizer.json"),
        "required_class": parent["required_class"],
        "abstain_token_id": parent["abstain_token_id"],
        "abstain_logit_bias": parent["abstain_logit_bias"],
        "forbidden_identity_assets_absent": True,
    }
