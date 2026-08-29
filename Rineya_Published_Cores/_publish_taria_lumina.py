#!/usr/bin/env python3
"""Archival publisher for the former Taria and Lumina dense-core identities.

Taria and Lumina were renamed to Supa and Baya, respectively.  The historical
materialization logic remains here so the original publication can be audited,
but every action refuses to run once either canonical successor is present in
the vault or catalog.  This prevents recreating the retired directories or
adding duplicate catalog rows after the metadata-only rename.

Before retirement, this publisher was deliberately separate from
``_materialize_core.py`` and had two phases:

1. ``stage`` reconstructs a BF16 dense checkpoint in an immutable staging
   directory and writes provisional provenance.
2. ``run-parity`` records a provenance-bound public Demo-256 dense evaluation.
   ``record-parity`` is diagnostic only and cannot authorize publication.
   ``finalize`` then atomically promotes the staged pair and updates the catalog
   without rebuilding it or dropping archived entries.

``status`` remains safe and reports the retirement/successor state.  No action
overwrites an existing core, staging directory, parity packet, or prediction
file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VAULT = Path(__file__).resolve().parent
REPO = VAULT.parent
BASE_MODEL = VAULT / "Rineya-Nimbus-1.6.0-devfreeze"
CATALOG = VAULT / "CATALOG.json"

DEMO = REPO / "Rineya_Long_Context_Demo_v1"
DEMO_DATA = DEMO / "data"
RUNNER = DEMO / "scripts" / "run_local_hf.py"
EVALUATOR = DEMO / "scripts" / "evaluate_predictions.py"
DEFAULT_PARITY_ROOT = (
    REPO
    / "artifacts"
    / "rineya_long_context_weight_geometry_v1"
    / "published_dense_parity_v2"
)
LEGACY_PARITY_ARCHIVE_ROOT = (
    REPO
    / "artifacts"
    / "rineya_long_context_weight_geometry_v1"
    / "published_dense_parity_legacy_packets"
)

BASE_CORE_ID = (
    "rineya.core.nimbus.v1.6.0-devfreeze.28L.qwen3-x-halo-uncensored-alpha-0.40"
)
TRAINING_BASE_CORE_ID = (
    "rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4"
)
BASE_WEIGHT_SHA256 = "7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c"
TRAINING_BASE_WEIGHT_SHA256 = (
    "38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544"
)
ADAPTER_WEIGHT_SHA256 = (
    "517966d7564c083f12504e4c3ff1ac3e82a0abb97d2845ea3dd116174b01efb4"
)
BASE_TENSOR_INVENTORY_SHA256 = (
    "836380aa644ad2640d546039e1a393c0408ac915630a15d2f3687be46eab856e"
)
RUNNER_SHA256 = "4e9a73e78fe706a7e0491b57534d19753228cb743e2a3811c9307077815f4119"
EVALUATOR_SHA256 = "92c9e17762ff5979ac6c0657be26ee0a99bfce0620453886a2e2bc81632af1b5"
CONTROLLED_PARITY_MODE = "publisher_controlled_demo256_v2"

# These files are copied back after save_pretrained.  This prevents the
# installed Transformers version from silently rewriting model or tokenizer
# semantics while materializing the dense weight file.
PRESERVED_BASE_FILE_SHA256 = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": (
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
    ),
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": (
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
    ),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}

DEMO_FILE_SHA256 = {
    "all_prompts.jsonl": "2619bd2c4a503e8d65e1d22d5090738b91229365b400332ade29c110dece0af5",
    "all_answer_key.jsonl": (
        "b95cf52c29a44b3f4eb18d2979c18e3b941d93709a06d908c30295ac6ed66398"
    ),
    "showcase_answer_key.jsonl": (
        "dc1541b6053b7a945774e3d87ca02cd3f06adb0a27a2f707f61d655001b6d53b"
    ),
    "stability_answer_key.jsonl": (
        "248bdc5592ba9eac05737e6cf160d5e6ec85d03590ee4ba8778f28e4098d01d9"
    ),
}

EXPECTED_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
PREDICTION_FIELDS = ("prediction", "response", "output", "generated_text", "answer")


@dataclass(frozen=True)
class CoreSpec:
    key: str
    codename: str
    directory: str
    core_id: str
    version: str
    adapter_relpath: str
    adapter_config_sha256: str
    scale: float
    encoded_alpha: int
    reference_predictions_relpath: str
    reference_predictions_sha256: str
    showcase_hits: int
    stability_hits: int
    all_hits: int
    role: str

    @property
    def adapter_path(self) -> Path:
        return REPO / self.adapter_relpath

    @property
    def final_path(self) -> Path:
        return VAULT / self.directory

    @property
    def staging_path(self) -> Path:
        return VAULT / f".{self.directory}.staging-v1"

    @property
    def parity_path(self) -> Path:
        return self.staging_path / "PARITY.v2.json"

    @property
    def legacy_parity_path(self) -> Path:
        return self.staging_path / "PARITY.json"

    @property
    def analysis_parity_path(self) -> Path:
        return self.staging_path / "PARITY.analysis-only.json"

    def archived_parity_path(self, source: Path) -> Path:
        return LEGACY_PARITY_ARCHIVE_ROOT / self.key / source.name

    @property
    def reference_predictions_path(self) -> Path:
        return REPO / self.reference_predictions_relpath


CORE_SPECS = {
    "taria": CoreSpec(
        key="taria",
        codename="Taria",
        directory="Rineya-Taria-1.7.0-devfreeze",
        core_id=(
            "rineya.core.taria.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
        ),
        version="1.7.0-devfreeze",
        adapter_relpath=(
            "artifacts/rineya_long_context_weight_geometry_v1/selected/"
            "nimbus_step200_scale0p8125_adapter"
        ),
        adapter_config_sha256=(
            "1abf76a796f86c52273ad0d88d650e4174076476a91dac78c1864a70f1d7160d"
        ),
        scale=0.8125,
        encoded_alpha=26,
        reference_predictions_relpath=(
            "artifacts/rineya_long_context_weight_geometry_v1/final/"
            "nimbus_step200_scale0p8125_all256.jsonl"
        ),
        reference_predictions_sha256=(
            "dd8476ed28f4a57d64fd18a64a4f20fae7084382d16cd87c552a3611d4775ab7"
        ),
        showcase_hits=42,
        stability_hits=121,
        all_hits=163,
        role="low-scale paired-answer robustness arm",
    ),
    "lumina": CoreSpec(
        key="lumina",
        codename="Lumina",
        directory="Rineya-Lumina-1.7.0-devfreeze",
        core_id=(
            "rineya.core.lumina.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-1p25"
        ),
        version="1.7.0-devfreeze",
        adapter_relpath=(
            "artifacts/rineya_long_context_weight_geometry_v1/selected/"
            "nimbus_step200_scale1p25_adapter"
        ),
        adapter_config_sha256=(
            "40200036e8ff271f6b3eb45732c137493530dbb20d9c0ae8ca6e60af4d771b4b"
        ),
        scale=1.25,
        encoded_alpha=40,
        reference_predictions_relpath=(
            "artifacts/rineya_long_context_weight_geometry_v1/final/"
            "nimbus_step200_scale1p25_all256.jsonl"
        ),
        reference_predictions_sha256=(
            "917ef4dbd6eb14d19b21c1bc0e71fedcb79090471af41f23b03b908563cbe075"
        ),
        showcase_hits=40,
        stability_hits=113,
        all_hits=153,
        role="high-scale abstention and counterfactual arm",
    ),
}

# These identities are the canonical, metadata-only successors of the frozen
# historical CORE_SPECS above.  Do not rewrite CORE_SPECS: its values bind the
# old merge/parity evidence and are intentionally retained for auditability.
RENAMED_SUCCESSORS: dict[str, dict[str, str]] = {
    "taria": {
        "codename": "Supa",
        "directory": "Rineya-Supa-1.7.0-devfreeze",
        "core_id": (
            "rineya.core.supa.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
        ),
    },
    "lumina": {
        "codename": "Baya",
        "directory": "Rineya-Baya-1.7.0-devfreeze",
        "core_id": (
            "rineya.core.baya.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-1p25"
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def require_within_repo(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != REPO.resolve() and REPO.resolve() not in resolved.parents:
        raise ValueError(f"{label} must stay inside the repository: {resolved}")
    return resolved


def resolve_repo_relative(value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be stored as a repository-relative path")
    return require_within_repo(REPO / candidate, label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _catalog_successor_markers(catalog_path: Path) -> list[str]:
    """Return renamed identities found in an existing catalog."""
    if not catalog_path.is_file():
        return []
    catalog = load_json(catalog_path)
    entries = catalog.get("cores")
    if not isinstance(entries, list):
        raise TypeError("CATALOG.json cores must be a list")

    markers: list[str] = []
    for successor in RENAMED_SUCCESSORS.values():
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            core = entry.get("core")
            records = [entry, core] if isinstance(core, dict) else [entry]
            if any(
                record.get("core_id") == successor["core_id"]
                or record.get("directory") == successor["directory"]
                or record.get("directory_name") == successor["directory"]
                or record.get("codename") == successor["codename"]
                for record in records
            ):
                markers.append(
                    f"catalog:{successor['codename']} ({successor['directory']})"
                )
                break
    return markers


def legacy_publisher_retirement_markers(
    *, catalog_path: Path | None = None
) -> list[str]:
    """Find evidence that the Taria/Lumina publisher has been retired."""
    markers = [
        f"vault:{successor['codename']} ({VAULT / successor['directory']})"
        for successor in RENAMED_SUCCESSORS.values()
        if (VAULT / successor["directory"]).exists()
    ]
    if catalog_path is not None:
        markers.extend(_catalog_successor_markers(catalog_path))
    return markers


def refuse_retired_legacy_publication(*, catalog_path: Path | None = None) -> None:
    """Fail closed when Supa/Baya prove the paired legacy release was renamed."""
    markers = legacy_publisher_retirement_markers(catalog_path=catalog_path)
    if markers:
        raise RuntimeError(
            "Legacy Taria/Lumina publisher is retired after the canonical "
            "Taria->Supa and Lumina->Baya rename; refusing legacy publication. "
            "Detected: " + "; ".join(markers)
        )


def require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"Pinned {label} hash mismatch for {path}:\n"
            f"  expected={expected.lower()}\n  observed={observed.lower()}"
        )
    return observed.lower()


def snapshot_files(root: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(set(names)):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def safetensor_inventory(path: Path) -> list[dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("Install safetensors before inventory validation") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return [
            {
                "key": key,
                "shape": [int(value) for value in handle.get_slice(key).get_shape()],
                "dtype": str(handle.get_slice(key).get_dtype()),
            }
            for key in sorted(handle.keys())
        ]


def source_snapshot(spec: CoreSpec) -> dict[str, Any]:
    return {
        "base": snapshot_files(
            BASE_MODEL,
            ["model.safetensors", *PRESERVED_BASE_FILE_SHA256],
        ),
        "adapter": snapshot_files(
            spec.adapter_path,
            ["adapter_config.json", "adapter_model.safetensors"],
        ),
    }


def assert_source_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any], spec: CoreSpec
) -> None:
    if before != after:
        raise RuntimeError(
            f"Source files changed while staging {spec.codename}; refusing publication."
        )


def _resolve_adapter_base(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate.resolve()


def validate_pinned_sources(spec: CoreSpec) -> dict[str, Any]:
    if not BASE_MODEL.is_dir():
        raise FileNotFoundError(BASE_MODEL)
    base_core = load_json(BASE_MODEL / "CORE.json")
    if base_core.get("core_id") != BASE_CORE_ID:
        raise RuntimeError(
            f"Nimbus CORE.json identity mismatch: {base_core.get('core_id')!r}"
        )
    require_sha256(
        BASE_MODEL / "model.safetensors",
        BASE_WEIGHT_SHA256,
        "Nimbus model",
    )
    base_inventory = safetensor_inventory(BASE_MODEL / "model.safetensors")
    if canonical_json_sha256(base_inventory) != BASE_TENSOR_INVENTORY_SHA256:
        raise RuntimeError("Pinned Nimbus tensor inventory mismatch")
    for name, digest in PRESERVED_BASE_FILE_SHA256.items():
        require_sha256(BASE_MODEL / name, digest, f"Nimbus {name}")

    adapter = spec.adapter_path
    require_sha256(
        adapter / "adapter_model.safetensors",
        ADAPTER_WEIGHT_SHA256,
        f"{spec.codename} adapter weights",
    )
    require_sha256(
        adapter / "adapter_config.json",
        spec.adapter_config_sha256,
        f"{spec.codename} adapter config",
    )
    config = load_json(adapter / "adapter_config.json")
    adapter_base = str(config.get("base_model_name_or_path") or "")
    if _resolve_adapter_base(adapter_base) != BASE_MODEL.resolve():
        raise RuntimeError(
            f"{spec.codename} adapter application base mismatch: {adapter_base!r}"
        )
    if int(config.get("r") or 0) != 16:
        raise RuntimeError(f"{spec.codename} adapter rank is not 16")
    if int(config.get("lora_alpha") or 0) != spec.encoded_alpha:
        raise RuntimeError(f"{spec.codename} encoded LoRA alpha mismatch")
    if not math.isclose(spec.encoded_alpha / 32.0, spec.scale):
        raise RuntimeError(f"{spec.codename} alpha does not encode its selected scale")
    if set(config.get("target_modules") or []) != EXPECTED_TARGET_MODULES:
        raise RuntimeError(f"{spec.codename} adapter target modules mismatch")
    required = {
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"{spec.codename} adapter config mismatch: {mismatches}")
    return config


def refuse_publish_paths(specs: Sequence[CoreSpec], *, require_staged: bool) -> None:
    refuse_retired_legacy_publication()
    for spec in specs:
        if spec.final_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing published core: {spec.final_path}"
            )
        if require_staged:
            if not spec.staging_path.is_dir():
                raise FileNotFoundError(spec.staging_path)
        elif spec.staging_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing staging core: {spec.staging_path}"
            )


def copy_preserved_assets(
    base: Path,
    destination: Path,
    expected_hashes: Mapping[str, str] = PRESERVED_BASE_FILE_SHA256,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name, expected in expected_hashes.items():
        source = base / name
        require_sha256(source, expected, f"preserved source {name}")
        shutil.copy2(source, destination / name)
        copied[name] = require_sha256(
            destination / name,
            expected,
            f"preserved destination {name}",
        )
    return copied


def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "safetensors", "accelerate"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": packages,
    }


def verify_dense_checkpoint(path: Path, *, reload_model: bool = True) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    peft_files = sorted(
        str(item.relative_to(path))
        for item in path.rglob("*")
        if item.is_file()
        and (item.name.startswith("adapter_") or item.name == "adapter_config.json")
    )
    if peft_files:
        raise RuntimeError(f"Dense core contains PEFT remnants: {peft_files}")

    preserved = {
        name: require_sha256(path / name, digest, f"output {name}")
        for name, digest in PRESERVED_BASE_FILE_SHA256.items()
    }
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    extra_weight_files = sorted(
        item.name
        for item in path.glob("*.safetensors")
        if item.name != "model.safetensors"
    )
    if extra_weight_files or (path / "model.safetensors.index.json").exists():
        raise RuntimeError(
            "Publisher requires one unsharded model.safetensors; found "
            f"extras={extra_weight_files}"
        )

    base_inventory = safetensor_inventory(BASE_MODEL / "model.safetensors")
    if canonical_json_sha256(base_inventory) != BASE_TENSOR_INVENTORY_SHA256:
        raise RuntimeError(
            "Pinned Nimbus tensor inventory mismatch during verification"
        )
    output_inventory = safetensor_inventory(weights)
    base_by_key = {row["key"]: row for row in base_inventory}
    output_by_key = {row["key"]: row for row in output_inventory}
    missing_keys = sorted(set(base_by_key) - set(output_by_key))
    unexpected_keys = sorted(set(output_by_key) - set(base_by_key))
    if unexpected_keys or missing_keys not in ([], ["lm_head.weight"]):
        raise RuntimeError(
            "Dense tensor inventory mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    inventory_mismatches = {
        key: {"base": base_by_key[key], "output": output_by_key[key]}
        for key in output_by_key
        if base_by_key[key]["shape"] != output_by_key[key]["shape"]
        or output_by_key[key]["dtype"] != "BF16"
    }
    if inventory_mismatches:
        raise RuntimeError(
            "Dense tensor shape/dtype inventory mismatch: "
            f"{list(inventory_mismatches)[:8]}"
        )
    if "model.embed_tokens.weight" not in output_by_key:
        raise RuntimeError("Dense checkpoint is missing model.embed_tokens.weight")

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("Install torch and safetensors before verification") from exc

    dtype_counts: dict[str, int] = {}
    tensor_count = 0
    parameter_count = 0
    non_finite: list[str] = []
    lora_keys: list[str] = []
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            if "lora" in key.lower() or "adapter" in key.lower():
                lora_keys.append(key)
            view = handle.get_slice(key)
            dtype = str(view.get_dtype())
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            shape = tuple(int(value) for value in view.get_shape())
            count = math.prod(shape)
            parameter_count += count
            tensor_count += 1
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                non_finite.append(key)
            del tensor
    if lora_keys:
        raise RuntimeError(
            f"Dense checkpoint contains LoRA tensor keys: {lora_keys[:8]}"
        )
    if dtype_counts != {"BF16": tensor_count}:
        raise RuntimeError(f"Dense checkpoint is not BF16-only: {dtype_counts}")
    if non_finite:
        raise RuntimeError(
            f"Dense checkpoint contains non-finite tensors: {non_finite[:8]}"
        )

    config = load_json(path / "config.json")
    geometry = {
        "model_type": config.get("model_type"),
        "architecture": (config.get("architectures") or [None])[0],
        "num_hidden_layers": int(config.get("num_hidden_layers") or 0),
        "hidden_size": int(config.get("hidden_size") or 0),
        "vocab_size": int(config.get("vocab_size") or 0),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
    }
    expected_geometry = {
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if geometry != expected_geometry:
        raise RuntimeError(f"Dense checkpoint geometry mismatch: {geometry}")

    reload_checks: dict[str, Any] = {"performed": False}
    if reload_model:
        from transformers import AutoModelForCausalLM

        kwargs = {
            "device_map": {"": "cpu"},
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "output_loading_info": True,
        }
        try:
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                str(path), dtype=torch.bfloat16, **kwargs
            )
        except TypeError as exc:  # Transformers 4.x compatibility
            if "dtype" not in str(exc):
                raise
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                str(path), torch_dtype=torch.bfloat16, **kwargs
            )
        model.eval()
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        reload_checks = {
            "performed": True,
            "class": type(model).__name__,
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "tied_embeddings": input_weight.data_ptr() == output_weight.data_ptr(),
            "missing_keys": list(loading_info.get("missing_keys") or []),
            "unexpected_keys": list(loading_info.get("unexpected_keys") or []),
            "mismatched_keys": list(loading_info.get("mismatched_keys") or []),
            "error_messages": list(loading_info.get("error_msgs") or []),
        }
        del output_weight, input_weight, model
        gc.collect()
        if reload_checks["num_hidden_layers"] != 28:
            raise RuntimeError(f"Reloaded layer count mismatch: {reload_checks}")
        if reload_checks["parameter_dtype"] != "torch.bfloat16":
            raise RuntimeError(f"Reloaded dtype mismatch: {reload_checks}")
        if not reload_checks["tied_embeddings"]:
            raise RuntimeError(f"Reloaded embeddings are not tied: {reload_checks}")
        loading_missing = set(reload_checks["missing_keys"])
        if loading_missing - {"lm_head.weight"}:
            raise RuntimeError(f"Reload reported missing keys: {reload_checks}")
        if (
            reload_checks["unexpected_keys"]
            or reload_checks["mismatched_keys"]
            or reload_checks["error_messages"]
        ):
            raise RuntimeError(f"Reload reported incompatible weights: {reload_checks}")

    return {
        **geometry,
        "weight_bytes": weights.stat().st_size,
        "weight_sha256": sha256_file(weights),
        "tensor_count": tensor_count,
        "parameter_count_serialized": parameter_count,
        "tensor_inventory_sha256": canonical_json_sha256(output_inventory),
        "tensor_inventory_missing_from_base": missing_keys,
        "tensor_inventory_unexpected": unexpected_keys,
        "dtype_counts": dtype_counts,
        "all_finite": True,
        "no_peft_remnants": True,
        "preserved_asset_sha256": preserved,
        "reload": reload_checks,
    }


def reference_metrics(spec: CoreSpec) -> dict[str, Any]:
    return {
        "classification": "peft_reference_before_dense_materialization",
        "showcase64": {
            "hits": spec.showcase_hits,
            "items": 64,
            "strict_exact": spec.showcase_hits / 64,
            "selection_split": True,
        },
        "stability192": {
            "hits": spec.stability_hits,
            "items": 192,
            "strict_exact": spec.stability_hits / 192,
            "frozen_validation": True,
        },
        "all256": {
            "hits": spec.all_hits,
            "items": 256,
            "strict_exact": spec.all_hits / 256,
        },
        "prediction_path": spec.reference_predictions_relpath,
        "prediction_sha256": spec.reference_predictions_sha256,
        "caveat": (
            "Reference used PEFT-on-Nimbus inference. Dense scores must be measured "
            "and recorded before final publication."
        ),
    }


def _pair_metadata(spec: CoreSpec) -> dict[str, Any]:
    sibling = CORE_SPECS["lumina" if spec.key == "taria" else "taria"]
    return {
        "sibling_core_id": sibling.core_id,
        "sibling_directory": sibling.directory,
        "role": spec.role,
        "ensemble_method": "high_scale_unknown_else_low_scale",
        "recipe": (
            "Rineya_Long_Context_Weight_Geometry_Lab_v1/configs/final/"
            "nimbus_two_scale_unknown_gate.json"
        ),
        "prior_peft_pair_result": {
            "showcase64_hits": 45,
            "stability192_hits": 124,
            "all256_hits": 169,
            "caveat": "Must be recomputed using the two dense cores.",
        },
    }


def render_model_readme(
    spec: CoreSpec,
    *,
    provisional: bool,
    dense_results: Mapping[str, Any] | None = None,
) -> str:
    status = (
        "staged; dense parity pending" if provisional else "dense validation recorded"
    )
    lines = [
        "---",
        "library_name: transformers",
        "license: apache-2.0",
        "pipeline_tag: text-generation",
        "base_model: Qwen/Qwen3-0.6B",
        "---",
        "",
        f"# Rineya {spec.codename} {spec.version}",
        "",
        f"Status: **{status}**.",
        "",
        (
            f"{spec.codename} is a BF16 dense reconstruction of Nimbus 1.6 plus "
            "the disjoint Halo long-context Stage-A checkpoint-200 LoRA at "
            f"effective scale `{spec.scale}` (rank 16, encoded alpha "
            f"`{spec.encoded_alpha}`)."
        ),
        "",
        (
            "The LoRA was trained on Halo 1.2 and transferred to Nimbus; it was "
            "not trained on Nimbus. Taria and Lumina are separate dense cores. "
            "The answer-blind UNKNOWN gate is a two-model inference recipe, not "
            "a property of either core alone."
        ),
        "",
        "## Frozen evaluation protocol",
        "",
        (
            "BF16, greedy decoding, seed 17, thinking disabled, 24 generated "
            "tokens, 4096-token context cap, and no truncation."
        ),
        "",
    ]
    if dense_results:
        lines.extend(
            [
                "## Dense Demo results",
                "",
                (f"- Showcase (selection): {dense_results['showcase64']['hits']}/64"),
                (
                    f"- Stability (frozen validation): "
                    f"{dense_results['stability192']['hits']}/192"
                ),
                f"- Complete public Demo: {dense_results['all256']['hits']}/256",
                "",
                (
                    "These are public Demo results. Showcase was used during model "
                    "selection and is not an untouched test result."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Evaluation state",
                "",
                (
                    "The source PEFT arm has stored public Demo results, but this "
                    "staged dense checkpoint is not publishable until its own 256-row "
                    "evaluation and prediction-parity packet are recorded."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def provisional_metadata(
    spec: CoreSpec,
    *,
    started_at: str,
    source_before: Mapping[str, Any],
    source_after: Mapping[str, Any],
    checks: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    common_source = {
        "application_base": {
            "path": rel_repo(BASE_MODEL),
            "core_id": BASE_CORE_ID,
            "weight_sha256": BASE_WEIGHT_SHA256,
            "config_sha256": PRESERVED_BASE_FILE_SHA256["config.json"],
            "tokenizer_sha256": PRESERVED_BASE_FILE_SHA256["tokenizer.json"],
        },
        "training_base": {
            "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze",
            "core_id": TRAINING_BASE_CORE_ID,
            "weight_sha256": TRAINING_BASE_WEIGHT_SHA256,
            "note": "The Stage-A LoRA was trained on Halo, then transferred to Nimbus.",
        },
        "adapter": {
            "path": spec.adapter_relpath,
            "original_checkpoint": "artifacts/halo_long_context_sft_v1/stage_a/checkpoint-200",
            "weight_sha256": ADAPTER_WEIGHT_SHA256,
            "config_sha256": spec.adapter_config_sha256,
            "training_step": 200,
            "rank": 16,
            "original_alpha": 32,
            "effective_scale": spec.scale,
            "encoded_alpha": spec.encoded_alpha,
            "effective_alpha_over_rank": spec.encoded_alpha / 16,
            "target_modules": sorted(EXPECTED_TARGET_MODULES),
            "dropout": adapter_config.get("lora_dropout"),
            "bias": adapter_config.get("bias"),
            "peft_version_recorded": adapter_config.get("peft_version"),
        },
    }
    core = {
        "schema_version": "1.1",
        "core_id": spec.core_id,
        "codename": spec.codename,
        "directory_name": spec.directory,
        "version": spec.version,
        "kind": "lora_transfer_dense_merge",
        "status": "staged_pending_dense_parity",
        "classification": "showcase_selected_stability_validated_source_arm",
        "public_claim": (
            "PROVISIONAL: BF16 dense merge staged. Source PEFT metrics are not "
            "assigned to the dense core until dense parity is measured."
        ),
        "provisional": True,
        "created_at_utc": started_at,
        "staged_at_utc": utc_now(),
        "device": "cpu",
        "sources_unmodified": source_before == source_after,
        "source": common_source,
        "merge": {
            "formula": "BF16(FP32(Nimbus) + FP32(LoRA_B @ LoRA_A) * alpha/r)",
            "accumulator_dtype": "float32",
            "output_dtype": "bfloat16",
            "safe_merge": True,
            "serialization": "safetensors",
            "preserved_base_assets_byte_exact": True,
        },
        "pair": _pair_metadata(spec),
        "reference_adapter_evaluation": reference_metrics(spec),
        "dense_validation": None,
        "checks": dict(checks),
        "note": (
            "Staging metadata only. Finalization requires a complete dense Demo-256 "
            "parity packet and never overwrites an existing core."
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "kind": "taria_lumina_bf16_dense_lora_transfer_merge",
        "status": "staged_pending_dense_parity",
        "core_id": spec.core_id,
        "codename": spec.codename,
        "directory_name": spec.directory,
        "created_at_utc": started_at,
        "staged_at_utc": core["staged_at_utc"],
        "source": common_source,
        "source_snapshot_before": source_before,
        "source_snapshot_after": source_after,
        "sources_unmodified": source_before == source_after,
        "algorithm": core["merge"],
        "environment": environment_manifest(),
        "output": {
            "staging_directory": spec.staging_path.name,
            "final_directory": spec.directory,
            "checks": dict(checks),
        },
        "dense_parity": None,
    }
    return core, manifest


def _safe_cleanup_building(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != VAULT.resolve() or ".building-" not in resolved.name:
        raise RuntimeError(f"Refusing unsafe staging cleanup target: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _safe_cleanup_staging(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != VAULT.resolve() or not resolved.name.endswith(".staging-v1"):
        raise RuntimeError(f"Refusing unsafe staging cleanup target: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def require_release_pair(specs: Sequence[CoreSpec], action: str) -> None:
    keys = [spec.key for spec in specs]
    expected = list(CORE_SPECS)
    if keys != expected:
        raise ValueError(
            f"{action} is a paired-release operation and requires exactly "
            f"{expected} in that order"
        )


def stage_core(spec: CoreSpec) -> Path:
    refuse_publish_paths([spec], require_staged=False)
    adapter_config = validate_pinned_sources(spec)
    source_before = source_snapshot(spec)
    started_at = utc_now()
    building = VAULT / f".{spec.directory}.building-{os.getpid()}-{uuid.uuid4().hex}"
    if building.exists():
        raise FileExistsError(building)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError(
            "Install torch, transformers, peft, and safetensors"
        ) from exc

    model = None
    wrapped = None
    merged = None
    try:
        model_kwargs = {
            "device_map": {"": "cpu"},
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(BASE_MODEL), dtype=torch.float32, **model_kwargs
            )
        except TypeError as exc:  # Transformers 4.x compatibility
            if "dtype" not in str(exc):
                raise
            model = AutoModelForCausalLM.from_pretrained(
                str(BASE_MODEL), torch_dtype=torch.float32, **model_kwargs
            )
        wrapped = PeftModel.from_pretrained(
            model,
            str(spec.adapter_path),
            is_trainable=False,
            local_files_only=True,
        )
        merged = wrapped.merge_and_unload(safe_merge=True)
        merged.eval()
        merged.to(dtype=torch.bfloat16)
        building.mkdir(parents=False, exist_ok=False)
        merged.save_pretrained(
            str(building),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        copy_preserved_assets(BASE_MODEL, building)

        del merged, wrapped, model
        merged = wrapped = model = None
        gc.collect()

        checks = verify_dense_checkpoint(building, reload_model=True)
        source_after = source_snapshot(spec)
        assert_source_unchanged(source_before, source_after, spec)
        core, manifest = provisional_metadata(
            spec,
            started_at=started_at,
            source_before=source_before,
            source_after=source_after,
            checks=checks,
            adapter_config=adapter_config,
        )
        atomic_write_json(building / "CORE.json", core)
        atomic_write_json(building / "merge_manifest.json", manifest)
        atomic_write_text(
            building / "README.md",
            render_model_readme(spec, provisional=True),
        )
        if spec.final_path.exists() or spec.staging_path.exists():
            raise FileExistsError(
                f"Publish target appeared during staging: {spec.final_path} / "
                f"{spec.staging_path}"
            )
        os.replace(building, spec.staging_path)
        return spec.staging_path
    except Exception:
        del merged, wrapped, model
        gc.collect()
        _safe_cleanup_building(building)
        raise


def stage_many(specs: Sequence[CoreSpec]) -> list[Path]:
    require_release_pair(specs, "stage")
    refuse_publish_paths(specs, require_staged=False)
    for spec in specs:
        validate_pinned_sources(spec)
    staged: list[Path] = []
    try:
        for spec in specs:
            staged.append(stage_core(spec))
    except Exception:
        for path in reversed(staged):
            _safe_cleanup_staging(path)
        raise
    return staged


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"{path}:{line_no}: expected object with id")
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {row_id}")
            seen.add(row_id)
            rows.append(row)
    return rows


def prediction_text(row: Mapping[str, Any]) -> str:
    for field in PREDICTION_FIELDS:
        if row.get(field) is not None:
            return str(row[field])
    return ""


def normalize_answer(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", value)
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_prediction_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "id": str(row["id"]),
            "status": str(row.get("status", "ok")),
            "prediction": prediction_text(row),
        }
        for row in sorted(rows, key=lambda item: str(item["id"]))
    ]
    return canonical_json_sha256(canonical)


def score_prediction_subset(
    prediction_by_id: Mapping[str, Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    key_ids = [str(row["id"]) for row in key_rows]
    missing = sorted(set(key_ids) - set(prediction_by_id))
    if missing:
        raise ValueError(f"Dense predictions miss answer-key ids: {missing[:8]}")
    hits = 0
    for key in key_rows:
        row = prediction_by_id[str(key["id"])]
        accepted = {
            normalize_answer(value)
            for value in key.get("accepted_answers", [key.get("answer", "")])
        }
        hits += int(normalize_answer(prediction_text(row)) in accepted)
    return {
        "hits": hits,
        "items": len(key_rows),
        "strict_exact": hits / len(key_rows) if key_rows else 0.0,
    }


def _validate_dense_report(
    report_path: Path | None,
    computed_all: Mapping[str, Any],
) -> dict[str, Any] | None:
    if report_path is None:
        return None
    report = load_json(report_path)
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError("Dense report must contain exactly one run")
    run = runs[0]
    expected = {
        "key_items": computed_all["items"],
        "covered_items": computed_all["items"],
        "prediction_rows": computed_all["items"],
    }
    for field, value in expected.items():
        if int(run.get(field) or 0) != int(value):
            raise ValueError(
                f"Dense report {field} mismatch: {run.get(field)} != {value}"
            )
    if run.get("duplicate_prediction_ids") or run.get("unexpected_prediction_ids"):
        raise ValueError("Dense report contains duplicate or unexpected prediction ids")
    if not math.isclose(
        float(run.get("strict_exact") or 0.0),
        float(computed_all["strict_exact"]),
        abs_tol=1e-6,
    ):
        raise ValueError("Dense report strict_exact does not match independent scoring")
    return {
        "path": rel_repo(report_path),
        "sha256": sha256_file(report_path),
        "run": run,
    }


def build_parity_evidence(
    spec: CoreSpec,
    *,
    model_checks: Mapping[str, Any],
    predictions_path: Path,
    dense_report_path: Path | None = None,
    reference_path: Path | None = None,
    answer_key_paths: Mapping[str, Path] | None = None,
    enforce_pins: bool = True,
    run_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference_path = reference_path or spec.reference_predictions_path
    answer_key_paths = answer_key_paths or {
        "showcase64": DEMO_DATA / "showcase_answer_key.jsonl",
        "stability192": DEMO_DATA / "stability_answer_key.jsonl",
        "all256": DEMO_DATA / "all_answer_key.jsonl",
    }
    if enforce_pins:
        predictions_path = require_within_repo(
            predictions_path, "dense prediction artifact"
        )
        reference_path = require_within_repo(
            reference_path, "PEFT reference prediction artifact"
        )
        if dense_report_path is not None:
            dense_report_path = require_within_repo(
                dense_report_path, "dense report artifact"
            )
        answer_key_paths = {
            name: require_within_repo(path, f"{name} answer key")
            for name, path in answer_key_paths.items()
        }
        require_sha256(
            reference_path,
            spec.reference_predictions_sha256,
            f"{spec.codename} PEFT reference predictions",
        )
        for name, path in answer_key_paths.items():
            filename = path.name
            require_sha256(path, DEMO_FILE_SHA256[filename], f"Demo {name} answer key")

    dense_rows = load_jsonl(predictions_path)
    reference_rows = load_jsonl(reference_path)
    if len(dense_rows) != 256 or len(reference_rows) != 256:
        raise ValueError(
            f"Parity requires 256 rows per arm; dense={len(dense_rows)}, "
            f"reference={len(reference_rows)}"
        )
    dense_by_id = {str(row["id"]): row for row in dense_rows}
    reference_by_id = {str(row["id"]): row for row in reference_rows}
    if set(dense_by_id) != set(reference_by_id):
        raise ValueError("Dense and PEFT reference predictions cover different ids")
    non_ok = sorted(
        row_id
        for row_id, row in dense_by_id.items()
        if str(row.get("status", "ok")) != "ok"
    )
    if non_ok:
        raise ValueError(f"Dense parity contains non-ok rows: {non_ok[:8]}")
    if any("adapter_scale" in row for row in dense_rows):
        raise ValueError(
            "Dense prediction file unexpectedly contains adapter_scale fields"
        )
    if enforce_pins:
        bad_protocol_rows = sorted(
            row_id
            for row_id, row in dense_by_id.items()
            if type(row.get("seed")) is not int
            or row.get("seed") != 17
            or row.get("decoding") != "greedy"
        )
        if bad_protocol_rows:
            raise ValueError(
                "Dense prediction rows do not record the frozen seed/decoding: "
                f"{bad_protocol_rows[:8]}"
            )

    raw_matches = 0
    normalized_matches = 0
    differing_ids: list[str] = []
    for row_id in sorted(dense_by_id):
        dense_prediction = prediction_text(dense_by_id[row_id])
        reference_prediction = prediction_text(reference_by_id[row_id])
        raw_equal = dense_prediction == reference_prediction
        normalized_equal = normalize_answer(dense_prediction) == normalize_answer(
            reference_prediction
        )
        raw_matches += int(raw_equal)
        normalized_matches += int(normalized_equal)
        if not raw_equal:
            differing_ids.append(row_id)

    dense_results = {
        name: score_prediction_subset(
            dense_by_id,
            load_jsonl(path),
        )
        for name, path in answer_key_paths.items()
    }
    if dense_results["showcase64"]["items"] != 64:
        raise ValueError("Showcase answer key is not 64 rows")
    if dense_results["stability192"]["items"] != 192:
        raise ValueError("Stability answer key is not 192 rows")
    if dense_results["all256"]["items"] != 256:
        raise ValueError("Complete answer key is not 256 rows")
    report = _validate_dense_report(dense_report_path, dense_results["all256"])

    return {
        "schema_version": "1.0",
        "kind": "dense_vs_peft_prediction_parity",
        "verified_at_utc": utc_now(),
        "core_id": spec.core_id,
        "codename": spec.codename,
        "model": {
            "directory": spec.staging_path.name,
            "weight_sha256": model_checks["weight_sha256"],
            "dtype_counts": model_checks["dtype_counts"],
        },
        "protocol": {
            "dtype": "bfloat16",
            "decoding": "greedy",
            "seed": 17,
            "thinking": False,
            "max_new_tokens": 24,
            "max_context": 4096,
            "truncation": "forbidden",
            "prompts_sha256": DEMO_FILE_SHA256["all_prompts.jsonl"],
        },
        "candidate": {
            "path": rel_repo(predictions_path),
            "file_sha256": sha256_file(predictions_path),
            "canonical_prediction_sha256": canonical_prediction_sha256(dense_rows),
            "rows": len(dense_rows),
        },
        "peft_reference": {
            "path": rel_repo(reference_path),
            "file_sha256": sha256_file(reference_path),
            "canonical_prediction_sha256": canonical_prediction_sha256(reference_rows),
            "rows": len(reference_rows),
        },
        "parity": {
            "raw_prediction_matches": raw_matches,
            "normalized_prediction_matches": normalized_matches,
            "items": len(dense_rows),
            "exact_raw_parity": raw_matches == len(dense_rows),
            "exact_normalized_parity": normalized_matches == len(dense_rows),
            "differing_ids": differing_ids,
        },
        "dense_results": dense_results,
        "dense_report": report,
        "run_manifest": dict(run_manifest) if run_manifest else None,
        "publishable_provenance": bool(
            run_manifest and run_manifest.get("mode") == CONTROLLED_PARITY_MODE
        ),
        "selection_caveat": (
            "Showcase was used during model selection. Stability was frozen before "
            "its first selected-recipe run; all scores are on the public Demo suite."
        ),
    }


def record_parity(
    spec: CoreSpec,
    predictions_path: Path,
    dense_report_path: Path | None,
    *,
    require_exact: bool,
    run_manifest: Mapping[str, Any] | None = None,
) -> Path:
    refuse_retired_legacy_publication()
    if not spec.staging_path.is_dir():
        raise FileNotFoundError(spec.staging_path)
    controlled = bool(
        run_manifest and run_manifest.get("mode") == CONTROLLED_PARITY_MODE
    )
    output_path = spec.parity_path if controlled else spec.analysis_parity_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite parity evidence: {output_path}")
    checks = verify_dense_checkpoint(spec.staging_path, reload_model=True)
    evidence = build_parity_evidence(
        spec,
        model_checks=checks,
        predictions_path=predictions_path.resolve(strict=True),
        dense_report_path=(
            dense_report_path.resolve(strict=True) if dense_report_path else None
        ),
        run_manifest=run_manifest,
    )
    if require_exact and not evidence["parity"]["exact_raw_parity"]:
        raise RuntimeError(
            f"{spec.codename} dense predictions differ from PEFT on "
            f"{256 - evidence['parity']['raw_prediction_matches']}/256 rows"
        )
    atomic_write_json(output_path, evidence)
    return output_path


def parity_commands(
    spec: CoreSpec,
    *,
    output_root: Path = DEFAULT_PARITY_ROOT,
) -> list[str]:
    refuse_retired_legacy_publication()
    python = REPO / ".venv" / "Scripts" / "python.exe"
    return [
        " ".join(
            [
                f'& "{python}"',
                f'"{Path(__file__).resolve()}" run-parity {spec.key}',
                f'--output-root "{output_root}"',
                "--max-gpu-memory 2GiB --max-cpu-memory 32GiB",
            ]
        ),
    ]


def _safe_cleanup_parity_building(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    root = output_root.resolve()
    if resolved.parent != root or not resolved.name.startswith(".parity-building-"):
        raise RuntimeError(f"Refusing unsafe parity cleanup target: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def run_parity(
    spec: CoreSpec,
    *,
    output_root: Path,
    max_gpu_memory: str,
    max_cpu_memory: str,
    require_exact: bool,
) -> Path:
    refuse_retired_legacy_publication()
    if not spec.staging_path.is_dir():
        raise FileNotFoundError(spec.staging_path)
    safe_output_root = require_within_repo(output_root.resolve(), "parity output root")
    output_dir = safe_output_root / spec.key
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite parity output: {output_dir}")
    building = safe_output_root / f".parity-building-{spec.key}-{uuid.uuid4().hex}"
    if building.exists():
        raise FileExistsError(building)
    building_predictions = building / f"{spec.key}_dense_all256.jsonl"
    building_report_json = building / f"{spec.key}_dense_all256_report.json"
    building_report_md = building / f"{spec.key}_dense_all256_report.md"
    predictions = output_dir / f"{spec.key}_dense_all256.jsonl"
    report_json = output_dir / f"{spec.key}_dense_all256_report.json"
    if spec.parity_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite parity evidence: {spec.parity_path}"
        )
    safe_output_root.mkdir(parents=True, exist_ok=True)
    building.mkdir(parents=False, exist_ok=False)
    checks_before = verify_dense_checkpoint(spec.staging_path, reload_model=False)
    require_sha256(
        DEMO_DATA / "all_prompts.jsonl",
        DEMO_FILE_SHA256["all_prompts.jsonl"],
        "Demo all prompts",
    )
    require_sha256(RUNNER, RUNNER_SHA256, "frozen local HF runner")
    require_sha256(EVALUATOR, EVALUATOR_SHA256, "frozen Demo evaluator")
    started = utc_now()
    runner_command = [
        sys.executable,
        str(RUNNER),
        "--model",
        str(spec.staging_path),
        "--input",
        str(DEMO_DATA / "all_prompts.jsonl"),
        "--output",
        str(building_predictions),
        "--max-new-tokens",
        "24",
        "--max-context",
        "4096",
        "--on-overflow",
        "error",
        "--seed",
        "17",
        "--device-map",
        "auto",
        "--disable-thinking",
        "--torch-dtype",
        "bfloat16",
        "--max-gpu-memory",
        max_gpu_memory,
        "--max-cpu-memory",
        max_cpu_memory,
    ]
    evaluator_command = [
        sys.executable,
        str(EVALUATOR),
        "--key",
        str(DEMO_DATA / "all_answer_key.jsonl"),
        "--predictions",
        str(building_predictions),
        "--json-report",
        str(building_report_json),
        "--markdown-report",
        str(building_report_md),
    ]
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONUTF8"] = "1"
    try:
        subprocess.run(
            runner_command,
            check=True,
            cwd=str(REPO),
            env=subprocess_env,
        )
        subprocess.run(
            evaluator_command,
            check=True,
            cwd=str(REPO),
            env=subprocess_env,
        )
        checks_after = verify_dense_checkpoint(spec.staging_path, reload_model=False)
        if checks_before["weight_sha256"] != checks_after["weight_sha256"]:
            raise RuntimeError("Staged dense weights changed during parity evaluation")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.replace(building, output_dir)
    except Exception:
        _safe_cleanup_parity_building(building, safe_output_root)
        raise
    run_manifest = {
        "mode": CONTROLLED_PARITY_MODE,
        "core_id": spec.core_id,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "python_version": sys.version.split()[0],
        "environment": environment_manifest(),
        "runner": {"path": rel_repo(RUNNER), "sha256": RUNNER_SHA256},
        "evaluator": {"path": rel_repo(EVALUATOR), "sha256": EVALUATOR_SHA256},
        "invocation": {
            "model": spec.staging_path.name,
            "prompts": rel_repo(DEMO_DATA / "all_prompts.jsonl"),
            "predictions": rel_repo(predictions),
            "report": rel_repo(report_json),
            "dtype": "bfloat16",
            "decoding": "greedy",
            "seed": 17,
            "thinking": False,
            "max_new_tokens": 24,
            "max_context": 4096,
            "truncation": "forbidden",
            "max_gpu_memory": max_gpu_memory,
            "max_cpu_memory": max_cpu_memory,
        },
        "model_weight_sha256_before": checks_before["weight_sha256"],
        "model_weight_sha256_after": checks_after["weight_sha256"],
    }
    return record_parity(
        spec,
        predictions,
        report_json,
        require_exact=require_exact,
        run_manifest=run_manifest,
    )


def validate_parity_evidence(
    spec: CoreSpec,
    evidence: Mapping[str, Any],
    checks: Mapping[str, Any],
    *,
    require_exact: bool,
) -> None:
    if evidence.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported parity schema for {spec.codename}")
    if evidence.get("kind") != "dense_vs_peft_prediction_parity":
        raise ValueError(f"Parity kind mismatch for {spec.codename}")
    if evidence.get("core_id") != spec.core_id:
        raise ValueError(f"Parity core identity mismatch for {spec.codename}")
    if evidence.get("publishable_provenance") is not True:
        raise ValueError(
            f"{spec.codename} parity was not produced by controlled run-parity"
        )

    model = evidence.get("model")
    if not isinstance(model, dict):
        raise TypeError(f"Parity model metadata must be an object for {spec.codename}")
    if model.get("weight_sha256") != checks.get("weight_sha256"):
        raise ValueError(f"Parity weight identity mismatch for {spec.codename}")
    if model.get("dtype_counts") != checks.get("dtype_counts"):
        raise ValueError(f"Parity dtype identity mismatch for {spec.codename}")

    expected_protocol = {
        "dtype": "bfloat16",
        "decoding": "greedy",
        "seed": 17,
        "thinking": False,
        "max_new_tokens": 24,
        "max_context": 4096,
        "truncation": "forbidden",
        "prompts_sha256": DEMO_FILE_SHA256["all_prompts.jsonl"],
    }
    if evidence.get("protocol") != expected_protocol:
        raise ValueError(f"Pinned parity protocol mismatch for {spec.codename}")

    parity = evidence.get("parity")
    if not isinstance(parity, dict):
        raise TypeError(f"Parity metrics must be an object for {spec.codename}")
    integer_fields = (
        "raw_prediction_matches",
        "normalized_prediction_matches",
        "items",
    )
    for field in integer_fields:
        if type(parity.get(field)) is not int:
            raise TypeError(f"Parity {field} must be an integer for {spec.codename}")
    boolean_fields = ("exact_raw_parity", "exact_normalized_parity")
    for field in boolean_fields:
        if type(parity.get(field)) is not bool:
            raise TypeError(f"Parity {field} must be a boolean for {spec.codename}")
    if parity["items"] != 256:
        raise ValueError(f"Parity coverage is not 256 for {spec.codename}")
    raw_matches = parity["raw_prediction_matches"]
    normalized_matches = parity["normalized_prediction_matches"]
    if not (0 <= raw_matches <= normalized_matches <= 256):
        raise ValueError(f"Parity match counts are inconsistent for {spec.codename}")
    if parity["exact_raw_parity"] != (raw_matches == 256):
        raise ValueError(f"Raw parity flag/count mismatch for {spec.codename}")
    if parity["exact_normalized_parity"] != (normalized_matches == 256):
        raise ValueError(f"Normalized parity flag/count mismatch for {spec.codename}")
    differing_ids = parity.get("differing_ids")
    if (
        not isinstance(differing_ids, list)
        or any(not isinstance(value, str) for value in differing_ids)
        or len(differing_ids) != len(set(differing_ids))
        or len(differing_ids) != 256 - raw_matches
    ):
        raise ValueError(f"Parity differing_ids are inconsistent for {spec.codename}")
    if require_exact and not parity["exact_raw_parity"]:
        raise RuntimeError(f"Exact dense/PEFT parity required for {spec.codename}")

    dense_results = evidence.get("dense_results")
    if not isinstance(dense_results, dict):
        raise TypeError(f"Dense results must be an object for {spec.codename}")
    expected_items = {"showcase64": 64, "stability192": 192, "all256": 256}
    for split, items in expected_items.items():
        result = dense_results.get(split)
        if not isinstance(result, dict):
            raise TypeError(f"Dense {split} result must be an object")
        if type(result.get("items")) is not int or result["items"] != items:
            raise ValueError(f"Dense {split} coverage mismatch for {spec.codename}")
        if type(result.get("hits")) is not int:
            raise TypeError(f"Dense {split} hits must be an integer")
        hits = result["hits"]
        if hits < 0 or hits > items:
            raise ValueError(f"Dense {split} hit count is invalid for {spec.codename}")
        strict_exact = result.get("strict_exact")
        if (
            isinstance(strict_exact, bool)
            or not isinstance(strict_exact, (int, float))
            or not math.isfinite(float(strict_exact))
            or not math.isclose(float(strict_exact), hits / items, abs_tol=1e-12)
        ):
            raise ValueError(f"Dense {split} exact score is inconsistent")

    candidate = evidence.get("candidate")
    reference = evidence.get("peft_reference")
    report = evidence.get("dense_report")
    if not all(isinstance(value, dict) for value in (candidate, reference, report)):
        raise TypeError(
            "Publishable parity requires candidate, reference, and report objects"
        )
    candidate_path = resolve_repo_relative(
        candidate["path"], "dense prediction artifact"
    )
    reference_path = resolve_repo_relative(reference["path"], "PEFT reference artifact")
    report_path = resolve_repo_relative(report["path"], "dense report artifact")

    run_manifest = evidence.get("run_manifest")
    if not isinstance(run_manifest, dict):
        raise TypeError("Publishable parity requires a controlled run manifest")
    if run_manifest.get("mode") != CONTROLLED_PARITY_MODE:
        raise ValueError("Parity run manifest is not publisher-controlled")
    if run_manifest.get("core_id") != spec.core_id:
        raise ValueError("Parity run manifest core identity mismatch")
    if run_manifest.get("model_weight_sha256_before") != checks.get(
        "weight_sha256"
    ) or run_manifest.get("model_weight_sha256_after") != checks.get("weight_sha256"):
        raise ValueError("Parity run manifest model hash mismatch")
    expected_runner = {"path": rel_repo(RUNNER), "sha256": RUNNER_SHA256}
    expected_evaluator = {"path": rel_repo(EVALUATOR), "sha256": EVALUATOR_SHA256}
    if run_manifest.get("runner") != expected_runner:
        raise ValueError("Parity runner identity mismatch")
    if run_manifest.get("evaluator") != expected_evaluator:
        raise ValueError("Parity evaluator identity mismatch")
    require_sha256(RUNNER, RUNNER_SHA256, "frozen local HF runner")
    require_sha256(EVALUATOR, EVALUATOR_SHA256, "frozen Demo evaluator")
    invocation = run_manifest.get("invocation")
    if not isinstance(invocation, dict):
        raise TypeError("Parity run invocation must be an object")
    expected_invocation = {
        "model": spec.staging_path.name,
        "prompts": rel_repo(DEMO_DATA / "all_prompts.jsonl"),
        "predictions": candidate["path"],
        "report": report["path"],
        "dtype": "bfloat16",
        "decoding": "greedy",
        "seed": 17,
        "thinking": False,
        "max_new_tokens": 24,
        "max_context": 4096,
        "truncation": "forbidden",
        "max_gpu_memory": invocation.get("max_gpu_memory"),
        "max_cpu_memory": invocation.get("max_cpu_memory"),
    }
    if invocation != expected_invocation:
        raise ValueError("Parity run invocation does not match its artifacts/protocol")

    rebuilt = build_parity_evidence(
        spec,
        model_checks=checks,
        predictions_path=candidate_path,
        dense_report_path=report_path,
        reference_path=reference_path,
        run_manifest=run_manifest,
    )
    compared_fields = (
        "model",
        "protocol",
        "candidate",
        "peft_reference",
        "parity",
        "dense_results",
        "dense_report",
        "run_manifest",
        "publishable_provenance",
        "selection_caveat",
    )
    for field in compared_fields:
        if evidence.get(field) != rebuilt.get(field):
            raise ValueError(f"Parity evidence field was altered or is stale: {field}")


def finalized_metadata(
    spec: CoreSpec,
    core: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dense = evidence["dense_results"]
    parity = evidence["parity"]
    finalized_at = utc_now()
    core_out = dict(core)
    core_out.update(
        {
            "status": "showcase_selected_stability_validated",
            "public_claim": (
                f"BF16 dense Nimbus + Stage-A step-200 LoRA at scale {spec.scale}. "
                f"Public Demo: {dense['showcase64']['hits']}/64 Showcase (selection), "
                f"{dense['stability192']['hits']}/192 Stability (frozen validation), "
                f"{dense['all256']['hits']}/256 overall. Sealed TEST not run."
            ),
            "provisional": False,
            "finished_at_utc": finalized_at,
            "published_at_utc": finalized_at,
            "dense_validation": dict(evidence),
            "note": (
                "Dense validation is recorded. Showcase is selection data; Stability "
                "was the frozen validation split. The paired UNKNOWN gate remains a "
                "separate two-core inference recipe."
            ),
        }
    )
    manifest_out = dict(manifest)
    manifest_out.update(
        {
            "status": "finalized",
            "finalized_at_utc": finalized_at,
            "dense_parity": dict(evidence),
            "parity_classification": (
                "exact_raw_prediction_parity"
                if parity["exact_raw_parity"]
                else "measured_dense_prediction_drift"
            ),
        }
    )
    return core_out, manifest_out


def update_catalog_atomically(
    catalog_path: Path,
    published: Sequence[tuple[CoreSpec, Path]],
) -> dict[str, Any]:
    refuse_retired_legacy_publication(catalog_path=catalog_path)
    catalog = load_json(catalog_path)
    entries = catalog.get("cores")
    if not isinstance(entries, list):
        raise TypeError("CATALOG.json cores must be a list")
    # Keep every existing top-level key (especially `superseded`) and every
    # unrelated core entry byte-for-semantics.  Only exact new identities are
    # appended or refreshed.
    output_entries = list(entries)
    for spec, directory in published:
        core = load_json(directory / "CORE.json")
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(directory / "model.safetensors")
        if core.get("core_id") != spec.core_id:
            raise RuntimeError(f"Final CORE.json identity mismatch for {spec.codename}")
        if core.get("directory_name") != spec.directory:
            raise RuntimeError(
                f"Final CORE.json directory mismatch for {spec.codename}"
            )
        new_entry = {
            "core_id": spec.core_id,
            "codename": spec.codename,
            "directory": spec.directory,
            "present": (directory / "model.safetensors").is_file(),
            "core": core,
        }
        matching = [
            index
            for index, entry in enumerate(output_entries)
            if isinstance(entry, dict)
            and (
                entry.get("core_id") == spec.core_id
                or entry.get("directory") == spec.directory
            )
        ]
        conflicts = [
            entry
            for entry in output_entries
            if isinstance(entry, dict)
            and entry.get("codename") == spec.codename
            and entry.get("core_id") != spec.core_id
        ]
        if conflicts:
            raise RuntimeError(
                f"Catalog already assigns codename {spec.codename} to another core"
            )
        if len(matching) > 1:
            raise RuntimeError(
                f"Catalog has duplicate identity rows for {spec.codename}"
            )
        if matching:
            index = matching[0]
            old = output_entries[index]
            if (
                old.get("core_id") != spec.core_id
                or old.get("directory") != spec.directory
            ):
                raise RuntimeError(f"Catalog identity conflict for {spec.codename}")
            merged_entry = dict(old)
            merged_entry.update(new_entry)
            output_entries[index] = merged_entry
        else:
            output_entries.append(new_entry)

    ids = [entry.get("core_id") for entry in output_entries if isinstance(entry, dict)]
    directories = [
        entry.get("directory") for entry in output_entries if isinstance(entry, dict)
    ]
    if len(ids) != len(set(ids)) or len(directories) != len(set(directories)):
        raise RuntimeError("Catalog would contain duplicate core ids or directories")
    catalog["cores"] = output_entries
    catalog["updated_at_utc"] = utc_now()
    catalog["last_written"] = [spec.directory for spec, _ in published]
    atomic_write_json(catalog_path, catalog)
    return catalog


def finalize_many(specs: Sequence[CoreSpec], *, require_exact: bool) -> list[Path]:
    require_release_pair(specs, "finalize")
    refuse_publish_paths(specs, require_staged=True)
    archive_plan: list[tuple[Path, Path]] = []
    prepared: list[
        tuple[
            CoreSpec,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, str],
        ]
    ] = []
    for spec in specs:
        validate_pinned_sources(spec)
        checks = verify_dense_checkpoint(spec.staging_path, reload_model=True)
        if not spec.parity_path.is_file():
            raise FileNotFoundError(
                f"Dense parity evidence is required before finalization: {spec.parity_path}"
            )
        evidence = load_json(spec.parity_path)
        validate_parity_evidence(spec, evidence, checks, require_exact=require_exact)
        core = load_json(spec.staging_path / "CORE.json")
        manifest = load_json(spec.staging_path / "merge_manifest.json")
        if core.get("core_id") != spec.core_id or not core.get("provisional"):
            raise RuntimeError(f"Invalid provisional CORE.json for {spec.codename}")
        if manifest.get("core_id") != spec.core_id:
            raise RuntimeError(f"Invalid merge manifest for {spec.codename}")
        core_out, manifest_out = finalized_metadata(
            spec,
            core,
            manifest,
            evidence,
        )
        core_out["checks"] = dict(checks)
        manifest_output = dict(manifest_out.get("output") or {})
        manifest_output["checks"] = dict(checks)
        manifest_out["output"] = manifest_output
        nonpublishable_archives: list[str] = []
        for source in (spec.legacy_parity_path, spec.analysis_parity_path):
            if not source.is_file():
                continue
            destination = spec.archived_parity_path(source)
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite archived parity packet: {destination}"
                )
            archive_plan.append((source, destination))
            nonpublishable_archives.append(rel_repo(destination))
        if nonpublishable_archives:
            core_out["nonpublishable_parity_evidence_archives"] = (
                nonpublishable_archives
            )
            manifest_out["nonpublishable_parity_evidence_archives"] = (
                nonpublishable_archives
            )
        originals = {
            name: (spec.staging_path / name).read_text(encoding="utf-8")
            for name in ("CORE.json", "merge_manifest.json", "README.md")
        }
        prepared.append((spec, core_out, manifest_out, evidence, originals))

    promoted: list[tuple[CoreSpec, Path]] = []
    archived: list[tuple[Path, Path]] = []
    try:
        for source, destination in archive_plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            archived.append((source, destination))

        for spec, core, manifest, evidence, _ in prepared:
            atomic_write_json(spec.staging_path / "CORE.json", core)
            atomic_write_json(spec.staging_path / "merge_manifest.json", manifest)
            atomic_write_text(
                spec.staging_path / "README.md",
                render_model_readme(
                    spec,
                    provisional=False,
                    dense_results=evidence["dense_results"],
                ),
            )

        for spec, _, _, _, _ in prepared:
            if spec.final_path.exists():
                raise FileExistsError(spec.final_path)
            os.replace(spec.staging_path, spec.final_path)
            promoted.append((spec, spec.final_path))
        update_catalog_atomically(CATALOG, promoted)
    except Exception:
        for spec, final_path in reversed(promoted):
            if final_path.is_dir() and not spec.staging_path.exists():
                os.replace(final_path, spec.staging_path)
        for source, destination in reversed(archived):
            if destination.is_file() and not source.exists():
                os.replace(destination, source)
        # Leave a failed pair publication in its exact retryable provisional
        # state, even if failure happened after metadata preparation.
        for spec, _, _, _, originals in prepared:
            if spec.staging_path.is_dir():
                for name, content in originals.items():
                    atomic_write_text(spec.staging_path / name, content)
        raise
    return [path for _, path in promoted]


def selected_specs(values: Sequence[str]) -> list[CoreSpec]:
    if "all" in values and len(values) != 1:
        raise ValueError("Select 'all' by itself, or list individual cores")
    keys = list(CORE_SPECS) if "all" in values else list(values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate core selections: {keys}")
    return [CORE_SPECS[key] for key in keys]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    choices = [*CORE_SPECS, "all"]

    stage_parser = subparsers.add_parser(
        "stage", help="Archival action; refused after the Supa/Baya rename"
    )
    stage_parser.add_argument("cores", nargs="+", choices=choices)

    commands_parser = subparsers.add_parser(
        "commands", help="Archival action; refused after the Supa/Baya rename"
    )
    commands_parser.add_argument("cores", nargs="+", choices=choices)
    commands_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_PARITY_ROOT
    )

    record_parser = subparsers.add_parser(
        "record-parity",
        help=(
            "Record existing predictions for diagnostics only; final publication "
            "requires controlled run-parity"
        ),
    )
    record_parser.add_argument("core", choices=CORE_SPECS)
    record_parser.add_argument("--predictions", type=Path, required=True)
    record_parser.add_argument("--dense-report", type=Path)
    record_parser.add_argument("--require-exact-parity", action="store_true")

    run_parser = subparsers.add_parser(
        "run-parity", help="Run and record the frozen Demo-256 dense evaluation"
    )
    run_parser.add_argument("core", choices=CORE_SPECS)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_PARITY_ROOT)
    run_parser.add_argument("--max-gpu-memory", default="2GiB")
    run_parser.add_argument("--max-cpu-memory", default="32GiB")
    run_parser.add_argument("--require-exact-parity", action="store_true")

    finalize_parser = subparsers.add_parser(
        "finalize", help="Promote validated staging cores and update the catalog"
    )
    finalize_parser.add_argument("cores", nargs="+", choices=choices)
    finalize_parser.add_argument("--require-exact-parity", action="store_true")

    subparsers.add_parser("status", help="Inspect final/staging/parity state")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stage":
        specs = selected_specs(args.cores)
        for path in stage_many(specs):
            print(f"STAGED {path}")
        print("Dense parity is required before finalization.")
        return 0
    if args.command == "commands":
        specs = selected_specs(args.cores)
        require_release_pair(specs, "commands")
        for spec in specs:
            print(f"# {spec.codename}")
            for command in parity_commands(spec, output_root=args.output_root):
                print(command)
            print()
        print(
            f'& "{REPO / ".venv" / "Scripts" / "python.exe"}" '
            f'"{Path(__file__).resolve()}" finalize '
            + " ".join(spec.key for spec in specs)
        )
        return 0
    if args.command == "record-parity":
        path = record_parity(
            CORE_SPECS[args.core],
            args.predictions,
            args.dense_report,
            require_exact=args.require_exact_parity,
        )
        print(f"RECORDED ANALYSIS-ONLY {path}")
        print("Final publication still requires a fresh controlled run-parity packet.")
        return 0
    if args.command == "run-parity":
        path = run_parity(
            CORE_SPECS[args.core],
            output_root=args.output_root,
            max_gpu_memory=args.max_gpu_memory,
            max_cpu_memory=args.max_cpu_memory,
            require_exact=args.require_exact_parity,
        )
        print(f"RECORDED {path}")
        return 0
    if args.command == "finalize":
        specs = selected_specs(args.cores)
        for path in finalize_many(specs, require_exact=args.require_exact_parity):
            print(f"PUBLISHED {path}")
        return 0
    if args.command == "status":
        payload = {
            spec.key: {
                "core_id": spec.core_id,
                "final": spec.final_path.is_dir(),
                "staged": spec.staging_path.is_dir(),
                "publishable_parity_v2": spec.parity_path.is_file(),
                "legacy_parity_v1": spec.legacy_parity_path.is_file(),
                "analysis_only_parity": spec.analysis_parity_path.is_file(),
            }
            for spec in CORE_SPECS.values()
        }
        markers = legacy_publisher_retirement_markers(catalog_path=CATALOG)
        payload["_publisher"] = {
            "archival": True,
            "retired": bool(markers),
            "retirement_markers": markers,
            "canonical_successors": {
                key: {
                    **successor,
                    "present": (VAULT / successor["directory"]).is_dir(),
                }
                for key, successor in RENAMED_SUCCESSORS.items()
            },
        }
        print(json.dumps(payload, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
