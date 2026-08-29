from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_rank_pruned_adapter.py"
SPEC = importlib.util.spec_from_file_location("build_rank_pruned_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


PREFIXES = (
    "base_model.model.model.layers.0.self_attn.q_proj",
    "base_model.model.model.layers.0.mlp.down_proj",
)
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(base: Path, rank: int) -> dict[str, Any]:
    return {
        "alpha_pattern": {},
        "base_model_name_or_path": str(base.resolve()),
        "bias": "none",
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "layer_replication": None,
        "lora_alpha": rank * 2,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["q_proj", "down_proj"],
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def write_adapter(
    path: Path,
    base: Path,
    tensors: dict[str, np.ndarray],
    rank: int,
) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(config(base, rank), indent=2) + "\n", encoding="utf-8"
    )
    save_file(
        {key: np.ascontiguousarray(value.astype(np.float32)) for key, value in tensors.items()},
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


def controlled_tensors(rank: int = 4) -> dict[str, np.ndarray]:
    assert rank == 4
    tensors: dict[str, np.ndarray] = {}
    for offset, prefix in enumerate(PREFIXES, 1):
        a = np.diag(np.array([1.0, 2.0, 3.0, 4.0]) * offset)
        b = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        tensors[prefix + A_SUFFIX] = a
        tensors[prefix + B_SUFFIX] = b
    return tensors


def random_tensors(rank: int, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for prefix in PREFIXES:
        tensors[prefix + A_SUFFIX] = rng.normal(size=(rank, 5)).astype(np.float32)
        tensors[prefix + B_SUFFIX] = rng.normal(size=(4, rank)).astype(np.float32)
    return tensors


def load_pair(path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    with safe_open(str(path / "adapter_model.safetensors"), framework="numpy") as handle:
        return handle.get_tensor(prefix + A_SUFFIX), handle.get_tensor(prefix + B_SUFFIX)


def dense_norm(path: Path, prefix: str, scale: float) -> float:
    a, b = load_pair(path, prefix)
    return scale * float(
        np.linalg.norm(b.astype(np.float64) @ a.astype(np.float64))
    )


def test_keeps_top_components_and_original_scale_without_rescale(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    source_tensors = controlled_tensors()
    write_adapter(source, base, source_tensors, rank=4)
    source_hash = file_hash(source / "adapter_model.safetensors")
    output = tmp_path / "rank2"

    manifest = MODULE.build_rank_pruned_adapter(source, 2, output)

    assert file_hash(source / "adapter_model.safetensors") == source_hash
    assert not list(tmp_path.glob(".rank2.tmp-*"))
    output_config = json.loads(
        (output / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert output_config["r"] == 2
    assert output_config["lora_alpha"] == 4
    assert output_config["lora_alpha"] / output_config["r"] == 2.0

    for prefix in PREFIXES:
        a_out, b_out = load_pair(output, prefix)
        np.testing.assert_array_equal(a_out, source_tensors[prefix + A_SUFFIX][[2, 3]])
        np.testing.assert_array_equal(b_out, source_tensors[prefix + B_SUFFIX][:, [2, 3]])
    assert manifest["pruning"]["norm_preserving_rescale"] is False
    assert manifest["pruning"]["kept_component_energy_fraction"] == pytest.approx(
        25 / 30
    )
    for row in manifest["pruning"]["modules"]:
        assert row["kept_indices"] == [2, 3]
        assert row["norm_rescale_factor"] == 1.0
        assert row["dense_delta_frobenius_before"] is None
        assert row["kept_component_energy_fraction"] == pytest.approx(25 / 30)
    saved = json.loads(
        (output / "rank_pruning_manifest.json").read_text(encoding="utf-8")
    )
    assert saved == manifest
    assert saved["source"]["adapter_model_sha256"] == source_hash
    assert saved["output"]["adapter_model_sha256"] == file_hash(
        output / "adapter_model.safetensors"
    )


def test_optional_rescale_preserves_each_dense_delta_norm(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    write_adapter(source, base, random_tensors(3), rank=3)
    output = tmp_path / "rank1_rescaled"

    manifest = MODULE.build_rank_pruned_adapter(
        source, 1, output, norm_preserving_rescale=True
    )

    output_config = json.loads(
        (output / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert output_config["r"] == 1
    assert output_config["lora_alpha"] == 2
    for prefix in PREFIXES:
        assert dense_norm(output, prefix, 2.0) == pytest.approx(
            dense_norm(source, prefix, 2.0), rel=2e-6, abs=2e-6
        )
    assert manifest["verification"]["norm_preservation_verified"] is True
    assert manifest["verification"]["worst_norm_relative_error"] < 2e-6
    assert all(
        row["norm_rescale_factor"] > 0.0
        for row in manifest["pruning"]["modules"]
    )


@pytest.mark.parametrize("target_rank", [0, -1, 5])
def test_rejects_invalid_target_rank(tmp_path: Path, target_rank: int) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    write_adapter(source, base, controlled_tensors(), rank=4)
    output = tmp_path / "output"

    with pytest.raises(MODULE.SoupValidationError, match="target rank"):
        MODULE.build_rank_pruned_adapter(source, target_rank, output)

    assert not output.exists()


def test_source_validation_rejects_nonfinite_tensor(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    tensors = controlled_tensors()
    tensors[PREFIXES[0] + A_SUFFIX][0, 0] = np.nan
    write_adapter(source, base, tensors, rank=4)

    with pytest.raises(MODULE.SoupValidationError, match="NaN or infinite"):
        MODULE.build_rank_pruned_adapter(source, 2, tmp_path / "output")


def test_refuses_existing_or_source_nested_output(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    write_adapter(source, base, controlled_tensors(), rank=4)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        MODULE.build_rank_pruned_adapter(source, 2, existing)
    assert marker.read_text(encoding="utf-8") == "keep"

    with pytest.raises(MODULE.SoupValidationError, match="must not be inside"):
        MODULE.build_rank_pruned_adapter(source, 2, source / "nested")
    assert not (source / "nested").exists()
