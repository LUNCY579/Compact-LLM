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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_exact_adapter_soup.py"
SPEC = importlib.util.spec_from_file_location("build_exact_adapter_soup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


MODULE_PREFIXES = (
    "base_model.model.model.layers.0.self_attn.q_proj",
    "base_model.model.model.layers.0.mlp.down_proj",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_config(base: Path, *, rank: int = 2) -> dict[str, Any]:
    return {
        "alpha_pattern": {},
        "base_model_name_or_path": str(base.resolve()),
        "bias": "none",
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
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


def make_adapter(
    path: Path,
    base: Path,
    *,
    seed: int,
    rank: int = 2,
    config_updates: dict[str, Any] | None = None,
    drop_key: str | None = None,
    nonfinite_key: str | None = None,
    shape_mismatch: bool = False,
) -> dict[str, np.ndarray]:
    path.mkdir()
    config = adapter_config(base, rank=rank)
    config.update(config_updates or {})
    (path / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    dimensions = ((3, 4), (4, 3))
    for prefix, (in_features, out_features) in zip(MODULE_PREFIXES, dimensions):
        a_key = prefix + MODULE.A_SUFFIX
        b_key = prefix + MODULE.B_SUFFIX
        tensors[a_key] = rng.normal(size=(rank, in_features)).astype(np.float32)
        tensors[b_key] = rng.normal(size=(out_features, rank)).astype(np.float32)
    if shape_mismatch:
        key = MODULE_PREFIXES[0] + MODULE.A_SUFFIX
        tensors[key] = rng.normal(size=(rank, 5)).astype(np.float32)
    if drop_key is not None:
        tensors.pop(drop_key)
    if nonfinite_key is not None:
        tensors[nonfinite_key][0, 0] = np.nan
    save_file(tensors, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    return tensors


def dense_delta(path: Path, prefix: str, scale: float) -> np.ndarray:
    with safe_open(str(path / "adapter_model.safetensors"), framework="numpy") as handle:
        a = handle.get_tensor(prefix + MODULE.A_SUFFIX).astype(np.float64)
        b = handle.get_tensor(prefix + MODULE.B_SUFFIX).astype(np.float64)
    return scale * (b @ a)


def test_builds_exact_negative_nonnormalized_concat_atomically(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_adapter(first, base, seed=7)
    make_adapter(second, base, seed=19)
    source_hashes = {
        first: file_hash(first / "adapter_model.safetensors"),
        second: file_hash(second / "adapter_model.safetensors"),
    }
    output = tmp_path / "soup"
    coefficients = (1.25, -0.375)

    manifest = MODULE.build_exact_adapter_soup(
        [(first, coefficients[0]), (second, coefficients[1])],
        output,
        verification_mode="all",
    )

    assert output.is_dir()
    assert not list(tmp_path.glob(".soup.tmp-*"))
    assert file_hash(first / "adapter_model.safetensors") == source_hashes[first]
    assert file_hash(second / "adapter_model.safetensors") == source_hashes[second]

    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["r"] == 4
    assert config["lora_alpha"] == 4
    assert config["use_rslora"] is False
    assert config["rank_pattern"] == {}
    assert config["alpha_pattern"] == {}

    for prefix in MODULE_PREFIXES:
        desired = coefficients[0] * dense_delta(first, prefix, 2.0)
        desired += coefficients[1] * dense_delta(second, prefix, 2.0)
        actual = dense_delta(output, prefix, 1.0)
        np.testing.assert_allclose(actual, desired, rtol=2e-6, atol=2e-6)

    saved_manifest = json.loads(
        (output / "soup_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["coefficient_sum"] == pytest.approx(0.875)
    assert [row["coefficient"] for row in saved_manifest["sources"]] == list(
        coefficients
    )
    assert saved_manifest["output"]["rank"] == 4
    assert saved_manifest["output"]["effective_scale"] == 1.0
    assert saved_manifest["verification"]["checked_module_count"] == 2
    assert saved_manifest["verification"]["worst_relative_frobenius_error"] < 2e-6
    assert manifest == saved_manifest


def test_rejects_base_identity_mismatch_without_output(tmp_path: Path) -> None:
    base_a = tmp_path / "base_a"
    base_b = tmp_path / "base_b"
    base_a.mkdir()
    base_b.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_adapter(first, base_a, seed=1)
    make_adapter(second, base_b, seed=2)
    output = tmp_path / "soup"

    with pytest.raises(MODULE.SoupValidationError, match="Base-model identity mismatch"):
        MODULE.build_exact_adapter_soup([(first, 0.5), (second, 0.5)], output)

    assert not output.exists()


def test_rejects_config_and_tensor_topology_mismatches(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    first = tmp_path / "first"
    config_bad = tmp_path / "config_bad"
    shape_bad = tmp_path / "shape_bad"
    make_adapter(first, base, seed=1)
    make_adapter(
        config_bad,
        base,
        seed=2,
        config_updates={"target_modules": ["q_proj"]},
    )
    make_adapter(shape_bad, base, seed=3, shape_mismatch=True)

    with pytest.raises(MODULE.SoupValidationError, match="configuration/topology"):
        MODULE.build_exact_adapter_soup(
            [(first, 0.5), (config_bad, 0.5)], tmp_path / "config_output"
        )
    with pytest.raises(MODULE.SoupValidationError, match="tensor topology mismatch"):
        MODULE.build_exact_adapter_soup(
            [(first, 0.5), (shape_bad, 0.5)], tmp_path / "shape_output"
        )


@pytest.mark.parametrize("defect", ["unpaired", "nonfinite"])
def test_rejects_unpaired_or_nonfinite_tensors(tmp_path: Path, defect: str) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    b_key = MODULE_PREFIXES[0] + MODULE.B_SUFFIX
    a_key = MODULE_PREFIXES[0] + MODULE.A_SUFFIX
    kwargs: dict[str, Any] = {}
    if defect == "unpaired":
        kwargs["drop_key"] = b_key
    else:
        kwargs["nonfinite_key"] = a_key
    make_adapter(source, base, seed=4, **kwargs)

    expected = "unpaired" if defect == "unpaired" else "NaN or infinite"
    with pytest.raises(MODULE.SoupValidationError, match=expected):
        MODULE.build_exact_adapter_soup([(source, 1.0)], tmp_path / "output")


def test_refuses_existing_or_source_nested_output(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "source"
    make_adapter(source, base, seed=5)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        MODULE.build_exact_adapter_soup([(source, 1.0)], existing)
    assert marker.read_text(encoding="utf-8") == "untouched"

    with pytest.raises(MODULE.SoupValidationError, match="must not be inside"):
        MODULE.build_exact_adapter_soup([(source, 1.0)], source / "nested")
    assert not (source / "nested").exists()
