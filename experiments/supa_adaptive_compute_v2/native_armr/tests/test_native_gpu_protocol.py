"""GPU registration changes execution resources while retaining native science."""
import json

import pytest

from native_armr.src.protocol import (DEFAULT_CONFIG, GPU_CONFIG,
                                      GPU_RUNTIME_AMENDMENT, load_protocol,
                                      require_config_path)


def test_gpu_protocol_preserves_all_reference_scientific_choices():
    reference = load_protocol(DEFAULT_CONFIG)
    gpu = load_protocol(GPU_CONFIG)
    assert gpu == {**reference, **GPU_RUNTIME_AMENDMENT}
    assert gpu["dtype"] == "float32" and gpu["attention_implementation"] == "eager"
    assert gpu["threads"] == 4 and gpu["use_cache"] is False and gpu["tf32"] is False
    assert gpu["runtime_qualification"] == "native_gpu_fp32_v1"
    assert gpu["min_available_gib"] < reference["min_available_gib"]
    assert gpu["min_available_vram_gib"] > 0
    assert gpu["max_cuda_memory_gib"] > gpu["min_available_vram_gib"]


@pytest.mark.parametrize("mutation", ["arm", "data", "analysis", "control", "qualification", "tf32"])
def test_gpu_protocol_rejects_science_changes_or_relaxed_qualification(tmp_path, mutation):
    gpu = json.loads(GPU_CONFIG.read_text(encoding="utf-8"))
    if mutation == "arm":
        gpu["arms"]["A16"]["layer"] = 17
    elif mutation == "data":
        gpu["data"]["target_primary_examples"] = 64
    elif mutation == "analysis":
        gpu["analysis"]["minimum_absolute_accuracy_gain"] = 0
    elif mutation == "control":
        gpu["control"]["seed"] = 7302
    elif mutation == "qualification":
        del gpu["runtime_qualification"]
    else:
        gpu["tf32"] = True
    path = tmp_path / "unregistered_gpu.json"
    path.write_text(json.dumps(gpu), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        load_protocol(path)


def test_native_execution_accepts_only_named_protocol_paths(tmp_path):
    assert require_config_path(DEFAULT_CONFIG) == DEFAULT_CONFIG.resolve()
    assert require_config_path(GPU_CONFIG) == GPU_CONFIG.resolve()
    copied = tmp_path / GPU_CONFIG.name
    copied.write_bytes(GPU_CONFIG.read_bytes())
    with pytest.raises(ValueError, match="registered CPU or GPU protocol"):
        require_config_path(copied)


def test_gpu_caller_passes_exact_protocol_to_shared_runner(monkeypatch, tmp_path):
    from native_armr.scripts import run_large_eval
    from shared import evaluation
    calls = []

    def run_shard(**kwargs):
        calls.append(kwargs)
        return tmp_path / "synthetic_run"

    monkeypatch.setattr(evaluation, "run_shard", run_shard)
    assert run_large_eval.main(["--config", str(GPU_CONFIG), "--manifest",
                                str(tmp_path / "manifest.json"), "--arm", "A0",
                                "--role", "architecture_dev"]) == 0
    assert len(calls) == 1
    assert calls[0]["config_path"] == GPU_CONFIG.resolve()
    assert calls[0]["config"]["device"] == "cuda:0"
    assert calls[0]["shard_count"] == 32
