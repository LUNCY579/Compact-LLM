import pytest
from experiments.multi_reasoning_scout_v1 import budget, core, freeze, runner


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, None, "100000000000"])
def test_invalid_resource_measurement_is_not_a_pass(value):
    observed = dict(host_available_bytes=value, gpu_free_bytes=10*budget.GIB,
                    disk_free_bytes=10*budget.GIB, host_rss_bytes=budget.GIB,
                    gpu_allocated_bytes=budget.GIB)
    with pytest.raises(RuntimeError, match="measurement"):
        budget.check(observed, preload=True)


@pytest.mark.parametrize("tokens", [0, -1, 1.5, True])
def test_allowance_cannot_be_refunded(tmp_path, tokens):
    meter = budget.Meter(tmp_path)
    with pytest.raises(ValueError):
        meter.reserve(tokens)
    assert meter.calls == meter.allowance == 0


def test_actual_registered_bank_has_no_lexical_only_world_isolation():
    receipt = freeze.verify_bank()
    assert receipt["structural_worlds_without_names"] == receipt["cases"] == 120


def test_future_registration_cannot_alias_old_checkpoint(monkeypatch, tmp_path):
    old = {"id": "d1-seed17-bf16-export", "path": str(tmp_path/"old"),
           "assets": {"model.safetensors": "a"*64, "config.json": "b"*64},
           "policy": "same", "runtime_dtype": "float32", "storage_dtype": ["BF16"],
           "attention_backend": "eager", "decoding": {}}
    new = {**old, "id": "future", "path": str(tmp_path/"new")}
    registration = {"frozen_protocol_sha256": "f"*64, "model": new, "training_provenance": {}}
    monkeypatch.setattr(runner, "read_json", lambda path: registration if str(path) == "future" else {"selected": [old]})
    monkeypatch.setattr(runner, "file_hash", lambda path: "f"*64)
    with pytest.raises(ValueError, match="distinct weight bytes"):
        runner.candidate("future", "future")
