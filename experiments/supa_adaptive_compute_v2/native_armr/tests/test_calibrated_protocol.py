"""Prospective v2.3 admission boundaries; synthetic artifacts, no real models."""
from copy import deepcopy
import json

import pytest

from native_armr.src import protocol
from shared import legacy
from shared.hashing import sha256_file


@pytest.fixture
def calibrated_fixture(monkeypatch, tmp_path):
    from native_armr.src import control_calibration
    root = tmp_path / "programme"
    monkeypatch.setattr(legacy, "REPO", tmp_path)
    monkeypatch.setattr(legacy, "ROOT", root)
    artifact = root / "native_armr/evidence/00_control_calibration/synthetic/report.json"
    artifact.parent.mkdir(parents=True)
    report = {"alpha": .25,
              "control": {"seed": 7301, "std": .02, "intermediate_size": 10240, "insertion_layer": 13},
              "checkpoint_files_before": {"model.safetensors": json.loads(protocol.DEFAULT_CONFIG.read_text())["checkpoint_sha256"]},
              "control_state_sha256": "a" * 64}
    artifact.write_text(json.dumps(report), encoding="utf-8")
    binding = {"artifact_path": artifact.relative_to(tmp_path).as_posix(),
               "artifact_sha256": sha256_file(artifact)}
    calls = []

    def verify(path, digest, *, require_committed=True):
        calls.append((path, digest, require_committed))
        assert path == artifact and digest == sha256_file(artifact)
        assert require_committed is True
        return deepcopy(report)

    monkeypatch.setattr(control_calibration, "verify_calibration_artifact", verify)
    config = protocol.calibrated_gpu_config(binding, report["alpha"])
    return config, artifact, calls, report


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_v23_changes_only_registered_control_calibration(calibrated_fixture, tmp_path):
    config, artifact, calls, report = calibrated_fixture
    loaded = protocol.load_protocol(write_config(tmp_path, config))
    reference = protocol.load_protocol(protocol.GPU_CONFIG)
    changed = {key for key in loaded if loaded[key] != reference.get(key)}
    assert changed == {"protocol_version", "runtime_qualification", "control", "control_calibration"}
    assert {key for key in loaded["control"] if loaded["control"][key] != reference["control"][key]} == {"alpha", "selection"}
    assert loaded["runtime_qualification"] == "native_gpu_fp32_v2_3"
    assert loaded["qualification"] == reference["qualification"]
    assert list(loaded["arms"]) == ["A0", "A1", "A3", "A16", "AM"]
    assert calls == [(artifact, config["control_calibration"]["artifact_sha256"], True)]


@pytest.mark.parametrize("mutation", ["arm", "data", "analysis", "seed", "width", "qualification", "device", "selection", "extra"])
def test_v23_rejects_all_unregistered_changes(calibrated_fixture, tmp_path, mutation):
    config, _, _, _ = calibrated_fixture
    if mutation == "arm":
        config["arms"]["A3"]["layers"] = [11, 13]
    elif mutation == "data":
        config["data"]["target_primary_examples"] = 64
    elif mutation == "analysis":
        config["analysis"]["minimum_absolute_accuracy_gain"] = 0
    elif mutation in {"seed", "width"}:
        config["control"]["seed" if mutation == "seed" else "intermediate_size"] += 1
    elif mutation == "qualification":
        config["qualification"]["require_greedy_identity"] = False
    elif mutation == "device":
        config["device"] = "cpu"
    elif mutation == "selection":
        config["control"]["selection"] = "architecture_accuracy"
    else:
        config["alpha_override"] = .1
    with pytest.raises(ValueError, match="v2.3 may change only"):
        protocol.load_protocol(write_config(tmp_path, config))


@pytest.mark.parametrize("alpha", [0., -.1, float("nan"), float("inf"), True, .125])
def test_v23_rejects_invalid_or_edited_alpha(calibrated_fixture, tmp_path, alpha):
    config, _, _, _ = calibrated_fixture
    config["control"]["alpha"] = alpha
    with pytest.raises(ValueError, match="alpha"):
        protocol.load_protocol(write_config(tmp_path, config))


@pytest.mark.parametrize("mutation", ["missing", "extra", "bad_hash", "outside", "noncanonical", "filename"])
def test_v23_rejects_unbound_artifacts(calibrated_fixture, mutation):
    config, _, _, _ = calibrated_fixture
    binding = config["control_calibration"]
    if mutation == "missing":
        config.pop("control_calibration")
    elif mutation == "extra":
        binding["alpha_override"] = .1
    elif mutation == "bad_hash":
        binding["artifact_sha256"] = "unbound"
    elif mutation == "outside":
        binding["artifact_path"] = "programme/cross_scale/evidence/report.json"
    elif mutation == "noncanonical":
        binding["artifact_path"] = binding["artifact_path"].replace("/synthetic/", "/synthetic/../synthetic/")
    else:
        binding["artifact_path"] = binding["artifact_path"].replace("report.json", "edited.json")
    with pytest.raises(ValueError):
        protocol.verify_control_calibration(config)


@pytest.mark.parametrize("key", ["seed", "checkpoint"])
def test_v23_rejects_artifact_from_different_operator_or_backbone(calibrated_fixture, key):
    config, _, _, report = calibrated_fixture
    if key == "seed":
        report["control"]["seed"] += 1
    else:
        report["checkpoint_files_before"]["model.safetensors"] = "b" * 64
    with pytest.raises(ValueError, match="calibration artifact"):
        protocol.verify_control_calibration(config)


def test_v23_requires_committed_artifact_before_caller_runs(calibrated_fixture, monkeypatch, tmp_path):
    from native_armr.src import control_calibration
    config, _, _, _ = calibrated_fixture

    def reject(*args, **kwargs):
        raise RuntimeError("calibration artifact is not committed")

    monkeypatch.setattr(control_calibration, "verify_calibration_artifact", reject)
    with pytest.raises(RuntimeError, match="not committed"):
        protocol.load_protocol(write_config(tmp_path, config))


def test_v23_named_path_and_exclusive_derived_config_freeze(calibrated_fixture, monkeypatch, tmp_path):
    from native_armr.scripts import freeze_calibrated_control as freeze
    config, artifact, calls, _ = calibrated_fixture
    original_bytes = [path.read_bytes() for path in (protocol.DEFAULT_CONFIG, protocol.GPU_CONFIG)]
    output = tmp_path / "protocol_gpu_v2_3.json"
    monkeypatch.setattr(protocol, "CALIBRATED_GPU_CONFIG", output)
    # The real script's programme root is fixed by its location.
    monkeypatch.setattr(freeze, "PROGRAM", legacy.ROOT)
    assert freeze.freeze_calibrated_control(artifact) == output
    assert protocol.require_config_path(output) == output.resolve()
    assert protocol.load_protocol(output) == config
    with pytest.raises(FileExistsError):
        freeze.freeze_calibrated_control(artifact)
    with pytest.raises(ValueError, match="registered CPU or GPU protocol"):
        protocol.require_config_path(tmp_path / "copied.json")
    assert original_bytes == [path.read_bytes() for path in (protocol.DEFAULT_CONFIG, protocol.GPU_CONFIG)]
    assert all(call[2] is True for call in calls)


def test_operator_uses_bound_alpha_and_identical_transform_for_cpu_qualifier(calibrated_fixture, monkeypatch):
    from native_armr.src import operators
    from shared import integrity
    config, _, _, report = calibrated_fixture
    config["device"] = "cpu"  # The generic qualifier makes this runtime-only copy.
    seen = []

    class FakeControl:
        def __init__(self, model, **kwargs):
            seen.append(kwargs)
            self.transform = object()

    monkeypatch.setattr(operators, "NativeComputeControl", FakeControl)
    monkeypatch.setattr(integrity, "state_hash", lambda _: report["control_state_sha256"])
    assert operators.operator_factory(object(), "AM", config).arm == "AM"
    assert seen[0]["alpha"] == .25
    assert seen[0]["seed"] == 7301 and seen[0]["std"] == .02
    assert seen[0]["intermediate_size"] == 10240
    monkeypatch.setattr(integrity, "state_hash", lambda _: "wrong-transform")
    with pytest.raises(ValueError, match="tensors differ"):
        operators.operator_factory(object(), "AM", config)
