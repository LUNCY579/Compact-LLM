"""v2.4 provenance/admission and unequal-budget regressions; no real models."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from native_armr.src import protocol
from shared import legacy
from shared.hashing import sha256_file


@pytest.fixture
def learned_fixture(monkeypatch, tmp_path):
    from native_armr.src import learned_control_calibration
    root = tmp_path / "programme"
    monkeypatch.setattr(legacy, "REPO", tmp_path)
    monkeypatch.setattr(legacy, "ROOT", root)
    artifact = root / "native_armr/evidence/00_learned_control_calibration/synthetic/report.json"
    artifact.parent.mkdir(parents=True)
    report = {
        "alpha": .25, "control": deepcopy(protocol.LEARNED_CONTROL_METADATA),
        "checkpoint_files_before": {"model.safetensors": json.loads(protocol.DEFAULT_CONFIG.read_text())["checkpoint_sha256"]},
        "control_state_sha256": "a" * 64,
        "branch_provenance": {str(i): {"source": "b" * 64, "copy": "b" * 64} for i in (12, 13, 14)},
        "source_hashes": {str(i): {"mlp": "b" * 64, "post_attention_layernorm": "c" * 64} for i in (12, 13, 14)},
        "copy_hashes": {str(i): {"mlp": "b" * 64, "post_attention_layernorm": "c" * 64} for i in (12, 13, 14)},
        "parameter_inventory": {"synthetic_count": 28314624},
    }
    artifact.write_text(json.dumps(report), encoding="utf-8")
    binding = {"artifact_path": artifact.relative_to(tmp_path).as_posix(),
               "artifact_sha256": sha256_file(artifact)}
    calls = []

    def verify(path, digest, *, require_committed=True):
        calls.append((path, digest, require_committed))
        assert path == artifact and digest == sha256_file(artifact)
        assert require_committed is True
        return deepcopy(report)

    monkeypatch.setattr(learned_control_calibration, "verify_calibration_artifact", verify)
    return protocol.learned_gpu_config(binding, report["alpha"]), artifact, calls, report


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_v24_only_changes_control_and_its_binding(learned_fixture, tmp_path):
    config, artifact, calls, _ = learned_fixture
    loaded = protocol.load_protocol(write_config(tmp_path, config))
    reference = protocol.load_protocol(protocol.GPU_CONFIG)
    assert {key for key in loaded if loaded[key] != reference.get(key)} == {
        "protocol_version", "runtime_qualification", "control", "control_calibration", "arms"}
    assert {arm for arm in loaded["arms"] if loaded["arms"][arm] != reference["arms"][arm]} == {"AM"}
    assert loaded["qualification"] == reference["qualification"]
    assert loaded["analysis"] == reference["analysis"]
    assert loaded["data"] == reference["data"]
    assert loaded["runtime_qualification"] == "native_gpu_fp32_v2_4"
    assert tuple(loaded["arms"]) == ("A0", "A1", "A3", "A16", "AM")
    assert "seed" not in loaded["control"] and "std" not in loaded["control"]
    assert calls == [(artifact, config["control_calibration"]["artifact_sha256"], True)]


@pytest.mark.parametrize("mutation", ["native_arm", "branch", "order", "coefficient", "width", "norm", "copies", "budget", "data", "analysis", "qualification", "device", "guard", "extra"])
def test_v24_rejects_unregistered_scientific_or_runtime_changes(learned_fixture, tmp_path, mutation):
    config, _, _, _ = learned_fixture
    if mutation == "native_arm":
        config["arms"]["A3"]["layers"] = [11, 13]
    elif mutation == "branch":
        config["control"]["source_layers"] = [12, 13, 16]
    elif mutation == "order":
        config["control"]["summation_order"] = "F12+(F13+F14)"
    elif mutation == "coefficient":
        config["control"]["branch_coefficients"] = [1, 1, .5]
    elif mutation == "width":
        config["control"]["intermediate_size"] = 4096
    elif mutation == "norm":
        config["control"]["normalization"] = "shared_norm"
    elif mutation == "copies":
        config["control"]["additional_parameters"] -= 3072
    elif mutation == "budget":
        config["control"]["additional_matmul_flops_per_token"] = 62914560
    elif mutation == "data":
        config["data"]["target_primary_examples"] = 64
    elif mutation == "analysis":
        config["analysis"]["minimum_absolute_accuracy_gain"] = 0
    elif mutation == "qualification":
        config["qualification"]["require_greedy_identity"] = False
    elif mutation == "device":
        config["device"] = "cpu"
    elif mutation == "guard":
        config["hidden_rms_ratio_bounds"] = [.01, 10]
    else:
        config["alpha_override"] = .1
    with pytest.raises(ValueError, match="v2.4 may change only"):
        protocol.load_protocol(write_config(tmp_path, config))


@pytest.mark.parametrize("alpha", [0., -.1, float("nan"), float("inf"), True, .125])
def test_v24_rejects_invalid_or_edited_alpha(learned_fixture, tmp_path, alpha):
    config, _, _, _ = learned_fixture
    config["control"]["alpha"] = alpha
    with pytest.raises(ValueError, match="alpha"):
        protocol.load_protocol(write_config(tmp_path, config))


@pytest.mark.parametrize("mutation", ["missing", "extra", "bad_hash", "outside", "noncanonical", "filename"])
def test_v24_rejects_unbound_artifacts(learned_fixture, mutation):
    config, _, _, _ = learned_fixture
    binding = config["control_calibration"]
    if mutation == "missing":
        config.pop("control_calibration")
    elif mutation == "extra":
        binding["alpha_override"] = .1
    elif mutation == "bad_hash":
        binding["artifact_sha256"] = "unbound"
    elif mutation == "outside":
        binding["artifact_path"] = "programme/native_armr/evidence/00_control_calibration/synthetic/report.json"
    elif mutation == "noncanonical":
        binding["artifact_path"] = binding["artifact_path"].replace("/synthetic/", "/synthetic/../synthetic/")
    else:
        binding["artifact_path"] = binding["artifact_path"].replace("report.json", "edited.json")
    with pytest.raises(ValueError):
        protocol.verify_learned_control_calibration(config)


@pytest.mark.parametrize("key", ["control", "checkpoint"])
def test_v24_rejects_artifact_from_different_operator_or_backbone(learned_fixture, key):
    config, _, _, report = learned_fixture
    if key == "control":
        report["control"]["source_layers"] = [12, 13, 16]
    else:
        report["checkpoint_files_before"]["model.safetensors"] = "b" * 64
    with pytest.raises(ValueError, match="calibration artifact"):
        protocol.verify_learned_control_calibration(config)


def test_v24_uncommitted_artifact_cannot_create_final_config(learned_fixture, monkeypatch, tmp_path):
    from native_armr.src import learned_control_calibration
    from native_armr.scripts import freeze_learned_control as freeze
    config, artifact, _, _ = learned_fixture
    output = tmp_path / "protocol_gpu_v2_4.json"
    monkeypatch.setattr(protocol, "LEARNED_GPU_CONFIG", output)
    monkeypatch.setattr(freeze, "PROGRAM", legacy.ROOT)

    def reject(*args, **kwargs):
        raise RuntimeError("calibration artifact is not committed")

    monkeypatch.setattr(learned_control_calibration, "verify_calibration_artifact", reject)
    with pytest.raises(RuntimeError, match="not committed"):
        protocol.load_protocol(write_config(tmp_path, config))
    with pytest.raises(RuntimeError, match="not committed"):
        freeze.freeze_learned_control(artifact)
    assert not output.exists()


def test_v24_exclusive_derived_config_preserves_all_earlier_configs(learned_fixture, monkeypatch, tmp_path):
    from native_armr.scripts import freeze_learned_control as freeze
    config, artifact, calls, _ = learned_fixture
    originals = (protocol.DEFAULT_CONFIG, protocol.GPU_CONFIG, protocol.CALIBRATED_GPU_CONFIG)
    before = [path.read_bytes() for path in originals]
    output = tmp_path / "protocol_gpu_v2_4.json"
    monkeypatch.setattr(protocol, "LEARNED_GPU_CONFIG", output)
    monkeypatch.setattr(freeze, "PROGRAM", legacy.ROOT)
    assert freeze.freeze_learned_control(artifact) == output
    assert protocol.require_config_path(output) == output.resolve()
    assert protocol.load_protocol(output) == config
    with pytest.raises(FileExistsError):
        freeze.freeze_learned_control(artifact)
    with pytest.raises(ValueError, match="registered CPU or GPU protocol"):
        protocol.require_config_path(tmp_path / "copied.json")
    assert before == [path.read_bytes() for path in originals]
    assert all(call[2] is True for call in calls)


@pytest.mark.parametrize("corruption", [None, "state", "branch", "source", "copy", "inventory", "actual_count", "actual_work"])
def test_factory_verifies_scale_copies_and_actual_state_before_cpu_qualification_forward(learned_fixture, monkeypatch, corruption):
    from native_armr.src import learned_control, operators
    from shared import integrity
    config, _, _, report = learned_fixture
    config["device"] = "cpu"  # The existing qualifier's runtime-only copy.
    seen = []

    class FakeControl:
        def __init__(self, model, **kwargs):
            seen.append(kwargs)
            self.transform = SimpleNamespace(**{key: deepcopy(report[key]) for key in (
                "branch_provenance", "source_hashes", "copy_hashes", "parameter_inventory")})
            self.added_parameter_count = self.added_active_parameters = 28314624
            self.extra_matmul_flops_per_token = 56623104
            if corruption in {"branch", "source", "copy", "inventory"}:
                field = {"branch": "branch_provenance", "source": "source_hashes", "copy": "copy_hashes", "inventory": "parameter_inventory"}[corruption]
                setattr(self.transform, field, {"tampered": True})
            elif corruption == "actual_count":
                self.added_parameter_count -= 3072
            elif corruption == "actual_work":
                self.extra_matmul_flops_per_token = 62914560

    monkeypatch.setattr(learned_control, "NativeFFNControl", FakeControl)
    monkeypatch.setattr(integrity, "state_hash", lambda _: "wrong-state" if corruption == "state" else report["control_state_sha256"])
    if corruption:
        with pytest.raises(ValueError, match="(differ|differs)"):
            operators.operator_factory(object(), "AM", config)
    else:
        assert operators.operator_factory(object(), "AM", config).arm == "AM"
        assert seen == [{"alpha": .25, "emergency_bounds": (.25, 4.)}]


def test_audit_native_cost_is_independent_of_control_budget_and_counts_norm_copies(learned_fixture):
    from native_armr.src.audit import expected_operator_inventory
    config, _, _, _ = learned_fixture
    for arm, work in {"A0": 0, "A1": 31457280, "A3": 62914560, "A16": 31457280}.items():
        assert expected_operator_inventory(config, arm) == {
            "added_parameter_count": 0, "added_active_parameters": 0,
            "extra_matmul_flops_per_token": work}
    assert expected_operator_inventory(config, "AM") == {
        "added_parameter_count": 28314624, "added_active_parameters": 28314624,
        "extra_matmul_flops_per_token": 56623104}
    old = protocol.load_protocol(protocol.DEFAULT_CONFIG)
    assert expected_operator_inventory(old, "AM") == {
        "added_parameter_count": 31457280, "added_active_parameters": 31457280,
        "extra_matmul_flops_per_token": 62914560}


def test_analysis_reports_actual_resident_norm_copies_and_approximate_matching(learned_fixture):
    from native_armr.src.analysis import analyze_complete_bank
    from shared.datasets import generate_rows
    from test_native_analysis import evidence
    config, _, _, _ = learned_fixture
    config["analysis"]["bootstrap_replicates"] = 50
    cases = generate_rows("native_armr", "architecture_dev")
    report = analyze_complete_bank(cases, evidence(cases, lambda arm, i, c: True), config, "architecture_dev")
    assert report["quality_cost_frontier"]["arms"]["AM"]["added_parameters"] == 28314624
    matching = report["control_compute_matching"]
    assert matching["normalization_parameters"] == 3072
    assert matching["newly_learned_parameters"] == 0
    assert matching["logical_unique_coefficients_beyond_backbone"] == 0
    assert matching["control_added_matmul_flops_per_token"] / matching["a3_added_matmul_flops_per_token"] == .9
    assert report["decision"] == "MECHANICS_PASS_NO_EFFICACY_CLAIM"
    assert report["efficacy_claim_licensed"] is False
