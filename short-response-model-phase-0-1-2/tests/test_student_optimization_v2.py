from __future__ import annotations

import json
from pathlib import Path

from scripts.student_optimization_v2.context_budget_sweep import split_dev, group_key
from scripts.student_optimization_v2.state_manager import default_state, render_report, save_state


ROOT = Path(__file__).resolve().parents[1]


def test_v2_config_has_complete_efficiency_ladder() -> None:
    config = json.loads((ROOT / "configs/student_optimization_v2.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == "2.0"
    assert config["layer_sensitivity"]["enabled"] is True
    assert config["context_sweep"]["enabled"] is True
    assert len(config["context_sweep"]["profiles"]) >= 4
    assert config["sparse_runtime"]["enabled"] is True
    assert "bitsandbytes_nf4" in config["quantization_sweep"]["backends"]
    assert config["release"]["complete_only_when_package_exists"] is True


def test_phase10_v2_has_importance_and_uniform_controls() -> None:
    config = json.loads((ROOT / "configs/phase_10_student_architecture_v2.json").read_text(encoding="utf-8"))
    full = {row["name"]: row for row in config["candidate_sets"]["full"]}
    assert full["depth_drop_1_importance"]["depth_mode"] == "importance-drop"
    assert full["depth_drop_1_uniform_control"]["depth_mode"] == "uniform-drop"
    assert full["combined_small_target"]["depth_mode"] == "importance-drop"
    assert full["combined_deployment_target"]["target_parameter_max"] == 120_000_000


def test_phase10_runner_supports_external_scan_and_custom_registry() -> None:
    text = (ROOT / "scripts/phase_10_student_architecture/run_phase_10.ps1").read_text(encoding="utf-8")
    assert "[string]$StructureProfile" in text
    assert "[string]$LayerSensitivityScan" in text
    assert 'if ($RequestedDepthMode -eq "importance-drop")' in text
    assert '"--drop-layer-ids"' in text
    assert '"--registry", $Registry' in text


def test_structural_candidate_supports_explicit_layer_deletion() -> None:
    text = (ROOT / "scripts/phase_10_student_architecture/create_structural_candidate.py").read_text(encoding="utf-8")
    assert '"explicit-drop"' in text
    assert '"--drop-layer-ids"' in text
    assert 'retained_current = [index for index in range(original_layer_count) if index not in set(explicit)]' in text


def test_context_dev_split_is_group_disjoint_and_deterministic() -> None:
    rows = []
    for group in range(12):
        for index in range(3):
            rows.append({"id": f"g{group}_{index}", "group_id": f"g{group}"})
    left1, right1 = split_dev(rows, 0.5)
    left2, right2 = split_dev(rows, 0.5)
    assert [row["id"] for row in left1] == [row["id"] for row in left2]
    assert [row["id"] for row in right1] == [row["id"] for row in right2]
    assert {group_key(row) for row in left1}.isdisjoint({group_key(row) for row in right1})
    assert len(left1) + len(right1) == len(rows)


def test_state_report_never_defaults_to_complete(tmp_path: Path) -> None:
    state = default_state()
    assert state["status"] == "NOT_STARTED"
    assert state["terminal"] is False
    state["status"] = "WAITING_FOR_PHASE10_HUMAN_PASS"
    state["checkpoint"] = "WAITING_FOR_PHASE10_HUMAN_PASS"
    path = tmp_path / "state.json"
    report = tmp_path / "report.md"
    save_state(path, state)
    render_report(state, report)
    text = report.read_text(encoding="utf-8")
    assert "WAITING_FOR_PHASE10_HUMAN_PASS" in text
    assert "COMPLETE means" in text or "`COMPLETE` means" in text
    assert state["terminal"] is False


def test_master_runner_contains_human_checkpoint_and_release_completion_semantics() -> None:
    text = (ROOT / "scripts/student_optimization_v2/run_student_optimization_v2.ps1").read_text(encoding="utf-8")
    for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"):
        assert f'"{stage}"' in text
    assert "WAITING_FOR_PHASE10_HUMAN_PASS" in text
    assert "WAITING_FOR_PHASE11_HUMAN_PASS" in text
    assert "run_phase_10.ps1" in text
    assert "run_phase_11.ps1" in text
    assert "context_budget_sweep.py" in text
    assert "sparse_student_pipeline.py" in text
    assert "quantization_sweep.py" in text
    assert "phase_11_release.zip" in text
    assert "Update-State -Action complete" in text


def test_selection_stages_explicitly_reject_sealed_inputs() -> None:
    context = (ROOT / "scripts/student_optimization_v2/context_budget_sweep.py").read_text(encoding="utf-8")
    sparse = (ROOT / "scripts/student_optimization_v2/sparse_student_pipeline.py").read_text(encoding="utf-8")
    quant = (ROOT / "scripts/student_optimization_v2/quantization_sweep.py").read_text(encoding="utf-8")
    scan = (ROOT / "scripts/student_optimization_v2/scan_student_layer_sensitivity.py").read_text(encoding="utf-8")
    for text in (context, sparse, quant, scan):
        assert "sealed" in text.casefold()
        assert "test" in text.casefold()


def test_master_runner_preserves_human_gates_on_resume() -> None:
    text = (ROOT / "scripts/student_optimization_v2/run_student_optimization_v2.ps1").read_text(encoding="utf-8")
    assert 'Test-Path $Phase10Registry' in text
    assert '-not $ForcePhase10Rerun' in text
    assert 'preserving the human gate' in text
    assert 'Test-Path $Phase11Registry' in text
    assert '-not $ForcePhase11Rerun' in text
    assert 'human gate was preserved' in text
