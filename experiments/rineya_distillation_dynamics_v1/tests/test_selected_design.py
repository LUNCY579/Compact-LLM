"""Contract-only fabricated fixtures: no model loads, gradients or optimizer updates."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments.rineya_distillation_dynamics_v1 import cli, registry
from experiments.rineya_distillation_dynamics_v1.contracts import (
    Arm, ContractError, Plan, file_hash, matrix, read_json, sealed,
    unseal, validate_arm, write_new)
from experiments.rineya_distillation_dynamics_v1.selection import select


CONFIG = Path(__file__).parents[1] / "configs" / "selected-pilot.json"


def selected_plan():
    return Plan.load(CONFIG)


def fabricated_selected_runs(root, plan):
    """Selection metadata fixtures; adapter bytes are intentionally not real weights."""
    paths = []
    for arm in matrix(plan):
        directory = root / arm.name
        checkpoints = []
        for step in range(1, plan.steps + 1):
            checkpoint = directory / "checkpoints" / f"{step:06d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter.safetensors").write_bytes(b"CONTRACT_FIXTURE_NOT_WEIGHTS")
            receipt = {"step": step, "adapter_sha256": file_hash(checkpoint / "adapter.safetensors")}
            write_new(checkpoint / "receipt.json", sealed(receipt))
            checkpoints.append({"step": step, "receipt": receipt, "meter": {}, "summary": {"groups": {
                "dev_acquisition/fresh": {"gain": .1, "n": 10, "damaged": 0, "correct": 7},
                "dev_retention/independent": {"gain": 0, "n": 10, "damaged": 0, "correct": 7},
                "dev_hard/fresh-transfer": {"gain": .1, "n": 10, "damaged": 0, "correct": 7}}}})
        result = {"status": "COMPLETE", "plan": plan.fingerprint, "identity": "contract-fixture-id",
                  "arm": asdict(arm), "initial_adapter": "paired-seed-17", "frozen": {"fixture": True},
                  "evidence_kind": "mechanical_fixture", "checkpoints": checkpoints}
        path = directory / "RESULT.json"
        write_new(path, sealed(result))
        paths.append(path)
    return paths


def test_selected_contrasts_and_control_are_explicit():
    plan = selected_plan()
    arms = matrix(plan)
    assert len(arms) == 4 and matrix(plan, "selected") == arms
    assert {a.name: (a.source, a.objective) for a in arms} == {
        "A-lr0-s17": ("teacher", "forward"), "B-lr0-s17": ("student", "forward"),
        "D-lr0-s17": ("student", "reverse"), "SFT-lr0-s17": ("solution", "ce")}
    assert {a.learning_rate for a in arms} == {1e-5}
    assert plan.steps == 2 and plan.checkpoint_every == 1
    for arm in arms:
        validate_arm(plan, arm)
    with pytest.raises(ContractError, match="frozen treatment"):
        validate_arm(plan, Arm("C-lr0-s17", "teacher", "reverse", 1e-5, 17))


@pytest.mark.parametrize("change", [
    {"selected_cells": ()}, {"selected_cells": ("A", "B", "D")},
    {"selected_cells": ("A", "B", "C", "D", "SFT")},
    {"selected_cells": ("A", "B", "C", "SFT")},
    {"selected_cells": ("A", "B", "D", "D", "SFT")},
    {"selected_cells": ("B", "A", "D", "SFT")},
    {"learning_rates": (1e-5, 2e-5)}, {"learning_rates": ()},
    {"learning_rates": (float("nan"),)}, {"learning_rates": (True,)},
    {"seeds": (17, 29)}, {"checkpoint_every": 2}])
def test_selected_plan_rejects_ambiguous_or_widened_scope(change):
    with pytest.raises(ContractError):
        replace(selected_plan(), **change)


@pytest.mark.parametrize("stage", ["screen", "full"])
def test_selected_matrix_cannot_be_widened(stage):
    with pytest.raises(ContractError, match="widen"):
        matrix(selected_plan(), stage)


def test_factorial_behavior_and_selected_scope_remain_separate():
    assert len(matrix(Plan())) == 30
    assert len(matrix(Plan(), "screen")) == 18
    with pytest.raises(ContractError, match="widen"):
        matrix(Plan(), "selected")
    with pytest.raises(ContractError, match="cannot carry"):
        Plan(selected_cells=("A", "B", "D", "SFT"))


def test_plan_cli_is_offline_and_auto_resolves_selected(capsys):
    assert cli.main(["plan", "--plan", str(CONFIG)]) == 0
    result = __import__("json").loads(capsys.readouterr().out)
    assert result["stage"] == "selected" and len(result["arms"]) == 4
    assert result["native_execution"] is False
    assert cli.main(["plan", "--plan", str(CONFIG), "--stage", "full"]) == 2
    assert "widen" in capsys.readouterr().out


def test_complete_selected_selection_includes_sft_and_per_step_checkpoints(tmp_path):
    plan = selected_plan()
    paths = fabricated_selected_runs(tmp_path / "runs", plan)
    chosen = select(plan, paths, tmp_path / "selection", identity="contract-fixture-id")
    assert chosen["stage"] == "selected" and chosen["candidates_examined"] == 8
    assert chosen["evidence_kind"] == "mechanical_fixture"
    assert chosen["promotion_authorized"] is False


@pytest.mark.parametrize("failure", ["missing_control", "missing_step", "full", "screen", "failed_receipt", "restore_receipt"])
def test_selected_selection_fails_closed(tmp_path, failure):
    plan = selected_plan()
    paths = fabricated_selected_runs(tmp_path / "runs", plan)
    if failure == "missing_control":
        paths.pop()
    if failure == "missing_step":
        record = unseal(read_json(paths[0]))
        record["checkpoints"].pop()
        paths[0].write_text(__import__("json").dumps(sealed(record)), encoding="utf-8")
    if failure in {"failed_receipt", "restore_receipt"}:
        marker = "FAILED.json" if failure == "failed_receipt" else "RESTORE_FAILED.json"
        write_new(paths[0].parent / marker, {"mechanical_fixture": True})
    kwargs = {"stage": failure} if failure in {"full", "screen"} else {}
    with pytest.raises(ContractError):
        select(plan, paths, tmp_path / "selection", identity="contract-fixture-id", **kwargs)
    assert not (tmp_path / "selection").exists()


def test_selected_registration_blocks_unreviewed_learning_rate_before_asset_access(tmp_path, monkeypatch):
    plan = selected_plan()
    plan_path, binding_path = tmp_path / "plan.json", tmp_path / "binding.json"
    write_new(plan_path, asdict(plan))
    binding = read_json(CONFIG.parent / "native-binding.example.json")
    assert binding["learning_rates_reviewed"] is False
    write_new(binding_path, binding)
    monkeypatch.setattr(registry, "inventory", lambda *_: pytest.fail("unreviewed LR must fail before asset access"))
    with pytest.raises(ContractError, match="explicit review"):
        registry.register(plan_path, binding_path, tmp_path / "registration.json")
    assert not (tmp_path / "registration.json").exists()


@pytest.mark.parametrize("change", ["missing_scope", "missing_control", "unreviewed_lr"])
def test_selected_registration_rejects_scope_drift_before_asset_access(tmp_path, monkeypatch, change):
    plan = selected_plan()
    payload = {"schema": 1, "source": registry.source_identity(), "plan": asdict(plan),
               "plan_sha256": plan.fingerprint, "stage_scope": "selected",
               "treatment_cells": [asdict(arm) for arm in matrix(plan)],
               "binding": {"learning_rates_reviewed": True, "reviewed_by": "contract-fixture-reviewer"}}
    if change == "missing_scope":
        payload.pop("stage_scope")
    if change == "missing_control":
        payload["treatment_cells"].pop()
    if change == "unreviewed_lr":
        payload["binding"]["learning_rates_reviewed"] = False
    path = tmp_path / "registration.json"
    write_new(path, sealed(payload))
    monkeypatch.setattr(registry, "inventory", lambda *_: pytest.fail("scope drift must fail before asset access"))
    with pytest.raises(ContractError, match="scope|review"):
        registry.check_registration(path)


def test_qualify_cli_routes_only_to_supervised_disposable_mode(tmp_path, monkeypatch):
    from experiments.rineya_distillation_dynamics_v1 import host
    calls = []
    monkeypatch.setattr(host, "launch", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert cli.main(["qualify", "--registration", "not-opened", "--output", str(tmp_path / "future"),
                     "--allow-model-load", "--allow-training", "--ack-host-available"]) == 0
    assert len(calls) == 1
    assert calls[0][1]["qualification_only"] is True
    assert calls[0][1]["arm"] is None and calls[0][1]["selection"] is None
    assert calls[0][1]["allow_training"] is True
    assert not (tmp_path / "future").exists()
