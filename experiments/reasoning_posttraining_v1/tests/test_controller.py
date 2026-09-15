"""Fixture gates for the supervised controller and its all-attempt budget journal.

None of these tests load pretrained weights, reach CUDA or perform an optimizer step.
They cover the refusals that must hold *before* the native lane is ever entered.
"""
import json
from pathlib import Path

import pytest

from experiments.model_quality_gpu_v1.integrity import digest, read_json, write_new
from experiments.reasoning_posttraining_v1 import controller
from experiments.reasoning_posttraining_v1.budget import Journal, validate_grant
from experiments.reasoning_posttraining_v1.preflight import LIMITS

DATA_SHA = "a" * 64
PLAN_SHA = "b" * 64
EXCLUDES = ["broad_SFT", "RL", "larger_teacher", "cloud", "release", "architectural_change"]


def grant(**over) -> dict:
    value = {"experiment": "RRP-001-MICROFIT", "decision": "APPROVED",
             "approved_by": "operator", "authorization_reference": "session-grant",
             "data_contract_sha256": DATA_SHA, "plan_sha256": PLAN_SHA,
             "limits": dict(LIMITS), "excludes": list(EXCLUDES),
             "training_permission": True}
    value.update(over)
    return value


def terminal(wall=1.0) -> dict:
    return {"status": "EXITED", "returncode": 0, "child_pid": 4242,
            "wall_seconds": wall, "cleanup": None}


# ---------------------------------------------------------------- grant validation

def test_valid_engineering_grant_is_accepted():
    validate_grant(grant(), DATA_SHA, PLAN_SHA)


@pytest.mark.parametrize("bad,match", [
    ({"decision": "PENDING"}, "approval is absent"),
    ({"training_permission": False}, "approval is absent"),
    ({"approved_by": None}, "approval is absent"),
    ({"approved_by": "   "}, "approval is absent"),
    ({"authorization_reference": ""}, "approval is absent"),
    ({"experiment": "RRP-001-LADDER-B"}, "approval is absent"),
    ({"data_contract_sha256": "c" * 64}, "exact prepared data"),
    ({"plan_sha256": "c" * 64}, "exact prepared data"),
    ({"excludes": ["broad_SFT"]}, "later research lanes"),
    ({"excludes": EXCLUDES[:-1] + ["something_else"]}, "later research lanes"),
])
def test_grant_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_grant(grant(**bad), DATA_SHA, PLAN_SHA)


def test_grant_cannot_expand_or_blur_any_numeric_limit():
    # An approval may be smaller than the ceiling, never larger.
    validate_grant(grant(limits={**LIMITS, "model_worker_launches": 1}), DATA_SHA, PLAN_SHA)
    for key in ("model_worker_launches", "processed_training_tokens_all_attempts"):
        with pytest.raises(ValueError, match="invalid or expanded"):
            validate_grant(grant(limits={**LIMITS, key: LIMITS[key] + 1}), DATA_SHA, PLAN_SHA)
    for value in (1.0, True, "4", None):
        with pytest.raises(ValueError, match="invalid or expanded"):
            validate_grant(grant(limits={**LIMITS, "model_worker_launches": value}),
                           DATA_SHA, PLAN_SHA)
    missing = {k: v for k, v in LIMITS.items() if k != "retries"}
    with pytest.raises(ValueError, match="every numeric budget"):
        validate_grant(grant(limits=missing), DATA_SHA, PLAN_SHA)
    with pytest.raises(ValueError, match="every numeric budget"):
        validate_grant(grant(limits={**LIMITS, "extra_lane": 0}), DATA_SHA, PLAN_SHA)


def test_rl_and_cloud_lanes_are_pinned_shut_at_zero():
    assert LIMITS["on_policy_rl_rollout_tokens"] == 0
    assert LIMITS["cloud_or_teacher_spend"] == 0
    for key in ("on_policy_rl_rollout_tokens", "cloud_or_teacher_spend"):
        with pytest.raises(ValueError, match="invalid or expanded"):
            validate_grant(grant(limits={**LIMITS, key: 1}), DATA_SHA, PLAN_SHA)


# ---------------------------------------------------------------- journal lifecycle

def test_journal_binds_one_grant_and_is_create_only(tmp_path):
    g = grant()
    Journal(tmp_path / "j", g, create=True)
    Journal(tmp_path / "j", g)                       # reopening the same grant is fine
    with pytest.raises(ValueError, match="different grant"):
        Journal(tmp_path / "j", grant(approved_by="someone else"))
    with pytest.raises(FileExistsError):
        Journal(tmp_path / "j", g, create=True)


def test_unsettled_launch_retains_reservation_and_blocks_the_next(tmp_path):
    j = Journal(tmp_path / "j", grant(), create=True)
    j.start("qualify-001", "qualify", 300)
    reserved = j.read()["used_or_reserved"]
    assert reserved["model_worker_launches"] == 1
    assert reserved["model_worker_wall_seconds_and_conservative_gpu_seconds"] == 320
    with pytest.raises(RuntimeError, match="unsettled prior worker"):
        j.start("train-001", "train", 600)
    # A fresh process sees the same unsettled reservation.
    assert Journal(tmp_path / "j", grant()).active == {"qualify-001": j.active["qualify-001"]}


def test_settlement_replaces_reservation_with_measured_wall_time(tmp_path):
    j = Journal(tmp_path / "j", grant(), create=True)
    j.start("qualify-001", "qualify", 600)
    j.settle("qualify-001", terminal(wall=42.2), "c" * 64, owned_wall=50.9)
    used = j.read()["used_or_reserved"]
    # 620 reserved collapses to the measured 43s, so the next launch regains headroom.
    assert used["model_worker_wall_seconds_and_conservative_gpu_seconds"] == 43
    assert used["total_owned_work_wall_seconds"] == 51
    assert not j.read()["active_reservations"]
    j.start("train-001", "train", 1500)              # now affordable
    assert j.read()["used_or_reserved"]["adaptive_model_selection_rounds"] == 1


def test_settlement_refuses_unverified_or_impossible_receipts(tmp_path):
    j = Journal(tmp_path / "j", grant(), create=True)
    j.start("qualify-001", "qualify", 300)
    with pytest.raises(RuntimeError, match="termination unverified"):
        j.settle("qualify-001", {**terminal(), "cleanup": {"child_exited": False}}, "c" * 64, 5.0)
    with pytest.raises(RuntimeError, match="no observed exit code"):
        j.settle("qualify-001", {**terminal(), "returncode": None}, "c" * 64, 5.0)
    with pytest.raises(ValueError, match="cannot omit worker time"):
        j.settle("qualify-001", terminal(wall=90.0), "c" * 64, owned_wall=10.0)
    with pytest.raises(ValueError, match="no matching launch"):
        j.settle("other-run", terminal(), "c" * 64, 5.0)
    assert j.read()["active_reservations"] == ["qualify-001"]


def test_exhausted_budget_refuses_before_the_charge_lands(tmp_path):
    j = Journal(tmp_path / "j", grant(limits={**LIMITS, "processed_training_tokens_all_attempts": 100}),
                create=True)
    j.reserve({"processed_training_tokens_all_attempts": 100})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        j.reserve({"processed_training_tokens_all_attempts": 1})
    assert j.read()["used_or_reserved"]["processed_training_tokens_all_attempts"] == 100
    assert not j.breached
    with pytest.raises(ValueError, match="invalid charge"):
        j.reserve({"unknown_lane": 1})
    with pytest.raises(ValueError, match="invalid charge"):
        j.reserve({"processed_training_tokens_all_attempts": -1})


def test_edited_event_and_mid_chain_gap_are_refused(tmp_path):
    j = Journal(tmp_path / "j", grant(), create=True)
    for amount in (3, 4, 5):
        j.reserve({"evaluation_and_qualification_model_calls": amount})
    events = tmp_path / "j" / "events"
    original = (events / "000001.json").read_bytes()

    body = read_json(events / "000001.json")
    body["charges"]["evaluation_and_qualification_model_calls"] = 1
    (events / "000001.json").write_bytes(json.dumps(body).encode())
    with pytest.raises(ValueError, match="digest mismatch"):
        Journal(tmp_path / "j", grant())

    (events / "000001.json").write_bytes(original)
    assert Journal(tmp_path / "j", grant()).used["evaluation_and_qualification_model_calls"] == 12
    (events / "000001.json").unlink()
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        Journal(tmp_path / "j", grant())


def test_tail_truncation_is_a_documented_limitation_not_a_detection(tmp_path):
    """Deleting the newest event leaves a valid prefix, so spend can be silently undone.

    Create-only files cannot carry a rewritable head pointer, and any extra marker file
    would be equally deletable. The mitigation that does hold is asserted below: a launch
    cannot be erased while its run evidence exists, which covers every worker attempt.
    """
    j = Journal(tmp_path / "j", grant(), create=True)
    j.reserve({"evaluation_and_qualification_model_calls": 9})
    (tmp_path / "j" / "events" / "000000.json").unlink()
    assert Journal(tmp_path / "j", grant()).used["evaluation_and_qualification_model_calls"] == 0

    j = Journal(tmp_path / "j2", grant(), create=True)
    j.start("qualify-001", "qualify", 300)
    (tmp_path / "j2" / "runs" / "qualify-001").mkdir()
    (tmp_path / "j2" / "events" / "000000.json").unlink()
    with pytest.raises(ValueError, match="no corresponding budget reservation"):
        Journal(tmp_path / "j2", grant())


def test_run_evidence_without_a_reservation_is_refused(tmp_path):
    Journal(tmp_path / "j", grant(), create=True)
    (tmp_path / "j" / "runs" / "smuggled-001").mkdir()
    with pytest.raises(ValueError, match="no corresponding budget reservation"):
        Journal(tmp_path / "j", grant())


def test_reused_run_id_and_unsafe_run_id_are_refused(tmp_path):
    j = Journal(tmp_path / "j", grant(), create=True)
    j.start("qualify-001", "qualify", 300)
    j.settle("qualify-001", terminal(), "c" * 64, 5.0)
    # Caught before the duplicate event is written, so the journal stays readable.
    with pytest.raises(ValueError, match="reused run ID"):
        j.start("qualify-001", "qualify", 300)
    assert Journal(tmp_path / "j", grant()).used["model_worker_launches"] == 1
    for bad in ("../escape", "Qualify", "with space", "", "x" * 61):
        with pytest.raises(ValueError, match="unsafe run ID"):
            j.start(bad, "qualify", 300)
    with pytest.raises(ValueError, match="invalid phase"):
        j.start("ladder-001", "ladder_b", 300)
    for bad_timeout in (0, -1, 1.5):
        with pytest.raises(ValueError, match="invalid phase/timeout"):
            j.start("qualify-002", "qualify", bad_timeout)


def test_three_planned_launches_fit_the_approved_envelope(tmp_path):
    """qualify -> train -> resume must be affordable under the real ceiling."""
    j = Journal(tmp_path / "j", grant(), create=True)
    for run_id, phase, timeout, actual in [("qualify-001", "qualify", 600, 250.0),
                                           ("train-001", "train", 1500, 900.0),
                                           ("resume-001", "resume", 1200, 850.0)]:
        j.start(run_id, phase, timeout)
        j.settle(run_id, terminal(wall=actual), "c" * 64, owned_wall=actual + 30)
    state = j.read()
    assert not state["budget_breached"]
    assert state["used_or_reserved"]["model_worker_launches"] == 3
    assert state["remaining"]["model_worker_launches"] == 1      # one retry still available


# ---------------------------------------------------------------- plan/data binding

def plan_fixture(rows):
    from experiments.reasoning_posttraining_v1.preflight import build_plan
    contract = {"microfit_source": "forge_v1_primitives", "selected_worlds": 32,
                "selected_ids": [r["id"] for r in rows]}
    return build_plan(contract, rows)


def rows_fixture():
    return [{"id": f"w{i:02d}", "parent_id": f"w{i:02d}", "source_split": "train",
             "mode": "direct", "truncated": False, "ids": [5, 6, 7, 2],
             "labels": [-100, -100, 7, 2], "prompt_ids": [5, 6]} for i in range(32)]


def write_prepared(tmp_path, rows):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    body = b"".join(json.dumps(r).encode() + b"\n" for r in rows)
    (prepared / "microfit-train.jsonl").write_bytes(body)
    import hashlib
    write_new(prepared / "DATA_CONTRACT.json",
              {"files": {"microfit-train.jsonl": hashlib.sha256(body).hexdigest()}})
    return prepared


def test_design_tracks_the_frozen_plan_and_rejects_a_changed_recipe():
    plan = plan_fixture(rows_fixture())
    d = controller.design_for(plan)
    assert d.processed_budget == plan["processed_tokens"] == 1024
    assert [d.beta1, d.beta2] == plan["optimizer"]["betas"]
    for key, value in [("peak_lr", 3e-5), ("gradient_clip", 2), ("weight_decay", 0.1)]:
        with pytest.raises(ValueError, match="frozen plan disagrees"):
            controller.design_for({**plan, "optimizer": {**plan["optimizer"], key: value}})
    with pytest.raises(ValueError, match="frozen plan disagrees"):
        controller.design_for({**plan, "max_sequence": 1024})


def test_presentations_recompile_the_plan_inventory(tmp_path):
    rows = rows_fixture()
    plan = plan_fixture(rows)
    prepared = write_prepared(tmp_path, rows)
    probe, repeated = controller.presentations(prepared, plan)
    assert len(probe) == 32 and len(repeated) == plan["presentations"] == 256
    assert sum(len(r["ids"]) for r in repeated) == plan["processed_tokens"]


@pytest.mark.parametrize("bad", ["order", "split", "mode", "truncated", "exposures", "worlds"])
def test_presentations_refuse_any_drift_from_the_plan(tmp_path, bad):
    rows = rows_fixture()
    plan = plan_fixture(rows)
    if bad == "order":
        rows = list(reversed(rows))
    if bad == "split":
        rows[0] = {**rows[0], "source_split": "validation"}
    if bad == "mode":
        rows[0] = {**rows[0], "mode": "reasoning"}
    if bad == "truncated":
        rows[0] = {**rows[0], "truncated": True}
    if bad == "worlds":
        rows[0] = {**rows[0], "parent_id": rows[1]["parent_id"]}
    if bad == "exposures":
        plan = {**plan, "exposures_per_world": 9}
    with pytest.raises(ValueError):
        controller.presentations(write_prepared(tmp_path, rows), plan)


def test_changed_prepared_bytes_refuse_before_any_model_work(tmp_path):
    rows = rows_fixture()
    plan = plan_fixture(rows)
    prepared = write_prepared(tmp_path, rows)
    write_new(tmp_path / "plan.json", plan)
    write_new(tmp_path / "grant.json",
              grant(data_contract_sha256="x", plan_sha256=digest(plan)))

    class Args:
        pass
    args = Args()
    args.prepared, args.plan = str(prepared), str(tmp_path / "plan.json")
    args.grant = str(tmp_path / "grant.json")
    (prepared / "microfit-train.jsonl").write_bytes(b'{"id": "tampered"}\n')
    with pytest.raises(ValueError, match="prepared artifact changed"):
        controller.inputs(args)


# ---------------------------------------------------------------- identity and entry

def test_package_identity_is_stable_and_content_sensitive(tmp_path, monkeypatch):
    first = controller.package_identity()
    assert first == controller.package_identity()
    assert "controller.py" in first["files"] and "budget.py" in first["files"]
    assert not any("__pycache__" in name for name in first["files"])
    monkeypatch.setattr(controller, "PACKAGE", tmp_path)
    (tmp_path / "only.py").write_bytes(b"x = 1\n")
    assert controller.package_identity()["sha256"] != first["sha256"]


def test_checkout_requires_the_dedicated_clean_branch(monkeypatch):
    calls = {"branch": controller.BRANCH, "status": "", "top": str(controller.ROOT)}

    def fake(args, **kw):
        # args == ["git", "-C", <root>, <subcommand>, ...]
        assert args[:3] == ["git", "-C", str(controller.ROOT)]
        if args[3] == "rev-parse" and args[4] == "--show-toplevel":
            return calls["top"] + "\n"
        if args[3] == "branch":
            return calls["branch"] + "\n"
        if args[3] == "status":
            return calls["status"] + "\n"
        return "f" * 40 + "\n"

    monkeypatch.setattr(controller.subprocess, "check_output", fake)
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: None)
    assert controller.checkout() == "f" * 40

    calls["branch"] = "main"
    with pytest.raises(ValueError, match="requires research/rineya"):
        controller.checkout()

    calls["branch"] = controller.BRANCH
    calls["status"] = "?? experiments/reasoning_posttraining_v1/scratch.py"
    with pytest.raises(ValueError, match="clean checkout"):
        controller.checkout()


def test_eos_set_must_be_complete_and_contain_the_tokenizer_eos(tmp_path):
    class Tok:
        eos_token_id = 151645

    write_new(tmp_path / "generation_config.json", {"eos_token_id": [151645, 151643]})
    assert controller.eos_set(tmp_path, Tok()) == [151645, 151643]

    other = tmp_path / "other"
    other.mkdir()
    write_new(other / "generation_config.json", {"eos_token_id": [151643]})
    with pytest.raises(ValueError, match="invalid or incomplete"):
        controller.eos_set(other, Tok())


def test_private_worker_refuses_without_a_supervisor_launch(tmp_path, capsys):
    class Args:
        pass
    args = Args()
    args.out = str(tmp_path / "unsupervised")
    (tmp_path / "unsupervised").mkdir()
    assert controller.worker(args) == 2
    assert "refused" in capsys.readouterr().err
    # Nothing was written into the output directory by the refused worker.
    assert not list((tmp_path / "unsupervised").iterdir())


@pytest.mark.parametrize("argv,match", [
    (["qualify"], "--source is required"),
    (["train", "--source", "s", "--prepared", "p", "--plan", "l", "--grant", "g",
      "--journal", "j", "--run-id", "r", "--timeout", "60"], "--confirm-model-execution"),
    (["resume", "--source", "s", "--prepared", "p", "--plan", "l", "--grant", "g",
      "--journal", "j", "--run-id", "r", "--timeout", "60",
      "--confirm-model-execution"], "explicit --resume checkpoint"),
])
def test_model_commands_refuse_incomplete_invocations(argv, match, capsys):
    with pytest.raises(SystemExit):
        controller.main(argv)
    assert match in capsys.readouterr().err


def test_phases_match_the_journal_and_exclude_unregistered_lanes():
    assert controller.PHASES == ("qualify", "train", "resume")
    assert controller.DISK_FLOOR == 14 * 2 ** 30
