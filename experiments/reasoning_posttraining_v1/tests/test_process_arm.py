"""RRP-007 process arm: the invariants that make the scored result mean anything."""
from __future__ import annotations

import json

import pytest

from experiments.model_quality_gpu_v1.config import Design
from experiments.reasoning_posttraining_v1 import budget, process_arm
from experiments.reasoning_posttraining_v1.arm import frozen_optimizer


def view(parent, family, system="RULE TEXT"):
    return {"id": f"RRP-process:{parent}-step0-valid", "parent_id": parent, "family": family,
            "prompt": [{"role": "system", "content": system},
                       {"role": "user", "content": '{"before":{"A":1},"event":{"kind":"add"}}'}],
            "reference": '{"A":2}', "source_id": f"{parent}-step0-valid",
            "source_split": "train", "view_role": "CANDIDATE_NOT_ADMITTED_OR_TRAINED"}


def write_views(tmp_path, rows):
    path = tmp_path / "CANDIDATE_VIEWS.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def compiled_row(row_id, *, parent=None, family="ledger_replay", ids=None):
    ids = ids or [1, 2, 3, 4]
    return {"id": row_id, "parent_id": parent or row_id, "family": family,
            "source_split": "train", "mode": "direct", "truncated": False,
            "ids": ids, "prompt_ids": ids[:2],
            "labels": [-100, -100] + ids[2:], "processed_tokens": len(ids),
            "supervised_tokens": len(ids) - 2}


ALL_FAMILIES = ("ledger_replay", "prerequisite_plan", "register_machine")


def test_the_arm_registers_its_own_ceiling_and_does_not_widen_rrp001():
    assert process_arm.EXPERIMENT in budget.REGISTERED
    entry = budget.REGISTERED[process_arm.EXPERIMENT]
    assert entry["ceiling"] is not budget.REGISTERED["RRP-001-MICROFIT"]["ceiling"]
    assert entry["ceiling"]["on_policy_rl_rollout_tokens"] == 0
    assert entry["ceiling"]["cloud_or_teacher_spend"] == 0
    assert entry["phases"] == ("train", "resume")
    # RL, teachers, cloud, release and architecture stay out of this lane.
    assert set(entry["excludes"]) == {"RL", "larger_teacher", "cloud", "release",
                                      "architectural_change"}


def test_the_controller_accepts_the_plan_identity():
    from experiments.reasoning_posttraining_v1 import controller
    assert process_arm.EXPERIMENT in controller.REGISTERED_PLANS


def test_rules_are_read_from_the_views_so_train_and_eval_cannot_drift(tmp_path):
    rows = [view(f"p{i}", fam, system=f"{fam} rule")
            for fam in ALL_FAMILIES for i in range(2)]
    rules = process_arm.rules_from_views(write_views(tmp_path, rows))
    assert rules == {fam: f"{fam} rule" for fam in ALL_FAMILIES}


def test_two_different_rules_for_one_family_is_a_build_failure(tmp_path):
    rows = [view("p0", "ledger_replay", system="A"), view("p1", "ledger_replay", system="B")]
    rows += [view(f"q{i}", fam) for i, fam in enumerate(ALL_FAMILIES[1:])]
    with pytest.raises(ValueError, match="more than one system rule"):
        process_arm.rules_from_views(write_views(tmp_path, rows))


def test_a_missing_family_is_a_build_failure(tmp_path):
    rows = [view("p0", "ledger_replay")]
    with pytest.raises(ValueError, match="expected exactly"):
        process_arm.rules_from_views(write_views(tmp_path, rows))


def test_held_out_parents_are_exactly_the_scored_view_parents(tmp_path):
    rows = [view(f"p{i}", fam) for fam in ALL_FAMILIES for i in range(3)]
    held = process_arm.held_out_parents(write_views(tmp_path, rows))
    assert held == {f"p{i}" for i in range(3)}


def test_selection_is_deterministic_and_balanced():
    per_parent = {fam: {f"{fam}-{i}": {"id": f"{fam}-{i}-step0-valid", "parent_id": f"{fam}-{i}",
                                       "family": fam} for i in range(40)}
                  for fam in ALL_FAMILIES}
    first = process_arm.select(per_parent, per_family=10)
    second = process_arm.select(per_parent, per_family=10)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) == 30
    counts = {fam: sum(1 for r in first if r["family"] == fam) for fam in ALL_FAMILIES}
    assert counts == {fam: 10 for fam in ALL_FAMILIES}


def test_the_plan_is_computed_from_the_compiled_bytes():
    compiled = [compiled_row("a", ids=[1, 2, 3, 4]), compiled_row("b", ids=[5, 6, 7, 8, 9])]
    plan = process_arm.build_plan(compiled, exposures=2, probe_per_family=1)

    assert plan["experiment"] == "RRP-007-PROCESS-A1"
    assert plan["objective"] == "assistant_response_plus_eos"
    assert plan["row_order_per_exposure"] == ["a", "b"]
    assert plan["unique_worlds"] == 2
    assert plan["presentations"] == 4
    assert plan["processed_tokens"] == 2 * (4 + 5)
    # labels[1:] supervised counts: row a contributes 2, row b contributes 3
    assert plan["supervised_tokens"] == 2 * (2 + 3)
    assert plan["max_sequence"] == 2048 == Design().max_sequence
    assert plan["optimizer"] == frozen_optimizer(Design())
    assert plan["precision"] == "fp32" and plan["trainable"] == "blocks8-27"


def test_the_plan_passes_the_controllers_own_design_validator():
    from experiments.reasoning_posttraining_v1.controller import design_for
    compiled = [compiled_row("a", ids=[1, 2, 3, 4])]
    design_for(process_arm.build_plan(compiled, exposures=1, probe_per_family=1))


def test_write_prepared_records_the_held_out_count_and_is_create_only(tmp_path):
    compiled = [compiled_row("a")]
    report = {"train_window": 2048, "held_out_parents": 384}
    contract = process_arm.write_prepared(tmp_path / "prep", compiled, report)
    assert contract["experiment"] == "RRP-007-PROCESS-A1"
    assert contract["held_out_parents_excluded"] == 384
    assert contract["source_view"].endswith("positives only")
    assert contract["confirmation"] == "NOT_ESTABLISHED"
    with pytest.raises(ValueError, match="create-only"):
        process_arm.write_prepared(tmp_path / "prep", compiled, report)


def test_probe_subset_is_balanced_across_families_and_deterministic():
    compiled = [compiled_row(f"{fam}-{i}", family=fam)
                for fam in ALL_FAMILIES for i in range(20)]
    first = process_arm.probe_subset(compiled, per_family=4)
    assert first == process_arm.probe_subset(compiled, per_family=4)
    assert len(first) == 12
    by = {fam: sum(1 for i in first if i.startswith(fam)) for fam in ALL_FAMILIES}
    assert by == {fam: 4 for fam in ALL_FAMILIES}


def test_the_ceiling_actually_admits_a_train_launch():
    """Regression: the ceiling pinned adaptive_model_selection_rounds to 0 and refused.

    budget.start() charges exactly 1 of that counter on the first train launch of a
    journal -- it counts "a training round happened", not "the agent chose a model". A
    ceiling of 0 raises "insufficient approved budget" before anything runs, which is a
    failure mode no other test could see because it lives in the arithmetic between the
    ceiling and start()'s fixed charges.
    """
    ceiling = budget.REGISTERED[process_arm.EXPERIMENT]["ceiling"]
    timeout = 2000
    charges = {"model_worker_launches": 1, "new_disk_bytes": 4 * 2**20,
               "model_worker_wall_seconds_and_conservative_gpu_seconds": timeout + 20,
               "total_owned_work_wall_seconds": timeout + 140,
               "adaptive_model_selection_rounds": 1}
    too_small = {k: (v, ceiling[k]) for k, v in charges.items() if v > ceiling[k]}
    assert not too_small, f"ceiling refuses its own launch: {too_small}"


def test_the_ceiling_covers_both_checkpoints_the_engine_writes():
    """engine.py saves at group 2 and at the final group, reserving sum(numel)*12 + 1 MiB."""
    trainable_parameters = 314_618_880
    per_checkpoint = trainable_parameters * 12 + 2**20
    ceiling = budget.REGISTERED[process_arm.EXPERIMENT]["ceiling"]
    assert ceiling["new_disk_bytes"] >= 2 * per_checkpoint + 4 * 2**20


def test_compiled_rows_carry_every_field_the_native_evaluator_reads(tmp_path):
    """Regression: rows lacked `reference` and the launch died with KeyError at probe time.

    native_runtime.evaluate reads ids/labels/prompt_ids/reference/id/parent_id/family off
    each probe row. A missing key is not caught by any plan or contract check, so it costs a
    full model load and a budget launch to discover.
    """
    from experiments.model_quality_gpu_v1.native import load_tokenizer
    tokenizer = load_tokenizer(
        __import__("pathlib").Path(__file__).resolve().parents[3] / "Supa")
    rows = [{"id": "p0-step0-valid", "parent_id": "p0", "family": "ledger_replay",
             "before": {"A": 1}, "event": {"kind": "add", "account": "A", "amount": 1},
             "proposed_after": {"A": 2}}]
    rules = {"ledger_replay": "Apply one event to the supplied account balances."}
    compiled, rejected = process_arm.compile_training(rows, rules, tokenizer)

    assert not rejected and len(compiled) == 1
    row = compiled[0]
    for field in ("ids", "labels", "prompt_ids", "reference", "id", "parent_id", "family"):
        assert field in row, f"native_runtime.evaluate reads {field!r} and it is absent"
    assert row["reference"] == row["reference_text"] == '{"A":2}'
    assert row["source_split"] == "train" and row["mode"] == "direct"
    assert row["truncated"] is False
    assert row["labels"][0] == -100


def test_seed_17_keeps_seed_1s_exact_salt():
    """A replication that rebuilds a different corpus has silently stopped being one."""
    assert process_arm.salt_for(17) == process_arm.SALT
    assert process_arm.salt_for(23) == process_arm.SALT + "-s23"
    assert process_arm.salt_for(23) != process_arm.salt_for(29)
    with pytest.raises(ValueError):
        process_arm.salt_for(-1)


def test_a_different_selection_seed_draws_different_parents():
    per_parent = {fam: {f"{fam}-{i}": {"id": f"{fam}-{i}-step0-valid",
                                       "parent_id": f"{fam}-{i}", "family": fam}
                        for i in range(200)}
                  for fam in ALL_FAMILIES}
    a = {r["parent_id"] for r in process_arm.select(
        per_parent, per_family=64, salt=process_arm.salt_for(17))}
    b = {r["parent_id"] for r in process_arm.select(
        per_parent, per_family=64, salt=process_arm.salt_for(23))}
    assert len(a) == len(b) == 192
    assert a != b
    # Overlap is expected and is reported with the result; total identity is not.
    assert len(a & b) < len(a)
