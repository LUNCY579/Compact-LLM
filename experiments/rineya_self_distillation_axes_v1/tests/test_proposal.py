"""Offline packet and synthetic-result fixtures; no native capability evidence."""
import copy
import json

import pytest

from experiments.rineya_self_distillation_axes_v1.comparison import paired_screen, summarize_cell
from experiments.rineya_self_distillation_axes_v1.proposal import (
    SOURCE_CORE, SOURCE_SHA, TINY_ARMS, external_destination, prepare, validate_packet,
)
from experiments.rineya_self_distillation_axes_v1.data import load_rows


def source_receipts(tmp_path):
    custody = tmp_path / "custody.json"
    inspection = tmp_path / "inspection.json"
    custody.write_text(json.dumps({"sha256": SOURCE_SHA, "checkpoint_path": "/fixture-not-loaded"}))
    inspection.write_text(json.dumps({"checkpoint_manifest_sha256": SOURCE_SHA,
        "source_identity": {"core_id": SOURCE_CORE}, "eligibility": {"fixture": True}}))
    return custody, inspection


def test_packet_is_create_only_bounded_and_not_permission(tmp_path):
    custody, inspection = source_receipts(tmp_path)
    packet = prepare(tmp_path / "packet", custody, inspection, tmp_path / "source", tmp_path)
    assert validate_packet(packet)
    assert packet["design"]["cells"] == 6 and packet["limits"]["total_updates"] == 72
    assert not packet["execution_authorized"]
    assert len(load_rows(tmp_path / "packet" / "screen.jsonl", "modular_state")) == 56
    assert len(packet["data"]["cohorts"]["teacher_gate"]) == 8
    assert all(len(ids) == 12 and len(set(ids)) == 12 for ids in packet["design"]["train_row_streams"].values())
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "packet", custody, inspection, tmp_path / "source", tmp_path)
    for field, value in (("execution_authorized", True), ("status", "APPROVED")):
        changed = copy.deepcopy(packet)
        changed[field] = value
        with pytest.raises(ValueError): validate_packet(changed)
    changed = copy.deepcopy(packet)
    changed["design"]["arms"][0]["ema_rate"] = .02
    with pytest.raises(ValueError): validate_packet(changed)
    changed = copy.deepcopy(packet)
    changed["limits"]["max_response_tokens"] = 64
    with pytest.raises(ValueError): validate_packet(changed)
    changed = copy.deepcopy(packet)
    changed["teacher_gate"]["min_correct"] = 6
    with pytest.raises(ValueError): validate_packet(changed)


def test_external_paths_reject_source_codex_and_outside(tmp_path):
    workspace = tmp_path / "source"
    for path in (workspace / "outputs", tmp_path / ".codex" / "outputs", tmp_path.parent / "outside"):
        with pytest.raises(ValueError): external_destination(path, workspace, tmp_path)
    other = tmp_path / "other-checkout"
    other.mkdir()
    (other / ".git").write_text("gitdir: fixture-only")
    with pytest.raises(ValueError, match="another Git"):
        external_destination(other / "outputs", workspace, tmp_path)
    assert external_destination(tmp_path / "external", workspace, tmp_path) == tmp_path / "external"


def result_fixture(acquisition_gain=True):
    cohorts = {"iid_acquisition": ["iid0", "iid1", "iid2"],
               "harder_transfer": ["hard0", "hard1"],
               "retention_arithmetic": [f"arith{i}" for i in range(8)],
               "retention_spatial": [f"spatial{i}" for i in range(8)]}
    baseline = []
    for cohort, ids in cohorts.items():
        for identifier in ids:
            retention = cohort.startswith("retention")
            baseline.append({"id": identifier, "parent_id": identifier, "task": cohort if retention else "modular_state",
                "purpose": "retention" if retention else "acquisition", "correct": retention})
    final = copy.deepcopy(baseline)
    for row in final:
        if acquisition_gain and row["id"] in ("iid0", "iid1"):
            row["correct"] = True
        if row["id"] == "arith0": row["correct"] = False
        if row["id"].startswith("hard"): row["correct"] = False
    result = {"arm_id": "fixture", "baseline": baseline,
        "checkpoints": [{"step": s, "rows": copy.deepcopy(final)} for s in (10, 11, 12)],
        "steps": [{"step": s, "update_l2": .001, "preclip_gradient_norm": .01} for s in range(1, 13)],
        "all_forward_counts_including_eval": {"processed_input_tokens": 10, "sampled_tokens": 10, "forward_calls": 10}}
    return result, cohorts


def test_harder_transfer_and_source_damage_never_hide_in_iid():
    result, cohorts = result_fixture()
    summary = summarize_cell(result, cohorts)
    assert summary["iid_acquisition_pp"] == pytest.approx(200 / 3)
    assert summary["harder_transfer_pp"] == 0
    arithmetic = next(t for t in summary["retention_tasks"] if t["task"] == "retention_arithmetic")
    assert arithmetic["source_correct_retained"] == 7 and arithmetic["damages"] == 1
    assert summary["measured_tradeoff_condition_met"] and not summary["promotion_allowed"]
    result["checkpoints"][0]["rows"].pop()
    with pytest.raises(ValueError): summarize_cell(result, cohorts)


def test_no_acquisition_or_zero_source_denominator_is_not_retention_success():
    result, cohorts = result_fixture(False)
    summary = summarize_cell(result, cohorts)
    assert summary["retention_margin_met"] and not summary["measured_tradeoff_condition_met"]
    for row in result["baseline"]:
        if row["purpose"] == "retention": row["correct"] = False
    summary = summarize_cell(result, cohorts)
    assert not summary["adequate_source_success_denominators"]
    assert all(t["retained_fraction_of_source_correct"] is None for t in summary["retention_tasks"])
    assert not summary["measured_tradeoff_condition_met"]


def test_retention_task_relabeling_is_rejected():
    result, cohorts = result_fixture()
    for records in [result["baseline"]] + [c["rows"] for c in result["checkpoints"]]:
        for row in records:
            if row["purpose"] == "retention": row["task"] = "retention_arithmetic"
    with pytest.raises(ValueError, match="cannot be pooled"):
        summarize_cell(result, cohorts)


def test_paired_analysis_binds_seed_config_and_per_row_identity(tmp_path):
    custody, inspection = source_receipts(tmp_path)
    packet = prepare(tmp_path / "packet", custody, inspection, tmp_path / "source", tmp_path)
    rows = load_rows(tmp_path / "packet" / "screen.jsonl", "modular_state")
    baseline = [{"id": r["id"], "parent_id": r["parent_id"], "task": r["task"],
                 "purpose": r["purpose"], "correct": r["purpose"] == "retention"}
                for r in rows if r["split"] == "dev"]
    results = {}
    for seed in (17, 29):
        for arm in TINY_ARMS:
            results[(seed, arm.name)] = {"arm_id": arm.name, "arm": vars(arm),
                "initial_parameters_sha256": "a" * 64,
                "identity": {"config_sha256": packet["design"]["config_sha256_by_seed"][str(seed)],
                    "code_sha256": packet["code_sha256"], "runtime": "synthetic result fixture only",
                    "checkpoint_sha256": SOURCE_SHA,
                    "dataset_sha256": packet["data"]["bindings"]["screen.jsonl"]["sha256"]},
                "baseline": copy.deepcopy(baseline),
                "checkpoints": [{"step": s, "rows": copy.deepcopy(baseline)} for s in (10, 11, 12)],
                "steps": [{"step": s, "row_id": identifier, "update_l2": .001, "preclip_gradient_norm": .01}
                          for s, identifier in enumerate(packet["design"]["train_row_streams"][str(seed)], 1)],
                "all_forward_counts_including_eval": {"processed_input_tokens": 10, "sampled_tokens": 10, "forward_calls": 10}}
    report = paired_screen(results, packet)
    assert report["independent_training_draws"] == 1 and not report["promotion_allowed"]
    changed = copy.deepcopy(results)
    key = (17, TINY_ARMS[0].name)
    changed[key]["identity"]["config_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="exact reviewed"):
        paired_screen(changed, packet)
    changed = copy.deepcopy(results)
    changed[key]["baseline"][0]["parent_id"] = "relabeled-family"
    with pytest.raises(ValueError, match="per-row"):
        paired_screen(changed, packet)
