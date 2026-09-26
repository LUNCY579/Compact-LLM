"""Standalone auditor arithmetic and recovery-evidence tests; no model/GPU imports."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("phase1_independent_auditor", ROOT / "scripts/audit_phase1_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def training_fixture():
    trajectory = []
    applied = skipped = safety = tokens = 0
    for attempt in range(1, 7):
        apply, skip, safe = attempt not in (2, 4), attempt == 2, attempt == 4
        applied += apply
        skipped += skip
        safety += safe
        tokens += 16 if apply else 0
        trajectory.append({
            "attempt": attempt, "step": attempt, "optimizer_update_applied": apply,
            "step_skipped_by_scaler": skip, "step_skipped_by_safety_check": safe,
            "finite_nonzero_gradient": apply, "gradients_finite": not skip,
            "grad_norm": 1.0 if apply else None, "answer_tokens": 16,
            "examples_consumed": attempt * 8, "optimizer_updates_applied": applied,
            "steps_skipped_by_scaler_total": skipped, "steps_skipped_by_safety_check_total": safety,
            "examples_in_applied_updates": applied * 8, "answer_tokens_in_applied_updates": tokens,
        })
    config = {"max_steps": 6, "grad_accum": 8, "lr": 1e-4, "seed": 17}
    training = {**config, "optimizer_attempts": 6, "steps_applied": applied,
                "steps_skipped_by_scaler": skipped, "steps_skipped_by_safety_check": safety,
                "examples_consumed": 48, "examples_in_applied_updates": 32,
                "answer_tokens_in_applied_updates": 64, "finite_nonzero_gradient_attempts": 4,
                "gradient_checkpointing": True, "gate_disabled_for_training": True,
                "gate_on_for_validation_evaluations": True, "gate_restored_after_training": True,
                "trajectory": trajectory, "validation": [{"step": i} for i in range(1, 7)]}
    resource = {"optimizer_steps_attempted": 6, "optimizer_steps_applied": applied,
                "steps_skipped_by_scaler": skipped, "steps_skipped_by_safety_check": safety,
                "examples_consumed": 48}
    return training, resource, config


def test_logical_training_counts_applied_scaler_and_safety_skips_once():
    training, resource, config = training_fixture()
    checks, counts = audit.trajectory_checks(training, resource, config)
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    assert counts["attempted_from_trajectory"] == 6
    assert counts["applied_from_trajectory"] == 4


@pytest.mark.parametrize("corruption", ["duplicated_prefix", "missing_attempt", "remaining_budget", "double_cost",
                                         "cursor", "cumulative", "missing_validation", "token_total", "nonfinite_update"])
def test_bad_logical_training_cannot_pass(corruption):
    training, resource, config = training_fixture()
    if corruption == "duplicated_prefix":
        training["trajectory"] = training["trajectory"][:2] + training["trajectory"]
    elif corruption == "missing_attempt":
        training["trajectory"].pop(2)
    elif corruption == "remaining_budget":
        training["max_steps"] = 4
    elif corruption == "double_cost":
        resource["optimizer_steps_attempted"] = 8
    elif corruption == "cursor":
        training["trajectory"][2]["examples_consumed"] = 8
    elif corruption == "cumulative":
        training["trajectory"][2]["optimizer_updates_applied"] = 1
    elif corruption == "missing_validation":
        training["validation"].pop()
    elif corruption == "token_total":
        training["answer_tokens_in_applied_updates"] += 1
    else:
        training["trajectory"][0]["grad_norm"] = float("inf")
    checks, _ = audit.trajectory_checks(training, resource, config)
    assert not all(checks.values())


def test_resource_unknowns_remain_lower_bounds():
    exact = {key: 2 for key in (*audit.SUM_RESOURCES, *audit.PEAK_RESOURCES)}
    exact.update(model_loads=1, model_loads_upper_bound=1, resource_accounting_complete=True)
    interrupted = {**exact, "wall_seconds": 3, "peak_gpu_allocated_bytes": None,
                   "model_loads": 0, "model_loads_upper_bound": 1,
                   "resource_accounting_complete": False, "unknown_segment_ids": ["s2"],
                   "incomplete_metric_fields": ["wall_seconds", "peak_gpu_allocated_bytes"]}
    result = audit.aggregate_resources([exact, interrupted])
    assert result["wall_seconds"] == 5
    assert result["model_loads"] == 1 and result["model_loads_upper_bound"] == 2
    assert result["resource_accounting_complete"] is False
    assert result["unknown_segment_ids"] == ["s2"]
    assert "lower bounds" in result["cost_interpretation"]


def test_no_resource_receipts_is_not_exact_zero():
    result = audit.aggregate_resources([None, None])
    assert result["resource_accounting_complete"] is False


def test_raw_preservation_last_number_and_short_text_scoring():
    assert audit.preservation_score("73 + 11 - 3 + 4 = 85", "85.")["matched"]
    assert not audit.preservation_score("85 - 3 = 82", "85.")["matched"]
    assert audit.preservation_score("The result is ALLOW.", "allow")["matched"]


def test_partial_audit_never_emits_winner_or_license(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LAB", tmp_path)
    monkeypatch.setattr(audit, "read_jsonl", lambda path: [])
    monkeypatch.setattr(audit, "expected_preservation", lambda smoke: {})
    result = audit.audit_stage("phase1", str.lower)
    assert result["decision"]["status"] == "INCOMPLETE"
    assert result["decision"]["winner"] is None
    assert result["decision"]["phase_2_licensed"] is False


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


def recovery_fixture(out, *, interrupted):
    training, logical_dose, config = training_fixture()
    config["arm"] = "B"
    ids = ["000001-first", "000002-final"]
    encoded = {"train": [{"n_answer_tokens": 2, "input_ids": [1, 2], "labels": [1, 2]}],
               "dev": [{"n_answer_tokens": 2, "input_ids": [1, 2], "labels": [1, 2]}]}
    data = {"train": audit.fingerprint(encoded["train"]), "dev": audit.fingerprint(encoded["dev"]),
            "train_count": 1, "dev_count": 1}
    write_json(out / "encoded_data.json", encoded)
    initializer = {"algorithm": "sha256-native-trainable-parameters-v1", "sha256": "3" * 64}
    recipe = {**config, "source_sha256": {"source.py": "4" * 64},
              "optimizer": "AdamW", "weight_decay": 0.0, "max_grad_norm": 1.0,
              "gradient_checkpointing": True, "adapter_dtype": "float32", "initialization": "lora"}
    identity = {"schema_version": 3, "run_id": "fixture", "config": config,
                "source_sha256": recipe["source_sha256"], "canonical_core": "canonical",
                "checkpoint_sha256": {"model.safetensors": "5" * 64},
                "scientific_recipe": recipe, "dataset_fingerprint": data,
                "adapter_initialization": initializer, "frozen_parameter_sha256": "6" * 64}
    fingerprints = {"recipe_sha256": audit.fingerprint({"scientific_recipe": recipe, "dataset_fingerprint": data}),
                    "train_data_sha256": data["train"], "eval_data_sha256": data["dev"],
                    "base_checkpoint_sha256": audit.fingerprint(identity["checkpoint_sha256"]),
                    "adapter_initialization_sha256": initializer["sha256"]}
    identity["checkpoint_fingerprints"] = fingerprints
    write_json(out / "identity.json", identity)
    info = {"training": training, "dataset_fingerprint": data, "frozen_base": {"before": "6" * 64},
            "run_id": "fixture", "segment_id": ids[-1], "checkpoint_fingerprints": fingerprints}
    segment_resources = []
    for index, sid in enumerate(ids):
        directory = out / "segments" / sid
        saved = {"run_id": "fixture", "segment_id": sid, "config": config,
                 "source_sha256": identity["source_sha256"], "canonical_core": "canonical",
                 "checkpoint_sha256_before": identity["checkpoint_sha256"],
                 "resumed_from_attempt": 0 if index == 0 else 2,
                 "resumed_from_checkpoint_sha256": None if index == 0 else "7" * 64,
                 "status": "STARTED"}
        write_json(directory / "segment_identity.json", saved)
        write_json(directory / "started_manifest.json", saved)
        write_json(directory / "training_entry.json", {"run_id": "fixture", "total_max_steps": 6,
                   "checkpoint_fingerprints": fingerprints, "resume_from_checkpoint": index == 1})
        write_json(directory / "adapter_initialization.json", initializer)
        write_json(directory / "frozen_parameters_before.json", {"algorithm": "sha256-native-frozen-parameters-v1", "sha256": identity["frozen_parameter_sha256"]})
        write_json(directory / "model_load.json", {"status": "LOADED", "model_loads": 1,
                   "model_loads_upper_bound": 1, "model_load_attempts": 1, "model_load_seconds": 1})
        resource = {key: 0 for key in (*audit.SUM_RESOURCES, *audit.PEAK_RESOURCES)}
        resource.update(wall_seconds=3 + index, model_loads=1, model_loads_upper_bound=1,
                        model_load_attempts=1, model_load_seconds=1,
                        peak_rss_bytes=10 + index, resource_accounting_complete=True)
        selected = training["trajectory"][:2] if index == 0 else training["trajectory"][2:]
        resource.update(optimizer_steps_attempted=len(selected),
                        optimizer_steps_applied=sum(r["optimizer_update_applied"] for r in selected),
                        steps_skipped_by_scaler=sum(r["step_skipped_by_scaler"] for r in selected),
                        steps_skipped_by_safety_check=sum(r["step_skipped_by_safety_check"] for r in selected),
                        examples_consumed=len(selected) * 8)
        segment_resources.append(resource)
        if index != 0 or not interrupted:
            write_json(directory / "run_manifest.json", {**saved, "status": "FAILED" if index == 0 else "COMPLETE",
                       "end_utc": "2026-09-05T01:00:00+00:00", "end_committed_attempt": 2 if index == 0 else 6,
                       "canonical_checkpoint_unchanged": True, "checkpoint_sha256_after": identity["checkpoint_sha256"],
                       "resource": resource})
            write_json(directory / "resource.json", resource)
        selected = training["trajectory"][:2] if index == 0 else training["trajectory"][2:]
        write_jsonl(directory / "training_progress.jsonl", [{"event": "optimizer_attempt", **row} for row in selected])
    checkpoint_path = out / "checkpoints/latest.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"opaque test checkpoint: auditor hashes but never deserializes")
    receipt = {"sha256": audit.file_digest(checkpoint_path), "completed_attempts": 6, "pending_validation": None,
               "trajectory_sha256": audit.fingerprint(training["trajectory"]),
               "evals_sha256": audit.fingerprint(training["validation"])}
    write_json(out / "committed_checkpoint.json", {"receipt": receipt,
               "progress": {"completed_attempts": 6, "cursor": 48, "pending_validation": None,
                            "trajectory": training["trajectory"], "evals": training["validation"]}})
    events = []
    for row, validation in zip(training["trajectory"], training["validation"]):
        events += [{"event": "optimizer_attempt", **row}, {"event": "validation", **validation}]
    write_jsonl(out / "training_progress.jsonl", events)
    if interrupted:
        segment_resources[0] = {**segment_resources[0], "resource_accounting_complete": False,
                                "unknown_segment_ids": [ids[0]],
                                **{key: None for key in ("wall_seconds", *audit.PEAK_RESOURCES)}}
    resource = {**audit.aggregate_resources(segment_resources), **logical_dose,
                "numeric_costs_are_lower_bounds": interrupted}
    manifest = {"schema_version": 3, "status": "COMPLETE", "run_id": "fixture", "config": config,
                "source_sha256": identity["source_sha256"], "canonical_core": "canonical",
                "checkpoint_sha256_before": identity["checkpoint_sha256"], "checkpoint_fingerprints": fingerprints,
                "checkpoint_receipt": receipt, "segment_ids": ids, "resume_count": 1,
                "committed_training": logical_dose}
    for name in ("arm_B_info.json", "arm_B_target.jsonl", "arm_B_preservation.jsonl", "per_example.jsonl", "dataset.json"):
        content = info if name == "arm_B_info.json" else {}
        write_json(out / name, content)
        write_json(out / "segments" / ids[-1] / name, content)
    return manifest, resource, info


@pytest.mark.parametrize("interrupted", [False, True])
def test_resumed_arm_accepts_two_physical_loads_and_honest_interrupted_cost(tmp_path, interrupted):
    manifest, resource, info = recovery_fixture(tmp_path, interrupted=interrupted)
    checks, computed = audit.segment_checks(tmp_path, "B", manifest, resource, info)
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    assert computed["physical_cost_aggregation"]["model_loads"] == 2
    assert computed["physical_cost_aggregation"]["resource_accounting_complete"] is (not interrupted)


@pytest.mark.parametrize("corruption", ["missing_segment", "missing_identity", "hidden_load", "hidden_unknown",
                                         "missing_checkpoint", "pending_validation", "duplicate_logical_prefix",
                                         "altered_source", "missing_cost_fields", "attempt_gap", "wrong_initializer",
                                         "missing_segment_initializer", "encoded_data_mutation", "hidden_load_seconds",
                                         "physical_prefix_double_count"])
def test_recovery_receipt_corruption_is_rejected(tmp_path, corruption):
    manifest, resource, info = recovery_fixture(tmp_path, interrupted=True)
    first = tmp_path / "segments" / manifest["segment_ids"][0]
    if corruption == "missing_segment":
        manifest["segment_ids"].pop(0)
    elif corruption == "missing_identity":
        (first / "segment_identity.json").unlink()
    elif corruption == "hidden_load":
        resource["model_loads"] = 1
    elif corruption == "hidden_unknown":
        resource["resource_accounting_complete"] = True
    elif corruption == "missing_checkpoint":
        (tmp_path / "checkpoints/latest.pt").unlink()
    elif corruption == "pending_validation":
        manifest["checkpoint_receipt"]["pending_validation"] = 6
    elif corruption == "duplicate_logical_prefix":
        path = tmp_path / "training_progress.jsonl"
        events = audit.read_jsonl(path)
        write_jsonl(path, events[:2] + events)
    elif corruption == "altered_source":
        path = first / "started_manifest.json"
        saved = audit.json_file(path)
        saved["source_sha256"] = {"changed": "old"}
        write_json(path, saved)
    elif corruption == "missing_cost_fields":
        resource["incomplete_metric_fields"] = []
    elif corruption == "attempt_gap":
        path = first / "started_manifest.json"
        saved = audit.json_file(path)
        saved["resumed_from_attempt"] = 1
        write_json(path, saved)
    elif corruption == "wrong_initializer":
        path = tmp_path / "identity.json"
        saved = audit.json_file(path)
        saved["adapter_initialization"]["sha256"] = "9" * 64
        write_json(path, saved)
    elif corruption == "missing_segment_initializer":
        (first / "adapter_initialization.json").unlink()
    elif corruption == "encoded_data_mutation":
        path = tmp_path / "encoded_data.json"
        saved = audit.json_file(path)
        saved["train"][0]["n_answer_tokens"] = 3
        write_json(path, saved)
    elif corruption == "hidden_load_seconds":
        resource["model_load_seconds"] = 0
    else:
        final = tmp_path / "segments" / manifest["segment_ids"][-1]
        saved = audit.json_file(final / "run_manifest.json")
        saved["resource"]["optimizer_steps_attempted"] += 2
        write_json(final / "run_manifest.json", saved)
        write_json(final / "resource.json", saved["resource"])
    checks, _ = audit.segment_checks(tmp_path, "B", manifest, resource, info)
    assert not all(checks.values())


@pytest.mark.parametrize("smoke", [False, True])
def test_exact_one_point_preservation_boundary_and_smoke_never_licenses(tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(audit, "LAB", tmp_path)
    monkeypatch.setattr(audit, "read_jsonl", lambda path: [{"id": str(i)} for i in range(100)])
    monkeypatch.setattr(audit, "expected_preservation", lambda smoke: {})
    monkeypatch.setattr(audit, "bootstrap", lambda *args: {"test_stub": True})
    monkeypatch.setitem(audit.PUBLISHED, "a11", {"showcase": 60, "stability_test": 0, "all": 60})
    keys = ("max_steps", "grad_accum", "lr", "max_length", "seed", "rank", "alpha", "dropout", "trainable_layers", "dtype", "data_sha256", "external_scorer_sha256")
    config = {key: "fixed" for key in keys}
    def fake_arm(stage, arm, *args):
        nt = {"A": 60, "B": 62, "C": 61}[arm]
        np = {"A": 42, "B": 41, "C": 41}[arm]
        target = {str(i): {"correct": i < nt, "split": "showcase"} for i in range(100)}
        preserved = {str(i): {"matched": i < np} for i in range(100)}
        return {"status": "VALID", "dataset_fingerprint": {"fixed": True}, "config": config,
                "recorded_resource": {key: 0 for key in (*audit.SUM_RESOURCES, *audit.PEAK_RESOURCES)}}, {"target": target, "preservation": preserved}
    monkeypatch.setattr(audit, "audit_arm", fake_arm)
    result = audit.audit_stage("phase1_smoke" if smoke else "phase1", str.lower)
    decision = result["decision"]
    assert decision["arms"]["B"]["preservation_regression_fraction"] == "1"
    assert decision["arms"]["B"]["target_gain_fraction"] == "2"
    assert decision["arms"]["B"]["meaningfully_positive"] is True
    assert decision["arms"]["C"]["meaningfully_positive"] is False
    assert decision["winner"] == "LoRA"
    assert decision["phase_2_licensed"] is (not smoke)


def raw_arm_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LAB", tmp_path)
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setitem(audit.CORE_PATHS, "a11", tmp_path / "canonical")
    hashes = {"fixed": "1" * 64}
    monkeypatch.setattr(audit, "current_source_hashes", lambda: hashes)
    monkeypatch.setattr(audit, "current_checkpoint_hashes", lambda core: hashes)
    monkeypatch.setattr(audit, "current_data_hashes", lambda: hashes)
    monkeypatch.setattr(audit, "file_digest", lambda path: "2" * 64)
    target = {"t1": {"split": "showcase", "variant": "base", "task_type": "test", "answerable": True,
                      "context": "context", "question": "question?", "accepted_answers": ["yes"]}}
    preserved = {"p1": {"example_id": "p1", "task_family": "counting_sets", "reference": "1.",
                         "messages": [], "reference_numbers": ["1"]}}
    target_item = {"example_id": "t1", "condition": "A_full", **target["t1"]}
    config = {"arm": "A", "core": "a11", "smoke": True, "max_steps": 6, "rank": 4, "alpha": 8,
              "dropout": 0.0, "trainable_layers": [17, 18, 19], "protected_layers": [0, 1, 2],
              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "seed": 17,
              "max_length": 3840, "grad_accum": 8, "lr": 1e-4, "train_packs": 8, "dev_packs": 4,
              "step_budget_unit": "optimizer attempts", "target_context": "FULL",
              "preservation_classes": list(audit.CLASSES), "preservation_per_class": 8,
              "eval_max_new_tokens": 24, "dtype": "float16", "data_sha256": hashes,
              "external_scorer_sha256": "2" * 64}
    resource = {key: 0 for key in (*audit.SUM_RESOURCES, *audit.PEAK_RESOURCES)}
    resource.update(model_loads=1, wall_seconds=1)
    info = {"target": audit.summarize([True]), "preservation": {"matched": audit.summarize([True]), "exact": audit.summarize([True])},
            "loaded_core_path": str(tmp_path / "canonical"), "loaded_dtype": "float16",
            "evaluation_gate_on": True, "frozen_base": {"before": "frozen", "after": "frozen", "unchanged": True},
            "training": None, "adapter": {"trainable_params": 0},
            "dataset_fingerprint": {"train": "3" * 64, "dev": "4" * 64, "train_count": 32, "dev_count": 16,
                                    "target": audit.fingerprint([target_item]), "target_count": 1,
                                    "preservation": audit.fingerprint(list(preserved.values())), "preservation_count": 1}}
    manifest = {"status": "COMPLETE", "config": config, "canonical_checkpoint_unchanged": True,
                "checkpoint_sha256_before": hashes, "checkpoint_sha256_after": hashes,
                "canonical_core": str(tmp_path / "canonical"), "source_sha256": hashes, "resource": resource}
    target_record = {**target_item, "raw_prediction": "yes", "normalized_prediction": "yes", "correct": True, "malformed": False}
    preservation_record = {**preserved["p1"], "raw_prediction": "1.", **audit.preservation_score("1.", "1.")}
    out = tmp_path / "results/phase1_smoke/arm_A"
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "resource.json", resource)
    write_json(out / "arm_A_info.json", info)
    write_json(out / "dataset.json", {"fingerprint": info["dataset_fingerprint"],
               "train": {"n_input": 32, "n_kept": 32, "n_dropped_too_long": 0},
               "dev": {"n_input": 16, "n_kept": 16, "n_dropped_too_long": 0}})
    write_jsonl(out / "arm_A_target.jsonl", [target_record])
    write_jsonl(out / "arm_A_preservation.jsonl", [preservation_record])
    rows = [{"evaluation": "target", "arm": "A", **target_record},
            {"evaluation": "preservation", "arm": "A", **preservation_record}]
    write_jsonl(out / "per_example.jsonl", rows)
    return target, preserved, out, rows


def test_raw_arm_rescoring_with_valid_metadata(tmp_path, monkeypatch):
    target, preserved, out, rows = raw_arm_fixture(tmp_path, monkeypatch)
    result, records = audit.audit_arm("phase1_smoke", "A", "a11", True, target, preserved, str.lower)
    assert result["status"] == "VALID", result
    assert records["target"]["t1"]["correct"] is True


@pytest.mark.parametrize("corruption", ["cached_scores", "duplicate_id", "wrong_reference", "source_hash"])
def test_raw_arm_corruption_is_recomputed_or_rejected(tmp_path, monkeypatch, corruption):
    target, preserved, out, rows = raw_arm_fixture(tmp_path, monkeypatch)
    if corruption == "cached_scores":
        rows[0]["correct"] = False
        rows[1]["matched"] = False
    elif corruption == "duplicate_id":
        rows.append(rows[0])
    elif corruption == "wrong_reference":
        rows[1]["reference"] = "2."
    else:
        manifest = audit.json_file(out / "run_manifest.json")
        manifest["source_sha256"] = {"old": "receipt"}
        write_json(out / "run_manifest.json", manifest)
    write_jsonl(out / "per_example.jsonl", rows)
    result, records = audit.audit_arm("phase1_smoke", "A", "a11", True, target, preserved, str.lower)
    assert result["status"] == "INVALID"
    assert records["target"]["t1"]["correct"] is True
    assert records["preservation"]["p1"]["matched"] is True


@pytest.mark.parametrize("terminal,newline", [(False, False), (False, True), (True, False)])
def test_only_interrupted_unterminated_last_physical_line_can_be_ignored(tmp_path, terminal, newline):
    path = tmp_path / "physical.jsonl"
    path.write_bytes(b'{"attempt":1}\n{"attempt":' + (b'\n' if newline else b''))
    if not terminal and not newline:
        assert audit.physical_jsonl(path, interrupted=True) == ([{"attempt": 1}], True)
    else:
        with pytest.raises(json.JSONDecodeError):
            audit.physical_jsonl(path, interrupted=not terminal)


def test_incomplete_arm_scans_physical_cost_without_stale_root_receipt(tmp_path):
    manifest, resource, info = recovery_fixture(tmp_path, interrupted=True)
    computed = audit.incomplete_resources(tmp_path)
    assert computed["model_loads"] == 2
    assert computed["resource_accounting_complete"] is False
    assert computed["optimizer_steps_attempted"] == 6
    assert computed["logical_dose_receipt_current"] is True
    saved = audit.json_file(tmp_path / "committed_checkpoint.json")
    saved["receipt"]["sha256"] = "0" * 64
    write_json(tmp_path / "committed_checkpoint.json", saved)
    unknown = audit.incomplete_resources(tmp_path)
    assert unknown["optimizer_steps_attempted"] is None
    assert unknown["logical_dose_receipt_current"] is False


@pytest.mark.parametrize("action", ["fail_mid_accumulation", "fail_eval"])
def test_independent_segment_auditor_accepts_real_cpu_runner_recovery(tmp_path, action):
    # The imported test fixture builds only a tiny CPU adapter; no canonical
    # checkpoint, tokenizer or experiment evaluation is ever loaded.
    from test_phase1_lifecycle import execute
    with pytest.raises(RuntimeError, match="injected"):
        execute(tmp_path, action=action)
    execute(tmp_path, resume=True)
    arm = tmp_path / "arm_B"
    manifest = audit.json_file(arm / "run_manifest.json")
    resource = audit.json_file(arm / "resource.json")
    info = audit.json_file(arm / "arm_B_info.json")
    checks, _ = audit.segment_checks(arm, "B", manifest, resource, info)
    assert all(checks.values()), [key for key, value in checks.items() if not value]
