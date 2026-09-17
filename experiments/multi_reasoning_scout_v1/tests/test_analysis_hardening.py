"""Synthetic evidence-integrity checks; these do not measure model efficacy."""
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json

import pytest

from experiments.multi_reasoning_scout_v1 import analysis, core, runner
from experiments.multi_reasoning_scout_v1.tests.test_scout import FixtureBackend, factorial_fixture


def replace_json(path, value):
    # Test tampering is deliberate; production evidence writers remain create-only.
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    records, run = tmp_path / "records", tmp_path / "run"
    (records / "bank").mkdir(parents=True)
    (run / "raw").mkdir(parents=True)
    source = {"repository": "fixture", "branch": "fixture", "sha": "source", "files": {"code.py": "hash"}}
    freeze = {"source": source, "source_files": source["files"]}
    core.write_new(records / "FREEZE.json", freeze)
    monkeypatch.setattr(analysis, "RECORDS", records)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "load_frozen", lambda: freeze)
    monkeypatch.setattr(analysis, "git", lambda *args: "source")
    model = {"id": "fixture"}
    binding = {"freeze_sha256": core.file_hash(records / "FREEZE.json"), "source": source,
               "model": model, "model_identity": core.digest(model), "runtime": {"fixture": True}}
    core.write_new(run / "binding.json", binding)
    public = [core.PublicItem("smoke", "world-smoke", "register_machine", "SMOKE", "Fixture problem"),
              core.PublicItem("check", "world-check", "register_machine", "CHECK", "Fixture problem")]
    core.dump_lines_new(records / "bank/public.jsonl", [asdict(p) for p in public])
    core.dump_lines_new(records / "bank/gold.jsonl", [{**asdict(p), "answer": "0", "answerable": True,
                                                     "difficulty": 3} for p in public])
    core.write_new(records / "BANK_MANIFEST.json", {"n": 2, "case_ids": [p.id for p in public],
                                                    "families": ["register_machine"]})
    for name in ("PROMPT_MANIFEST.json", "protocol.json"):
        core.write_new(records / name, {"fixture": True})
    for item in public:
        runner.run_item(item, FixtureBackend(), run / "raw", binding)
    for name, stage in (("qualification.json", "NATIVE_QUALIFICATION"), ("smoke.json", "SCHEMA_SMOKE")):
        core.write_new(run / name, {"status": "PASS", "stage": stage, "binding": binding,
                                   "causality": True, "repeat_ids_identical": True, "training_steps": 0, "items": 1,
                                   "qualification_forward_calls": 10,
                                   "qualification_generations": [FixtureBackend().generate([], 16) for _ in range(2)]})
    core.write_new(run / "generation.json", {"status": "PASS", "stage": "FROZEN_SCOUT_GENERATION", "binding": binding,
        "items": 2, "state_unchanged": True, "training_steps": 0, "resources": {},
        "raw_files": {p.name: core.file_hash(p) for p in (run / "raw").iterdir()}})
    launch = {"schema": 1, "out": str(run), "cwd": str(tmp_path), "supervisor_pid": 100,
              "timeout_seconds": 1800, "argv": ["python", "-m", "experiments.multi_reasoning_scout_v1.runner",
                                                "_worker", "--model", "fixture", "--out", str(run)]}
    core.write_new(run / "launch.json", launch)
    core.write_new(run / "spawn.json", {"child_pid": 101, "supervisor_pid": 100, "launch_sha256": core.digest(launch)})
    core.write_new(run / "worker_claim.json", {"schema": 1, "pid": 101, "parent_pid": 100,
        "supervisor_pid": 100, "launch_sha256": core.digest(launch), "launcher_redirected": False})
    core.write_new(run / "supervisor.json", {"status": "EXITED", "returncode": 0, "child_pid": 101})
    return run, records, tmp_path / "scored"


def rehash_test_evidence(run):
    generation = core.read_json(run / "generation.json")
    generation["raw_files"] = {p.name: core.file_hash(p) for p in (run / "raw").iterdir()}
    replace_json(run / "generation.json", generation)


def test_complete_synthetic_run_scores_difficulty_and_each_arm(complete_run):
    run, _, out = complete_run
    result = analysis.analyze_run(run, out)
    assert result["per_item"][0]["difficulty"] == 3
    assert result["per_difficulty"]["3"]["arms"]["A0"]["correct"] == 1
    assert result["per_family_difficulty"]["register_machine"]["3"]["arms"]["A3"]["correct"] == 1
    for arm in ("A0", "A1", "A2", "A3", "S1"):
        assert result["arms"][arm]["state_validity"]["rate"] == 1
        assert result["per_item"][0]["state_validity"][arm]["valid"]
    assert all(result["per_item"][0]["role_correct"].values())
    assert result["actual_unique_cost_all_partitions"]["generation_calls"] == 22
    assert result["actual_unique_cost_including_qualification"]["generation_calls"] == 24
    assert result["actual_unique_cost_including_qualification"]["forward_calls"] == 76
    assert result["qualification_cost"]["numerical_forward_calls"] == 4


@pytest.mark.parametrize("mutation", ["extra", "missing_manifest_entry", "subrecord", "prompt", "parser", "public", "majority"])
def test_scoring_rejects_raw_evidence_inconsistency(complete_run, mutation):
    run, _, out = complete_run
    raw = run / "raw"
    if mutation == "extra":
        core.write_new(raw / "extra.json", {})
        rehash_test_evidence(run)
    elif mutation == "missing_manifest_entry":
        generation = core.read_json(run / "generation.json")
        del generation["raw_files"]["check.R0.json"]
        replace_json(run / "generation.json", generation)
    else:
        item_path = raw / "check.item.json"
        row = core.read_json(item_path)
        if mutation == "subrecord":
            row["outputs"]["R0"]["parsed"]["answer"] = "1"
        elif mutation in ("prompt", "parser"):
            record = row["outputs"]["R0"]
            if mutation == "prompt":
                record["prompt"][1]["content"] += "Injected instruction"
                record["prompt_sha256"] = core.digest(record["prompt"])
            else:
                record["parsed"]["answer"] = "1"
            replace_json(raw / "check.R0.json", record)
        elif mutation == "public":
            row["item"]["problem"] = "Changed question"
        else:
            row["S0"] = "1"
        replace_json(item_path, row)
        rehash_test_evidence(run)
    with pytest.raises(ValueError):
        analysis.analyze_run(run, out)
    assert not out.exists()


@pytest.mark.parametrize("name,key,value", [
    ("qualification.json", "status", "FAIL"), ("qualification.json", "causality", False),
    ("smoke.json", "status", "FAIL"), ("smoke.json", "items", 8),
    ("supervisor.json", "returncode", 2), ("spawn.json", "launch_sha256", "wrong"),
    ("worker_claim.json", "pid", 999), ("generation.json", "state_unchanged", False),
])
def test_scoring_rejects_failed_or_unlinked_receipts(complete_run, name, key, value):
    run, _, out = complete_run
    record = core.read_json(run / name)
    record[key] = value
    replace_json(run / name, record)
    with pytest.raises(ValueError):
        analysis.analyze_run(run, out)


def test_scoring_rejects_stale_source_even_with_rebound_receipts(complete_run):
    run, _, out = complete_run
    for name in ("binding.json", "generation.json", "qualification.json", "smoke.json"):
        record = core.read_json(run / name)
        binding = record if name == "binding.json" else record["binding"]
        binding["source"]["files"]["code.py"] = "drift"
        replace_json(run / name, record)
    with pytest.raises(ValueError, match="source"):
        analysis.analyze_run(run, out)


def test_all_patterns_and_future_difficulty_breakdowns():
    old, new = factorial_fixture((False, False, False, False))
    old["per_item"], new["per_item"] = [], []
    for i, bits in enumerate(product((False, True), repeat=4)):
        a, b = factorial_fixture(bits)
        for target, example in ((old, a), (new, b)):
            row = example["per_item"][0]
            row.update(id=str(i), scenario_id=str(i), difficulty=3+i % 2)
            target["per_item"].append(row)
    result = analysis.factorial(old, new)
    assert set(result["overall"]["patterns_ABCD"].values()) == {1}
    assert result["per_difficulty"]["3"]["n"] == 8
    for row in result["per_item"]:
        A, B, C, D = [int(row[k]) for k in "ABCD"]
        assert row["training_makes_multi_newly_useful"] == (B-A <= 0 and D-C > 0)
        assert row["training_makes_multi_unnecessary"] == (not A and B and C and D)
        assert row["training_makes_multi_harmful"] == (B-A >= 0 and D-C < 0)


def native_pair(monkeypatch):
    old, new = factorial_fixture((False, True, True, True))
    model = {"id": "d1", "path": "fixture-old", "assets": {"model.safetensors": "old", "config.json": "config", "tokenizer.json": "tokenizer"},
             "source_repository": "repo", "source_branch": "old", "source_sha": "old",
             **{k: "same" for k in ("policy", "runtime_dtype", "storage_dtype", "attention_backend", "decoding")}}
    old.update(model=model, native_execution_pass=True, runtime={"test": 1})
    future = deepcopy(model)
    future.update(id="new", path="fixture-new", source_branch="new", source_sha="new")
    future["assets"]["model.safetensors"] = "new"
    new.update(model=future, native_execution_pass=True, runtime={"test": 1})
    monkeypatch.setattr(analysis, "load_frozen", lambda: {})
    monkeypatch.setattr(analysis, "file_hash", lambda path: "fixed")
    monkeypatch.setattr(analysis, "read_json", lambda path: {"selected": [deepcopy(model)], "primary_old_candidate_for_future": "d1"})
    return old, new


@pytest.mark.parametrize("mutation", ["old_comparator", "old_hash", "same_weights", "same_path", "same_source", "tokenizer", "config", "runtime", "difficulty", "both_stale_freezes"])
def test_native_factorial_rejects_unregistered_comparator_and_incompatible_future(monkeypatch, mutation):
    old, new = native_pair(monkeypatch)
    if mutation == "old_comparator":
        old["model"] = {**old["model"], "id": "supa"}
    elif mutation == "old_hash":
        old["model"] = deepcopy(old["model"])
        old["model"]["assets"]["model.safetensors"] = "altered"
    elif mutation == "same_weights":
        new["model"]["assets"]["model.safetensors"] = "old"
    elif mutation == "same_path":
        new["model"]["path"] = old["model"]["path"]
    elif mutation == "same_source":
        new["model"].update(source_branch="old", source_sha="old")
    elif mutation in ("tokenizer", "config"):
        new["model"]["assets"][mutation + ".json"] = "changed"
    elif mutation == "runtime":
        new["runtime"] = {"test": 2}
    elif mutation == "both_stale_freezes":
        old["freeze_sha256"] = new["freeze_sha256"] = "wrong"
    else:
        new["per_item"][0]["difficulty"] = 4
    with pytest.raises(ValueError):
        analysis.factorial(old, new)


def test_native_factorial_preserves_valid_registered_comparison(monkeypatch):
    old, new = native_pair(monkeypatch)
    result = analysis.factorial(old, new)
    assert result["overall"]["deltas"]["interaction"] == -1
    assert result["evidence_class"] == "OPEN_DEV_NATIVE_PAIRED"
    assert result["native_execution_pass"] is True
    assert result["confirmatory_claim"] is result["release_claim"] is False


def test_fixture_factorial_cannot_imply_native_evidence():
    old, new = factorial_fixture((False, True, True, True))
    result = analysis.factorial(old, new)
    assert result["evidence_class"] == "SOFTWARE_FIXTURE"
    assert result["native_execution_pass"] is result["open_dev_result"] is False
    assert result["confirmatory_claim"] is result["release_claim"] is False
    new["native_execution_pass"] = True
    with pytest.raises(ValueError, match="requires registered model"):
        analysis.factorial(old, new)
