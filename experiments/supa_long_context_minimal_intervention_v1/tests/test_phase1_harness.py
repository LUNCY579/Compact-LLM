"""Failure-path checks for isolated arms; no checkpoint load, training or GPU."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("phase1_harness", ROOT / "scripts/phase1_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from slcmi import evaluation, provenance


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(provenance, "checkpoint_files", lambda path: {"model.safetensors": "fixed"})
    monkeypatch.setattr(runner, "checkpoint_files", lambda path: {"model.safetensors": "fixed"})
    monkeypatch.setattr(runner, "source_hashes", lambda: {"source.py": "current"})
    monkeypatch.setattr(provenance, "source_hashes", lambda: {"source.py": "current"})
    monkeypatch.setattr(provenance, "git", lambda *args: "test")
    out = tmp_path / "phase1_smoke"
    core = {"path": tmp_path / "canonical", "dtype": "float16"}
    config = {"core": "a11", "smoke": True, "data_sha256": {"curriculum": "same"}}
    targets = [{"example_id": "t1", "split": "showcase", "variant": "base", "task_type": "test",
                "answerable": True, "condition": "A_full", "accepted_answers": ["yes"]}]
    preserved = [{"example_id": "p1", "task_family": "counting_sets", "reference": "1.",
                  "reference_numbers": ["1"]}]

    def add(arm, status="COMPLETE"):
        arm_out = out / f"arm_{arm}"
        arm_out.mkdir(parents=True)
        manifest = {
            "status": status, "config": {**config, "arm": arm},
            "source_sha256": {"source.py": "current"}, "canonical_core": str(core["path"]),
            "canonical_checkpoint_unchanged": True,
            "checkpoint_sha256_before": {"model.safetensors": "fixed"},
            "checkpoint_sha256_after": {"model.safetensors": "fixed"},
            "resource": {"wall_seconds": 2, "model_loads": 1, "peak_rss_bytes": 10,
                         "peak_gpu_allocated_bytes": 0, "peak_gpu_reserved_bytes": 0,
                         "optimizer_steps_attempted": 6 if arm != "A" else 0,
                         "optimizer_steps_applied": 5 if arm != "A" else 0,
                         "steps_skipped_by_scaler": 1 if arm != "A" else 0,
                         "trainable_params": 12 if arm != "A" else 0},
        }
        provenance.write_json(arm_out / "run_manifest.json", manifest)
        provenance.write_json(arm_out / f"arm_{arm}_info.json", {
            "loaded_core_path": str(core["path"]), "loaded_dtype": core["dtype"],
            "evaluation_gate_on": True, "frozen_base": {"before": "same", "after": "same", "unchanged": True},
            "dataset_fingerprint": {"train": "same", "dev": "same",
                                    "target": runner._digest_rows(targets),
                                    "preservation": runner._digest_rows(preserved),
                                    "target_count": 1, "preservation_count": 1},
            "adapter": {"trainable_params": 12 if arm != "A" else 0}, "training": None,
        })
        # Deliberately wrong cached booleans: the raw predictions must win.
        provenance.write_jsonl(arm_out / "per_example.jsonl", [
            {**targets[0], "evaluation": "target", "arm": arm, "raw_prediction": "yes", "correct": False},
            {**preserved[0], "evaluation": "preservation", "arm": arm, "raw_prediction": "1.",
             "matched": False, "exact": False, "response_words": 99},
        ])
        return arm_out

    return out, config, core, targets, preserved, add


def test_partial_arm_cannot_decide_or_license_and_counts_failed_cost(stage):
    out, config, core, targets, preservation, add = stage
    add("A")
    add("B", status="FAILED")
    summary = runner.combine_stage(out, config, core, targets, preservation)
    assert summary["status"] == "PARTIAL"
    assert summary["completed_arms"] == ["A"]
    assert summary["missing_arms"] == ["B", "C"]
    assert summary["decision"]["winner"] is None
    assert summary["decision"]["phase_2_licensed"] is False
    resource = json.loads((out / "resource.json").read_text())
    assert resource["wall_seconds"] == 4  # Failed B cost is not hidden.
    assert resource["model_loads"] == 2
    assert resource["peak_rss_bytes"] == 10


def test_complete_combine_rescores_raw_predictions_and_sums_cost(stage):
    out, config, core, targets, preservation, add = stage
    for arm in runner.REQUIRED_ARMS:
        add(arm)
    summary = runner.combine_stage(out, config, core, targets, preservation)
    assert summary["status"] == "COMPLETE"
    assert all(info["target"]["correct"] == 1 for info in summary["arms"].values())
    assert all(info["preservation"]["matched"]["correct"] == 1 for info in summary["arms"].values())
    assert summary["decision"]["phase_2_licensed"] is False  # Smoke never licenses.
    resource = json.loads((out / "resource.json").read_text())
    assert resource["wall_seconds"] == 6
    assert resource["optimizer_steps_applied"] == 10
    assert resource["steps_skipped_by_scaler"] == 2
    assert len(provenance.read_jsonl(out / "per_example.jsonl")) == 6


@pytest.mark.parametrize("field,value", [
    ("source_sha256", {"source.py": "old"}),
    ("checkpoint_sha256_after", {"model.safetensors": "changed"}),
    ("canonical_core", "wrong-core"),
])
def test_incompatible_receipt_erases_stale_complete_decision(stage, field, value):
    out, config, core, targets, preservation, add = stage
    arm_out = add("A")
    provenance.write_json(out / "summary.json", {"status": "COMPLETE", "decision": {"phase_2_licensed": True}})
    path = arm_out / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    provenance.write_json(path, manifest, overwrite=True)
    with pytest.raises(ValueError):
        runner.combine_stage(out, config, core, targets, preservation)
    assert json.loads((out / "summary.json").read_text())["decision"]["phase_2_licensed"] is False
    assert json.loads((out / "run_manifest.json").read_text())["status"] == "FAILED"


def test_data_configuration_mismatch_is_refused(stage):
    out, config, core, targets, preservation, add = stage
    arm_out = add("A")
    path = arm_out / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["data_sha256"]["curriculum"] = "different"
    provenance.write_json(path, manifest, overwrite=True)
    with pytest.raises(ValueError, match="configuration mismatch"):
        runner.combine_stage(out, config, core, targets, preservation)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "wrong_reference"])
def test_row_corruption_is_refused(stage, kind):
    out, config, core, targets, preservation, add = stage
    arm_out = add("A")
    path = arm_out / "per_example.jsonl"
    rows = provenance.read_jsonl(path)
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "missing":
        rows.pop()
    else:
        rows[1]["reference"] = "2."
    provenance.write_jsonl(path, rows, overwrite=True)
    with pytest.raises(ValueError):
        runner.combine_stage(out, config, core, targets, preservation)


def test_nonempty_arm_refusal_precedes_checkpoint_access(tmp_path, monkeypatch):
    arm = tmp_path / "arm_A"
    arm.mkdir()
    (arm / "evidence.json").write_text("original")
    def unexpected(*args):
        pytest.fail("checkpoint should not be touched before refusing stale output")
    monkeypatch.setattr(provenance, "checkpoint_files", unexpected)
    with pytest.raises(FileExistsError):
        provenance.Run("arm_A", {}, results_dir=tmp_path, refuse_nonempty=True)
    assert (arm / "evidence.json").read_text() == "original"


def test_preservation_exact_one_point_boundary_is_accepted():
    arms = {arm: {} for arm in runner.REQUIRED_ARMS}
    target = {
        arm: [{"example_id": str(i), "correct": i < (60 if arm == "A" else 62)} for i in range(100)]
        for arm in runner.REQUIRED_ARMS
    }
    preserved = {
        arm: [{"example_id": str(i), "matched": i < (42 if arm == "A" else 41)} for i in range(100)]
        for arm in runner.REQUIRED_ARMS
    }
    decision = runner.decide(arms, target, preserved)
    assert decision["arms"]["B"]["meets_preservation_gate"] is True
    assert decision["arms"]["B"]["meaningfully_positive"] is True


def test_baseline_validity_requires_core_specific_split_counts():
    rows = [{"split": "showcase", "correct": i < 43} for i in range(64)]
    rows += [{"split": "stability_test", "correct": i < 125} for i in range(192)]
    assert runner.baseline_validity(rows, "a1", False)["matches"] is True
    assert runner.baseline_validity(rows, "a11", False)["matches"] is False


def test_loader_respects_declared_bfloat16_when_dtype_omitted(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    captured = {}
    class Qwen3AbstainForCausalLM:
        def eval(self): return self
        def to(self, device): return self
        def requires_grad_(self, flag): return self
    def model_load(path, **kwargs):
        captured.update(path=path, **kwargs)
        return Qwen3AbstainForCausalLM()
    monkeypatch.setattr(evaluation.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(evaluation.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(dtype=torch.bfloat16))
    monkeypatch.setattr(evaluation.AutoModelForCausalLM, "from_pretrained", model_load)
    evaluation.load_model(tmp_path, "cpu")
    assert captured["path"] == tmp_path
    assert captured["dtype"] == torch.bfloat16


def test_frozen_hash_ignores_adapter_but_detects_backbone_mutation():
    import torch
    model = torch.nn.Linear(4, 2)
    model.weight.requires_grad_(False)
    before = provenance.frozen_parameter_hash(model)
    with torch.no_grad():
        model.bias.add_(1)
    assert provenance.frozen_parameter_hash(model) == before
    with torch.no_grad():
        model.weight.add_(1)
    assert provenance.frozen_parameter_hash(model) != before
