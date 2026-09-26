"""Runner lifecycle evidence using tiny CPU models, including actual hard exit."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "tests"))
from slcmi import provenance
from test_training_safety import TinyAdapter, row

spec = importlib.util.spec_from_file_location("phase1_lifecycle_runner", LAB / "scripts/phase1_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Probe(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class Tokenizer:
    def __call__(self, *args, **kwargs):
        return Probe(input_ids=torch.tensor([[1]]))


class LifecycleModel(TinyAdapter):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.calls = 0

    def forward(self, input_ids, labels=None, **kwargs):
        if labels is None:
            return SimpleNamespace(logits=torch.zeros(1, 1, 2))
        if torch.is_grad_enabled():
            self.calls += 1
            if self.action == "kill_mid_accumulation" and self.calls == 3:
                os._exit(37)
            if self.action == "fail_mid_accumulation" and self.calls == 3:
                raise RuntimeError("injected accumulation failure")
        return super().forward(input_ids=input_ids, labels=labels, **kwargs)

    def save_pretrained(self, out):
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"lora_A": self.lora_A.weight.detach().cpu()}, out / "adapter.pt")

    def unload(self):
        return self


def install_fixture(patch, directory: Path, arm="B", action="success"):
    patch.setattr(provenance.torch.cuda, "is_available", lambda: False)
    patch.setattr(provenance, "git", lambda *args: "cpu-fixture")
    source = {"fixture.py": "a" * 64}
    base = {"model.safetensors": "b" * 64, "config.json": "c" * 64}
    for module in (runner, provenance):
        patch.setattr(module, "source_hashes", lambda: source)
        patch.setattr(module, "checkpoint_files", lambda path: base)
    patch.setattr(runner, "load_packs", lambda path: [])
    patch.setattr(runner, "sample_curriculum", lambda packs, split, count: [row(1), row(2)])
    patch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    patch.setattr(runner, "build_dataset", lambda tokenizer, rows, length: (rows, {"n_dropped_too_long": 0}))
    loads = []

    def load(*args, **kwargs):
        if action == "fail_load":
            raise RuntimeError("injected model load failure")
        model = LifecycleModel(action).eval()
        if arm == "A":
            model.requires_grad_(False)
        loads.append(model)
        return Tokenizer(), model

    patch.setattr(runner, "load_model", load)
    patch.setattr(runner, "build_adapter", lambda model, *a: (model, runner.count_trainable(model)))
    patch.setattr(runner, "verify_adapter", lambda *a: {"all_passed": True, "trained_layers": [17, 18, 19]})
    targets = [{"example_id": "t1", "split": "showcase", "variant": "base", "task_type": "test",
                "answerable": True, "condition": "A_full", "accepted_answers": ["yes"]}]
    preservation = [{"example_id": "p1", "task_family": "counting_sets", "reference": "1.",
                     "reference_numbers": ["1"]}]

    def evaluate(*args, **kwargs):
        if action == "fail_eval":
            raise RuntimeError("injected evaluation failure")
        return [{**targets[0], "raw_prediction": "yes", "correct": True, "malformed": False}]

    patch.setattr(runner, "evaluate_rows", evaluate)
    patch.setattr(runner, "evaluate_preservation", lambda *a, **kw: [
        {**preservation[0], "raw_prediction": "1.", "matched": True, "exact": True, "response_words": 1}])
    if action == "kill_after_commit":
        original_append = runner.append_jsonl

        def append(path, record):
            if record["event"] == "optimizer_attempt" and record["attempt"] == 1:
                # Atomic checkpoint survived; callback delivery did not.
                with path.open("ab") as stream:
                    stream.write(b'{"event":"optimizer_attempt",')
                    stream.flush()
                    os.fsync(stream.fileno())
                os._exit(37)
            original_append(path, record)

        patch.setattr(runner, "append_jsonl", append)
    config = {"schema_version": 3, "core": "a1", "smoke": True, "device": "cpu", "max_steps": 6,
              "grad_accum": 2, "lr": 1e-4, "train_packs": 2, "dev_packs": 2,
              "data_sha256": {"curriculum": "fixed"}}
    core = {"path": directory / "canonical", "dtype": "bfloat16"}
    return config, core, targets, preservation, loads


def execute(directory, *, arm="B", action="success", resume=False):
    with pytest.MonkeyPatch.context() as patch:
        config, core, targets, preservation, loads = install_fixture(patch, directory, arm, action)
        runner.execute_arm(directory / f"arm_{arm}", arm, config, core, targets, preservation, resume=resume)
        return len(loads)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_refusal_precedes_core_access_and_resume_identity_is_strict(tmp_path, monkeypatch):
    config, core, _, _, _ = install_fixture(monkeypatch, tmp_path)
    arm = tmp_path / "arm_B"
    identity = runner.prepare_arm(arm, "B", config, core, resume=False)
    monkeypatch.setattr(runner, "checkpoint_files", lambda path: pytest.fail("fresh refusal must precede hashing"))
    with pytest.raises(FileExistsError):
        runner.prepare_arm(arm, "B", config, core, resume=False)
    monkeypatch.setattr(runner, "checkpoint_files", lambda path: identity["checkpoint_sha256"])
    with pytest.raises(ValueError, match="config"):
        runner.prepare_arm(arm, "B", {**config, "lr": 2e-4}, core, resume=True)
    assert read(arm / "identity.json") == identity


def test_zero_progress_resume_requires_durable_no_training_entry(tmp_path, monkeypatch):
    config, core, _, _, _ = install_fixture(monkeypatch, tmp_path)
    arm = tmp_path / "arm_B"
    identity = runner.prepare_arm(arm, "B", config, core, resume=False)
    assert runner.prepare_arm(arm, "B", config, core, resume=True) == identity
    marker = arm / "segments/000001-test/training_entry.json"
    provenance.write_json(marker, {})
    with pytest.raises(ValueError, match="training was entered"):
        runner.prepare_arm(arm, "B", config, core, resume=True)


def test_completed_arm_resume_is_idempotent_and_keeps_one_load(tmp_path):
    assert execute(tmp_path) == 1
    before = read(tmp_path / "arm_B/run_manifest.json")
    assert execute(tmp_path, resume=True) == 0
    assert read(tmp_path / "arm_B/run_manifest.json") == before
    assert before["resource"]["model_loads"] == 1
    assert before["committed_training"]["optimizer_steps_attempted"] == 6


def test_frozen_a_explicit_evaluation_retry_has_no_optimizer(tmp_path):
    with pytest.raises(RuntimeError, match="evaluation failure"):
        execute(tmp_path, arm="A", action="fail_eval")
    failed = read(tmp_path / "arm_A/run_manifest.json")
    assert failed["status"] == "FAILED"
    assert execute(tmp_path, arm="A", resume=True) == 1
    receipt = read(tmp_path / "arm_A/run_manifest.json")
    assert receipt["status"] == "COMPLETE"
    assert receipt["resource"]["model_loads"] == 2
    assert receipt["committed_training"]["optimizer_steps_attempted"] == 0
    assert not (tmp_path / "arm_A/checkpoints").exists()


def test_failure_receipt_resume_total_budget_and_segment_deltas(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="accumulation failure"):
        execute(tmp_path, action="fail_mid_accumulation")
    arm = tmp_path / "arm_B"
    failed = read(arm / "run_manifest.json")
    assert failed["committed_training"]["optimizer_steps_attempted"] == 1
    assert execute(tmp_path, resume=True) == 1
    manifest = read(arm / "run_manifest.json")
    info = read(arm / "arm_B_info.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["resource"]["model_loads"] == 2
    assert manifest["resource"]["optimizer_steps_attempted"] == 6
    assert manifest["resource"]["examples_consumed"] == 12
    segments = sorted((arm / "segments").iterdir())
    assert read(segments[0] / "resource.json")["optimizer_steps_attempted"] == 1
    assert read(segments[1] / "resource.json")["optimizer_steps_attempted"] == 5
    assert read(segments[1] / "resource.json")["optimizer_steps_applied"] == 5
    assert read(segments[1] / "resource.json")["examples_consumed"] == 10
    assert [r["attempt"] for r in info["training"]["trajectory"]] == list(range(1, 7))
    runner.validate_lifecycle_receipt(arm, "B", manifest, info)


@pytest.mark.parametrize("action", ["kill_mid_accumulation", "kill_after_commit"])
def test_actual_hard_kill_preserves_segments_reconciles_checkpoint_and_unknown_costs(tmp_path, action):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), action],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 37, child.stdout + child.stderr
    arm = tmp_path / "arm_B"
    first = next((arm / "segments").iterdir())
    assert not (first / "run_manifest.json").exists()
    state = torch.load(arm / "checkpoints/latest.pt", map_location="cpu", weights_only=True)
    assert state["progress"]["completed_attempts"] == 1
    with pytest.MonkeyPatch.context() as patch:
        config, core, targets, preservation, _ = install_fixture(patch, tmp_path)
        partial = runner.combine_stage(tmp_path, config, core, targets, preservation)
    assert partial["status"] == "PARTIAL"
    assert partial["decision"]["winner"] is None
    assert partial["decision"]["phase_2_licensed"] is False
    partial_cost = read(tmp_path / "resource.json")
    assert partial_cost["model_loads"] == 1
    assert partial_cost["resource_accounting_complete"] is False
    assert partial_cost["optimizer_steps_attempted"] == 1
    assert execute(tmp_path, resume=True) == 1
    manifest = read(arm / "run_manifest.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["resource"]["model_loads"] == 2
    assert manifest["resource"]["model_loads_upper_bound"] == 2
    assert manifest["resource"]["resource_accounting_complete"] is False
    assert manifest["resource"]["numeric_costs_are_lower_bounds"] is True
    assert manifest["resource"]["unknown_segment_ids"] == [first.name]
    assert "wall_seconds" in manifest["resource"]["incomplete_metric_fields"]
    assert not (first / "run_manifest.json").exists()  # No fabricated exit receipt.
    events = provenance.read_jsonl(arm / "training_progress.jsonl")
    assert [r["attempt"] for r in events if r["event"] == "optimizer_attempt"] == list(range(1, 7))
    assert manifest["checkpoint_receipt"]["pending_validation"] is None
    runner.validate_lifecycle_receipt(arm, "B", manifest, read(arm / "arm_B_info.json"))


def test_resume_of_training_complete_evaluation_failure_does_not_add_attempts(tmp_path):
    with pytest.raises(RuntimeError, match="evaluation failure"):
        execute(tmp_path, action="fail_eval")
    assert execute(tmp_path, resume=True) == 1
    arm = tmp_path / "arm_B"
    final = read(arm / "run_manifest.json")
    segment = sorted((arm / "segments").iterdir())[-1]
    assert final["committed_training"]["optimizer_steps_attempted"] == 6
    assert read(segment / "resource.json")["optimizer_steps_attempted"] == 0
    assert read(segment / "resource.json")["optimizer_steps_applied"] == 0
    assert read(segment / "run_manifest.json")["resumed_from_attempt"] == 6


def test_corrupt_latest_never_falls_back_to_older_or_temporary_checkpoint(tmp_path):
    with pytest.raises(RuntimeError):
        execute(tmp_path, action="fail_mid_accumulation")
    checkpoint = tmp_path / "arm_B/checkpoints/latest.pt"
    (checkpoint.parent / "older.pt").write_bytes(checkpoint.read_bytes())
    checkpoint.write_bytes(b"corrupt latest transaction")
    with pytest.raises(Exception):
        execute(tmp_path, resume=True)
    assert len(list((tmp_path / "arm_B/segments").iterdir())) == 1


def test_actual_changed_initializer_refused_before_restore_or_training(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError):
        execute(tmp_path, action="fail_mid_accumulation")
    arm = tmp_path / "arm_B"
    original_checkpoint = (arm / "checkpoints/latest.pt").read_bytes()
    config, core, targets, preservation, _ = install_fixture(monkeypatch, tmp_path)
    original_load = runner.load_model

    def changed(*args, **kwargs):
        tokenizer, model = original_load(*args, **kwargs)
        with torch.no_grad():
            model.lora_A.weight.add_(0.25)
        return tokenizer, model

    monkeypatch.setattr(runner, "load_model", changed)
    with pytest.raises(ValueError, match="adapter_initialization"):
        runner.execute_arm(arm, "B", config, core, targets, preservation, resume=True)
    assert (arm / "checkpoints/latest.pt").read_bytes() == original_checkpoint
    assert read(arm / "run_manifest.json")["committed_training"]["optimizer_steps_attempted"] == 1


def test_preparation_failure_during_resume_preserves_original_committed_budget(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError):
        execute(tmp_path, action="fail_mid_accumulation")
    config, core, targets, preservation, _ = install_fixture(monkeypatch, tmp_path)

    def fail(*args):
        raise RuntimeError("injected dataset preparation failure")

    monkeypatch.setattr(runner, "load_packs", fail)
    arm = tmp_path / "arm_B"
    with pytest.raises(RuntimeError, match="preparation failure"):
        runner.execute_arm(arm, "B", config, core, targets, preservation, resume=True)
    receipt = read(arm / "run_manifest.json")
    assert receipt["committed_training"]["optimizer_steps_attempted"] == 1
    segment = sorted((arm / "segments").iterdir())[-1]
    assert read(segment / "started_manifest.json")["resumed_from_attempt"] == 1
    assert read(segment / "resource.json")["model_loads"] == 0


def test_os_lock_blocks_live_owner_and_is_released_by_hard_exit(tmp_path):
    lock = tmp_path / "arm.lock"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    script = """import os,sys,time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from slcmi.provenance import ExclusiveFileLock
with ExclusiveFileLock(Path(sys.argv[2])):
    print('LOCKED', flush=True)
    sys.stdin.readline()
    os._exit(37)
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(LAB / "src"), str(lock)],
                             cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="owns the run lock"):
            with provenance.ExclusiveFileLock(lock):
                pytest.fail("concurrent owner entered")
        child.stdin.write("exit\n")
        child.stdin.flush()
        assert child.wait(timeout=30) == 37
        assert lock.exists()  # A stale file is harmless; ownership was OS state.
        with provenance.ExclusiveFileLock(lock):
            pass
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=30)


def test_cuda_metric_failure_cannot_mask_exact_cpu_checkpoint_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "checkpoint_files", lambda path: {"model.safetensors": "unchanged"})
    monkeypatch.setattr(provenance, "source_hashes", lambda: {})
    monkeypatch.setattr(provenance, "git", lambda *args: "fixture")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def sticky(*args):
        raise RuntimeError("sticky CUDA device assert")

    for name in ("reset_peak_memory_stats", "mem_get_info", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, sticky)
    with pytest.raises(RuntimeError, match="original training failure"):
        with provenance.Run("segment", {}, results_dir=tmp_path, core_path=tmp_path) as run:
            raise RuntimeError("original training failure")
    receipt = read(tmp_path / "segment/run_manifest.json")
    assert receipt["status"] == "FAILED"
    assert receipt["checkpoint_sha256_before"] == receipt["checkpoint_sha256_after"]
    assert receipt["canonical_checkpoint_unchanged"] is True
    assert receipt["resource"]["peak_gpu_allocated_bytes"] is None
    assert receipt["resource"]["metric_errors"]["peak_gpu_allocated_bytes"]
    assert read(tmp_path / "segment/checkpoint_integrity.json")["canonical_checkpoint_unchanged"] is True


@pytest.mark.parametrize("argument", [["--max-steps", "301"], ["--grad-accum", "4"], ["--lr", "0.0002"]])
def test_cli_refuses_scientific_budget_or_recipe_changes_before_loading(argument, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["phase1_run.py", "--arms", "B", "--device", "cpu", *argument])
    monkeypatch.setattr(runner, "load_model", lambda *a, **kw: pytest.fail("model must not load"))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_cli_pins_gpu_policy_before_scientific_inputs_or_model_access(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["phase1_run.py", "--arms", "B"])
    for key, value in runner.GPU_EXECUTION_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setattr(runner, "sha256", lambda *a: pytest.fail("reject environment before data access"))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_completed_lifecycle_rejects_duplicate_logical_prefix(tmp_path):
    execute(tmp_path)
    arm = tmp_path / "arm_B"
    rows = provenance.read_jsonl(arm / "training_progress.jsonl")
    provenance.write_jsonl(arm / "training_progress.jsonl", [rows[0], *rows], overwrite=True)
    with pytest.raises(ValueError, match="logical training progress"):
        runner.validate_lifecycle_receipt(arm, "B", read(arm / "run_manifest.json"), read(arm / "arm_B_info.json"))


if __name__ == "__main__":
    execute(Path(sys.argv[1]), action=sys.argv[2])
