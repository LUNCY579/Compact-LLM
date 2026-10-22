"""Synthetic teardown accounting only; never creates a subprocess or GPU query."""
import itertools
import json
from types import SimpleNamespace

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_supervisor as parent


def mocked_run(tmp_path, monkeypatch, mode):
    campaign = tmp_path / "campaign"; v3 = campaign / "v3"; v3.mkdir(parents=True)
    admission_path = v3 / "admission.json"; admission_path.write_text("{}", encoding="utf-8")
    (v3 / "review.json").write_text("{}", encoding="utf-8")
    admission = {"study_id": "SYNTHETIC-TEARDOWN-ONLY", "lane": "SYNTHETIC_NO_MODEL",
                 "review_receipt": "review.json", "head": "fixture", "dispatch_hashes": {}, "gpu_uuid": "fixture"}
    monkeypatch.setattr(parent, "ROOT", tmp_path); monkeypatch.setattr(parent, "CAMPAIGN", campaign)
    monkeypatch.setattr(parent, "V3", v3)
    monkeypatch.setattr(parent, "check_parent", lambda *args: (admission, {"uuid": "fixture", "free_bytes": 8 * parent.GIB}))
    clock = itertools.count(100., 5.)
    monkeypatch.setattr(parent.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(parent.psutil, "Process", lambda *args: SimpleNamespace(create_time=lambda: 1., exe=lambda: "FAKE-NOT-A-PROCESS"))
    actions = []
    class Child:
        pid = 987654
        returncode = 0
        polls = 0
        def poll(self):
            self.polls += 1
            return None if mode == "terminate_error_alive" and self.polls > 1 else self.returncode
        def wait(self, timeout):
            actions.append("wait")
            if mode in ("wait_timeout", "terminate_error_alive"):
                raise parent.subprocess.TimeoutExpired("SYNTHETIC", timeout)
            return self.returncode
        def kill(self): actions.append("kill")
    child = Child()
    class Job:
        def assign(self, item): actions.append("assign")
        def resume_suspended(self, item): actions.append("resume")
        def terminate(self):
            actions.append("terminate")
            if mode == "terminate_error_alive": raise RuntimeError("synthetic terminate failed")
        def close(self):
            actions.append("close")
            if mode in ("close_error", "no_child_close_error"): raise RuntimeError("synthetic close failed")
            if mode == "foreign_lock":
                lock = campaign / "native_worker.lock"
                record = json.loads(lock.read_text(encoding="utf-8")); record["token"] = "foreign"
                lock.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(parent, "WindowsJob", Job)
    def popen(*args, **kwargs):
        actions.append("popen")
        assert kwargs["creationflags"] & 0x4 and kwargs["env"]["TMP"].startswith(str(v3))
        if mode in ("no_child", "no_child_close_error"): raise OSError("synthetic Popen refusal")
        return child
    monkeypatch.setattr(parent.subprocess, "Popen", popen)
    def forbidden(*args): raise AssertionError("real GPU/process sample must not be queried")
    monkeypatch.setattr(parent, "gpu_inventory", forbidden); monkeypatch.setattr(parent, "resource_sample", forbidden)
    out = v3 / "run"
    return campaign, out, admission_path, child, actions


@pytest.mark.parametrize("mode", ["wait_timeout", "terminate_error_alive"])
def test_child_teardown_exceptions_still_charge_and_journal(tmp_path, monkeypatch, mode):
    campaign, out, admission, child, actions = mocked_run(tmp_path, monkeypatch, mode)
    expected = parent.subprocess.TimeoutExpired if mode == "wait_timeout" else RuntimeError
    with pytest.raises(expected): parent.run(tmp_path / "source", out, admission)
    terminal = json.loads((out / "terminal_receipt.json").read_text(encoding="utf-8"))
    ledger = [json.loads(x) for x in (campaign / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    assert terminal["native_worker_seconds"] > 0 and ledger[-1]["native_worker_seconds"] == terminal["native_worker_seconds"]
    assert terminal["peak_rss_bytes"] is None and ledger[-1]["event"] == "native_completed"
    assert "close" in actions
    if mode == "terminate_error_alive":
        assert terminal["status"] == "SUPERVISOR_CLEANUP_UNVERIFIED"
        assert not terminal["worker_termination_verified"] and not terminal["native_accounting_complete"]
        assert (campaign / "native_worker.lock").exists()
    else:
        assert terminal["worker_termination_verified"] and not (campaign / "native_worker.lock").exists()


def test_close_error_keeps_unknown_descendant_lock_even_when_launcher_poll_terminal(tmp_path, monkeypatch):
    campaign, out, admission, child, actions = mocked_run(tmp_path, monkeypatch, "close_error")
    assert parent.run(tmp_path / "source", out, admission) == 2
    terminal = json.loads((out / "terminal_receipt.json").read_text(encoding="utf-8"))
    ledger = [json.loads(x) for x in (campaign / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    assert terminal["status"] == "SUPERVISOR_CLEANUP_UNVERIFIED" and terminal["cleanup_error"]
    assert not terminal["worker_termination_verified"] and not terminal["native_accounting_complete"]
    assert terminal["native_worker_seconds"] > 0 and ledger[-1]["native_worker_seconds"] == terminal["native_worker_seconds"]
    assert terminal["peak_rss_bytes"] is None and (campaign / "native_worker.lock").exists()


@pytest.mark.parametrize("mode", ["no_child", "no_child_close_error"])
def test_no_child_is_zero_charge_null_measurements_and_no_started_event(tmp_path, monkeypatch, mode):
    campaign, out, admission, child, actions = mocked_run(tmp_path, monkeypatch, mode)
    assert parent.run(tmp_path / "source", out, admission) == 2
    terminal = json.loads((out / "terminal_receipt.json").read_text(encoding="utf-8"))
    ledger = [json.loads(x) for x in (campaign / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    assert terminal["native_worker_seconds"] == 0 and terminal["peak_rss_bytes"] is None
    assert "worker_pid" not in terminal and not any(x["event"] == "native_started" for x in ledger)
    assert terminal["worker_termination_verified"] and terminal["native_accounting_complete"]
    assert not (campaign / "native_worker.lock").exists()


@pytest.mark.parametrize("mode", ["normal", "foreign_lock"])
def test_normal_closed_child_clears_only_own_lock(tmp_path, monkeypatch, mode):
    campaign, out, admission, child, actions = mocked_run(tmp_path, monkeypatch, mode)
    assert parent.run(tmp_path / "source", out, admission) == 0
    terminal = json.loads((out / "terminal_receipt.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "WORKER_TERMINAL" and terminal["native_worker_seconds"] > 0
    assert terminal["worker_termination_verified"] and terminal["native_accounting_complete"]
    assert (campaign / "native_worker.lock").exists() == (mode == "foreign_lock")
