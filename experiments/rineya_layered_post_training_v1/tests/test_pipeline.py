from dataclasses import replace
import json
import pytest
from experiments.rineya_layered_post_training_v1.config import ARMS
from experiments.rineya_layered_post_training_v1.data import fixtures
from experiments.rineya_layered_post_training_v1.engine import run, confirm
from experiments.rineya_layered_post_training_v1.runtime import tiny_runtime


@pytest.mark.parametrize("arm", ARMS)
def test_end_to_end_all_arms(cfg, tmp_path, arm):
    config = replace(cfg, arm=arm)
    result = run(tiny_runtime(config), fixtures(), tmp_path / arm)
    assert result["status"] == "SYNTHETIC_TORCH_MECHANICS_PASS"
    assert not result["confirmation_used"]
    assert result["budget"]["actor_steps"] == config.actor_steps
    assert result["selected_scores"]["retention"] >= result["baseline_scores"]["retention"] - config.max_retention_drop
    assert (tmp_path / arm / "selection.lock.json").exists()
    assert not (tmp_path / arm / "confirmation.json").exists()
    if "soup2" in ARMS[arm]:
        event = next(e for e in result["events"] if e["stage"] == "soup2")
        assert event["recipes"]["three_parent"]["parents"][0] == "base"
    confirmed = confirm(tiny_runtime(config), fixtures(), tmp_path / arm)
    assert confirmed["status"] == "SYNTHETIC_CONFIRMATION_ONLY"
    with pytest.raises(ValueError, match="already opened"):
        confirm(tiny_runtime(config), fixtures(), tmp_path / arm)


def test_lock_and_confirmation_fingerprints(cfg, tmp_path):
    output = tmp_path / "run"
    run(tiny_runtime(cfg), fixtures(), output)
    rows = fixtures()
    rows[-1] = replace(rows[-1], revision="tampered")
    with pytest.raises(ValueError, match="data changed"):
        confirm(tiny_runtime(cfg), rows, output)
    lockpath = output / "selection.lock.json"
    lock = json.loads(lockpath.read_text())
    lock["checkpoint"] = "base"
    lockpath.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="lock tampered"):
        confirm(tiny_runtime(cfg), fixtures(), output)


def test_native_critic_failure_blocks_rl3(cfg, tmp_path, monkeypatch):
    from experiments.rineya_layered_post_training_v1 import engine
    config = replace(cfg, arm="full")
    runtime = tiny_runtime(config)
    runtime.native = True  # Unit-test injected native policy, NOT a native model test.
    monkeypatch.setattr(engine, "resource_report", lambda *a, **kw: {"blockers": []})
    monkeypatch.setattr(engine, "critic_audit", lambda *a: {"auroc": .1, "mse": 1., "constant_mse": .2})
    with pytest.raises(ValueError, match="critic failed"):
        run(runtime, fixtures(), tmp_path / "blocked")
    failure = json.loads((tmp_path / "blocked" / "failure.json").read_text())
    assert failure["status"] == "FAILED_OR_BLOCKED_NO_PROMOTION"
    assert not (tmp_path / "blocked" / "selection.lock.json").exists()
    assert "rl3" not in runtime.budget.by_phase


def test_no_successful_trace_is_not_substituted(cfg, tmp_path, monkeypatch):
    from experiments.rineya_layered_post_training_v1 import engine
    monkeypatch.setattr(engine, "score", lambda *a: 0.)
    with pytest.raises(ValueError, match="no verifier-approved"):
        run(tiny_runtime(cfg), fixtures(), tmp_path / "no_success")


def test_truncation_fails_closed(cfg, tmp_path):
    runtime = tiny_runtime(replace(cfg, max_new_tokens=1))
    with pytest.raises(ValueError, match="truncated rollout group"):
        run(runtime, fixtures(), tmp_path / "truncated")


def test_no_overwrite(cfg, tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run(tiny_runtime(cfg), fixtures(), output)
