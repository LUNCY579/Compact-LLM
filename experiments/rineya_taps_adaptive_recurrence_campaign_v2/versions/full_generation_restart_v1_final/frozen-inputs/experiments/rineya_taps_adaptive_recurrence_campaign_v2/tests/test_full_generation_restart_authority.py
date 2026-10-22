"""Temporary fabricated grants test refusals; no actual authorization/model."""
import json
from pathlib import Path
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import full_generation_restart_authority as authority


def packet(tmp_path, monkeypatch):
    package = tmp_path / "FABRICATED_AUTHORITY_NOT_NATIVE"
    package.mkdir()
    monkeypatch.setattr(authority, "PACKAGE", package)
    worker = package / "bulk_restart_native.py"
    worker.write_text("# fabricated nonexecuted bytes\n")
    ledger = package / "BUDGET_LEDGER.jsonl"
    ledger.write_text("")
    grant_path = package / "HUMAN_FULL_GENERATION_RESTART_ALLOWANCE_001.json"
    grant = {"status": "EXPLICIT_ADDITIONAL_FULL_GENERATION_RESTART_AUTHORIZED",
             "id": "FULL-GENERATION-RESTART-001", "lane": "GPU_FP32_FULL_GENERATION_RESTART",
             "attempts": 1, "max_native_worker_seconds": 16000, "model_API_attempts": 28436,
             "state_adjoints": 0, "cases": 288, "physical_evaluations": 3296, "maximum_prefix_tokens": 357,
             "scientific_generation_cap": 8, "SMOKE_generation_cap": 24, "no_further_retry": True,
             "parent_run": "native-bulk-001", "training": False,
             "full_bulk_matrix_retry": True, "worker_sha256": authority.sha(worker)}
    grant_path.write_text(json.dumps(grant))
    admission = {"requires_additional_human_authorization": True,
                 "human_extension_path": str(grant_path), "human_extension_sha256": authority.sha(grant_path),
                 "hard_deadline_seconds": 16000, "max_new_tokens": 8,
                 "qualified_maximum_prefix_tokens": 357, "native_finalization_reserve_seconds": 1800,
                 "forecast": {"physical_case_evaluations": 3296, "maximum_model_API_attempts": 28436},
                 "dispatch_hashes": {"experiments/rineya_taps_adaptive_recurrence_campaign_v2/bulk_restart_native.py": authority.sha(worker)}}
    return grant_path, grant, admission, ledger


def test_missing_actual_grant_and_template_cannot_authorize(tmp_path, monkeypatch):
    path, grant, admission, _ = packet(tmp_path, monkeypatch)
    proposed = path.parent / "TEMPLATE_ONLY.json"
    proposed.write_bytes(path.read_bytes())
    path.unlink()
    with pytest.raises(PermissionError, match="actual human allowance absent"):
        authority.validate_human_extension(admission)
    admission["human_extension_path"] = str(proposed)
    with pytest.raises(PermissionError):
        authority.validate_human_extension(admission)


@pytest.mark.parametrize("key,value", [("attempts", 2), ("model_API_attempts", 28437),
                                        ("full_bulk_matrix_retry", False), ("scientific_generation_cap", 64), ("SMOKE_generation_cap", 8), ("no_further_retry", False), ("worker_sha256", "wrong")])
def test_wrong_scope_refuses_even_when_hash_matches(tmp_path, monkeypatch, key, value):
    path, grant, admission, _ = packet(tmp_path, monkeypatch)
    grant[key] = value
    path.write_text(json.dumps(grant))
    admission["human_extension_sha256"] = authority.sha(path)
    with pytest.raises(PermissionError, match="exact original-config-preserving"):
        authority.validate_human_extension(admission)


@pytest.mark.parametrize("event", [{"event": "native_started"}, {"event": "native_completed", "worker_pid": 123}])
def test_started_or_launched_failure_consumes_single_attempt(tmp_path, monkeypatch, event):
    _, _, admission, ledger = packet(tmp_path, monkeypatch)
    assert authority.validate_human_extension(admission)
    ledger.write_text(json.dumps({**event, "human_extension_sha256": admission["human_extension_sha256"]})+"\n")
    # The contained child must validate the scope after its parent's start row.
    assert authority.validate_human_scope(admission)
    with pytest.raises(PermissionError, match="already consumed"):
        authority.validate_human_extension(admission)


@pytest.mark.parametrize("key,value", [("hard_deadline_seconds", 16001), ("max_new_tokens", 64),
                                        ("native_finalization_reserve_seconds", 1)])
def test_admission_cannot_outgrow_exact_grant(tmp_path, monkeypatch, key, value):
    _, _, admission, _ = packet(tmp_path, monkeypatch)
    admission[key] = value
    with pytest.raises(PermissionError, match="admission differs"):
        authority.validate_human_extension(admission)
