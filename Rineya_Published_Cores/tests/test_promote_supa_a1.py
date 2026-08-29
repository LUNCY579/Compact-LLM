from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

VAULT = Path(__file__).resolve().parents[1]


def load_module():
    path = VAULT / "_promote_supa_a1.py"
    spec = importlib.util.spec_from_file_location("promote_supa_a1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROMOTE = load_module()


def test_process_liveness_probe_is_read_only_for_the_current_process() -> None:
    assert PROMOTE.process_is_alive(os.getpid())


def test_identity_and_measured_results_are_pinned() -> None:
    core = PROMOTE.core_payload("2026-08-27T00:00:00Z")
    assert core["codename"] == "Supa-A1"
    assert core["tracking_label"] == "Supa-A1"
    assert "posthoc" in core["status"]
    assert core["checks"]["weight_sha256"] == PROMOTE.MODEL_SHA256
    assert core["evaluation"]["checkpoint_results"]["showcase64"]["hits"] == 43
    assert core["evaluation"]["checkpoint_results"]["stability192"]["hits"] == 125
    assert core["evaluation"]["offline_gate_reference"]["showcase64"]["hits"] == 46


def test_no_bias_control_is_not_published_as_nimbus_a1() -> None:
    core = PROMOTE.core_payload("2026-08-27T00:00:00Z")
    control = core["evaluation"]["no_bias_control"]
    assert control["classification"] == "abstention_ablation_control_not_a_named_core"
    catalog = {"cores": [], "updated_at_utc": "old"}
    updated = PROMOTE.catalog_with_supa_a1(catalog, core)
    assert len(updated["cores"]) == 1
    assert updated["tracking_labels"]["Nimbus-A1"]["directory"] == (
        "Rineya-Nimbus-1.6.0-devfreeze"
    )


def test_catalog_promotion_refuses_identity_collision() -> None:
    core = PROMOTE.core_payload("2026-08-27T00:00:00Z")
    catalog = {"cores": [{"core_id": PROMOTE.CORE_ID}]}
    with pytest.raises(FileExistsError):
        PROMOTE.catalog_with_supa_a1(catalog, core)


def test_model_card_states_loader_and_evaluation_caveats() -> None:
    card = PROMOTE.model_card()
    assert "trust_remote_code=True" in card
    assert "greedy" in card
    assert "43/64" in card
    assert "125/192" in card
    assert "does not inherit" in card
    assert "subfolder" in card
    assert "previously inspected validation" in card


def test_source_manifest_contains_no_absolute_machine_path() -> None:
    encoded = json.dumps(
        [PROMOTE.source_manifest(), PROMOTE.sanitized_original_manifest()]
    )
    assert "C:\\\\Users" not in encoded
    assert ".claude/worktrees" not in encoded


def test_release_manifest_hashes_every_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    manifest = PROMOTE.release_manifest(tmp_path, "now")
    assert set(manifest["files"]) == {"a.txt", "nested/b.txt"}
    assert manifest["files"]["a.txt"]["sha256"] == PROMOTE.sha256_file(
        tmp_path / "a.txt"
    )


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_apply_rolls_back_post_commit_failures_including_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    vault = tmp_path / "vault"
    source = tmp_path / "source"
    vault.mkdir()
    source.mkdir()
    catalog = vault / "CATALOG.json"
    original_catalog = '{"cores": [], "marker": "original"}\n'
    catalog.write_text(original_catalog, encoding="utf-8")
    (source / "modeling_abstain.py").write_text("runtime", encoding="utf-8")

    monkeypatch.setattr(PROMOTE, "VAULT", vault)
    monkeypatch.setattr(PROMOTE, "CATALOG", catalog)
    monkeypatch.setattr(PROMOTE, "SOURCE_ROOT", source)
    monkeypatch.setattr(PROMOTE, "DESTINATION", vault / "core")
    monkeypatch.setattr(PROMOTE, "ROOT_RUNTIME_MODULE", vault / "modeling_supa_a1.py")
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", vault / ".transaction.json")
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", vault / ".transaction.lock")
    monkeypatch.setattr(
        PROMOTE,
        "SOURCE_MODELING_SHA256",
        PROMOTE.sha256_file(source / "modeling_abstain.py"),
    )
    monkeypatch.setattr(PROMOTE, "validate_sources", dict)
    monkeypatch.setattr(PROMOTE, "verify_directory", lambda _: None)

    def fake_build(staging: Path, published_at: str) -> None:
        del published_at
        staging.mkdir()
        PROMOTE.atomic_json(staging / "CORE.json", {"core_id": PROMOTE.CORE_ID})

    monkeypatch.setattr(PROMOTE, "build_staging", fake_build)
    monkeypatch.setattr(
        PROMOTE,
        "catalog_with_supa_a1",
        lambda current, core: {"cores": [core], "previous": current},
    )
    monkeypatch.setattr(
        PROMOTE,
        "verify",
        lambda: (_ for _ in ()).throw(failure_type("injected verification failure")),
    )

    with pytest.raises(failure_type, match="injected verification failure"):
        PROMOTE.apply()

    assert catalog.read_text(encoding="utf-8") == original_catalog
    assert not (vault / "core").exists()
    assert not (vault / "modeling_supa_a1.py").exists()
    assert not (vault / ".transaction.json").exists()


def test_recover_rejects_a_journal_targeting_the_vault_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    marker = vault / "must-survive.txt"
    marker.write_text("safe", encoding="utf-8")
    journal = vault / ".transaction.json"
    lock = vault / ".transaction.lock"
    lock.write_text(
        json.dumps({"transaction_id": "tx", "owner_pid": 999999}),
        encoding="utf-8",
    )
    journal.write_text(
        json.dumps(
            {
                "transaction_id": "tx",
                "owner_pid": 999999,
                "destination": str(vault),
                "staging": str(vault / f".{PROMOTE.DESTINATION_NAME}.staging-safe"),
                "catalog_backup": str(vault / ".CATALOG.json.backup-safe"),
                "runtime_module": str(vault / PROMOTE.RUNTIME_MODULE_NAME),
                "runtime_was_absent": True,
                "original_catalog_sha256": "0" * 64,
                "new_catalog_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(PROMOTE, "VAULT", vault)
    monkeypatch.setattr(PROMOTE, "DESTINATION", vault / PROMOTE.DESTINATION_NAME)
    monkeypatch.setattr(PROMOTE, "CATALOG", vault / "CATALOG.json")
    monkeypatch.setattr(
        PROMOTE, "ROOT_RUNTIME_MODULE", vault / PROMOTE.RUNTIME_MODULE_NAME
    )
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", journal)
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", lock)
    monkeypatch.setattr(PROMOTE, "process_is_alive", lambda _: False)

    with pytest.raises(ValueError, match="unexpected destination"):
        PROMOTE.recover()

    assert marker.read_text(encoding="utf-8") == "safe"
    assert journal.exists()


def test_apply_refuses_an_existing_transaction_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / ".transaction.lock"
    lock.write_text(
        json.dumps({"transaction_id": "tx", "owner_pid": 999999}),
        encoding="utf-8",
    )
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", lock)
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", tmp_path / ".journal.json")

    with pytest.raises(RuntimeError, match="Another Supa-A1 transaction"):
        PROMOTE.apply()

    assert lock.exists()


def test_recover_refuses_a_live_transaction_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    catalog = vault / "CATALOG.json"
    catalog.write_text('{"state": "original"}\n', encoding="utf-8")
    original_hash = PROMOTE.sha256_file(catalog)
    transaction_id = "live-tx"
    owner_pid = 12345
    lock = vault / ".transaction.lock"
    journal = vault / ".transaction.json"
    lock.write_text(
        json.dumps({"transaction_id": transaction_id, "owner_pid": owner_pid}),
        encoding="utf-8",
    )
    journal.write_text(
        json.dumps(
            {
                "transaction_id": transaction_id,
                "owner_pid": owner_pid,
                "destination": str(vault / PROMOTE.DESTINATION_NAME),
                "staging": str(vault / f".{PROMOTE.DESTINATION_NAME}.staging-live"),
                "catalog_backup": str(vault / ".CATALOG.json.backup-live"),
                "runtime_module": str(vault / PROMOTE.RUNTIME_MODULE_NAME),
                "runtime_was_absent": True,
                "original_catalog_sha256": original_hash,
                "new_catalog_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(PROMOTE, "VAULT", vault)
    monkeypatch.setattr(PROMOTE, "CATALOG", catalog)
    monkeypatch.setattr(PROMOTE, "DESTINATION", vault / PROMOTE.DESTINATION_NAME)
    monkeypatch.setattr(
        PROMOTE, "ROOT_RUNTIME_MODULE", vault / PROMOTE.RUNTIME_MODULE_NAME
    )
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", journal)
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", lock)
    monkeypatch.setattr(PROMOTE, "process_is_alive", lambda pid: pid == owner_pid)

    with pytest.raises(RuntimeError, match="owner PID 12345 is alive"):
        PROMOTE.recover()

    assert journal.is_file()
    assert lock.is_file()


def test_recover_clears_a_dead_owner_lock_without_a_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / ".transaction.lock"
    lock.write_text(
        json.dumps({"transaction_id": "dead-tx", "owner_pid": 999999}),
        encoding="utf-8",
    )
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", lock)
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", tmp_path / ".journal.json")
    monkeypatch.setattr(PROMOTE, "process_is_alive", lambda _: False)

    PROMOTE.recover()

    assert not lock.exists()


def test_recover_fails_closed_when_changed_catalog_has_no_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    catalog = vault / "CATALOG.json"
    catalog.write_text('{"state": "original"}\n', encoding="utf-8")
    original_hash = PROMOTE.sha256_file(catalog)
    catalog.write_text('{"state": "new"}\n', encoding="utf-8")
    new_hash = PROMOTE.sha256_file(catalog)
    destination = vault / PROMOTE.DESTINATION_NAME
    destination.mkdir()
    (destination / "marker").write_text("keep", encoding="utf-8")
    runtime = vault / PROMOTE.RUNTIME_MODULE_NAME
    runtime.write_text("runtime", encoding="utf-8")
    journal = vault / ".transaction.json"
    lock = vault / ".transaction.lock"
    lock.write_text(
        json.dumps({"transaction_id": "tx", "owner_pid": 999999}),
        encoding="utf-8",
    )
    journal.write_text(
        json.dumps(
            {
                "transaction_id": "tx",
                "owner_pid": 999999,
                "destination": str(destination),
                "staging": str(vault / f".{PROMOTE.DESTINATION_NAME}.staging-safe"),
                "catalog_backup": str(vault / ".CATALOG.json.backup-missing"),
                "runtime_module": str(runtime),
                "runtime_was_absent": True,
                "original_catalog_sha256": original_hash,
                "new_catalog_sha256": new_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(PROMOTE, "VAULT", vault)
    monkeypatch.setattr(PROMOTE, "CATALOG", catalog)
    monkeypatch.setattr(PROMOTE, "DESTINATION", destination)
    monkeypatch.setattr(PROMOTE, "ROOT_RUNTIME_MODULE", runtime)
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", journal)
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", lock)
    monkeypatch.setattr(PROMOTE, "process_is_alive", lambda _: False)

    with pytest.raises(RuntimeError, match="backup is missing"):
        PROMOTE.recover()

    assert (destination / "marker").is_file()
    assert runtime.is_file()
    assert journal.is_file()
    assert lock.is_file()


def test_catalog_restore_refuses_an_unrelated_third_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "CATALOG.json"
    backup = tmp_path / ".CATALOG.json.backup-test"
    catalog.write_text('{"state": "original"}\n', encoding="utf-8")
    original_hash = PROMOTE.sha256_file(catalog)
    backup.write_bytes(catalog.read_bytes())
    catalog.write_text('{"state": "new"}\n', encoding="utf-8")
    new_hash = PROMOTE.sha256_file(catalog)
    catalog.write_text('{"state": "unrelated"}\n', encoding="utf-8")
    unrelated = catalog.read_text(encoding="utf-8")
    monkeypatch.setattr(PROMOTE, "CATALOG", catalog)

    with pytest.raises(RuntimeError, match="neither the recorded original"):
        PROMOTE.restore_catalog_or_fail(backup, original_hash, new_hash)

    assert catalog.read_text(encoding="utf-8") == unrelated
    assert backup.is_file()


def test_failed_rollback_preserves_journal_and_catalog_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    source = tmp_path / "source"
    vault.mkdir()
    source.mkdir()
    catalog = vault / "CATALOG.json"
    catalog.write_text('{"cores": [], "marker": "original"}\n', encoding="utf-8")
    (source / "modeling_abstain.py").write_text("runtime", encoding="utf-8")

    monkeypatch.setattr(PROMOTE, "VAULT", vault)
    monkeypatch.setattr(PROMOTE, "CATALOG", catalog)
    monkeypatch.setattr(PROMOTE, "SOURCE_ROOT", source)
    monkeypatch.setattr(PROMOTE, "DESTINATION", vault / "core")
    monkeypatch.setattr(PROMOTE, "ROOT_RUNTIME_MODULE", vault / "modeling_supa_a1.py")
    monkeypatch.setattr(PROMOTE, "TRANSACTION_JOURNAL", vault / ".transaction.json")
    monkeypatch.setattr(PROMOTE, "TRANSACTION_LOCK", vault / ".transaction.lock")
    monkeypatch.setattr(
        PROMOTE,
        "SOURCE_MODELING_SHA256",
        PROMOTE.sha256_file(source / "modeling_abstain.py"),
    )
    monkeypatch.setattr(PROMOTE, "validate_sources", dict)
    monkeypatch.setattr(PROMOTE, "verify_directory", lambda _: None)

    def fake_build(staging: Path, published_at: str) -> None:
        del published_at
        staging.mkdir()
        PROMOTE.atomic_json(staging / "CORE.json", {"core_id": PROMOTE.CORE_ID})

    monkeypatch.setattr(PROMOTE, "build_staging", fake_build)
    monkeypatch.setattr(
        PROMOTE,
        "catalog_with_supa_a1",
        lambda current, core: {"cores": [core], "previous": current},
    )
    monkeypatch.setattr(
        PROMOTE,
        "verify",
        lambda: (_ for _ in ()).throw(RuntimeError("injected verification failure")),
    )
    real_replace = PROMOTE.os.replace

    def fail_catalog_restore(source_path, destination_path):
        source_path = Path(source_path)
        if source_path.name.startswith(".CATALOG.json.backup-"):
            raise OSError("injected rollback failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(PROMOTE.os, "replace", fail_catalog_restore)

    with pytest.raises(RuntimeError, match="rollback failed"):
        PROMOTE.apply()

    journal = PROMOTE.load_json(vault / ".transaction.json")
    assert Path(journal["catalog_backup"]).is_file()
    assert (vault / "core").is_dir()
