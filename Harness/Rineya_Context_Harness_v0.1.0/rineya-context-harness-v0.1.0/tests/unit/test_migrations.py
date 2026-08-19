from __future__ import annotations

import sqlite3
from pathlib import Path

from rineya_harness.config import DatabaseConfig, MemoryPolicyConfig
from rineya_harness.state.store import SQLiteStateStore


def test_migration_runner_upgrades_legacy_v0001_database(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript((root / "migrations" / "0001_initial.sql").read_text())
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            ("0001", "2026-08-17T00:00:00+00:00"),
        )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteStateStore(
        DatabaseConfig(path=database),
        MemoryPolicyConfig(),
        migration_directory=root / "migrations",
    )
    store.initialize()

    connection = sqlite3.connect(database)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(idempotency_records)")
        }
        versions = {
            row[0] for row in connection.execute("SELECT version FROM schema_migrations")
        }
    finally:
        connection.close()
    assert "request_hash" in columns
    assert versions == {"0001", "0002"}
