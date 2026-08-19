from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from rineya_harness.clock import Clock, SystemClock
from rineya_harness.config import DatabaseConfig, MemoryPolicyConfig
from rineya_harness.domain.enums import MemoryStatus, RunStatus, SupportStatus, WorkflowNode
from rineya_harness.domain.models import (
    CheckpointRecord,
    ClaimRecord,
    EvidenceRecord,
    MemoryItem,
    PatchResult,
    RunRecord,
    RunSnapshot,
    StateEvent,
    StatePatch,
)
from rineya_harness.errors import (
    ConflictError,
    NotFoundError,
    RevisionConflictError,
    StateInvariantError,
)
from rineya_harness.ids import new_id

from .invariants import StateInvariantValidator
from .reducer import project_mutation
from .serialization import canonical_json, sha256_json


class SQLiteStateStore:
    """Local-first materialized state plus append-only transition events."""

    def __init__(
        self,
        settings: DatabaseConfig,
        memory_policy: MemoryPolicyConfig,
        *,
        migration_directory: Path | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.path = Path(settings.path)
        self.migration_directory = migration_directory or (
            Path(__file__).resolve().parents[1] / "migrations"
        )
        self.clock = clock or SystemClock()
        self.validator = StateInvariantValidator(memory_policy)
        self._lock = threading.RLock()
        self._initialized = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=self.settings.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.settings.busy_timeout_ms}")
        if self.settings.wal:
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(f"PRAGMA synchronous = {self.settings.synchronous}")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            migration_files = sorted(self.migration_directory.glob("[0-9][0-9][0-9][0-9]_*.sql"))
            if not migration_files:
                raise RuntimeError(f"No migrations found in: {self.migration_directory}")
            with self._connect() as connection:
                try:
                    rows = connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                    applied = {str(row["version"]) for row in rows}
                except sqlite3.OperationalError:
                    applied = set()
                for migration in migration_files:
                    version = migration.stem.split("_", 1)[0]
                    if version in applied:
                        continue
                    sql = migration.read_text(encoding="utf-8")
                    try:
                        connection.executescript("BEGIN IMMEDIATE;\n" + sql)
                        connection.execute(
                            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                            (version, self.clock.now().isoformat()),
                        )
                        connection.execute("COMMIT")
                    except Exception:
                        if connection.in_transaction:
                            connection.execute("ROLLBACK")
                        raise
                    applied.add(version)
            self._initialized = True

    def create_run(self, run: RunRecord) -> RunRecord:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO runs(
                      id, goal, mode, status, revision, current_node, config_json,
                      metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.goal,
                        run.mode.value,
                        run.status.value,
                        run.revision,
                        run.current_node.value,
                        canonical_json(run.config_snapshot),
                        canonical_json(run.metadata),
                        run.created_at.isoformat(),
                        run.updated_at.isoformat(),
                    ),
                )
                self._insert_event(
                    connection,
                    StateEvent(
                        run_id=run.id,
                        revision=run.revision,
                        event_type="run_created",
                        payload={"run": run.model_dump(mode="json")},
                    ),
                )
                connection.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                connection.execute("ROLLBACK")
                raise ConflictError(f"Run already exists: {run.id}") from exc
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return run

    def get_run(self, run_id: str) -> RunRecord:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("run", run_id)
        return self._run_from_row(row)

    def list_runs(self, *, limit: int = 100) -> list[RunRecord]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        current_node: WorkflowNode | None = None,
        metadata_patch: dict[str, Any] | None = None,
        event_type: str = "run_updated",
        event_payload: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> RunRecord:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
                if row is None:
                    raise NotFoundError("run", run_id)
                run = self._run_from_row(row)
                if status is not None:
                    run.status = status
                if current_node is not None:
                    run.current_node = current_node
                if metadata_patch:
                    run.metadata = {**run.metadata, **metadata_patch}
                run.updated_at = self.clock.now()
                connection.execute(
                    """
                    UPDATE runs SET status = ?, current_node = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        run.status.value,
                        run.current_node.value,
                        canonical_json(run.metadata),
                        run.updated_at.isoformat(),
                        run.id,
                    ),
                )
                self._insert_event(
                    connection,
                    StateEvent(
                        run_id=run.id,
                        revision=run.revision,
                        event_type=event_type,
                        payload=event_payload or {},
                        trace_id=trace_id,
                    ),
                )
                connection.execute("COMMIT")
                return run
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list_memory(self, run_id: str, *, include_inactive: bool = False) -> list[MemoryItem]:
        self.initialize()
        sql = "SELECT record_json FROM memory_items WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if not include_inactive:
            sql += " AND status = ?"
            parameters.append(MemoryStatus.ACTIVE.value)
        sql += " ORDER BY revision_updated, id"
        with self._connect() as connection:
            self._ensure_run(connection, run_id)
            rows = connection.execute(sql, parameters).fetchall()
        return [MemoryItem.model_validate_json(row["record_json"]) for row in rows]

    def get_memory(self, run_id: str, memory_id: str) -> MemoryItem:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM memory_items WHERE run_id = ? AND id = ?",
                (run_id, memory_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("memory", memory_id)
        return MemoryItem.model_validate_json(row["record_json"])

    def apply_patch(
        self,
        patch: StatePatch,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> PatchResult:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                request_hash = self._patch_request_hash(patch)
                cached = connection.execute(
                    """
                    SELECT request_hash, result_json FROM idempotency_records
                    WHERE run_id = ? AND idempotency_key = ?
                    """,
                    (patch.run_id, patch.idempotency_key),
                ).fetchone()
                if cached is not None:
                    if cached["request_hash"] == "legacy-unverifiable":
                        raise ConflictError(
                            "Legacy idempotency record cannot be safely replayed; use a new key: "
                            f"{patch.idempotency_key}"
                        )
                    if cached["request_hash"] != request_hash:
                        raise ConflictError(
                            "Idempotency key was already used for a different state patch: "
                            f"{patch.idempotency_key}"
                        )
                    result = PatchResult.model_validate_json(cached["result_json"])
                    result.idempotent_replay = True
                    connection.execute("COMMIT")
                    return result

                run_row = connection.execute("SELECT * FROM runs WHERE id = ?", (patch.run_id,)).fetchone()
                if run_row is None:
                    raise NotFoundError("run", patch.run_id)
                run = self._run_from_row(run_row)
                if run.revision != patch.expected_revision:
                    raise RevisionConflictError(patch.run_id, patch.expected_revision, run.revision)

                before = self._load_memory_map(connection, patch.run_id)
                after = {key: value.model_copy(deep=True) for key, value in before.items()}
                next_revision = run.revision + 1
                now = self.clock.now()
                changed: list[str] = []
                for mutation in patch.mutations:
                    changed.append(
                        project_mutation(
                            after,
                            mutation,
                            run_id=patch.run_id,
                            revision=next_revision,
                            now=now,
                        )
                    )

                evidence_rows = connection.execute(
                    "SELECT id FROM evidence_records WHERE run_id = ?",
                    (patch.run_id,),
                ).fetchall()
                known_evidence_ids = {str(row["id"]) for row in evidence_rows}
                reasons = self.validator.validate(
                    before=before,
                    after=after,
                    patch=patch,
                    known_evidence_ids=known_evidence_ids,
                )
                if reasons:
                    raise StateInvariantError(reasons)

                self._replace_memory_state(connection, patch.run_id, after)
                connection.execute(
                    "UPDATE runs SET revision = ?, updated_at = ? WHERE id = ?",
                    (next_revision, now.isoformat(), patch.run_id),
                )
                removed = [
                    memory_id
                    for memory_id in changed
                    if memory_id in before
                    and before[memory_id].status is MemoryStatus.ACTIVE
                    and after[memory_id].status is not MemoryStatus.ACTIVE
                ]
                archived = [
                    memory_id
                    for memory_id in changed
                    if after[memory_id].status is MemoryStatus.ARCHIVED
                ]
                purged = [
                    memory_id
                    for memory_id in changed
                    if after[memory_id].status is MemoryStatus.PURGED
                ]
                state_hash = self._state_hash(after)
                event_id = new_id("evt")
                result = PatchResult(
                    patch_id=patch.id,
                    run_id=patch.run_id,
                    previous_revision=run.revision,
                    new_revision=next_revision,
                    committed=True,
                    changed_memory_ids=list(dict.fromkeys(changed)),
                    removed_from_active_context=list(dict.fromkeys(removed)),
                    archived_memory_ids=list(dict.fromkeys(archived)),
                    purged_memory_ids=list(dict.fromkeys(purged)),
                    prompt_rebuild_required=bool(changed),
                    state_hash=state_hash,
                    event_id=event_id,
                )
                self._insert_event(
                    connection,
                    StateEvent(
                        id=event_id,
                        run_id=patch.run_id,
                        revision=next_revision,
                        event_type="state_patch_committed",
                        payload={
                            "patch": patch.model_dump(mode="json"),
                            "result": result.model_dump(mode="json"),
                        },
                        trace_id=trace_id,
                        span_id=span_id,
                        created_at=now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO idempotency_records(
                      run_id, idempotency_key, patch_id, request_hash, result_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patch.run_id,
                        patch.idempotency_key,
                        patch.id,
                        request_hash,
                        result.model_dump_json(),
                        now.isoformat(),
                    ),
                )
                connection.execute("COMMIT")
                return result
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def add_evidence(self, records: Iterable[EvidenceRecord]) -> None:
        self.initialize()
        records = list(records)
        if not records:
            return
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for record in records:
                    self._ensure_run(connection, record.run_id)
                    self._insert_immutable_record(
                        connection,
                        table="evidence_records",
                        run_id=record.run_id,
                        record_id=record.id,
                        record_json=record.model_dump_json(),
                        resource_type="evidence",
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list_evidence(self, run_id: str) -> list[EvidenceRecord]:
        self.initialize()
        with self._connect() as connection:
            self._ensure_run(connection, run_id)
            rows = connection.execute(
                "SELECT record_json FROM evidence_records WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [EvidenceRecord.model_validate_json(row["record_json"]) for row in rows]

    def add_claims(self, records: Iterable[ClaimRecord]) -> None:
        self.initialize()
        records = list(records)
        if not records:
            return
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for record in records:
                    self._ensure_run(connection, record.run_id)
                    evidence_rows = connection.execute(
                        "SELECT id FROM evidence_records WHERE run_id = ?",
                        (record.run_id,),
                    ).fetchall()
                    known_evidence_ids = {str(row["id"]) for row in evidence_rows}
                    unknown = record.evidence_ids - known_evidence_ids
                    if unknown:
                        raise ConflictError(
                            f"Claim {record.id} cites unknown evidence IDs: {sorted(unknown)}"
                        )
                    if (
                        record.support_status is SupportStatus.SUPPORTED
                        and not record.evidence_ids
                    ):
                        raise ConflictError(
                            f"Supported claim requires evidence IDs: {record.id}"
                        )
                    self._insert_immutable_record(
                        connection,
                        table="claim_records",
                        run_id=record.run_id,
                        record_id=record.id,
                        record_json=record.model_dump_json(),
                        resource_type="claim",
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list_claims(self, run_id: str) -> list[ClaimRecord]:
        self.initialize()
        with self._connect() as connection:
            self._ensure_run(connection, run_id)
            rows = connection.execute(
                "SELECT record_json FROM claim_records WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [ClaimRecord.model_validate_json(row["record_json"]) for row in rows]

    def get_snapshot(self, run_id: str) -> RunSnapshot:
        return RunSnapshot(
            run=self.get_run(run_id),
            memories=self.list_memory(run_id, include_inactive=True),
            evidence=self.list_evidence(run_id),
            claims=self.list_claims(run_id),
        )

    def create_checkpoint(
        self,
        run_id: str,
        *,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._run_from_row(self._required_run_row(connection, run_id))
                snapshot = self._snapshot_payload(connection, run_id)
                checkpoint = CheckpointRecord(
                    run_id=run_id,
                    source_revision=run.revision,
                    label=label,
                    state_hash=sha256_json(snapshot),
                    metadata=metadata or {},
                    created_at=self.clock.now(),
                )
                connection.execute(
                    """
                    INSERT INTO checkpoints(
                      id, run_id, source_revision, label, state_hash,
                      snapshot_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.id,
                        checkpoint.run_id,
                        checkpoint.source_revision,
                        checkpoint.label,
                        checkpoint.state_hash,
                        canonical_json(snapshot),
                        canonical_json(checkpoint.metadata),
                        checkpoint.created_at.isoformat(),
                    ),
                )
                self._insert_event(
                    connection,
                    StateEvent(
                        run_id=run_id,
                        revision=run.revision,
                        event_type="checkpoint_created",
                        payload={"checkpoint": checkpoint.model_dump(mode="json")},
                    ),
                )
                connection.execute("COMMIT")
                return checkpoint
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list_checkpoints(self, run_id: str) -> list[CheckpointRecord]:
        self.initialize()
        with self._connect() as connection:
            self._ensure_run(connection, run_id)
            rows = connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY created_at DESC", (run_id,)
            ).fetchall()
        return [self._checkpoint_from_row(row) for row in rows]

    def restore_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        expected_revision: int,
        trace_id: str | None = None,
    ) -> PatchResult:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._run_from_row(self._required_run_row(connection, run_id))
                if run.revision != expected_revision:
                    raise RevisionConflictError(run_id, expected_revision, run.revision)
                row = connection.execute(
                    "SELECT * FROM checkpoints WHERE run_id = ? AND id = ?",
                    (run_id, checkpoint_id),
                ).fetchone()
                if row is None:
                    raise NotFoundError("checkpoint", checkpoint_id)
                snapshot = json.loads(row["snapshot_json"])
                memories = [MemoryItem.model_validate(item) for item in snapshot["memories"]]
                next_revision = run.revision + 1
                now = self.clock.now()
                restored: dict[str, MemoryItem] = {}
                for item in memories:
                    item.revision_updated = next_revision
                    item.updated_at = now
                    restored[item.id] = item
                # Validate the restored materialized state with a synthetic no-op-like patch omitted;
                # checkpoint snapshots were already validated when committed.
                self._replace_memory_state(connection, run_id, restored)
                connection.execute("DELETE FROM evidence_records WHERE run_id = ?", (run_id,))
                connection.execute("DELETE FROM claim_records WHERE run_id = ?", (run_id,))
                for record in snapshot["evidence"]:
                    evidence = EvidenceRecord.model_validate(record)
                    connection.execute(
                        "INSERT INTO evidence_records(run_id, id, record_json) VALUES (?, ?, ?)",
                        (run_id, evidence.id, evidence.model_dump_json()),
                    )
                for record in snapshot["claims"]:
                    claim = ClaimRecord.model_validate(record)
                    connection.execute(
                        "INSERT INTO claim_records(run_id, id, record_json) VALUES (?, ?, ?)",
                        (run_id, claim.id, claim.model_dump_json()),
                    )
                connection.execute(
                    "UPDATE runs SET revision = ?, updated_at = ? WHERE id = ?",
                    (next_revision, now.isoformat(), run_id),
                )
                state_hash = self._state_hash(restored)
                event_id = new_id("evt")
                result = PatchResult(
                    patch_id=f"restore_{checkpoint_id}",
                    run_id=run_id,
                    previous_revision=run.revision,
                    new_revision=next_revision,
                    committed=True,
                    changed_memory_ids=sorted(restored),
                    prompt_rebuild_required=True,
                    state_hash=state_hash,
                    event_id=event_id,
                )
                self._insert_event(
                    connection,
                    StateEvent(
                        id=event_id,
                        run_id=run_id,
                        revision=next_revision,
                        event_type="checkpoint_restored",
                        payload={
                            "checkpoint_id": checkpoint_id,
                            "source_revision": int(row["source_revision"]),
                            "result": result.model_dump(mode="json"),
                        },
                        trace_id=trace_id,
                        created_at=now,
                    ),
                )
                connection.execute("COMMIT")
                return result
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list_events(self, run_id: str, *, limit: int = 1_000) -> list[StateEvent]:
        self.initialize()
        with self._connect() as connection:
            self._ensure_run(connection, run_id)
            rows = connection.execute(
                """
                SELECT * FROM state_events WHERE run_id = ?
                ORDER BY created_at, id LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def export_run(self, run_id: str) -> dict[str, Any]:
        snapshot = self.get_snapshot(run_id)
        checkpoints = self.list_checkpoints(run_id)
        events = self.list_events(run_id)
        payload = {
            "snapshot": snapshot.model_dump(mode="json"),
            "checkpoints": [item.model_dump(mode="json") for item in checkpoints],
            "events": [item.model_dump(mode="json") for item in events],
        }
        payload["export_sha256"] = sha256_json(payload)
        return payload

    def current_state_hash(self, run_id: str) -> str:
        items = {item.id: item for item in self.list_memory(run_id, include_inactive=True)}
        return self._state_hash(items)

    @staticmethod
    def _patch_request_hash(patch: StatePatch) -> str:
        semantic_request = patch.model_dump(
            mode="json",
            exclude={"id", "created_at"},
        )
        return sha256_json(semantic_request)

    def _snapshot_payload(self, connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
        memories = self._load_memory_map(connection, run_id)
        evidence_rows = connection.execute(
            "SELECT record_json FROM evidence_records WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        claim_rows = connection.execute(
            "SELECT record_json FROM claim_records WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return {
            "memories": [
                item.model_dump(mode="json") for item in sorted(memories.values(), key=lambda x: x.id)
            ],
            "evidence": [json.loads(row["record_json"]) for row in evidence_rows],
            "claims": [json.loads(row["record_json"]) for row in claim_rows],
        }

    @staticmethod
    def _state_hash(items: dict[str, MemoryItem]) -> str:
        payload = [
            item.model_dump(mode="json") for item in sorted(items.values(), key=lambda value: value.id)
        ]
        return sha256_json(payload)

    @staticmethod
    def _replace_memory_state(
        connection: sqlite3.Connection,
        run_id: str,
        items: dict[str, MemoryItem],
    ) -> None:
        connection.execute("DELETE FROM memory_items WHERE run_id = ?", (run_id,))
        connection.executemany(
            """
            INSERT INTO memory_items(run_id, id, record_json, status, pinned, revision_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.id,
                    item.model_dump_json(),
                    item.status.value,
                    int(item.pinned),
                    item.revision_updated,
                )
                for item in sorted(items.values(), key=lambda value: value.id)
            ],
        )

    @staticmethod
    def _load_memory_map(connection: sqlite3.Connection, run_id: str) -> dict[str, MemoryItem]:
        rows = connection.execute(
            "SELECT id, record_json FROM memory_items WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {row["id"]: MemoryItem.model_validate_json(row["record_json"]) for row in rows}

    @staticmethod
    def _insert_immutable_record(
        connection: sqlite3.Connection,
        *,
        table: str,
        run_id: str,
        record_id: str,
        record_json: str,
        resource_type: str,
    ) -> None:
        if table not in {"evidence_records", "claim_records"}:
            raise ValueError(f"Unsupported immutable table: {table}")
        existing = connection.execute(
            f"SELECT record_json FROM {table} WHERE run_id = ? AND id = ?",
            (run_id, record_id),
        ).fetchone()
        if existing is not None:
            if existing["record_json"] != record_json:
                raise ConflictError(
                    f"Immutable {resource_type} record already exists with different content: "
                    f"{record_id}"
                )
            return
        connection.execute(
            f"INSERT INTO {table}(run_id, id, record_json) VALUES (?, ?, ?)",
            (run_id, record_id, record_json),
        )

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, event: StateEvent) -> None:
        connection.execute(
            """
            INSERT INTO state_events(
              id, run_id, revision, event_type, payload_json,
              trace_id, span_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.run_id,
                event.revision,
                event.event_type,
                canonical_json(event.payload),
                event.trace_id,
                event.span_id,
                event.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _ensure_run(connection: sqlite3.Connection, run_id: str) -> None:
        if connection.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
            raise NotFoundError("run", run_id)

    @staticmethod
    def _required_run_row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("run", run_id)
        return row

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        from rineya_harness.domain.enums import RunMode

        return RunRecord(
            id=row["id"],
            goal=row["goal"],
            mode=RunMode(row["mode"]),
            status=RunStatus(row["status"]),
            revision=int(row["revision"]),
            current_node=WorkflowNode(row["current_node"]),
            config_snapshot=json.loads(row["config_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _checkpoint_from_row(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            id=row["id"],
            run_id=row["run_id"],
            source_revision=int(row["source_revision"]),
            label=row["label"],
            state_hash=row["state_hash"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> StateEvent:
        return StateEvent(
            id=row["id"],
            run_id=row["run_id"],
            revision=int(row["revision"]),
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            created_at=row["created_at"],
        )

    @staticmethod
    def memory_map(items: Iterable[MemoryItem]) -> dict[str, MemoryItem]:
        return {item.id: item for item in items}
