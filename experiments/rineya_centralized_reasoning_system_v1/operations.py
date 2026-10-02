"""Durable CRS coordination. This module contains no model launcher.

Production callers must share one host database across worktrees. SQLite and
receipts provide cooperative coordination, not OS access isolation. Tests may use
temporary databases. A stale lease always requires explicit reconciliation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid


class OperationsError(RuntimeError):
    pass


ACTIVE = {"RESERVED", "RUNNING", "OUTCOME_UNKNOWN"}
TERMINAL = {"SUCCEEDED", "FAILED", "ABORTED", "SKIPPED"}
STATES = {"PENDING", "READY", "BLOCKED"} | ACTIVE | TERMINAL
EPOCH_SECONDS = 7200
ROLLING_SECONDS = 14400
WINDOW_SECONDS = 86400


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def host_state_path():
    """Return the proposed host-wide path without reading or creating it."""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise OperationsError("LOCALAPPDATA unavailable; explicit host path required")
    return Path(base) / "Rineya" / "CRS-001" / "host_state.sqlite3"


class OperationsStore:
    def __init__(self, db_path, receipt_dir, *, clock=time.time):
        self.db_path = Path(db_path).resolve()
        self.receipt_dir = Path(receipt_dir).resolve()
        self.clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.db = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA busy_timeout=5000")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise OperationsError("corrupt operations database; preserve and reconcile")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs(
                  key TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                  payload TEXT NOT NULL, epoch TEXT NOT NULL,
                  state TEXT NOT NULL, created REAL NOT NULL,
                  reserved REAL, reserved_seconds REAL NOT NULL DEFAULT 0,
                  actual_seconds REAL, finished REAL,
                  owner TEXT, fence INTEGER, worker TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS dependencies(
                  job TEXT NOT NULL REFERENCES jobs(key),
                  parent TEXT NOT NULL REFERENCES jobs(key),
                  PRIMARY KEY(job,parent));
                CREATE TABLE IF NOT EXISTS leases(
                  kind TEXT PRIMARY KEY, owner TEXT NOT NULL, fence INTEGER NOT NULL,
                  expires REAL NOT NULL, status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  time REAL NOT NULL, kind TEXT NOT NULL, key TEXT,
                  detail TEXT NOT NULL);
            """)
        except (sqlite3.DatabaseError, OperationsError) as exc:
            if hasattr(self, "db"):
                self.db.close()
            raise OperationsError("database unavailable/corrupt; no launch allowed") from exc

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _event(self, kind, key, detail, *, at=None):
        self.db.execute("INSERT INTO events(time,kind,key,detail) VALUES(?,?,?,?)",
                        (self.clock() if at is None else at, kind, key, canonical(detail).decode()))

    def _job(self, key):
        row = self.db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        if row is None:
            raise OperationsError("unregistered job")
        try:
            payload = json.loads(row["payload"])
            parents = [r[0] for r in self.db.execute("SELECT parent FROM dependencies WHERE job=? ORDER BY parent", (key,))]
            expected = digest({"payload": payload, "epoch": row["epoch"], "dependencies": parents})
            if expected != row["request_hash"] or canonical(payload).decode() != row["payload"]:
                raise OperationsError("job request/dependency integrity mismatch")
            if row["state"] not in STATES:
                raise OperationsError("unregistered job state")
            event = self.db.execute("SELECT kind,detail FROM events WHERE key=? AND kind IN "
                "('JOB_REGISTERED','JOB_READY','CPU_FINISHED','NATIVE_RESERVED','WORKER_OBSERVED',"
                "'OUTCOME_UNKNOWN','OUTCOME_RECONCILED','ORPHAN_WORKER') ORDER BY id DESC LIMIT 1", (key,)).fetchone()
            if event is None:
                raise OperationsError("job has no durable state event")
            states = {"JOB_REGISTERED": "PENDING", "JOB_READY": "READY", "NATIVE_RESERVED": "RESERVED",
                      "WORKER_OBSERVED": "RUNNING", "OUTCOME_UNKNOWN": "OUTCOME_UNKNOWN", "ORPHAN_WORKER": "OUTCOME_UNKNOWN"}
            if event[0] in {"CPU_FINISHED", "OUTCOME_RECONCILED"}:
                detail = json.loads(event[1]); expected_state = detail["outcome"]
                if row["result"] != canonical(detail).decode():
                    raise OperationsError("terminal outcome differs from durable event")
            else:
                expected_state = states[event[0]]
            if row["state"] != expected_state:
                raise OperationsError("job state differs from durable event")
            if row["state"] in ACTIVE and row["reserved"] is None:
                raise OperationsError("active job missing reservation")
            if row["state"] == "RUNNING" and row["worker"] is None:
                raise OperationsError("running job missing worker identity")
            origin = self.db.execute("SELECT kind,time,detail FROM events WHERE key=? "
                "AND kind IN ('JOB_REGISTERED','ORPHAN_WORKER') ORDER BY id ASC LIMIT 1", (key,)).fetchone()
            if origin is None or row["created"] != origin["time"]:
                raise OperationsError("job creation time differs from durable origin")
            reservation = self.db.execute("SELECT time,detail FROM events WHERE key=? "
                "AND kind='NATIVE_RESERVED' ORDER BY id ASC LIMIT 1", (key,)).fetchone()
            if reservation is not None:
                facts = json.loads(reservation["detail"])
                if ((row["reserved"], row["reserved_seconds"], row["owner"], row["fence"])
                        != (reservation["time"], facts["seconds"], facts["owner"], facts["fence"])):
                    raise OperationsError("reservation accounting differs from durable admission")
            elif origin["kind"] == "ORPHAN_WORKER":
                if (row["reserved"] != origin["time"] or row["reserved_seconds"] != ROLLING_SECONDS
                        or row["owner"] is not None or row["fence"] is not None):
                    raise OperationsError("orphan accounting differs from durable observation")
            elif (row["reserved"] is not None or row["reserved_seconds"] != 0
                    or row["owner"] is not None or row["fence"] is not None):
                raise OperationsError("unadmitted job has native accounting")
            worker_event = self.db.execute("SELECT detail FROM events WHERE key=? "
                "AND kind='WORKER_OBSERVED' ORDER BY id ASC LIMIT 1", (key,)).fetchone()
            expected_worker = (worker_event["detail"] if worker_event else
                               canonical(json.loads(origin["detail"])["identity"]).decode()
                               if origin["kind"] == "ORPHAN_WORKER" else None)
            if row["worker"] != expected_worker:
                raise OperationsError("worker identity differs from durable observation")
            if row["state"] in TERMINAL:
                terminal_event = self.db.execute("SELECT kind,time,detail FROM events WHERE key=? "
                    "AND kind IN ('CPU_FINISHED','OUTCOME_RECONCILED') ORDER BY id DESC LIMIT 1", (key,)).fetchone()
                details = json.loads(terminal_event["detail"])
                expected_actual = details["actual_native_seconds"] if terminal_event["kind"] == "OUTCOME_RECONCILED" else None
                if row["actual_seconds"] != expected_actual or row["finished"] != terminal_event["time"]:
                    raise OperationsError("actual cost/finish time differs from durable outcome")
            elif row["actual_seconds"] is not None or row["finished"] is not None:
                raise OperationsError("nonterminal work cannot have resolved cost/finish time")
        except (ValueError, KeyError, TypeError) as exc:
            raise OperationsError("job logical state corrupt") from exc
        return row

    def get_job(self, key):
        row = dict(self._job(key))
        row["payload"] = json.loads(row["payload"])
        for field in ("worker", "result"):
            row[field] = json.loads(row[field]) if row[field] else None
        return row

    def register_job(self, key, payload, epoch, dependencies=()):
        if not isinstance(key, str) or not key or not isinstance(epoch, str) or not epoch:
            raise OperationsError("nonempty durable job key and epoch required")
        if key in dependencies or len(set(dependencies)) != len(dependencies):
            raise OperationsError("invalid dependency list")
        request_hash = digest({"payload": payload, "epoch": epoch,
                               "dependencies": sorted(dependencies)})
        with self._transaction():
            old = self.db.execute("SELECT request_hash FROM jobs WHERE key=?", (key,)).fetchone()
            if old:
                if old[0] != request_hash:
                    raise OperationsError("idempotency key reused with different request")
                return self.get_job(key)
            for parent in dependencies:
                self._job(parent)
            created = self.clock()
            self.db.execute("INSERT INTO jobs(key,request_hash,payload,epoch,state,created) "
                            "VALUES(?,?,?,?,?,?)", (key, request_hash, canonical(payload).decode(),
                            epoch, "PENDING", created))
            self.db.executemany("INSERT INTO dependencies(job,parent) VALUES(?,?)",
                                [(key, parent) for parent in dependencies])
            self._event("JOB_REGISTERED", key, {"request_hash": request_hash}, at=created)
        return self.get_job(key)

    def ready(self, key):
        with self._transaction():
            row = self._job(key)
            if row["state"] == "READY":
                return self.get_job(key)
            if row["state"] not in {"PENDING", "BLOCKED"}:
                raise OperationsError("job cannot become ready from current state")
            failed = self.db.execute("SELECT j.key,j.state FROM dependencies d JOIN jobs j "
                                    "ON j.key=d.parent WHERE d.job=? AND j.state!='SUCCEEDED'",
                                    (key,)).fetchall()
            for parent in self.db.execute("SELECT parent FROM dependencies WHERE job=?", (key,)):
                self._job(parent[0])
            if failed:
                raise OperationsError("dependencies not successfully completed")
            self.db.execute("UPDATE jobs SET state='READY' WHERE key=?", (key,))
            self._event("JOB_READY", key, {})
        return self.get_job(key)

    @staticmethod
    def _seconds(value, *, maximum=None, zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise OperationsError("finite seconds required")
        if value < 0 or (not zero and value == 0) or (maximum is not None and value > maximum):
            raise OperationsError("seconds outside registered envelope")
        return float(value)

    def _require_lease(self, owner, fence, kind="model"):
        row = self.db.execute("SELECT * FROM leases WHERE kind=?", (kind,)).fetchone()
        if (row is None or row["owner"] != owner or row["fence"] != fence
                or row["status"] != "LIVE" or row["expires"] <= self.clock()):
            raise OperationsError("lease expired, mismatched or fenced")
        return row

    def acquire_lease(self, owner, seconds, *, kind="model"):
        seconds = self._seconds(seconds, maximum=600)
        if not isinstance(owner, str) or not owner or kind not in {"model", "coordinator"}:
            raise OperationsError("invalid lease identity")
        stale = False
        with self._transaction():
            row = self.db.execute("SELECT * FROM leases WHERE kind=?", (kind,)).fetchone()
            if row and row["status"] == "LIVE" and row["expires"] > self.clock():
                raise OperationsError("another live lease exists; use explicit heartbeat")
            if row and row["status"] != "RECONCILED":
                stale = True
                self.db.execute("UPDATE leases SET status='RECONCILIATION_REQUIRED' WHERE kind=?", (kind,))
                if kind == "model":
                    affected = self.db.execute("SELECT key FROM jobs WHERE state IN ('RESERVED','RUNNING')").fetchall()
                    for active in affected:
                        self._job(active[0])
                        self.db.execute("UPDATE jobs SET state='OUTCOME_UNKNOWN' WHERE key=?", (active[0],))
                        self._event("OUTCOME_UNKNOWN", active[0], {"reason": "stale model lease"})
                self._event("STALE_LEASE", kind, {"owner": row["owner"], "fence": row["fence"]})
            else:
                fence = 1 if row is None else row["fence"] + 1
                self.db.execute("INSERT OR REPLACE INTO leases VALUES(?,?,?,?,?)",
                                (kind, owner, fence, self.clock() + seconds, "LIVE"))
                self._event("LEASE_ACQUIRED", kind, {"owner": owner, "fence": fence})
        if stale:
            raise OperationsError("stale lease permits reconciliation only")
        return {"owner": owner, "fence": fence, "expires": self.clock() + seconds, "kind": kind}

    def heartbeat_lease(self, owner, fence, seconds, *, kind="model"):
        seconds = self._seconds(seconds, maximum=600)
        with self._transaction():
            self._require_lease(owner, fence, kind)
            self.db.execute("UPDATE leases SET expires=? WHERE kind=?", (self.clock() + seconds, kind))
            self._event("LEASE_HEARTBEAT", kind, {"owner": owner, "fence": fence})

    def release_lease(self, owner, fence, *, kind="model"):
        with self._transaction():
            self._require_lease(owner, fence, kind)
            if kind == "model" and self.db.execute("SELECT 1 FROM jobs WHERE state IN "
                    "('RESERVED','RUNNING','OUTCOME_UNKNOWN') LIMIT 1").fetchone():
                raise OperationsError("active/unknown worker prevents lease release")
            self.db.execute("UPDATE leases SET status='RECONCILED' WHERE kind=?", (kind,))
            self._event("LEASE_RELEASED", kind, {"owner": owner, "fence": fence})

    def reconcile_lease(self, owner, fence, evidence, *, processes_clear, kind="model"):
        """Explicit process/GPU reconciliation, never automatic stale takeover."""
        required = {"host_process_check", "gpu_activity_check", "worker_status"}
        if (processes_clear is not True or not isinstance(evidence, dict) or not required <= evidence.keys()
                or any(not isinstance(evidence[k], str) or not evidence[k].strip() for k in required)):
            raise OperationsError("complete current host/process reconciliation evidence required")
        with self._transaction():
            row = self.db.execute("SELECT * FROM leases WHERE kind=?", (kind,)).fetchone()
            if row is None or row["owner"] != owner or row["fence"] != fence:
                raise OperationsError("reconciliation lease identity mismatch")
            if row["status"] == "LIVE" and row["expires"] > self.clock():
                raise OperationsError("live lease must not be reconciled as stale")
            if kind == "model" and self.db.execute("SELECT 1 FROM jobs WHERE state IN "
                    "('RESERVED','RUNNING','OUTCOME_UNKNOWN') LIMIT 1").fetchone():
                raise OperationsError("reconcile every unknown job before lease takeover")
            self.db.execute("UPDATE leases SET status='RECONCILED' WHERE kind=?", (kind,))
            self._event("LEASE_RECONCILED", kind, evidence)

    def budget_snapshot(self, epoch):
        now = self.clock()
        rows = self.db.execute("SELECT * FROM jobs WHERE reserved IS NOT NULL").fetchall()
        for row in rows:
            self._job(row["key"])
        def charged(row):
            return row["reserved_seconds"] if row["actual_seconds"] is None else row["actual_seconds"]
        epoch_used = sum(charged(r) for r in rows if r["epoch"] == epoch)
        rolling_used = sum(charged(r) for r in rows if r["state"] in ACTIVE
                           or (r["finished"] is not None and r["finished"] >= now - WINDOW_SECONDS))
        return {"epoch": epoch, "epoch_charged_seconds": epoch_used,
                "rolling_charged_seconds": rolling_used,
                "epoch_limit_seconds": EPOCH_SECONDS, "rolling_limit_seconds": ROLLING_SECONDS,
                "unknown_reservations_charged_in_full": True}

    def reserve(self, key, owner, fence, native_seconds):
        native_seconds = self._seconds(native_seconds, maximum=EPOCH_SECONDS)
        with self._transaction():
            self._require_lease(owner, fence)
            row = self._job(key)
            if row["state"] in ACTIVE or row["state"] in TERMINAL:
                if (row["owner"], row["fence"], row["reserved_seconds"]) != (owner, fence, native_seconds):
                    raise OperationsError("reservation retry identity mismatch")
                return self.get_job(key)
            if row["state"] != "READY":
                raise OperationsError("job is not ready")
            if self.db.execute("SELECT 1 FROM jobs WHERE state IN "
                    "('RESERVED','RUNNING','OUTCOME_UNKNOWN') LIMIT 1").fetchone():
                raise OperationsError("one native worker; outstanding job requires reconciliation")
            budget = self.budget_snapshot(row["epoch"])
            if (budget["epoch_charged_seconds"] + native_seconds > EPOCH_SECONDS
                    or budget["rolling_charged_seconds"] + native_seconds > ROLLING_SECONDS):
                raise OperationsError("native time budget exhausted")
            reserved_at = self.clock()
            self.db.execute("UPDATE jobs SET state='RESERVED',reserved=?,reserved_seconds=?,"
                            "owner=?,fence=? WHERE key=?", (reserved_at, native_seconds, owner, fence, key))
            self._event("NATIVE_RESERVED", key, {"seconds": native_seconds, "owner": owner, "fence": fence}, at=reserved_at)
        return self.get_job(key)

    def mark_running(self, key, owner, fence, worker_identity):
        if not isinstance(worker_identity, dict) or not {"pid", "creation_time", "command_hash"} <= worker_identity.keys():
            raise OperationsError("PID plus process creation time and command identity required")
        if (isinstance(worker_identity["pid"], bool) or not isinstance(worker_identity["pid"], int)
                or worker_identity["pid"] <= 0 or not isinstance(worker_identity["creation_time"], str)
                or not worker_identity["creation_time"] or not isinstance(worker_identity["command_hash"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", worker_identity["command_hash"])):
            raise OperationsError("invalid worker identity values")
        with self._transaction():
            self._require_lease(owner, fence)
            row = self._job(key)
            if row["state"] == "RUNNING" and json.loads(row["worker"]) == worker_identity:
                return self.get_job(key)
            if row["state"] != "RESERVED" or (row["owner"], row["fence"]) != (owner, fence):
                raise OperationsError("unknown/nonowned outcome must be reconciled before dispatch")
            self.db.execute("UPDATE jobs SET state='RUNNING',worker=? WHERE key=?",
                            (canonical(worker_identity).decode(), key))
            self._event("WORKER_OBSERVED", key, worker_identity)
        return self.get_job(key)

    def mark_unknown(self, key, reason):
        with self._transaction():
            row = self._job(key)
            if row["state"] not in ACTIVE:
                raise OperationsError("only an admitted job can have unknown native outcome")
            self.db.execute("UPDATE jobs SET state='OUTCOME_UNKNOWN' WHERE key=?", (key,))
            self._event("OUTCOME_UNKNOWN", key, {"reason": str(reason)})

    def reconcile(self, key, outcome, actual_native_seconds, evidence, *, owner=None, fence=None,
                  stale_reconciliation=False, coordinator_owner=None, coordinator_fence=None):
        """Resolve admitted work using identified evidence; never retries a job.

        Unknown costs remain the full reservation when actual_native_seconds=None.
        A proved no-dispatch outcome may use zero with a failed/aborted outcome.
        Actual overruns are preserved and reported rather than rejected or erased.
        """
        if outcome not in TERMINAL or not isinstance(evidence, dict) or not evidence:
            raise OperationsError("terminal outcome and nonempty reconciliation evidence required")
        actual = None if actual_native_seconds is None else self._seconds(actual_native_seconds, zero=True)
        proof_ref = evidence.get("native_outcome_proof_ref")
        if not isinstance(proof_ref, dict):
            raise OperationsError("identified process-exit/no-dispatch observation receipt required")
        proof_receipt = self.verify_receipt(proof_ref)
        if proof_receipt.get("kind") != "native_outcome_observation" or proof_receipt.get("key") != key:
            raise OperationsError("outcome observation receipt identity mismatch")
        proof = proof_receipt["payload"]
        if not isinstance(proof, dict) or not isinstance(proof.get("observer"), str) or not proof["observer"].strip():
            raise OperationsError("identified outcome observer required")
        with self._transaction():
            row = self._job(key)
            lease = self.db.execute("SELECT * FROM leases WHERE kind='model'").fetchone()
            if lease is None or (lease["owner"], lease["fence"]) != (owner, fence):
                raise OperationsError("outcome reconciliation owner/fence mismatch")
            if stale_reconciliation:
                if lease["status"] != "RECONCILIATION_REQUIRED" or lease["expires"] > self.clock():
                    raise OperationsError("explicit stale reconciliation requires expired marked lease")
                self._require_lease(coordinator_owner, coordinator_fence, "coordinator")
                if proof.get("reconciliation_owner") != coordinator_owner:
                    raise OperationsError("stale reconciliation owner witness mismatch")
            else:
                self._require_lease(owner, fence)
            if row["owner"] is not None and (row["owner"], row["fence"]) != (owner, fence):
                raise OperationsError("job owner/fence mismatch")
            if proof.get("request_hash") != row["request_hash"] or (proof.get("owner"), proof.get("fence")) != (owner, fence):
                raise OperationsError("outcome observation request/lease binding mismatch")
            result = {"evidence": evidence, "outcome": outcome,
                      "actual_native_seconds": actual,
                      "reservation_exceeded": actual is not None and actual > row["reserved_seconds"]}
            if row["state"] in TERMINAL:
                if row["result"] != canonical(result).decode():
                    raise OperationsError("terminal outcome immutable; conflicting reconciliation")
                return self.get_job(key)
            if row["worker"] is not None:
                if actual == 0 or proof.get("never_dispatched") or proof.get("process_exited") is not True:
                    raise OperationsError("observed worker cannot erase cost or omit verified exit")
                if proof.get("worker_identity") != json.loads(row["worker"]):
                    raise OperationsError("process exit identity mismatch")
                observed_at = proof.get("exit_observed_at")
                if (type(observed_at) not in (int, float) or not math.isfinite(observed_at)
                        or not row["reserved"] <= observed_at <= self.clock()):
                    raise OperationsError("current measured process exit time required")
            else:
                if row["state"] != "RESERVED" or proof.get("never_dispatched") is not True or proof.get("launch_attempted") is not False:
                    raise OperationsError("unobserved dispatch outcome needs explicit never-dispatched proof")
                if actual not in (0, None):
                    raise OperationsError("never-dispatched job cannot have measured native work")
            if proof.get("elapsed_native_seconds") != actual:
                raise OperationsError("measured native cost differs from exit observation")
            if row["state"] not in ACTIVE:
                raise OperationsError("job has no admitted native work to reconcile")
            if outcome == "SUCCEEDED" and actual is None:
                raise OperationsError("success requires measured native cost")
            finished_at = self.clock()
            self.db.execute("UPDATE jobs SET state=?,actual_seconds=?,finished=?,result=? WHERE key=?",
                            (outcome, actual, finished_at, canonical(result).decode(), key))
            self._event("OUTCOME_RECONCILED", key, result, at=finished_at)
        return self.get_job(key)

    def finish_cpu(self, key, outcome, evidence):
        """Record non-native DAG nodes without creating native budget exposure."""
        if outcome not in TERMINAL or not isinstance(evidence, dict) or not evidence:
            raise OperationsError("CPU outcome requires identified evidence")
        result = {"evidence": evidence, "native_seconds": 0, "outcome": outcome}
        with self._transaction():
            row = self._job(key)
            if row["state"] in TERMINAL and row["reserved"] is None and row["result"] == canonical(result).decode():
                return self.get_job(key)
            if row["reserved"] is not None or row["state"] not in {"READY", "PENDING", "BLOCKED"}:
                raise OperationsError("CPU completion cannot erase native reservations or outcomes")
            if outcome == "SUCCEEDED" and row["state"] != "READY":
                raise OperationsError("CPU success requires dependencies and READY state")
            finished_at = self.clock()
            self.db.execute("UPDATE jobs SET state=?,finished=?,result=? WHERE key=?",
                            (outcome, finished_at, canonical(result).decode(), key))
            self._event("CPU_FINISHED", key, result, at=finished_at)
        return self.get_job(key)

    def write_receipt(self, kind, key, payload):
        if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", kind):
            raise OperationsError("invalid receipt kind")
        body = {"schema_version": 1, "kind": kind, "key": key, "payload": payload}
        raw = canonical(body)
        filename = kind + "-" + hashlib.sha256(str(key).encode()).hexdigest() + ".json"
        destination = self.receipt_dir / filename
        if destination.exists():
            if destination.read_bytes() != raw:
                raise OperationsError("receipt conflict/corruption; existing bytes preserved")
            return {"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest()}
        temporary = self.receipt_dir / (".partial-" + uuid.uuid4().hex)
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Same-volume hard-link publication is atomic and refuses overwrite.
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != raw:
                raise OperationsError("concurrent receipt conflict; partial bytes retained")
        except OSError as exc:
            raise OperationsError("receipt publication failed; partial bytes retained") from exc
        temporary.unlink()
        return {"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest()}

    def verify_receipt(self, reference):
        try:
            path = Path(reference["path"]).resolve()
            if path.parent != self.receipt_dir or path.suffix != ".json":
                raise OperationsError("receipt path escape")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                raise OperationsError("receipt hash mismatch")
            value = json.loads(raw)
            if canonical(value) != raw:
                raise OperationsError("receipt noncanonical/corrupt")
            if (type(value) is not dict or set(value) != {"schema_version", "kind", "key", "payload"}
                    or value["schema_version"] != 1 or not isinstance(value["kind"], str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["kind"])):
                raise OperationsError("receipt schema mismatch")
            filename = value["kind"] + "-" + hashlib.sha256(str(value["key"]).encode()).hexdigest() + ".json"
            if path.name != filename:
                raise OperationsError("receipt filename/identity mismatch")
            return value
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
            raise OperationsError("receipt unavailable/corrupt; no result inferred") from exc

    def orphan_worker(self, worker_identity, reason):
        """Journal an observed unregistered worker; block future admission."""
        key = "orphan-" + digest(worker_identity)
        with self._transaction():
            existing = self.db.execute("SELECT key FROM jobs WHERE key=?", (key,)).fetchone()
            if not existing:
                observed_at = self.clock()
                self.db.execute("INSERT INTO jobs(key,request_hash,payload,epoch,state,created,"
                                "reserved,reserved_seconds,worker) VALUES(?,?,?,?,?,?,?,?,?)",
                                (key, digest({"payload": worker_identity, "epoch": "ORPHAN_UNKNOWN", "dependencies": []}), canonical(worker_identity).decode(),
                                 "ORPHAN_UNKNOWN", "OUTCOME_UNKNOWN", observed_at, observed_at,
                                 ROLLING_SECONDS, canonical(worker_identity).decode()))
                self._event("ORPHAN_WORKER", key, {"reason": reason, "identity": worker_identity}, at=observed_at)
        return self.get_job(key)
