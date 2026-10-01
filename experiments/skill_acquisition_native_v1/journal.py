"""Durable physical-cost and logical-progress journal; no framework imports.

This is an accounting adapter after production admission, not an admission
substitute. Every accepted start is fsynced before its caller may work. Missing
returns retain their conservative reservations. Corrupt/partial logs fail closed.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from . import plan
from .checkpoint import validate_progress

SCHEMA = "SKILL-ACQUISITION-DURABLE-LEDGER-1"
MAX_LINE = 2 * 1024 * 1024
IO_FIELDS = ("identity_hash_attempts", "identity_hash_bytes", "chunk_io_read_bytes",
             "chunk_io_write_bytes", "checkpoint_restore_attempts", "model_load_attempts")
RESTORE_BOOTSTRAP = frozenset(("module_import", "input_io", "tokenizer_load", "model_load",
    "inventory_configuration", "process_rng_seed", "fingerprint", "canary", "dataset_read", "checkpoint_construct",
    "checkpoint_adapter_io", "checkpoint_restore", "cleanup", "wall_watchdog", "receipt_write"))


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hex64(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate journal JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite journal JSON")))


def _exclusive(path, value):
    raw = canonical(value) + b"\n"
    with Path(path).open("xb") as stream:
        require(stream.write(raw) == len(raw), "short ledger header write")
        stream.flush()
        os.fsync(stream.fileno())


def bootstrap_charges(observations):
    require(type(observations) is list and len(observations)<=20000 and len(canonical(observations))<=1024**2,
            "bounded bootstrap observations")
    pending, finished = {}, set()
    charges={"chunk_io_read_bytes":0,"chunk_io_write_bytes":0,"identity_hash_attempts":0,"identity_hash_bytes":0}
    mapping={"chunk_io_read_bytes":"read_bytes_upper","chunk_io_write_bytes":"write_bytes_upper",
             "identity_hash_attempts":"identity_hash_attempts_upper","identity_hash_bytes":"identity_hash_bytes_upper"}
    for event in observations:
        require(type(event) is dict and set(event)=={"kind","phase","operation_id","metadata"}
                and event["kind"]=="input_io" and type(event["operation_id"]) is str and event["operation_id"]
                and type(event["metadata"]) is dict,"bootstrap event schema")
        identifier,fields=event["operation_id"],event["metadata"]
        if event["phase"]=="started":
            require(identifier not in pending and identifier not in finished,"bootstrap start identity reuse")
            require(all(type(fields.get(key)) is int and fields[key]>=0 for key in mapping.values()),"bootstrap reservation fields/types")
            pending[identifier]=fields
            for counter,key in mapping.items():charges[counter]+=fields[key]
        else:
            require(event["phase"] in ("returned","failed") and identifier in pending,"bootstrap return identity/order")
            prior=pending.pop(identifier);finished.add(identifier)
            key="bytes_read" if event["phase"]=="returned" else "known_bytes_read"
            require(type(fields.get(key)) is int and 0<=fields[key]<=prior["read_bytes_upper"],"bootstrap observed read exceeded reservation")
    require(not pending,"bootstrap metadata admission left an unresolved input")
    return charges


class DurableLedger:
    def __init__(self, directory, planned, caps, bindings, extra_limits, *, resume=False,
                 clock=time.perf_counter):
        self.directory = Path(directory).resolve()
        self.plan, self.clock = planned, clock
        self.bindings = _load(canonical(bindings))
        self.extra_limits = dict(extra_limits)
        require(all(type(self.extra_limits.get(key)) is int and self.extra_limits[key] > 0 for key in IO_FIELDS),
                "explicit positive identity/IO/load/restore limits required")
        self.core = plan.Ledger(planned, caps)
        self.progress = plan.Progress(planned, self.core)
        self.plan_rows = [[p.row_id for p in boundary.prompts] for boundary in planned.boundaries]
        self.extra = {key: 0 for key in IO_FIELDS}
        self.external_ops, self.samples, self.invocations = {}, {}, {}
        self.used_restores=set()
        self.events, self.previous = [], None
        self.current_invocation = None
        self._wall_last = None
        self.restoration_pending = False
        self.fatal = False
        header = {"schema": SCHEMA, "bindings": self.bindings, "plan_sha256": planned.sha256,
                  "caps": caps, "extra_limits": self.extra_limits}
        self.binding_sha256 = digest(header)
        self.path = self.directory / "PHYSICAL_COSTS.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)
        header_path = self.directory / "LEDGER_BINDINGS.json"
        if resume:
            require(header_path.stat().st_size<=1024**2,"ledger header byte bound")
            self.header_bytes=header_path.stat().st_size
            require(header_path.is_file() and _load(header_path.read_bytes()) == header,
                    "resume ledger source/plan/recipe/caps binding drift")
            self._replay()
        else:
            require(not header_path.exists() and not self.path.exists(), "ledger is create-only")
            self._extra_check({"chunk_io_write_bytes": len(canonical(header)) + 1})
            _exclusive(header_path, header)
        self.header_bytes = header_path.stat().st_size
        self._extra_add({"chunk_io_write_bytes": self.header_bytes})
        if resume:
            self._event("setup_io", {"read_bytes": self.header_bytes + (self.path.stat().st_size if self.path.exists() else 0)})

    @property
    def generation(self):
        return self.core.generation

    @property
    def symbolic_snapshot(self):
        return self.core.symbolic_snapshot

    def _extra_check(self, changes):
        updated = dict(self.extra)
        for key, amount in changes.items():
            require(key in updated and type(amount) is int and amount >= 0, "physical IO counter/type")
            updated[key] += amount
            if updated[key] > self.extra_limits[key]:
                raise plan.CeilingRejected("cumulative " + key + " hard ceiling")
        return updated

    def _extra_add(self, changes):
        self.extra = self._extra_check(changes)

    def _replay(self):
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            while True:
                raw = stream.readline(MAX_LINE + 1)
                if not raw:
                    break
                require(len(raw) <= MAX_LINE and raw.endswith(b"\n"), "corrupt/partial cost journal; no truncation")
                entry = _load(raw)
                require(canonical(entry)+b"\n"==raw,"cost journal must retain exact canonical bytes")
                body = {key: value for key, value in entry.items() if key != "sha256"}
                require(body.get("seq") == len(self.events) and body.get("previous") == self.previous
                        and body.get("binding_sha256") == self.binding_sha256 and digest(body) == entry.get("sha256"),
                        "cost journal hash/order/source binding drift")
                self._extra_add({"chunk_io_write_bytes": len(raw)})
                self._apply(body["kind"], body["payload"])
                self.events.append(entry)
                self.previous = entry["sha256"]

    def _event(self, kind, payload):
        require(not self.fatal, "ledger failed; no further execution")
        body = {"seq": len(self.events), "previous": self.previous, "binding_sha256": self.binding_sha256,
                "kind": kind, "payload": payload,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        entry = dict(body, sha256=digest(body))
        raw = canonical(entry) + b"\n"
        require(len(raw) <= MAX_LINE, "cost event byte budget")
        # In-memory admission can conservatively advance before an IO failure.
        # No caller may start physical work until this method successfully fsyncs.
        self._extra_check({"chunk_io_write_bytes": len(raw)})
        try:
            self._apply(kind, payload)
            with self.path.open("ab") as stream:
                require(stream.write(raw) == len(raw), "short cost journal write")
                stream.flush()
                os.fsync(stream.fileno())
            self._extra_add({"chunk_io_write_bytes": len(raw)})
        except BaseException:
            self.fatal = True
            raise
        self.events.append(entry)
        self.previous = entry["sha256"]
        return entry

    def _rekey(self, table, generated, declared):
        require(declared not in table or declared == generated, "duplicate physical attempt")
        table[declared] = table.pop(generated)

    def _apply(self, kind, payload):
        if kind == "wall":
            self.core.charge_wall(payload["seconds"])
        elif kind == "unknown_wall":
            self.core.unresolved_wall()
        elif kind == "setup_io":
            if "observations" in payload:
                require(digest(payload["observations"])==payload["observations_sha256"]
                        and bootstrap_charges(payload["observations"])==payload["charges"],"bootstrap observation digest/charges drift")
                self._extra_add(payload["charges"])
            else:
                expected=self.header_bytes+sum(len(canonical(event))+1 for event in self.events)
                require(payload["read_bytes"]==expected,"resume ledger observed read size drift")
                self._extra_add({"chunk_io_read_bytes":payload["read_bytes"]})
        elif kind == "invocation_start":
            self.core.invocation(resume=payload["resume"])
            require(payload["id"] not in self.invocations, "duplicate invocation")
            self.invocations[payload["id"]] = dict(payload, ended=False)
        elif kind == "invocation_end":
            require(payload["id"] in self.invocations and not self.invocations[payload["id"]]["ended"], "invocation end identity")
            self.invocations[payload["id"]]["ended"] = True
            self.invocations[payload["id"]]["outcome"] = payload["outcome"]
        elif kind == "rollout_start":
            require(not self.restoration_pending, "native restore fence not installed")
            require(hex64(payload["policy_id"]), "sampled behavior policy hash")
            generated = self.core.rollout_start(payload["trajectory"], payload["prompt_tokens"])
            self._rekey(self.core.pending_rollouts, generated, payload["attempt"])
            self.core.pending_rollouts[payload["attempt"]]["policy_id"] = payload["policy_id"]
        elif kind == "rollout_return":
            sample = payload["sample"]
            started = self.core.pending_rollouts.get(payload["attempt"], {})
            require(sample.get("rollout_id") == payload["attempt"] and hex64(sample.get("policy_id"))
                    and sample["policy_id"] == started.get("policy_id")
                    and sample.get("prompt_id") == started.get("row_id"), "returned rollout identity")
            require(type(sample.get("reward")) is int and sample["reward"] in (0, 1)
                    and type(sample.get("completion_ids")) is list
                    and len(sample["completion_ids"]) == sample.get("generated_tokens"), "returned rollout reward/token count")
            self.core.rollout_return(payload["attempt"], payload["sample"]["generated_tokens"])
            self.samples[payload["attempt"]] = payload["sample"]
        elif kind == "rollout_failed":
            require(payload["attempt"] in self.core.pending_rollouts, "failed rollout must have a durable start")
            # The whole generated-token reservation and prefill stay charged.
        elif kind == "operation":
            self._apply_operation(payload)
        elif kind == "boundary_commit":
            require(not self.restoration_pending, "restore fence not installed")
            record = payload["update"]
            require(record["generation"] == self.generation and record["symbolic_snapshot"] == self.symbolic_snapshot
                    and record["boundary_id"] == self.progress.C and record["U_before"] == self.progress.U, "returned update C/U/fence")
            attempts = payload["attempts"]
            require(len(attempts) == 16 and len(set(attempts)) == 16 and all(a in self.samples for a in attempts), "complete returned rollout batch")
            require(all(self.samples[a]["policy_id"] == record["start_policy_id"] for a in attempts), "one behavior policy for committed batch")
            expected = [self.plan.trajectory(self.progress.C, i, k)["logical_id"] for i in range(2) for k in range(8)]
            require([self.core.rollouts[a]["logical_id"] for a in attempts] == expected, "rollout order must be exact prompt-major G8")
            rewards = [[self.samples[a]["reward"] for a in attempts[i:i + 8]] for i in (0, 8)]
            signal = any(0 < sum(group) < 8 for group in rewards)
            optimizer = record.get("optimizer_operation")
            op_id = optimizer["operation_id"] if optimizer is not None else None
            require(type(record["updated"]) is bool and record["updated"] == signal
                    and record["U_after"] == self.progress.U + int(signal), "actual Adam advancement proof")
            pre_op = record.get("identity_operation") or {}
            pre = self.external_ops.get(pre_op.get("operation_id"), {})
            require(pre.get("phase") == "returned" and pre.get("operation") == "fingerprint"
                    and all(pre["metadata"].get(k) == record[k] for k in
                            ("generation", "symbolic_snapshot", "boundary_id", "start_policy_id")), "pre-update identity proof absent/stale")
            require(pre["result"].get("policy_sha256")==record["start_policy_id"]
                    and pre["result"].get("U")==record["U_before"] and hex64(record.get("frozen_state_sha256"))
                    and pre["result"].get("frozen_state_sha256")==record["frozen_state_sha256"],"pre-update native hash/U/frozen proof drift")
            if record["updated"]:
                require(record.get("parity") == "PASS" and hex64(record.get("final_policy_id"))
                        and optimizer is not None and self.external_ops.get(op_id, {}).get("phase") == "returned", "returned proved optimizer required")
                step = self.external_ops[op_id]
                require(step["operation"] == "optimizer" and step["metadata"].get("U_before") == self.progress.U
                        and step["metadata"].get("U_after") == self.progress.U + 1
                        and all(step["metadata"].get(k) == record[k] for k in
                                ("generation", "symbolic_snapshot", "boundary_id", "start_policy_id")), "optimizer C/U proof drift")
                final_op = record.get("final_identity_operation") or {}
                post = self.external_ops.get(final_op.get("operation_id"), {})
                require(post.get("phase") == "returned" and post.get("operation") == "fingerprint"
                        and all(post["metadata"].get(k) == record[k] for k in
                                ("generation", "symbolic_snapshot", "boundary_id", "start_policy_id")), "post-update identity proof absent/stale")
                require(post["result"].get("policy_sha256")==record["final_policy_id"]
                        and post["result"].get("U")==record["U_after"]
                        and post["result"].get("frozen_state_sha256")==record["frozen_state_sha256"],"post-update native hash/U/frozen proof drift")
            self.progress.commit(rewards, attempts, op_id)
        elif kind == "install_restore":
            progress = payload["progress"]
            validate_progress(progress, self.plan_rows)
            require(progress["C"] <= self.progress.C and progress["history"] == self.progress.history[:progress["C"]]
                    and progress["generation"] > self.generation, "restore must be an actual ancestor with a fresh fence")
            identifier=payload.get("restore_operation_id")
            op=self.external_ops.get(identifier,{})
            require(identifier not in self.used_restores and hex64(payload.get("manifest_sha256"))
                    and op.get("operation")=="checkpoint_restore" and op.get("phase")=="returned"
                    and op["result"].get("manifest_sha256")==payload["manifest_sha256"]
                    and all(op["result"].get(k)==progress[v] for k,v in
                            (("C","C"),("U","U"),("generation","generation"),("symbolic_snapshot","snapshot"))),
                    "native returned restore/manifest/fence proof missing or reused")
            self.progress.C, self.progress.U = progress["C"], progress["U"]
            self.progress.history = _load(canonical(progress["history"]))
            self.core.generation, self.core.symbolic_snapshot = progress["generation"], progress["snapshot"]
            self.used_restores.add(identifier)
            self.restoration_pending = False
        else:
            raise ValueError("unknown cost/progress event: " + kind)

    def _apply_operation(self, payload):
        kind, phase, identifier = payload["operation"], payload["phase"], payload["operation_id"]
        metadata = payload["metadata"]
        mapped = {"loss_forward": "loss_forward", "backward": "backward", "optimizer": "optimizer", "checkpoint_save": "checkpoint"}
        if phase == "started":
            require(identifier not in self.external_ops, "duplicate operation start")
            require(kind in RESTORE_BOOTSTRAP or not self.restoration_pending, "restore fence not installed")
            if "generation" in metadata:
                require(metadata["generation"] == self.generation and metadata.get("symbolic_snapshot") == self.symbolic_snapshot,
                        "operation start fence drift")
            changes = {"chunk_io_read_bytes": metadata.get("read_bytes_upper", 0),
                       "chunk_io_write_bytes": metadata.get("write_bytes_upper", 0),
                       "identity_hash_attempts": metadata.get("identity_hash_attempts_upper", 0),
                       "identity_hash_bytes": metadata.get("identity_hash_bytes_upper", 0)}
            if kind == "fingerprint":
                changes["identity_hash_attempts"] += 1
                changes["identity_hash_bytes"] += metadata["state_bytes"]
            if kind == "checkpoint_restore":
                changes["checkpoint_restore_attempts"] = 1
            if kind == "model_load":
                changes["model_load_attempts"] = 1
            self._extra_check(changes)
            for stem in ("identity_hash_attempts","identity_hash_bytes"):
                if stem+"_lower" in metadata:
                    require(type(metadata[stem+"_lower"]) is int and 0<=metadata[stem+"_lower"]<=changes[stem],
                            "opaque helper hash lower exceeds reservation")
            if kind in mapped:
                generated = self.core.op_start(mapped[kind], boundary=metadata.get("boundary_id", metadata.get("C")),
                        selected_tokens=metadata.get("selected_tokens", 0), full_sequence_tokens=metadata.get("full_sequence_tokens", 0))
                self._rekey(self.core.pending_ops, generated, identifier)
            self._extra_add(changes)
            self.external_ops[identifier] = {"operation": kind, "phase": "started", "metadata": metadata,
                                            "reservations":changes}
            if kind == "checkpoint_restore":
                self.restoration_pending = True
        else:
            require(phase in ("returned", "failed") and identifier in self.external_ops
                    and self.external_ops[identifier]["phase"] == "started"
                    and self.external_ops[identifier]["operation"] == kind, "operation return/failure identity")
            prior = self.external_ops[identifier]["metadata"]
            reservations=self.external_ops[identifier]["reservations"]
            for field,counter in (("identity_hash_attempts_actual","identity_hash_attempts"),
                                  ("identity_hash_bytes_actual","identity_hash_bytes"),
                                  ("known_identity_hash_attempts","identity_hash_attempts"),
                                  ("known_identity_hash_bytes","identity_hash_bytes")):
                if field in metadata:
                    require(type(metadata[field]) is int and 0<=metadata[field]<=reservations[counter],
                            "actual hash work exceeded reservation")
            for observed, upper in (("bytes_read", "read_bytes_upper"), ("bytes_written", "write_bytes_upper"),
                                    ("known_bytes_read", "read_bytes_upper"), ("known_bytes_written", "write_bytes_upper")):
                if observed in metadata:
                    require(type(metadata[observed]) is int and 0 <= metadata[observed] <= prior.get(upper, 0), "physical IO exceeded reserved upper")
            if kind != "checkpoint_restore":
                for field in ("generation", "symbolic_snapshot", "boundary_id", "start_policy_id", "U_before", "U_after"):
                    if field in prior and field in metadata:
                        require(metadata[field] == prior[field], "operation return fence/C/U drift")
            # IO reservations stay charged unless exact instrumentation is available.
            # This adapter deliberately never refunds checkpoint/IO uncertainty.
            if kind in mapped and phase == "returned":
                self.core.op_return(identifier)
            self.external_ops[identifier]["phase"] = phase
            self.external_ops[identifier]["result"] = metadata

    def tick_wall(self):
        if self.current_invocation is None or self._wall_last is None:
            return
        now = self.clock()
        require(now >= self._wall_last, "worker monotonic clock moved backwards")
        self._event("wall", {"seconds": now - self._wall_last, "invocation": self.current_invocation})
        self._wall_last = now

    def start_invocation(self, *, resume=False, reason=None, started=None, boot_id=None):
        require(self.current_invocation is None, "invocation already active")
        if resume:
            require(type(reason) is str and reason.strip(), "documented resume reason required")
        now = self.clock()
        for identifier, old in self.invocations.items():
            if old["ended"]:
                continue
            if boot_id is None or old.get("boot_id") != boot_id or now < old["perf_started"]:
                self._event("unknown_wall", {"invocation": identifier})
                raise plan.UnresolvedBudget("interrupted invocation has no defensible wall upper bound")
            charged = sum(e["payload"]["seconds"] for e in self.events if e["kind"] == "wall" and e["payload"].get("invocation") == identifier)
            self._event("wall", {"seconds": max(0.0, now - old["perf_started"] - charged), "invocation": identifier,
                                 "rule": "same-boot upper bound includes restart delay"})
            self._event("invocation_end", {"id": identifier, "outcome": "interrupted_upper_bound_charged"})
        identifier = uuid.uuid4().hex
        begin = now if started is None else started
        require(type(begin) in (int, float) and math.isfinite(begin) and begin <= now, "invocation start clock")
        self._event("invocation_start", {"id": identifier, "resume": bool(resume), "reason": reason,
                    "perf_started": begin, "boot_id": boot_id})
        self.current_invocation, self._wall_last = identifier, begin
        self.tick_wall()
        return identifier

    def end_invocation(self, outcome):
        self.tick_wall()
        self._event("invocation_end", {"id": self.current_invocation, "outcome": outcome})
        self.current_invocation, self._wall_last = None, None

    def operation(self, kind, phase, operation_id, **metadata):
        self.tick_wall()
        self._event("operation", {"operation": kind, "phase": phase, "operation_id": operation_id, "metadata": metadata})

    def charge_setup(self, observations):
        """Charge bounded bootstrap reads that preceded plan admission, before framework work."""
        changes=bootstrap_charges(observations)
        self._event("setup_io", {"charges": changes, "observations_sha256": digest(observations),
                                 "observations": observations,
                                 "scope": "bounded bootstrap before admitted plan; charged before framework"})

    def rollout_start(self, trajectory, prompt_tokens, policy_id):
        self.tick_wall()
        attempt = uuid.uuid4().hex
        self._event("rollout_start", {"attempt": attempt, "trajectory": trajectory,
                    "prompt_tokens": prompt_tokens, "policy_id": policy_id})
        return attempt

    def rollout_return(self, attempt, sample):
        self.tick_wall()
        self._event("rollout_return", {"attempt": attempt, "sample": sample})

    def rollout_failed(self, attempt, error):
        self.tick_wall()
        self._event("rollout_failed", {"attempt":attempt,"error":{"type":type(error).__name__,"message":str(error)[:1000]}})

    def commit_boundary(self, attempts, update_record):
        self.tick_wall()
        self._event("boundary_commit", {"attempts": attempts, "update": update_record})

    def progress_record(self):
        return {"C": self.progress.C, "U": self.progress.U, "history": _load(canonical(self.progress.history)),
                "generation": self.generation, "snapshot": self.symbolic_snapshot}

    def install_restore(self, restored):
        self.tick_wall()
        matching=[identifier for identifier,op in self.external_ops.items() if identifier not in self.used_restores
                  and op["operation"]=="checkpoint_restore" and op["phase"]=="returned"
                  and op["result"].get("manifest_sha256")==restored["manifest_sha256"]
                  and op["result"].get("generation")==restored["progress"]["generation"]]
        require(len(matching)==1,"one unused returned restore/manifest proof required")
        self._event("install_restore", {"progress": restored["progress"], "manifest_sha256": restored["manifest_sha256"],
                                        "restore_operation_id":matching[0]})

    def snapshot(self):
        return {"schema": SCHEMA, "binding_sha256": self.binding_sha256, "events": len(self.events),
                "last_event_sha256": self.previous, "progress": {k: v for k, v in self.progress_record().items() if k != "history"},
                "history_sha256": digest(self.progress.history), "cumulative": dict(self.core.counters),
                "identity_and_IO_cumulative": dict(self.extra), "unknown_wall": self.core.wall_unknown,
                "pending_rollouts": len(self.core.pending_rollouts),
                "pending_operations": sum(op["phase"] == "started" for op in self.external_ops.values())}
