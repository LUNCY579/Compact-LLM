"""Durable independent attempts, failure-preserving resume, and hard token reservations."""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Callable, Protocol

from .contracts import (Blocked, VERSION, digest, directory_bytes, encoded, exact, integer, loads,
                        read_json, require, seed_for, source_identity, write_new)


class Backend(Protocol):
    identity: dict

    def generate(self, task: dict, seed: int, greedy: bool, config: dict,
                 emit_token: Callable[[int, float, float], None]) -> dict: ...


def slots(study: dict, public: list[dict]) -> list[dict]:
    config = study["config"]
    result = []
    for task in public:
        for index in range(config["samples"] + 1):
            arm = "greedy" if index == 0 else "sample"
            result.append({"slot": len(result), "task": task["id"], "index": index,
                           "arm": arm, "seed": seed_for(config["seed"], task["id"], arm, index)})
    return result


class Journal:
    """Hash-chained JSONL detects corruption, not malicious rewriting by its owner."""
    def __init__(self, path: Path, max_bytes: int):
        self.path, self.max_bytes = Path(path), max_bytes
        self.sequence, self.previous = 0, "0" * 64
        self.records: dict[int, dict] = {}
        self.pending: int | None = None
        self.tokens = 0
        if self.path.exists():
            require(self.path.stat().st_size <= max_bytes, "journal size limit")
            with self.path.open("rb") as f:
                for line in f:
                    require(len(line) <= 2**20 and line.endswith(b"\n"), "torn/oversize journal; preserve and investigate")
                    event = loads(line)
                    h = event.pop("hash")
                    require(event["seq"] == self.sequence and event["prev"] == self.previous
                            and digest(event) == h, "journal hash/sequence drift")
                    self._apply(event["event"], event["data"])
                    self.sequence += 1
                    self.previous = h

    def _apply(self, kind: str, data: dict) -> None:
        slot = data["slot"]
        integer(slot, 0, 1024*65, "slot")
        if kind == "begin":
            exact(data, {"slot", "task", "index", "arm", "seed"}, "attempt")
            require(self.pending is None and slot not in self.records and slot == len(self.records),
                    "attempt must be the next unused slot")
            self.records[slot] = {**data, "state": "pending", "token_ids": [],
                                  "raw_logprobs": [], "policy_logprobs": []}
            self.pending = slot
        elif kind == "token":
            exact(data, {"slot", "id", "raw_lp", "policy_lp"}, "token")
            require(self.pending == slot, "token without matching pending attempt")
            integer(data["id"], 0, 2**31-1, "token ID")
            for key in ("raw_lp", "policy_lp"):
                require(type(data[key]) in (int, float) and -1e9 <= data[key] <= 1e-6,
                        "invalid token logprob")
            r = self.records[slot]
            r["token_ids"].append(data["id"])
            r["raw_logprobs"].append(data["raw_lp"])
            r["policy_logprobs"].append(data["policy_lp"])
            self.tokens += 1
        elif kind in {"returned", "failed"}:
            require(self.pending == slot, "terminal event without matching pending attempt")
            r = self.records[slot]
            if kind == "returned":
                exact(data, {"slot", "text", "stop", "prompt_tokens", "seconds", "forward_positions"}, "return")
                require(type(data["text"]) is str and len(data["text"]) <= 262144,
                        "invalid decoded output")
                require(data["stop"] in {"eos", "length"} and bool(r["token_ids"]), "invalid termination")
                integer(data["prompt_tokens"], 1, 896, "prompt tokens")
                integer(data["forward_positions"], 1, 1024*128, "forward positions")
                require(type(data["seconds"]) in (int, float) and 0 <= data["seconds"] <= 21600,
                        "invalid generation duration")
            else:
                exact(data, {"slot", "reason", "uncertain_work"}, "failure")
                require(type(data["reason"]) is str and len(data["reason"]) <= 1000
                        and type(data["uncertain_work"]) is bool, "invalid failure")
            r.update(data)
            r["state"] = kind
            self.pending = None
        else:
            raise Blocked("unknown journal event")

    def append(self, kind: str, data: dict) -> None:
        if kind in {"begin", "returned", "failed"}:
            require(directory_bytes(self.path.parent) < self.max_bytes, "aggregate output storage budget")
        event = {"seq": self.sequence, "prev": self.previous, "event": kind, "data": data}
        h = digest(event)
        raw = encoded({**event, "hash": h}) + b"\n"
        require((self.path.stat().st_size if self.path.exists() else 0) + len(raw) <= self.max_bytes,
                "journal storage budget")
        # Validate before persisting. A write failure aborts the process; do not reuse this object.
        self._apply(kind, data)
        with self.path.open("ab") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        self.sequence += 1
        self.previous = h


def validate_bank(journal: Journal, inventory: list[dict], config: dict) -> None:
    for i, record in journal.records.items():
        require(i < len(inventory), "unknown attempt slot")
        require(all(record[k] == v for k, v in inventory[i].items()), "attempt identity or seed drift")
        require(len(record["token_ids"]) <= config["max_new_tokens"], "sample token ceiling exceeded")
    require(len(journal.records)*config["max_new_tokens"] <= config["max_generated_tokens"],
            "attempt reservations exceed generation budget")


def collect(study: dict, public: list[dict], backend: Backend, out: Path, *, resume: bool = False,
            max_attempts: int | None = None, abandon_interrupted: bool = False,
            pause_after_seconds: float | None = None) -> dict:
    config = study["config"]
    out.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    bound = {"schema": VERSION, "study_id": study["study_id"], "backend": backend.identity,
             "code_digest": identity["code_digest"]}
    manifest_path = out / "run.json"
    if manifest_path.exists():
        require(resume, "run already exists; use explicit resume")
        require(read_json(manifest_path)["binding"] == bound, "run/model/runtime/source drift")
    else:
        require(not resume and not (out / "events.jsonl").exists(), "no resumable run manifest")
        write_new(manifest_path, {"binding": bound, "source": identity})
    journal = Journal(out / "events.jsonl", config["max_output_bytes"])
    inventory = slots(study, public)
    validate_bank(journal, inventory, config)
    if journal.pending is not None:
        require(abandon_interrupted, "pending attempt: review evidence, then explicitly abandon it; never redraw")
        journal.append("failed", {"slot": journal.pending, "reason": "explicitly_abandoned_interrupted_attempt",
                                  "uncertain_work": True})
    tasks = {row["id"]: row for row in public}
    if max_attempts is None:
        max_attempts = len(inventory)
    integer(max_attempts, 1, 1024*65, "max attempts")
    started, executed = time.monotonic(), 0
    error = None
    for item in inventory[len(journal.records):]:
        if executed >= max_attempts or (pause_after_seconds is not None and executed > 0
                                       and time.monotonic()-started >= pause_after_seconds):
            break
        require((len(journal.records)+1)*config["max_new_tokens"] <= config["max_generated_tokens"],
                "reserved token budget exhausted")
        journal.append("begin", item)
        def token(tok: int, raw_lp: float, policy_lp: float) -> None:
            require(len(journal.records[item["slot"]]["token_ids"]) < config["max_new_tokens"],
                    "per-attempt token budget exhausted")
            journal.append("token", {"slot": item["slot"], "id": tok, "raw_lp": raw_lp, "policy_lp": policy_lp})
        try:
            result = backend.generate(tasks[item["task"]], item["seed"], item["arm"] == "greedy", config, token)
            journal.append("returned", {"slot": item["slot"], **result})
        except Exception as exc:
            # Returned-token journal can bound, but not fully meter, failed asynchronous work.
            if journal.pending is not None:
                journal.append("failed", {"slot": item["slot"], "reason": f"{type(exc).__name__}: {exc}"[:1000],
                                          "uncertain_work": True})
            error = f"{type(exc).__name__}: {exc}"[:1000]
            break
        executed += 1
    validate_bank(journal, inventory, config)
    complete = len(journal.records) == len(inventory) and journal.pending is None
    return {"status": "BLOCKED" if error else ("COMPLETE" if complete else "PAUSED"),
            "reason": error, "scheduled": len(inventory), "started": len(journal.records),
            "returned": sum(r["state"] == "returned" for r in journal.records.values()),
            "failed": sum(r["state"] == "failed" for r in journal.records.values()),
            "durably_recorded_tokens": journal.tokens, "token_reservation_upper": len(journal.records)*config["max_new_tokens"],
            "collection_seconds_this_segment": time.monotonic()-started,
            "evidence": backend.identity["evidence"]}


class FixtureBackend:
    """Scripted public-prompt reader, solely for mechanics demonstrations."""
    def __init__(self, scenario: str = "minority"):
        require(scenario in {"minority", "flat", "all_correct", "vote_gain"}, "unknown fixture scenario")
        self.identity = {"name": "scripted_fixture", "scenario": scenario, "evidence": "SYNTHETIC_ONLY"}
        self.calls = 0

    def generate(self, task: dict, seed: int, greedy: bool, config: dict, emit_token: Callable) -> dict:
        import re
        self.calls += 1
        text = task["messages"][-1]["content"]
        match = re.search(r'\{"answer":([0-9]+)\}', text)
        require(match is not None, "fixture only accepts its own public prompts")
        expected = int(match.group(1))
        scenario = self.identity["scenario"]
        good = (scenario == "all_correct" or
                (not greedy and ((scenario == "minority" and seed % 5 == 0) or
                                 (scenario == "vote_gain" and seed % 5 != 0))))
        output = encoded({"answer": expected if good else -1}).decode()
        emit_token(100 if good else 101, -0.7, -0.7)
        emit_token(151645, -0.1, -0.1)
        return {"text": output, "stop": "eos", "prompt_tokens": 8, "seconds": 0.0, "forward_positions": 9}
