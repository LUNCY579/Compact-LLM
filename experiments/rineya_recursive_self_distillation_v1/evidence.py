"""Write-once checkpoints and strict clean-boundary resume.

An interrupted transaction is deliberately not auto-resumed: exact spending is
unknown and the run is quarantined. Completed checkpoints resume only within the
same locked run directory and exact binding. Nothing refunds previous spending.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from .contracts import canonical, digest, file_hash, read_json, require


def write_new(path: str | Path, value) -> None:
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_new(temporary, value)
    os.replace(temporary, path)


class RunStore:
    def __init__(self, path: str | Path, binding: dict, plan, *, resume=False):
        self.path, self.plan = Path(path), plan
        if resume:
            require(self.path.is_dir(), "resume requires existing run")
        else:
            self.path.mkdir(parents=True, exist_ok=False)
        self.lock = self.path / "RUN.lock"
        self.lock_fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self.lock_fd, str(os.getpid()).encode())
        self.started = time.monotonic()
        self.last_manifest = None
        self.pending = None
        try:
            if resume:
                require(not (self.path / "INFLIGHT.json").exists(),
                        "interrupted transaction: quarantine; no automatic retry or spend reset")
                record = read_json(self.path / "RUN.json")
                require(record["binding"] == binding, "run/model/data/source/plan identity drift")
                pointer = read_json(self.path / "LATEST.json")
                checkpoint = self.path / pointer["directory"]
                require(checkpoint.parent == self.path and checkpoint.name.startswith("checkpoint-"), "checkpoint path")
                require(file_hash(checkpoint / "MANIFEST.json") == pointer["manifest_sha256"], "manifest drift")
                manifest = read_json(checkpoint / "MANIFEST.json")
                require(manifest["binding"] == binding, "checkpoint binding drift")
                require(set(manifest["files"]) == {"ADAPTER_STATE.pt", "STATE.json"}, "checkpoint inventory drift")
                for name, expected in manifest["files"].items():
                    require(Path(name).name == name and file_hash(checkpoint / name) == expected, "checkpoint file drift")
                self.state = read_json(checkpoint / "STATE.json")
                require(self.state["status"] == "PAUSED", "only a clean PAUSED checkpoint may resume")
                self.last_manifest = pointer["manifest_sha256"]
                self.checkpoint_path = checkpoint
            else:
                write_new(self.path / "RUN.json", {"binding": binding, "protocol": "RSD-001-v1",
                          "evidence": binding["evidence"], "native_efficacy_claim": False})
                self.state = {"step": 0, "status": "NEW", "processed_tokens": 0, "generated_tokens": 0,
                              "model_calls": 0, "optimizer_steps": 0, "wall_seconds": 0.0,
                              "history": [], "evaluations": []}
            self.binding = binding
            self.prior_wall = self.state["wall_seconds"]
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "lock_fd", None) is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
            self.lock.unlink()

    def begin(self, operation: str):
        require(self.pending is None, "nested transaction")
        self.pending = operation
        write_new(self.path / "INFLIGHT.json", {"operation": operation, "step": self.state["step"],
                  "before": self.state, "interrupted_spending": "unknown; bounded by remaining run limits"})

    def meter(self, kind: str, processed: int, generated: int):
        require(self.pending is not None, "unaccounted model work")
        require(processed >= 0 and generated >= 0, "negative accounting")
        calls = int(kind == "hidden")
        for field, delta, limit in (("processed_tokens", processed, self.plan.processed_token_limit),
                                    ("generated_tokens", generated, self.plan.generated_token_limit),
                                    ("model_calls", calls, self.plan.model_call_limit)):
            require(self.state[field] + delta <= limit, field + " budget exhausted")
        self.state["wall_seconds"] = self.prior_wall + time.monotonic() - self.started
        require(self.state["wall_seconds"] <= self.plan.wall_seconds, "cumulative wall budget exhausted")
        self.state["processed_tokens"] += processed
        self.state["generated_tokens"] += generated
        self.state["model_calls"] += calls

    def save(self, backend, optimizer, *, status: str):
        import torch
        self.state["status"] = status
        self.state["wall_seconds"] = self.prior_wall + time.monotonic() - self.started
        directory = self.path / ("checkpoint-" + uuid.uuid4().hex)
        directory.mkdir()
        payload = {"parameters": backend.snapshot(), "initial_teacher": backend.initial,
                   "initial_digest": backend.initial_digest, "version": backend.version,
                   "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else []}
        with (directory / "ADAPTER_STATE.pt").open("xb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        write_new(directory / "STATE.json", self.state)
        manifest = {"binding": self.binding, "previous_manifest": self.last_manifest,
                    "files": {n: file_hash(directory / n) for n in ("ADAPTER_STATE.pt", "STATE.json")}}
        write_new(directory / "MANIFEST.json", manifest)
        self.last_manifest = file_hash(directory / "MANIFEST.json")
        atomic_json(self.path / "LATEST.json", {"directory": directory.name, "manifest_sha256": self.last_manifest})
        if self.pending:
            # Crash after pointer replacement but before unlink also fails closed.
            (self.path / "INFLIGHT.json").unlink()
            self.pending = None
        self.checkpoint_path = directory

    def load(self, backend, optimizer):
        import torch
        from .backend import tensor_hash
        saved = torch.load(self.checkpoint_path / "ADAPTER_STATE.pt", map_location="cpu", weights_only=True)
        require(saved["initial_digest"] == backend.initial_digest
                and tensor_hash(saved["initial_teacher"]) == backend.initial_digest, "frozen teacher drift")
        backend.restore(saved["parameters"])
        backend.version = saved["version"]
        require(backend.version == self.state["step"], "policy/checkpoint version mismatch")
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if backend.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def fail(self, exc: BaseException):
        name = "FAILURE-" + uuid.uuid4().hex + ".json"
        write_new(self.path / name, {"status": "FAILED_QUARANTINED", "error": str(exc),
                  "exception": type(exc).__name__, "last_state": self.state,
                  "native_efficacy_claim": False, "inflight_retained": self.pending is not None})
