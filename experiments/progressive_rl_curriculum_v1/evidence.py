"""Single-writer, atomic checkpoint transactions and conservative compute accounting."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .schema import Limits, canonical, digest, require


class BudgetExhausted(RuntimeError):
    pass


def file_hash(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Store:
    def __init__(self, root: Path, identity: dict):
        self.root = root
        self.identity = identity
        self.root.mkdir(parents=True, exist_ok=True)
        require(not root.is_symlink(), "run directory may not be a symlink")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        path = self.root / "RUNNING.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError("run is locked; do not auto-delete a possibly live writer's lock") from error
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            path.unlink(missing_ok=True)

    def commit(self, state: dict, backend, *, good: bool = False) -> str:
        previous = None
        sequence = 0
        latest = self.root / "LATEST.json"
        if latest.exists():
            previous = json.loads(latest.read_text())
            sequence = previous["sequence"] + 1
        existing = [int(p.name[11:]) for p in self.root.glob("checkpoint-*")
                    if p.name[11:].isdigit()]
        sequence = max([sequence, *(n + 1 for n in existing)])
        name = f"checkpoint-{sequence:08d}"
        destination = self.root / name
        require(not destination.exists(), "checkpoint collision; refusing overwrite")
        if good:
            state["last_good"] = name
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.root))
        try:
            backend.save(temporary)
            atomic_json(temporary / "state.json", state)
            files = {p.name: file_hash(p) for p in temporary.iterdir()}
            receipt = {"identity": self.identity, "files": files,
                       "previous": previous, "sequence": sequence}
            atomic_json(temporary / "receipt.json", receipt)
            os.replace(temporary, destination)
            atomic_json(latest, {"name": name, "sequence": sequence, "receipt": digest(receipt)})
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return name

    def inspect(self, name: str | None = None) -> tuple[Path, dict]:
        pointer = json.loads((self.root / "LATEST.json").read_text())
        if name is not None:
            seen = set()
            while pointer["name"] != name:
                require(pointer["name"] not in seen, "cycle in checkpoint history")
                seen.add(pointer["name"])
                parent_name = pointer["name"]
                require(parent_name.startswith("checkpoint-") and parent_name[11:].isdigit(), "invalid ancestor path")
                ancestor = json.loads((self.root / parent_name / "receipt.json").read_text())
                require(digest(ancestor) == pointer["receipt"], "ancestor receipt integrity failure")
                pointer = ancestor["previous"]
                require(pointer is not None, "checkpoint is not in committed history")
        name = pointer["name"]
        require(isinstance(name, str) and name.startswith("checkpoint-") and name[11:].isdigit(),
                "invalid checkpoint path")
        path = self.root / name
        require(path.is_dir() and not path.is_symlink(), "missing checkpoint directory")
        receipt = json.loads((path / "receipt.json").read_text())
        require(receipt["identity"] == self.identity, "plan/catalog/model/runtime identity changed")
        if pointer:
            require(digest(receipt) == pointer["receipt"], "checkpoint receipt changed")
        require(set(p.name for p in path.iterdir()) == set(receipt["files"]) | {"receipt.json"},
                "unexpected or missing checkpoint files")
        for filename, expected in receipt["files"].items():
            require(Path(filename).name == filename and file_hash(path / filename) == expected,
                    f"checkpoint integrity failure: {filename}")
        return path, json.loads((path / "state.json").read_text())

    def restore(self, backend, name: str | None = None) -> dict:
        path, state = self.inspect(name)
        backend.load(path)
        return state


class Ledger:
    """Reserve worst-case work BEFORE I/O. Interrupted work keeps the reservation."""
    KEYS = ("generated_tokens", "processed_tokens", "rollout_calls", "optimizer_steps")

    def __init__(self, limits: Limits, state: dict):
        self.limits = limits
        self.state = state
        state.setdefault("used", {**dict.fromkeys(self.KEYS, 0), "wall_seconds": 0.0})
        state.setdefault("pending", None)

    def reserve(self, **cost: int) -> None:
        require(self.state["pending"] is None, "unsettled operation")
        require(set(cost) <= set(self.KEYS), "unknown budget unit")
        require(all(type(v) is int and v >= 0 for v in cost.values()), "invalid budget charge")
        if self.state["used"]["wall_seconds"] >= self.limits.wall_seconds:
            raise BudgetExhausted("wall time budget exhausted")
        for key, value in cost.items():
            if self.state["used"][key] + value > getattr(self.limits, key):
                raise BudgetExhausted(f"{key} budget exhausted")
        for key, value in cost.items():
            self.state["used"][key] += value
        self.state["pending"] = {"cost": cost, "started": time.time()}

    def settle(self, actual: dict[str, int] | None = None) -> None:
        pending = self.state["pending"]
        require(pending is not None, "no pending operation")
        if actual is not None:
            require(set(actual) == set(pending["cost"]), "incomplete settlement")
            for key, value in actual.items():
                require(type(value) is int and 0 <= value <= pending["cost"][key], "budget overrun")
            for key, value in actual.items():
                self.state["used"][key] -= pending["cost"][key] - value
        self.state["used"]["wall_seconds"] += max(0, time.time() - pending["started"])
        self.state["pending"] = None


def resource_snapshot(run_dir: Path, *, cuda: bool) -> dict:
    import psutil
    result = {"host_free_gib": psutil.virtual_memory().available / 2**30,
              "disk_free_gib": shutil.disk_usage(run_dir).free / 2**30, "gpu_free_gib": None}
    if cuda:
        import torch
        require(torch.cuda.is_available(), "CUDA unavailable")
        result["gpu_free_gib"] = torch.cuda.mem_get_info()[0] / 2**30
    return result


def check_resources(limits: Limits, run_dir: Path, *, cuda: bool, preflight: bool) -> list[dict]:
    snapshots = []
    for index in range(2 if preflight else 1):
        if index:
            time.sleep(1)
        snapshot = resource_snapshot(run_dir, cuda=cuda)
        threshold = limits.host_free_gib if preflight else limits.worker_free_gib
        require(snapshot["host_free_gib"] >= threshold, "BLOCKED_HOST_RAM")
        require(snapshot["disk_free_gib"] >= limits.disk_free_gib, "BLOCKED_DISK")
        if cuda and preflight:
            require(snapshot["gpu_free_gib"] >= limits.gpu_free_gib, "BLOCKED_GPU_RAM")
        snapshots.append(snapshot)
    return snapshots
