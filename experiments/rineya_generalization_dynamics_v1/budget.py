"""Cumulative, immutable native-attempt accounting; no model or bank access.

Wall deadlines are cooperative: an in-flight load or forward can overrun the
deadline. An owned-child supervisor is still required for a hard wall limit.
"""
from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import Blocked, read, require, signed, verify, write_new


class NativeBudget:
    """One attempt, charged against all earlier attempts in the same split.

Call only while holding the run's single-writer lock. A STARTED receipt without
its finished partner blocks resumption, because its consumption is unknown.
"""

    def __init__(self, run: Path, split: str, manifest: dict, *, require_remaining: bool = True):
        require(split in {"dev", "confirm"}, "Unknown native-budget split")
        self.directory = run / "native_budget" / split
        self.invalidation_directory = run / "native_registry_invalidations" / split
        self.binding = {"manifest_sha256": manifest["sha256"], "split": split,
                        "max_native_wall_seconds": manifest["runtime"]["max_native_wall_seconds"],
                        "max_native_forward_calls": manifest["runtime"]["max_native_forward_calls"]}
        if self.invalidation_directory.exists():
            for path in self.invalidation_directory.iterdir():
                require(re.fullmatch(r"[0-9]{6}\.json", path.name) is not None and path.is_file(),
                        "Unexpected native registry invalidation entry")
                receipt = verify(read(path))
                require(all(receipt.get(k) == v for k, v in self.binding.items()),
                        "Native registry invalidation binding changed")
            require(False, "Native registry verification failed; run permanently invalidated, start a new run")
        self.prior_wall_seconds = 0.0
        self.prior_forward_calls = 0
        self.forward_calls = 0
        self.started_at = None
        self.started_receipt = None
        self.last_finished = None
        numbers = []
        if self.directory.exists():
            for path in self.directory.iterdir():
                match = re.fullmatch(r"([0-9]{6})\.(STARTED|finished)\.json", path.name)
                require(match is not None and path.is_file(), "Unexpected native-budget ledger entry")
                if match[2] == "STARTED":
                    numbers.append(int(match[1]))
            numbers.sort()
            require(numbers == list(range(1, len(numbers) + 1)), "Native-budget attempt sequence changed")
            require(len(list(self.directory.iterdir())) == 2 * len(numbers),
                    "Interrupted native-budget attempt; consumption unknown, resume blocked")
        for number in numbers:
            started = verify(read(self.directory / f"{number:06d}.STARTED.json"))
            finished_path = self.directory / f"{number:06d}.finished.json"
            require(finished_path.is_file(), "Interrupted native-budget attempt; consumption unknown, resume blocked")
            finished = verify(read(finished_path))
            require(all(started.get(k) == v for k, v in self.binding.items()), "Native-budget binding changed")
            require(started.get("attempt") == number and finished.get("attempt") == number
                    and finished.get("started_sha256") == started["sha256"], "Native-budget receipt linkage changed")
            require(started.get("prior_wall_seconds") == self.prior_wall_seconds
                    and started.get("prior_forward_calls") == self.prior_forward_calls, "Native-budget consumption changed")
            wall, calls = finished.get("wall_seconds"), finished.get("forward_calls")
            require(type(wall) in {int, float} and math.isfinite(wall) and wall >= 0,
                    "Invalid native-budget wall consumption")
            require(type(calls) is int and calls >= 0, "Invalid native-budget forward consumption")
            self.prior_wall_seconds += wall
            self.prior_forward_calls += calls
            require(finished.get("total_wall_seconds") == self.prior_wall_seconds
                    and finished.get("total_forward_calls") == self.prior_forward_calls, "Native-budget totals changed")
            self.last_finished = finished
        self.attempt = len(numbers) + 1
        if require_remaining:
            require(self.prior_wall_seconds < self.binding["max_native_wall_seconds"], "Aggregate native wall budget exhausted")
            require(self.prior_forward_calls < self.binding["max_native_forward_calls"], "Aggregate native forward budget exhausted")

    @classmethod
    def inspect_complete(cls, run: Path, split: str, manifest: dict) -> dict:
        """Validate finished accounting without starting or resetting an attempt."""
        ledger = cls(run, split, manifest, require_remaining=False)
        require(ledger.last_finished is not None, "No completed native-budget receipt")
        require(ledger.last_finished.get("status") == "COMPLETE", "Latest native-budget attempt did not complete")
        require(ledger.prior_wall_seconds <= ledger.binding["max_native_wall_seconds"],
                "Aggregate native wall budget exceeded")
        require(ledger.prior_forward_calls <= ledger.binding["max_native_forward_calls"],
                "Aggregate native forward budget exceeded")
        return ledger.last_finished

    def __enter__(self):
        self.started_at = time.monotonic()
        self.started_receipt = signed({**self.binding, "attempt": self.attempt,
                "prior_wall_seconds": self.prior_wall_seconds, "prior_forward_calls": self.prior_forward_calls,
                "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                "wall_limit": "cooperative; one in-flight load/forward can exceed deadline"})
        write_new(self.directory / f"{self.attempt:06d}.STARTED.json", self.started_receipt)
        return self

    def snapshot(self) -> dict:
        require(self.started_at is not None, "Native budget attempt is not started")
        elapsed = time.monotonic() - self.started_at
        return {"attempt": self.attempt, "wall_seconds": elapsed, "forward_calls": self.forward_calls,
                "total_wall_seconds": self.prior_wall_seconds + elapsed,
                "total_forward_calls": self.prior_forward_calls + self.forward_calls}

    def check_time(self) -> None:
        require(self.snapshot()["total_wall_seconds"] < self.binding["max_native_wall_seconds"],
                "Aggregate native wall budget exhausted")

    def before_forward(self, module=None, inputs=None) -> None:
        """A pre-hook on the HF causal model; PEFT generation delegates there."""
        self.check_time()
        require(self.prior_forward_calls + self.forward_calls < self.binding["max_native_forward_calls"],
                "Aggregate native forward budget exhausted")
        self.forward_calls += 1

    def invalidate_registry(self, stage: str, exc: BaseException) -> None:
        """Preserve a permanent barrier when scored assets cannot be verified.

        Restoring files cannot prove which bytes a failed attempt scored. Old
        predictions remain for audit, but this split cannot be resumed or selected.
        """
        require(self.started_receipt is not None or self.last_finished is not None,
                "Registry invalidation needs an accounted native attempt")
        active = self.started_receipt is not None
        attempt = self.attempt if active else self.last_finished["attempt"]
        started_sha256 = self.started_receipt["sha256"] if active else self.last_finished["started_sha256"]
        receipt = signed({**self.binding, "attempt": attempt, "started_sha256": started_sha256,
                          "stage": stage, "reason": f"{type(exc).__name__}: {exc}",
                          "meaning": "Permanent invalidation; restoring registered assets cannot rehabilitate scored receipts"})
        write_new(self.invalidation_directory / f"{attempt:06d}.json", receipt)

    def __exit__(self, exc_type, exc, traceback):
        receipt = {**self.snapshot(), "started_sha256": self.started_receipt["sha256"],
                   "status": "COMPLETE" if exc is None else "BLOCKED" if isinstance(exc, Blocked) else "FAILED",
                   "reason": None if exc is None else f"{type(exc).__name__}: {exc}"}
        write_new(self.directory / f"{self.attempt:06d}.finished.json", signed(receipt))
        return False
