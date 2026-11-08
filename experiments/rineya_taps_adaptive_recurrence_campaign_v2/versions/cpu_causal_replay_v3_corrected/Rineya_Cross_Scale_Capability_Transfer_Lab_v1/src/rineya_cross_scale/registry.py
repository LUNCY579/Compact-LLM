from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .hashing import canonical_sha256


class RegistryError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def alpha_label(alpha: float) -> str:
    scaled = int(round(abs(float(alpha)) * 10_000))
    prefix = "AN" if alpha < 0 else "A"
    return f"{prefix}{scaled:04d}"


def _clean(value: str) -> str:
    result = "".join(character for character in value.upper() if character.isalnum())
    if not result:
        raise RegistryError(f"empty identifier component from {value!r}")
    return result


@dataclass(frozen=True)
class ExperimentSpec:
    donor: str
    projection_strategy: str
    depth_mapping: str
    width_mapping: str
    merge_family: str
    coefficient: float
    seed: int
    output_dtype: str
    source_hashes: dict[str, str]
    layer_mask: tuple[int, ...] = ()
    module_mask: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return canonical_sha256(asdict(self))

    @property
    def experiment_id(self) -> str:
        stem = "_".join(
            (
                "CSM",
                _clean(self.projection_strategy),
                _clean(self.depth_mapping),
                _clean(self.width_mapping),
                _clean(self.donor),
                _clean(self.merge_family),
                alpha_label(self.coefficient),
                f"S{self.seed}",
            )
        )
        return f"{stem}_{self.digest[:8].upper()}"


@contextmanager
def registry_lock(path: Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RegistryError(f"timed out acquiring registry lock: {lock}")
            time.sleep(0.05)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\ncreated={utc_now()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


class ExperimentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        result = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RegistryError(
                        f"invalid registry JSON at line {line_number}"
                    ) from error
                result.append(value)
        return result

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def create(self, spec: ExperimentSpec) -> str:
        experiment_id = spec.experiment_id
        with registry_lock(self.path):
            events = self.events()
            existing = [
                event
                for event in events
                if event.get("experiment_id") == experiment_id
                and event.get("event") == "created"
            ]
            if existing:
                if existing[0].get("spec_digest") != spec.digest:
                    raise RegistryError(
                        f"experiment ID collision for {experiment_id}"
                    )
                return experiment_id
            self._append(
                {
                    "schema_version": "1.0",
                    "at_utc": utc_now(),
                    "event": "created",
                    "experiment_id": experiment_id,
                    "spec_digest": spec.digest,
                    "spec": asdict(spec),
                    "status": "registered",
                }
            )
        return experiment_id

    def status(self, experiment_id: str, status: str, **details: Any) -> None:
        allowed = {
            "building",
            "built",
            "validating",
            "valid",
            "evaluating",
            "complete",
            "failed",
            "eliminated",
        }
        if status not in allowed:
            raise RegistryError(f"unsupported experiment status: {status}")
        with registry_lock(self.path):
            created = any(
                event.get("event") == "created"
                and event.get("experiment_id") == experiment_id
                for event in self.events()
            )
            if not created:
                raise RegistryError(f"unknown experiment ID: {experiment_id}")
            self._append(
                {
                    "schema_version": "1.0",
                    "at_utc": utc_now(),
                    "event": "status",
                    "experiment_id": experiment_id,
                    "status": status,
                    "details": details,
                }
            )

    def record_diagnostic(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        transformation: str,
        verdict: str,
        resource_usage: dict[str, Any] | None = None,
        quantitative_result: dict[str, Any] | None = None,
        interpretation: str = "",
        next_action: str = "",
        source_hashes: dict[str, str] | None = None,
    ) -> None:
        """Append a v3+ diagnostic event (PASS / FAIL / INCONCLUSIVE)."""
        if verdict not in {"PASS", "FAIL", "INCONCLUSIVE"}:
            raise RegistryError(f"unsupported diagnostic verdict: {verdict}")
        with registry_lock(self.path):
            self._append(
                {
                    "schema_version": "3.0",
                    "at_utc": utc_now(),
                    "event": "diagnostic",
                    "experiment_id": experiment_id,
                    "hypothesis": hypothesis,
                    "transformation": transformation,
                    "verdict": verdict,
                    "resource_usage": resource_usage or {},
                    "quantitative_result": quantitative_result or {},
                    "interpretation": interpretation,
                    "next_action": next_action,
                    "source_hashes": source_hashes or {},
                }
            )
