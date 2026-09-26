"""Run provenance, resource accounting and canonical-checkpoint protection.

Conventions follow ``experiments/anchor_routed_middle_recurrence_v1/src/armr/provenance.py``:
exclusive-create JSON writes, LF newlines, ``ensure_ascii=False``, and a
manifest that pins git state plus the SHA-256 of every source file that ran.

The canonical Supa-A1.1 checkpoint is read-only for this lab. ``checkpoint_files``
is hashed before and after every stage and compared; any difference is a hard
failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers

EXPERIMENT = Path(__file__).resolve().parents[2]
REPO = EXPERIMENT.parents[1]

# Large weights are gitignored, so a worktree never carries them. The canonical
# core lives only in the primary checkout.
_VAULT = REPO.parent / "Chat-Bubble" / "Rineya_Published_Cores"

CANONICAL_CORE = _VAULT / "Rineya-Supa-A1.1-1.1.0-devfreeze"
STATE_HASH_ALGORITHM = "sha256-native-state-v1"
FROZEN_HASH_ALGORITHM = "sha256-native-frozen-parameters-v1"

# Both A-series cores carry the SAME baked gate -- identical
# abstain_token_id 51378 and abstain_logit_bias 5.156249985098839, and a
# byte-identical modeling file. A1 is not an ungated model; it is the bfloat16
# store of the same recipe. Removing the gate is a separate axis, applied
# in memory at load time and never written to disk.
CORES = {
    "a11": {
        "path": CANONICAL_CORE,
        "dtype": "float16",
        "label": "Supa-A1.1 (float16 store)",
        "modeling_file": "modeling_supa_a11.py",
    },
    "a1": {
        "path": _VAULT / "Rineya-Supa-A1-1.0.0-devfreeze",
        "dtype": "bfloat16",
        "label": "Supa-A1 (bfloat16 store)",
        "modeling_file": "modeling_supa_a1.py",
    },
}

DEMO_SUITE = REPO / "Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl"

CHECKPOINT_FILE_NAMES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "modeling_supa_a11.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "CORE.json",
    "FILES.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value: Any) -> Any:
    """Replace inf/nan with a tagged string so a result file always survives.

    ``allow_nan=False`` is kept deliberately -- emitting bare ``Infinity`` would
    produce JSON that other tools reject. But one non-finite metric must not
    truncate a whole run's evidence, which is what happened when a float16
    scaler overflow put ``inf`` into a gradient norm.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return f"NON_FINITE:{value}"
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_finite(value), ensure_ascii=False, indent=2, allow_nan=False)
    if overwrite:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(payload + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    else:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(payload + "\n")
            f.flush()
            os.fsync(f.fileno())


def append_jsonl(path: Path, record: dict) -> None:
    """Durable observability event; checkpoints remain the transaction authority."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_finite(record), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class ExclusiveFileLock:
    """A process-owned OS lock, automatically released on process death.

    The persistent file is deliberately never unlinked. Its presence or PID is
    not ownership evidence; unlinking would let contenders lock different inodes.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(f"another process owns the run lock: {self.path}") from exc
        self.stream = stream
        return self

    def __exit__(self, *exc):
        if self.stream is not None:
            try:
                self.stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None
        return False


def write_jsonl(path: Path, rows: list[dict], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_jsonl_prefix(path: Path) -> list[dict]:
    """Read a killed writer's intact prefix; preserve its torn final line on disk."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith(("\n", "\r")):
                raise
    return rows


def checkpoint_files(path: Path = CANONICAL_CORE) -> dict[str, str]:
    names = set(CHECKPOINT_FILE_NAMES) | {c["modeling_file"] for c in CORES.values()}
    return {name: sha256(path / name) for name in sorted(names) if (path / name).is_file()}


def state_hash(model: Any, chunk_elements: int = 8 << 20) -> str:
    """Hash of every parameter value. Used to prove a frozen model was frozen.

    Streams in chunks and hashes the parameters in their *native* dtype. An
    earlier version upcast each tensor to float32 on CPU and then called
    ``.numpy().tobytes()``, which materialises two full copies at once -- about
    1.2 GB for the tied embedding alone. On a 16 GB laptop already sharing RAM
    with another experiment, that was enough to page the process out and stall
    it. The version is recorded in each new manifest; older algorithm digests
    must not be compared across runs.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            piece = flat[start : start + chunk_elements].to("cpu").contiguous()
            h.update(memoryview(piece.view(torch.uint8).numpy()))
            del piece
    return h.hexdigest()


def frozen_parameter_hash(model: Any, chunk_elements: int = 8 << 20) -> str:
    """Hash frozen parameters within an adapter run, excluding trainable LoRA."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        if tensor.requires_grad:
            continue
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            piece = flat[start:start + chunk_elements].to("cpu").contiguous()
            h.update(memoryview(piece.view(torch.uint8).numpy()))
    return h.hexdigest()


def trainable_parameter_receipt(model: Any) -> dict:
    """Native values, exact names, shapes and dtypes of the actual initializer."""
    h = hashlib.sha256()
    parameters = []
    for name, tensor in sorted(model.named_parameters()):
        if not tensor.requires_grad:
            continue
        metadata = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        parameters.append(metadata)
        h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        h.update(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return {"algorithm": "sha256-native-trainable-parameters-v1", "sha256": h.hexdigest(),
            "parameters": parameters}


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    except Exception:  # pragma: no cover - provenance must never break a run
        return "UNAVAILABLE"


def gpu_summary() -> dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {"available": True, "name": torch.cuda.get_device_name(0),
                "total_bytes": int(total), "free_bytes_at_start": int(free),
                "used_by_others_bytes": int(total - free)}
    except Exception as exc:
        return {"available": None, "measurement_error": f"{type(exc).__name__}: {exc}"}


def source_hashes() -> dict[str, str]:
    files = sorted(
        [*EXPERIMENT.glob("src/**/*.py"), *EXPERIMENT.glob("scripts/*.py"),
         *EXPERIMENT.glob("configs/*.json")]
    )
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha256(p) for p in files}


class Run:
    """One stage of the pilot: manifest in, resource accounting out."""

    def __init__(self, stage: str, config: dict, *, results_dir: Path | None = None,
                 core_path: Path | None = None, refuse_nonempty: bool = False):
        self.stage = stage
        self.config = config
        self.core_path = core_path or CANONICAL_CORE
        self.started_utc = datetime.now(timezone.utc)
        self.out = (results_dir or (EXPERIMENT / "results")) / stage
        if refuse_nonempty and self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"refusing nonempty stage directory: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        self.checkpoint_before = checkpoint_files(self.core_path)
        self.model_loads = 0
        self.model_load_attempts = 0
        self.resource_extra: dict[str, Any] = {}
        self._t0 = None

    def __enter__(self) -> "Run":
        import time

        self.metric_errors = {}
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            self.metric_errors["reset_gpu_peaks"] = f"{type(exc).__name__}: {exc}"
        self._t0 = time.perf_counter()
        self.manifest = {
            "status": "STARTED",
            "stage": self.stage,
            "start_utc": self.started_utc.isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "git_status": git("status", "--short"),
            "config": self.config,
            "canonical_core": str(self.core_path),
            "checkpoint_sha256_before": self.checkpoint_before,
            "source_sha256": source_hashes(),
            "state_hash_algorithm": STATE_HASH_ALGORITHM,
            "frozen_parameter_hash_algorithm": FROZEN_HASH_ALGORITHM,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "os": platform.platform(),
            "logical_cpus": psutil.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "gpu": gpu_summary(),
        }
        write_json(self.out / "started_manifest.json", self.manifest, overwrite=True)
        return self

    def model_load_started(self) -> None:
        if self.model_load_attempts:
            raise RuntimeError("a physical segment may load the base model only once")
        self.model_load_attempts = 1
        self._load_t0 = time.perf_counter()
        write_json(self.out / "model_load.json", {
            "status": "LOADING", "start_utc": datetime.now(timezone.utc).isoformat(),
            "model_load_attempts": 1, "model_loads": 0, "model_loads_upper_bound": 1,
        }, overwrite=True)

    def model_load_completed(self) -> None:
        self.model_loads = 1
        previous = json.loads((self.out / "model_load.json").read_text(encoding="utf-8"))
        write_json(self.out / "model_load.json", {
            **previous, "status": "LOADED", "end_utc": datetime.now(timezone.utc).isoformat(),
            "model_loads": 1, "model_loads_upper_bound": 1,
            "model_load_seconds": round(time.perf_counter() - self._load_t0, 3),
        }, overwrite=True)

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A stage that raised still records what it consumed and still proves the
        # canonical checkpoint survived. finish() is idempotent enough to be
        # called explicitly on the success path.
        if exc_type is not None and self.manifest.get("status") == "STARTED":
            self.finish("FAILED", {"error": f"{exc_type.__name__}: {exc}"})
        return False

    def finish(self, status: str, decision: dict, *, resource_overrides: dict | None = None) -> dict:
        import time

        after = checkpoint_files(self.core_path)
        unchanged = after == self.checkpoint_before
        # Disk hashing is CPU-only and precedes all potentially poisoned CUDA
        # operations. Even a sticky device assert cannot erase this receipt.
        write_json(self.out / "checkpoint_integrity.json", {
            "checkpoint_sha256_before": self.checkpoint_before,
            "checkpoint_sha256_after": after, "canonical_checkpoint_unchanged": unchanged,
            "checked_utc": datetime.now(timezone.utc).isoformat(),
        }, overwrite=True)

        def measure(name, callback):
            try:
                return int(callback())
            except Exception as exc:
                self.metric_errors[name] = f"{type(exc).__name__}: {exc}"
                return None

        def rss_peak():
            memory = self.process.memory_info()
            return getattr(memory, "peak_wset", memory.rss)

        def gpu_peak(callback):
            return callback() if torch.cuda.is_available() else 0

        resource = {
            "label": self.stage,
            "wall_seconds": round(time.perf_counter() - self._t0, 3),
            "peak_rss_bytes": measure("peak_rss_bytes", rss_peak),
            "peak_gpu_allocated_bytes": measure("peak_gpu_allocated_bytes", lambda: gpu_peak(torch.cuda.max_memory_allocated)),
            "peak_gpu_reserved_bytes": measure("peak_gpu_reserved_bytes", lambda: gpu_peak(torch.cuda.max_memory_reserved)),
            "model_loads": self.model_loads,
            "model_load_attempts": self.model_load_attempts,
            "metric_errors": self.metric_errors,
            "resource_accounting_complete": not bool(self.metric_errors),
            **self.resource_extra,
        }
        if resource_overrides is not None:
            resource["aggregation_wall_seconds"] = resource["wall_seconds"]
            resource.update(resource_overrides)
        self.manifest.update(
            status=status if unchanged else "FAILED",
            end_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint_sha256_after=after,
            canonical_checkpoint_unchanged=unchanged,
            resource=resource,
            decision=decision,
        )
        write_json(self.out / "run_manifest.json", self.manifest, overwrite=True)
        write_json(self.out / "resource.json", resource, overwrite=True)
        if not unchanged:
            raise RuntimeError(
                "CANONICAL CHECKPOINT MUTATED -- this lab is read-only against "
                f"{self.core_path}"
            )
        return resource
