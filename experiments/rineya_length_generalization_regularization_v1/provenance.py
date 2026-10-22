"""Content-bound receipts, create-only outputs, and fail-closed native resources."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from .data import digest


def file_sha256(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Expected an ordinary local file")
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_fingerprint(directory):
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink() or not (directory / "config.json").is_file():
        raise ValueError("An exact local Hugging Face checkpoint directory is required")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Native lane requires safetensors; no pickle checkpoint loading")
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Checkpoint symlinks are not qualified; materialize an isolated snapshot")
    return digest({str(p.relative_to(directory)): file_sha256(p) for p in files})


def code_fingerprint():
    root = Path(__file__).parent
    paths = sorted(list(root.rglob("*.py")) + list((root / "configs").glob("*.json")))
    return digest({str(p.relative_to(root)): file_sha256(p) for p in paths})


def write_json_new(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)


def runtime_identity(torch):
    from importlib.metadata import version
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": version("transformers"), "safetensors": version("safetensors"),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0))}


def binding(checkpoint, bank, plan, eligibility, runtime):
    eligibility = json.loads(Path(eligibility).read_text(encoding="utf-8"))
    data_hash = file_sha256(bank)
    if eligibility.get("bank_sha256") != data_hash or any(eligibility.get(k) is not True for k in (
            "train_eligible", "quarantine_excluded", "cross_split_parent_and_prompt_audit_passed")):
        raise ValueError("Missing/mismatched data-eligibility attestation")
    # Attestations are operator assertions, not cryptographic proof of decontamination.
    return {"checkpoint_sha256": checkpoint_fingerprint(checkpoint), "bank_sha256": data_hash,
            "plan_sha256": digest(plan.to_dict()), "code_sha256": code_fingerprint(),
            "eligibility_sha256": digest(eligibility), "runtime": runtime}


def verify_receipt(receipt, expected_binding, *, now=None):
    now = now or datetime.now(timezone.utc)
    if receipt.get("status") != "READY_FOR_EXPLICIT_BOUNDED_TRAINING" or receipt.get("binding") != expected_binding:
        raise ValueError("Missing/stale native qualification receipt")
    try:
        created = datetime.fromisoformat(receipt["created_at"])
        age = (now - created).total_seconds()
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Malformed qualification timestamp") from exc
    if not 0 <= age <= 24 * 3600:
        raise ValueError("Qualification receipt expired or is future-dated")
    checks = ("identity_parity", "finite_backward", "serialization_parity", "merge_parity")
    if any(receipt.get("checks", {}).get(k) is not True for k in checks):
        raise ValueError("Incomplete native qualification")


@contextmanager
def exclusive_run(output_dir):
    """Create-only output directory. Failed attempts are never silently reused."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    lock = output / "RUN_LOCK.json"
    write_json_new(lock, {"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()})
    try:
        yield output
    finally:
        # Retain the receipt/lock: a completed or failed run ID stays spent.
        pass


class NativeGuard:
    """Single local CUDA device, host RAM/RSS, headroom, and foreign-job checks.

    Unsupported nvidia-smi/psutil telemetry blocks native execution. These are
    conservative research controls, not a cluster scheduler or security boundary.
    """
    def __init__(self):
        import psutil
        import torch
        self.psutil, self.torch = psutil, torch
        self.process = psutil.Process()

    def _foreign_jobs(self):
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=10)
        own = {self.process.pid} | {p.pid for p in self.process.children(recursive=True)}
        pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        if pids - own:
            raise RuntimeError("Foreign GPU compute processes detected; do not interrupt them")

    def before_load(self):
        if not self.torch.cuda.is_available() or self.torch.cuda.device_count() != 1:
            raise RuntimeError("This native lane requires exactly one visible CUDA GPU")
        for _ in range(2):
            if self.psutil.virtual_memory().available < 7400 * 1024**2:
                raise RuntimeError("Pre-load free host RAM below 7400 MiB")
            time.sleep(0.25)
        self._foreign_jobs()

    def __call__(self):
        if self.psutil.virtual_memory().available < 6144 * 1024**2:
            raise RuntimeError("Runtime free host RAM below 6144 MiB")
        processes = [self.process] + self.process.children(recursive=True)
        if sum(p.memory_info().rss for p in processes) > 4096 * 1024**2:
            raise RuntimeError("Process-tree RSS exceeds 4096 MiB")
        free, _ = self.torch.cuda.mem_get_info()
        reusable = self.torch.cuda.memory_reserved() - self.torch.cuda.memory_allocated()
        if free + reusable < 256 * 1024**2:
            raise RuntimeError("CUDA headroom below 256 MiB")
        self._foreign_jobs()
