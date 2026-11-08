"""RQSFT custody/identity. Pure helpers confer no scientific execution authority."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import uuid

from experiments.model_quality_gpu_v1.integrity import (
    canonical, decode, digest, seal, unseal,
)
from . import BASE_COMMIT
from .contract import external_output

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
BRANCH = "codex/rineya-quality-sft-v1"


def materialized(path):
    target = Path(path).absolute()
    if ".." in target.parts:
        raise ValueError("Traversal paths are forbidden")
    for ancestor in (target, *target.parents):
        if ancestor.is_symlink():
            raise ValueError("Materialized paths required; symlinks forbidden")
        if ancestor.exists():
            attributes = getattr(ancestor.lstat(), "st_file_attributes", 0)
            if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError("Materialized paths required; Windows reparse points forbidden")
    return target


check_input_path = materialized


def regular(path):
    target = materialized(path)
    if not target.is_file():
        raise ValueError("Missing regular file: " + target.name)
    return target


def file_hash(path):
    h = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, limit=32 * 1024 * 1024):
    target = regular(path)
    if target.stat().st_size > limit:
        raise ValueError("JSON exceeds bounded reader limit")
    return decode(target.read_text(encoding="utf-8"))


def write_new(path, value):
    target = materialized(path)
    with target.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def new_dir(path):
    target = external_output(materialized(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def code_identity():
    files = {}
    dependency = ROOT / "experiments" / "model_quality_gpu_v1"
    for folder in (PACKAGE, dependency):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not set(path.parts) & {"__pycache__", ".pytest_cache", "evidence", "outputs"}:
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(regular(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"sha256": digest(files), "files": files}


def checkout_identity(require_clean=False):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT or git("branch", "--show-current") != BRANCH:
        raise ValueError("RQSFT native work requires its dedicated full checkout")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], check=True, capture_output=True)
    changes = git("status", "--porcelain", "--untracked-files=all")
    if require_clean and changes:
        raise ValueError("Native RQSFT work requires a clean committed reviewed checkout")
    return {"branch": BRANCH, "head": git("rev-parse", "HEAD"), "base": BASE_COMMIT,
            "clean": not bool(changes)}


@contextmanager
def external_caches(directory):
    root = Path(directory) / "caches"
    temp = Path(directory) / "temp"
    root.mkdir(exist_ok=False)
    temp.mkdir(exist_ok=False)
    values = {"HF_HOME": str(root / "hf"), "HF_HUB_CACHE": str(root / "hf" / "hub"),
              "HUGGINGFACE_HUB_CACHE": str(root / "hf" / "hub"), "TRANSFORMERS_CACHE": str(root / "transformers"),
              "TORCH_HOME": str(root / "torch"), "TORCH_EXTENSIONS_DIR": str(root / "extensions"),
              "CUDA_CACHE_PATH": str(root / "cuda"), "XDG_CACHE_HOME": str(root),
              "TRITON_CACHE_DIR": str(root / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
              "TMP": str(temp), "TEMP": str(temp)}
    previous = {k: os.environ.get(k) for k in values}
    old_bytecode, old_temp = sys.dont_write_bytecode, tempfile.tempdir
    try:
        os.environ.update(values)
        sys.dont_write_bytecode, tempfile.tempdir = True, str(temp)
        yield
    finally:
        sys.dont_write_bytecode, tempfile.tempdir = old_bytecode, old_temp
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def gpu_lease(path=None):
    # RQSFT cooperates through its fixed BusinessProject lock. Legacy MQ's temp
    # lock is checked separately before launch; it is never created/removed here.
    target = external_output(materialized(path or "C:/Users/clclc/BusinessProject/rineya-gpu-0.lock"))
    token = uuid.uuid4().hex
    write_new(target, {"experiment": "RQSFT-001", "pid": os.getpid(), "token": token})
    try:
        yield
    finally:
        if target.exists() and read_json(target).get("token") == token:
            target.unlink()
