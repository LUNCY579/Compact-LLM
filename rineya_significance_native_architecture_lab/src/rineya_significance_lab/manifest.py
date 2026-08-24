"""Auditable run manifests and atomic JSON writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_snapshot(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        return completed.stdout.strip()

    return {
        "repo_root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "remote": run("remote", "get-url", "origin"),
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependency_versions(
            ["numpy", "torch", "transformers", "PyYAML", "scipy", "scikit-learn", "matplotlib"]
        ),
    }


def build_manifest(
    *,
    repo: str | Path,
    config: Mapping[str, Any],
    run_id: str,
    execution_mode: str,
    model: Mapping[str, Any] | None = None,
    benchmark: Mapping[str, Any] | None = None,
    architecture_variant: str,
    seed: int,
    compute_tier: str = "A",
    status: str = "RUNNING",
    evidence_class: str = "NEW_CHATBUBBLE_EVIDENCE",
    resolved_config_path: str = "resolved_config.yaml",
) -> dict[str, Any]:
    """Build a manifest conforming to ``schemas/manifest.schema.json``."""

    snapshot = git_snapshot(repo)
    environment = environment_snapshot()
    dependencies = environment["dependencies"]
    model_value: dict[str, Any] = {
        "status": "NOT_APPLICABLE",
        "id": None,
        "path": None,
        "architecture": None,
        "hashes": {"weights": None, "tokenizer": None, "config": None, "chat_template": None},
    }
    model_value.update(dict(model or {}))
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    benchmark_value = dict(benchmark or {})
    query_conditioned = config.get("query", {}).get("mode") == "conditioned"
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "evidence_class": evidence_class,
        "created_at": utc_now(),
        "completed_at": None,
        "repositories": {
            "chatbubble": {
                "remote": snapshot["remote"],
                "commit": snapshot["commit"],
                "dirty": snapshot["dirty"],
            },
            "source": {
                "remote": "https://github.com/LUNCY579/RineyaBeta.git",
                "implementation": {
                    "ref": "refs/heads/main",
                    "commit": "9e00be74098878290dc2b0d54cbffee4652c6088",
                    "path": "scripts/kv/significance.py",
                    "blob": "30ee29724dce52de5a288f6421123fb718240541",
                },
                "paper": {
                    "ref": "refs/heads/r2-healing-harness",
                    "commit": "1664f54de0bbdfde45e311b75bbf3038dbfffe42",
                    "path": "paper/significance_matrix/main.tex",
                    "blob": "b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3",
                },
            },
        },
        "experiment": {
            "name": str(config.get("experiment", {}).get("name", run_id)),
            "compute_tier": compute_tier,
            "architecture_variant": architecture_variant,
            "benchmark": benchmark_value.get("name"),
            "seed": int(seed),
            "query_conditioned": query_conditioned,
            "execution_mode": execution_mode.upper(),
        },
        "config": {
            "resolved_path": resolved_config_path,
            "sha256": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        },
        "model": model_value,
        "runtime": {
            "python": environment["python"],
            "torch": dependencies.get("torch"),
            "transformers": dependencies.get("transformers"),
            "device": str(config.get("runtime", {}).get("device", "cpu")),
            "precision": config.get("runtime", {}).get("dtype"),
            "hardware": environment,
        },
        "resolved_config": dict(config),
        "artifacts": {},
        "notes": ["Cornerstone-1 only; inherited source evidence is not a new result."],
    }


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
