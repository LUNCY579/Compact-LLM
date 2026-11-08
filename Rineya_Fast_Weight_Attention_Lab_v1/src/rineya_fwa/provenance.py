"""Minimal provenance sidecar. Never writes into published cores."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rineya_fwa.paths import REPO_ROOT, assert_readonly_target


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip()


def stamp(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "config": config or {},
        "config_sha256": hashlib.sha256(
            json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def write_json(path: Path, payload: dict[str, Any], *, config: dict[str, Any] | None = None) -> None:
    assert_readonly_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(stamp(config), indent=2) + "\n", encoding="utf-8")
