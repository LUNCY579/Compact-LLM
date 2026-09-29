"""Strict public/private contracts, immutable files, and source fingerprints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
VERSION = "rvs001-implementation-1"
BRIDGE = "experiments/post_training_mimo_centralized_v1/scripts/bridge_data.py"
WATCHDOG = "experiments/opallo_native_collector_v1/supervisor.py"
BRIDGE_BLOB = "2c17f08e35bf13ca5baf3888c01df55b0138b9c1"
WATCHDOG_BLOB = "dc8f25bfbed2f7f2159ada79a9fec55622f8010d"


class Blocked(RuntimeError):
    """A failed contract or resource gate; never silently repair the run."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Blocked(message)


def integer(x: Any, lo: int, hi: int, name: str) -> int:
    require(type(x) is int and lo <= x <= hi, f"invalid {name}")
    return x


def exact(obj: Any, keys: set[str], name: str = "object") -> None:
    require(type(obj) is dict and set(obj) == keys, f"unexpected/missing {name} fields")


def _pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise Blocked("nonfinite JSON: " + value)


def loads(text: str | bytes) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise Blocked("invalid JSON") from exc


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def blob_sha(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def matches_git_blob(path: Path, expected: str) -> bool:
    """Accept Git's Windows CRLF checkout conversion only; hash active bytes separately."""
    data = Path(path).read_bytes()
    for candidate in (data, data.replace(b"\r\n", b"\n")):
        identity = hashlib.sha1(b"blob " + str(len(candidate)).encode() + b"\0" + candidate).hexdigest()
        if identity == expected:
            return True
    return False


def directory_bytes(root: Path) -> int:
    """Observed regular-file bytes; caller's output directory must not contain symlinks."""
    total = 0
    for path in Path(root).rglob("*"):
        require(not path.is_symlink(), "linked output path prohibited")
        if path.is_file():
            total += path.stat().st_size
    return total


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(encoded(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path: Path) -> Any:
    require(Path(path).stat().st_size <= 16 * 1024**2, "JSON file too large")
    return loads(Path(path).read_bytes())


def read_jsonl(path: Path, limit: int = 4096) -> list[dict]:
    path = Path(path)
    require(path.stat().st_size <= 64 * 1024**2, "catalog too large")
    result = []
    with path.open("rb") as f:
        for line in f:
            require(len(line) <= 1024**2 and line.endswith(b"\n"), "oversize/torn JSONL row")
            require(len(result) < limit, "too many rows")
            result.append(loads(line))
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with Path(path).open("xb") as f:
        for row in rows:
            f.write(encoded(row) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def source_identity(native: bool = False) -> dict:
    files = {p.relative_to(ROOT).as_posix(): file_sha(p)
             for p in sorted(PACKAGE.glob("*.py"))}
    for rel in (BRIDGE, WATCHDOG, "experiments/opallo_native_collector_v1/contract.py"):
        p = ROOT / rel
        if p.exists():
            files[rel] = file_sha(p)
        elif native:
            raise Blocked("missing inherited dependency: " + rel)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, "not-a-git-checkout"
    if native:
        require(head is not None and not dirty, "native execution requires a clean committed checkout")
        require(matches_git_blob(ROOT / BRIDGE, BRIDGE_BLOB), "centralized bridge source drift")
        require(matches_git_blob(ROOT / WATCHDOG, WATCHDOG_BLOB), "watchdog source drift")
    return {"version": VERSION, "files": files, "code_digest": digest(files),
            "git_head": head, "dirty": bool(dirty)}


def validate_config(c: dict) -> dict:
    exact(c, {"schema", "seed", "samples", "budgets", "temperature", "max_prompt_tokens",
              "max_new_tokens", "max_context_tokens", "cache", "segment_attempts",
              "segment_seconds", "total_wall_seconds", "max_generated_tokens",
              "max_output_bytes", "bootstrap_replicates"}, "config")
    require(c["schema"] == VERSION, "config schema drift")
    integer(c["seed"], 0, 2**63-1, "seed")
    integer(c["samples"], 1, 64, "samples")
    require(type(c["budgets"]) is list and bool(c["budgets"]), "empty budgets")
    for k in c["budgets"]:
        integer(k, 1, c["samples"], "budget")
    require(c["budgets"] == sorted(set(c["budgets"])) and c["budgets"][0] == 1
            and c["budgets"][-1] == c["samples"], "budgets must include 1 and n")
    require(type(c["temperature"]) in (int, float) and math.isfinite(c["temperature"])
            and 0 < c["temperature"] <= 2, "invalid temperature")
    for name, lo, hi in [("max_prompt_tokens", 8, 896), ("max_new_tokens", 1, 128),
                         ("max_context_tokens", 16, 1024), ("segment_attempts", 1, 128),
                         ("segment_seconds", 30, 1800), ("total_wall_seconds", 30, 21600),
                         ("max_generated_tokens", 1, 2**24), ("max_output_bytes", 65536, 2**30),
                         ("bootstrap_replicates", 0, 2000)]:
        integer(c[name], lo, hi, name)
    require(c["max_prompt_tokens"] + c["max_new_tokens"] <= c["max_context_tokens"], "context cap")
    require(type(c["cache"]) is bool, "cache must be boolean")
    return c


def validate_public(rows: list[dict]) -> None:
    require(0 < len(rows) <= 1024, "public population must have 1..1024 rows")
    ids, prompts = set(), set()
    for r in rows:
        exact(r, {"id", "messages"}, "public task")
        require(type(r["id"]) is str and re.fullmatch(r"[0-9a-f]{64}", r["id"]) is not None,
                "public ID must be opaque hash")
        require(r["id"] not in ids, "duplicate public ID")
        ids.add(r["id"])
        require(type(r["messages"]) is list and 1 <= len(r["messages"]) <= 32, "messages count")
        for m in r["messages"]:
            exact(m, {"role", "content"}, "message")
            require(type(m["role"]) is str and m["role"] in {"system", "user", "assistant"} and type(m["content"]) is str
                    and len(m["content"]) <= 65536, "invalid message")
        require(r["messages"][-1]["role"] == "user", "last message must be user")
        key = digest(r["messages"])
        require(key not in prompts, "duplicate public prompt")
        prompts.add(key)


def parse_answer(text: str) -> str | None:
    """Label-blind canonical JSON. None means invalid; empty strings/lists are valid."""
    try:
        require(type(text) is str and len(text) <= 4096, "answer too large")
        value = loads(text)
        exact(value, {"answer"}, "answer envelope")
        answer = value["answer"]
        # Only the types used by the four deterministic bridge domains.
        require(type(answer) in (int, str, list), "unsupported answer type")
        if type(answer) is list:
            require(len(answer) <= 128 and all(type(v) is int for v in answer), "noninteger answer list")
        return encoded(value).decode("utf-8")
    except (Blocked, ValueError, TypeError, RecursionError):
        return None


def seed_for(seed: int, task: str, arm: str, index: int) -> int:
    return int(digest([VERSION, seed, task, arm, index])[:16], 16) % (2**63-1)
