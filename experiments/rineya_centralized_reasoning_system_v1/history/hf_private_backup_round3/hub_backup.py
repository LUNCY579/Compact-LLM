"""Bounded CRS-001 private research archive. No model imports or background jobs.

Examples (all preparation is local):
  python hub_backup.py --dry-run
  python hub_backup.py --prepare
  python hub_backup.py --upload --snapshot <content_sha256>
  python hub_backup.py --reconcile --snapshot <content_sha256>

An exception worktree requires --exception-worktree and, when preparing,
--exception-sha256 from a fresh inventory after its owner freezes the work.
Raw, failed, qualified and quarantined observations remain research evidence;
this helper never makes a record training eligible.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
from typing import Callable

REPO_ID = "Luncy1978/rineya-crs-001-research"
PACKAGE = Path("experiments/rineya_centralized_reasoning_system_v1")
PRIMARY = Path("C:/Users/clclc/BusinessProject/ChatBubble-rineya-centralized-reasoning-system-v1")
EXCEPTION = Path("C:/Users/clclc/.codex/worktrees/crs-native-desktop-v2/ChatBubble-rineya-centralized-reasoning-system-v1")
LOCAL_NAME = ".hub_backup_local"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_FILES = 4096
MAX_ENTRIES = 16384
MAX_DEPTH = 24
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
ARCHIVE_BOUND = MAX_TOTAL_BYTES + MAX_FILES * 2048 + 10240
HERMES_PYTHON = Path("C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe")
SDK_ROOT = HERMES_PYTHON.parent.parent / "Lib/site-packages/huggingface_hub"
SDK_VERSION = "1.27.0"
SDK_PINS = {"hf_api.py": "44da5ff83a41741410e7e9588c1f003e545a08e21c48ae255944cc92f4223dc0", "_commit_api.py": "95a96a8492a5f3f456547800767cd9286fdfcb813b7dbc8721ebfa9c32eb8f8c", "utils/_validators.py": "b6e0b5538cb107ee172886a364c9ca65448abd9eaa3e9c9d63c2c8b5f6b4838d"}
COVERAGE = "CRS-001 package source/docs/open fixture/raw/results/quality/failure/quarantined history; root charter and AGENTS provenance only"
EXCLUSION_POLICY = "Never follow JSON paths, import tensors, open databases, traverse links, upload credentials/weights/protected banks/caches/backup-local state, or promote research into training data."
QUALIFICATION = "research_evidence_original_status_retained"
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
TEXT_SUFFIXES = {".py", ".rs", ".md", ".json", ".jsonl", ".txt", ".log", ".csv", ".tsv", ".toml", ".patch", ".yaml", ".yml"}
CACHE_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".cache", LOCAL_NAME}
WEIGHT_DIRS = {"models", "weights", "checkpoints", "adapters", "model_weights", "canonical_weights"}
PROTECTED_DIRS = {"check", "final", "check_bank", "final_bank", "protected_banks", "central_check", "central_final"}
DB_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".sqlite3-wal", ".sqlite3-shm"}
TENSOR_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx", ".gguf", ".npy", ".npz", ".pkl", ".pickle"}
SECRET_NAMES = {"credentials", "credentials.json", "secrets.json", "token", "tokens.json", "id_rsa", "id_ed25519", ".netrc", ".pypirc", "auth.json"}
SECRET_BYTES = re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|AKIA[A-Z0-9]{16})")
# Existing UTF-8 paper extraction contains two literal NULs. Preserve these
# exact provenance bytes; this is not a general permission for binary text.
KNOWN_NUL_TEXT = {"9e70f23c4afe2a9831001390d28556368f3bf2662f99843edcdb6c6ce53c261c"}
# Candidate3: inert whole-byte Microsoft provenance only. Case is significant;
# no generic HTML permission, rendering, URL following or executable content.
KNOWN_HTML = {
    "evidence/cpu_only_tool_version_source_007/windows-apisets.html": "56ca145d908728c3a33d47b3112b43ff8c2306570062f6cf752203e40f2e4004",
    "evidence/cpu_only_tool_version_source_007/dll-search-order.html": "ecf5f31efc88b3d5c1bed7e279b5227dad5711004f155963d2519c8914f23509",
    "evidence/cpu_only_tool_version_source_007/WaitForSingleObjectEx.html": "d0e013e092ec11a829a66f70a6cf0b85be757bb68c656470875e0c61b56c4f8b",
}
CARD = """---
pretty_name: CRS-001 private research preservation archive
task_categories: []
tags:
  - research-artifact-archive
  - not-training-eligible
---
# CRS-001 private research preservation archive

This private dataset repository preserves the complementary Rineya CRS-001
research track. It is an artifact archive, not a training dataset or model
release. Snapshots retain supplied source, research documents, open TRAIN/DEV
fixtures, raw predictions, quality reviews, failed attempts and quarantined
history with their original paths, byte counts and SHA-256 hashes.

All included records have `training_eligible: false` in the archive inventory.
Qualified CPU fixtures, native observations, failures and quarantined records
retain their original qualification boundaries. Backup does not reopen a gate,
approve a hypothesis, establish model gains, or authorize training.

`snapshots/<content_sha256>/archive.tar` contains separately prefixed `primary`
and, only when explicitly frozen and supplied, `exception` sources. The adjacent
manifest records source Git HEAD as provenance, inventory, coverage, exclusions,
limits and archive integrity. External paths mentioned inside JSON are never
followed. Model weights, credentials, protected central CHECK/FINAL banks, live
databases, caches, opaque compressed source bundles and backup-local state are
excluded. Checked individual upstream source, license and paper evidence may
remain as research provenance; their original licensing still applies.

Snapshots are content addressed. Existing snapshot bytes must verify before a
same-content no-op; changed content receives a new directory and Hub commit.
No remote deletions, public visibility changes or scheduled uploader are used.
"""


class BackupError(RuntimeError):
    """A bounded safety, integrity, privacy or reconciliation refusal."""


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def strict_json(data: bytes | str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BackupError("Duplicate JSON metadata key refused")
            result[key] = value
        return result
    def constant(_):
        raise BackupError("Non-finite JSON metadata refused")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def exact_keys(value: object, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise BackupError("Unexpected or missing metadata fields")


def payload_limits() -> dict:
    return {"max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES, "max_files": MAX_FILES}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(rows: list[dict]) -> str:
    # HEAD, exclusions and local receipts cannot make an unchanged payload fire.
    return sha(canonical([{k: row[k] for k in ("path", "bytes", "sha256")} for row in sorted(rows, key=lambda r: r["path"])]))


def no_link(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise BackupError("Link/reparse point refused")
    return info


def safe_path(root: Path, path: Path) -> Path:
    """Reject aliases/escapes before any source content is read."""
    root = root.absolute()
    path = path.absolute()
    if root.resolve() != root:
        raise BackupError("Source boundary has a linked ancestor")
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise BackupError("Path outside source boundary") from exc
    no_link(root)
    current = root
    for part in rel.parts:
        if part in {".", ".."}:
            raise BackupError("Traversal refused")
        current /= part
        no_link(current)
    if not path.resolve().is_relative_to(root.resolve()):
        raise BackupError("Resolved path escaped boundary")
    return path


def classify(relative: PurePosixPath) -> str | None:
    if SECRET_BYTES.search(str(relative).encode("utf-8")):
        raise BackupError("Sensitive artifact path refused")
    lower = [p.lower() for p in relative.parts]
    name = lower[-1]
    if any(p in CACHE_DIRS for p in lower):
        return "cache_or_backup_local_state"
    if any(p in WEIGHT_DIRS for p in lower):
        return "canonical_model_weights"
    if any(p in PROTECTED_DIRS for p in lower):
        return "protected_CHECK_FINAL_bank"
    if any(p in SECRET_NAMES or p == ".env" or p.startswith(".env.") or p.endswith((".pem", ".key", ".p12", ".pfx")) for p in lower):
        raise BackupError(f"Unexpected sensitive name refused: {relative}")
    if name.startswith(("hf_private_backup_receipt", "hub_backup_receipt", "hf_backup_receipt")) or "hub_backup_receipts" in lower or "hf_backup_receipts" in lower:
        return "backup_receipt_only"
    suffix = PurePosixPath(name).suffix
    if suffix in DB_SUFFIXES or name.endswith(("-wal", "-shm", ".db-journal", ".sqlite3-journal")):
        return "database_not_opened"
    if suffix in TENSOR_SUFFIXES:
        return "tensor_weight_or_activation_ambiguity_not_opened"
    if suffix == ".pyc":
        return "cache_bytecode"
    if name.endswith((".tar.gz", ".tgz", ".gz", ".zip", ".7z", ".tar")):
        return "opaque_archive_not_inspected"
    if suffix == ".html":
        if relative.as_posix() not in KNOWN_HTML:
            raise BackupError(f"Unlisted HTML provenance path refused: {relative}")
        return None
    if suffix in TEXT_SUFFIXES or name in {"license", ".gitignore"} or suffix in {".pdf", ".png"}:
        return None
    raise BackupError(f"Unexpected file type refused: {relative}")


def read_stable(root: Path, path: Path, expected: dict | None = None) -> bytes:
    path = safe_path(root, path)
    before = no_link(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1:
        raise BackupError("Non-regular or hard-linked source refused")
    if before.st_size > MAX_FILE_BYTES:
        raise BackupError("File exceeds maximum safe artifact size")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise BackupError("Source changed during open")
        data = stream.read(MAX_FILE_BYTES + 1)
        after_open = os.fstat(stream.fileno())
    after = no_link(path)
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(data) > MAX_FILE_BYTES or stamp(before) != stamp(after_open) or stamp(before) != stamp(after):
        raise BackupError("Source changed during read")
    validate_bytes(PurePosixPath(path.relative_to(root).as_posix()), data)
    if expected and (expected["bytes"] != len(data) or expected["sha256"] != sha(data)):
        raise BackupError("Source changed after inventory")
    return data


def validate_bytes(relative: PurePosixPath, data: bytes) -> None:
    if SECRET_BYTES.search(data):
        raise BackupError(f"Credential material refused: {relative}")
    suffix = relative.suffix.lower()
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise BackupError("Unexpected PDF content")
    elif suffix == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BackupError("Unexpected PNG content")
    else:
        try:
            data.decode("utf-8-sig")
        except UnicodeError as exc:
            raise BackupError("Unexpected non-UTF-8 artifact") from exc
        if b"\0" in data and sha(data) not in KNOWN_NUL_TEXT:
            raise BackupError(f"Unexpected binary content: {relative}")
        if suffix == ".html":
            parts = relative.parts
            # These are the three explicit contexts used by source inventory,
            # source read and prepared tar validation. Basenames are never keys.
            if parts[:len(PACKAGE.parts)] == PACKAGE.parts:
                parts = parts[len(PACKAGE.parts):]
            elif len(parts) >= 2 and parts[:2] in (("primary", "package"), ("exception", "package")):
                parts = parts[2:]
            expected = KNOWN_HTML.get(PurePosixPath(*parts).as_posix())
            if expected is None:
                raise BackupError(f"Unlisted HTML provenance path refused: {relative}")
            if sha(data) != expected:
                raise BackupError(f"HTML provenance hash differs: {relative}")


def inventory(root: Path, prefix: str) -> dict:
    if prefix not in {"primary", "exception"}:
        raise BackupError("Unknown source prefix")
    root = root.absolute()
    base = safe_path(root, root / PACKAGE)
    rows, exclusions = [], []
    visited = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal visited
        if depth > MAX_DEPTH:
            raise BackupError("Source depth limit exceeded")
        with os.scandir(directory) as entries:
            names = []
            for entry in entries:
                if visited + len(names) >= MAX_ENTRIES:
                    raise BackupError("Source entry limit exceeded")
                names.append(entry.name)
            names.sort()
        for name in names:
            visited += 1
            if visited > MAX_ENTRIES:
                raise BackupError("Source entry limit exceeded")
            path = directory / name
            info = no_link(path)
            relative = PurePosixPath(path.relative_to(base).as_posix())
            reason = classify(relative) if not stat.S_ISDIR(info.st_mode) else directory_reason(relative)
            if reason:
                exclusions.append({"path": f"{prefix}/package/{relative}", "bytes": None if stat.S_ISDIR(info.st_mode) else info.st_size, "reason": reason})
                continue
            if stat.S_ISDIR(info.st_mode):
                walk(path, depth + 1)
            else:
                add(path, f"{prefix}/package/{relative}")

    def add(path: Path, archive_path: str) -> None:
        data = read_stable(root, path)
        rows.append({"path": archive_path, "bytes": len(data), "sha256": sha(data), "type": path.suffix.lower() or "text", "training_eligible": False, "qualification": QUALIFICATION})
        if len(rows) > MAX_FILES or sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES:
            raise BackupError("Source inventory count/byte limit exceeded")

    walk(base, 0)
    for name in ("AGENTS.md", "RINEYA_CENTRALIZED_REASONING_SYSTEM.md"):
        path = root / name
        if path.exists():
            add(path, f"{prefix}/provenance/{name}")
    rows.sort(key=lambda row: row["path"])
    types = {}
    for row in rows:
        group = types.setdefault(row["type"], {"files": 0, "bytes": 0})
        group["files"] += 1
        group["bytes"] += row["bytes"]
    return {"prefix": prefix, "root": str(root), "content_sha256": digest(rows), "files": rows, "exclusions": exclusions, "types": types, "included_files": len(rows), "included_bytes": sum(row["bytes"] for row in rows), "largest_file_bytes": max((row["bytes"] for row in rows), default=0)}


def directory_reason(relative: PurePosixPath) -> str | None:
    # classify() intentionally does not grant arbitrary extensionless directories.
    if SECRET_BYTES.search(str(relative).encode("utf-8")):
        raise BackupError("Sensitive artifact directory refused")
    lower = [p.lower() for p in relative.parts]
    if any(p in CACHE_DIRS for p in lower):
        return "cache_or_backup_local_state"
    if any(p in WEIGHT_DIRS for p in lower):
        return "canonical_model_weights"
    if any(p in PROTECTED_DIRS for p in lower):
        return "protected_CHECK_FINAL_bank"
    if any(p in SECRET_NAMES or p == ".env" or p.startswith(".env.") for p in lower):
        raise BackupError("Sensitive directory refused")
    if "hub_backup_receipts" in lower or "hf_backup_receipts" in lower:
        return "backup_receipt_only"
    return None


def git_head(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15, check=False)
    head = result.stdout.strip()
    return head if result.returncode == 0 and COMMIT.fullmatch(head) else None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    no_link(path.parent)
    if path.exists():
        no_link(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def mkdir_beneath(root: Path, path: Path) -> None:
    root, path = root.absolute(), path.absolute()
    safe_path(root, root)
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise BackupError("Local staging path outside boundary") from exc
    current = root
    for part in parts:
        if part in {".", ".."}:
            raise BackupError("Local staging traversal refused")
        current /= part
        if current.exists() or current.is_symlink():
            no_link(current)
        else:
            current.mkdir()
        if not current.is_dir():
            raise BackupError("Local staging ancestor is not a directory")
    safe_path(root, path)


def local_root(primary: Path) -> Path:
    base = safe_path(primary, primary / PACKAGE)
    target = base / LOCAL_NAME
    if target.exists():
        safe_path(primary, target)
    else:
        target.mkdir()
    return target


def require_digest(value: str) -> str:
    if not HEX.fullmatch(value):
        raise BackupError("Snapshot must be a lowercase SHA-256 digest")
    return value


def prepare(primary: Path, exception: Path | None = None, exception_sha256: str | None = None, head_reader: Callable = git_head) -> dict:
    inventories = [inventory(primary, "primary")]
    if exception:
        if not exception_sha256:
            raise BackupError("Explicit frozen exception content digest required")
        found = inventory(exception, "exception")
        if found["content_sha256"] != require_digest(exception_sha256):
            raise BackupError("Exception worktree differs from supplied frozen inventory")
        inventories.append(found)
    rows = sorted([row for entry in inventories for row in entry["files"]], key=lambda r: r["path"])
    if len(rows) > MAX_FILES or sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES:
        raise BackupError("Combined snapshot count/byte limit exceeded")
    content = digest(rows)
    local = local_root(primary)
    frozen = local / "snapshots" / content
    if frozen.exists():
        verify_prepared(frozen, content)
        return {"content_sha256": content, "prepared_path": str(frozen), "status": "LOCAL_SAME_CONTENT_NOOP"}
    building = local / "building"
    if building.exists():
        raise BackupError("Incomplete preparation retained; review local building directory before retry")
    building.mkdir()
    snapshot = building / "snapshots" / content
    mkdir_beneath(local, snapshot)
    archive = snapshot / "archive.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as writer:
        for row in rows:
            source = next(entry for entry in inventories if row["path"].startswith(entry["prefix"] + "/"))
            relative = PurePosixPath(row["path"])
            root = Path(source["root"])
            path = root / PACKAGE / Path(*relative.parts[2:]) if relative.parts[1] == "package" else root / Path(*relative.parts[2:])
            data = read_stable(root, path, row)
            info = tarfile.TarInfo(row["path"])
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            writer.addfile(info, io.BytesIO(data))
    for entry in inventories:
        refreshed = inventory(Path(entry["root"]), entry["prefix"])
        if refreshed["files"] != entry["files"]:
            raise BackupError("Source changed during preparation; partial local archive retained")
    manifest = {"schema_version": 1, "repo_id": REPO_ID, "repo_type": "dataset", "visibility_required": "private", "content_sha256": content, "training_eligible": False, "coverage": COVERAGE, "limits": payload_limits(), "sources": [{**entry, "git_head_provenance_only": head_reader(Path(entry["root"]))} for entry in inventories], "files": rows, "archive": {"name": "archive.tar", "bytes": archive.stat().st_size, "sha256": file_sha(archive)}, "exclusion_policy": EXCLUSION_POLICY}
    manifest_bytes = canonical(manifest)
    (snapshot / "manifest.json").write_bytes(manifest_bytes)
    (building / "README.md").write_text(CARD, encoding="utf-8", newline="\n")
    binding_path = local / "bindings" / f"{content}.json"
    mkdir_beneath(local, binding_path.parent)
    binding = {"content_sha256": content, "manifest_sha256": sha(manifest_bytes), "archive_sha256": manifest["archive"]["sha256"], "card_sha256": sha(CARD.encode("utf-8"))}
    if binding_path.exists():
        raise BackupError("Existing preparation binding retained; review before retry")
    atomic_json(binding_path, binding)
    verify_prepared(building, content)
    mkdir_beneath(local, frozen.parent)
    building.rename(frozen)
    return {"content_sha256": content, "prepared_path": str(frozen), "status": "PREPARED_LOCAL_ONLY", "included_files": len(rows), "included_bytes": sum(row["bytes"] for row in rows)}


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def manifest_metadata(manifest: dict, rows: list[dict], primary: Path) -> None:
    exact_keys(manifest, {"schema_version", "repo_id", "repo_type", "visibility_required", "content_sha256", "training_eligible", "coverage", "limits", "sources", "files", "archive", "exclusion_policy"})
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["coverage"] != COVERAGE or manifest["exclusion_policy"] != EXCLUSION_POLICY or manifest["limits"] != payload_limits() or not all(type(v) is int for v in manifest["limits"].values()):
        raise BackupError("Manifest fixed metadata differs")
    exact_keys(manifest["archive"], {"name", "bytes", "sha256"})
    if manifest["archive"]["name"] != "archive.tar" or type(manifest["archive"]["bytes"]) is not int or not 0 <= manifest["archive"]["bytes"] <= ARCHIVE_BOUND or not isinstance(manifest["archive"]["sha256"], str) or not HEX.fullmatch(manifest["archive"]["sha256"]):
        raise BackupError("Manifest archive metadata invalid")
    sources = manifest["sources"]
    if not isinstance(sources, list) or len(sources) not in {1, 2}:
        raise BackupError("Manifest source count invalid")
    for source, prefix, root in zip(sources, ("primary", "exception"), (primary, EXCEPTION)):
        exact_keys(source, {"prefix", "root", "content_sha256", "files", "exclusions", "types", "included_files", "included_bytes", "largest_file_bytes", "git_head_provenance_only"})
        if source["prefix"] != prefix or source["root"] != str(root.absolute()):
            raise BackupError("Manifest source provenance boundary differs")
        head = source["git_head_provenance_only"]
        if head is not None and (not isinstance(head, str) or not COMMIT.fullmatch(head)):
            raise BackupError("Manifest Git provenance invalid")
        selected = [row for row in rows if row["path"].startswith(prefix + "/")]
        if not selected or canonical(source["files"]) != canonical(selected) or source["content_sha256"] != digest(selected):
            raise BackupError("Manifest source/content association differs")
        expected_types = {}
        for row in selected:
            group = expected_types.setdefault(row["type"], {"files": 0, "bytes": 0})
            group["files"] += 1
            group["bytes"] += row["bytes"]
        expected_totals = {"included_files": len(selected), "included_bytes": sum(row["bytes"] for row in selected), "largest_file_bytes": max(row["bytes"] for row in selected)}
        if canonical(source["types"]) != canonical(expected_types) or any(type(source[k]) is not int or source[k] != v for k, v in expected_totals.items()):
            raise BackupError("Manifest source inventory aggregates differ")
        exclusions = source["exclusions"]
        if not isinstance(exclusions, list) or len(exclusions) > MAX_ENTRIES:
            raise BackupError("Manifest exclusions exceed bound")
        seen = {row["path"] for row in selected}
        for excluded in exclusions:
            exact_keys(excluded, {"path", "bytes", "reason"})
            raw = excluded["path"]
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > 512:
                raise BackupError("Manifest excluded path invalid")
            p = PurePosixPath(raw)
            if p.as_posix() != raw or len(p.parts) < 3 or p.parts[:2] != (prefix, "package") or any(part in {"..", "."} or "\\" in part or ":" in part for part in p.parts) or raw in seen:
                raise BackupError("Manifest excluded path association invalid")
            count = excluded["bytes"]
            if count is not None and (type(count) is not int or not 0 <= count <= 2**63 - 1):
                raise BackupError("Manifest excluded byte count invalid")
            reason = directory_reason(PurePosixPath(*p.parts[2:])) if count is None else classify(PurePosixPath(*p.parts[2:]))
            if not reason or reason != excluded["reason"]:
                raise BackupError("Manifest exclusion reason differs")
            seen.add(raw)
    if {r["path"].split("/")[0] for r in rows} != {source["prefix"] for source in sources}:
        raise BackupError("Manifest source coverage differs")


def prepared_local(frozen: Path, content: str) -> Path:
    if frozen.name == "building":
        local = frozen.parent
    elif frozen.name == content and frozen.parent.name == "snapshots":
        local = frozen.parent.parent
    else:
        raise BackupError("Prepared local path shape differs")
    if local.name != LOCAL_NAME or local.parent != local.parent.parent.parent / PACKAGE:
        raise BackupError("Prepared package boundary differs")
    safe_path(local, frozen)
    return local


def verify_prepared(frozen: Path, content: str) -> dict:
    require_digest(content)
    local = prepared_local(frozen, content)
    safe_path(frozen.parent, frozen)
    snapshot = frozen / "snapshots" / content
    expected_paths = {"README.md", f"snapshots/{content}/archive.tar", f"snapshots/{content}/manifest.json"}
    for directory, expected_names in ((frozen, {"README.md", "snapshots"}), (frozen / "snapshots", {content}), (snapshot, {"archive.tar", "manifest.json"})):
        safe_path(frozen, directory)
        names = set()
        with os.scandir(directory) as entries:
            for entry in entries:
                no_link(directory / entry.name)
                names.add(entry.name)
                if len(names) > 3:
                    raise BackupError("Prepared folder count bound exceeded")
        if names != expected_names:
            raise BackupError("Prepared folder has unexpected/missing files")
    for relative in expected_paths:
        if not stat.S_ISREG(no_link(frozen / relative).st_mode) or no_link(frozen / relative).st_nlink > 1:
            raise BackupError("Prepared files must be regular and not hard-linked")
    card_bytes = (frozen / "README.md").read_bytes()
    validate_bytes(PurePosixPath("README.md"), card_bytes)
    if card_bytes != CARD.encode("utf-8"):
        raise BackupError("Research dataset card differs")
    manifest_path = snapshot / "manifest.json"
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise BackupError("Prepared manifest exceeds bound")
    manifest_bytes = manifest_path.read_bytes()
    validate_bytes(PurePosixPath("manifest.json"), manifest_bytes)
    manifest = strict_json(manifest_bytes)
    if not isinstance(manifest, dict) or canonical(manifest) != manifest_bytes:
        raise BackupError("Manifest canonical metadata differs")
    if manifest.get("repo_id") != REPO_ID or manifest.get("repo_type") != "dataset" or manifest.get("visibility_required") != "private" or manifest.get("training_eligible") is not False:
        raise BackupError("Prepared destination/scope changed")
    rows = manifest.get("files", [])
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
        raise BackupError("Prepared inventory count invalid")
    indexed = {}
    for row in rows:
        exact_keys(row, {"path", "bytes", "sha256", "type", "training_eligible", "qualification"})
        if not isinstance(row["path"], str) or len(row["path"].encode("utf-8")) > 512:
            raise BackupError("Manifest artifact path invalid")
        p = PurePosixPath(row["path"])
        if p.as_posix() != row["path"] or p.is_absolute() or len(p.parts) < 3 or any(x in {"..", "."} or "\\" in x or ":" in x for x in p.parts) or p.parts[0] not in {"primary", "exception"} or p.parts[1] not in {"package", "provenance"}:
            raise BackupError("Prepared inventory path invalid")
        if p.parts[1] == "provenance" and (len(p.parts) != 3 or p.name not in {"AGENTS.md", "RINEYA_CENTRALIZED_REASONING_SYSTEM.md"}):
            raise BackupError("Prepared root provenance expanded")
        if classify(PurePosixPath(*p.parts[2:])) is not None or row["type"] != (p.suffix.lower() or "text") or row["qualification"] != QUALIFICATION or row.get("training_eligible") is not False or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_FILE_BYTES or not isinstance(row["sha256"], str) or not HEX.fullmatch(row["sha256"]):
            raise BackupError("Prepared artifact outside scope")
        if p.suffix.lower() == ".html" and row["sha256"] != KNOWN_HTML.get(PurePosixPath(*p.parts[2:]).as_posix()):
            raise BackupError("Prepared HTML provenance hash differs")
        if row["path"] in indexed:
            raise BackupError("Prepared duplicate path")
        indexed[row["path"]] = row
    if sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES or digest(rows) != content or manifest.get("content_sha256") != content:
        raise BackupError("Prepared content digest/size differs")
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise BackupError("Manifest artifact order differs")
    manifest_metadata(manifest, rows, local.parent.parent.parent)
    binding_path = safe_path(local, local / "bindings" / f"{content}.json")
    if binding_path.stat().st_size > 2048:
        raise BackupError("Preparation binding exceeds bound")
    binding_bytes = binding_path.read_bytes()
    validate_bytes(PurePosixPath("binding.json"), binding_bytes)
    binding = strict_json(binding_bytes)
    exact_keys(binding, {"content_sha256", "manifest_sha256", "archive_sha256", "card_sha256"})
    expected_binding = {"content_sha256": content, "manifest_sha256": sha(manifest_bytes), "archive_sha256": manifest["archive"]["sha256"], "card_sha256": sha(card_bytes)}
    if binding != expected_binding or binding_bytes != canonical(binding):
        raise BackupError("Prepared metadata/provenance binding differs")
    archive = snapshot / "archive.tar"
    # Raw tar overhead is bounded independently of the payload bound.
    if archive.stat().st_size > ARCHIVE_BOUND or archive.stat().st_size != manifest["archive"]["bytes"] or file_sha(archive) != manifest["archive"]["sha256"]:
        raise BackupError("Prepared archive integrity differs")
    seen = set()
    with tarfile.open(archive, "r:") as reader:
        for member in reader:
            if not member.isfile() or member.name not in indexed or member.name in seen or member.size != indexed[member.name]["bytes"]:
                raise BackupError("Prepared archive membership differs")
            stream = reader.extractfile(member)  # bytes only; never extract to disk
            if stream is None:
                raise BackupError("Prepared archive member unreadable")
            data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) != member.size or sha(data) != indexed[member.name]["sha256"]:
                raise BackupError("Prepared archive artifact hash differs")
            validate_bytes(PurePosixPath(member.name), data)
            seen.add(member.name)
    if seen != set(indexed):
        raise BackupError("Prepared archive missing artifact")
    return manifest


SDK_BRIDGE = r"""
import importlib.util, json, sys
from pathlib import Path
try:
    helper_path = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-centralized-reasoning-system-v1/experiments/rineya_centralized_reasoning_system_v1/hub_backup.py')
    spec = importlib.util.spec_from_file_location('crs_private_backup_bridge', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for relative, expected in helper.SDK_PINS.items():
        path = helper.safe_path(helper.SDK_ROOT, helper.SDK_ROOT / relative)
        if helper.file_sha(path) != expected:
            raise helper.BackupError('Installed SDK source pin differs')
    import huggingface_hub
    from huggingface_hub import HfApi, CommitOperationAdd
    if huggingface_hub.__version__ != helper.SDK_VERSION or Path(huggingface_hub.__file__).resolve().parent != helper.SDK_ROOT.resolve():
        raise helper.BackupError('Installed SDK identity/version differs')
    raw = sys.stdin.buffer.read(16385)
    if len(raw) > 16384:
        raise helper.BackupError('SDK request exceeds bound')
    request = helper.strict_json(raw)
    result = helper.sdk_commit(request, HfApi(), CommitOperationAdd)
    print(json.dumps(result, sort_keys=True))
except BaseException:
    print('{"status":"REFUSED"}')
    sys.exit(2)
"""


class HFCLI:
    """Fixed dataset CLI reads plus a pinned SDK expected-parent write bridge."""
    def run(self, arguments: list[str], timeout: int = 120) -> str:
        try:
            result = subprocess.run(["hf", *arguments], capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackupError("HF CLI response unavailable; reconcile any pending upload") from exc
        if result.returncode:
            raise BackupError("HF CLI command failed; raw output withheld")
        return result.stdout.strip()

    def info(self) -> dict:
        info = strict_json(self.run(["datasets", "info", REPO_ID, "--expand", "private,sha,siblings", "--format", "json"]))
        verified_info(info)
        return info

    def download(self, remote: str, revision: str, directory: Path, maximum_bytes: int) -> Path:
        plan = json.loads(self.run(["download", REPO_ID, remote, "--type", "dataset", "--revision", revision, "--dry-run", "--format", "json"]))
        # hf 1.27 emits human-sized JSON values (e.g. "2.5K"). Refuse unknown
        # formats; conservatively bound rounding and a possible 1024 base.
        if not isinstance(plan, list) or len(plan) != 1 or plan[0].get("file") != remote:
            raise BackupError("Remote download inventory differs")
        size = plan[0].get("size")
        if isinstance(size, int) and not isinstance(size, bool):
            bound = size
        elif isinstance(size, str) and (match := re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", size)):
            power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[match[2]]
            bound = int((float(match[1]) + (1 if power else 0)) * 1024 ** power)
        else:
            raise BackupError("Remote download byte size unavailable")
        if not 0 <= bound <= maximum_bytes:
            raise BackupError("Remote download exceeds safe byte bound")
        self.run(["download", REPO_ID, remote, "--type", "dataset", "--revision", revision, "--local-dir", str(directory), "--quiet"])
        return directory / remote

    def upload(self, folder: Path, content: str, parent_commit: str) -> None:
        require_digest(content)
        if not isinstance(parent_commit, str) or not COMMIT.fullmatch(parent_commit):
            raise BackupError("Expected parent commit invalid")
        verify_prepared(folder, content)
        request = {"folder": str(folder), "content_sha256": content, "parent_commit": parent_commit, "repo_id": REPO_ID, "repo_type": "dataset", "visibility_required": "private"}
        try:
            result = subprocess.run([str(HERMES_PYTHON), "-c", SDK_BRIDGE], input=canonical(request).decode("utf-8"), capture_output=True, text=True, timeout=600, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackupError("SDK write response unavailable; reconcile pending upload") from exc
        if result.returncode:
            raise BackupError("SDK expected-parent write failed; raw output withheld")
        response = strict_json(result.stdout)
        exact_keys(response, {"status", "remote_commit", "parent_commit"})
        if response["status"] != "COMMITTED" or response["parent_commit"] != parent_commit or not isinstance(response["remote_commit"], str) or not COMMIT.fullmatch(response["remote_commit"]):
            raise BackupError("SDK write response not verified")


def verified_info(info: dict) -> None:
    exact_keys(info, {"id", "private", "sha", "siblings"})
    if info["id"] != REPO_ID or info["private"] is not True or not isinstance(info["sha"], str) or not COMMIT.fullmatch(info["sha"]):
        raise BackupError("Fixed dataset identity/private visibility/commit not verified")
    sibling_paths(info)


def sibling_paths(info: dict) -> set[str]:
    if not isinstance(info, dict):
        raise BackupError("Remote file inventory invalid")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > MAX_ENTRIES:
        raise BackupError("Remote file inventory unavailable")
    paths, folded = set(), set()
    for item in siblings:
        exact_keys(item, {"rfilename"})
        raw = item["rfilename"]
        if not isinstance(raw, str) or not raw or len(raw.encode("utf-8")) > 512 or any(ord(c) < 32 or ord(c) == 127 for c in raw):
            raise BackupError("Remote inventory path invalid")
        p = PurePosixPath(raw)
        if p.is_absolute() or p.as_posix() != raw or any(part in {"..", "."} or "\\" in part or ":" in part or part.endswith((" ", ".")) for part in p.parts) or raw in paths or raw.casefold() in folded:
            raise BackupError("Remote inventory path ambiguous/duplicate/traversal")
        paths.add(raw)
        folded.add(raw.casefold())
    for raw in paths:
        p = PurePosixPath(raw)
        if any(parent.as_posix().casefold() in folded for parent in p.parents if parent.as_posix() != "."):
            raise BackupError("Remote inventory file/directory ambiguity")
    return paths


def transport_bytes(root: Path, path: Path, maximum: int, expected_hash: str) -> bytes:
    safe_path(root, path)
    before = no_link(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1 or not 0 <= before.st_size <= maximum:
        raise BackupError("SDK prepared file size/type differs")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(maximum + 1)
        after_open = os.fstat(stream.fileno())
    after = no_link(path)
    stamp = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if stamp(before) != stamp(opened) or stamp(before) != stamp(after_open) or stamp(before) != stamp(after) or len(data) != before.st_size or sha(data) != expected_hash:
        raise BackupError("SDK prepared bytes changed after verification")
    return data


def sdk_commit(request: dict, api, add_operation) -> dict:
    """Sole SDK write, tested with fake API. Bind three immutable byte additions."""
    exact_keys(request, {"folder", "content_sha256", "parent_commit", "repo_id", "repo_type", "visibility_required"})
    if request["repo_id"] != REPO_ID or request["repo_type"] != "dataset" or request["visibility_required"] != "private" or not isinstance(request["parent_commit"], str) or not COMMIT.fullmatch(request["parent_commit"]):
        raise BackupError("SDK fixed private destination/parent differs")
    content = require_digest(request["content_sha256"])
    expected = PRIMARY / PACKAGE / LOCAL_NAME / "snapshots" / content
    if request["folder"] != str(expected):
        raise BackupError("SDK prepared folder boundary differs")
    folder = expected
    manifest = verify_prepared(folder, content)
    prefix = f"snapshots/{content}/"
    files = [("README.md", MAX_FILE_BYTES, sha(CARD.encode("utf-8"))), (prefix + "manifest.json", MAX_MANIFEST_BYTES, sha(canonical(manifest))), (prefix + "archive.tar", ARCHIVE_BOUND, manifest["archive"]["sha256"])]
    operations = []
    for relative, maximum, expected_hash in files:
        data = transport_bytes(folder, folder / relative, maximum, expected_hash)
        if not relative.endswith("archive.tar"):
            validate_bytes(PurePosixPath(relative), data)
        operations.append(add_operation(path_in_repo=relative, path_or_fileobj=data))
    observed = api.repo_info(repo_id=REPO_ID, repo_type="dataset", revision="main", files_metadata=False)
    if not isinstance(observed.siblings, list):
        raise BackupError("SDK remote inventory unavailable")
    info = {"id": observed.id, "private": observed.private, "sha": observed.sha, "siblings": [{"rfilename": getattr(item, "rfilename", None)} for item in observed.siblings]}
    verified_info(info)
    if info["sha"] != request["parent_commit"] or any(path.startswith(prefix) for path in sibling_paths(info)):
        raise BackupError("SDK private parent/absence changed before write")
    committed = api.create_commit(repo_id=REPO_ID, repo_type="dataset", revision="main", operations=operations, commit_message=f"Preserve CRS-001 research snapshot {content}", parent_commit=request["parent_commit"], create_pr=False, run_as_future=False)
    if not isinstance(getattr(committed, "oid", None), str) or not COMMIT.fullmatch(committed.oid):
        raise BackupError("SDK commit response unavailable")
    return {"status": "COMMITTED", "remote_commit": committed.oid, "parent_commit": request["parent_commit"]}


def state_path(local: Path, content: str) -> Path:
    return local / "state" / f"{require_digest(content)}.json"


def read_state(local: Path, content: str) -> dict:
    path = state_path(local, content)
    if not path.exists():
        return {"content_sha256": content, "status": "UNATTEMPTED"}
    safe_path(local, path)
    if path.stat().st_size > 16384:
        raise BackupError("Local state exceeds bound")
    value = json.loads(path.read_bytes())
    if value.get("content_sha256") != content:
        raise BackupError("Local state content association differs")
    return value


@contextlib.contextmanager
def lock(local: Path):
    path = local / "operation.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BackupError("Another backup operation or retained crash lock requires review") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def reconcile_local(local: Path, content: str, client: HFCLI) -> dict:
    frozen = local / "snapshots" / require_digest(content)
    safe_path(local, frozen)
    manifest = verify_prepared(frozen, content)
    info = client.info()
    verified_info(info)
    prefix = f"snapshots/{content}/"
    expected = {prefix + "archive.tar", prefix + "manifest.json"}
    present = {path for path in sibling_paths(info) if path.startswith(prefix)}
    state = {"content_sha256": content, "remote_commit": info["sha"], "repo_id": REPO_ID, "private_verified": True}
    if not present:
        state["status"] = "RECONCILED_ABSENT"
    elif present != expected:
        raise BackupError("Remote snapshot partial/collision; no overwrite allowed")
    else:
        destination = local / "reconcile" / content / info["sha"]
        mkdir_beneath(local, destination)
        safe_path(local, destination)
        remote_manifest = client.download(prefix + "manifest.json", info["sha"], destination, 8 * 1024 * 1024)
        remote_archive = client.download(prefix + "archive.tar", info["sha"], destination, MAX_TOTAL_BYTES + MAX_FILES * 2048 + 10240)
        safe_path(local, remote_manifest)
        safe_path(local, remote_archive)
        if remote_manifest.stat().st_size != (frozen / prefix / "manifest.json").stat().st_size or remote_manifest.read_bytes() != (frozen / prefix / "manifest.json").read_bytes() or remote_archive.stat().st_size != manifest["archive"]["bytes"] or file_sha(remote_archive) != manifest["archive"]["sha256"]:
            raise BackupError("Remote snapshot bytes differ; no overwrite allowed")
        state["status"] = "VERIFIED"
    mkdir_beneath(local, state_path(local, content).parent)
    atomic_json(state_path(local, content), state)
    return state


def upload_local(local: Path, content: str, client: HFCLI) -> dict:
    frozen = local / "snapshots" / require_digest(content)
    safe_path(local, frozen)
    verify_prepared(frozen, content)
    prior = read_state(local, content)
    if prior["status"] in {"PENDING", "UNKNOWN"}:
        raise BackupError("Unknown/pending upload must be explicitly reconciled before retry")
    observed = reconcile_local(local, content, client)
    if observed["status"] == "VERIFIED":
        return {**observed, "status": "VERIFIED_REMOTE_SAME_CONTENT_NOOP"}
    # Refresh privacy immediately before the sole write command.
    before_write = client.info()
    verified_info(before_write)
    if any(path.startswith(f"snapshots/{content}/") for path in sibling_paths(before_write)):
        raise BackupError("Remote target appeared before write; explicitly reconcile")
    verify_prepared(frozen, content)
    atomic_json(state_path(local, content), {"content_sha256": content, "status": "PENDING", "repo_id": REPO_ID, "before_commit": before_write["sha"]})
    try:
        client.upload(frozen, content, before_write["sha"])
        after = reconcile_local(local, content, client)
        if after["status"] != "VERIFIED":
            raise BackupError("Upload returned without verified remote snapshot")
        return after
    except Exception as exc:
        atomic_json(state_path(local, content), {"content_sha256": content, "status": "UNKNOWN", "repo_id": REPO_ID, "before_commit": before_write["sha"], "action_required": "Run --reconcile; do not repeat upload blindly"})
        raise BackupError("Upload outcome unknown; run explicit --reconcile before any retry") from exc


def authorized(primary: Path) -> None:
    path = safe_path(primary, primary / PACKAGE / "authorization/OWNER_EXCEPTION_001_GRANTED_2026-10-04.json")
    receipt = json.loads(read_stable(primary, path))
    scope = receipt.get("private_backup_scope", {})
    if receipt.get("authorization_received") is not True or scope.get("repo_id") != REPO_ID or scope.get("repo_type") != "dataset" or scope.get("visibility_required") != "private" or scope.get("no_remote_deletions_or_public_visibility_changes") is not True:
        raise BackupError("Owner's fixed private-backup authorization not verified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--upload", action="store_true")
    actions.add_argument("--reconcile", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--exception-worktree", type=Path)
    parser.add_argument("--exception-sha256")
    args = parser.parse_args(argv)
    try:
        if args.exception_worktree and args.exception_worktree.absolute() != EXCEPTION.absolute():
            raise BackupError("Only the explicitly named exception worktree is in scope")
        if args.exception_sha256 and not args.exception_worktree:
            raise BackupError("Exception digest requires exception worktree")
        if args.upload or args.reconcile:
            if not args.snapshot or args.exception_worktree or args.exception_sha256:
                raise BackupError("Upload/reconcile requires prepared --snapshot only")
            authorized(PRIMARY)
            local = local_root(PRIMARY)
            with lock(local):
                result = upload_local(local, args.snapshot, HFCLI()) if args.upload else reconcile_local(local, args.snapshot, HFCLI())
        elif args.snapshot:
            raise BackupError("Snapshot applies only to upload/reconcile")
        elif args.dry_run:
            found = [inventory(PRIMARY, "primary")]
            if args.exception_worktree:
                found.append(inventory(args.exception_worktree, "exception"))
            result = {"status": "READ_ONLY_INVENTORY_NO_UPLOAD", "repo_id": REPO_ID, "repo_type": "dataset", "visibility_required": "private", "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES, "max_files": MAX_FILES}, "sources": found}
        else:
            local = local_root(PRIMARY)
            with lock(local):
                result = prepare(PRIMARY, args.exception_worktree, args.exception_sha256)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (BackupError, OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        # CLI/token-bearing subprocess outputs are never rendered here.
        message = str(exc) if isinstance(exc, BackupError) else "Backup refused: local/remote data did not satisfy the bounded contract"
        print(json.dumps({"status": "REFUSED", "reason": message}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
