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
    validate_bytes(PurePosixPath(path.name), data)
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
        rows.append({"path": archive_path, "bytes": len(data), "sha256": sha(data), "type": path.suffix.lower() or "text", "training_eligible": False, "qualification": "research_evidence_original_status_retained"})
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
    manifest = {"schema_version": 1, "repo_id": REPO_ID, "repo_type": "dataset", "visibility_required": "private", "content_sha256": content, "training_eligible": False, "coverage": "CRS-001 package source/docs/open fixture/raw/results/quality/failure/quarantined history; root charter and AGENTS provenance only", "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES, "max_files": MAX_FILES}, "sources": [{**entry, "git_head_provenance_only": head_reader(Path(entry["root"]))} for entry in inventories], "files": rows, "archive": {"name": "archive.tar", "bytes": archive.stat().st_size, "sha256": file_sha(archive)}, "exclusion_policy": "Never follow JSON paths, import tensors, open databases, traverse links, upload credentials/weights/protected banks/caches/backup-local state, or promote research into training data."}
    (snapshot / "manifest.json").write_bytes(canonical(manifest))
    (building / "README.md").write_text(CARD, encoding="utf-8", newline="\n")
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


def verify_prepared(frozen: Path, content: str) -> dict:
    require_digest(content)
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
    if (frozen / "README.md").read_bytes() != CARD.encode("utf-8"):
        raise BackupError("Research dataset card differs")
    manifest_path = snapshot / "manifest.json"
    if manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise BackupError("Prepared manifest exceeds bound")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("repo_id") != REPO_ID or manifest.get("repo_type") != "dataset" or manifest.get("visibility_required") != "private" or manifest.get("training_eligible") is not False:
        raise BackupError("Prepared destination/scope changed")
    rows = manifest.get("files", [])
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
        raise BackupError("Prepared inventory count invalid")
    indexed = {}
    for row in rows:
        p = PurePosixPath(row["path"])
        if p.is_absolute() or len(p.parts) < 3 or any(x in {"..", "."} or "\\" in x or ":" in x for x in p.parts) or p.parts[0] not in {"primary", "exception"} or p.parts[1] not in {"package", "provenance"}:
            raise BackupError("Prepared inventory path invalid")
        if p.parts[1] == "provenance" and (len(p.parts) != 3 or p.name not in {"AGENTS.md", "RINEYA_CENTRALIZED_REASONING_SYSTEM.md"}):
            raise BackupError("Prepared root provenance expanded")
        if classify(PurePosixPath(*p.parts[2:])) is not None or row.get("training_eligible") is not False or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_FILE_BYTES or not HEX.fullmatch(row["sha256"]):
            raise BackupError("Prepared artifact outside scope")
        if row["path"] in indexed:
            raise BackupError("Prepared duplicate path")
        indexed[row["path"]] = row
    if sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES or digest(rows) != content or manifest.get("content_sha256") != content:
        raise BackupError("Prepared content digest/size differs")
    archive = snapshot / "archive.tar"
    # Raw tar overhead is bounded independently of the payload bound.
    if archive.stat().st_size > MAX_TOTAL_BYTES + MAX_FILES * 2048 + 10240 or archive.stat().st_size != manifest["archive"]["bytes"] or file_sha(archive) != manifest["archive"]["sha256"]:
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


class HFCLI:
    """Only fixed-repository info/download/upload. CLI output is never printed."""
    def run(self, arguments: list[str], timeout: int = 120) -> str:
        try:
            result = subprocess.run(["hf", *arguments], capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackupError("HF CLI response unavailable; reconcile any pending upload") from exc
        if result.returncode:
            raise BackupError("HF CLI command failed; raw output withheld")
        return result.stdout.strip()

    def info(self) -> dict:
        info = json.loads(self.run(["datasets", "info", REPO_ID, "--expand", "private,sha,siblings", "--format", "json"]))
        if info.get("id") != REPO_ID or info.get("private") is not True or not COMMIT.fullmatch(info.get("sha", "")):
            raise BackupError("Fixed dataset identity/private visibility/commit not verified")
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

    def upload(self, folder: Path, content: str) -> None:
        self.run(["upload", REPO_ID, str(folder), ".", "--type", "dataset", "--private", "--commit-message", f"Preserve CRS-001 research snapshot {content}", "--quiet"], timeout=600)


def sibling_paths(info: dict) -> set[str]:
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        raise BackupError("Remote file inventory unavailable")
    return {item["rfilename"] for item in siblings if isinstance(item, dict) and isinstance(item.get("rfilename"), str)}


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
    if any(path.startswith(f"snapshots/{content}/") for path in sibling_paths(before_write)):
        raise BackupError("Remote target appeared before write; explicitly reconcile")
    verify_prepared(frozen, content)
    atomic_json(state_path(local, content), {"content_sha256": content, "status": "PENDING", "repo_id": REPO_ID, "before_commit": observed["remote_commit"]})
    try:
        client.upload(frozen, content)
        after = reconcile_local(local, content, client)
        if after["status"] != "VERIFIED":
            raise BackupError("Upload returned without verified remote snapshot")
        return after
    except Exception as exc:
        atomic_json(state_path(local, content), {"content_sha256": content, "status": "UNKNOWN", "repo_id": REPO_ID, "before_commit": observed["remote_commit"], "action_required": "Run --reconcile; do not repeat upload blindly"})
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
