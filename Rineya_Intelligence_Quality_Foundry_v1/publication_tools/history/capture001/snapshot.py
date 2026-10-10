#!/usr/bin/env python3
"""Bounded-memory, byte-preserving foundry publication archive (stdlib only).

    python snapshot.py snapshot SOURCE ARCHIVE [--extra LOGICAL=FILE ...]
    python snapshot.py verify ARCHIVE
    python snapshot.py restore ARCHIVE EMPTY_DESTINATION

The source is never written. Archives and restored files are created exclusively.
Objects are one deterministic gzip stream split at compressed byte boundaries;
individual parts need not be independently decompressible. Catalogs are streamed
JSONL gzip files. SQLite indexes live in a temporary directory, not the archive.
Two matching reads plus matching file stats are evidence of stability during a
file's capture, not an atomic filesystem snapshot or a claim about dataset quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
from typing import Iterable


FORMAT = "rineya-foundry-publication-v1"
CHUNK = 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
PROGRAM_PREFIX = "program/20261010_resource_precedence_cohort_design_v1"
PROGRAM_FILES = (
    "program/20261005_supa_admission_lane_v1/CONTINUATION_STATUS_060.json",
    "program/20261005_supa_admission_lane_v1/CONTINUATION_STATUS_061.json",
    "program/20261010_resource_planning_quality_v1/REVIEWS_AND_HORIZONS001.md",
    "program/20261010_resource_planning_quality_v1/support_horizon158/HORIZON_RESULT158.json",
)
SUPPLEMENT_NAMES = frozenset((
    "PRESERVATION_AFTER_v10.json", "REPAIR_RECEIPT_v10.json",
    "SOURCE_PIN_v10.json", "TEST_ONLY_REPAIR_v10.diff",
))
PROTECTED_WORDS = frozenset((
    "check", "final", "eval", "evaluation", "spent", "privatebank", "private",
    "sealedpayloads", "sealedpayload", "sealed_payloads", "private_bank",
))
SKIP_DIRS = frozenset((
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".cache", "cache", "caches", "node_modules", ".venv",
    "venv", "env", "site-packages", "dist", "build", "weights", "models",
    "checkpoints", "packages", "downloads", ".tox", ".nox", "wandb",
))
WEIGHT_EXTENSIONS = frozenset((
    ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".gguf", ".ggml",
    ".h5", ".hdf5", ".pkl", ".pickle", ".whl", ".egg", ".pyc", ".pyo",
    ".dll", ".exe", ".so", ".dylib", ".sqlite", ".sqlite3", ".db",
    ".mdb", ".db3", ".db-wal", ".db-shm",
))
METHOD_ROOTS = frozenset(("scripts", "tests", "registry_tools", "mechanisms"))
METHOD_EXTENSIONS = frozenset((".py", ".ps1", ".sh", ".js", ".ts", ".bat"))
PILOT_PREFIXES = tuple(
    f"{root}/pilot-v{version}"
    for root, versions in (("candidates", (1, 2, 3)), ("packets", (1, 2, 3)), ("releases", (2, 3)))
    for version in versions
)
RAW_CORPUS_NAME = re.compile(
    r"(?:^|[_.-])(?:train(?:_sft)?|corpus|modelraw)\.(?:jsonl|parquet|arrow)$", re.I
)
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_RESERVED = re.compile(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z", re.I)


class ArchiveError(Exception):
    """An archive failed its structural or byte-integrity requirements."""


class SourceChanged(ArchiveError):
    pass


class SecretFound(ArchiveError):
    pass


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError("duplicate JSON key")
        result[key] = value
    return result


def _parse_json(data):
    try:
        value = json.loads(data, object_pairs_hook=_unique_json)
    except (ValueError, UnicodeError) as exc:
        raise ArchiveError("invalid JSON metadata") from exc
    if not isinstance(value, dict):
        raise ArchiveError("metadata row must be an object")
    return value


def canonical_path(value):
    """Reject traversal, drive/ADS names and Windows path aliases on every OS."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ArchiveError("invalid logical path")
    if value.startswith("/") or any(ord(char) < 32 for char in value):
        raise ArchiveError("absolute or control-character path")
    parts = value.split("/")
    for part in parts:
        if part in ("", ".", "..") or part.endswith((".", " ")) or WINDOWS_RESERVED.fullmatch(part):
            raise ArchiveError("unsafe or nonportable logical path")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArchiveError("invalid " + name)
    return value


def _sha(value):
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ArchiveError("invalid SHA256")
    return value


def _fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _within(child, parent):
    try:
        return os.path.commonpath((os.path.normcase(str(child)), os.path.normcase(str(parent)))) == os.path.normcase(str(parent))
    except ValueError:
        return False


def _check_ancestors(path):
    """Do not create or read output through a symlink/reparse-point ancestor."""
    for item in (path,) + tuple(path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ArchiveError("symlink or reparse-point ancestor")


def _regular_file(path):
    _check_ancestors(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ArchiveError("expected a regular file")
    return value


def _open_source(path, expected):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    stream = os.fdopen(descriptor, "rb")
    if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(expected):
        stream.close()
        raise SourceChanged("source changed before open")
    return stream


class SecretScanner:
    """Streaming credential signatures; errors never contain the matched value."""
    _patterns = (
        ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"), False),
        ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), True),
        ("github_token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,255}\b"), True),
        ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), False),
        ("openai_key", re.compile(rb"\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{32,512}\b"), True),
    )

    def __init__(self):
        self.tail = b""

    @staticmethod
    def _high_entropy(value):
        counts = Counter(value)
        entropy = -sum((count / len(value)) * math.log2(count / len(value)) for count in counts.values())
        return len(counts) >= 12 and entropy >= 3.5

    def feed(self, data):
        window = self.tail + data
        for kind, expression, entropy_required in self._patterns:
            for match in expression.finditer(window):
                if not entropy_required or self._high_entropy(match.group(0)):
                    raise SecretFound(kind)
        self.tail = window[-2048:]


class Policy:
    def __init__(self, allowed_program_files=PROGRAM_FILES):
        self.allowed_program_files = tuple(canonical_path(value) for value in allowed_program_files)
        if set(self.allowed_program_files) != set(PROGRAM_FILES):
            raise ArchiveError("only the four documented program references are permitted")

    @staticmethod
    def _words(name):
        lower = name.casefold()
        words = set(re.split(r"[^a-z0-9]+", lower))
        if "privatebank" in lower or "sealedpayload" in lower or "private_bank" in lower or "sealed_payload" in lower:
            words.add("privatebank")
        return words

    def reason(self, relative, is_dir=None):
        """Called on the name before lstat; blocked subtrees are not enumerated."""
        parts = relative.split("/")
        lower = [item.casefold() for item in parts]
        base = lower[-1]
        if lower[0] in ("program", "programs"):
            prefix_allowed = (relative == PROGRAM_PREFIX or relative.startswith(PROGRAM_PREFIX + "/")
                              or PROGRAM_PREFIX.startswith(relative + "/"))
            file_allowed = any(relative == item or item.startswith(relative + "/")
                               for item in self.allowed_program_files)
            if not (prefix_allowed or file_allowed):
                return "unreferenced_program_subtree"
        if any(part in SKIP_DIRS for part in lower):
            return "weights_packages_or_cache"
        if base in (".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", "credentials", "credentials.json") or base.startswith(".env."):
            return "credential_filename"
        if Path(base).suffix in WEIGHT_EXTENSIONS or base.endswith((".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm")):
            return "weights_packages_or_database"
        if (base == "pytorch_model.bin" or base.startswith("pytorch_model-")
                or base == "model.bin" or re.fullmatch(r"model-\d+-of-\d+\.bin", base)):
            return "model_weight_filename"
        pilot = any(relative == prefix or relative.startswith(prefix + "/") for prefix in PILOT_PREFIXES)
        if (not pilot and Path(base).suffix in (".jsonl", ".parquet", ".arrow")
                and RAW_CORPUS_NAME.search(base)):
            return "possible_raw_corpus_outside_owned_pilot"
        if base.endswith((".lock", ".tmp", ".temp", ".swp", ".swo", ".part", ".pid")) or base.startswith(("~$", ".~", ".tmp")):
            return "live_lock_or_temporary_file"
        # Safe manifests are filename exceptions only; protected ancestors stay blocked.
        safe_manifest = (base == "seal.json" or base == "final_manifest" or base.startswith("final_manifest.")
                         or base in ("final_delivery_manifest.json", "final_source_manifest.json"))
        fixture_manifest = lower[0] == "fixtures" and ("metadata" in base or "manifest" in base) and Path(base).suffix in (".json", ".jsonl", ".yaml", ".yml", ".md")
        check_method = Path(base).suffix in METHOD_EXTENSIONS and "check" in self._words(base)
        for index, part in enumerate(parts):
            protected = self._words(part) & PROTECTED_WORDS
            if not protected:
                continue
            if index == len(parts) - 1 and is_dir is not True:
                if (safe_manifest or fixture_manifest) and not (protected - {"check", "final", "eval", "evaluation"}):
                    continue
                if check_method and protected == {"check"}:
                    continue
            return "protected_subtree_or_filename"
        return None

    def materialize(self, relative):
        parts = relative.split("/")
        base = parts[-1].casefold()
        if relative == PROGRAM_PREFIX or relative.startswith(PROGRAM_PREFIX + "/") or relative in self.allowed_program_files:
            return True
        if len(parts) == 1 and (base.startswith("readme") or base.startswith("data_card")):
            return True
        if parts[0].casefold() in ("docs", "config", "scripts", "tests", "registry_tools", "mechanisms"):
            return True
        return parts[0].casefold() == "cycles" and len(parts) <= 3 and Path(base).suffix in (".md", ".json", ".yaml", ".yml", ".txt")

    def describe(self):
        return {
            "protected_name_words": sorted(PROTECTED_WORDS),
            "name_gate_before_stat": True,
            "protected_ancestors_are_never_enumerated": True,
            "safe_filename_exceptions": ["SEAL.json", "FINAL_MANIFEST[.*]", "fixtures metadata/manifest files", "CHECK method source outside protected ancestors"],
            "allowed_program_prefix": PROGRAM_PREFIX,
            "allowed_program_files": list(self.allowed_program_files),
            "owned_pilot_raw_corpus_exceptions": list(PILOT_PREFIXES),
            "possible_raw_corpus_filenames_elsewhere_excluded_before_stat": True,
            "excluded_directory_names": sorted(SKIP_DIRS),
            "excluded_extensions": sorted(WEIGHT_EXTENSIONS),
            "credential_and_live_runtime_filenames_are_excluded": True,
            "secret_signatures": ["PEM private keys", "high-entropy GitHub tokens", "AWS access key IDs", "high-entropy OpenAI keys"],
            "secret_handling": "omit entire occurrence; never redact immutable bytes or print credential values",
            "stability": "stat / hash read / stat / compress-and-hash read / stat, with open-file stat checks; matching reads required",
            "scope": "root and core foundry trees, cycles/evidence/audits/profiles/reviews/candidates/packets/releases and registry/state metadata, subject to explicit omissions",
            "mirror": "README/DATA_CARD, docs/config/scripts/tests/registry_tools/mechanisms, allowed program paths, shallow cycle reports/manifests",
            "omission_rows": "blocked directories produce one unenumerated-subtree row, not an inferred count of descendant files",
            "claims": "byte preservation and integrity only; no dataset QA, training result or model-gain claim",
        }


class HashWriter:
    def __init__(self, file):
        self.file = file
        self.hash = hashlib.sha256()
        self.size = 0

    def write(self, value):
        self.hash.update(value)
        self.size += len(value)
        return self.file.write(value)

    def flush(self):
        self.file.flush()


class CatalogWriter:
    def __init__(self, path):
        self.file = path.open("xb")
        self.output = HashWriter(self.file)
        self.gzip = gzip.GzipFile(filename="", fileobj=self.output, mode="wb", compresslevel=9, mtime=0)
        self.hash = hashlib.sha256()
        self.size = 0
        self.rows = 0
        self.closed = False

    def add(self, value):
        data = _json_bytes(value) + b"\n"
        if len(data) > MAX_RECORD_BYTES:
            raise ArchiveError("catalog record exceeds bounded line limit")
        self.hash.update(data)
        self.size += len(data)
        self.rows += 1
        self.gzip.write(data)

    def close(self):
        if not self.closed:
            self.gzip.close()
            self.file.close()
            self.closed = True
        return {"compressed_sha256": self.output.hash.hexdigest(), "compressed_size": self.output.size,
                "sha256": self.hash.hexdigest(), "size": self.size, "records": self.rows}


def _database(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-2048")
    db.executescript("""
        CREATE TABLE objects (sha TEXT PRIMARY KEY, size INTEGER, compressed_size INTEGER, parts INTEGER);
        CREATE TABLE parts (object TEXT, idx INTEGER, path TEXT UNIQUE, sha TEXT, size INTEGER,
                            PRIMARY KEY(object,idx));
        CREATE TABLE paths (path TEXT PRIMARY KEY, key TEXT UNIQUE, object TEXT, size INTEGER, materialized INTEGER);
        CREATE TABLE omissions (key TEXT PRIMARY KEY);
        CREATE TABLE staging (idx INTEGER PRIMARY KEY, path TEXT, sha TEXT, size INTEGER);
        CREATE TABLE directories (path TEXT PRIMARY KEY, key TEXT UNIQUE);
    """)
    return db


class SplitWriter:
    def __init__(self, directory, db, maximum):
        self.directory = directory
        self.db = db
        self.maximum = maximum
        self.file = None
        self.idx = 0
        self.size = 0
        self.total = 0
        self.digest = None

    def _finish(self):
        if self.file is not None:
            self.file.close()
            self.db.execute("INSERT INTO staging VALUES(?,?,?,?)", (self.idx, str(self.path), self.digest.hexdigest(), self.size))
            self.idx += 1
            self.file = None

    def write(self, data):
        remaining = memoryview(data)
        original = len(remaining)
        while remaining:
            if self.file is None:
                self.path = self.directory / ("%06d.part" % self.idx)
                self.file = self.path.open("xb")
                self.size = 0
                self.digest = hashlib.sha256()
            amount = min(len(remaining), self.maximum - self.size)
            piece = remaining[:amount]
            self.file.write(piece)
            self.digest.update(piece)
            self.size += amount
            self.total += amount
            remaining = remaining[amount:]
            if self.size == self.maximum:
                self._finish()
        return original

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def close(self):
        self._finish()


class DiscardWriter:
    def write(self, data):
        return len(data)

    def flush(self):
        pass


def _stream_source(path, expected, output=None):
    digest = hashlib.sha256()
    scanner = SecretScanner()
    size = 0
    with _open_source(path, expected) as stream:
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                break
            scanner.feed(chunk)
            digest.update(chunk)
            size += len(chunk)
            if output is not None:
                output.write(chunk)
        if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(expected):
            raise SourceChanged("source changed during read")
    return digest.hexdigest(), size


def _source_stat(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
        raise SourceChanged("source no longer a regular file")
    return value


def capture_file(path, archive, db, part_max_bytes):
    """Return stable (object hash, size), never accepting only one source read."""
    before = _source_stat(path)
    first_hash, size = _stream_source(path, before)
    mid = _source_stat(path)
    if _fingerprint(before) != _fingerprint(mid) or size != before.st_size:
        raise SourceChanged("source changed after hash read")
    existing = db.execute("SELECT size FROM objects WHERE sha=?", (first_hash,)).fetchone()
    if existing is not None and existing[0] != size:
        raise ArchiveError("content hash size collision")
    if existing is not None:
        # Recheck every occurrence, but reuse the already verified compressed object.
        # Compressing duplicate bytes into a discard sink adds no integrity evidence.
        second_hash, second_size = _stream_source(path, mid)
        after = _source_stat(path)
        if (_fingerprint(before) != _fingerprint(after)
                or first_hash != second_hash or size != second_size):
            raise SourceChanged("source changed during duplicate verification read")
        return first_hash, size
    db.execute("DELETE FROM staging")
    with tempfile.TemporaryDirectory(prefix=".building-object-", dir=archive) as temporary:
        sink = SplitWriter(Path(temporary), db, part_max_bytes)
        try:
            with gzip.GzipFile(filename="", fileobj=sink, mode="wb", compresslevel=9, mtime=0) as compressed:
                second_hash, second_size = _stream_source(path, mid, compressed)
        finally:
            if isinstance(sink, SplitWriter):
                sink.close()
        after = _source_stat(path)
        if _fingerprint(before) != _fingerprint(after) or first_hash != second_hash or size != second_size:
            raise SourceChanged("source changed during compression read")
        if not existing:
            directory = archive / "objects" / first_hash[:2] / first_hash
            prefix = directory.parent
            prefix.mkdir(exist_ok=True)
            directory.mkdir(exist_ok=False)
            db.execute("SAVEPOINT object_capture")
            try:
                for index, filename, digest, compressed_size in db.execute("SELECT idx,path,sha,size FROM staging ORDER BY idx"):
                    relative = "objects/%s/%s/%06d.gz" % (first_hash[:2], first_hash, index)
                    target = archive / relative
                    if target.exists():
                        raise ArchiveError("object target already exists")
                    os.rename(filename, target)
                    db.execute("INSERT INTO parts VALUES(?,?,?,?,?)", (first_hash, index, relative, digest, compressed_size))
                db.execute("INSERT INTO objects VALUES(?,?,?,?)", (first_hash, size, sink.total, sink.idx))
                db.execute("RELEASE object_capture")
            except BaseException:
                db.execute("ROLLBACK TO object_capture")
                db.execute("RELEASE object_capture")
                # Only this exclusive, newly created object directory is cleaned.
                for child in directory.iterdir():
                    child.unlink()
                directory.rmdir()
                if not any(prefix.iterdir()):
                    prefix.rmdir()
                raise
        db.execute("DELETE FROM staging")
    return first_hash, size


def _walk_source(root, policy, omitted):
    """Streaming DFS; one scandir iterator per depth, no directory-size list."""
    stack = []
    try:
        stack.append(("", os.scandir(root)))
        while stack:
            parent, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                stack.pop()
                continue
            relative = parent + "/" + entry.name if parent else entry.name
            reason = policy.reason(relative)
            if reason:
                omitted(relative, reason)
                continue
            try:
                canonical_path(relative)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    omitted(relative, "symlink_or_reparse_point")
                elif stat.S_ISDIR(info.st_mode):
                    reason = policy.reason(relative, is_dir=True)
                    if reason:
                        omitted(relative, reason)
                    else:
                        try:
                            stack.append((relative, os.scandir(entry.path)))
                        except OSError:
                            omitted(relative, "unreadable_subtree")
                elif stat.S_ISREG(info.st_mode):
                    yield relative, Path(entry.path)
                else:
                    omitted(relative, "nonregular_file")
            except ArchiveError:
                omitted(relative, "unsafe_logical_path")
            except OSError:
                omitted(relative, "unreadable_entry")
    finally:
        for _, entries in stack:
            entries.close()


class MultiPartReader(io.RawIOBase):
    """Read split gzip streams without joining or allocating a whole object.

    read(-1) deliberately returns at most CHUNK; callers must iterate. Gzip's
    bounded reads and readinto are supported. Every part is hashed as consumed.
    """
    def __init__(self, archive, rows):
        super().__init__()
        self.archive = archive
        self.rows = iter(rows)
        self.file = None
        self.current = None
        self.digest = None
        self.size = 0
        self.finished = False

    def readable(self):
        return True

    def _next(self):
        try:
            relative, digest, size = next(self.rows)
        except StopIteration:
            self.finished = True
            return False
        path = self.archive / canonical_path(relative)
        _regular_file(path)
        self.file = path.open("rb")
        self.current = (digest, size)
        self.digest = hashlib.sha256()
        self.size = 0
        return True

    def _finish_part(self):
        self.file.close()
        self.file = None
        if (self.digest.hexdigest(), self.size) != self.current:
            raise ArchiveError("compressed part hash or size mismatch")

    def readinto(self, buffer):
        if self.closed:
            raise ValueError("reader is closed")
        view = memoryview(buffer)
        count = 0
        while count < len(view):
            if self.file is None and not self._next():
                break
            amount = self.file.readinto(view[count:])
            if amount:
                self.digest.update(view[count:count + amount])
                self.size += amount
                count += amount
                if self.size > self.current[1]:
                    raise ArchiveError("compressed part exceeds declared size")
            else:
                self._finish_part()
        return count

    def read(self, size=-1):
        size = CHUNK if size is None or size < 0 else min(size, CHUNK)
        data = bytearray(size)
        count = self.readinto(data)
        return bytes(memoryview(data)[:count])

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        super().close()


def _object_bytes(archive, db, sha, size, output=None):
    rows = db.execute("SELECT path,sha,size FROM parts WHERE object=? ORDER BY idx", (sha,))
    digest = hashlib.sha256()
    total = 0
    try:
        with MultiPartReader(archive, rows) as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
                while True:
                    chunk = compressed.read(CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > size:
                        raise ArchiveError("object exceeds declared uncompressed size")
                    digest.update(chunk)
                    if output is not None:
                        output.write(chunk)
            # gzip must have consumed the complete compressed stream, including
            # concatenated members. Drain only to validate physical trailing parts.
            if raw.read(1):
                raise ArchiveError("unconsumed object bytes")
    except (OSError, EOFError, ValueError) as exc:
        raise ArchiveError("invalid or missing compressed object") from exc
    if digest.hexdigest() != sha or total != size:
        raise ArchiveError("object uncompressed hash or size mismatch")


def _make_parents(root, relative):
    canonical_path(relative)
    target = root.joinpath(*relative.split("/"))
    if not _within(target.absolute(), root.absolute()):
        raise ArchiveError("destination escapes root")
    current = root
    for part in relative.split("/")[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            value = current.lstat()
            if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
                raise ArchiveError("unsafe destination parent")
        else:
            current.mkdir()
    _check_ancestors(target.parent)
    return target


def _materialize(archive, db, relative, sha, size, root):
    target = _make_parents(root, relative)
    with target.open("xb") as output:
        _object_bytes(archive, db, sha, size, output)


def _physical_name_gate(path, is_dir=False):
    """Reject protected physical ancestors before any filesystem inspection."""
    policy = Policy()
    for index, component in enumerate(path.parts):
        if component == path.anchor:
            continue
        reason = policy.reason(component, is_dir=(is_dir or index != len(path.parts) - 1))
        if reason:
            raise ArchiveError("physical source path is outside publication policy")


def snapshot(source, archive, extras=(), allowed_program_files=PROGRAM_FILES, part_max_bytes=MAX_PART_BYTES):
    source, archive = Path(source).absolute(), Path(archive).absolute()
    _integer(part_max_bytes, "part byte limit", 1)
    if part_max_bytes > MAX_PART_BYTES:
        raise ArchiveError("part byte limit exceeds 32 MiB")
    _physical_name_gate(source, is_dir=True)
    validated_extras = []  # Fixed maximum of four external metadata files.
    for logical, physical in extras:
        canonical_path(logical)
        if logical not in {"_supplement/accidental_checkout_v10/" + name for name in SUPPLEMENT_NAMES}:
            raise ArchiveError("supplement must be one of the four documented v10 metadata files")
        if logical in {item[0] for item in validated_extras}:
            raise ArchiveError("duplicate supplement path")
        physical = Path(physical).absolute()
        _physical_name_gate(physical)
        if physical.name != logical.rsplit("/", 1)[1]:
            raise ArchiveError("supplement basename must match its documented logical identity")
        validated_extras.append((logical, physical))
    _check_ancestors(source)
    _check_ancestors(archive.parent)
    if not source.is_dir() or _within(archive.resolve(), source.resolve()):
        raise ArchiveError("archive must be outside the source tree")
    policy = Policy(allowed_program_files)
    for logical, physical in validated_extras:
        _regular_file(physical)
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "objects").mkdir()
    (archive / "source").mkdir()
    paths = CatalogWriter(archive / "paths.jsonl.gz")
    omissions = CatalogWriter(archive / "omissions.jsonl.gz")
    reasons = Counter()
    totals = Counter()
    with tempfile.TemporaryDirectory(prefix="publication-index-") as temporary:
        with contextlib.closing(_database(Path(temporary) / "index.sqlite3")) as db:
            def omitted(relative, reason):
                try:
                    db.execute("INSERT INTO omissions VALUES(?)", (relative.casefold(),))
                except sqlite3.IntegrityError as exc:
                    raise ArchiveError("duplicate omission path") from exc
                omissions.add({"path": relative, "reason": reason})
                reasons[reason] += 1

            def include(relative, physical, origin):
                try:
                    sha, size = capture_file(physical, archive, db, part_max_bytes)
                except SecretFound:
                    omitted(relative, "secret_signature")
                    return
                except SourceChanged:
                    omitted(relative, "source_changed")
                    return
                except OSError:
                    omitted(relative, "unreadable_or_changed_file")
                    return
                mirror = policy.materialize(relative)
                _remember_parents(db, relative)
                try:
                    db.execute("INSERT INTO paths VALUES(?,?,?,?,?)", (relative, relative.casefold(), sha, size, int(mirror)))
                except sqlite3.IntegrityError as exc:
                    raise ArchiveError("duplicate or case-alias source path") from exc
                paths.add({"path": relative, "object": sha, "size": size, "materialized": mirror, "origin": origin})
                totals["occurrences"] += 1
                totals["occurrence_bytes"] += size
                if mirror:
                    _materialize(archive, db, relative, sha, size, archive / "source")
                    totals["materialized_files"] += 1
                    totals["materialized_bytes"] += size
                if origin == "supplement":
                    totals["supplement_files"] += 1

            try:
                for relative, physical in _walk_source(source, policy, omitted):
                    include(relative, physical, "foundry")
                for relative, physical in sorted(validated_extras):
                    include(relative, physical, "supplement")
                objects = CatalogWriter(archive / "objects.jsonl.gz")
                try:
                    for sha, size, compressed_size, parts in db.execute("SELECT * FROM objects ORDER BY sha"):
                        objects.add({"type": "object", "sha256": sha, "size": size, "compressed_size": compressed_size, "parts": parts})
                        totals["unique_objects"] += 1
                        totals["object_bytes"] += size
                        totals["compressed_object_bytes"] += compressed_size
                        totals["object_parts"] += parts
                        for index, relative, digest, amount in db.execute("SELECT idx,path,sha,size FROM parts WHERE object=? ORDER BY idx", (sha,)):
                            objects.add({"type": "part", "object": sha, "index": index, "path": relative, "sha256": digest, "size": amount})
                finally:
                    object_descriptor = objects.close()
            finally:
                path_descriptor = paths.close()
                omission_descriptor = omissions.close()
    totals["omission_rows"] = sum(reasons.values())
    counts = {name: totals[name] for name in ("occurrences", "occurrence_bytes", "unique_objects", "object_bytes", "compressed_object_bytes", "object_parts", "materialized_files", "materialized_bytes", "supplement_files", "omission_rows")}
    capture_issues = sum(reasons[reason] for reason in ("secret_signature", "source_changed", "unreadable_or_changed_file", "unreadable_entry", "unreadable_subtree", "unsafe_logical_path"))
    report = {"format": FORMAT, "policy": policy.describe(), "counts": counts, "omissions_by_reason": dict(sorted(reasons.items())),
              "capture_issues": capture_issues, "all_included_files_passed_two_read_stability": True,
              "atomic_source_freeze_claim": False, "original_source_written": False,
              "status": "capture_issues_recorded" if capture_issues else "completed_with_explicit_policy_omissions",
              "limits": ["File stability is checked per capture, not atomically across the source tree.",
                         "Credential signatures are heuristic and cannot prove absence of every possible secret.",
                         "Filesystem iteration order may vary; gzip bytes and object hashes are deterministic.",
                         "Integrity requires a trusted catalog or externally pinned catalog SHA256; this is not a signed archive."]}
    report_data = _json_bytes(report) + b"\n"
    with (archive / "report.json").open("xb") as stream:
        stream.write(report_data)
    manifest = {"format": FORMAT, "object_encoding": "gzip-stream-split", "part_max_bytes": part_max_bytes,
                "catalogs": {"paths.jsonl.gz": path_descriptor, "objects.jsonl.gz": object_descriptor, "omissions.jsonl.gz": omission_descriptor},
                "report_sha256": hashlib.sha256(report_data).hexdigest(), "counts": counts}
    with (archive / "manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest) + b"\n")
    return report


def _hash_file(path):
    digest = hashlib.sha256()
    size = 0
    _regular_file(path)
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                return digest.hexdigest(), size
            digest.update(chunk)
            size += len(chunk)


def _small_json(path):
    _regular_file(path)
    with path.open("rb") as stream:
        value = stream.read(MAX_RECORD_BYTES + 1)
    if len(value) > MAX_RECORD_BYTES:
        raise ArchiveError("metadata exceeds size limit")
    return _parse_json(value)


def _catalog_rows(archive, name, expected):
    if not isinstance(expected, dict):
        raise ArchiveError("invalid catalog descriptor")
    path = archive / name
    if _hash_file(path) != (_sha(expected.get("compressed_sha256")), _integer(expected.get("compressed_size"), "compressed catalog size")):
        raise ArchiveError("compressed catalog hash or size mismatch")
    digest, size, rows = hashlib.sha256(), 0, 0
    try:
        with gzip.open(path, "rb") as stream:
            while True:
                data = stream.readline(MAX_RECORD_BYTES + 1)
                if not data:
                    break
                if len(data) > MAX_RECORD_BYTES or not data.endswith(b"\n"):
                    raise ArchiveError("oversized or unterminated JSONL record")
                digest.update(data)
                size += len(data)
                rows += 1
                if size > _integer(expected.get("size"), "catalog size") or rows > _integer(expected.get("records"), "catalog record count"):
                    raise ArchiveError("catalog exceeds declared limits")
                yield _parse_json(data)
    except (OSError, EOFError) as exc:
        raise ArchiveError("invalid gzip catalog") from exc
    if (digest.hexdigest(), size, rows) != (_sha(expected.get("sha256")), expected["size"], expected["records"]):
        raise ArchiveError("catalog uncompressed hash, size or record count mismatch")


def _scan_archive_tree(root, callback):
    stack = []
    try:
        _check_ancestors(root)
        if not root.is_dir():
            raise ArchiveError("missing archive directory")
        stack.append(("", os.scandir(root)))
        while stack:
            parent, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                stack.pop()
                continue
            relative = parent + "/" + entry.name if parent else entry.name
            canonical_path(relative)
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ArchiveError("symlink or reparse point in archive")
            is_dir = stat.S_ISDIR(info.st_mode)
            if not is_dir and not stat.S_ISREG(info.st_mode):
                raise ArchiveError("nonregular archive entry")
            callback(relative, Path(entry.path), is_dir)
            if is_dir:
                stack.append((relative, os.scandir(entry.path)))
    finally:
        for _, entries in stack:
            entries.close()


def _remember_parents(db, relative):
    parent = relative.rpartition("/")[0]
    while parent:
        existing = db.execute("SELECT path FROM directories WHERE key=?", (parent.casefold(),)).fetchone()
        if existing is not None and existing[0] != parent:
            raise ArchiveError("case-alias directory in catalog")
        db.execute("INSERT OR IGNORE INTO directories VALUES(?,?)", (parent, parent.casefold()))
        parent = parent.rpartition("/")[0]


def _verify_into(archive, db, expected_catalog_sha256=None):
    manifest = _small_json(archive / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("object_encoding") != "gzip-stream-split":
        raise ArchiveError("unsupported archive format")
    maximum = _integer(manifest.get("part_max_bytes"), "part byte limit", 1)
    if maximum > MAX_PART_BYTES:
        raise ArchiveError("part byte limit exceeds 32 MiB")
    catalogs = manifest.get("catalogs")
    if not isinstance(catalogs, dict) or set(catalogs) != {"paths.jsonl.gz", "objects.jsonl.gz", "omissions.jsonl.gz"}:
        raise ArchiveError("unexpected catalog set")
    if expected_catalog_sha256 is not None and _sha(expected_catalog_sha256) != catalogs["paths.jsonl.gz"].get("compressed_sha256"):
        raise ArchiveError("externally pinned path catalog SHA256 mismatch")
    report_hash, _ = _hash_file(archive / "report.json")
    if report_hash != _sha(manifest.get("report_sha256")):
        raise ArchiveError("report hash mismatch")
    report = _small_json(archive / "report.json")
    if report.get("format") != FORMAT:
        raise ArchiveError("unexpected report format")
    for required_directory in (archive / "objects", archive / "source"):
        _check_ancestors(required_directory)
        if not required_directory.is_dir():
            raise ArchiveError("missing archive object or mirror directory")
    for row in _catalog_rows(archive, "objects.jsonl.gz", catalogs["objects.jsonl.gz"]):
        try:
            if row.get("type") == "object":
                db.execute("INSERT INTO objects VALUES(?,?,?,?)", (_sha(row.get("sha256")), _integer(row.get("size"), "object size"), _integer(row.get("compressed_size"), "compressed object size", 1), _integer(row.get("parts"), "part count", 1)))
            elif row.get("type") == "part":
                sha = _sha(row.get("object"))
                index = _integer(row.get("index"), "part index")
                relative = canonical_path(row.get("path"))
                if relative != "objects/%s/%s/%06d.gz" % (sha[:2], sha, index):
                    raise ArchiveError("noncanonical object part path")
                size = _integer(row.get("size"), "part size", 1)
                if size > maximum or not db.execute("SELECT 1 FROM objects WHERE sha=?", (sha,)).fetchone():
                    raise ArchiveError("oversized part or unknown object")
                db.execute("INSERT INTO parts VALUES(?,?,?,?,?)", (sha, index, relative, _sha(row.get("sha256")), size))
            else:
                raise ArchiveError("unknown object catalog row type")
        except sqlite3.IntegrityError as exc:
            raise ArchiveError("duplicate object or object part") from exc
    totals = Counter()
    for sha, size, compressed_size, count in db.execute("SELECT * FROM objects"):
        observed = db.execute("SELECT COUNT(*),MIN(idx),MAX(idx),SUM(size) FROM parts WHERE object=?", (sha,)).fetchone()
        if observed != (count, 0, count - 1, compressed_size):
            raise ArchiveError("missing, noncontiguous or size-mismatched object parts")
        _object_bytes(archive, db, sha, size)
        totals["unique_objects"] += 1
        totals["object_bytes"] += size
        totals["compressed_object_bytes"] += compressed_size
        totals["object_parts"] += count
    for row in _catalog_rows(archive, "paths.jsonl.gz", catalogs["paths.jsonl.gz"]):
        relative = canonical_path(row.get("path"))
        sha = _sha(row.get("object"))
        size = _integer(row.get("size"), "path size")
        mirror = row.get("materialized")
        if not isinstance(mirror, bool) or row.get("origin") not in ("foundry", "supplement"):
            raise ArchiveError("invalid materialization or origin field")
        object_row = db.execute("SELECT size FROM objects WHERE sha=?", (sha,)).fetchone()
        if object_row != (size,):
            raise ArchiveError("missing object or path/object size mismatch")
        _remember_parents(db, relative)
        try:
            db.execute("INSERT INTO paths VALUES(?,?,?,?,?)", (relative, relative.casefold(), sha, size, int(mirror)))
        except sqlite3.IntegrityError as exc:
            raise ArchiveError("duplicate or case-alias catalog path") from exc
        # Also reject ancestor/file collisions before restore or mirror writes.
        parent = relative.rpartition("/")[0]
        while parent:
            if db.execute("SELECT 1 FROM paths WHERE key=?", (parent.casefold(),)).fetchone():
                raise ArchiveError("catalog file/ancestor collision")
            parent = parent.rpartition("/")[0]
        if db.execute("SELECT 1 FROM paths WHERE key LIKE ? ESCAPE '!'", (relative.casefold().replace("!", "!!").replace("%", "!%").replace("_", "!_") + "/%",)).fetchone():
            raise ArchiveError("catalog file/descendant collision")
        totals["occurrences"] += 1
        totals["occurrence_bytes"] += size
        if row["origin"] == "supplement":
            totals["supplement_files"] += 1
        if mirror:
            totals["materialized_files"] += 1
            totals["materialized_bytes"] += size
    if db.execute("SELECT 1 FROM objects WHERE NOT EXISTS(SELECT 1 FROM paths WHERE paths.object=objects.sha) LIMIT 1").fetchone():
        raise ArchiveError("unreferenced archive object")
    reasons = Counter()
    for row in _catalog_rows(archive, "omissions.jsonl.gz", catalogs["omissions.jsonl.gz"]):
        reason = row.get("reason")
        relative = row.get("path")
        if reason == "unsafe_logical_path":
            # Inert source-name evidence only: never join or restore this value.
            if not isinstance(relative, str) or not relative or len(relative.encode("utf-8")) > MAX_RECORD_BYTES:
                raise ArchiveError("invalid omitted source-name evidence")
            try:
                canonical_path(relative)
            except ArchiveError:
                pass
            else:
                raise ArchiveError("unsafe-path omission contains a canonical path")
        else:
            relative = canonical_path(relative)
        if not isinstance(reason, str) or reason not in {
            "unreferenced_program_subtree", "weights_packages_or_cache", "credential_filename", "weights_packages_or_database", "model_weight_filename", "possible_raw_corpus_outside_owned_pilot", "live_lock_or_temporary_file", "protected_subtree_or_filename", "symlink_or_reparse_point", "unreadable_subtree", "nonregular_file", "unsafe_logical_path", "unreadable_entry", "secret_signature", "source_changed", "unreadable_or_changed_file",
        }:
            raise ArchiveError("invalid omission reason")
        if db.execute("SELECT 1 FROM paths WHERE key=?", (relative.casefold(),)).fetchone():
            raise ArchiveError("path is both included and omitted")
        try:
            db.execute("INSERT INTO omissions VALUES(?)", (relative.casefold(),))
        except sqlite3.IntegrityError as exc:
            raise ArchiveError("duplicate omission path") from exc
        totals["omission_rows"] += 1
        reasons[reason] += 1
    counts = {name: totals[name] for name in ("occurrences", "occurrence_bytes", "unique_objects", "object_bytes", "compressed_object_bytes", "object_parts", "materialized_files", "materialized_bytes", "supplement_files", "omission_rows")}
    if manifest.get("counts") != counts or report.get("counts") != counts or report.get("omissions_by_reason") != dict(reasons):
        raise ArchiveError("archive count or omission summary mismatch")
    db.execute("DELETE FROM directories")
    for (relative,) in db.execute("SELECT path FROM parts"):
        _remember_parents(db, relative)
    for (relative,) in db.execute("SELECT path FROM paths WHERE materialized=1"):
        _remember_parents(db, "source/" + relative)
    db.execute("INSERT OR IGNORE INTO directories VALUES('objects','objects')")
    db.execute("INSERT OR IGNORE INTO directories VALUES('source','source')")
    required_files = {"manifest.json", "report.json", "paths.jsonl.gz", "objects.jsonl.gz", "omissions.jsonl.gz"}
    observed_fixed = set()
    observed_parts = 0
    observed_mirror = 0

    def inspect(relative, physical, is_dir):
        nonlocal observed_parts, observed_mirror
        if is_dir:
            if not db.execute("SELECT 1 FROM directories WHERE path=?", (relative,)).fetchone():
                raise ArchiveError("unknown archive directory")
        elif relative in required_files:
            observed_fixed.add(relative)
        elif relative.startswith("objects/"):
            if not db.execute("SELECT 1 FROM parts WHERE path=?", (relative,)).fetchone():
                raise ArchiveError("unknown archive object part")
            observed_parts += 1
        elif relative.startswith("source/"):
            record = db.execute("SELECT object,size FROM paths WHERE path=? AND materialized=1", (relative[7:],)).fetchone()
            if record is None or _hash_file(physical) != record:
                raise ArchiveError("unknown or changed materialized file")
            observed_mirror += 1
        else:
            raise ArchiveError("unknown archive file")

    _scan_archive_tree(archive, inspect)
    if observed_fixed != required_files or observed_parts != counts["object_parts"] or observed_mirror != counts["materialized_files"]:
        raise ArchiveError("missing archive or materialized files")
    return {"verified": True, "format": FORMAT, "counts": counts,
            "paths_catalog_sha256": catalogs["paths.jsonl.gz"]["compressed_sha256"], "capture_issues": report.get("capture_issues")}


def verify(archive, expected_catalog_sha256=None):
    archive = Path(archive).absolute()
    try:
        with tempfile.TemporaryDirectory(prefix="publication-verify-") as temporary:
            with contextlib.closing(_database(Path(temporary) / "index.sqlite3")) as db:
                return _verify_into(archive, db, expected_catalog_sha256)
    except OSError as exc:
        raise ArchiveError("missing or unreadable archive component") from exc


def restore(archive, destination, expected_catalog_sha256=None):
    archive, destination = Path(archive).absolute(), Path(destination).absolute()
    _check_ancestors(destination)
    if _within(destination.resolve(), archive.resolve()) or _within(archive.resolve(), destination.resolve()):
        raise ArchiveError("restore destination and archive must be disjoint")
    if destination.exists():
        if not destination.is_dir():
            raise ArchiveError("restore destination must be a directory")
        with os.scandir(destination) as entries:
            if next(entries, None) is not None:
                raise ArchiveError("restore destination must be empty; overwrites are forbidden")
    with tempfile.TemporaryDirectory(prefix="publication-restore-") as temporary:
        with contextlib.closing(_database(Path(temporary) / "index.sqlite3")) as db:
            result = _verify_into(archive, db, expected_catalog_sha256)
            if not destination.exists():
                destination.mkdir(parents=True, exist_ok=False)
            for relative, sha, size in db.execute("SELECT path,object,size FROM paths ORDER BY path"):
                _materialize(archive, db, relative, sha, size, destination)
            result["restored_files"] = result["counts"]["occurrences"]
            return result


def _extra_argument(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("extra must be LOGICAL=FILE")
    logical, physical = value.split("=", 1)
    return logical, physical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("snapshot", help="capture a new archive without writing the source")
    build.add_argument("source")
    build.add_argument("archive")
    build.add_argument("--extra", action="append", default=[], type=_extra_argument, metavar="LOGICAL=FILE")
    build.add_argument("--allow-program-file", action="append", dest="allowed_program_files", help="repeat exactly the four documented program reference paths")
    build.add_argument("--part-max-bytes", type=int, default=MAX_PART_BYTES)
    check = commands.add_parser("verify", help="fully verify catalogs, all object bytes and the readable mirror")
    check.add_argument("archive")
    check.add_argument("--catalog-sha256", help="trusted compressed paths.jsonl.gz SHA256")
    unpack = commands.add_parser("restore", help="verify then restore into an empty destination using exclusive writes")
    unpack.add_argument("archive")
    unpack.add_argument("destination")
    unpack.add_argument("--catalog-sha256", help="trusted compressed paths.jsonl.gz SHA256")
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            result = snapshot(args.source, args.archive, args.extra, args.allowed_program_files or PROGRAM_FILES, args.part_max_bytes)
            code = 2 if result["capture_issues"] else 0
        elif args.command == "verify":
            result, code = verify(args.archive, args.catalog_sha256), 0
        else:
            result, code = restore(args.archive, args.destination, args.catalog_sha256), 0
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return code
    except (ArchiveError, OSError, sqlite3.Error) as exc:
        # Exception texts here are structural messages or OS paths, never bytes.
        print("publication error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
