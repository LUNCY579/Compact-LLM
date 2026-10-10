#!/usr/bin/env python3
"""Read-only, byte-preserving TRAIN-declaration backup; Python standard library.

No network, model, tokenizer, training, or source-writing capabilities. Production
capture requires an independent review receipt matching the exact code/config.
The default CLI only describes the prospective selection and does not capture.
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
import stat
import sys
import tarfile
from datetime import datetime, timezone

SOURCE_ROOT = Path("C:/Users/clclc/BusinessProject/Rineya-Dataset")
INVENTORY_RELATIVE = "backup_20261010/INVENTORY_METADATA003.json"
INVENTORY_SHA256 = "a034f6a694868503afc402f0d2bd925497a49c8e99c0a638395651da0e816640"
DESTINATION = "Luncy1978/rineya-crs-001-research"
NAMESPACE = "rineya-dataset-snapshots"
FORMAT = "rineya-dataset-byte-backup-v1"
SELECTION_FORMAT = "rineya-dataset-byte-backup-selection-v1"
TRAIN_ROLES = frozenset(("train_candidate_or_train_audit", "declared_training_export_pending_source_status"))
METADATA_ROLE = "source_document_or_metadata"
CHUNK = 1024 * 1024
METADATA_LIMIT = 262144
CONFIG_LIMIT = 16 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
RESERVED = re.compile(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z", re.I)
BLOCKED_DIRS = frozenset((".git", ".cache", "cache", "caches", "weights", "models", "checkpoints", "node_modules", ".venv", "venv", "packages", "downloads", "program", "programs", "queue", "queues", "state", "locks", "__pycache__"))
PROTECTED = frozenset(("check", "final", "eval", "evaluation", "evaluator", "holdout", "spent", "private", "privatebank", "sealed"))
FORBIDDEN_EXTENSIONS = frozenset((".bin", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".gguf", ".ggml", ".pkl", ".pickle", ".sqlite", ".sqlite3", ".db", ".db3", ".mdb", ".zip", ".tar", ".7z", ".parquet", ".arrow", ".py", ".ps1", ".sh", ".js", ".ts", ".exe", ".dll"))
METADATA_NAME = re.compile(r"(?:readme.*|data(?:set)?_card.*|sources?.*|split.*|provenance.*|.*manifest.*)\.(?:md|txt|json|yaml|yml)\Z", re.I)
CLAIMS = {"source_owner": "UNKNOWN", "canonicality": "UNKNOWN", "native_boundary": "UNKNOWN", "model_boundary": "UNKNOWN", "rights_use": "UNKNOWN", "training_ready": False, "training_authorized": False, "training_eligible": False, "measured_gains": "UNMEASURED"}


class BackupError(Exception):
    """Structural failure whose text never includes source payload bytes."""


class SourceChanged(BackupError):
    pass


class SecretFound(BackupError):
    pass


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError("duplicate_json_key")
        result[key] = value
    return result


def _parse(data):
    try:
        value = json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=lambda _: (_ for _ in ()).throw(BackupError("nonfinite_json")))
    except (ValueError, UnicodeError) as exc:
        raise BackupError("invalid_json") from exc
    if not isinstance(value, dict):
        raise BackupError("json_object_required")
    return value


def canonical_path(value):
    """Portable logical path; never normalize a source name into another name."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or value.startswith("/") or any(ord(c) < 32 for c in value):
        raise BackupError("unsafe_logical_path")
    for part in value.split("/"):
        if part in ("", ".", "..") or part.endswith((".", " ")) or RESERVED.fullmatch(part):
            raise BackupError("unsafe_logical_path")
    return value


def file_id(relative, role):
    """Identity assigned from the original path and role, before any capture."""
    return hashlib.sha256(json_bytes([relative, role])).hexdigest()


def path_gate(relative, declared_role, size):
    """Pure name/role/size gate: no stat, resolve, glob, enumeration, or reads."""
    try:
        canonical_path(relative)
    except BackupError:
        return "unsafe_logical_path"
    if declared_role not in TRAIN_ROLES | {METADATA_ROLE}:
        return "undeclared_role"
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return "invalid_recorded_size"
    parts = relative.casefold().split("/")
    base = parts[-1]
    if any(p in BLOCKED_DIRS for p in parts):
        return "forbidden_subtree"
    for part in parts:
        words = set(re.split(r"[^a-z0-9]+", part))
        if words & PROTECTED or any(w.startswith(("evaluation", "evaluator")) for w in words) or any(p in part for p in ("privatebank", "private_bank", "sealedpayload", "sealed_payload")):
            return "protected_path"
    if base in (".env", ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json", "secrets", "secrets.json", "id_rsa", "id_ed25519") or base.startswith(".env."):
        return "credential_filename"
    if base.endswith((".lock", ".pid", ".tmp", ".temp", ".part", ".swp", ".sqlite-wal", ".sqlite-shm", ".db-wal", ".db-shm")) or base.startswith(("~$", ".~")):
        return "live_state_or_temporary"
    if Path(base).suffix in FORBIDDEN_EXTENSIONS:
        return "opaque_program_weight_or_database"
    if declared_role in TRAIN_ROLES:
        return None if base.endswith((".jsonl", ".jsonl.gz")) else "undeclared_train_encoding"
    if size > METADATA_LIMIT or not METADATA_NAME.fullmatch(base):
        return "not_small_metadata"
    if any(p in ("examples", "tests", "fixtures", "pilot", "evidence", "protocol", "protocols") for p in parts):
        return "metadata_scope_uncertain"
    return None


def prepare_selection(inventory, inventory_sha256):
    """Project inventory labels only; payloads are never inspected here."""
    if not SHA.fullmatch(inventory_sha256) or inventory.get("status") != "METADATA_ONLY_NOT_PREPARED":
        raise BackupError("inventory_pin_or_status_invalid")
    source = inventory.get("source")
    if source is not None and str(source).replace("\\", "/").casefold() != str(SOURCE_ROOT).replace("\\", "/").casefold():
        raise BackupError("inventory_source_mismatch")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise BackupError("inventory_files_required")
    train = [row for row in files if row.get("backup_role") in TRAIN_ROLES]
    roots = {row["path"].split("/")[0] for row in train if isinstance(row.get("path"), str)}
    rows = []
    for source_row in files:
        role, path, size = source_row.get("backup_role"), source_row.get("path"), source_row.get("bytes")
        kind = "declared_train" if role in TRAIN_ROLES else "small_metadata"
        if role not in TRAIN_ROLES:
            if role != METADATA_ROLE or not isinstance(path, str) or ("/" in path and path.split("/")[0] not in roots) or path_gate(path, role, size) is not None:
                continue
        if source_row.get("training_eligible") is not False:
            raise BackupError("inventory_eligibility_must_remain_false")
        reason = path_gate(path, role, size)
        rows.append({"path": path, "bytes": size, "backup_role": role, "training_eligible": False, "file_id": file_id(path, role), "selection_kind": kind, "path_gate_status": "BLOCKED" if reason else "ALLOWED_PROSPECTIVELY", "path_gate_reason": reason})
    rows.sort(key=lambda row: (row["path"], row["backup_role"]))
    return {"schema": SELECTION_FORMAT, "status": "PROSPECTIVE_NOT_CAPTURED", "source_root": str(SOURCE_ROOT).replace("\\", "/"), "inventory_sha256": inventory_sha256, "inventory_status": inventory["status"], "destination": DESTINATION, "namespace": NAMESPACE, "claims": dict(CLAIMS), "counts": {"declared_train_files": len(train), "declared_train_recorded_bytes": sum(row["bytes"] for row in train), "metadata_files": sum(row["selection_kind"] == "small_metadata" for row in rows), "selected_files": len(rows), "selected_recorded_bytes": sum(row["bytes"] for row in rows)}, "rows": rows}


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _within(child, parent):
    try:
        return os.path.commonpath((os.path.normcase(str(child)), os.path.normcase(str(parent)))) == os.path.normcase(str(parent))
    except ValueError:
        return False


def _check_ancestors(path):
    """Use lstat, including ancestors, without following a reparse point."""
    for item in reversed((path,) + tuple(path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise BackupError("symlink_or_reparse_point")


def _safe_file(root, row):
    reason = path_gate(row["path"], row["backup_role"], row["bytes"])
    if reason:
        raise BackupError(reason)
    # Name gate must run before every resolve/stat/open on the selected path.
    path = root.joinpath(*row["path"].split("/"))
    _check_ancestors(path)
    resolved = path.resolve(strict=True)
    if not _within(resolved, root) or resolved == root:
        raise BackupError("source_outside_exact_root")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise BackupError("nonregular_source")
    if info.st_size != row["bytes"]:
        raise SourceChanged("recorded_size_changed")
    return path, info


def _open_source(path, expected):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    stream = os.fdopen(descriptor, "rb")
    if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(expected):
        stream.close()
        raise SourceChanged("source_changed_before_open")
    return stream


class SecretScanner:
    """Streaming signatures, adapted from publication_tools/snapshot.py.

    A negative scan is not a claim that every possible secret is absent.
    No matched token or source text enters receipts or exceptions.
    """
    patterns = (
        ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"), False),
        ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), True),
        ("github_token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,255}\b"), True),
        ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), False),
        ("openai_key", re.compile(rb"\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{32,512}\b"), True),
        ("huggingface_token", re.compile(rb"\bhf_[A-Za-z0-9]{30,255}\b"), True),
    )

    def __init__(self):
        self.tail = b""

    def feed(self, data):
        window = self.tail + data
        for kind, expression, entropy_required in self.patterns:
            for match in expression.finditer(window):
                value = match.group(0)
                counts = Counter(value)
                entropy = -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values())
                if not entropy_required or (len(counts) >= 12 and entropy >= 3.5):
                    raise SecretFound("secret_signature_" + kind)
        self.tail = window[-2048:]


class HashingReader:
    def __init__(self, stream, scan=False):
        self.stream, self.digest, self.size = stream, hashlib.sha256(), 0
        self.scanner = SecretScanner() if scan else None

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.size += len(data)
        if self.scanner:
            self.scanner.feed(data)
        return data

    def tell(self):
        return self.size


def _assert_stable(root, row, path, stream, expected):
    if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(expected):
        raise SourceChanged("source_changed_during_read")
    checked_path, current = _safe_file(root, row)
    if checked_path != path or _fingerprint(current) != _fingerprint(expected):
        raise SourceChanged("source_changed_after_read")


def scan_file(root, row):
    path, before = _safe_file(root, row)
    with _open_source(path, before) as stream:
        reader = HashingReader(stream, scan=True)
        if row["path"].casefold().endswith(".jsonl.gz"):
            expanded_scanner = SecretScanner()
            try:
                with gzip.GzipFile(fileobj=reader, mode="rb") as expanded:
                    while True:
                        data = expanded.read(CHUNK)
                        if not data:
                            break
                        expanded_scanner.feed(data)
            except (EOFError, gzip.BadGzipFile, OSError) as exc:
                raise BackupError("invalid_declared_jsonl_gzip") from exc
        while reader.read(CHUNK):
            pass
        _assert_stable(root, row, path, stream, before)
        if reader.size != row["bytes"]:
            raise SourceChanged("source_read_size_changed")
    return {"file_id": row["file_id"], "path": row["path"], "backup_role": row["backup_role"], "training_eligible": False, "size": reader.size, "sha256": reader.digest.hexdigest(), "stat_fingerprint": list(_fingerprint(before)), "member": "source/" + row["path"]}


def _add_file(archive, root, row, expected):
    path, before = _safe_file(root, row)
    if list(_fingerprint(before)) != expected["stat_fingerprint"]:
        raise SourceChanged("source_changed_between_reads")
    with _open_source(path, before) as stream:
        reader = HashingReader(stream)
        info = tarfile.TarInfo(expected["member"])
        info.size, info.mtime, info.mode = expected["size"], 0, 0o444
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        archive.addfile(info, reader)
        extra = reader.read(1)
        _assert_stable(root, row, path, stream, before)
        if extra or reader.size != expected["size"] or reader.digest.hexdigest() != expected["sha256"]:
            raise SourceChanged("independent_hash_or_size_mismatch")


def _hash_file(path):
    _check_ancestors(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise BackupError("nonregular_local_file")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
        if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(before):
            raise BackupError("local_file_changed")
    if _fingerprint(path.lstat()) != _fingerprint(before):
        raise BackupError("local_file_changed")
    return digest.hexdigest(), size


def _content_descriptor(manifest):
    return {"format": FORMAT, "inventory_sha256": manifest["inventory_sha256"], "claims": CLAIMS, "files": [{k: row[k] for k in ("file_id", "path", "backup_role", "training_eligible", "size", "sha256", "member")} for row in manifest["files"]]}


def verify_archive(archive_path, manifest):
    """Hash every file member; reject extras, duplicates, traversal, and types.

    Drain gzip fully after tar EOF, verifying its checksum and requiring only
    zero tar padding. Hash the entire compressed file in this same readback.
    """
    if manifest.get("format") != FORMAT or manifest.get("claims") != CLAIMS:
        raise BackupError("invalid_manifest_configuration")
    archive_path = Path(archive_path)
    _check_ancestors(archive_path)
    before = archive_path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise BackupError("nonregular_archive")
    expected, aliases = {}, set()
    for row in manifest["files"]:
        canonical_path(row["path"])
        if path_gate(row["path"], row["backup_role"], row["size"]) is not None or not SHA.fullmatch(row["sha256"]):
            raise BackupError("manifest_path_or_digest_not_allowed")
        if row["member"] != "source/" + row["path"] or row["file_id"] != file_id(row["path"], row["backup_role"]) or row["training_eligible"] is not False:
            raise BackupError("manifest_identity_mismatch")
        if row["member"] in expected or row["member"].casefold() in aliases:
            raise BackupError("duplicate_manifest_member")
        expected[row["member"]] = row
        aliases.add(row["member"].casefold())
    observed, verified_bytes = set(), 0
    try:
        with archive_path.open("rb") as raw:
            compressed = HashingReader(raw)
            with gzip.GzipFile(fileobj=compressed, mode="rb") as expanded:
                with tarfile.open(fileobj=expanded, mode="r|") as archive:
                    for member in archive:
                        canonical_path(member.name)
                        if not member.isfile() or member.name not in expected or member.name in observed or member.linkname or member.sparse is not None:
                            raise BackupError("extra_duplicate_or_nonregular_archive_member")
                        row = expected[member.name]
                        if member.size != row["size"]:
                            raise BackupError("archive_member_size_mismatch")
                        digest, count = hashlib.sha256(), 0
                        with contextlib.closing(archive.extractfile(member)) as stream:
                            while True:
                                chunk = stream.read(CHUNK)
                                if not chunk:
                                    break
                                digest.update(chunk)
                                count += len(chunk)
                        if count != row["size"] or digest.hexdigest() != row["sha256"]:
                            raise BackupError("archive_member_hash_mismatch")
                        observed.add(member.name)
                        verified_bytes += count
                    # tarfile's stream adapter may have prefetched bytes beyond
                    # the first EOF block. Drain that buffer before gzip itself.
                    while True:
                        padding = archive.fileobj.read(CHUNK)
                        if not padding:
                            break
                        if padding.strip(b"\0"):
                            raise BackupError("nonzero_trailing_tar_bytes")
                while True:
                    padding = expanded.read(CHUNK)
                    if not padding:
                        break
                    if padding.strip(b"\0"):
                        raise BackupError("nonzero_trailing_tar_bytes")
            while compressed.read(CHUNK):
                raise BackupError("trailing_compressed_archive_bytes")
            if _fingerprint(os.fstat(raw.fileno())) != _fingerprint(before):
                raise BackupError("archive_changed_during_readback")
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, OSError) as exc:
        raise BackupError("invalid_or_unreadable_archive") from exc
    if observed != set(expected) or _fingerprint(archive_path.lstat()) != _fingerprint(before):
        raise BackupError("missing_member_or_archive_changed")
    result = {"verified": True, "archive_sha256": compressed.digest.hexdigest(), "archive_size": compressed.size, "verified_files": len(observed), "verified_bytes": verified_bytes}
    if manifest.get("archive_sha256") not in (None, result["archive_sha256"]) or manifest.get("archive_size") not in (None, result["archive_size"]):
        raise BackupError("archive_descriptor_mismatch")
    if manifest.get("content_sha256") not in (None, hashlib.sha256(json_bytes(_content_descriptor(manifest))).hexdigest()):
        raise BackupError("content_descriptor_mismatch")
    return result


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write(json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _local_json(path):
    _check_ancestors(path)
    with path.open("rb") as stream:
        data = stream.read(CONFIG_LIMIT + 1)
    if len(data) > CONFIG_LIMIT:
        raise BackupError("configuration_too_large")
    return _parse(data), hashlib.sha256(data).hexdigest()


def _validate_selection(selection):
    if selection.get("schema") != SELECTION_FORMAT or selection.get("status") != "PROSPECTIVE_NOT_CAPTURED" or selection.get("claims") != CLAIMS or selection.get("destination") != DESTINATION or selection.get("namespace") != NAMESPACE:
        raise BackupError("invalid_selection_configuration")
    paths, ids, directories = set(), set(), {}
    for row in selection["rows"]:
        if row["training_eligible"] is not False or row["file_id"] != file_id(row["path"], row["backup_role"]):
            raise BackupError("selection_identity_mismatch")
        if row["file_id"] in ids or row["path"].casefold() in paths:
            raise BackupError("duplicate_selection_identity")
        ids.add(row["file_id"])
        paths.add(row["path"].casefold())
        parent = row["path"].rpartition("/")[0]
        while parent:
            if directories.get(parent.casefold(), parent) != parent:
                raise BackupError("case_alias_selection_directory")
            directories[parent.casefold()] = parent
            parent = parent.rpartition("/")[0]
        reason = path_gate(row["path"], row["backup_role"], row["bytes"])
        if row.get("path_gate_reason") != reason or row.get("path_gate_status") != ("BLOCKED" if reason else "ALLOWED_PROSPECTIVELY"):
            raise BackupError("prospective_gate_mismatch")
    if paths & set(directories):
        raise BackupError("selection_file_ancestor_collision")


def _reason(exc):
    if isinstance(exc, FileNotFoundError):
        return "missing_source"
    if isinstance(exc, PermissionError):
        return "unreadable_source"
    if isinstance(exc, BackupError):
        return str(exc)
    if isinstance(exc, OSError):
        return "filesystem_error"
    return "unexpected_" + type(exc).__name__


def _now():
    return datetime.now(timezone.utc).isoformat()


def capture(selection, output, *, source_root=None, test_fixture=False, review_receipt=None, selection_file=None):
    """Capture only after external review; no upload code is present.

    Tests must use a disjoint temporary fixture root. Production uses exactly
    SOURCE_ROOT and re-derives the selection from the pinned inventory bytes.
    Caller must use review-gated CLI for real-source execution.
    """
    output = Path(output).absolute()
    root = Path(source_root if source_root is not None else SOURCE_ROOT).absolute()
    production = SOURCE_ROOT.absolute()
    if test_fixture:
        if source_root is None or _within(root, production) or _within(production, root):
            raise BackupError("synthetic_fixture_must_be_disjoint")
    elif source_root is not None and root != production:
        raise BackupError("production_source_override_forbidden")
    if not test_fixture:
        if review_receipt is None or selection_file is None:
            raise BackupError("independent_review_receipt_required")
        _review_gate(Path(review_receipt), Path(selection_file))
        pinned_selection, _ = _local_json(Path(selection_file))
        if pinned_selection != selection:
            raise BackupError("selection_file_argument_mismatch")
    if _within(output, root) or _within(root, output):
        raise BackupError("output_source_overlap")
    _check_ancestors(output)
    output.mkdir(parents=True, exist_ok=False)
    start = {"format": FORMAT, "event": "START", "pid": os.getpid(), "started_utc": _now(), "source_root": str(root), "source_mode": "synthetic_fixture" if test_fixture else "read_only_production", "selection_sha256": hashlib.sha256(json_bytes(selection)).hexdigest(), "inventory_sha256": selection.get("inventory_sha256", "UNKNOWN"), "tool_sha256": _hash_file(Path(__file__).absolute())[0], "config_sha256": hashlib.sha256(json_bytes({"format": FORMAT, "source_root": str(root), "destination": DESTINATION, "namespace": NAMESPACE, "claims": CLAIMS, "selection": selection})).hexdigest(), "destination": DESTINATION, "namespace": NAMESPACE}
    _write_json(output / "START.json", start)
    outcome = {"format": FORMAT, "event": "OUTCOME", "pid": os.getpid(), "status": "FAILED", "source_root": str(root), "selection_sha256": start["selection_sha256"], "tool_sha256": start["tool_sha256"], "config_sha256": start["config_sha256"], "inventory_sha256": start["inventory_sha256"], "counts": {name: "UNKNOWN" for name in ("selected_files", "captured_files", "failed_attempts", "verified_files")}, "failures": [], "claims": dict(CLAIMS), "upload_attempted": False}
    files, current_row, stage = [], None, "configuration"
    partial = output / "archive.partial.tar.gz"
    try:
        with partial.open("xb") as raw_archive, (output / "attempts.jsonl").open("xb") as attempts:
            def attempt(row, attempt_stage, status, reason=None):
                record = {"pid": os.getpid(), "file_id": row["file_id"], "path": row["path"], "backup_role": row["backup_role"], "training_eligible": False, "stage": attempt_stage, "status": status, "reason": reason}
                attempts.write(json_bytes(record) + b"\n")
                attempts.flush()
                if status == "FAILED":
                    outcome["failures"].append({k: record[k] for k in ("file_id", "path", "stage", "reason")})
            _validate_selection(selection)
            if not test_fixture:
                if root != production or selection["inventory_sha256"] != INVENTORY_SHA256:
                    raise BackupError("production_source_or_inventory_pin_mismatch")
                inventory, sha = _local_json(production / INVENTORY_RELATIVE)
                if sha != INVENTORY_SHA256 or prepare_selection(inventory, sha) != selection:
                    raise BackupError("selection_not_exact_pinned_inventory_projection")
                if selection["counts"]["declared_train_files"] != 102 or selection["counts"]["declared_train_recorded_bytes"] != 1311950205:
                    raise BackupError("declared_train_inventory_count_mismatch")
            _check_ancestors(root)
            root = root.resolve(strict=True)
            if not root.is_dir():
                raise BackupError("source_root_not_directory")
            for row in selection["rows"]:
                current_row, stage = row, "first_read"
                try:
                    files.append(scan_file(root, row))
                    attempt(row, stage, "VERIFIED_FIRST_READ")
                except (BackupError, OSError) as exc:
                    attempt(row, stage, "FAILED", _reason(exc))
            outcome["counts"] = {"selected_files": len(selection["rows"]), "captured_files": 0, "failed_attempts": len(outcome["failures"]), "verified_files": 0}
            if outcome["failures"]:
                return outcome
            current_row, stage = None, "archive_creation"
            with gzip.GzipFile(fileobj=raw_archive, mode="wb", filename="", mtime=0, compresslevel=9) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for row, descriptor in zip(selection["rows"], files):
                        current_row, stage = row, "second_read"
                        try:
                            _add_file(archive, root, row, descriptor)
                            outcome["counts"]["captured_files"] += 1
                            attempt(row, stage, "VERIFIED_SECOND_READ")
                        except (BackupError, OSError) as exc:
                            attempt(row, stage, "FAILED", _reason(exc))
                            raise BackupError("second_read_failed") from exc
            raw_archive.flush()
            os.fsync(raw_archive.fileno())
        current_row, stage = None, "archive_readback"
        manifest = {"format": FORMAT, "status": "VERIFIED_BYTE_BACKUP", "inventory_sha256": selection["inventory_sha256"], "source_root": selection["source_root"], "claims": dict(CLAIMS), "selection_sha256": start["selection_sha256"], "files": files}
        manifest["content_sha256"] = hashlib.sha256(json_bytes(_content_descriptor(manifest))).hexdigest()
        verification = verify_archive(partial, manifest)
        manifest.update({"archive_sha256": verification["archive_sha256"], "archive_size": verification["archive_size"], "verified_files": verification["verified_files"], "verified_bytes": verification["verified_bytes"], "destination": DESTINATION, "namespace_path": NAMESPACE + "/" + manifest["content_sha256"], "archive_filename": "backup.tar.gz", "source_inventory_status": "METADATA_ONLY_NOT_PREPARED", "restoration_scope": "Selected declared TRAIN-related files and small metadata only; not a complete evaluation-release restore."})
        # Same-directory rename after full readback; exclusive output directory.
        os.rename(partial, output / "backup.tar.gz")
        _write_json(output / "manifest.json", manifest)
        outcome.update({"status": "SUCCESS", "content_sha256": manifest["content_sha256"], "archive_sha256": manifest["archive_sha256"], "namespace_path": manifest["namespace_path"]})
        outcome["counts"]["verified_files"] = verification["verified_files"]
        return outcome
    except Exception as exc:
        # Never print arbitrary exception values: OS/library exceptions may carry
        # unexpected content. Failure receipts contain only controlled reasons.
        outcome["fatal"] = {"stage": stage, "reason": _reason(exc)}
        if current_row is not None and not any(r["file_id"] == current_row["file_id"] and r["stage"] == stage for r in outcome["failures"]):
            outcome["failures"].append({"file_id": current_row["file_id"], "path": current_row["path"], "stage": stage, "reason": _reason(exc)})
        outcome["counts"] = {name: "UNKNOWN" for name in outcome["counts"]}
        return outcome
    finally:
        outcome["completed_utc"] = _now()
        _write_json(output / "OUTCOME.json", outcome)


def _review_gate(path, selection_path):
    receipt, _ = _local_json(path)
    source_sha = _hash_file(Path(__file__).absolute())[0]
    selection_sha = _hash_file(selection_path)[0]
    tests_sha = _hash_file(Path(__file__).with_name("test_backup.py"))[0]
    if receipt.get("status") != "APPROVED_FOR_CAPTURE" or receipt.get("independent_review") is not True or receipt.get("synthetic_tests_passed") is not True or receipt.get("tool_sha256") != source_sha or receipt.get("selection_file_sha256") != selection_sha or receipt.get("test_source_sha256") != tests_sha or receipt.get("inventory_sha256") != INVENTORY_SHA256 or receipt.get("source_root") != str(SOURCE_ROOT).replace("\\", "/"):
        raise BackupError("independent_review_and_synthetic_test_gate_not_satisfied")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "capture", "verify"))
    parser.add_argument("--selection", type=Path, default=Path(__file__).with_name("prospective_selection.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            if args.archive is None or args.manifest is None:
                raise BackupError("archive_and_manifest_required")
            manifest, _ = _local_json(args.manifest)
            result = verify_archive(args.archive, manifest)
        else:
            selection, _ = _local_json(args.selection)
            _validate_selection(selection)
            if args.command == "describe":
                result = {"status": selection["status"], "counts": selection["counts"], "claims": selection["claims"], "capture_executed": False, "upload_attempted": False}
            else:
                if args.output is None or args.review_receipt is None:
                    raise BackupError("output_and_review_receipt_required")
                _review_gate(args.review_receipt, args.selection)
                base = Path(__file__).parent.resolve()
                if not _within(args.output.absolute(), base):
                    raise BackupError("production_output_must_be_inside_owned_backup_directory")
                result = capture(selection, args.output, review_receipt=args.review_receipt, selection_file=args.selection)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") != "FAILED" else 2
    except (BackupError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAILED_BEFORE_CAPTURE", "reason": _reason(exc), "capture_executed": False, "upload_attempted": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
