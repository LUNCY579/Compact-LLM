from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, canonical_sha256, sha256_file


class EvidenceError(RuntimeError):
    pass


def build_evidence_manifest(
    evidence_root: Path, output_path: Path, *, git_head: str | None = None
) -> dict[str, Any]:
    evidence_root = evidence_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path != evidence_root and evidence_root not in output_path.parents:
        raise EvidenceError("manifest output must be inside evidence root")
    records = {}
    private_present = False
    for directory, names, filenames in os.walk(evidence_root):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(evidence_root)
        if relative_directory.as_posix() == "10_sealed_eval/private":
            private_present = True
            names[:] = []
            continue
        retained_names = []
        for name in names:
            child_relative = (relative_directory / name).as_posix()
            if child_relative == "10_sealed_eval/private":
                private_present = True
                continue
            retained_names.append(name)
        names[:] = retained_names
        for filename in filenames:
            path = directory_path / filename
            if path.resolve() == output_path:
                continue
            if path.is_symlink():
                raise EvidenceError(f"symlink not allowed in evidence: {path}")
            relative = path.relative_to(evidence_root).as_posix()
            records[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    identity = [[name, record["bytes"], record["sha256"]] for name, record in sorted(records.items())]
    manifest = {
        "schema_version": "1.0",
        "git_head": git_head,
        "evidence_root": str(evidence_root),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records.values()),
        "sealed_private_directory_present_but_not_read": private_present,
        "root_digest": canonical_sha256(identity),
        "files": dict(sorted(records.items())),
    }
    atomic_write_json(output_path, manifest)
    return manifest


def verify_evidence_manifest(manifest_path: Path) -> dict[str, Any]:
    import json

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["evidence_root"])
    failures = []
    identity = []
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        identity.append([relative, actual_bytes, actual_hash])
        if actual_bytes != expected["bytes"] or actual_hash != expected["sha256"]:
            failures.append(f"mismatch:{relative}")
    digest = canonical_sha256(sorted(identity))
    if digest != manifest["root_digest"]:
        failures.append("root_digest_mismatch")
    return {"status": "verified" if not failures else "failed", "failures": failures}
