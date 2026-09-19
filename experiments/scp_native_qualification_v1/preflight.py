"""Read-only source/asset preparation; no worker, lease or permit is started."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
from .assets import audit_subject
from .contract import (BRANCH, PARENT, PACKAGE, RUNTIME, NATIVE_BLOCK, digest, file_hash,
                       require, safe_path, write_new)
from .evidence import external_directory


def validate_checkout_observation(branch, head, status, changed):
    from .contract import hash_string
    require(branch == BRANCH and hash_string(head, 40) and head != PARENT, "wrong/unpublished successor checkout")
    require(status == "", "dirty checkout refused, including untracked files")
    require(changed, "successor has no changes")
    prefix = "experiments/scp_native_qualification_v1/"
    require(all(kind == "A" and path.startswith(prefix) and ".." not in Path(path).parts
                for kind, path in changed), "historical code/evidence changed or unscoped successor diff")


def checkout_binding(root):
    root = safe_path(root).resolve()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == root, "repository root mismatch")
    require(git("rev-parse", "HEAD:" + PACKAGE.relative_to(root).as_posix()), "package not committed")
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    git("merge-base", "--is-ancestor", PARENT, head)
    lines = git("diff", "--no-ext-diff", "--no-renames", "--name-status", PARENT, head).splitlines()
    changed = [line.split("\t", 1) for line in lines]
    validate_checkout_observation(branch, head, status, changed)
    return {"parent_commit": PARENT, "successor_commit": head, "branch": branch,
            "changed_paths": [p for _, p in changed], "clean_checkout": True,
            "package_git_tree": git("rev-parse", "HEAD:experiments/scp_native_qualification_v1")}


def metadata_runtime():
    libraries = {}
    for name in RUNTIME:
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {"libraries": libraries,
            "pins_match": all((libraries[k] or "").split("+")[0] == v for k, v in RUNTIME.items()),
            "native_imports": False, "host_observer": "NOT_RUN", "gpu_qualification": "NOT_RUN"}


def prepare(root, source, partial, subject, out):
    root = safe_path(root)
    binding = checkout_binding(root)
    assets = audit_subject(source, partial, subject)
    report = {"schema": "SCP_NQ_PREFLIGHT_V1", "classification": "ASSET_PREFLIGHT_ONLY",
              "source_binding": binding, "assets": assets, "runtime_metadata": metadata_runtime(),
              "protocol_sha256": file_hash(PACKAGE / "PROTOCOL.md"),
              "native_execution_permitted": False, "permit_consumed": False, "native_status": NATIVE_BLOCK,
              "remaining": ["exact-host observer and fresh resource readings", "complete and approve one-use request",
                            "review native subject transport and inherited MQ/RRP attachment",
                            "supervised 2048-window parity and zero-update checks", "native receipt verifier qualification"]}
    report["report_content_sha256"] = digest(report)
    out = external_directory(out, root)
    write_new(out / "PREFLIGHT.json", report)
    return report
