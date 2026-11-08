#!/usr/bin/env python3
"""Verify the TGSA programme ledger against the repository.

This is a provenance check, not a scientific check.  It confirms three things:

1.  every imported copy still hashes to the value recorded in the ledger;
2.  every imported copy is byte-identical to its declared source commit and path,
    so `verbatim_copy` is a fact rather than an assertion;
3.  every stage commit recorded in the ledger still resolves in this repository.

A ledger that has drifted from the branches it indexes is worse than no ledger,
so this exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEDGER = HERE / "tgsa_programme_ledger.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_show(repo: Path, commit: str, path: str) -> bytes | None:
    """Return the blob at commit:path, or None if it cannot be read."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def commit_exists(repo: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def verify(ledger_path: Path = LEDGER, repo: Path = REPO) -> dict[str, object]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = 0

    for record in ledger["imported_records"]:
        local = repo / record["path"]
        checks += 1
        if not local.is_file():
            errors.append(f"missing imported copy: {record['path']}")
            continue

        payload = local.read_bytes()
        actual = sha256_bytes(payload)
        checks += 1
        if actual != record["sha256"]:
            errors.append(
                f"hash drift: {record['path']} recorded {record['sha256'][:12]}, "
                f"found {actual[:12]}"
            )

        checks += 1
        source = git_show(repo, record["source_commit"], record["source_path"])
        if source is None:
            errors.append(
                f"unreadable source: {record['source_commit']}:{record['source_path']}"
            )
        elif record.get("verbatim_copy") and source != payload:
            errors.append(
                f"not a verbatim copy: {record['path']} differs from "
                f"{record['source_commit']}:{record['source_path']}"
            )

    for stage in ledger["stages"]:
        checks += 1
        if not commit_exists(repo, stage["commit"]):
            errors.append(
                f"stage {stage['id']} commit {stage['commit']} does not resolve"
            )

    return {"passed": not errors, "checks": checks, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify(args.ledger, args.repo)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"checks: {report['checks']}")
    if report["passed"]:
        print("ledger provenance: PASSED")
        return 0
    print("ledger provenance: FAILED")
    for error in report["errors"]:
        print(f"  {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
