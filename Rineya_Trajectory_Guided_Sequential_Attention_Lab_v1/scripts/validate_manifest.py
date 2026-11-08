#!/usr/bin/env python3
"""Fail closed on incomplete or failed Phase-0 run manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "created_at_utc",
    "status",
    "branch",
    "config_sha256",
    "seed",
    "environment",
    "model",
    "input",
    "checks",
    "metrics",
    "claim_scope",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_TOP_LEVEL - set(manifest))
    errors: list[str] = []
    if missing:
        errors.append(f"missing fields: {missing}")
    if manifest.get("schema_version") != "tgsa.run-manifest.v1":
        errors.append("unexpected schema_version")
    if manifest.get("status") != "pass":
        errors.append("run status is not pass")
    checks = manifest.get("checks", {})
    failed_checks = sorted(name for name, value in checks.items() if value is not True)
    if failed_checks:
        errors.append(f"failed checks: {failed_checks}")
    digest = manifest.get("config_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("config_sha256 is not a 64-character digest")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"PASS: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
