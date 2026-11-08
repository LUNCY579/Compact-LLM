#!/usr/bin/env python3
"""Validate the candidate ledger, its strict gates, and git provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from rineya_tgsa.promotion import MANDATORY_PROMOTION_GATES, validate_ledger


def git_object_exists(repo: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LAB / "evidence" / "candidate_ledger.json",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=LAB / "schemas" / "candidate_ledger.schema.json",
    )
    parser.add_argument("--no-git", action="store_true", help="skip local git-object checks")
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise SystemExit("jsonschema is required; install the test extra") from exc

    errors = [
        f"schema: {error.json_path}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(ledger),
            key=lambda value: list(value.absolute_path),
        )
    ]
    errors.extend(validate_ledger(ledger))
    if tuple(ledger.get("promotion_predicate", ())) != MANDATORY_PROMOTION_GATES:
        errors.append("promotion_predicate must contain the canonical gates in canonical order")

    if not args.no_git:
        commits = {ledger.get("retrofit_commit")}
        commits.update(candidate.get("source_commit") for candidate in ledger.get("candidates", []))
        for commit in sorted(value for value in commits if isinstance(value, str)):
            if not git_object_exists(REPO, commit):
                errors.append(f"git provenance object is unavailable: {commit}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    decisions: dict[str, int] = {}
    for candidate in ledger["candidates"]:
        decisions[candidate["decision"]] = decisions.get(candidate["decision"], 0) + 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "candidates": len(ledger["candidates"]),
                "decisions": decisions,
                "promotable": [
                    value["id"] for value in ledger["candidates"] if value["decision"] == "PROMOTE"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
