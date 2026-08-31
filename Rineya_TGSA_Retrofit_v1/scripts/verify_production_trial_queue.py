#!/usr/bin/env python3
"""Validate historical trial records and their immutable git provenance."""

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

from rineya_tgsa.trial_queue import validate_trial_queue


def git_object_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", type=Path, default=LAB / "evidence" / "production_trial_queue.json"
    )
    parser.add_argument(
        "--schema", type=Path, default=LAB / "schemas" / "production_trial_queue.schema.json"
    )
    parser.add_argument("--no-git", action="store_true")
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    from jsonschema import Draft202012Validator, FormatChecker

    errors = [
        f"schema: {error.json_path}: {error.message}"
        for error in Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(queue)
    ]
    errors.extend(validate_trial_queue(queue))
    if not args.no_git:
        commits = {queue.get("retrofit_commit")}
        commits.update(item.get("source_commit") for item in queue.get("trials", []))
        for commit in sorted(value for value in commits if isinstance(value, str)):
            if not git_object_exists(commit):
                errors.append(f"git provenance object is unavailable: {commit}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    statuses: dict[str, int] = {}
    for trial in queue["trials"]:
        statuses[trial["trial_status"]] = statuses.get(trial["trial_status"], 0) + 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "trials": len(queue["trials"]),
                "selected_trial_id": queue["selected_trial_id"],
                "trial_statuses": statuses,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
