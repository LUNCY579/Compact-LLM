#!/usr/bin/env python3
"""Freeze v5 capacity selection or authorize the next MTP-free mechanism stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase1_runner import atomic_write_json, utc_now
from tgsa.phase1_v5 import (
    canonical_sha256,
    freeze_band_after_probe,
    load_v5_plan,
    select_qualified_exposure,
)

VERIFICATION_SCHEMA_VERSION = "tgsa.phase1-v5-verification.v1"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return dict(value)


def _load_verified_report(path: Path) -> dict[str, object]:
    report = _load_object(path)
    verification_path = path.with_name("verification.json")
    verification = _load_object(verification_path)
    if verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise ValueError(f"{verification_path} has an incompatible schema")
    if verification.get("passed") is not True:
        raise ValueError(f"{verification_path} did not pass")
    if verification.get("report_sha256") != canonical_sha256(report):
        raise ValueError(
            f"{verification_path} verifies a different report payload"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capacity = subparsers.add_parser(
        "capacity",
        help="select the first passing rung and emit a frozen apparatus band",
    )
    capacity.add_argument("--plan", type=Path, required=True)
    capacity.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="verified probe report; repeat in registered rung order",
    )
    capacity.add_argument("--output", type=Path, required=True)

    exposure = subparsers.add_parser(
        "exposure",
        help="select from the complete verified qualification grid",
    )
    exposure.add_argument("--plan", type=Path, required=True)
    exposure.add_argument("--frozen-band", type=Path, required=True)
    exposure.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="verified qualification report; repeat for every exposure",
    )
    exposure.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    plan = load_v5_plan(args.plan.resolve())
    reports = [_load_verified_report(path.resolve()) for path in args.report]
    if args.command == "capacity":
        result = freeze_band_after_probe(plan=plan, probe_reports=reports)
    else:
        frozen_band = _load_object(args.frozen_band.resolve())
        result = select_qualified_exposure(
            plan=plan,
            frozen_band=frozen_band,
            qualification_reports=reports,
        )
    result["created_at_utc"] = utc_now()
    result["source_reports"] = [str(path.resolve()) for path in args.report]
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    # NO_GO is a scientific result, not a broken selector.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
