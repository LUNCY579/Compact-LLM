#!/usr/bin/env python3
"""Select v5.1 capacity/exposure only from verified, prospectively locked reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase1_runner import utc_now
from tgsa.phase1_v5 import (
    canonical_sha256,
    freeze_band_after_probe,
    load_v5_plan,
    select_qualified_exposure,
)
from tgsa.v5_1_programme import atomic_write_json, verify_repository_contract

V5_VERIFICATION_SCHEMA = "tgsa.phase1-v5-verification.v1"
V51_RUN_REGISTRATION_SCHEMA = "tgsa.v5-1-run-registration.v1"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return dict(value)


def _load_registered_report(
    path: Path,
    *,
    programme_sha256: str,
    require_attestation: bool,
) -> dict[str, object]:
    report = _load_object(path)
    verification_path = path.with_name("verification.json")
    verification = _load_object(verification_path)
    if verification.get("schema_version") != V5_VERIFICATION_SCHEMA:
        raise ValueError(f"{verification_path} has an incompatible schema")
    if verification.get("passed") is not True:
        raise ValueError(f"{verification_path} did not pass")
    if verification.get("report_sha256") != canonical_sha256(report):
        raise ValueError(f"{verification_path} verifies a different report payload")

    registration_path = path.with_name("v5_1_registration_verification.json")
    registration = _load_object(registration_path)
    if registration.get("schema_version") != V51_RUN_REGISTRATION_SCHEMA:
        raise ValueError(f"{registration_path} has an incompatible schema")
    if registration.get("programme_sha256") != programme_sha256:
        raise ValueError(f"{registration_path} belongs to a different v5.1 programme")
    if registration.get("source_report_sha256") != canonical_sha256(report):
        raise ValueError(f"{registration_path} is not bound to {path}")
    attestation = registration.get("operator_attestation")
    if require_attestation and (
        not isinstance(attestation, Mapping) or attestation.get("accepted") is not True
    ):
        raise ValueError(
            f"{registration_path} lacks the required prospective operator attestation"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--plan", type=Path, required=True)
    capacity.add_argument("--report", type=Path, action="append", required=True)
    capacity.add_argument("--output", type=Path, required=True)

    exposure = subparsers.add_parser("exposure")
    exposure.add_argument("--plan", type=Path, required=True)
    exposure.add_argument("--frozen-band", type=Path, required=True)
    exposure.add_argument("--report", type=Path, action="append", required=True)
    exposure.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    programme = verify_repository_contract(
        repo_root=args.repo_root.resolve(),
        contract_path=args.contract.resolve(),
    )
    if programme["passed"] is not True:
        raise SystemExit(
            "v5.1 programme verification failed before selection output:\n- "
            + "\n- ".join(str(item) for item in programme["errors"])
        )

    plan = load_v5_plan(args.plan.resolve())
    reports = [
        _load_registered_report(
            path.resolve(),
            programme_sha256=str(programme["programme_sha256"]),
            require_attestation=not bool(plan["development_only"]),
        )
        for path in args.report
    ]
    if args.command == "capacity":
        result = freeze_band_after_probe(plan=plan, probe_reports=reports)
    else:
        result = select_qualified_exposure(
            plan=plan,
            frozen_band=_load_object(args.frozen_band.resolve()),
            qualification_reports=reports,
        )
    result["v5_1_programme_sha256"] = programme["programme_sha256"]
    result["v5_1_repo_head"] = programme["repo_head"]
    result["created_at_utc"] = utc_now()
    result["source_reports"] = [str(path.resolve()) for path in args.report]
    result["claim_scope"] = (
        str(result.get("claim_scope", "")).strip()
        + " Selection remains apparatus-only and cannot authorize mechanism, Rineya, "
        "broad-quality, serial-round, or wall-clock claims."
    )
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
