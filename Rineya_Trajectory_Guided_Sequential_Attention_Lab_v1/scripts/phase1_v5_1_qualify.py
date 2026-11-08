#!/usr/bin/env python3
"""Run one v5 apparatus cell behind the prospective v5.1 programme lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase1_v5 import load_v5_plan
from tgsa.phase1_v5_runner import run_v5_stage
from tgsa.v5_1_programme import (
    SOURCE_V5_PLAN_BLOB,
    SOURCE_V5_PLAN_PATH,
    V51ContractError,
    atomic_write_json,
    canonical_sha256,
    git_blob_sha1,
    verify_repository_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one v5 prerequisite-probe rung or qualification exposure only "
            "after the v5.1 prospective registration and lineage lock passes."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("probe", "qualification"), required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--exposure")
    parser.add_argument("--frozen-band", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--attest-no-prior-outcome-inspection",
        action="store_true",
        help=(
            "required for the nondevelopment plan; records the operator's "
            "attestation that no nondevelopment outcome was inspected before "
            "the prospective v5.1 lock"
        ),
    )
    args = parser.parse_args()
    if args.stage == "probe" and args.exposure is not None:
        parser.error("--exposure is forbidden for the fixed prerequisite probe")
    if args.stage == "qualification" and args.exposure is None:
        parser.error("--exposure is required for qualification")

    repo_root = args.repo_root.resolve()
    contract_path = args.contract.resolve()
    plan_path = args.plan.resolve()

    programme = verify_repository_contract(
        repo_root=repo_root,
        contract_path=contract_path,
    )
    if programme["passed"] is not True:
        raise SystemExit(
            "v5.1 programme verification failed before output creation:\n- "
            + "\n- ".join(str(item) for item in programme["errors"])
        )

    plan = load_v5_plan(plan_path)
    development_only = bool(plan["development_only"])
    if not development_only:
        expected_plan = (repo_root / SOURCE_V5_PLAN_PATH).resolve()
        if plan_path != expected_plan:
            raise V51ContractError(
                "nondevelopment execution must use the exact v5 plan path locked by v5.1"
            )
        plan_blob = git_blob_sha1(plan_path.read_bytes())
        if plan_blob != SOURCE_V5_PLAN_BLOB:
            raise V51ContractError(
                "nondevelopment execution plan bytes do not match the v5.1 lock"
            )
        if not args.attest_no_prior_outcome_inspection:
            raise V51ContractError(
                "nondevelopment execution requires --attest-no-prior-outcome-inspection; "
                "do not use the flag if any nondevelopment outcome was inspected before "
                "the v5.1 lock"
            )

    report = run_v5_stage(
        plan_path=plan_path,
        stage=args.stage,
        rung_id=args.rung,
        exposure_id=args.exposure,
        frozen_band_path=(
            None if args.frozen_band is None else args.frozen_band.resolve()
        ),
        output_dir=args.output_dir.resolve(),
        repo_root=repo_root,
        resume=not args.no_resume,
    )
    report_name = "probe_report.json" if args.stage == "probe" else "qualification_report.json"
    report_path = args.output_dir.resolve() / report_name
    companion = {
        "schema_version": "tgsa.v5-1-run-registration.v1",
        "programme_verification_schema": programme["schema_version"],
        "programme_sha256": programme["programme_sha256"],
        "programme_contract": str(contract_path),
        "repo_head": programme["repo_head"],
        "source_plan_blob_sha1": programme["source_plan_committed_blob_sha1"],
        "source_report": report_name,
        "source_report_sha256": canonical_sha256(report),
        "stage": args.stage,
        "rung_id": args.rung,
        "exposure_id": args.exposure,
        "development_only": development_only,
        "operator_attestation": {
            "required": not development_only,
            "accepted": bool(args.attest_no_prior_outcome_inspection),
            "statement": (
                "No nondevelopment outcome produced under the locked v5 plan was "
                "inspected before the prospective v5.1 lock."
            ),
        },
        "claim_scope": (
            "registration and provenance attachment only; the scientific verdict "
            "remains in the verified source report and cannot by itself authorize "
            "a mechanism, Rineya, broad-quality, serial-round, or wall-clock claim"
        ),
    }
    atomic_write_json(
        args.output_dir.resolve() / "v5_1_registration_verification.json",
        companion,
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "stage": report.get("stage_execution"),
                "rung_id": report.get("rung_id"),
                "exposure_id": report.get("exposure_id"),
                "verdict": report.get("verdict"),
                "decision": report.get("decision"),
                "programme_sha256": programme["programme_sha256"],
                "report": str(report_path),
                "note": "exit code reports software completion, not scientific GO/NO_GO",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
