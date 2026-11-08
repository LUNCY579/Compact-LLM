#!/usr/bin/env python3
"""Verify a closed TGSA Phase 1 v5 qualification evidence chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, cast

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase1_data import sha256_file, validate_dataset_manifest
from tgsa.phase1_runner import atomic_write_json, utc_now
from tgsa.phase1_v5 import (
    V5_PROBE_REPORT_SCHEMA_VERSION,
    V5_QUALIFICATION_REPORT_SCHEMA_VERSION,
    canonical_sha256,
    load_v5_plan,
    plan_sha256,
)
from tgsa.phase1_v5_runner import V5_CELL_SCHEMA_VERSION


VERIFICATION_SCHEMA_VERSION = "tgsa.phase1-v5-verification.v1"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return dict(value)


def verify(
    *,
    plan_path: Path,
    report_path: Path,
    require_go: bool = False,
) -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    plan = load_v5_plan(plan_path)
    report = _load_object(report_path)
    root = report_path.parent

    stage = str(report.get("stage_execution", ""))
    expected_schema = {
        "probe": V5_PROBE_REPORT_SCHEMA_VERSION,
        "qualification": V5_QUALIFICATION_REPORT_SCHEMA_VERSION,
    }.get(stage)
    checks["known_stage"] = expected_schema is not None
    checks["report_schema"] = report.get("schema_version") == expected_schema
    checks["software_complete"] = (
        report.get("status") == "complete"
        and report.get("software_success") is True
    )
    checks["plan_identity"] = report.get("plan_sha256") == plan_sha256(plan)

    manifest_relative = str(report.get("dataset_manifest", ""))
    manifest_path = root / manifest_relative
    try:
        validate_dataset_manifest(manifest_path)
        checks["dataset_manifest_valid"] = True
        checks["dataset_manifest_hash"] = (
            sha256_file(manifest_path)
            == report.get("dataset_manifest_sha256")
        )
    except Exception as error:  # noqa: BLE001 - collect every verification defect
        checks["dataset_manifest_valid"] = False
        checks["dataset_manifest_hash"] = False
        errors.append(f"dataset manifest: {type(error).__name__}: {error}")

    expected_arms = (
        ("cascade",)
        if stage == "probe"
        else tuple(str(value) for value in plan["registered_arms"])
    )
    expected_seeds = tuple(
        int(value)
        for value in (
            plan["probe"]["seeds"]
            if stage == "probe"
            else plan["qualification"]["seeds"]
        )
    )
    expected_cells = {
        (arm, seed) for arm in expected_arms for seed in expected_seeds
    }
    actual_cells: set[tuple[str, int]] = set()
    cells_ok = True
    artifact_hashes_ok = True
    identity_ok = True

    for entry_object in cast(list[object], report.get("cells", [])):
        if not isinstance(entry_object, Mapping):
            cells_ok = False
            continue
        entry = cast(Mapping[str, object], entry_object)
        arm = str(entry.get("arm", ""))
        seed = int(entry.get("seed", -1))
        key = (arm, seed)
        if key in actual_cells:
            cells_ok = False
        actual_cells.add(key)
        manifest_file = root / str(entry.get("path", ""))
        try:
            if sha256_file(manifest_file) != entry.get("sha256"):
                artifact_hashes_ok = False
            cell = _load_object(manifest_file)
            if (
                cell.get("schema_version") != V5_CELL_SCHEMA_VERSION
                or cell.get("status") != "complete"
                or cell.get("arm") != arm
                or int(cell.get("seed", -1)) != seed
                or cell.get("execution_fingerprint")
                != entry.get("execution_fingerprint")
            ):
                cells_ok = False
            identity = cast(Mapping[str, object], cell.get("identity", {}))
            if (
                identity.get("plan_sha256") != report.get("plan_sha256")
                or identity.get("stage") != stage
                or cast(Mapping[str, object], identity.get("rung", {})).get("id")
                != report.get("rung_id")
                or cast(Mapping[str, object], identity.get("exposure", {})).get("id")
                != report.get("exposure_id")
                or identity.get("dataset_manifest_sha256")
                != report.get("dataset_manifest_sha256")
            ):
                identity_ok = False
            files = cast(Mapping[str, object], cell.get("files", {}))
            for name in ("checkpoint", "predictions_dev", "predictions_test"):
                file_entry = cast(Mapping[str, object], files.get(name, {}))
                artifact_path = manifest_file.parent / str(
                    file_entry.get("path", "")
                )
                if (
                    not artifact_path.is_file()
                    or sha256_file(artifact_path) != file_entry.get("sha256")
                ):
                    artifact_hashes_ok = False
            training = cast(Mapping[str, object], cell.get("training", {}))
            if (
                int(training.get("optimizer_steps", -1))
                * int(training.get("batch_size", -1))
                != int(training.get("sample_presentations", -2))
            ):
                identity_ok = False
            if training.get("fixed_final_checkpoint") is not True:
                identity_ok = False
            if training.get("early_stopping") is not False:
                identity_ok = False
        except Exception as error:  # noqa: BLE001
            cells_ok = False
            artifact_hashes_ok = False
            errors.append(
                f"cell {arm}/{seed}: {type(error).__name__}: {error}"
            )

    checks["complete_registered_cell_matrix"] = (
        cells_ok and actual_cells == expected_cells
    )
    checks["cell_execution_identity"] = identity_ok
    checks["cell_artifact_hashes"] = artifact_hashes_ok

    authorization = cast(Mapping[str, object], report.get("authorization", {}))
    checks["single_cell_never_authorizes_mechanism"] = (
        authorization.get("mechanism_test") is False
        and authorization.get("rineya_retrofit") is False
    )
    checks["mtp_absent_from_v5_plan"] = (
        cast(Mapping[str, object], plan["mtp"]).get("enabled") is False
    )
    checks["apparatus_is_non_confirmatory"] = (
        plan["confirmatory"] is False
        and plan["mechanism_test_authorized"] is False
    )
    checks["verdict_is_explicit"] = report.get("verdict") in {"GO", "NO_GO"}
    if require_go:
        checks["required_scientific_go"] = report.get("verdict") == "GO"

    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    passed = all(checks.values())
    result = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "passed": passed,
        "verified_at_utc": utc_now(),
        "plan_sha256": plan_sha256(plan),
        "report_sha256": canonical_sha256(report),
        "report_file_sha256": sha256_file(report_path),
        "stage": stage,
        "scientific_verdict": report.get("verdict"),
        "require_go": require_go,
        "checks": dict(sorted(checks.items())),
        "errors": errors,
        "claim_scope": (
            "artifact and contract verification only; reproduces no model outcome "
            "and adds no scientific evidence"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-go",
        action="store_true",
        help="also fail when the scientifically valid verdict is NO_GO",
    )
    args = parser.parse_args()
    result = verify(
        plan_path=args.plan.resolve(),
        report_path=args.report.resolve(),
        require_go=args.require_go,
    )
    output = args.output or args.report.with_name("verification.json")
    atomic_write_json(output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
