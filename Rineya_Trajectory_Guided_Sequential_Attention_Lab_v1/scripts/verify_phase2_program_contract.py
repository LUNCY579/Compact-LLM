#!/usr/bin/env python3
"""Verify the canonical TGSA × MTP branch/stage contract.

This is an evidence-identity check, not a model-quality check.  It prevents the
sequential trajectory-conditioned Phase 2A intervention from being confused
with the independent-offset Phase 2B intervention and prevents the rejected v4
regime from being revived as confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

ALLOWED = {
    "mechanism_bridge": "sequential_teacher_forced_recursive_v1",
    "system_factorial": "parallel_independent_offsets_v1",
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return value


def verify(contract_path: Path, config_path: Path) -> dict[str, object]:
    contract = load_object(contract_path)
    config = load_object(config_path)
    errors: list[str] = []

    study_kind = str(config.get("study_kind", ""))
    semantics = str(config.get("mtp_semantics", ""))
    if study_kind not in ALLOWED:
        errors.append(f"unsupported or missing study_kind: {study_kind!r}")
    elif semantics != ALLOWED[study_kind]:
        errors.append(
            f"mtp_semantics {semantics!r} does not match study_kind {study_kind!r}; "
            f"expected {ALLOWED[study_kind]!r}"
        )

    expected_stage = "phase2a" if study_kind == "mechanism_bridge" else "phase2b"
    stages = contract.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(expected_stage), Mapping):
        errors.append(f"program contract has no stage definition for {expected_stage}")
    else:
        stage = stages[expected_stage]
        if stage.get("study_kind") != study_kind:
            errors.append("config study_kind disagrees with program contract")
        if stage.get("mtp_semantics") != semantics:
            errors.append("config mtp_semantics disagrees with program contract")

    registration = config.get("registration")
    if not isinstance(registration, Mapping):
        errors.append("config.registration must be an object")
        registration = {}

    phase1_status = str(
        config.get("phase1_regime_status", registration.get("phase1_regime_status", ""))
    )
    obsolete = bool(config.get("obsolete_regime", registration.get("obsolete_regime", False)))
    confirmatory = bool(config.get("confirmatory", False))
    development_only = bool(config.get("development_only", not confirmatory))

    if phase1_status != "rejected":
        errors.append("supplied v5 config must record phase1_regime_status='rejected'")
    if not obsolete:
        errors.append("supplied old-band v5 config must be marked obsolete_regime=true")
    if confirmatory:
        errors.append("a config tied to the rejected v4 regime may not be confirmatory")
    if not development_only:
        errors.append("a config tied to the rejected v4 regime must be development_only")

    order = None
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping):
        order = dataset.get("split_generation_order")
    if order not in (["test", "dev", "train"], ["holdout", "train"]):
        errors.append(
            "evaluation-first split_generation_order is required; expected "
            "['test','dev','train'] or ['holdout','train']"
        )

    checks = {
        "semantic_identity": study_kind in ALLOWED and semantics == ALLOWED.get(study_kind),
        "rejected_regime_disclosed": phase1_status == "rejected" and obsolete,
        "development_only": development_only and not confirmatory,
        "evaluation_first_splits": not any("split_generation_order" in item for item in errors),
        "cross_stage_pooling_forbidden": contract.get("evidence_may_be_pooled_across_stages") is False,
    }
    return {
        "schema_version": "tgsa.mtp-program-contract-verification.v1",
        "passed": not errors and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "contract_sha256": canonical_sha256(contract),
        "config_sha256": canonical_sha256(config),
        "study_kind": study_kind,
        "mtp_semantics": semantics,
        "claim_scope": "scientific/evidence identity only; not model-quality evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.contract, args.config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
