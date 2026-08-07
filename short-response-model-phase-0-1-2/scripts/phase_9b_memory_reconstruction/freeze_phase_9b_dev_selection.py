from __future__ import annotations

"""Freeze one Phase 9B strategy from DEV before any sealed-test evaluation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, sha256, write_json
from scripts.phase_9b_memory_reconstruction.phase9b_common import PIPELINE_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase 9B DEV-selected strategy.")
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assessment = read_json(args.assessment)
    strategy = str(assessment.get("recommended_candidate") or "").strip()
    if not strategy:
        raise SystemExit("NO_VALID_RECONSTRUCTION_FRONTIER: DEV produced no eligible Phase 9B strategy; sealed TEST remains untouched.")
    packet = (assessment.get("strategies") or {}).get(strategy) or {}
    artifact = packet.get("artifact") or {}
    payload = {
        "schema_version": "1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "selection_split": "dev",
        "sealed_test_touched": False,
        "strategy": strategy,
        "artifact": artifact,
        "dev_opportunity_margin": packet.get("opportunity_margin"),
        "dev_functional_success_rate": packet.get("functional_success_rate"),
        "dev_negative_transfer_rate": packet.get("negative_transfer_rate"),
        "assessment": str(args.assessment),
        "assessment_sha256": sha256(args.assessment),
        "summary": str(args.summary),
        "summary_sha256": sha256(args.summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
