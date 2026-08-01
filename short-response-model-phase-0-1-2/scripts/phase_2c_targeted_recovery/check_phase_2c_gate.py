from __future__ import annotations

"""Validate Phase 2C catalog against relative-opportunity evidence."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 2C proportional opportunity evidence.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/model_catalog.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/candidate_registry.json")
    args = parser.parse_args()
    catalog = read_json(args.catalog)
    registry = read_json(args.registry)
    rows = registry.get("base_seed_candidates") or []
    if not rows:
        raise SystemExit("Phase 2C registry contains no base-seed candidates.")
    if not any(row.get("name") == "all_remaining" for row in rows):
        raise SystemExit("Phase 2C is missing the all_remaining quality control.")
    by_name = {row.get("name"): row for row in rows}
    expected_repeat_seeds = set(int(value) for value in (catalog.get("repeat_seeds") or []))
    for profile_name, profile in (catalog.get("profiles") or {}).items():
        candidate_name = profile.get("candidate")
        if candidate_name not in by_name:
            raise SystemExit(f"Catalog profile {profile_name!r} points to missing candidate {candidate_name!r}.")
        source = by_name[candidate_name]
        assessment = source.get("assessment") or {}
        if candidate_name != "all_remaining" and not assessment.get("eligible"):
            raise SystemExit(
                f"Catalog profile {profile_name!r} points to candidate {candidate_name!r} whose "
                "proportional resource gain does not exceed quality opportunity loss."
            )
        if not assessment.get("technical_valid", True):
            raise SystemExit(f"Catalog profile {profile_name!r} has technically invalid evidence.")
        observed = {int(row.get("seed")) for row in (source.get("repeat_results") or [])}
        if expected_repeat_seeds and observed != expected_repeat_seeds:
            raise SystemExit(
                f"Catalog profile {profile_name!r} repeat evidence covers {sorted(observed)}, "
                f"expected configured seeds {sorted(expected_repeat_seeds)}."
            )
        if candidate_name != "all_remaining" and source.get("repeat_results") and not source.get("seed_stability_pass"):
            raise SystemExit(
                f"Catalog profile {profile_name!r} has a non-positive mean opportunity margin across recorded seeds."
            )
    print("Phase 2C automated evidence: PASS_PENDING_HUMAN_REVIEW (relative opportunity policy)")


if __name__ == "__main__":
    main()
