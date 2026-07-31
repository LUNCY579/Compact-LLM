from __future__ import annotations

"""Recompute the automated Phase 2C gate summary from the candidate registry."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    p = argparse.ArgumentParser(description="Check Phase 2C automated eligibility evidence.")
    p.add_argument("--catalog", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/model_catalog.json")
    p.add_argument("--registry", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/candidate_registry.json")
    args = p.parse_args()
    catalog = read_json(args.catalog)
    registry = read_json(args.registry)
    rows = registry.get("base_seed_candidates") or []
    if not rows:
        raise SystemExit("Phase 2C registry contains no base-seed candidates.")
    if not any(row.get("name") == "all_remaining" for row in rows):
        raise SystemExit("Phase 2C is missing the all_remaining quality control.")
    eligible = [row for row in rows if (row.get("assessment") or {}).get("eligible")]
    if not eligible:
        raise SystemExit("No Phase 2C candidate passed the automated gates.")
    eligible_by_name = {row.get("name"): row for row in eligible}
    expected_repeat_seeds = set(int(value) for value in (catalog.get("repeat_seeds") or []))
    for name, profile in (catalog.get("profiles") or {}).items():
        candidate = profile.get("candidate")
        if candidate not in eligible_by_name:
            raise SystemExit(f"Catalog profile {name!r} points to ineligible candidate {candidate!r}.")
        source = eligible_by_name[candidate]
        if not source.get("seed_stability_pass"):
            raise SystemExit(f"Catalog profile {name!r} lacks a passing repeated-seed stability check.")
        observed = {int(row.get("seed")) for row in (source.get("repeat_results") or [])}
        if expected_repeat_seeds and observed != expected_repeat_seeds:
            raise SystemExit(
                f"Catalog profile {name!r} repeat seeds are {sorted(observed)}, "
                f"expected {sorted(expected_repeat_seeds)}."
            )
    print("Phase 2C automated evidence: PASS_PENDING_HUMAN_REVIEW")


if __name__ == "__main__":
    main()
