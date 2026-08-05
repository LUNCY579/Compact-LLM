from __future__ import annotations

"""Validate Phase 2C catalog against complete relative-opportunity evidence."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json

EXPECTED_PROFILES = {"maximum_quality", "balanced_training", "minimum_training_memory"}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def adapter_complete(path: Path) -> bool:
    if not (path / "adapter_config.json").is_file():
        return False
    return any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in (path / "adapter_model.safetensors", path / "adapter_model.bin")
    )


def require_object(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} is not a JSON object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 2C proportional opportunity evidence.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/model_catalog.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/candidate_registry.json")
    args = parser.parse_args()

    catalog = require_object(read_json(args.catalog), "Phase 2C catalog")
    registry = require_object(read_json(args.registry), "Phase 2C registry")
    rows = registry.get("base_seed_candidates") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Phase 2C registry contains no base-seed candidates.")
    identities = [(str(row.get("name")), row.get("seed")) for row in rows if isinstance(row, dict)]
    if len(identities) != len(rows) or len(identities) != len(set(identities)):
        raise SystemExit("Phase 2C base registry contains malformed or duplicate candidate rows.")
    if not any(name == "all_remaining" for name, _ in identities):
        raise SystemExit("Phase 2C is missing the all_remaining quality control.")

    profiles = catalog.get("profiles") or {}
    if not isinstance(profiles, dict) or set(profiles) != EXPECTED_PROFILES:
        raise SystemExit(
            f"Phase 2C catalog profiles are {sorted(profiles) if isinstance(profiles, dict) else profiles!r}; "
            f"expected {sorted(EXPECTED_PROFILES)}."
        )
    by_name = {str(row.get("name")): row for row in rows}
    expected_repeat_seeds = {int(value) for value in (catalog.get("repeat_seeds") or [])}
    if not expected_repeat_seeds:
        raise SystemExit("Phase 2C catalog records no repeat seeds.")

    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise SystemExit(f"Catalog profile {profile_name!r} is not an object.")
        candidate_name = str(profile.get("candidate"))
        if candidate_name not in by_name:
            raise SystemExit(f"Catalog profile {profile_name!r} points to missing candidate {candidate_name!r}.")
        source = by_name[candidate_name]
        assessment = source.get("assessment") or {}
        if not isinstance(assessment, dict) or not assessment.get("technical_valid", True):
            raise SystemExit(f"Catalog profile {profile_name!r} has technically invalid evidence.")
        if not assessment.get("eligible"):
            raise SystemExit(
                f"Catalog profile {profile_name!r} points to ineligible candidate {candidate_name!r}."
            )
        margin = assessment.get("opportunity_margin_across_seeds", assessment.get("opportunity_margin"))
        try:
            if margin is not None and not math.isfinite(float(margin)):
                raise ValueError
        except (TypeError, ValueError):
            raise SystemExit(f"Catalog profile {profile_name!r} has invalid opportunity margin {margin!r}.")

        repeat_rows = source.get("repeat_results") or []
        observed = {int(row.get("seed")) for row in repeat_rows}
        if observed != expected_repeat_seeds:
            raise SystemExit(
                f"Catalog profile {profile_name!r} repeat evidence covers {sorted(observed)}, "
                f"expected configured seeds {sorted(expected_repeat_seeds)}."
            )
        if not source.get("seed_stability_pass"):
            raise SystemExit(
                f"Catalog profile {profile_name!r} failed or lacks recorded seed-stability approval."
            )

        adapter_value = profile.get("training_adapter")
        if not isinstance(adapter_value, str) or not adapter_complete(repo_path(adapter_value)):
            raise SystemExit(
                f"Catalog profile {profile_name!r} points to an incomplete adapter: {adapter_value!r}."
            )
        summary_value = profile.get("evaluation_summary")
        if not isinstance(summary_value, str) or not repo_path(summary_value).is_file():
            raise SystemExit(
                f"Catalog profile {profile_name!r} is missing its profile-validation summary: {summary_value!r}."
            )
        summary = require_object(read_json(repo_path(summary_value)), f"{profile_name} evaluation summary")
        if int(summary.get("case_count", 0)) <= 0 or "mean_quality_score" not in summary:
            raise SystemExit(f"Catalog profile {profile_name!r} has an incomplete evaluation summary.")

    print("Phase 2C automated evidence: PASS_PENDING_HUMAN_REVIEW (relative opportunity policy)")


if __name__ == "__main__":
    main()
