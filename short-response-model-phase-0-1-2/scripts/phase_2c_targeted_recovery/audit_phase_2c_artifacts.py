from __future__ import annotations

"""Read-only structural audit for a Phase 2C artifact tree."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Phase 2C artifacts without modifying them.")
    p.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery")
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        root / "run_config.json",
        root / "importance/combined_layer_ranking.json",
        root / "reference/reduced_frozen/generation/summary.json",
        root / "reference/reduced_frozen/teacher_forcing/summary.json",
        root / "candidate_registry.json",
        root / "model_catalog.json",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required file: {path}")
    candidate_count = 0
    seed_count = 0
    if not errors:
        registry = read_json(root / "candidate_registry.json")
        rows = [*(registry.get("base_seed_candidates") or []), *(registry.get("repeat_candidates") or [])]
        candidate_count = len(registry.get("base_seed_candidates") or [])
        seed_count = len(rows)
        for row in rows:
            candidate_root = Path(str(row.get("artifact_root", "")))
            if not candidate_root.is_absolute():
                candidate_root = ROOT / candidate_root
            for relative in (
                "training/adapter/adapter_config.json",
                "training/training_manifest.json",
                "evaluation/generation/results.jsonl",
                "evaluation/generation/summary.json",
                "evaluation/teacher_forcing/summary.json",
                "evaluation/output_deviation.json",
                "candidate_result.json",
            ):
                if not (candidate_root / relative).exists():
                    errors.append(f"{row.get('name')} seed {row.get('seed')}: missing {relative}")
        catalog = read_json(root / "model_catalog.json")
        profiles = catalog.get("profiles") or {}
        expected = {"maximum_quality", "balanced_training", "minimum_training_memory"}
        if set(profiles) != expected:
            errors.append(f"profile names are {sorted(profiles)}, expected {sorted(expected)}")
        expected_repeat_seeds = set(int(value) for value in (catalog.get("repeat_seeds") or []))
        base_by_name = {row.get("name"): row for row in (registry.get("base_seed_candidates") or [])}
        for profile_name, profile in profiles.items():
            candidate = profile.get("candidate")
            source = base_by_name.get(candidate)
            if source is None:
                errors.append(f"profile {profile_name}: candidate {candidate!r} is missing from base registry")
                continue
            if not (source.get("assessment") or {}).get("eligible"):
                errors.append(f"profile {profile_name}: candidate {candidate!r} is ineligible")
            if not source.get("seed_stability_pass"):
                errors.append(f"profile {profile_name}: candidate {candidate!r} failed or lacks seed stability")
            observed = {int(row.get("seed")) for row in (source.get("repeat_results") or [])}
            if expected_repeat_seeds and observed != expected_repeat_seeds:
                errors.append(
                    f"profile {profile_name}: repeat seeds {sorted(observed)} do not match "
                    f"{sorted(expected_repeat_seeds)}"
                )
            for key in ("training_adapter", "evaluation_summary"):
                value = profile.get(key)
                if not isinstance(value, str) or not value:
                    errors.append(f"profile {profile_name}: missing {key}")
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = ROOT / path
                if key == "training_adapter":
                    path = path / "adapter_config.json"
                if not path.exists():
                    errors.append(f"profile {profile_name}: missing {key} target {path}")
        ranking = read_json(root / "importance/combined_layer_ranking.json")
        ranked = ranking.get("ranking") or []
        if len(ranked) != int(ranking.get("layer_count", -1)):
            errors.append("importance ranking count does not match layer_count")
        if sorted(int(row.get("importance_rank", 0)) for row in ranked) != list(range(1, len(ranked) + 1)):
            errors.append("importance ranks are not contiguous")

    status = "CLEAN_COMPLETE" if not errors else "NEEDS_ATTENTION"
    payload: dict[str, Any] = {
        "status": status,
        "artifact_root": str(root),
        "base_candidate_count": candidate_count,
        "candidate_seed_count": seed_count,
        "errors": errors,
        "warnings": warnings,
    }
    print("PHASE 2C ARTIFACT AUDIT")
    print("=" * 72)
    print(f"Artifact root: {root}")
    print(f"Candidates:    {candidate_count} base configurations / {seed_count} total seeds")
    print(f"Errors:        {len(errors)}")
    for error in errors:
        print(f"[ERROR] {error}")
    print(f"\nFINAL STATUS: {status}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
