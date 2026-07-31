from __future__ import annotations

"""Resolve a human-approved Phase 2C adapter/checkpoint pair for Phase 3."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.phase_2c_targeted_recovery.phase2c_common import parse_approved_profiles

PROFILE_NAMES = ("maximum_quality", "balanced_training", "minimum_training_memory")


def human_pass(gate_text: str) -> bool:
    return bool(re.search(r"(?im)^\s*-\s*Human decision:\s*`?PASS`?\s*$", gate_text))


def resolve_profile(catalog: dict, profile: str, gate_text: str) -> dict:
    profiles = catalog.get("profiles") or {}
    if profile not in profiles:
        raise KeyError(f"Profile {profile!r} is not present. Available: {sorted(profiles)}")
    if not human_pass(gate_text):
        raise RuntimeError("Every Phase 2C profile requires Human decision: PASS.")
    if profile not in parse_approved_profiles(gate_text):
        raise RuntimeError(f"Profile {profile!r} is not listed in Approved profiles.")
    row = dict(profiles[profile])
    if not (row.get("assessment") or {}).get("eligible"):
        raise RuntimeError(f"Profile {profile!r} does not pass the recorded automated Phase 2C gates.")
    if not row.get("seed_stability_pass"):
        raise RuntimeError(f"Profile {profile!r} does not have passing repeated-seed evidence.")
    model_id = row.get("training_model_id")
    adapter = row.get("training_adapter")
    summary = row.get("evaluation_summary")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"Profile {profile!r} has no training_model_id.")
    if not isinstance(adapter, str) or not adapter:
        raise RuntimeError(f"Profile {profile!r} has no training_adapter.")
    if not isinstance(summary, str) or not summary:
        raise RuntimeError(f"Profile {profile!r} has no evaluation_summary.")
    row["profile"] = profile
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve an approved Phase 2C profile.")
    p.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    p.add_argument("--catalog", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery/model_catalog.json")
    p.add_argument("--gate", type=Path, default=ROOT / "reports/PHASE_2C_STOP_GATE.md")
    p.add_argument("--field", choices=("training_model_id", "training_adapter", "inference_model", "evaluation_summary"), default=None)
    args = p.parse_args()
    if not args.catalog.exists():
        raise FileNotFoundError(args.catalog)
    gate_text = args.gate.read_text(encoding="utf-8") if args.gate.exists() else ""
    row = resolve_profile(read_json(args.catalog), args.profile, gate_text)
    if args.field:
        print(row.get(args.field, ""))
    else:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
