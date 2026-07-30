from __future__ import annotations

"""Resolve an approved Phase 2B model profile for Phase 3.

``full_quality`` is always the original accepted all-layer SFT branch. Reduced
profiles require an explicit human PASS and must be named in the gate report's
``Approved profiles`` field.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def approved_profiles(gate_text: str) -> set[str]:
    match = re.search(r"(?im)^\s*-\s*Approved profiles:\s*`?([^`\n]+)`?\s*$", gate_text)
    if not match:
        return set()
    return {part.strip() for part in match.group(1).split(",") if part.strip()}


def human_pass(gate_text: str) -> bool:
    return bool(re.search(r"(?im)^\s*-\s*Human decision:\s*`?PASS`?\s*$", gate_text))


def resolve_profile(catalog: dict, profile: str, gate_text: str) -> dict:
    profiles = catalog.get("profiles") or {}
    if profile not in profiles:
        raise KeyError(f"Profile {profile!r} is not present. Available: {sorted(profiles)}")
    if profile != "full_quality":
        if not human_pass(gate_text):
            raise RuntimeError("Reduced Phase 2B profiles require Human decision: PASS.")
        if profile not in approved_profiles(gate_text):
            raise RuntimeError(
                f"Profile {profile!r} is not listed in the gate report's Approved profiles field."
            )
    row = dict(profiles[profile])
    row["profile"] = profile
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve a Phase 2B profile.")
    parser.add_argument("--profile", choices=("full_quality", "balanced", "min_memory"), required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "artifacts/phase_2_progressive_reduction/model_catalog.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports/PHASE_2B_STOP_GATE.md")
    parser.add_argument("--field", choices=("training_model_id", "training_adapter", "inference_model", "evaluation_summary"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.catalog.exists():
        raise FileNotFoundError(args.catalog)
    gate_text = args.gate.read_text(encoding="utf-8") if args.gate.exists() else ""
    row = resolve_profile(read_json(args.catalog), args.profile, gate_text)
    if args.field:
        value = row.get(args.field)
        print("" if value is None else value)
    else:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
