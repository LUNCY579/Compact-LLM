from __future__ import annotations

"""Static/runtime prerequisite validator for v2.3.1."""

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--v23-root", type=Path, required=True)
    ap.add_argument("--v22-root", type=Path, required=True)
    ap.add_argument("--direct-root", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cfg = read_json(args.config)
    checks = {
        "v23_reasoning_residual": args.v23_root / "residual_mining/mined/reasoning_residual.reviewed.jsonl",
        "v23_preservation": args.v23_root / "residual_mining/mined/preservation_replay.reviewed.jsonl",
        "v23_reasoning_mix": args.v23_root / "residual_mining/training_views/reasoning_residual_mixed.reviewed.jsonl",
        "v23_mining_audit": args.v23_root / "residual_mining/mined/RESIDUAL_MINING_AUDIT.json",
        "v22_greedy_anchor": args.v22_root / "track_d_greedy_soup/final_model",
        "direct_dev": args.direct_root / "splits/dev.reviewed.jsonl",
        "direct_ood": args.direct_root / "splits/ood_dev.reviewed.jsonl",
    }
    missing = [name for name, path in checks.items() if not path.exists()]
    sealed_references = [str(path) for path in checks.values() if "sealed" in str(path).casefold()]
    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if sealed_references:
        problems.append("sealed path unexpectedly referenced")
    if bool(cfg.get("controls", {}).get("sealed_test_access", True)):
        problems.append("config controls.sealed_test_access must be false")
    payload = {
        "schema_version": "2.3.1",
        "status": "PASS" if not problems else "FAIL",
        "checks": {name: str(path) for name, path in checks.items()},
        "missing": missing,
        "sealed_references": sealed_references,
        "problems": problems,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
