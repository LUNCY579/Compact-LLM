#!/usr/bin/env python3
"""Post-run verifier for the complete TGSA Phase-1 evidence chain.

Fails closed. Every link from the registered study configuration through to the
prediction files must hash-match the artifacts on disk, and every registered
arm/seed cell must be present, complete, and internally consistent.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_verify import verify_evidence_chain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--expect-confirmatory",
        choices=("true", "false"),
        help="fail unless the study configuration's confirmatory flag matches",
    )
    args = parser.parse_args()
    report = verify_evidence_chain(
        study_config_path=args.config.resolve(),
        output_root=args.output_dir.resolve(),
        dataset_dir=None if args.dataset_dir is None else args.dataset_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        require_confirmatory=None
        if args.expect_confirmatory is None
        else args.expect_confirmatory == "true",
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    if not report["passed"]:
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
