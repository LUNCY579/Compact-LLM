#!/usr/bin/env python3
"""Run the hardened Phase 2A trajectory-mechanism development workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase2_hardened import run_phase2_study

EXPECTED_STUDY_KIND = "mechanism_bridge"
EXPECTED_MTP_SEMANTICS = "sequential_teacher_forced_recursive_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--program-contract",
        type=Path,
        default=LAB_ROOT / "configs" / "phase2_program_contract.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.force_data:
        parser.error("--overwrite and --force-data cannot be combined")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("study_kind") != EXPECTED_STUDY_KIND:
        parser.error(
            f"Phase 2A runner requires study_kind={EXPECTED_STUDY_KIND!r}; "
            f"received {config.get('study_kind')!r}"
        )
    if config.get("mtp_semantics") != EXPECTED_MTP_SEMANTICS:
        parser.error(
            f"Phase 2A runner requires mtp_semantics={EXPECTED_MTP_SEMANTICS!r}; "
            f"received {config.get('mtp_semantics')!r}"
        )

    # Fail before dataset or evidence-root creation.  This is intentionally a
    # separate programme-identity gate from the model/competence runner.
    subprocess.run(
        [
            sys.executable,
            str(LAB_ROOT / "scripts" / "verify_phase2_program_contract.py"),
            "--contract",
            str(args.program_contract.resolve()),
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=LAB_ROOT,
    )

    report = run_phase2_study(
        config_path=config_path,
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        overwrite=args.overwrite,
        force_data=args.force_data,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
