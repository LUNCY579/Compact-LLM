"""Freeze, preflight or execute the prospective CPU label-free AM calibration."""
from pathlib import Path
import argparse
import json
import sys

PROGRAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROGRAM))


def main(argv=None):
    from native_armr.src.control_calibration import (DEFAULT_CONFIG, DEFAULT_MANIFEST,
        freeze_calibration, verify_calibration_registration, run_calibration)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze:
        print(f"REGISTRATION={freeze_calibration(args.config, args.manifest)}")
    elif args.preflight:
        registration = verify_calibration_registration(args.config, args.manifest)
        print(json.dumps({"status": "PREFLIGHT_PASSED", "registration_sha256": registration["registration_sha256"],
                          "fit_scenarios": 96, "holdout_scenarios": 32, "model_executed": False}, indent=2))
    else:
        print(f"RUN_DIR={run_calibration(args.config, args.manifest, args.model_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
