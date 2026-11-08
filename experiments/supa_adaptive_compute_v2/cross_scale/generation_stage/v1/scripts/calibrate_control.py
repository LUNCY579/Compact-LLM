"""Exclusive Track B XM registration/calibration/artifact verification CLI."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from cross_scale.generation_stage.v1.src.control_calibration import (
    DEFAULT_CONFIG, DEFAULT_MANIFEST, freeze_calibration_registration,
    run_calibration, verify_calibration_artifact, verify_calibration_registration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "preflight", "run", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "verify":
        if args.artifact is None:
            parser.error("verify requires --artifact")
        report = verify_calibration_artifact(args.artifact, args.sha256)
        print(json.dumps({"status": "VERIFIED", "alpha": report["alpha"],
                          "control_state_sha256": report["control_state_sha256"]}, sort_keys=True))
        return
    if args.export_manifest is None:
        parser.error("freeze/preflight/run require --export-manifest")
    if args.action == "freeze":
        result = freeze_calibration_registration(args.config, args.dataset_manifest, export_manifest_path=args.export_manifest)
    elif args.action == "preflight":
        registration = verify_calibration_registration(args.config, args.dataset_manifest, export_manifest_path=args.export_manifest)
        result = {"status": "VERIFIED", "registration_path": registration["registration_path"],
                  "registration_sha256": registration["registration_sha256"]}
    else:
        result = run_calibration(args.config, args.dataset_manifest, export_manifest_path=args.export_manifest)
    print(json.dumps(result, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
