"""Freeze the v2.3 GPU config from one committed calibration artifact.

This creates a new config exclusively. It neither calibrates alpha nor runs a
model, and never modifies the original v2.1/v2.2 protocol or registration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROGRAM))


def freeze_calibrated_control(artifact_path: Path) -> Path:
    from native_armr.src.control_calibration import verify_calibration_artifact
    from native_armr.src.protocol import (CALIBRATED_GPU_CONFIG, calibrated_gpu_config,
                                         verify_control_calibration)
    from shared.guards import require_within
    from shared.hashing import sha256_file
    from shared.legacy import REPO

    artifact_path = require_within(artifact_path, PROGRAM / "native_armr/evidence/00_control_calibration")
    digest = sha256_file(artifact_path)
    report = verify_calibration_artifact(artifact_path, digest, require_committed=True)
    binding = {"artifact_path": artifact_path.relative_to(REPO).as_posix(),
               "artifact_sha256": digest}
    config = calibrated_gpu_config(binding, report["alpha"])
    verify_control_calibration(config)
    # Arm order is part of the protocol. The evidence writer sorts dictionary
    # keys, so config serialization deliberately retains registration order.
    payload = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with CALIBRATED_GPU_CONFIG.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return CALIBRATED_GPU_CONFIG


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-artifact", required=True, type=Path)
    args = parser.parse_args(argv)
    path = freeze_calibrated_control(args.calibration_artifact)
    print(f"CONFIG={path}")
    print("Next: freeze and commit the new Track A registration, then pass its new five-arm CPU/GPU qualification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
