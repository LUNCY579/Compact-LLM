from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.v2 import verify_cached_artifacts, write_verification_report  # noqa: E402


def main() -> int:
    report = verify_cached_artifacts()
    destination = (
        LAB_ROOT
        / "cross_scale_evidence"
        / "12_v2_followup"
        / "protocol"
        / "artifact_verification.json"
    )
    write_verification_report(report, destination)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
