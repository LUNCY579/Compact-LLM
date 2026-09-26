from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.evaluation.banks import build_banks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build disjoint cross-scale evaluation banks.")
    parser.add_argument(
        "--output-directory", type=Path, default=LAB_ROOT / "data" / "evaluation"
    )
    parser.add_argument(
        "--sealed-path",
        type=Path,
        default=LAB_ROOT
        / "cross_scale_evidence"
        / "10_sealed_eval"
        / "private"
        / "sealed.jsonl",
    )
    args = parser.parse_args()
    manifest = build_banks(
        args.output_directory,
        args.sealed_path,
        specifications={
            "calibration": (271828, 4),
            "mergedev": (314159, 4),
            "dev_full": (161803, 12),
            "sealed": (141421, 16),
        },
    )
    public = {
        "manifest": str((args.output_directory / "bank_manifest.json").resolve()),
        "splits": {
            name: {
                "sha256": details["sha256"],
                "items": details["items"],
                "scenarios": details["scenarios"],
            }
            for name, details in manifest["splits"].items()
        },
        "prompt_hash_overlap": manifest["prompt_hash_overlap"],
    }
    print(json.dumps(public, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
