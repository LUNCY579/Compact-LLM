from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.evidence import (  # noqa: E402
    build_evidence_manifest,
    verify_evidence_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify the cross-scale evidence manifest.")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=LAB_ROOT / "cross_scale_evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "cross_scale_evidence" / "evidence_manifest.json",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_evidence_manifest(args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "verified" else 2
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=LAB_ROOT.parent, text=True
    ).strip()
    manifest = build_evidence_manifest(
        args.evidence_root, args.output, git_head=git_head
    )
    print(json.dumps({"file_count": manifest["file_count"], "total_bytes": manifest["total_bytes"], "root_digest": manifest["root_digest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
