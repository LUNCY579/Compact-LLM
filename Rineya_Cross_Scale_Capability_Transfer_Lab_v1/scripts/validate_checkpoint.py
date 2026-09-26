from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.validation import validate_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run structural and functional checkpoint GO tests.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    report = validate_checkpoint(
        checkpoint_path=args.checkpoint,
        template_path=args.template,
        output_path=args.output,
        threads=args.threads,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
