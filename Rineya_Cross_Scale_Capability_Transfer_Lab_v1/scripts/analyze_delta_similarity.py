from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.delta_analysis import analyze_delta_pair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze projected 4B/8B delta agreement.")
    parser.add_argument("--delta4", type=Path, required=True)
    parser.add_argument("--delta8", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_delta_pair(
        delta4_path=args.delta4,
        delta8_path=args.delta8,
        base_path=args.base,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report["global"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
