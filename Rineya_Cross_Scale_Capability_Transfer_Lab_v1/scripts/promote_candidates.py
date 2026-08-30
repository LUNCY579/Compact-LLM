from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.promotion import promote  # noqa: E402


def key_value(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected NAME=PATH: {value}")
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply deterministic preservation-first promotion.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    report = promote(
        baseline_path=args.baseline,
        candidate_paths=key_value(args.candidate),
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        top_n=args.top_n,
    )
    print(json.dumps({"eligible_count": report["eligible_count"], "promoted": report["promoted"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
