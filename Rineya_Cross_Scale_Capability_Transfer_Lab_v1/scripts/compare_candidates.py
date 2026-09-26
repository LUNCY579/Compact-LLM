from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.comparison import (  # noqa: E402
    compare_prediction_files,
    write_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired scenario-aware model comparison.")
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    report = compare_prediction_files(
        args.a,
        args.b,
        label_a=args.label_a,
        label_b=args.label_b,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    write_comparison(
        report, json_path=args.output_json, markdown_path=args.output_markdown
    )
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
