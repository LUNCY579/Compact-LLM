from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.evaluation.summary import write_summary_files
from consensus_lab.io import read_jsonl
from consensus_lab.paths import LabPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild summary JSON/CSV from authoritative raw JSONL rows.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--output-stem",
        help="Override the summary filename stem; valid only when rebuilding one input.",
    )
    args = parser.parse_args()
    if args.output_stem and len(args.inputs) != 1:
        parser.error("--output-stem requires exactly one input")
    paths = LabPaths.discover(LAB_ROOT)
    packets = []
    for value in args.inputs:
        source = paths.assert_output_path(value)
        rows = read_jsonl(source)
        output_stem = args.output_stem or source.stem
        json_path = paths.assert_output_path(LAB_ROOT / f"results/summaries/{output_stem}.json")
        csv_path = paths.assert_output_path(LAB_ROOT / f"results/summaries/{output_stem}.csv")
        summaries = write_summary_files(rows, json_path, csv_path)
        packets.append({"input": str(source), "rows": len(rows), "summaries": len(summaries), "json": str(json_path)})
    print(json.dumps(packets, indent=2))


if __name__ == "__main__":
    main()
