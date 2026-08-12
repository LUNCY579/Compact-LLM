from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.discovery.inventory import build_inventory
from consensus_lab.io import write_json_atomic
from consensus_lab.paths import LabPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit official dense models and PEFT adapters without mutation.")
    parser.add_argument("--output", type=Path, default=LAB_ROOT / "reports/candidate_inventory.json")
    parser.add_argument("--paths", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = LabPaths.from_config(args.paths) if args.paths else LabPaths.discover(LAB_ROOT)
    output = paths.assert_output_path(args.output)
    packet = build_inventory(paths)
    write_json_atomic(output, packet)
    print(json.dumps({"output": str(output), "models": packet["dense_model_count"], "adapters": packet["adapter_count"]}, indent=2))


if __name__ == "__main__":
    main()

