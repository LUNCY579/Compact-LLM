"""Create both independent frozen banks before any model evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.supa_adaptive_compute_v2.shared.datasets import build_banks
from experiments.supa_adaptive_compute_v2.shared.hashing import write_json
from experiments.supa_adaptive_compute_v2.shared.statistics import power_planning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "experiments/supa_adaptive_compute_v2/data/generated/v2")
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    manifests = build_banks(args.output, args.seed)
    write_json(args.output / "power_planning.json", power_planning())
    for track,path in manifests.items():
        print(f"{track}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
