#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from rineya_fwa.data import build_lab_mix  # noqa: E402
from rineya_fwa.paths import DATA  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build unsealed TRAIN/DEV (CC0 subset or synthetic).")
    p.add_argument("--train-rows", type=int, default=512)
    p.add_argument("--dev-rows", type=int, default=64)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--strata", default="2k,4k")
    args = p.parse_args()
    strata = [s.strip() for s in args.strata.split(",") if s.strip()]
    manifest = build_lab_mix(
        train_rows=args.train_rows,
        dev_rows=args.dev_rows,
        strata=strata,
        seed=args.seed,
        dest=DATA / "unsealed",
    )
    write_json(DATA / "unsealed" / "build_report.json", manifest, config=manifest)
    print(manifest)


if __name__ == "__main__":
    main()
