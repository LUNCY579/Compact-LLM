#!/usr/bin/env python3
"""Fail-closed Phase-D entrypoint; full SFT is forbidden without a passed gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if (
        gate.get("verdict") != "PASS_ADVANCE_TO_PHASE_D"
        or gate.get("passed") is not True
    ):
        raise RuntimeError(
            "Phase D blocked: the frozen Phase-C advancement gate did not pass"
        )
    raise RuntimeError(
        "Gate passed, but Phase-D implementation must be reviewed and committed as a "
        "separate milestone before any full SFT workload starts"
    )


if __name__ == "__main__":
    raise SystemExit(main())
