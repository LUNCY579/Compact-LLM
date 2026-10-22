#!/usr/bin/env python3
"""Fail-closed Phase-D entrypoint; full SFT is forbidden without a passed gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from rineya_pcsm.gate import GateInputs, authenticate_phase_d_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--gradient-summary", type=Path, required=True)
    parser.add_argument("--gradient-cache", type=Path, required=True)
    args = parser.parse_args()
    authenticate_phase_d_gate(
        args.gate,
        GateInputs(
            run=args.run,
            plans=args.plans,
            preregistration=args.preregistration,
            budget=args.budget,
            sample=args.sample_manifest,
            split=args.split_manifest,
            gradient_summary=args.gradient_summary,
            gradient_cache=args.gradient_cache,
        ),
    )
    raise RuntimeError(
        "Gate passed, but Phase-D implementation must be reviewed and committed as a "
        "separate milestone before any full SFT workload starts"
    )


if __name__ == "__main__":
    raise SystemExit(main())
