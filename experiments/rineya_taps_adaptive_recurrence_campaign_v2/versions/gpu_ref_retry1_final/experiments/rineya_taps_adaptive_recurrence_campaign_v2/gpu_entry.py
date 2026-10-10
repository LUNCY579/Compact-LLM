"""Explicit GPU successor bootstrap; no CPU runner fallback or model import."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--stage", choices=("qualification", "dev"), default="dev")
    args = parser.parse_args()
    started = time.monotonic()
    while not args.release.is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("parent did not establish GPU worker containment")
        time.sleep(.05)
    if json.loads(args.release.read_text(encoding="utf-8"))["contained"] is not True:
        raise RuntimeError("GPU containment not admitted")
    sys.argv = ["experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_native",
                "--stage", args.stage, "--source", str(args.source), "--out", str(args.out),
                "--admission", str(args.admission)]
    runpy.run_module("experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_native", run_name="__main__")


if __name__ == "__main__":
    main()
