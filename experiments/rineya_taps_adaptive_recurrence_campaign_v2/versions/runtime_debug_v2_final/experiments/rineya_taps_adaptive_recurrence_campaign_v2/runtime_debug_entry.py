"""Explicit runtime-debug GPU bootstrap; imports no model before containment."""
import argparse
import json
import os
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
    args = parser.parse_args()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.monotonic()
    while not args.release.is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("runtime-debug parent did not establish containment")
        time.sleep(.05)
    if json.loads(args.release.read_text(encoding="utf-8"))["contained"] is not True:
        raise RuntimeError("runtime-debug containment not admitted")
    sys.argv = ["experiments.rineya_taps_adaptive_recurrence_campaign_v2.runtime_debug",
                "--source", str(args.source), "--out", str(args.out), "--admission", str(args.admission)]
    runpy.run_module("experiments.rineya_taps_adaptive_recurrence_campaign_v2.runtime_debug", run_name="__main__")


if __name__ == "__main__":
    main()
