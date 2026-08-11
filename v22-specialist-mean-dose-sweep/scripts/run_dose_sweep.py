from __future__ import annotations

"""One-command CPU builder + smoke eval for specialist-mean dose sweep."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    py = sys.executable
    if not args.skip_build:
        cmd = [py, str(ROOT / "scripts" / "build_phase4_dose_models.py")]
        if args.force:
            cmd.append("--force")
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    if not args.skip_eval:
        subprocess.run(
            [py, str(ROOT / "scripts" / "smoke_eval_cpu.py"), "--limit", str(args.limit)],
            check=True,
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
