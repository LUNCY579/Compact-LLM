# Purpose: Compatibility wrapper that delegates human-gate validation to scripts/common.
from __future__ import annotations
import argparse, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def approved(path: Path) -> bool:
    if not path.exists(): return False
    return bool(re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$", path.read_text(encoding="utf-8")))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--report", type=Path, default=ROOT/"reports"/"PHASE_7_STOP_GATE.md"); a=p.parse_args()
    if not approved(a.report): raise SystemExit(f"{a.report} does not contain an explicit human PASS")
    print(f"Phase 7 human gate: PASS")
if __name__ == "__main__": main()
