from __future__ import annotations
import argparse, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',type=int,required=True);a=p.parse_args();report=ROOT/'reports'/f'PHASE_{a.phase}_STOP_GATE.md'
 if not report.exists(): raise SystemExit(f'Missing {report}')
 text=report.read_text(encoding='utf-8')
 if not re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$",text): raise SystemExit(f'Phase {a.phase} lacks explicit human PASS in {report}')
 print(f'Phase {a.phase} human gate: PASS')
if __name__=='__main__':main()
