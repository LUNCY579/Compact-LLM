from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import sha256,write_json
ROOT=Path(__file__).resolve().parents[2]
def gate(path):
 if not path.exists():return 'MISSING'
 t=path.read_text(encoding='utf-8');return 'PASS' if re.search(r'(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$',t) else 'NOT_APPROVED'
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'reports/RELEASE_CANDIDATE_AUDIT.md');a=p.parse_args(); gates={str(i):gate(ROOT/'reports'/f'PHASE_{i}_STOP_GATE.md') for i in range(1,8)}; files=[]
 for x in a.model_dir.rglob('*'):
  if x.is_file(): files.append({'path':str(x.relative_to(a.model_dir)),'sha256':sha256(x),'bytes':x.stat().st_size})
 required=['A base','B all-layer SFT','C best single','D best 2-4','E random control','F selective+DPO','G selective+DPO+soup','H all-layer+DPO+sparse','I full combined']
 status='PASS_PENDING_HUMAN_RELEASE_DECISION' if all(v=='PASS' for v in gates.values()) else 'BLOCKED'
 lines=['# RELEASE CANDIDATE AUDIT','',f'## Status\n\n`{status}`','','## Gate chain','']+[f'- Phase {k}: `{v}`' for k,v in gates.items()]+['','## Required ablation matrix','']+[f'- [ ] {x}' for x in required]+['','## Artifact hashes','']+[f"- `{x['path']}` — `{x['sha256']}` ({x['bytes']} bytes)" for x in files]+['','## Human release decision','','- Decision: `PENDING`','- Reviewer:','- Date:','- Notes:']
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(lines),encoding='utf-8');write_json(a.out.with_suffix('.json'),{'status':status,'gates':gates,'files':files});print(a.out)
if __name__=='__main__':main()
