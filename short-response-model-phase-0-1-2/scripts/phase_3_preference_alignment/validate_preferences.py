from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_jsonl, write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=ROOT/'data/preferences/train.jsonl'); p.add_argument('--report',type=Path,default=ROOT/'reports/PHASE_3_PREFERENCE_DATA_VALIDATION.json'); a=p.parse_args()
 rows=read_jsonl(a.input); errors=[]; seen=set()
 for i,r in enumerate(rows,1):
  rid=str(r.get('id',f'line_{i}'))
  if rid in seen: errors.append(f'{rid}: duplicate id')
  seen.add(rid)
  for k in ('prompt','chosen','rejected'):
   if not isinstance(r.get(k),str) or not r[k].strip(): errors.append(f'{rid}: missing {k}')
  if r.get('chosen','').strip()==r.get('rejected','').strip(): errors.append(f'{rid}: chosen equals rejected')
 write_json(a.report,{'records':len(rows),'errors':errors,'status':'PASS' if not errors else 'FAIL'})
 if errors: raise SystemExit('Preference validation failed; see '+str(a.report))
 print(f'Validated {len(rows)} preference pairs')
if __name__=='__main__': main()
