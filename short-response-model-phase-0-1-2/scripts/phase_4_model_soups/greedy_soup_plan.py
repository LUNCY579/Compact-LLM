from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json, write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('summaries',nargs='+',type=Path); p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_4_soup/greedy_plan.json'); a=p.parse_args()
 rows=[]
 for s in a.summaries:
  x=read_json(s); rows.append({'summary':str(s),'score':x['mean_heuristic_score'],'critical':x.get('critical_failure_count',0)})
 rows.sort(key=lambda r:(r['critical']>0,-r['score']))
 write_json(a.out,{'ranked_ingredients':rows,'procedure':'Start with best eligible ingredient. Add one ingredient, merge uniformly in FP32 delta space, evaluate, retain only if composite improves and no hard regression.'})
 print(a.out)
if __name__=='__main__': main()
