# Purpose: Legacy planning helper retained for dry-run inspection; the production runner uses build_greedy_soup.py.
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json, write_json
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score
from scripts.common.quality_gate import quality_index
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('summaries',nargs='+',type=Path); p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_4_soup/greedy_plan.json'); a=p.parse_args()
 rows=[]
 for s in a.summaries:
  x=read_json(s); require_quality_v2(x, str(s)); rows.append({'summary':str(s),'mean_quality':summary_quality_score(x),'quality_index':quality_index(x),'worst_decile':x.get('worst_decile_mean_quality_score',0.0),'quality_pass_rate':x.get('quality_pass_rate',0.0),'intelligibility_pass_rate':x.get('intelligibility_pass_rate',0.0),'hard_cases':x.get('hard_failure_case_count',0),'critical':x.get('critical_failure_count',0),'major_cases':x.get('major_case_count',0)})
 rows.sort(key=lambda r:(r['hard_cases']>0,r['critical']>0,r['major_cases']>0,-r['quality_index'],-r['worst_decile'],-r['mean_quality']))
 write_json(a.out,{'ranked_ingredients':rows,'procedure':'Start with best eligible ingredient. Add one ingredient, merge uniformly in FP32 delta space, evaluate, retain only if composite improves and no hard regression.'})
 print(a.out)
if __name__=='__main__': main()
