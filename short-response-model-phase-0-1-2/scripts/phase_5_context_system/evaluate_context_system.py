from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from context_system import JsonlMemoryStore,retrieve,assemble,Budget,tokens
from scripts.common.experiment_utils import read_jsonl,write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--cases',type=Path,default=ROOT/'data/context/context_eval.jsonl'); p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_5_context/evaluation.json'); a=p.parse_args()
 rows=read_jsonl(a.cases); hits=0; selected=[]; t=time.perf_counter()
 for r in rows:
  got=retrieve(r['query'],r['memories'],r.get('top_k',8)); ids={x.get('id') for x in got}; required=set(r.get('required_memory_ids',[])); hits+=int(required<=ids); selected.append(sum(len(tokens(str(x.get('text','')))) for x in got))
 write_json(a.out,{'cases':len(rows),'relevant_memory_recall':hits/max(1,len(rows)),'avg_selected_memory_tokens':sum(selected)/max(1,len(selected)),'runtime_seconds':time.perf_counter()-t})
 print(a.out)
if __name__=='__main__':main()
