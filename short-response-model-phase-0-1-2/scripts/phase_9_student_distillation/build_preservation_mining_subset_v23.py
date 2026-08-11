from __future__ import annotations

"""Build a TRAIN-only Phase 9 preservation mining benchmark for v2.3."""
import argparse, json, random
from collections import defaultdict
from pathlib import Path
from typing import Any

def read_jsonl(p:Path)->list[dict[str,Any]]:return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write_jsonl(p:Path,rows:list[dict[str,Any]])->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def klass(r:dict[str,Any])->str:
    d=str(r.get('decision') or '').lower();m=str(r.get('mode') or '').lower();c=str(r.get('category') or '').lower()
    if d=='social' or 'social' in m or c in {'social','conversation','acknowledgment','celebration'}:return 'social'
    if d=='conflict' or 'conflict' in c:return 'conflict'
    if d=='abstain':return 'boundary'
    if d=='answer' and ('procedure' in m or r.get('procedures')):return 'procedure_answer'
    if d=='answer':return 'supported_answer'
    return 'other'
def sample(rows:list[dict[str,Any]],limit:int,seed:int)->list[dict[str,Any]]:
    targets={'supported_answer':0.50,'procedure_answer':0.20,'social':0.10,'boundary':0.10,'conflict':0.10}
    rng=random.Random(seed);b=defaultdict(list)
    for r in rows:b[klass(r)].append(r)
    for xs in b.values():rng.shuffle(xs)
    out=[]
    for k,frac in targets.items():out.extend(b[k][:min(len(b[k]),round(limit*frac))])
    if len(out)<limit:
        used={str(r.get('id')) for r in out};rest=[r for r in rows if str(r.get('id')) not in used];rng.shuffle(rest);out.extend(rest[:limit-len(out)])
    rng.shuffle(out);return out[:limit]
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--train-file',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int,default=2500);ap.add_argument('--seed',type=int,default=23051);a=ap.parse_args();rows=read_jsonl(a.train_file);out=sample(rows,a.limit,a.seed);write_jsonl(a.output,out)
    counts={k:sum(klass(r)==k for r in out) for k in sorted({klass(r) for r in out})};print(json.dumps({'rows':len(out),'class_counts':counts,'train_only_source':str(a.train_file)},indent=2,sort_keys=True))
if __name__=='__main__':main()
