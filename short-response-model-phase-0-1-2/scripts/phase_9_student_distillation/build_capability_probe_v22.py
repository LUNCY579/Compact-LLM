from __future__ import annotations

"""Build deterministic stratified DEV probes for v2.2 greedy composition search."""

import argparse, json, random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read(path:Path)->list[dict[str,Any]]:
    with path.open('r',encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def write(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def sample(rows:list[dict[str,Any]],count:int,keys:tuple[str,...],seed:int)->list[dict[str,Any]]:
    rng=random.Random(seed); buckets=defaultdict(list)
    for r in rows:buckets[tuple(str(r.get(k) or '') for k in keys)].append(r)
    for v in buckets.values():rng.shuffle(v)
    names=sorted(buckets); out=[]; i=0
    while len(out)<min(count,len(rows)) and any(buckets.values()):
        k=names[i%len(names)]; i+=1
        if buckets[k]:out.append(buckets[k].pop())
    return out

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--original',type=Path,required=True); p.add_argument('--procedural',type=Path,required=True); p.add_argument('--multidomain',type=Path,required=True); p.add_argument('--v212',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--original-count',type=int,default=180); p.add_argument('--procedural-count',type=int,default=180); p.add_argument('--multidomain-count',type=int,default=540); p.add_argument('--v212-count',type=int,default=180); p.add_argument('--seed',type=int,default=22042); a=p.parse_args()
    outputs={
      'original':sample(read(a.original),a.original_count,('decision','mode','difficulty','category'),a.seed+1),
      'procedural':sample(read(a.procedural),a.procedural_count,('decision','mode','difficulty','procedure_family'),a.seed+2),
      'multidomain':sample(read(a.multidomain),a.multidomain_count,('domain','reasoning_family','decision'),a.seed+3),
      'v212':sample(read(a.v212),a.v212_count,('reasoning_family','decision','difficulty'),a.seed+4),
    }
    for name,rows in outputs.items():write(a.output_root/f'{name}.jsonl',rows)
    manifest={'schema_version':'2.2','counts':{k:len(v) for k,v in outputs.items()},'seed':a.seed}
    (a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
