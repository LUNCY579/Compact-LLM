from __future__ import annotations

"""Build preservation-heavy residual reasoning/procedure training views."""
import argparse, json, random
from pathlib import Path
from typing import Any

def read_jsonl(p:Path)->list[dict[str,Any]]:return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write_jsonl(p:Path,rows:list[dict[str,Any]])->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def repeat_sample(rows:list[dict[str,Any]],n:int,rng:random.Random,max_repeat:int)->list[dict[str,Any]]:
    if not rows or n<=0:return []
    if n>len(rows)*max_repeat:raise ValueError(f'requested {n} samples from {len(rows)} rows exceeds max repeat factor {max_repeat}')
    out=[];pool=list(rows)
    while len(out)<n:
        rng.shuffle(pool);out.extend(pool[:min(len(pool),n-len(out))])
    return out

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--reasoning-residual',type=Path,required=True);ap.add_argument('--procedure-residual',type=Path,required=True);ap.add_argument('--preservation',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--reasoning-total',type=int,default=6000);ap.add_argument('--reasoning-residual-fraction',type=float,default=.60);ap.add_argument('--procedure-total',type=int,default=5000);ap.add_argument('--procedure-residual-fraction',type=float,default=.48);ap.add_argument('--minimum-reasoning-residual',type=int,default=250);ap.add_argument('--minimum-procedure-residual',type=int,default=60);ap.add_argument('--maximum-repeat-factor',type=int,default=8);ap.add_argument('--seed',type=int,default=23071);a=ap.parse_args()
    rr=read_jsonl(a.reasoning_residual);pr=read_jsonl(a.procedure_residual);pres=read_jsonl(a.preservation)
    status={'reasoning_ready':len(rr)>=a.minimum_reasoning_residual,'procedure_ready':len(pr)>=a.minimum_procedure_residual}
    rng=random.Random(a.seed);manifest={'schema_version':'2.3','source_counts':{'reasoning_residual':len(rr),'procedure_residual':len(pr),'preservation':len(pres)},'readiness':status,'views':{}}
    if status['reasoning_ready']:
        nres=round(a.reasoning_total*a.reasoning_residual_fraction);npres=a.reasoning_total-nres;mix=repeat_sample(rr,nres,rng,a.maximum_repeat_factor)+repeat_sample(pres,npres,rng,a.maximum_repeat_factor);rng.shuffle(mix);write_jsonl(a.output_root/'reasoning_residual_mixed.reviewed.jsonl',mix);manifest['views']['reasoning']={'rows':len(mix),'residual_rows_sampled':nres,'preservation_rows_sampled':npres}
    if status['procedure_ready']:
        nres=round(a.procedure_total*a.procedure_residual_fraction);npres=a.procedure_total-nres;mix=repeat_sample(pr,nres,rng,a.maximum_repeat_factor)+repeat_sample(pres,npres,rng,a.maximum_repeat_factor);rng.shuffle(mix);write_jsonl(a.output_root/'procedure_residual_mixed.reviewed.jsonl',mix);manifest['views']['procedure']={'rows':len(mix),'residual_rows_sampled':nres,'preservation_rows_sampled':npres}
    a.output_root.mkdir(parents=True,exist_ok=True);(a.output_root/'training_mix_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
