from __future__ import annotations

"""Measure pairwise LoRA-delta interference for v2.3 diagnostics."""
import argparse, json, math, re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

LAYER_RE=re.compile(r'(?:^|\.)layers\.(\d+)\.')

def load_adapter(path:Path)->tuple[dict[str,torch.Tensor],float]:
    cfg=json.loads((path/'adapter_config.json').read_text(encoding='utf-8'))
    r=float(cfg.get('r') or 1);alpha=float(cfg.get('lora_alpha') or r);scale=alpha/r
    st=path/'adapter_model.safetensors'
    if not st.exists():raise FileNotFoundError(st)
    return load_file(str(st),device='cpu'),scale

def module_pairs(tensors:dict[str,torch.Tensor])->dict[str,tuple[torch.Tensor,torch.Tensor]]:
    out={}
    for k,a in tensors.items():
        if '.lora_A.' not in k:continue
        bkey=k.replace('.lora_A.','.lora_B.')
        if bkey in tensors:out[k.split('.lora_A.')[0]]=(a.float(),tensors[bkey].float())
    return out

def delta(pair:tuple[torch.Tensor,torch.Tensor],scale:float)->torch.Tensor:
    a,b=pair;return (b@a)*scale

def pair_metrics(pa:Path,pb:Path,top_fraction:float)->dict[str,Any]:
    ta,sa=load_adapter(pa);tb,sb=load_adapter(pb);ma=module_pairs(ta);mb=module_pairs(tb);shared=sorted(set(ma)&set(mb));dot=na=nb=0.0;sign_same=sign_total=0;top_inter=top_union=0;per_layer={}
    for key in shared:
        da=delta(ma[key],sa);db=delta(mb[key],sb)
        fa=da.flatten();fb=db.flatten();dot+=float(torch.dot(fa,fb));na+=float(torch.dot(fa,fa));nb+=float(torch.dot(fb,fb))
        nz=(fa.ne(0)|fb.ne(0));sign_same+=int((torch.sign(fa[nz])==torch.sign(fb[nz])).sum()) if bool(nz.any()) else 0;sign_total+=int(nz.sum())
        k=max(1,int(math.ceil(fa.numel()*top_fraction)));ia=torch.topk(fa.abs(),k,largest=True).indices;ib=torch.topk(fb.abs(),k,largest=True).indices;mask_a=torch.zeros(fa.numel(),dtype=torch.bool);mask_b=torch.zeros_like(mask_a);mask_a[ia]=True;mask_b[ib]=True;top_inter+=int((mask_a&mask_b).sum());top_union+=int((mask_a|mask_b).sum())
        m=LAYER_RE.search(key);layer=m.group(1) if m else 'other';p=per_layer.setdefault(layer,{'dot':0.0,'na':0.0,'nb':0.0,'modules':0});p['dot']+=float(torch.dot(fa,fb));p['na']+=float(torch.dot(fa,fa));p['nb']+=float(torch.dot(fb,fb));p['modules']+=1
    cos=dot/(math.sqrt(na)*math.sqrt(nb)) if na>0 and nb>0 else None
    layers={k:{'cosine_similarity':v['dot']/(math.sqrt(v['na'])*math.sqrt(v['nb'])) if v['na']>0 and v['nb']>0 else None,'modules':v['modules']} for k,v in per_layer.items()}
    return {'shared_modules':len(shared),'frobenius_norm_a':math.sqrt(na),'frobenius_norm_b':math.sqrt(nb),'cosine_similarity':cos,'sign_agreement':sign_same/sign_total if sign_total else None,'high_magnitude_overlap_jaccard':top_inter/top_union if top_union else None,'top_fraction':top_fraction,'per_layer':layers}

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',action='append',required=True,help='name=path');ap.add_argument('--output-json',type=Path,required=True);ap.add_argument('--output-md',type=Path,required=True);ap.add_argument('--top-fraction',type=float,default=.10);a=ap.parse_args();items=[]
    for spec in a.adapter:
        name,path=spec.split('=',1);items.append((name,Path(path)))
    pairs={}
    for i,(na,pa) in enumerate(items):
        for nb,pb in items[i+1:]:pairs[f'{na}__{nb}']=pair_metrics(pa,pb,a.top_fraction)
    out={'schema_version':'2.3','adapters':{n:str(p) for n,p in items},'pairs':pairs};a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# v2.3 Adapter Interference Map','', '| Pair | Cosine | Sign agreement | Top-magnitude overlap | Shared modules |','|---|---:|---:|---:|---:|']
    for k,v in pairs.items():lines.append(f"| {k} | {v['cosine_similarity']} | {v['sign_agreement']} | {v['high_magnitude_overlap_jaccard']} | {v['shared_modules']} |")
    a.output_md.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
