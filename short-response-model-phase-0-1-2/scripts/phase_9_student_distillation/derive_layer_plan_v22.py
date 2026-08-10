from __future__ import annotations

"""Derive a pre-registered non-overlapping central-layer plan for v2.2.

This is an interference experiment, not a semantic-localization claim.  The
plan uses the middle-layer prior from the project's cited layer-wise work, then
splits that central band among capabilities so their LoRA deltas cannot touch
the same transformer blocks.
"""

import argparse, json, math
from pathlib import Path

DEFAULT_ORDER=["grounding","procedure","general","abstention","social"]


def layer_count(model:Path)->int:
    cfg=json.loads((model/'config.json').read_text(encoding='utf-8'))
    for key in ('num_hidden_layers','n_layer','num_layers','n_layers'):
        if cfg.get(key) is not None: return int(cfg[key])
    text=json.dumps(cfg)
    raise KeyError(f'Cannot find transformer layer count in {model / "config.json"}: {text[:300]}')


def derive(n:int,central_fraction:float,order:list[str])->dict[str,list[int]]:
    if n < len(order): raise ValueError('Model has fewer layers than capability groups')
    if not 0 < central_fraction <= 1: raise ValueError('central_fraction must be in (0,1]')
    width=max(len(order), int(round(n*central_fraction)))
    width=min(n,width)
    start=max(0,(n-width)//2); indices=list(range(start,start+width))
    groups={name:[] for name in order}
    # Nearly equal contiguous chunks preserve the non-overlap invariant.
    for pos,layer in enumerate(indices):
        bucket=min(len(order)-1, (pos*len(order))//len(indices))
        groups[order[bucket]].append(layer)
    # Ensure every group owns at least one layer.
    if any(not v for v in groups.values()): raise RuntimeError(f'Failed to assign every capability: {groups}')
    return groups


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--central-fraction',type=float,default=.60); p.add_argument('--capability-order',default=','.join(DEFAULT_ORDER)); a=p.parse_args()
    n=layer_count(a.model); order=[x.strip() for x in a.capability_order.split(',') if x.strip()]; plan=derive(n,a.central_fraction,order)
    packet={'schema_version':'2.2','strategy':'central_disjoint','model':str(a.model),'num_hidden_layers':n,'central_fraction':a.central_fraction,'capability_order':order,'layers':plan,'interpretation':'Pre-registered non-overlap test using a middle-layer prior; not evidence that these exact layers uniquely encode the named capabilities.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(packet,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(packet,indent=2,sort_keys=True))
if __name__=='__main__': main()
