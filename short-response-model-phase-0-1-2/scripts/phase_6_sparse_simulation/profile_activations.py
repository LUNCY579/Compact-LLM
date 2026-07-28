from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.common.experiment_utils import read_jsonl,write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--calibration',type=Path,default=ROOT/'data/calibration/calibration.jsonl');p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_6_sparse/traces.pt');p.add_argument('--block-size',type=int,default=128);a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model); model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype='auto',device_map='auto'); traces=[]; hooks=[]
 for name,m in model.named_modules():
  if name.endswith('mlp.down_proj'):
   def hook(mod,inp,out,n=name):
    z=inp[0].detach().float().cpu(); w=mod.weight.detach().float().cpu(); score=z.abs()*w.norm(dim=0); pad=(-score.shape[-1])%a.block_size
    if pad: score=torch.nn.functional.pad(score,(0,pad))
    traces.append({'layer':n,'block_scores':score.reshape(*score.shape[:-1],-1,a.block_size).sum(-1)})
   hooks.append(m.register_forward_hook(hook))
 for r in read_jsonl(a.calibration):
  text=r.get('text') or r.get('prompt'); x=tok(text,return_tensors='pt').to(model.device)
  with torch.no_grad(): model(**x)
 for h in hooks:h.remove()
 a.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'block_size':a.block_size,'traces':traces},a.out);write_json(a.out.with_suffix('.json'),{'trace_entries':len(traces),'block_size':a.block_size})
 print(a.out)
if __name__=='__main__':main()
