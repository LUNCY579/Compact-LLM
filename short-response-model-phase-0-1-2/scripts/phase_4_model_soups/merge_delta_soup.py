from __future__ import annotations
import argparse, gc
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.common.experiment_utils import write_json
ROOT=Path(__file__).resolve().parents[2]
def load_merged(model_id,adapter):
 b=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,device_map='cpu')
 return PeftModel.from_pretrained(b,str(adapter)).merge_and_unload().state_dict()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-id',default='Qwen/Qwen3-0.6B'); p.add_argument('--adapters',nargs='+',type=Path,required=True); p.add_argument('--weights',nargs='+',type=float); p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/phase_4_soup/merged_model'); a=p.parse_args()
 ws=a.weights or [1/len(a.adapters)]*len(a.adapters)
 if len(ws)!=len(a.adapters) or abs(sum(ws)-1)>1e-6: raise SystemExit('Weights must match adapters and sum to 1')
 base=AutoModelForCausalLM.from_pretrained(a.model_id,torch_dtype=torch.float32,device_map='cpu'); base_sd=base.state_dict(); accum={k:torch.zeros_like(v,dtype=torch.float32) for k,v in base_sd.items() if v.is_floating_point()}
 for alpha,adapter in zip(ws,a.adapters):
  merged=load_merged(a.model_id,adapter)
  for k in accum: accum[k].add_((merged[k].float()-base_sd[k].float()),alpha=alpha)
  del merged; gc.collect()
 for k,v in base_sd.items():
  if k in accum: v.copy_((v.float()+accum[k]).to(v.dtype))
 a.output_dir.mkdir(parents=True,exist_ok=True); base.save_pretrained(a.output_dir,safe_serialization=True); AutoTokenizer.from_pretrained(a.model_id).save_pretrained(a.output_dir)
 write_json(a.output_dir/'soup_manifest.json',{'model_id':a.model_id,'adapters':[str(x) for x in a.adapters],'weights':ws,'merge_space':'FP32 reconstructed deltas'})
 print('Saved',a.output_dir)
if __name__=='__main__': main()
