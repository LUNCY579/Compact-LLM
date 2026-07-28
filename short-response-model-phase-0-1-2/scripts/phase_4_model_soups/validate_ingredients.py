from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json, write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('adapters',nargs='+',type=Path); p.add_argument('--out',type=Path,default=ROOT/'reports/PHASE_4_COMPATIBILITY.json'); a=p.parse_args()
 configs=[]
 for d in a.adapters:
  c=d/'adapter_config.json'
  if not c.exists(): raise SystemExit(f'Missing {c}')
  configs.append(read_json(c))
 keys=('base_model_name_or_path','r','lora_alpha','target_modules','peft_type','task_type')
 ref={k:configs[0].get(k) for k in keys}; mismatches=[]
 for i,c in enumerate(configs[1:],1):
  for k in keys:
   if c.get(k)!=ref[k]: mismatches.append({'ingredient':i,'field':k,'expected':ref[k],'actual':c.get(k)})
 write_json(a.out,{'ingredients':[str(x) for x in a.adapters],'reference':ref,'mismatches':mismatches,'status':'PASS' if not mismatches else 'FAIL'})
 if mismatches: raise SystemExit('Incompatible adapters')
 print('Compatibility PASS')
if __name__=='__main__': main()
