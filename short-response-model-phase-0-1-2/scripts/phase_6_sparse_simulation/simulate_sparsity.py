from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from scripts.common.experiment_utils import write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--ratios',nargs='+',type=float,default=[1,.8,.65,.5,.35]);p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_6_sparse/frontier.json');a=p.parse_args();data=torch.load(a.traces,map_location='cpu',weights_only=False); rows=[]
 for ratio in a.ratios:
  retained=[]
  for t in data['traces']:
   s=t['block_scores'].flatten(); k=max(1,int(round(len(s)*ratio))); retained.append(float(torch.topk(s,k).values.sum()/s.sum().clamp_min(1e-12)))
  rows.append({'active_ratio':ratio,'mean_oracle_contribution_retained':sum(retained)/max(1,len(retained)),'quality_evaluation_required':True,'speed_claim_allowed':False})
 write_json(a.out,{'frontier':rows,'warning':'This oracle simulation uses dense computation and makes no speed claim.'});print(a.out)
if __name__=='__main__':main()
