from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from scripts.common.experiment_utils import write_json
ROOT=Path(__file__).resolve().parents[2]
class Router(nn.Module):
 def __init__(self,h,w,b):super().__init__();self.net=nn.Sequential(nn.Linear(h,w,bias=False),nn.SiLU(),nn.Linear(w,b,bias=False))
 def forward(self,x):return self.net(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_7_router/router.pt');p.add_argument('--width',type=int,default=128);p.add_argument('--epochs',type=int,default=10);p.add_argument('--target-ratio',type=float,default=.65);a=p.parse_args();d=torch.load(a.dataset,map_location='cpu',weights_only=False);x=d['hidden'].float();y=d['mask'].float();m=Router(x.shape[-1],a.width,y.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(a.epochs):
  logits=m(x); mask_loss=nn.functional.binary_cross_entropy_with_logits(logits,y); budget=(torch.sigmoid(logits).mean()-a.target_ratio).square();loss=mask_loss+.1*budget;opt.zero_grad();loss.backward();opt.step()
 a.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'hidden_size':x.shape[-1],'router_width':a.width,'num_blocks':y.shape[-1],'target_ratio':a.target_ratio},a.out);write_json(a.out.with_suffix('.json'),{'final_loss':float(loss),'samples':len(x)})
if __name__=='__main__':main()
