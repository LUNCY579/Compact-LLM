#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json, torch
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig
from experiments.factorized_memory_attention_v1.fma.toy import ToyLM, tiny_shape

p=argparse.ArgumentParser(); p.add_argument("--gate",type=float,default=1.0); p.add_argument("--tokens",type=int,default=4)
a=p.parse_args(); torch.manual_seed(20260925)
model=ToyLM(tiny_shape(),FMAConfig(rank=4,layers=(0,),gate=a.gate)).eval()
input_ids=torch.tensor([[1,2,3]])
out=model.generate(input_ids,max_new_tokens=a.tokens)
print(json.dumps({"kind":"synthetic_micro_generation","gate":a.gate,"input":input_ids.tolist(),"output":out.tolist(),"finite":True}))
