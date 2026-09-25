#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse,json,platform
import torch
from experiments.factorized_memory_attention_v1.fma.budget import parameter_budget
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, load_shape

p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--rank",type=int,default=32); p.add_argument("--baseline-params",type=int,default=596049920)
a=p.parse_args(); shape=load_shape(a.config); b=parameter_budget(shape,FMAConfig(rank=a.rank),baseline_params=a.baseline_params)
result={"kind":"FMA_RESOURCE_PROBE","python":platform.python_version(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"shape":shape.to_dict(),"budget":b.to_dict()}
if torch.cuda.is_available():
    result["cuda"]={"device":torch.cuda.get_device_name(0),"total_bytes":torch.cuda.get_device_properties(0).total_memory,"allocated_bytes":torch.cuda.memory_allocated(0),"reserved_bytes":torch.cuda.memory_reserved(0)}
else:
    result["cuda"]={"status":"BLOCKED","reason":"agent runtime is CPU-only; no RTX 4070 VRAM claim made"}
print(json.dumps(result,indent=2))
