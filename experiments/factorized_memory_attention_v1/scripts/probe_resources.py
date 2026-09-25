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
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank

p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--rank",type=int,default=32); p.add_argument("--baseline-params",type=int,default=596049920)
a=p.parse_args(); shape=load_shape(a.config); b=parameter_budget(shape,FMAConfig(rank=a.rank),baseline_params=a.baseline_params)
result={"kind":"FMA_RESOURCE_PROBE","python":platform.python_version(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"shape":shape.to_dict(),"budget":b.to_dict()}
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    bank = FactorizedMemoryBank(shape, FMAConfig(rank=a.rank)).to(device=device, dtype=torch.bfloat16)
    sample_ids = torch.tensor([[0, 1, 151935]], device=device, dtype=torch.long)
    with torch.no_grad():
        sample = bank.memory(sample_ids, 0)
    torch.cuda.synchronize(device)
    result["cuda"]={"device":torch.cuda.get_device_name(0),"total_bytes":torch.cuda.get_device_properties(0).total_memory,"allocated_bytes":torch.cuda.memory_allocated(0),"reserved_bytes":torch.cuda.memory_reserved(0),"peak_bytes_during_factorized_probe":torch.cuda.max_memory_allocated(device),"probe_parameter_bytes":b.factorized_bf16_bytes,"probe_output_shape":list(sample.shape),"full_dense_table_instantiated":False}
    del sample, sample_ids, bank
    torch.cuda.empty_cache()
else:
    result["cuda"]={"status":"BLOCKED","reason":"agent runtime is CPU-only; no RTX 4070 VRAM claim made","full_dense_table_instantiated":False}
print(json.dumps(result,indent=2))
