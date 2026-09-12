"""Can this programme move to the GPU without moving its numbers?

Two separate questions get conflated when someone says "just use the GPU":

  1. DEVICE. CUDA FP32 is not CPU FP32. Different kernels, different reduction orders. On
     Ada the cuDNN TF32 path is ON by default, which silently drops matmul mantissas from 24
     bits to 10 -- and this programme's whole HOTFIX-001/002 parity investigation turns on
     elementwise differences at a 1e-4 tolerance.
  2. RUNTIME. requirements.txt pins `torch==2.10.0` and says explicitly that it "does not
     prescribe a CUDA build", so CUDA is permitted -- but the only CUDA environment on this
     host is torch 2.12.1, which violates the pin. A different torch version changes kernels
     on its own, independently of the device.

Running the same rows in three configurations separates them:

    cpu  @ 2.10.0+cpu   the reference every existing number was produced on
    cpu  @ 2.12.1+cu126 isolates the RUNTIME change
    cuda @ 2.12.1+cu126 isolates the DEVICE change (TF32 off, then on)

Writes raw logits so the configurations can be differenced exactly rather than compared
through summary statistics.

Exploratory. Outside the hashed root; no receipt, no gate. Open splits only.

usage: device_parity.py <tag> <device> [rows] [--tf32]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _load_tokenizer

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
         r'\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
           r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
OUT = E / 'outputs/device_parity'

TAG = sys.argv[1] if len(sys.argv) > 1 else 'cpu_ref'
DEVICE = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TF32 = '--tf32' in sys.argv

if DEVICE == 'cuda':
    # Default on Ada is matmul TF32 off but cuDNN TF32 ON. Both are set explicitly so the
    # configuration is recorded rather than inherited.
    torch.backends.cuda.matmul.allow_tf32 = TF32
    torch.backends.cudnn.allow_tf32 = TF32

tok = _load_tokenizer(SRC)
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
enc = lambda t: tok(t, return_tensors='pt', add_special_tokens=False)['input_ids']
dev = [x for x in (enc(r['text']) for r in jl(E / 'outputs/splits/dev.jsonl')[:ROWS])
       if x.shape[1] >= 2]

t_load = time.perf_counter()
# `_load` takes the device directly: it reads safetensors on CPU then casts to
# (device, float32). Loading on CPU and moving afterwards would be an extra copy and a
# second chance for a dtype to drift.
model, _, _ = _load(SRC, DEVICE)
model.eval()
if DEVICE == 'cuda':
    torch.cuda.synchronize()
t_load = time.perf_counter() - t_load

logits, ce, ntok = [], 0.0, 0
t0 = time.perf_counter()
with torch.inference_mode():
    for ids in dev:
        out = model(ids.to(DEVICE)).logits.float().cpu()
        logits.append(out[0].numpy().astype(np.float32))
        lp = F.log_softmax(out[:, :-1], -1)
        ce += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), ids[:, 1:].reshape(-1),
                               reduction='sum'))
        ntok += ids.shape[1] - 1
if DEVICE == 'cuda':
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

peak = (torch.cuda.max_memory_allocated() / 2**30) if DEVICE == 'cuda' else None
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / f'logits_{TAG}.npy', np.concatenate(logits, axis=0))
rec = {'tag': TAG, 'device': DEVICE, 'torch': torch.__version__,
       'cuda_available': torch.cuda.is_available(),
       'tf32_matmul': torch.backends.cuda.matmul.allow_tf32 if DEVICE == 'cuda' else None,
       'tf32_cudnn': torch.backends.cudnn.allow_tf32 if DEVICE == 'cuda' else None,
       'rows': len(dev), 'target_tokens': ntok, 'dev_ce_nats': ce / ntok,
       'forward_seconds': elapsed, 'tokens_per_second': ntok / elapsed,
       'load_seconds': t_load, 'peak_vram_gib': peak}
(OUT / f'PARITY_{TAG}.json').write_text(json.dumps(rec, indent=2) + '\n', encoding='utf-8')
print(json.dumps(rec, indent=2), flush=True)
