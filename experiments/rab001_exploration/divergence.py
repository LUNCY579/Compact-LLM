"""Locate the first point of chunk-schedule divergence in CED. Read-only diagnostic."""
import json, torch
from pathlib import Path
from experiments.architecture_benchmark_001.run import _restore, _load_tokenizer, _prompt, load_rows

CK = Path('experiments/architecture_benchmark_001/outputs/ladder-8d54f540-v1/G1_CED_C28/attempt-0001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
model = _restore(CK, 'cpu').eval()
tok = _load_tokenizer(SRC)
row = load_rows(Path('experiments/architecture_benchmark_001/data/pilot.jsonl'), 'pilot')[0]
ids = _prompt(tok, row, 'cpu')
T = ids.shape[1]

cap = {}
def enc_hook(i):
    def f(mod, inp, out):
        cap.setdefault(f'enc{i:02d}', []).append(out[0].detach().clone())
    return f
def mem_hook(mod, inp, out):
    cap.setdefault('memory', []).append((out[0].detach().clone(), out[1].detach().clone()))

hs = [blk.register_forward_hook(enc_hook(i)) for i, blk in enumerate(model.encoder)]
hs.append(model.memory.register_forward_hook(mem_hook))

def run(chunk):
    cap.clear()
    with torch.inference_mode():
        if chunk is None:
            model(ids, use_cache=True)
        else:
            cache = None
            for s in range(0, T, chunk):
                o = model(ids[:, s:s+chunk], past=cache, use_cache=True)
                cache = o.cache
    enc = {k: torch.cat(v, dim=1) for k, v in cap.items() if k.startswith('enc')}
    mem = cap['memory'][-1]
    return enc, mem

full_enc, full_mem = run(None)
for chunk in (7, 1):
    ch_enc, ch_mem = run(chunk)
    print(f'=== chunk size {chunk} vs full forward (T={T}) ===')
    first = None
    for k in sorted(full_enc):
        d = (full_enc[k] - ch_enc[k]).abs().max().item()
        if d > 0 and first is None: first = k
        if k in (sorted(full_enc)[0], sorted(full_enc)[1], sorted(full_enc)[-2], sorted(full_enc)[-1]):
            print(f'  {k} hidden max|delta| = {d:.3e}')
    print(f'  FIRST diverging encoder layer: {first}')
    dk = (full_mem[0] - ch_mem[0]).abs().max().item()
    dv = (full_mem[1] - ch_mem[1]).abs().max().item()
    print(f'  shared memory bank  max|dK| = {dk:.3e}   max|dV| = {dv:.3e}')
    print()
for h in hs: h.remove()
