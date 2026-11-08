"""Ordinary-control version: per-layer chunk divergence in the unmodified official B28."""
import torch
from pathlib import Path
from experiments.architecture_benchmark_001.run import _load, _prompt, load_rows

SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
model, tok, _ = _load(SRC, 'cpu')
model.eval()
row = load_rows(Path('experiments/architecture_benchmark_001/data/pilot.jsonl'), 'pilot')[0]
ids = _prompt(tok, row, 'cpu')
T = ids.shape[1]
print(f'B28 plain control: {len(model.encoder)} encoder layers, memory={model.memory}, T={T}')

cap = {}
def hook(i):
    def f(mod, inp, out):
        cap.setdefault(i, []).append(out[0].detach().clone())
    return f
hs = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(model.encoder)]

def run(chunk):
    cap.clear()
    with torch.inference_mode():
        if chunk is None:
            model(ids, use_cache=True)
        else:
            cache = None
            for s in range(0, T, chunk):
                cache = model(ids[:, s:s+chunk], past=cache, use_cache=True).cache
    return {k: torch.cat(v, dim=1) for k, v in cap.items()}

full = run(None)
for chunk in (7, 1):
    ch = run(chunk)
    d = [ (full[k]-ch[k]).abs().max().item() for k in sorted(full) ]
    print(f'=== B28 chunk {chunk}: per-layer max|delta| ===')
    print('   layer00 =', f'{d[0]:.3e}', ' layer01 =', f'{d[1]:.3e}',
          ' layer13 =', f'{d[13]:.3e}', ' layer26 =', f'{d[26]:.3e}', ' layer27 =', f'{d[27]:.3e}')
    print(f'   growth layer00 -> layer13: {d[13]/d[0] if d[0] else float("nan"):.1f}x ; -> layer27: {d[27]/d[0] if d[0] else float("nan"):.1f}x')
for h in hs: h.remove()
