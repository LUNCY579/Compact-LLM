"""Greedy generation on the open pilot split for every arm. Capability sanity check.

Confirms or refutes the CE finding with observable behaviour. Exploratory; outside the
experiment package, so the implementation digest is untouched. Not a receipt.
"""
import gc, json, math, sys
from pathlib import Path
import torch

# This host's console is cp1252. A model emitting non-ASCII token salad would otherwise
# kill a run inside a print AFTER the expensive compute was done -- which is exactly how
# three C28 runs were lost. Output formatting must never destroy a measurement.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _restore, _load_tokenizer, _prompt, load_rows

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
ARMS = [('B28_source', None), ('D40_width', LAD / 'G1_WIDTH/attempt-0001'),
        ('C28_ced', LAD / 'G1_CED_C28/attempt-0001'), ('C40_ced', LAD / 'G1_CED_C40/attempt-0001')]

tok = _load_tokenizer(SRC)
rows = load_rows(E / 'data/pilot.jsonl', 'pilot')
EOS = {151643, 151645}
out = {}

for name, ckpt in ARMS:
    model = (_load(SRC, 'cpu')[0] if ckpt is None else _restore(ckpt, 'cpu')).eval()
    hits, samples, terminated, uniq = 0, [], 0, set()
    with torch.inference_mode():
        for row in rows:
            ids = _prompt(tok, row, 'cpu')
            gen = []
            cur = ids
            cache = None
            for _ in range(24):
                o = model(cur, past=cache, use_cache=True, last_position_only=True)
                cache = o.cache
                nxt = int(o.logits[0, -1].argmax())
                if nxt in EOS:
                    terminated += 1
                    break
                gen.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long)
            text = tok.decode(gen, skip_special_tokens=True)
            uniq.add(text.strip())
            if text.strip() == row['answer'].strip():
                hits += 1
            if len(samples) < 4:
                samples.append({'id': row['id'], 'expected': row['answer'], 'got': text[:80]})
    del model; gc.collect()
    out[name] = {'exact_hits': hits, 'rows': len(rows), 'terminated': terminated,
                 'distinct_outputs': len(uniq), 'samples': samples}
    print(f"{name:12} hits={hits}/{len(rows)} terminated={terminated} distinct_outputs={len(uniq)}", flush=True)
    for s in samples:
        print(f"    {s['id']:14} expected={s['expected']!r:8} got={s['got']!r}", flush=True)
    print(flush=True)

d = E / 'outputs/arm-quality'
d.mkdir(parents=True, exist_ok=True)
(d / 'ARM_GENERATION.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_capability_diagnostic_only',
    'note': 'Greedy decode, 24 new tokens max, open pilot split. Outside the experiment '
            'package; not a receipt and authorizes nothing.',
    'arms': out, 'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('wrote', d / 'ARM_GENERATION.json')
