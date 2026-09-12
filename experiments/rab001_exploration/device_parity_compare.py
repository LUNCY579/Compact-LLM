"""Difference the raw logits from two `device_parity.py` runs.

Summary statistics hide the thing that matters. This programme's parity work turns on
ELEMENTWISE differences at a 1e-4 tolerance, so a device move has to be judged the same way:
max |delta| over every logit, not a mean, and top-1 agreement on top of that.

usage: device_parity_compare.py <tag_a> <tag_b> [...]
"""
import itertools, json, sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OUT = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
           r'\experiments\architecture_benchmark_001\outputs\device_parity')
TAGS = sys.argv[1:] or ['cpu_ref', 'cuda_fp32', 'cuda_tf32']

have, meta = {}, {}
for t in TAGS:
    f, m = OUT / f'logits_{t}.npy', OUT / f'PARITY_{t}.json'
    if f.is_file() and m.is_file():
        have[t] = np.load(f)
        meta[t] = json.loads(m.read_text(encoding='utf-8'))
    else:
        print(f'  missing: {t}')

print(f"{'tag':12}{'device':7}{'torch':16}{'tf32':7}{'devCE':>10}{'tok/s':>9}{'VRAM GiB':>10}")
for t, m in meta.items():
    tf = '-' if m['device'] == 'cpu' else ('on' if m.get('tf32_cudnn') else 'off')
    vr = f"{m['peak_vram_gib']:.2f}" if m.get('peak_vram_gib') else '-'
    print(f"{t:12}{m['device']:7}{m['torch']:16}{tf:7}{m['dev_ce_nats']:>10.6f}"
          f"{m['tokens_per_second']:>9.1f}{vr:>10}")

pairs = []
print(f"\n{'comparison':28}{'max|dlogit|':>13}{'mean|dlogit|':>14}{'top1 agree':>12}{'dCE nats':>11}")
for a, b in itertools.combinations(have, 2):
    x, y = have[a], have[b]
    if x.shape != y.shape:
        print(f'  {a} vs {b}: SHAPE MISMATCH {x.shape} vs {y.shape} -- tokenization differs, '
              f'these runs are not comparable at all')
        continue
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    agree = float((x.argmax(-1) == y.argmax(-1)).mean())
    rec = {'a': a, 'b': b, 'max_abs_logit_delta': float(d.max()),
           'mean_abs_logit_delta': float(d.mean()), 'top1_agreement': agree,
           'dev_ce_delta_nats': meta[b]['dev_ce_nats'] - meta[a]['dev_ce_nats'],
           'within_1e-4': bool(d.max() < 1e-4)}
    pairs.append(rec)
    print(f"{a + ' vs ' + b:28}{rec['max_abs_logit_delta']:>13.3e}"
          f"{rec['mean_abs_logit_delta']:>14.3e}{agree:>12.4f}{rec['dev_ce_delta_nats']:>11.2e}")

if pairs:
    (OUT / 'DEVICE_PARITY.json').write_text(json.dumps({
        'schema': 1, 'evidence_kind': 'exploratory_device_parity_only',
        'note': 'Raw-logit differences between runtimes/devices on identical rows. The '
                'programme tolerance for elementwise parity is 1e-4; `within_1e-4` says '
                'whether a configuration could be compared against the existing CPU FP32 '
                'corpus at that tolerance. No receipt, no gate, no efficacy claim.',
        'configurations': meta, 'comparisons': pairs,
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }, indent=2) + '\n', encoding='utf-8')
    print('\nwrote', OUT / 'DEVICE_PARITY.json')
