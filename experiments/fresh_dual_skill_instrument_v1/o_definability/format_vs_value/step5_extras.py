"""Step 5: the 17 Supa FORMAT rows with shapes, and the clock +60-minute error pattern."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all, OUT, sk  # noqa: E402

R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
scored, correct = score_all(R, arms, targets)
c3 = json.loads((OUT / 'step3_classify.json').read_text(encoding='utf-8'))
ids = [r['id'] for r in probe]
res = {}
fr = c3['supa_wrong|permissive']['format_rows']
res['supa_format_rows_permissive'] = fr
print('Supa FORMAT rows (permissive):', len(fr), Counter(r['shape'] for r in fr.values()))
for s, r in fr.items():
    print(f"  {s} [{r['shape']}] SUPA {r['supa']!r} -> BF16 {r['opallo_bf16']!r} ({r['opallo_bf16_cat']})")
print('Supa NONTERM rows:', c3['supa_wrong|permissive']['nonterm_rows'])
print('Supa new_tokens max:', max(arms['supa'][i]['new_tokens'] for i in ids),
      'nonterm new_tokens:', [arms['supa'][i]['new_tokens'] for i in ids if not arms['supa'][i]['terminated']])

# clock rows: offset of the whole-text parsed value from the target (terminated, parseable only)
clock = [i for i in ids if targets['permissive'][i]['kind'] == 'clock']
for arm in arms:
    off = Counter()
    for i in clock:
        s = scored[(arm, 'permissive')][i]
        if s['parsed'] is None:
            off['unparsed'] += 1
            continue
        d = (int(s['parsed']) - targets['permissive'][i]['minute_of_day']) % 1440
        d = d - 1440 if d > 720 else d
        off[d] += 1
    res[f'clock_offsets|{arm}'] = {str(k): v for k, v in off.items()}
    print(f'clock rows ({len(clock)}) whole-text offset from target, {arm}:', dict(off))
(OUT / 'step5_extras.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
