"""Q5: the known answer an S-side NLL endpoint must reproduce, re-derived from recorded generations.

For each recorded arm on T_BANK_V2: strict exact, period-tolerant (gold + exactly one '.'),
trailing-period count. Controls: the recorded 'correct' flags must equal my strict recompute
(identity); shuffled golds within family must give ~0 strict.
"""
import collections
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TBANK, dump, sha  # noqa: E402

D = TBANK.parent
assert sha(TBANK).startswith('fc094136')
gold = {r['id']: r['reference'] for r in json.loads(TBANK.read_text(encoding='utf-8'))['rows']}
fam = {r['id']: r['family'] for r in json.loads(TBANK.read_text(encoding='utf-8'))['rows']}
arms = {}
for f in ('CALIBRATION_V2.json', 'V2_ARMS.json'):
    d = json.loads((D / f).read_text(encoding='utf-8'))
    assert d['bank_sha256'].startswith('fc094136')
    for a in d['arms']:
        arms[a['arm']] = (f, sha(D / f)[:8], a)
rng = random.Random(0)
ids = sorted(gold)
res = {}
print(f"{'arm':15s} file              strict rec_strict identity  tol  trailing'.'  neg(shuffled gold)")
for name, (f, h, a) in arms.items():
    P = {p['id']: p['generation'] for p in a['predictions']}
    assert set(P) == set(gold)
    strict = {i for i in ids if P[i].strip() == gold[i]}
    rec = {p['id'] for p in a['predictions'] if p['correct']}
    tol = {i for i in ids if P[i].strip() in (gold[i], gold[i] + '.')}
    trail = sum(P[i].strip().endswith('.') for i in ids)
    by = collections.defaultdict(list)
    for i in ids:
        by[fam[i]].append(i)
    sh = {}
    for fm, L in by.items():
        M = L[:]
        rng.shuffle(M)
        sh.update(dict(zip(L, M)))
    neg = sum(P[i].strip() == gold[sh[i]] for i in ids if sh[i] != i)
    res[name] = {'file': f, 'sha8': h, 'strict': len(strict), 'recorded_correct': a['correct'],
                 'identity': strict == rec, 'period_tolerant': len(tol), 'trailing_period': trail, 'neg_shuffled': neg,
                 'tol_minus_strict_all_gold_plus_period': all(P[i].strip() == gold[i] + '.' for i in tol - strict)}
    print(f"{name:15s} {f:17s} {len(strict):6d} {a['correct']:10d} {str(strict == rec):8s} {len(tol):4d} {trail:11d} {neg:6d}")
print('wrote q5_s_known_answer.json', dump('q5_s_known_answer.json', res))
