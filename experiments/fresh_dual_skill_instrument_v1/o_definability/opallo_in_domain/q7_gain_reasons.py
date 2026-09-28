"""Q7 (side check): for the rows Opallo FP32 GAINED over Supa on the panel, why did Supa fail?
Uses retention.score_prediction's own reason/kind fields; positive control = Supa's correct rows."""
import collections, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_DOSE, AUDIT, SUPA_REF, dump, load_scorer, typed_targets  # noqa: E402
R, _ = load_scorer()
T = typed_targets(R, json.loads(AUDIT.read_text(encoding='utf-8')))
supa = {r['id']: (r['source']['raw'], r['source']['terminated']) for r in json.loads(SUPA_REF.read_text(encoding='utf-8'))['rows']}
arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
fp = {i: (v['text'], v['terminated']) for i, v in arm['checkpoints'][-1]['outputs'].items()}
out = {}
for pol in R.BARE_POLICIES:
    S = {i: R.score_prediction(*supa[i], T[pol][i]) for i in supa}
    O = {i: R.score_prediction(*fp[i], T[pol][i]) for i in fp}
    gained = sorted(i for i in S if O[i]['correct'] and not S[i]['correct'])
    reasons = collections.Counter(S[i].get('reason') for i in gained)
    # did Supa's output contain the gold value's surface anywhere? (value-present-but-rejected proxy)
    k0 = next(iter(S)); keys = sorted(S[k0].keys())
    ctrl = collections.Counter(S[i].get('reason') for i in S if S[i]['correct'])
    out[pol] = {'gained': len(gained), 'supa_fail_reasons_on_gained': dict(reasons), 'supa_reason_on_its_correct_rows': dict(ctrl), 'score_keys': keys}
    print(pol, 'gained', len(gained), '| Supa failure reasons on gained rows:', dict(reasons.most_common()))
    print('    POS control, reasons on Supa-correct rows:', dict(ctrl.most_common()), '| fields', keys)
    for i in gained[:4]:
        print('     ', i.split('__')[1], '| supa', repr(supa[i][0][:70]), '| opallo', repr(fp[i][0][:40]), '| parsed', S[i].get('parsed'), '| target', T[pol][i].get('value') if isinstance(T[pol][i], dict) else T[pol][i])
print('wrote q7_gain_reasons.json', dump('q7_gain_reasons.json', out))
