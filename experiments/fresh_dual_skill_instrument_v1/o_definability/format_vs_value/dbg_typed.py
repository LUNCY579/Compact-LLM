import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '.')
from common import load_all
from extract import presence, want_of, loose_values
R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
exec(open('step2_controls.py').read().split('neg = {}')[0].split('# ---- U:')[0])
def near_miss(t):
    t = dict(t)
    if t['kind'] == 'clock':
        t['minute_of_day'] = (t['minute_of_day'] + 10) % 1440
    elif t['kind'] == 'duration':
        t['value'] = str(R.parse_rational(str(t['value'])) + 600)
    else:
        t['value'] = str(R.parse_rational(str(t['value'])) + 1)
    return t
for arm in arms:
    for i in ids:
        t = near_miss(targets['permissive'][i])
        p = presence(arms[arm][i]['text'], t, R)
        if p['typed'] != p['loose']:
            print(arm, i[-5:], repr(arms[arm][i]['text']), t, p)
