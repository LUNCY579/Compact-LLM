"""Step 2: positive and negative controls for the value-presence extractor."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all, OUT, CLASSES, sk  # noqa: E402
from extract import presence, want_of, NUMERIC_KINDS  # noqa: E402

R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
scored, correct = score_all(R, arms, targets)
aud = {r['id']: r for r in audit['rows']}
ids = [r['id'] for r in probe]
cls_of = {r['id']: r['reply_class'] for r in probe}
UNIT_SCALE = {'minute': 60, 'hour': 3600, 'second': 1}
dscale = {i: UNIT_SCALE.get(aud[i]['proposed_typed_target'].get('unit'))
          for i in ids if aud[i]['proposed_typed_target']['type'] == 'duration'}
VARIANTS = ('loose', 'typed', 'final')
res = {}


def P(text, t, i):
    return presence(text, t, R, display_scale=dscale.get(i))


# ---- U: hand-built unit cases with a known answer ------------------------------------------
T = targets['permissive']
def tid(suf):
    return next(i for i in ids if i.endswith(suf))
unit_cases = [  # (row suffix, text, expected loose)
    ('02515', 'A 15% tip on a $60 bill is $9.', True),      # money 9
    ('02515', 'A 15% tip on a $60 bill is $19.', False),
    ('02677', '9:15 AM - 8:30 AM = 45 minutes', True),      # duration 2700
    ('02229', '2 hours 15 minutes', False),                 # 7200: sub-run must NOT count
    ('02229', '2 hours, 15 minutes later: 2 hours', True),
    ('00982', '1 hour and 30 minutes.', True),              # 5400
    ('00982', '90 minutes', True),
    ('00982', '1.5 hours', True),
    ('00982', '90', False),                                  # bare not permitted on this row
    ('02628', '11:30 AM + 50 minutes = 12:20 PM', True),    # clock 740
    ('02628', '12:20', False),                               # 24-hour 12:20 is 740? -> True!
    ('02842', '9:15 AM + 55 minutes = 9:70 AM', False),     # 610; overflow is not a value
    ('01293', '3 1/5 cups', False),                          # 16/5: mixed numbers not in grammar
    ('01293', '16/5 cups', True),
    ('01293', '3.2 cups', True),
    ('02752', '55000 m', True),                              # distance 55 km
    ('02025', '50% of 120 units = 60 units.', True),        # quantity 60
    ('02025', '50% of 120 units = 160 units.', False),
    ('01221', '25', True),                                   # bare-minute row 1500 s
    ('02961', '1 hour and 30 minutes', False),               # 1800: `30 minutes` sub-run
    ('02961', 'It is 30 minutes.', True),
    ('02681', '9:55 PM', False),                             # 595 = 9:55 AM
    ('02681', 'It ends at 9:55 AM.', True),
    ('02681', '9:55', True),                                 # 24-hour 9:55 == 595
]
TYPED_EXPECT = {('02961', '1 hour and 30 minutes'): False, ('02681', '9:55 PM'): False,
                ('02229', '2 hours 15 minutes'): False, ('02961', 'It is 30 minutes.'): True,
                ('02677', '9:15 AM - 8:30 AM = 45 minutes'): True,
                ('02628', '11:30 AM + 50 minutes = 12:20 PM'): True}
ucase = []
for suf, text, exp in unit_cases:
    i = tid(suf)
    got = P(text, T[i], i)
    ucase.append({'row': suf, 'text': text, 'expected_loose': exp, 'got': got['loose'],
                  'final': got['final'], 'typed': got['typed'], 'carry': got['carry']})
# 12:20 read as 24-hour IS 740 minutes (12:20 is 12:20 pm in 24h) -> the parser's semantics
for c in ucase:
    if c['row'] == '02628' and c['text'] == '12:20':
        c['expected_loose'] = True
        c['note'] = '24-hour 12:20 == 740 by the scorer parser; expectation corrected a priori'
for c in ucase:
    c['expected_typed'] = TYPED_EXPECT.get((c['row'], c['text']))
res['unit_cases'] = ucase
res['unit_cases_all_pass'] = all(
    c['expected_loose'] == c['got'] and c['expected_typed'] in (None, c['typed']) for c in ucase)
print('unit cases pass:', res['unit_cases_all_pass'], 'n =', len(ucase),
      'typed-expectations', sum(c['expected_typed'] is not None for c in ucase))
for c in ucase:
    if c['expected_loose'] != c['got'] or c['expected_typed'] not in (None, c['typed']):
        print('   FAIL', c)

# ---- P0: gold answers --------------------------------------------------------------------
p0 = {}
for pol in R.BARE_POLICIES:
    sc = Counter(); pr = Counter()
    for i in ids:
        g = aud[i]['original_answer']
        sc[R.score_prediction(g, True, targets[pol][i])['correct']] += 1
        p = P(g, targets[pol][i], i)
        for v in VARIANTS:
            pr[v] += p[v]
    p0[pol] = {'gold_scored_correct': sc[True], 'n': len(ids), **{f'{v}_present': pr[v] for v in VARIANTS}}
res['P0_gold'] = p0
print('P0 gold:', p0)

# ---- P1: every scorer-correct row must be value-present ----------------------------------
p1 = {}
for (arm, pol), cset in correct.items():
    cnt = Counter()
    miss = defaultdict(list)
    for i in cset:
        p = P(arms[arm][i]['text'], targets[pol][i], i)
        for v in VARIANTS:
            cnt[v] += p[v]
            if not p[v]:
                miss[v].append(i)
    p1[f'{arm}|{pol}'] = {'n_correct': len(cset), **{f'{v}_present': cnt[v] for v in VARIANTS},
                          'misses': dict(miss)}
res['P1_correct_rows'] = p1
for k, v in p1.items():
    print('P1', k, {kk: vv for kk, vv in v.items() if kk != 'misses'}, 'misses' if v['misses'] else '')

# ---- P2: synthetic FORMAT damage around the gold answer ----------------------------------
WRAPS = {
    'prose_prefix': lambda g: 'The final result comes to ' + g,
    'prose_suffix': lambda g: g.rstrip('.') + ' in total.',
    'working_then_answer': lambda g: 'Working it through step by step, we get ' + g.rstrip('.') + ', so that is it.',
    'restated_subject': lambda g: 'For this note the value is ' + g,
    'nonterminated': lambda g: g.rstrip('.') + ' because the',   # scored with terminated=False
}
p2 = {}
for pol in R.BARE_POLICIES:
    for w, f in WRAPS.items():
        rej = fired = fired_rej = 0
        for i in ids:
            s = f(aud[i]['original_answer'])
            term = (w != 'nonterminated')
            ok = R.score_prediction(s, term, targets[pol][i])['correct']
            p = P(s, targets[pol][i], i)['loose']
            rej += (not ok); fired += p; fired_rej += (p and not ok)
        p2[f'{pol}|{w}'] = {'n': len(ids), 'scorer_rejected': rej, 'loose_fired': fired,
                            'loose_fired_on_rejected': fired_rej}
res['P2_synthetic_format_damage'] = p2
for k, v in p2.items():
    print('P2', k, v)

# ---- N: negative controls ----------------------------------------------------------------
def fam(t):
    return 'numeric' if t['kind'] in NUMERIC_KINDS else t['kind']


def derange(key_fn):
    groups = defaultdict(list)
    for i in ids:
        groups[key_fn(i)].append(i)
    m = {}
    for g in groups.values():
        for k, i in enumerate(g):
            m[i] = g[(k + 1) % len(g)] if len(g) > 1 else None
    return m


D = {'D1_shift1_within_class': derange(lambda i: cls_of[i]),
     'D2_shift1_within_class_and_kind': derange(lambda i: (cls_of[i], targets['permissive'][i]['kind']))}


def near_miss(t):
    t = dict(t)
    if t['kind'] == 'clock':
        t['minute_of_day'] = (t['minute_of_day'] + 10) % 1440
    elif t['kind'] == 'duration':
        t['value'] = str(R.parse_rational(str(t['value'])) + 600)
    else:
        t['value'] = str(R.parse_rational(str(t['value'])) + 1)
    return t


neg = {}
for pol in R.BARE_POLICIES:
    for dname, dm in list(D.items()) + [('N3_near_miss_plus1_or_plus10min', None)]:
        for arm in arms:
            wrong = set(ids) - correct[(arm, pol)]
            cnt = Counter(); cnt_w = Counter(); n = n_w = coll = 0
            fps = defaultdict(list)
            for i in ids:
                own = targets[pol][i]
                if dm is None:
                    t = near_miss(own)
                else:
                    if dm[i] is None:
                        continue
                    t = targets[pol][dm[i]]
                if fam(t) == fam(own) and want_of(t, R) == want_of(own, R):
                    coll += 1
                    continue
                p = P(arms[arm][i]['text'], t, i if dm is None else dm[i])
                n += 1
                inw = i in wrong
                n_w += inw
                for v in VARIANTS + ('carry',):
                    cnt[v] += p[v]
                    cnt_w[v] += (p[v] and inw)
                    if p[v] and v == 'loose':
                        fps[v].append(sk(i))
            neg[f'{pol}|{dname}|{arm}'] = {
                'pairs': n, 'collisions_excluded': coll,
                **{f'{v}_fp': cnt[v] for v in VARIANTS + ('carry',)},
                'pairs_on_arm_wrong_rows': n_w,
                **{f'{v}_fp_on_wrong_rows': cnt_w[v] for v in VARIANTS},
                'loose_fp_rows': fps['loose']}
res['N_negative_controls'] = neg
for k, v in neg.items():
    print('N', k, {kk: vv for kk, vv in v.items() if kk != 'loose_fp_rows'}, v['loose_fp_rows'])
(OUT / 'step2_controls.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
