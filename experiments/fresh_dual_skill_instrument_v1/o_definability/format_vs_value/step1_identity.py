"""Step 1: reproduce recorded correct sets BY IDENTITY, then list gained / lost rows."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all, OUT, CLASSES  # noqa: E402

R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
print('hashes:', {k: v[:8] for k, v in hashes.items()})
scored, correct = score_all(R, arms, targets)
checks = {}
# Supa recorded row targets (permissive receipt) vs audit-built permissive targets
checks['supa_row_targets_equal_permissive_overlay'] = all(
    r['target'] == targets['permissive'][r['id']] for r in supa['rows'])
checks['supa_recorded_source_correct_flags_equal'] = all(
    r['source']['correct'] == (r['id'] in correct[('supa', 'permissive')]) for r in supa['rows'])
for pol in R.BARE_POLICIES:
    checks[f'supa_{pol}_ids_identical'] = (
        set(supa['reference_ids_by_policy'][pol]) == correct[('supa', pol)])
    checks[f'bf16_{pol}_ids_identical'] = (
        set(panel['by_policy'][pol]['correct_ids']) == correct[('opallo_bf16', pol)])
checks['supa_source_correct_ids_identical'] = (
    set(supa['source_correct_ids']) == correct[('supa', 'permissive')])
fr = fp32['checkpoints'][-1]['retention']
checks['fp32_per_row_equals_permissive'] = (
    {i for i, v in fr['per_row'].items() if v} == correct[('opallo_fp32', 'permissive')])
for pol in R.BARE_POLICIES:
    r = fr[pol]
    ref = set(supa['reference_ids_by_policy'][pol])
    got = correct[('opallo_fp32', pol)]
    rec_lost = set().union(*[set(c['lost_ids']) for c in r['per_class'].values()])
    checks[f'fp32_{pol}_total'] = (r['correct_total'], len(got), r['correct_total'] == len(got))
    checks[f'fp32_{pol}_lost_ids_identical'] = rec_lost == (ref - got)
    checks[f'fp32_{pol}_per_class_arm_correct'] = all(
        c['arm_correct'] == sum(1 for i in got if i.split('__')[1] == cls)
        for cls, c in r['per_class'].items())
    checks[f'fp32_{pol}_gained_total'] = r['gained_total'] == len(got - ref)
cp = panel['cast_parity']
diff = sorted(i for i in arms['supa']
              if arms['opallo_bf16'][i]['text'] != arms['opallo_fp32'][i]['text'])
checks['cast_parity_differing_ids'] = sorted(cp['differing_ids']) == diff
checks['cast_parity_lost_vs_fp32'] = sorted(cp['lost_vs_float32']) == sorted(
    correct[('opallo_fp32', 'permissive')] - correct[('opallo_bf16', 'permissive')])
checks['cast_parity_gained_vs_fp32'] = sorted(cp['gained_vs_float32']) == sorted(
    correct[('opallo_bf16', 'permissive')] - correct[('opallo_fp32', 'permissive')])
for k, v in checks.items():
    print(f'  {k}: {v}')
ok = all((v[-1] if isinstance(v, tuple) else v) for v in checks.values())
counts = {f'{a}|{p}': len(v) for (a, p), v in correct.items()}
print('counts', counts)
print('IDENTITY OK' if ok else 'IDENTITY FAILED')
cls_of = {r['id']: r['reply_class'] for r in probe}
res = {'identity_ok': ok,
       'checks': {k: (list(v) if isinstance(v, tuple) else v) for k, v in checks.items()},
       'hashes': hashes, 'counts': counts}
for opal in ('opallo_bf16', 'opallo_fp32'):
    for pol in R.BARE_POLICIES:
        S, O = correct[('supa', pol)], correct[(opal, pol)]
        g, l = sorted(O - S), sorted(S - O)
        bycls = {c: {'gained': sum(cls_of[i] == c for i in g),
                     'lost': sum(cls_of[i] == c for i in l),
                     'supa': sum(cls_of[i] == c for i in S),
                     'opallo': sum(cls_of[i] == c for i in O)} for c in CLASSES}
        print(f'{opal} vs supa [{pol}]: supa {len(S)} opallo {len(O)} gained {len(g)} '
              f'lost {len(l)} net {len(g) - len(l)}')
        for c in CLASSES:
            print('     ', c, bycls[c])
        res[f'{opal}|{pol}'] = {'gained': g, 'lost': l, 'by_class': bycls}
(OUT / 'step1.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
(OUT / 'correct_sets.json').write_text(
    json.dumps({f'{a}|{p}': sorted(v) for (a, p), v in correct.items()}, indent=1),
    encoding='utf-8')
