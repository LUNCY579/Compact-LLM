"""Step 4: prompt-echo positive control, format-blind value accuracy, examples and row lists."""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all, OUT, CLASSES, sk  # noqa: E402
from extract import presence, loose_values  # noqa: E402

R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
scored, correct = score_all(R, arms, targets)
c3 = json.loads((OUT / 'step3_classify.json').read_text(encoding='utf-8'))
aud = {r['id']: r for r in audit['rows']}
ids = [r['id'] for r in probe]
cls_of = {r['id']: r['reply_class'] for r in probe}
full = {sk(i): i for i in ids}
assert len(full) == 150
UNIT_SCALE = {'minute': 60, 'hour': 3600, 'second': 1}
dscale = {i: UNIT_SCALE.get(aud[i]['proposed_typed_target'].get('unit'))
          for i in ids if aud[i]['proposed_typed_target']['type'] == 'duration'}
res = {}

# ---- (a) positive control for the target_in_prompt check ------------------------------------
# For every row, take each value the extractor reads out of the PROMPT under the row's own kind
# and build a target carrying that value; the prompt check must fire on it.
pos = fire = rows_with_values = 0
own_hits = []
for i in ids:
    t = targets['permissive'][i]
    pr = aud[i]['operative_prompt']
    vals = [v for v in loose_values(pr, t, R) if v[3]]
    if vals:
        rows_with_values += 1
    for v in vals:
        tt = dict(t)
        if t['kind'] == 'clock':
            tt['minute_of_day'] = int(v[2])
        else:
            tt['value'] = str(v[2])
        pos += 1
        fire += presence(pr, tt, R)['loose']
    if presence(pr, t, R, display_scale=dscale.get(i))['loose']:
        own_hits.append(sk(i))
res['prompt_echo_positive_control'] = {'rows_with_prompt_values': rows_with_values,
                                       'constructed_targets': pos, 'fired': fire,
                                       'rows_whose_OWN_target_is_in_prompt': own_hits}
print('prompt-echo positive control:', res['prompt_echo_positive_control'])

# ---- (b) format-blind value accuracy: value-present = CORRECT or FORMAT or NONTERM-with-value -
def vp(arm, pol, i):
    s = scored[(arm, pol)][i]
    return s['correct'] or presence(s['raw'], targets[pol][i], R,
                                    display_scale=dscale.get(i))['loose']


def binom_two_sided(k, n):
    """Exact two-sided sign test p for k successes of n at p=1/2."""
    tail = sum(math.comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


for pol in R.BARE_POLICIES:
    for opal in ('opallo_bf16', 'opallo_fp32'):
        S = {i for i in ids if vp('supa', pol, i)}
        O = {i for i in ids if vp(opal, pol, i)}
        sc_S, sc_O = correct[('supa', pol)], correct[(opal, pol)]
        blk = {'supa_value_present': len(S), 'opallo_value_present': len(O),
               'opallo_only': len(O - S), 'supa_only': len(S - O), 'net': len(O) - len(S),
               'sign_test_p_two_sided': round(binom_two_sided(len(O - S), len(O - S) + len(S - O)), 4),
               'scorer_opallo_only': len(sc_O - sc_S), 'scorer_supa_only': len(sc_S - sc_O),
               'scorer_sign_test_p_two_sided': round(binom_two_sided(len(sc_O - sc_S),
                                                     len(sc_O - sc_S) + len(sc_S - sc_O)), 6),
               'supa_only_rows': sorted(sk(i) for i in S - O),
               'by_class': {c: {'supa_vp': sum(cls_of[i] == c for i in S),
                                'opallo_vp': sum(cls_of[i] == c for i in O),
                                'supa_scorer': sum(cls_of[i] == c for i in sc_S),
                                'opallo_scorer': sum(cls_of[i] == c for i in sc_O)}
                            for c in CLASSES}}
        res[f'value_accuracy|{opal}|{pol}'] = blk
        print(f'value accuracy {opal} {pol}:', {k: v for k, v in blk.items() if k != 'by_class'})
        for c in CLASSES:
            print('     ', c, blk['by_class'][c])

# ---- (c) rows that matter, verbatim -----------------------------------------------------------
def row_line(suf, pol='permissive'):
    i = full[suf]
    t = targets[pol][i]
    tv = t.get('minute_of_day', t.get('value'))
    disp = aud[i]['proposed_typed_target'].get('canonical_display') or \
        f"{aud[i]['proposed_typed_target']['value']['rational']} {aud[i]['proposed_typed_target'].get('unit') or ''}".strip()
    return {'row': suf, 'class': cls_of[i], 'target': f"{t['kind']}:{tv} ({disp})",
            'prompt_tail': aud[i]['operative_prompt'][-110:],
            'supa': arms['supa'][i]['text'], 'supa_terminated': arms['supa'][i]['terminated'],
            'supa_reason': scored[('supa', pol)][i]['reason'],
            'opallo_bf16': arms['opallo_bf16'][i]['text'],
            'opallo_fp32': arms['opallo_fp32'][i]['text']}


for pol in R.BARE_POLICIES:
    blk = c3[f'opallo_bf16|{pol}']
    bycat = {}
    for suf, r in blk['gained_rows'].items():
        bycat.setdefault(r['cat'], []).append(suf)
    fp32_only = sorted(set(c3[f'opallo_fp32|{pol}']['gained_rows']) - set(blk['gained_rows']))
    bf16_only = sorted(set(blk['gained_rows']) - set(c3[f'opallo_fp32|{pol}']['gained_rows']))
    res[f'gained_ids|{pol}'] = {'bf16_by_cat': bycat, 'fp32_only_gains': fp32_only,
                                'bf16_only_gains': bf16_only,
                                'lost_bf16': {s: r['cat'] for s, r in blk['lost_rows'].items()},
                                'lost_fp32': {s: r['cat'] for s, r in
                                              c3[f'opallo_fp32|{pol}']['lost_rows'].items()}}
    print(f'\n[{pol}] BF16 gained by category:', {k: len(v) for k, v in bycat.items()})
    for k, v in bycat.items():
        print('   ', k, v)
    print('    fp32-only gains', fp32_only, 'bf16-only gains', bf16_only)
    print('    lost bf16', res[f'gained_ids|{pol}']['lost_bf16'])
    print('    lost fp32', res[f'gained_ids|{pol}']['lost_fp32'])
    if pol == 'permissive':
        ex = {k: [row_line(s) for s in v] for k, v in bycat.items()}
        res['examples_gained_permissive'] = ex
        # the reverse direction: Supa had the value in prose, Opallo stated a WRONG terse value
        fv = [s for s, r in c3['supa_wrong|permissive']['format_rows'].items()
              if r['opallo_bf16_cat'] == 'VALUE']
        ff = [s for s, r in c3['supa_wrong|permissive']['format_rows'].items()
              if r['opallo_bf16_cat'] == 'FORMAT']
        res['examples_supa_format_to_opallo_value'] = [row_line(s) for s in fv]
        res['examples_supa_format_to_opallo_format'] = [row_line(s) for s in ff]
        res['examples_lost_permissive_bf16'] = [row_line(s) for s in blk['lost_rows']]
        # VALUE gains flagged by a sensitivity (sub-run / carry)
        sens_rows = [s for s, r in blk['gained_rows'].items()
                     if r['cat'] == 'VALUE' and (r['subrun'] or r['carry'])]
        res['examples_value_gains_with_sensitivity_flag'] = [
            {**row_line(s), 'subrun': blk['gained_rows'][s]['subrun'],
             'carry': blk['gained_rows'][s]['carry']} for s in sens_rows]
    if pol == 'strict':
        res['examples_strict_format_losses_bf16'] = [
            row_line(s, 'strict') for s, r in blk['lost_rows'].items() if r['cat'] == 'FORMAT']

(OUT / 'step4_detail.json').write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                       encoding='utf-8')
for k in ('examples_gained_permissive',):
    for cat, rows in res[k].items():
        print(f'\n--- gained {cat} ({len(rows)}) ---')
        for r in rows:
            print(f"  {r['row']} {r['class'][:4]} {r['target']}")
            print(f"      prompt: ...{r['prompt_tail']}")
            print(f"      SUPA : {r['supa']!r} term={r['supa_terminated']} [{r['supa_reason']}]")
            print(f"      OPALLO bf16: {r['opallo_bf16']!r}  fp32: {r['opallo_fp32']!r}")
for k in ('examples_supa_format_to_opallo_value', 'examples_supa_format_to_opallo_format',
          'examples_lost_permissive_bf16', 'examples_value_gains_with_sensitivity_flag',
          'examples_strict_format_losses_bf16'):
    print(f'\n--- {k} ({len(res[k])}) ---')
    for r in res[k]:
        print(f"  {r['row']} {r['class'][:4]} {r['target']}")
        print(f"      prompt: ...{r['prompt_tail']}")
        print(f"      SUPA : {r['supa']!r} [{r['supa_reason']}]")
        print(f"      OPALLO bf16: {r['opallo_bf16']!r}  fp32: {r['opallo_fp32']!r}")
