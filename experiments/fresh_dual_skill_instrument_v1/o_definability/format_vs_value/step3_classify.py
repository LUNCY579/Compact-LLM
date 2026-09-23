"""Step 3-5: classify gained / lost rows and every Supa-wrong row as FORMAT / VALUE / NONTERM."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all, OUT, CLASSES, sk  # noqa: E402
from extract import presence  # noqa: E402

R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
scored, correct = score_all(R, arms, targets)
aud = {r['id']: r for r in audit['rows']}
ids = [r['id'] for r in probe]
cls_of = {r['id']: r['reply_class'] for r in probe}
UNIT_SCALE = {'minute': 60, 'hour': 3600, 'second': 1}
dscale = {i: UNIT_SCALE.get(aud[i]['proposed_typed_target'].get('unit'))
          for i in ids if aud[i]['proposed_typed_target']['type'] == 'duration'}
OPALLO = ('opallo_bf16', 'opallo_fp32')
short = sk


def shape(text, reason):
    """Surface description of WHY the scorer rejected a value-present text (descriptive only)."""
    b = R.normalise(text)
    if reason and 'bare' in reason:
        return 'unit_omitted_bare_value'
    if re.search(r'[=+]', b) or re.search(r'\d\s+-\s+\d', b):
        return 'shown_working_equation'
    if re.search(r'\d(:\d\d)?\s*(am|pm)?\s+(to|until)\s+\d', b):
        return 'range_restated'
    if re.match(r'[a-z]', b):
        return 'prose_sentence_prefix'
    return 'trailing_words_or_unit'


def classify(arm, pol, i):
    s = scored[(arm, pol)][i]
    rec = {'text': s['raw'], 'terminated': s['terminated'], 'reason': s['reason'],
           'correct': s['correct']}
    p = presence(s['raw'], targets[pol][i], R, display_scale=dscale.get(i))
    rec.update({f'vp_{k}': v for k, v in p.items()})
    prompt_p = presence(aud[i]['operative_prompt'], targets[pol][i], R,
                        display_scale=dscale.get(i))
    rec['target_in_prompt'] = prompt_p['loose']
    if s['correct']:
        rec['cat'] = 'CORRECT'
    elif not s['terminated']:
        rec['cat'] = 'NONTERM'
    elif p['loose']:
        rec['cat'] = 'FORMAT'
        rec['shape'] = shape(s['raw'], s['reason'])
    else:
        rec['cat'] = 'VALUE'
    return rec


C = {(arm, pol): {i: classify(arm, pol, i) for i in ids} for arm in arms for pol in R.BARE_POLICIES}
res = {'hashes': hashes}


def tally(rows, key, arm, pol):
    out = {}
    for c in CLASSES + ('ALL',):
        rr = [i for i in rows if c == 'ALL' or cls_of[i] == c]
        cnt = Counter(C[(arm, pol)][i]['cat'] for i in rr)
        nt_vp = sum(1 for i in rr if C[(arm, pol)][i]['cat'] == 'NONTERM'
                    and C[(arm, pol)][i]['vp_loose'])
        out[c] = {'n': len(rr), 'FORMAT': cnt['FORMAT'], 'VALUE': cnt['VALUE'],
                  'NONTERM': cnt['NONTERM'], 'NONTERM_value_present': nt_vp}
    return out


# ---- Q3: gained rows (Supa's text) and lost rows (Opallo's text) --------------------------
for opal in OPALLO:
    for pol in R.BARE_POLICIES:
        S, O = correct[('supa', pol)], correct[(opal, pol)]
        gained, lost = sorted(O - S, key=ids.index), sorted(S - O, key=ids.index)
        g = tally(gained, 'gained', 'supa', pol)
        l = tally(lost, 'lost', opal, pol)
        G, L = g['ALL'], l['ALL']
        net = len(gained) - len(lost)
        fmt_net = G['FORMAT'] - L['FORMAT']
        # sensitivities on the gained rows' Supa texts
        sens = Counter()
        for i in gained:
            r = C[('supa', pol)][i]
            if r['cat'] == 'VALUE':
                for k in ('vp_carry', 'vp_subrun', 'vp_bare_display'):
                    sens[k] += bool(r[k])
            if r['cat'] == 'FORMAT':
                sens['format_typed'] += r['vp_typed']
                sens['format_final'] += r['vp_final']
                sens['format_target_in_prompt'] += r['target_in_prompt']
                sens[f"shape:{r['shape']}"] += 1
        # variant headlines: FORMAT under TYPED / FINAL instead of LOOSE
        var = {}
        for v in ('vp_loose', 'vp_typed', 'vp_final'):
            gf = sum(1 for i in gained if C[('supa', pol)][i]['terminated'] and C[('supa', pol)][i][v])
            lf = sum(1 for i in lost if C[(opal, pol)][i]['terminated'] and C[(opal, pol)][i][v])
            var[v] = {'format_gains': gf, 'format_losses': lf, 'net_format': gf - lf,
                      'frac_of_net': round((gf - lf) / net, 4) if net else None}
        rows_g = {short(i): {'class': cls_of[i], 'cat': C[('supa', pol)][i]['cat'],
                             'supa': C[('supa', pol)][i]['text'],
                             'opallo': C[(opal, pol)][i]['text'],
                             'supa_values': C[('supa', pol)][i]['vp_values'],
                             'shape': C[('supa', pol)][i].get('shape'),
                             'typed': C[('supa', pol)][i]['vp_typed'],
                             'final': C[('supa', pol)][i]['vp_final'],
                             'target_in_prompt': C[('supa', pol)][i]['target_in_prompt'],
                             'carry': C[('supa', pol)][i]['vp_carry'],
                             'subrun': C[('supa', pol)][i]['vp_subrun']} for i in gained}
        rows_l = {short(i): {'class': cls_of[i], 'cat': C[(opal, pol)][i]['cat'],
                             'supa': C[('supa', pol)][i]['text'],
                             'opallo': C[(opal, pol)][i]['text']} for i in lost}
        head = {
            'gained': len(gained), 'lost': len(lost), 'net': net,
            'format_gains': G['FORMAT'], 'value_gains': G['VALUE'], 'nonterm_gains': G['NONTERM'],
            'nonterm_gains_value_present': G['NONTERM_value_present'],
            'format_losses': L['FORMAT'], 'value_losses': L['VALUE'], 'nonterm_losses': L['NONTERM'],
            'net_format': fmt_net, 'net_value': G['VALUE'] - L['VALUE'],
            'net_nonterm': G['NONTERM'] - L['NONTERM'],
            'frac_net_format_lower(nonterm->value)': round(fmt_net / net, 4) if net else None,
            'frac_net_format_upper(nonterm->format)':
                round((fmt_net + G['NONTERM'] - L['NONTERM']) / net, 4) if net else None,
            'frac_gross_gains_format': round(G['FORMAT'] / len(gained), 4) if gained else None,
        }
        res[f'{opal}|{pol}'] = {'headline': head, 'gained_by_class': g, 'lost_by_class': l,
                                'sensitivity': dict(sens), 'variants': var,
                                'gained_rows': rows_g, 'lost_rows': rows_l}
        print(f'\n=== {opal} vs supa [{pol}] ===')
        print('  headline', head)
        for c in CLASSES:
            print(f'  gained {c:22s}', g[c], '| lost', l[c])
        print('  sensitivity', dict(sens))
        print('  variants', var)

# ---- Q4: every Supa-wrong row ---------------------------------------------------------------
for pol in R.BARE_POLICIES:
    wrong = [i for i in ids if i not in correct[('supa', pol)]]
    t = tally(wrong, 'wrong', 'supa', pol)
    extra = Counter()
    for i in wrong:
        r = C[('supa', pol)][i]
        if r['cat'] == 'FORMAT':
            extra['typed'] += r['vp_typed']; extra['final'] += r['vp_final']
            extra['target_in_prompt'] += r['target_in_prompt']
            extra[f"shape:{r['shape']}"] += 1
        if r['cat'] == 'VALUE':
            extra['value_carry'] += r['vp_carry']; extra['value_subrun'] += r['vp_subrun']
            extra['value_bare_display'] += r['vp_bare_display']
    fate = {}
    for opal in OPALLO:
        fate[opal] = dict(Counter(f"{C[('supa', pol)][i]['cat']}->{C[(opal, pol)][i]['cat']}"
                                  for i in wrong))
    fmt_rows = [i for i in wrong if C[('supa', pol)][i]['cat'] == 'FORMAT']
    res[f'supa_wrong|{pol}'] = {
        'n_wrong': len(wrong), 'by_class': t, 'format_detail': dict(extra),
        'transition_to_opallo': fate,
        'format_rows': {short(i): {'class': cls_of[i], 'supa': C[('supa', pol)][i]['text'],
                                   'shape': C[('supa', pol)][i]['shape'],
                                   'opallo_bf16': C[('opallo_bf16', pol)][i]['text'],
                                   'opallo_bf16_cat': C[('opallo_bf16', pol)][i]['cat'],
                                   'opallo_fp32_cat': C[('opallo_fp32', pol)][i]['cat']}
                        for i in fmt_rows},
        'nonterm_rows': {short(i): C[('supa', pol)][i]['text'] for i in wrong
                         if C[('supa', pol)][i]['cat'] == 'NONTERM'}}
    print(f'\n=== all Supa-wrong rows [{pol}] n={len(wrong)} ===')
    for c in CLASSES + ('ALL',):
        print(f'  {c:22s}', t[c])
    print('  format detail', dict(extra))
    for opal in OPALLO:
        print('  transitions', opal, fate[opal])

# ---- whole-panel transition matrix (permissive and strict) ------------------------------------
for pol in R.BARE_POLICIES:
    for opal in OPALLO:
        m = Counter((C[('supa', pol)][i]['cat'], C[(opal, pol)][i]['cat']) for i in ids)
        res[f'matrix|{opal}|{pol}'] = {f'{a}->{b}': n for (a, b), n in sorted(m.items())}
        print(f'matrix {opal} {pol}:', res[f'matrix|{opal}|{pol}'])
        # Opallo's own wrong rows: how many are value-present?
        ow = Counter(C[(opal, pol)][i]['cat'] for i in ids)
        print(f'   {opal} {pol} category totals:', dict(ow))
        res[f'totals|{opal}|{pol}'] = dict(ow)
    res[f'totals|supa|{pol}'] = dict(Counter(C[('supa', pol)][i]['cat'] for i in ids))
    print(f'   supa {pol} category totals:', res[f'totals|supa|{pol}'])

# ---- verbose-text negative control (the population where a chance hit is possible) -----------
from step2_helpers import deranged_fp  # noqa: E402
res['verbose_negative_control'] = deranged_fp(R, ids, cls_of, targets, arms, correct, dscale)
print('\nverbose negative control', res['verbose_negative_control'])
(OUT / 'step3_classify.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
