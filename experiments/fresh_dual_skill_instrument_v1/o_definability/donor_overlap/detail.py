"""Row-id detail for the key arms, lost-set overlap, the samples-only view of Titan-Sparse, and the texts
of the shared-core and donor-specific rows (read the text, not just the structure)."""
import json
import sys
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
import arms as A  # noqa: E402

ARMS = A.ARMS
OP = 'rab_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95'
OPS = [OP, 'rab_B28last20_lr1e-05_b5200_seed23_wsd0.2-0.2_b20.95', 'rab_B28last20_lr1e-05_b5200_seed29_wsd0.2-0.2_b20.95',
       'rab_B28last20_lr1e-05_b5200_seed41_wsd0.2-0.2_b20.95']
TIS = ['int001_a100', 'int001_rep_s23_ctrl', 'int001_rep_s29_ctrl']
KEY = ['int001_a100', 'rrp015_control', 'int001_s25d', 'opallo_bf16', OP]
short = lambda i: i.replace('rineya_v3_sft__', '').replace('reas_', '')


def hg_upper(N, K, n, k):
    tot = comb(N, n)
    return sum(comb(K, j) * comb(N - K, n - j) for j in range(k, min(K, n) + 1)) / tot


out = {}
for p in C.POLS:
    cor = {n: C.score(ARMS[n]['preds'], p) for n in set(KEY + OPS + TIS)}
    S = C.SUPA_REF[p]
    G = {n: cor[n] - S for n in cor}
    L = {n: S - cor[n] for n in cor}
    r = {'per_arm': {}}
    for n in KEY:
        r['per_arm'][n] = {'gained': sorted(map(short, G[n])), 'lost': sorted(map(short, L[n])),
                           'gained_and_opallo_fp32_gained': sorted(map(short, G[n] & G[OP])),
                           'gained_not_opallo_fp32': sorted(map(short, G[n] - G[OP])),
                           'opallo_fp32_gained_not_arm': sorted(map(short, G[OP] - G[n])),
                           'lost_inter_opallo_fp32': sorted(map(short, L[n] & L[OP])),
                           'lost_hg_p_upper_within_supa_correct': hg_upper(len(S), len(L[OP]), len(L[n]), len(L[n] & L[OP]))}
    fo = {i: sum(i in G[o] for o in OPS) for i in set(C.IDS) - S}
    ft = {i: sum(i in G[t] for t in TIS) for i in set(C.IDS) - S}
    core7 = sorted(i for i in fo if fo[i] == 4 and ft[i] == 3)
    ospec = sorted(i for i in fo if fo[i] >= 3 and ft[i] == 0)
    tspec = sorted(i for i in fo if ft[i] >= 2 and fo[i] == 0)
    r['shared_core_all7'] = sorted(map(short, core7))
    r['opallo_ge3of4_titan_0of3'] = sorted(map(short, ospec))
    r['titan_ge2of3_opallo_0of4'] = sorted(map(short, tspec))
    r['counts'] = {'opallo_ge3of4': sum(1 for i in fo if fo[i] >= 3), 'of_which_titan_ge2of3': sum(1 for i in fo if fo[i] >= 3 and ft[i] >= 2),
                   'of_which_titan_0': len(ospec), 'titan_ge2of3': sum(1 for i in ft if ft[i] >= 2),
                   'of_which_opallo_ge3': sum(1 for i in ft if ft[i] >= 2 and fo[i] >= 3), 'of_which_opallo_0': len(tspec)}
    if p == 'permissive':
        tx = {}
        for tag, ids in (('core7', core7), ('opallo_specific', ospec), ('titan_specific', tspec)):
            tx[tag] = [{'id': short(i), 'target': C.TARGETS[p][i], 'supa': C.SUPA_PREDS[i]['text'],
                        'opallo_s17': ARMS[OP]['preds'][i]['text'], 'titan_s17': ARMS['int001_a100']['preds'][i]['text']} for i in ids]
        r['texts'] = tx
    out[p] = r

# ---- Titan-Sparse from its RETENTION_PROBE alone (what the task feared was all we had)
rp = json.loads(Path('C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/sparse001-20260917/evidence/RETENTION_PROBE_int001_s25d.json').read_text(encoding='utf-8'))
sv = {}
for p in C.POLS:
    b = rp['by_bare_policy'][p]
    lost = sorted(i for c in b['per_class'].values() for i in c['lost_ids'])
    samples = rp['samples']
    tg = C.TARGETS[p]
    s_correct = {s['id'] for s in samples if C.R.score_prediction(s['arm'], s['arm_terminated'], tg[s['id']])['correct']}
    s_ids = {s['id'] for s in samples}
    gained_visible = s_correct - C.SUPA_REF[p]
    Gop = C.score(ARMS[OP]['preds'], p) - C.SUPA_REF[p]
    sv[p] = {'arm_correct_total': b['arm_correct_total'], 'arm_correct_outside_source': b['arm_correct_outside_source'],
             'lost_ids': sorted(map(short, lost)), 'lost_inter_opallo_fp32': sorted(map(short, set(lost) & (C.SUPA_REF[p] - C.score(ARMS[OP]['preds'], p)))),
             'per_class_gained_from_counts': {c: v['reference_n'] - v['source_correct_terminated'] for c, v in b['per_class'].items()},
             'samples_n': len(samples), 'samples_supa_wrong': len(s_ids - C.SUPA_REF[p]),
             'gained_rows_visible_in_samples': len(gained_visible),
             'visible_gains_also_opallo_gained': len(gained_visible & Gop),
             'opallo_gains_among_supa_wrong_samples': len((s_ids - C.SUPA_REF[p]) & Gop)}
out['titan_sparse_from_probe_file_only'] = sv
(HERE / 'detail.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({p: {k: v for k, v in out[p].items() if k not in ('per_arm', 'texts')} for p in C.POLS}, indent=1))
print(json.dumps(out['titan_sparse_from_probe_file_only'], indent=1))
for n in KEY:
    x = out['permissive']['per_arm'][n]
    print(n, 'lost', x['lost'], 'lost∩op', len(x['lost_inter_opallo_fp32']), 'p', round(x['lost_hg_p_upper_within_supa_correct'], 4))
