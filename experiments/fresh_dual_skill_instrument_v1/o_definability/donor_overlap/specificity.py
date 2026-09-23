"""Is the Supa->donor gain donor-SPECIFIC? Exact label-permutation tests over independent training draws.

Arms are independent training draws only (no transforms of one delta):
  OPALLO recipe: seeds 17, 23, 29, 41   (general-reasoning SFT, last-20 blocks, WSD)
  TITAN recipe : s17 (int001_a100 = FP32 dense RRP-015 control), rep s23, rep s29 (transition rows)
  D1           : panel001_d1 s17, s23, s29 (transition rows, dose rung 1)
Statistic 1 (pairwise): mean within-family Jaccard of gained sets minus mean cross-family Jaccard.
Statistic 2 (row-level): sum over Supa-wrong rows of (gain-rate in family A - gain-rate in family B)^2.
Null: family labels exchangeable across the draws; enumerate EVERY relabelling with the same sizes.
Positive control for the test: split ONE family's draws into two pseudo-families -> must NOT reject
(it can't be specific to itself); and a planted-specific control: move rows so a family is specific.
"""
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
import arms as A  # noqa: E402

ARMS = A.ARMS
POLS = C.POLS
OP = ['rab_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95', 'rab_B28last20_lr1e-05_b5200_seed23_wsd0.2-0.2_b20.95',
      'rab_B28last20_lr1e-05_b5200_seed29_wsd0.2-0.2_b20.95', 'rab_B28last20_lr1e-05_b5200_seed41_wsd0.2-0.2_b20.95']
TI = ['int001_a100', 'int001_rep_s23_ctrl', 'int001_rep_s29_ctrl']
D1 = ['panel001_d1_s17', 'panel001_d1_s23', 'panel001_d1_s29']


def gsets(names, p):
    return [frozenset(C.score(ARMS[n]['preds'], p) - C.SUPA_REF[p]) for n in names]


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def stat_pair(sets, lab):
    w, c = [], []
    for i, j in itertools.combinations(range(len(sets)), 2):
        (w if lab[i] == lab[j] else c).append(jac(sets[i], sets[j]))
    return sum(w) / len(w) - sum(c) / len(c)


def stat_row(sets, lab, universe):
    s = 0.0
    fa = [k for k in range(len(sets)) if lab[k] == 0]
    fb = [k for k in range(len(sets)) if lab[k] == 1]
    for i in universe:
        ra = sum(i in sets[k] for k in fa) / len(fa)
        rb = sum(i in sets[k] for k in fb) / len(fb)
        s += (ra - rb) ** 2
    return s


def perm_test(sets, n_a, universe):
    n = len(sets)
    obs_lab = [0] * n_a + [1] * (n - n_a)
    labs = []
    for idx in itertools.combinations(range(n), n_a):
        labs.append([0 if k in idx else 1 for k in range(n)])
    op, orow = stat_pair(sets, obs_lab), stat_row(sets, obs_lab, universe)
    sp = [stat_pair(sets, l) for l in labs]
    sr = [stat_row(sets, l, universe) for l in labs]
    return {'labellings': len(labs), 'min_attainable_p': round(1 / len(labs), 4),
            'pair_stat_obs': round(op, 4), 'pair_p': sum(x >= op - 1e-12 for x in sp) / len(labs),
            'row_stat_obs': round(orow, 4), 'row_p': sum(x >= orow - 1e-12 for x in sr) / len(labs)}


out = {}
for p in POLS:
    W = sorted(set(C.IDS) - C.SUPA_REF[p])
    o, t, d = gsets(OP, p), gsets(TI, p), gsets(D1, p)
    r = {}
    r['opallo_vs_titan'] = perm_test(o + t, 4, W)
    r['opallo_vs_d1'] = perm_test(o + d, 4, W)
    r['titan_vs_d1'] = perm_test(t + d, 3, W)
    r['opallo_vs_titan_plus_d1'] = perm_test(o + t + d, 4, W)
    # CONTROL 1 (should NOT reject): the 4 Opallo seeds split 2 vs 2 -- a family cannot be specific to itself
    r['CONTROL_null_opallo_2v2'] = perm_test(o, 2, W)
    r['CONTROL_null_titan_plus_d1_3v3'] = perm_test(t + d, 3, W)
    # CONTROL 2 (should reject): plant specificity -- give the Titan draws a disjoint copy of their gains
    # by relabelling row ids into a private namespace; Opallo cannot share them by construction.
    planted = [frozenset(('PLANT_' + i) for i in s) for s in t]
    r['CONTROL_planted_specific'] = perm_test(o + planted, 4, W + ['PLANT_' + i for i in W])
    out[p] = r

(HERE / 'specificity.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
