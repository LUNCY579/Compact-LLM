"""Gained/lost sets vs Supa, and gained-set overlap with Opallo, for every admitted arm.

Nulls:
  HG   hypergeometric: B's gained set is an independent uniform draw from Supa's wrong rows
  HGs  class-stratified hypergeometric (draws within each reply_class), exact convolution
Baselines (the real comparator for "donor-specific"):
  within-recipe seed pairs (Opallo s17/23/29/41; Titan s17/s23/s29) vs cross-recipe pairs.
Controls:
  POS  same weight measured twice -> identical gained sets; FP32 vs BF16 Opallo
  NEG  (a) scorer against deranged targets fires ~never; (b) overlap against random
       equal-size subsets of Supa's wrong rows reproduces the HG expectation and ~5% p<0.05
"""
import itertools
import json
import random
import sys
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
import arms as A  # noqa: E402

POLS = C.POLS
ARMS = A.ARMS
assert all(a['collision_ok'] for a in ARMS.values()), [n for n, a in ARMS.items() if not a['collision_ok']]
CORRECT = {n: {p: C.score(a['preds'], p) for p in POLS} for n, a in ARMS.items()}
WRONG = {p: set(C.IDS) - C.SUPA_REF[p] for p in POLS}
CLASSES = sorted(set(C.CLASS.values()))


def gained(n, p):
    return CORRECT[n][p] - C.SUPA_REF[p]


def lost(n, p):
    return C.SUPA_REF[p] - CORRECT[n][p]


def hg_pmf(N, K, n):
    tot = comb(N, n)
    return [comb(K, k) * comb(N - K, n - k) / tot if 0 <= n - k <= N - K else 0.0 for k in range(0, min(K, n) + 1)]


def hg_upper(N, K, n, k):
    pmf = hg_pmf(N, K, n)
    return sum(pmf[k:]) if k < len(pmf) else 0.0


def strat_pmf(sets_a, sets_b, universe):
    """Exact pmf of |A∩B| when, within each class, B is a uniform draw of its size from the class universe."""
    dist = [1.0]
    for c in CLASSES:
        U = {i for i in universe if C.CLASS[i] == c}
        pm = hg_pmf(len(U), len(sets_b & U), len(sets_a & U))
        new = [0.0] * (len(dist) + len(pm) - 1)
        for i, x in enumerate(dist):
            for j, y in enumerate(pm):
                new[i + j] += x * y
        dist = new
    return dist


def overlap_stats(ga, gb, universe):
    N, K, n = len(universe), len(gb), len(ga)
    k = len(ga & gb)
    u = len(ga | gb)
    sp = strat_pmf(ga, gb, universe)
    return {'n_a': n, 'n_b': K, 'inter': k, 'union': u, 'jaccard': round(k / u, 4) if u else None,
            'overlap_frac_of_a': round(k / n, 4) if n else None,
            'hg_expected': round(n * K / N, 2), 'hg_p_upper': hg_upper(N, K, n, k),
            'strat_expected': round(sum(i * x for i, x in enumerate(sp)), 2), 'strat_p_upper': sum(sp[k:]),
            'expected_jaccard_null': round((n * K / N) / (n + K - n * K / N), 4) if (n + K) else None}


OPALLO = {'fp32': 'rab_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95', 'bf16': 'opallo_bf16'}
OPALLO_SEEDS = {'s17': OPALLO['fp32'], 's23': 'rab_B28last20_lr1e-05_b5200_seed23_wsd0.2-0.2_b20.95',
                's29': 'rab_B28last20_lr1e-05_b5200_seed29_wsd0.2-0.2_b20.95', 's41': 'rab_B28last20_lr1e-05_b5200_seed41_wsd0.2-0.2_b20.95'}
TITAN_SEEDS = {'s17': 'int001_a100', 's23': 'int001_rep_s23_ctrl', 's29': 'int001_rep_s29_ctrl'}
D1_SEEDS = {'s17': 'panel001_d1_s17', 's23': 'panel001_d1_s23', 's29': 'panel001_d1_s29'}
for d in (OPALLO, OPALLO_SEEDS, TITAN_SEEDS, D1_SEEDS):
    assert all(v in ARMS for v in d.values()), d

out = {'policies': {}}
for p in POLS:
    W = WRONG[p]
    res = {'supa_correct_n': len(C.SUPA_REF[p]), 'supa_wrong_n': len(W),
           'opallo_fp32_gained': sorted(gained(OPALLO['fp32'], p)), 'opallo_bf16_gained': sorted(gained(OPALLO['bf16'], p)),
           'opallo_fp32_lost': sorted(lost(OPALLO['fp32'], p)), 'opallo_bf16_lost': sorted(lost(OPALLO['bf16'], p))}
    # --- every arm vs Opallo
    table = []
    for n in sorted(ARMS, key=lambda n: (ARMS[n]['family'], n)):
        row = {'arm': n, 'family': ARMS[n]['family'], 'weight8': ARMS[n]['weight8'], 'correct': len(CORRECT[n][p]),
               'gained': len(gained(n, p)), 'lost': len(lost(n, p)),
               'gained_by_class': {c: len({i for i in gained(n, p) if C.CLASS[i] == c}) for c in CLASSES}}
        for tag, on in OPALLO.items():
            row['vs_opallo_' + tag] = overlap_stats(gained(n, p), gained(on, p), W)
            row['vs_opallo_' + tag]['lost_inter'] = len(lost(n, p) & lost(on, p))
            cs = overlap_stats(CORRECT[n][p], CORRECT[on][p], set(C.IDS))
            row['correct_vs_opallo_' + tag] = {k: cs[k] for k in ('n_a', 'n_b', 'inter', 'jaccard', 'hg_expected', 'hg_p_upper')}
        table.append(row)
    res['arms'] = table

    # --- seed-pair baselines
    def pairs(da, db, same):
        it = itertools.combinations(sorted(da), 2) if same else itertools.product(sorted(da), sorted(db))
        rows = []
        for x, y in it:
            na, nb = da[x], (da if same else db)[y]
            s = overlap_stats(gained(na, p), gained(nb, p), W)
            rows.append({'a': na, 'b': nb, **{k: s[k] for k in ('n_a', 'n_b', 'inter', 'jaccard', 'hg_expected', 'hg_p_upper', 'strat_p_upper')}})
        return rows
    base = {'opallo_x_opallo': pairs(OPALLO_SEEDS, None, True), 'titan_x_titan': pairs(TITAN_SEEDS, None, True),
            'd1_x_d1': pairs(D1_SEEDS, None, True),
            'titan_x_opallo': pairs(TITAN_SEEDS, OPALLO_SEEDS, False), 'd1_x_opallo': pairs(D1_SEEDS, OPALLO_SEEDS, False),
            'titan_x_d1': pairs(TITAN_SEEDS, D1_SEEDS, False)}
    res['seed_pair_baselines'] = base
    res['seed_pair_summary'] = {k: {'pairs': len(v), 'mean_jaccard': round(sum(r['jaccard'] for r in v) / len(v), 4),
                                    'min_jaccard': min(r['jaccard'] for r in v), 'max_jaccard': max(r['jaccard'] for r in v),
                                    'mean_inter': round(sum(r['inter'] for r in v) / len(v), 2),
                                    'mean_hg_expected': round(sum(r['hg_expected'] for r in v) / len(v), 2)}
                                for k, v in base.items()}

    # --- row-level: how many of the 4 Opallo seeds and 3 Titan draws gain each Supa-wrong row
    fo = {i: sum(i in gained(v, p) for v in OPALLO_SEEDS.values()) for i in W}
    ft = {i: sum(i in gained(v, p) for v in TITAN_SEEDS.values()) for i in W}
    fd = {i: sum(i in gained(v, p) for v in D1_SEEDS.values()) for i in W}
    grid = {f'O{a}_T{b}': sum(1 for i in W if fo[i] == a and ft[i] == b) for a in range(5) for b in range(4)}
    res['row_frequency_grid_opallo4_x_titan3'] = grid
    res['rows_all4_opallo_and_0_titan'] = sorted(i for i in W if fo[i] == 4 and ft[i] == 0)
    res['rows_all3_titan_and_0_opallo'] = sorted(i for i in W if ft[i] == 3 and fo[i] == 0)
    res['rows_gained_by_all_7'] = sorted(i for i in W if fo[i] == 4 and ft[i] == 3)
    res['rows_gained_by_none_of_7'] = len([i for i in W if fo[i] == 0 and ft[i] == 0])

    def spearman(x, y):
        ids = sorted(W)
        def rank(v):
            order = sorted(ids, key=lambda i: v[i])
            r, j = {}, 0
            while j < len(order):
                k = j
                while k + 1 < len(order) and v[order[k + 1]] == v[order[j]]:
                    k += 1
                for m in range(j, k + 1):
                    r[order[m]] = (j + k) / 2 + 1
                j = k + 1
            return r
        rx, ry = rank(x), rank(y)
        mx, my = sum(rx.values()) / len(ids), sum(ry.values()) / len(ids)
        num = sum((rx[i] - mx) * (ry[i] - my) for i in ids)
        den = (sum((rx[i] - mx) ** 2 for i in ids) * sum((ry[i] - my) ** 2 for i in ids)) ** 0.5
        return num / den
    # split-half within Opallo (2 seeds vs 2 seeds, all 3 splits) as the ceiling for a cross-family correlation
    seeds = sorted(OPALLO_SEEDS)
    splits = []
    for a in itertools.combinations(seeds, 2):
        b = tuple(s for s in seeds if s not in a)
        if a > b:
            continue
        xa = {i: sum(i in gained(OPALLO_SEEDS[s], p) for s in a) for i in W}
        xb = {i: sum(i in gained(OPALLO_SEEDS[s], p) for s in b) for i in W}
        splits.append({'a': a, 'b': b, 'spearman': round(spearman(xa, xb), 4)})
    res['row_spearman'] = {'opallo4_vs_titan3': round(spearman(fo, ft), 4), 'opallo4_vs_d1_3': round(spearman(fo, fd), 4),
                           'titan3_vs_d1_3': round(spearman(ft, fd), 4), 'opallo_split_half_2v2': splits}

    # --- permutation null for the row-frequency Spearman: permute Titan frequencies within class
    rng = random.Random(20260923)
    obs = spearman(fo, ft)
    ge = 0
    NP = 2000
    byc = {c: sorted(i for i in W if C.CLASS[i] == c) for c in CLASSES}
    for _ in range(NP):
        perm = {}
        for c, ids in byc.items():
            vals = [ft[i] for i in ids]
            rng.shuffle(vals)
            perm.update(dict(zip(ids, vals)))
        ge += spearman(fo, perm) >= obs
    res['row_spearman_perm_within_class'] = {'observed': round(obs, 4), 'n_perm': NP, 'p_ge': (ge + 1) / (NP + 1)}
    out['policies'][p] = res

# --- POSITIVE controls: same weight measured twice; FP32 vs BF16
pos = {}
for a_, b_ in (('int001_a100', 'panel001_gate_a100'), ('int001_lora596', 'panel001_lora_r16'),
               ('int001_lora596b', 'panel001_lora_r16_lr1e4'), ('doseint001_dual899_a100_retention', 'panel001_dose_a100'),
               ('doseint001_dual899_a025_retention', 'panel001_dose_a025'), ('int001_s25', 'int001_s25d'),
               (OPALLO['fp32'], OPALLO['bf16'])):
    same_text = sum(ARMS[a_]['preds'][i]['text'] == ARMS[b_]['preds'][i]['text'] for i in C.IDS)
    pos[f'{a_} | {b_}'] = {'weights': (ARMS[a_]['weight8'], ARMS[b_]['weight8']), 'identical_text_rows': same_text,
                           **{p: overlap_stats(gained(a_, p), gained(b_, p), WRONG[p]) for p in POLS}}
out['positive_controls'] = pos

# --- NEGATIVE control (a): scorer against deranged targets must essentially never fire
rng = random.Random(17)
neg_a = {}
for n in ('int001_a100', 'rrp015_control', 'int001_s25d', OPALLO['fp32'], OPALLO['bf16']):
    res = {}
    for p in POLS:
        within, across = [], []
        for _ in range(20):
            # derangement within class
            perm = {}
            for c in CLASSES:
                ids = [i for i in C.IDS if C.CLASS[i] == c]
                while True:
                    sh = ids[:]
                    rng.shuffle(sh)
                    if all(x != y for x, y in zip(ids, sh)):
                        break
                perm.update(dict(zip(ids, sh)))
            within.append(sum(C.R.score_prediction(ARMS[n]['preds'][i]['text'], ARMS[n]['preds'][i]['terminated'],
                                                   C.TARGETS[p][perm[i]])['correct'] for i in C.IDS))
            ids = C.IDS[:]
            while True:
                sh = ids[:]
                rng.shuffle(sh)
                if all(x != y for x, y in zip(ids, sh)):
                    break
            across.append(sum(C.R.score_prediction(ARMS[n]['preds'][i]['text'], ARMS[n]['preds'][i]['terminated'],
                                                   C.TARGETS[p][j])['correct'] for i, j in zip(ids, sh)))
        res[p] = {'true_targets': len(CORRECT[n][p]), 'deranged_within_class_mean': sum(within) / 20, 'max': max(within),
                  'deranged_across_mean': sum(across) / 20, 'across_max': max(across)}
    neg_a[n] = res
out['negative_control_scorer_deranged_targets'] = neg_a

# --- NEGATIVE control (b): overlap vs random equal-size subsets of Supa-wrong rows
rng = random.Random(23)
neg_b = {}
for n in ('int001_a100', 'rrp015_control', 'int001_s25d'):
    res = {}
    for p in POLS:
        W = sorted(WRONG[p])
        go = gained(OPALLO['fp32'], p)
        ga = gained(n, p)
        inters, small_p = [], 0
        R = 5000
        for _ in range(R):
            fake = set(rng.sample(W, len(go)))
            k = len(ga & fake)
            inters.append(k)
            small_p += hg_upper(len(W), len(fake), len(ga), k) <= 0.05
        res[p] = {'draws': R, 'mean_inter': round(sum(inters) / R, 3), 'hg_expected': round(len(ga) * len(go) / len(W), 3),
                  'frac_p_le_0.05': round(small_p / R, 4), 'max_inter': max(inters), 'observed_inter_real_opallo': len(ga & go)}
    neg_b[n] = res
out['negative_control_random_subsets'] = neg_b

(HERE / 'overlap.json').write_text(json.dumps(out, indent=1, default=list), encoding='utf-8')
print('written overlap.json')
