"""Negative control restricted to MULTI-VALUE texts: the only texts where a chance hit is possible.

For every Supa text that states >= 2 values of its own kind family, pair it with EVERY other
same-class, same-kind row's target whose value differs from the row's own target, and count
how often the LOOSE extractor fires. This is an all-pairs derangement, not a single shift, so
it measures the per-comparison false-positive probability on exactly the verbose population
where FORMAT calls are made.
"""
from extract import presence, want_of, NUMERIC_KINDS


def deranged_fp(R, ids, cls_of, targets, arms, correct, dscale):
    out = {}
    for pol in R.BARE_POLICIES:
        T = targets[pol]
        for arm in ('supa',):
            wrong = [i for i in ids if i not in correct[(arm, pol)]]
            verbose = []
            for i in wrong:
                p = presence(arms[arm][i]['text'], T[i], R, display_scale=dscale.get(i))
                if p['n_values'] >= 2:
                    verbose.append(i)
            pairs = fires = 0
            fire_rows = set()
            for i in verbose:
                own = want_of(T[i], R)
                for j in ids:
                    if j == i or cls_of[j] != cls_of[i] or T[j]['kind'] != T[i]['kind']:
                        continue
                    if want_of(T[j], R) == own:
                        continue
                    pairs += 1
                    if presence(arms[arm][i]['text'], T[j], R, display_scale=dscale.get(j))['loose']:
                        fires += 1
                        fire_rows.add(i.split('__')[1][:4] + '_' + i.split('__')[-1])
            out[f'{arm}|{pol}'] = {'verbose_wrong_texts': len(verbose), 'pairs': pairs,
                                   'loose_fires': fires,
                                   'fp_rate': round(fires / pairs, 4) if pairs else None,
                                   'texts_with_any_fire': sorted(fire_rows)}
    return out
