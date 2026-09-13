"""Paired comparison across arms: is the difference real, or is it 150 rows of noise?

`arm_dose.py` runs each arm on identical rows in identical order, so every probe row is a matched
pair and the right test is McNemar's exact test on the DISCORDANT rows -- the ones where the arms
disagree. Aggregate counts are not evidence: two arms can both reach 73/150 on different rows.

    python experiments/rab001_exploration/arm_compare.py [armA armB ...]

Thresholds come from `evidence/retention-design-v1/DISTINGUISHABILITY_PREREG.md`, fixed before any
arm was trained. The decisive number there: with an exact two-sided sign test this panel cannot
certify an effect below **6 discordant rows split 6-0** (p = 0.031). A 4-0 split is p = 0.125 and
means "no effect demonstrated" -- not "a small effect".

Every gain is also split by the `HEADROOM_AUDIT.json` strata, because 24 of the 103 headroom rows
are ones where the source already asserted the right answer and the grammar rejected it. Winning
those is format compliance, not arithmetic. Only `wrong_answer`-stratum gains are credited as
capability.

Refuses to compare arms whose per-step token sequences differ: unmatched data order would make
the pairing a lie.

Pure arithmetic, no model, no torch. Writes `evidence/retention-design-v1/ARM_COMPARISON.json`.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001.source import ROOT           # noqa: E402

OUT = ROOT / 'evidence' / 'retention-design-v1'
ALPHA = Fraction(1, 20)
ARMS = sys.argv[1:] or ['B28', 'B28last8', 'H', 'P']


def mcnemar_exact(b: int, c: int) -> Fraction:
    """Two-sided exact binomial on the discordant pairs. Exact rationals, no floats."""
    n = b + c
    if n == 0:
        return Fraction(1)
    tail = sum(Fraction(comb(n, k), 2 ** n) for k in range(min(b, c) + 1))
    return min(Fraction(1), 2 * tail)


def load(arm: str) -> dict | None:
    hits = sorted(OUT.glob(f'ARM_DOSE_{arm}_lr*.json'))
    if not hits:
        return None
    body = json.loads(hits[-1].read_text(encoding='utf-8'))
    final = body['checkpoints'][-1]
    body['_per_row'] = final['retention']['per_row']
    body['_final'] = final
    body['_path'] = hits[-1]
    return body


def main() -> int:
    audit = json.loads((OUT / 'HEADROOM_AUDIT.json').read_text(encoding='utf-8'))
    strata = audit['policies']['permissive']['ids']
    of_stratum = {rid: name for name, ids in strata.items() for rid in ids}
    reference = None

    loaded = {}
    for arm in ARMS:
        body = load(arm)
        if body is None:
            print(f'  (no ARM_DOSE artifact for {arm}; skipping)')
            continue
        loaded[arm] = body
    if len(loaded) < 1:
        raise SystemExit('nothing to compare')

    orders = {a: b.get('per_step_target_tokens') for a, b in loaded.items()}
    report = {'schema': 1, 'evidence_kind': 'exploratory_paired_comparison_only',
              'note': 'McNemar exact two-sided on matched probe rows. Thresholds pre-registered '
                      'in DISTINGUISHABILITY_PREREG.md before any arm was trained. Outside the '
                      'hashed root: no receipt, no gate, no efficacy claim.',
              'alpha': '1/20', 'arms': {}, 'pairs': {}}

    print(f'{"arm":<10} {"params":>13} {"trainable":>12} {"correct":>9} {"gain":>5} '
          f'{"real":>5} {"fmt":>4} {"lost":>5} {"net":>5} {"devCE":>7}')
    for arm, body in loaded.items():
        rowmap = body['_per_row']
        ref_ids = {i for i in rowmap if i not in of_stratum}
        reference = ref_ids if reference is None else reference
        gained = [i for i, ok in rowmap.items() if ok and i in of_stratum]
        real = sum(1 for i in gained if of_stratum[i] == 'wrong_answer')
        fmt = sum(1 for i in gained if of_stratum[i] == 'format_only')
        lost = sum(1 for i in ref_ids if not rowmap[i])
        h = body['headline']
        report['arms'][arm] = {
            'path': str(body['_path']), 'parameters': body['parameters'],
            'trainable_parameters': body['trainable_parameters'],
            'correct_total': h['correct_end'], 'gained': len(gained),
            'gained_wrong_answer_stratum': real, 'gained_format_only_stratum': fmt,
            'gained_runaway_stratum': len(gained) - real - fmt,
            'lost': lost, 'net': len(gained) - lost,
            'dev_ce_start': h['dev_ce_start'], 'dev_ce_end': h['dev_ce_end'],
            'gate_status_end': h['status_end'],
            'supervised_target_tokens': body['supervised_target_tokens'],
            'gained_ids': sorted(gained),
            'lost_ids': sorted(i for i in ref_ids if not rowmap[i])}
        print(f'{arm:<10} {body["parameters"]:>13,} {body["trainable_parameters"]:>12,} '
              f'{h["correct_end"]:>6}/150 {len(gained):>5} {real:>5} {fmt:>4} {lost:>5} '
              f'{len(gained)-lost:>+5} {h["dev_ce_end"]:>7.4f}')

    names = list(loaded)
    print(f'\n{"pair":<16} {"b":>3} {"c":>3} {"n":>3} {"p":>9}  verdict')
    for i, a in enumerate(names):
        for b_arm in names[i + 1:]:
            ra, rb = loaded[a]['_per_row'], loaded[b_arm]['_per_row']
            shared = sorted(set(ra) & set(rb))
            if orders[a] and orders[b_arm] and orders[a] != orders[b_arm]:
                print(f'{a}~{b_arm}: REFUSED, per-step token sequences differ '
                      '(unmatched data order)')
                report['pairs'][f'{a}~{b_arm}'] = {'status': 'REFUSED_UNMATCHED_DATA_ORDER'}
                continue
            bb = sum(1 for r in shared if ra[r] and not rb[r])
            cc = sum(1 for r in shared if rb[r] and not ra[r])
            p = mcnemar_exact(bb, cc)
            sig = p < ALPHA
            better = a if bb > cc else (b_arm if cc > bb else 'tie')
            verdict = (f'{better} better (p={float(p):.4f})' if sig
                       else 'no effect demonstrated')
            report['pairs'][f'{a}~{b_arm}'] = {
                'b_only_correct': bb, 'c_only_correct': cc, 'discordant': bb + cc,
                'p_exact': f'{p.numerator}/{p.denominator}', 'p_float': float(p),
                'significant_at_alpha': sig, 'favours': better if sig else None,
                'rows_compared': len(shared)}
            print(f'{a+"~"+b_arm:<16} {bb:>3} {cc:>3} {bb+cc:>3} {float(p):>9.4f}  {verdict}')

    path = OUT / 'ARM_COMPARISON.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('\nwrote', path)
    print('minimum detectable effect on this panel: 6 discordant rows split 6-0 (p=0.031)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
