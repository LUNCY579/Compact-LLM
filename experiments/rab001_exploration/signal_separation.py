"""Does ANY recorded signal separate the capable arm from the incapable one?

The retention gate rests on a negative claim: no likelihood- or teacher-agreement-shaped
quantity recorded in this programme can tell a working model from token salad. HOTFIX-005 cites
that claim, so it needs an artifact rather than an assertion.

This re-analyses the two committed 500k artifacts. No model, no torch, no training: it reads
recorded numbers and does arithmetic, so it is safe to run beside a model job.

Method. The matched pair gives 18 checkpoints, 9 per arm. Label them by the task score itself:

    capable    task_numeric_hits >= 16      (every source_control checkpoint, 16..26)
    incapable  task_numeric_hits <= 4       (every C28 checkpoint, 0..4)

No checkpoint falls between 4 and 16, so the labelling needs no tie-break and no tuning. For
each recorded signal, ask the most generous question available: does ANY threshold separate the
two groups -- i.e. are the two groups' ranges disjoint? A signal that cannot separate its own
training run's endpoints cannot certify anything.

    python experiments/rab001_exploration/signal_separation.py

Exploratory. Outside the hashed root: no receipt, no gate, open splits only, no efficacy claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PKG = Path(__file__).resolve().parents[1] / 'architecture_benchmark_001'
OUT = PKG / 'evidence' / 'retention-design-v1' / 'SIGNAL_SEPARATION.json'

ARMS = ('C28', 'source_control')
#: Every per-checkpoint quantity the run recorded, other than the task score itself.
SIGNALS = ('dev_ce_nats', 'dev_response_ce_nats', 'dev_kl_to_source_nats',
           'dev_response_kl_nats', 'dev_top1_agreement', 'dev_response_top1_agreement',
           'task_terminated', 'task_distinct')
#: Lower is better for a loss, higher is better for an agreement or a variety count.
LOWER_IS_BETTER = {'dev_ce_nats', 'dev_response_ce_nats',
                   'dev_kl_to_source_nats', 'dev_response_kl_nats'}
CAPABLE_AT_LEAST, INCAPABLE_AT_MOST = 16, 4


def load() -> list[dict]:
    points = []
    for arm in ARMS:
        path = PKG / 'evidence' / 'trainability' / f'LONGBUDGET_{arm}_seed17_b500000_lr1e-05.json'
        body = json.loads(path.read_text(encoding='utf-8'))
        if not body.get('completed'):
            raise SystemExit(f'{path.name} is not a completed run')
        for checkpoint in body['checkpoints']:
            points.append({'arm': arm, 'tokens': checkpoint['tokens'],
                           'task': checkpoint['task_numeric_hits'], 'n': checkpoint['task_n'],
                           **{s: checkpoint[s] for s in SIGNALS}})
    return points


def main() -> int:
    points = load()
    capable = [p for p in points if p['task'] >= CAPABLE_AT_LEAST]
    incapable = [p for p in points if p['task'] <= INCAPABLE_AT_MOST]
    unlabelled = [p for p in points if CAPABLE_AT_LEAST > p['task'] > INCAPABLE_AT_MOST]
    if unlabelled:
        raise SystemExit(f'{len(unlabelled)} checkpoints fall in the labelling gap; '
                         'the margin is no longer clean and the rule must be restated')

    results = {}
    for signal in SIGNALS:
        good = [p[signal] for p in capable]
        bad = [p[signal] for p in incapable]
        if signal in LOWER_IS_BETTER:
            # A separating threshold exists iff every capable point beats every incapable one.
            separates = max(good) < min(bad)
            worst_good = max(good)
            best_bad = min(bad)
        else:
            separates = min(good) > max(bad)
            worst_good = min(good)
            best_bad = max(bad)
        witness = None
        if not separates:
            # The single pair that kills the signal: an incapable checkpoint that looks at
            # least as good as the weakest capable one.
            g = min(capable, key=lambda p: p[signal]) if signal in LOWER_IS_BETTER else None
            g = max(capable, key=lambda p: p[signal]) if g is None else g
            worst = max(capable, key=lambda p: p[signal]) if signal in LOWER_IS_BETTER \
                else min(capable, key=lambda p: p[signal])
            best = min(incapable, key=lambda p: p[signal]) if signal in LOWER_IS_BETTER \
                else max(incapable, key=lambda p: p[signal])
            witness = {
                'capable_point': {'arm': worst['arm'], 'tokens': worst['tokens'],
                                  'task': f"{worst['task']}/{worst['n']}", signal: worst[signal]},
                'incapable_point': {'arm': best['arm'], 'tokens': best['tokens'],
                                    'task': f"{best['task']}/{best['n']}", signal: best[signal]},
                'reads_better': bool(
                    best[signal] < worst[signal] if signal in LOWER_IS_BETTER
                    else best[signal] > worst[signal])}
        results[signal] = {
            'separates_capable_from_incapable': bool(separates),
            'capable_range': [min(good), max(good)], 'incapable_range': [min(bad), max(bad)],
            'weakest_capable': worst_good, 'strongest_incapable': best_bad,
            'inverted_witness': witness}

    separating = [s for s, r in results.items() if r['separates_capable_from_incapable']]

    # A signal that separates today may only be separating because the run stopped where it
    # did. For each separating signal, ask whether the capable arm is still moving TOWARD the
    # incapable arm's plateau, and how many milestones of its recent drift the gap is worth.
    for signal in separating:
        trail = [p[signal] for p in sorted((q for q in points if q['arm'] == 'source_control'),
                                           key=lambda q: q['tokens'])]
        deltas = [round(b - a, 6) for a, b in zip(trail, trail[1:])][-3:]
        gap = abs(results[signal]['weakest_capable'] - results[signal]['strongest_incapable'])
        recent = abs(deltas[-1]) if deltas and deltas[-1] else None
        toward = all(d < 0 for d in deltas) if signal not in LOWER_IS_BETTER \
            else all(d > 0 for d in deltas)
        results[signal]['durability'] = {
            'capable_arm_last_three_deltas': deltas,
            'capable_arm_still_moving_toward_incapable': bool(toward),
            'gap': gap,
            'milestones_of_recent_drift_the_gap_is_worth': (
                round(gap / recent, 2) if recent else None),
            'note': 'The separation is a property of where this run stopped, not an '
                    'established threshold. A capable arm still drifting toward the incapable '
                    'plateau will erase it at a larger budget.'}
    body = {
        'schema': 1,
        'evidence_kind': 'exploratory_signal_separation_reanalysis_only',
        'note': 'Re-analysis of two committed 500k artifacts. No model was loaded and no new '
                'measurement was taken. Outside the hashed root: no receipt, no gate, open '
                'splits only, no efficacy claim.',
        'question': 'Does any threshold on a recorded non-task signal separate the capable '
                    'checkpoints from the incapable ones?',
        'labelling_rule': {'capable_task_at_least': CAPABLE_AT_LEAST,
                           'incapable_task_at_most': INCAPABLE_AT_MOST,
                           'checkpoints_in_gap': 0,
                           'basis': 'source_control spans 16..26 and C28 spans 0..4 over the '
                                    'nine milestones, so the margin needs no tie-break'},
        'checkpoints': len(points), 'capable': len(capable), 'incapable': len(incapable),
        'signals_tested': list(SIGNALS),
        'signals_that_separate': separating,
        'separating_signal_count': len(separating),
        'results': results,
        'points': points,
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    print(f'{len(points)} checkpoints: {len(capable)} capable, {len(incapable)} incapable, '
          f'0 in the gap')
    for signal, r in results.items():
        verdict = 'SEPARATES' if r['separates_capable_from_incapable'] else 'no threshold works'
        print(f'  {signal:32s} {verdict}')
        w = r['inverted_witness']
        if w and w['reads_better']:
            print(f"      inverted: {w['incapable_point']['arm']}@{w['incapable_point']['tokens']} "
                  f"({w['incapable_point']['task']}) reads BETTER than "
                  f"{w['capable_point']['arm']}@{w['capable_point']['tokens']} "
                  f"({w['capable_point']['task']})")
    print(f'\nsignals that separate: {separating or "NONE"}')
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
