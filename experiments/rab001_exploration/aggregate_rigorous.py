"""Aggregate the multi-seed matched comparison into one table with honest uncertainty.

Reports per-arm mean and spread across seeds, the CED penalty with its seed-level range, and
Wilson confidence intervals on the task rates. Also runs a built-in control: step-0 metrics
must be IDENTICAL across seeds for a given arm, because the seed only shuffles the training
order and cannot touch the untrained model. If that control fails, something other than the
training order is consuming the seed and the comparison is not matched.

Exploratory. Outside the hashed root; no receipt, no gate.
"""
import json, math, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
R = E / 'outputs/rigorous'
ARMS = ['source_control', 'C28']
SEEDS = [17, 29, 43]


def wilson(hits, n, z=1.96):
    """Wilson score interval; normal approximation is unreliable at these n."""
    if not n:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


runs, missing = {}, []
for arm in ARMS:
    for seed in SEEDS:
        f = R / f'RIGOROUS_{arm}_seed{seed}_lr1e-05_s40.json'
        if f.is_file():
            runs[(arm, seed)] = json.loads(f.read_text(encoding='utf-8'))
        else:
            missing.append(f.name)

if missing:
    print(f'MISSING {len(missing)} run(s): ' + ', '.join(missing))
print(f'loaded {len(runs)} runs\n')

# ---- control: step 0 must be identical across seeds ------------------------
print('CONTROL: step-0 metrics must be identical across seeds (seed only shuffles TRAIN)')
control_ok = True
for arm in ARMS:
    got = [(s, runs[(arm, s)]['checkpoints'][0]) for s in SEEDS if (arm, s) in runs]
    if len(got) < 2:
        print(f'  {arm:16} only {len(got)} run(s); control not applicable')
        continue
    base = got[0][1]
    same = all(abs(c['dev_ce_nats'] - base['dev_ce_nats']) < 1e-9
               and c['task_numeric_hits'] == base['task_numeric_hits'] for _, c in got)
    control_ok &= same
    print(f"  {arm:16} devCE {base['dev_ce_nats']:.6f} task {base['task_numeric_hits']}/{base['task_n']} "
          f"-> identical across {len(got)} seeds: {same}")
print(f'  CONTROL {"PASS" if control_ok else "FAIL - investigate before trusting the comparison"}\n')

# ---- per-arm aggregation ---------------------------------------------------
summary = {}
print(f"{'arm':16}{'step':>5}{'devCE mean':>12}{'range':>16}{'task rate':>11}{'95% CI':>18}{'seeds':>7}")
for arm in ARMS:
    have = [s for s in SEEDS if (arm, s) in runs]
    if not have:
        continue
    steps = [c['step'] for c in runs[(arm, have[0])]['checkpoints']]
    summary[arm] = {}
    for i, step in enumerate(steps):
        ces = [runs[(arm, s)]['checkpoints'][i]['dev_ce_nats'] for s in have]
        kls = [runs[(arm, s)]['checkpoints'][i]['dev_kl_to_source_nats'] for s in have]
        hits = sum(runs[(arm, s)]['checkpoints'][i]['task_numeric_hits'] for s in have)
        n = sum(runs[(arm, s)]['checkpoints'][i]['task_n'] for s in have)
        lo, hi = wilson(hits, n)
        summary[arm][step] = {
            'dev_ce_mean': mean(ces), 'dev_ce_min': min(ces), 'dev_ce_max': max(ces),
            'dev_kl_mean': mean(kls), 'task_hits_pooled': hits, 'task_n_pooled': n,
            'task_rate_pooled': hits / n if n else 0.0, 'task_ci95': [lo, hi],
            'per_seed_dev_ce': dict(zip(have, ces)), 'seeds': have,
        }
        print(f"{arm:16}{step:>5}{mean(ces):>12.4f}{f'[{min(ces):.4f},{max(ces):.4f}]':>16}"
              f"{hits}/{n:<7}{f'[{lo:.1%},{hi:.1%}]':>18}{len(have):>7}")
print()

# ---- the comparison --------------------------------------------------------
verdict = {}
if 'source_control' in summary and 'C28' in summary:
    for step in sorted(set(summary['source_control']) & set(summary['C28'])):
        a, b = summary['source_control'][step], summary['C28'][step]
        pen = b['dev_ce_mean'] - a['dev_ce_mean']
        # Does the CED penalty exceed the seed-to-seed spread of either arm?
        spread = max(a['dev_ce_max'] - a['dev_ce_min'], b['dev_ce_max'] - b['dev_ce_min'])
        separated = (b['dev_ce_min'] > a['dev_ce_max'])
        task_disjoint = (b['task_ci95'][1] < a['task_ci95'][0]) or (a['task_ci95'][1] < b['task_ci95'][0])
        verdict[step] = {'ced_ce_penalty_nats': pen, 'max_within_arm_seed_spread': spread,
                         'ce_ranges_disjoint': separated, 'task_ci95_disjoint': task_disjoint,
                         'source_task_rate': a['task_rate_pooled'], 'c28_task_rate': b['task_rate_pooled']}
        print(f"step {step:>3}: CED CE penalty {pen:+.4f} nats | max within-arm seed spread "
              f"{spread:.4f} | CE ranges disjoint across arms: {separated}")
        print(f"          task  source {a['task_rate_pooled']:.1%} {a['task_ci95']} vs "
              f"C28 {b['task_rate_pooled']:.1%} {b['task_ci95']} | CIs disjoint: {task_disjoint}")

out = R / 'AGGREGATE_MULTISEED.json'
out.write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_multiseed_aggregate_only',
    'note': 'Pooled across seeds with Wilson intervals on task rates. Seeds differ only in '
            'training-row order, so between-seed spread measures run-to-run variance at a '
            'fixed budget, not sampling of the eval. Open splits only; no receipt, no gate.',
    'runs_loaded': len(runs), 'missing': missing,
    'step0_identity_control_pass': control_ok,
    'per_arm': summary, 'comparison': verdict,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out)
